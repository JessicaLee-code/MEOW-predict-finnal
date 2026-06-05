#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交付管线全窗口演练 + 内存峰值采样（跨平台：Windows / macOS / Linux 通用）。

用途
----
在 ≥32GB 机器上跑正式提交链 `fit(Jun–Nov) + eval(Dec)`，一次性核验三件事：
1. 两成员（X1 ridge + lgbm_d4）整窗训练不 OOM；
2. 内存峰值是否落在预期（fit 持续峰 ~20GB；lgbm 列抽 numpy 处可有 ~28GB 瞬时尖峰）；
3. 三指标（Pearson / R² / MSE）量纲健康（Dec 只当 sanity，不回灌选型）。

全程 fire-and-forget：所有输出（meow 内部日志、`Preallocating window matrix` 行、
eval 三指标、内存心跳与峰值）都 tee 到一个固定日志文件，事后供核对——
只把文件末尾的「FINAL 汇总」几行带回即可判断峰值是否达标。

为什么不用 experiments/run_with_memory_guard.py
------------------------------------------------
那个看门狗依赖 `ps -ax` / `os.killpg` / `signal` / `start_new_session`，仅 Unix 可用。
本脚本改用 psutil 在「同一进程内」采样；lgbm 是多线程（不是多进程），RSS 全落在本进程，
因此同进程采样最准，且天然跨平台、可在 Windows 直接跑。

运行（在仓库根目录、已装好依赖的环境里）
----------------------------------------
    python experiments/run_submission_full_window.py

可选环境变量（不设则用全窗口默认）：
    MEOW_DATA_DIR        数据目录（默认 <repo>/data）
    MEOW_TRAIN_START/END 训练窗口（默认 20230601 / 20231130）
    MEOW_EVAL_START/END  评测窗口（默认 20231201 / 20231229）
    MEOW_MEM_POLL_SEC    采样间隔秒（默认 2.0；越小越能抓到瞬时尖峰）
    MEOW_MEM_HEARTBEAT_SEC 心跳打印间隔秒（默认 20.0）
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MEOW_DIR = REPO_ROOT / "meow"
SRC_DIR = REPO_ROOT / "src"

try:
    import psutil
except ImportError:
    sys.stderr.write(
        "缺少 psutil（内存采样必需）。请先安装：\n    pip install psutil\n"
    )
    raise


class Tee:
    """把写入同时分发到多个流（控制台 + 日志文件），带锁避免多线程交错。"""

    def __init__(self, *streams):
        self._streams = [s for s in streams if s is not None]
        self._lock = threading.Lock()

    def write(self, data):
        with self._lock:
            for s in self._streams:
                s.write(data)
                s.flush()

    def flush(self):
        with self._lock:
            for s in self._streams:
                s.flush()


class PeakSampler(threading.Thread):
    """后台线程：定时采样本进程（含子进程）RSS，跟踪全程峰值，并周期打印心跳。"""

    def __init__(self, poll_sec=2.0, heartbeat_sec=20.0):
        super().__init__(daemon=True)
        self._poll = poll_sec
        self._heartbeat = heartbeat_sec
        self._proc = psutil.Process(os.getpid())
        # 注意：属性名不能叫 `_stop`——会覆盖 threading.Thread 内部的 `_stop()` 方法，
        # 导致 join() 收尾时把 Event 当方法调用 → TypeError: 'Event' object is not callable。
        self._stop_event = threading.Event()
        self.peak_gb = 0.0

    def _rss_gb(self):
        total = self._proc.memory_info().rss
        for child in self._proc.children(recursive=True):
            try:
                total += child.memory_info().rss
            except psutil.Error:
                pass
        return total / (1024 ** 3)

    def run(self):
        last_hb = 0.0
        while not self._stop_event.is_set():
            cur = self._rss_gb()
            if cur > self.peak_gb:
                self.peak_gb = cur
            now = time.time()
            if now - last_hb >= self._heartbeat:
                _log("[mem] current={:.2f} GB  peak={:.2f} GB".format(cur, self.peak_gb))
                last_hb = now
            self._stop_event.wait(self._poll)

    def stop(self):
        self._stop_event.set()


def _log(msg):
    """统一带时间戳打印（经 Tee → 控制台 + 日志文件）。"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("[{}] {}".format(ts, msg))


def _load_meow_engine():
    """
    按文件路径加载 meow/meow.py 的 MeowEngine。

    meow 内部用裸 import（dl/feat/mdl…）；这里把 meow/ 置于 sys.path 最前，
    保证 `dl` 解析到带 `countDate` 的 meow/dl.py（而非 src/ 下同名旧副本）。
    本脚本是进程入口、最先 import，故无需清 sys.modules 缓存。
    """
    sys.path.insert(0, str(SRC_DIR))
    sys.path.insert(0, str(MEOW_DIR))
    spec = importlib.util.spec_from_file_location(
        "meow_entry_full_window", str(MEOW_DIR / "meow.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MeowEngine


def main():
    data_dir = os.environ.get("MEOW_DATA_DIR", str((REPO_ROOT / "data").resolve()))
    train_start = int(os.environ.get("MEOW_TRAIN_START", "20230601"))
    train_end = int(os.environ.get("MEOW_TRAIN_END", "20231130"))
    eval_start = int(os.environ.get("MEOW_EVAL_START", "20231201"))
    eval_end = int(os.environ.get("MEOW_EVAL_END", "20231229"))
    poll_sec = float(os.environ.get("MEOW_MEM_POLL_SEC", "2.0"))
    hb_sec = float(os.environ.get("MEOW_MEM_HEARTBEAT_SEC", "20.0"))

    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / "submission_full_window_{}.log".format(stamp)
    log_fp = log_path.open("w", encoding="utf-8")

    # tee：控制台 + 文件，捕获一切（含 meow log 默认的 print 输出）。
    sys.stdout = Tee(sys.__stdout__, log_fp)
    sys.stderr = Tee(sys.__stderr__, log_fp)

    vm = psutil.virtual_memory()
    _log("==== 交付管线全窗口演练开始 ====")
    _log("日志文件：{}".format(log_path))
    _log("平台：{}    Python：{}    CPU逻辑核：{}".format(
        sys.platform, sys.version.split()[0], psutil.cpu_count()))
    _log("物理内存总量：{:.1f} GB".format(vm.total / 1024 ** 3))
    _log("数据目录：{}".format(data_dir))
    _log("训练窗口：{}–{}    评测窗口：{}–{}".format(
        train_start, train_end, eval_start, eval_end))
    _log("预期：fit 持续峰 ~20GB；lgbm 列抽 numpy 处瞬时尖峰可达 ~28GB；32GB 可 survive。")
    _log("内存采样：每 {:.1f}s 取样、每 {:.1f}s 打印心跳".format(poll_sec, hb_sec))
    if vm.total / 1024 ** 3 < 31.0:
        _log("⚠️ 本机物理内存 < 32GB，全窗口可能 OOM；建议换 ≥32GB 机器跑。")

    sampler = PeakSampler(poll_sec=poll_sec, heartbeat_sec=hb_sec)
    sampler.start()

    t0 = time.time()
    MeowEngine = _load_meow_engine()
    engine = MeowEngine(h5dir=data_dir, cacheDir=None)

    _log("===== PHASE fit START =====")
    tf = time.time()
    engine.fit(train_start, train_end)
    peak_after_fit = sampler.peak_gb
    _log("===== PHASE fit DONE  耗时 {:.0f}s  fit后峰值={:.2f} GB =====".format(
        time.time() - tf, peak_after_fit))

    _log("===== PHASE eval START =====")
    te = time.time()
    engine.eval(eval_start, eval_end)
    _log("===== PHASE eval DONE  耗时 {:.0f}s =====".format(time.time() - te))

    sampler.stop()
    sampler.join(timeout=5)

    # 显眼的 FINAL 汇总：只把这几行带回 Mac 即可判断峰值是否达标。
    _log("======================== FINAL 汇总 ========================")
    _log("总耗时：{:.0f}s".format(time.time() - t0))
    _log("FINAL PEAK RSS（全程最高）：{:.2f} GB".format(sampler.peak_gb))
    _log("  其中 fit 阶段峰值：{:.2f} GB".format(peak_after_fit))
    _log("核对口径：持续心跳应稳定在 ~15–20GB；全程峰值（含瞬时）≤ ~28GB 即符合中档预期。")
    _log("============================================================")
    log_fp.flush()
    log_fp.close()


if __name__ == "__main__":
    main()
