# -*- coding: utf-8 -*-
"""DL-on-raw serve 腿：把截面模型（直吃 raw 59 通道）接进 `meow.py` 的 fit/predict。

fit() 现训 K=3 seed（不落权重文件），predict() 取 seed 平均交给 MeowEngine 与传统等权融合。
防御式降级：torch/CUDA/任一 DL 环节出错则 available=False、自动回落纯传统（绝不崩、绝不 NaN）。
训练段切 15% early-stop、归一化只用训练区统计、predict 复用训练期 Normalizer（零泄漏，防 overfit）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from log import log  # meow/ 入口已在 path 上
except Exception:  # 极端兜底：连日志模块都没有也不让它把提交链拖崩
    log = None


def _loginf(msg):
    """信息日志：优先 MeowLogger.inf，缺失/出错则退化到 print。防御代码的日志自身绝不能崩。"""
    fn = getattr(log, "inf", None)
    if callable(fn):
        try:
            fn(msg)
            return
        except Exception:
            pass
    print(msg)


def _logerr(msg):
    """错误日志：MeowLogger 无 .err，用 .red；都没有则退 .inf / print。"""
    for name in ("red", "inf"):
        fn = getattr(log, name, None)
        if callable(fn):
            try:
                fn(msg)
                return
            except Exception:
                break
    print(msg)


def _warn_banner(lines):
    """打印醒目红色横幅，确保老师在大量训练日志里也能一眼看到「降级/缺依赖」提示。

    逐行走 log.red（项目既有 ANSI 约定）；log 不可用则 print 兜底。横幅自身绝不抛异常。
    """
    body = ["=" * 60] + list(lines) + ["=" * 60]
    fn = getattr(log, "red", None)
    for ln in body:
        if callable(fn):
            try:
                fn(ln)
                continue
            except Exception:
                pass
        print(ln)


def _warn_torch_missing(err=None):
    """缺 PyTorch（老师最可能遇到）：打醒目提示，明确告诉装什么、不装的后果。"""
    lines = [
        "  注意：未检测到 PyTorch，深度学习增强支路【无法启用】。",
        "  程序不会崩溃、会照常输出传统模型结果（约 0.081），",
        "  但无法叠加深度学习增强（完整融合约 0.092）。",
        "  ▶ 若要拿到完整结果，请先安装依赖再运行：  pip install torch",
        "  这是环境依赖缺失提示，不是程序结束或报错。",
    ]
    if err is not None:
        lines.append("  底层错误：{}".format(repr(err)[:160]))
    _warn_banner(lines)


def _warn_runtime_degraded(err):
    """torch 在、但 DL 运行中途异常（显存/内存不足等）：打醒目提示，说明已回落、不是程序结束。"""
    _warn_banner([
        "  注意：深度学习支路本次运行异常，已自动回落纯传统模型（约 0.081）。",
        "  程序未崩溃、已照常输出结果；深度学习增强（约 0.092）本次未生效。",
        "  常见原因：GPU 显存或系统内存不足，可释放资源后重跑。",
        "  底层错误：{}".format(repr(err)[:160]),
    ])


# —— 默认 DL 配置：照搬机器2 proven 冠军 run（summary.json.champion），不臆造 —— #
# seq_len 走 SequenceDataset（结构旋钮，独立于卡带 hparams）；其余进卡带 hparams。
_DEFAULT_SEQ_LEN = 32
_DEFAULT_HPARAMS = {
    "dropout": 0.2,
    "hidden_size": 32,
    "num_layers": 1,
    "lambda_corr": 0.3,
    "max_epochs": 15,
    "patience": 5,
    "weight_decay": 0.001,
}
_DEFAULT_SEEDS = (42, 43, 44)        # K=3 seed 平均（交付真正会用的形态）
_EARLYSTOP_FRAC = 0.15               # 训练段尾部切 15% 做 early-stop（防 overfit）
_NORMALIZER_MODE = "zscore"          # 与机器2 训练一致（run 名 *_zscore）
_FUSION_WEIGHT_DL = 0.5              # 等权融合（零自由参数，离线已证 best_w≈0.5）


def _read_seeds_env() -> tuple:
    """允许用 MEOW_DL_SEEDS=42,43 临时改 seed 数（serve 测试提速用）；非法则回默认。"""
    raw = os.environ.get("MEOW_DL_SEEDS", "").strip()
    if not raw:
        return _DEFAULT_SEEDS
    try:
        seeds = tuple(int(s) for s in raw.split(",") if s.strip() != "")
        return seeds or _DEFAULT_SEEDS
    except Exception:
        return _DEFAULT_SEEDS


def _ensure_dl_path() -> None:
    """把 src/config/models 三目录平铺加进 sys.path（DL 机理依赖三目录平铺 import）。"""
    repo = Path(__file__).resolve().parent.parent
    for sub in ("src", "config", "models"):
        p = str(repo / sub)
        if p not in sys.path:
            sys.path.append(p)


class DLServe:
    """
    DL-on-raw serve 腿。两阶段：`fit(train_dates)` 现训 K seed；`predict(eval_dates)` 出 K seed 平均。

    任一环节抛异常都被吞掉并置 `available=False`——调用方据此回落纯传统，绝不让 DL 把提交链拖崩。
    """

    def __init__(self, raw_loader, seeds=None, seq_len=_DEFAULT_SEQ_LEN, hparams=None):
        # raw_loader: dates -> raw DataFrame（含 META + raw 列 + fret12），serve 直接传 MeowEngine.dloader.loadDates。
        self.raw_loader = raw_loader
        self.seeds = tuple(seeds) if seeds else _read_seeds_env()
        self.seq_len = int(seq_len)
        self.hparams = dict(hparams or _DEFAULT_HPARAMS)
        self.available = False          # 训练成功才置 True
        self._cartridges = []           # 每 seed 一个训练好的卡带
        self._normalizer = None         # 训练期 fit 的 Normalizer，predict 复用（零泄漏）
        self._adapter = None            # RawChannelAdapter（无状态，fit/predict 共用）
        self._device = "cpu"
        # 构造即探测 torch：缺失就当场打醒目提示，老师可立刻安装，不必等传统训练 ~50min 后才发现。
        self._torch_ready = self._probe_dependencies()

    def _probe_dependencies(self) -> bool:
        """只探测 torch 是否可用、不训练；缺失则立刻打「请装 torch」醒目横幅。返回是否就绪。"""
        try:
            _ensure_dl_path()
            import torch  # noqa: F401
            return True
        except Exception as e:
            _warn_torch_missing(e)
            return False

    # ---- 训练阶段：现训 K seed ---- #
    def fit(self, train_dates) -> None:
        """在训练窗现训 K 个 seed 的 DL-on-raw；任何失败都置 available=False（回落传统）。"""
        train_dates = list(train_dates)
        if not self._torch_ready:
            # 启动时已打过醒目「请装 torch」横幅，这里只补一行简短说明、不重复刷屏。
            self.available = False
            self._cartridges = []
            _logerr("[DLServe] 跳过 DL 训练（未检测到 PyTorch，详见启动时提示）→ 本次纯传统。")
            return
        try:
            _ensure_dl_path()
            import torch  # noqa: F401  —— import 失败即触发降级
            from sequence_dataset import (
                Normalizer,
                SequenceDataset,
                build_sequence_arrays,
                subset_by_dates,
            )
            from dl_protocol import split_train_earlystop
            from registry import build_adapter, build_cartridge
            from adapter_config import AdapterConfig, AdapterKind
            from model_config import ModelConfig, ModelKind

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            hp = dict(self.hparams)
            hp["device"] = self._device

            # RawChannelAdapter：固定 59 通道、无需 bind_data_sources（纯 raw 行内变换）。
            adapter = build_adapter(AdapterConfig(kind=AdapterKind.RAW_CHANNELS))
            self._adapter = adapter

            # 训练窗尾部切 15% 做 early-stop（与实验一致，防 overfit）。
            core_dates, es_dates = split_train_earlystop(tuple(train_dates), _EARLYSTOP_FRAC)
            _loginf(
                "[DLServe] 现算训练 raw 序列：core={} 天 / earlystop={} 天，device={}".format(
                    len(core_dates), len(es_dates), self._device
                )
            )
            core_arrays = build_sequence_arrays(self.raw_loader(list(core_dates)), adapter)
            # earlystop 段（日期与 core 互斥、无泄漏）；为空则用空切分得到形状对的空 arrays。
            es_arrays = (
                build_sequence_arrays(self.raw_loader(list(es_dates)), adapter)
                if es_dates else subset_by_dates(core_arrays, ())
            )

            # Normalizer 只用训练区（core+es）统计、分块 fit（不 concatenate、不物化整张副本）。
            normalizer = Normalizer(_NORMALIZER_MODE).fit_chunked(
                [core_arrays.features, es_arrays.features]
            )
            self._normalizer = normalizer

            cartridges = []
            for seed in self.seeds:
                # own_features=False：每 seed 各自拷贝白化，绝不原地改 core_arrays（否则下一 seed 双重白化）。
                core_ds = SequenceDataset(core_arrays, self.seq_len, normalizer, own_features=False)
                es_ds = SequenceDataset(es_arrays, self.seq_len, normalizer, own_features=False)
                cart = build_cartridge(ModelConfig(kind=ModelKind.XSECTION_RAW, hparams=hp))
                _loginf("[DLServe] 训练 DL seed={} ...".format(seed))
                cart.fit(core_ds, es_ds, hp, int(seed))
                cartridges.append(cart)

            self._cartridges = cartridges
            self.available = True
            _loginf("[DLServe] DL 现训完成：{} seed 就绪（device={}）".format(len(cartridges), self._device))
        except Exception as e:  # torch 已就绪却失败 = 运行时异常（显存/内存等）→ 降级，绝不抛给提交链
            self.available = False
            self._cartridges = []
            _warn_runtime_degraded(e)

    # ---- 推理阶段：K seed 平均 ---- #
    def predict(self, eval_dates):
        """返回 DataFrame(date,symbol,interval,pred_dl)（K seed 平均）；不可用/空/失败返回 None。

        因序列 warmup，每票每日前 seq_len-1 个 interval 无 DL 预测、不在结果里，MeowEngine 对这些行用纯传统填。
        """
        if not self.available or not self._cartridges:
            return None
        try:
            from sequence_dataset import SequenceDataset, build_sequence_arrays

            arrays = build_sequence_arrays(self.raw_loader(list(eval_dates)), self._adapter)
            if arrays.n_rows == 0:
                return None

            preds = []
            label_frame = None
            for cart in self._cartridges:
                # 每 seed 一份拷贝白化的 dataset；推理顺序=label_frame 行序（卡带 predict 已映射回窗口序）。
                ds = SequenceDataset(arrays, self.seq_len, self._normalizer, own_features=False)
                p = np.asarray(cart.predict(ds), dtype=np.float64).reshape(-1)
                preds.append(p)
                if label_frame is None:
                    label_frame = ds.label_frame()
            if not preds or label_frame is None or len(label_frame) == 0:
                return None
            # 各 seed predict 同 eval 数据、同 seq_len → 同窗口集、行序一致，可直接按列平均。
            avg = np.mean(np.column_stack(preds), axis=1)
            out = label_frame[["date", "symbol", "interval"]].copy()
            out["pred_dl"] = np.asarray(avg, dtype=np.float64)
            return out
        except Exception as e:
            _warn_runtime_degraded(e)
            return None


def fuse_traditional_with_dl(xdf, trad_pred, dl_df, weight_dl: float = _FUSION_WEIGHT_DL):
    """按 (date,symbol,interval) 等权融合 `(1-w)*trad + w*dl`（默认 w=0.5），输出对齐 xdf 行序的一维数组。

    dl_df 为 None / 只覆盖部分行时，缺的行（DL warmup）用纯传统填，绝不引入 NaN。量纲留在 fret12 保 MSE/R²。
    """
    trad = np.asarray(trad_pred, dtype=np.float64).reshape(-1)
    if dl_df is None or len(dl_df) == 0:
        return trad.astype(np.float32)

    # 用行号保住 xdf 原始顺序：merge 后按 _row 复原（meta 每行唯一 → 1:1 左连接）。
    key = xdf[["date", "symbol", "interval"]].copy()
    key["_row"] = np.arange(len(key), dtype=np.int64)
    merged = key.merge(dl_df, on=["date", "symbol", "interval"], how="left").sort_values("_row")
    dl_aligned = merged["pred_dl"].to_numpy(dtype=np.float64)   # 缺行为 NaN

    has_dl = np.isfinite(dl_aligned)
    fused = trad.copy()
    fused[has_dl] = (1.0 - weight_dl) * trad[has_dl] + weight_dl * dl_aligned[has_dl]
    n_dl = int(has_dl.sum())
    log.inf(
        "[DLServe] 融合完成：{}/{} 行用上 DL（其余 warmup 行纯传统），融合权重 w_dl={}".format(
            n_dl, len(trad), weight_dl
        )
    )
    return fused.astype(np.float32)
