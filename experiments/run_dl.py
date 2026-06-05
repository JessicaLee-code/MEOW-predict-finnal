"""
Orchestrator —— DL run 的发起 + 组装冻结 RunConfig + 两阶段交接 + 落盘（torch-free 编排）

对应规格 §2.1 控制层级最顶层、§7 配置管理。职责**只有组装 + 派发 + 落盘**，不写任何块旋钮
（避免 god-object，规格 §7.1 反模式警告）：

1. 吃一份**已组装冻结**的 ``RunConfig``；按 registry 建 ``adapter`` + ``cartridge_factory``、
   按 ``MeowDataLoader`` 建 ``raw_loader``（测试可注入合成 loader + 参考卡带）、按 ``protocol``
   派生折（并跑 ``assert_folds_causal`` 防泄漏闸）。
2. dump ``config.json``（含 ``config_fingerprint``）到 ``out_dir/<run_id>/``，可复现可审计。
3. 按 ``stage`` 分派：
   - ``SEARCH``  → 交 ``Searcher`` 海选（落 ``trials.csv`` + ``best_config.json``）。
   - ``VALIDATION`` → 定参跑 expanding 少折 × 多种子认证（落 ``fold_metrics.csv`` + ``summary.json``）。
   - ``SWEEP`` → 一命令两档（主路径，§十一·11.6）：档1 网格筛选 + 档2 冠军认证。

**增量落盘（防长跑中断打水漂）**：SWEEP / VALIDATION 是数小时级长跑，``trials.csv`` /
``fold_metrics.csv`` 不再"全算完一次性写"，而是**每个 trial / 每折一完成就 append + flush + fsync**
（``_IncrementalCsvWriter``），并同步逐事件写 ``progress.jsonl`` 时间线 + 每折刷新
``summary.partial.json`` 实时快照。中途崩溃也留下全部已完成步骤的价值；``--resume`` 可复用这些
落盘跳过已完成的不重算。``summary.json`` 仅在成功收官时落 → 它"在 ⇔ 跑完"，是完成标记。

**内存约定**：numpy 参考卡带走 ``gather_all``（一次性物化全部窗口），全量真实数据会爆内存；
故 CLI smoke 默认用 ``--max-symbols`` 抽样降规模。真正的 TCN 卡带按 ``iter_batches`` 流式喂，
不物化、无此限制。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence

# —— import 约定：src / config / models 三目录平铺（README / CLAUDE 已记） —— #
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _sub in ("src", "config", "models"):
    _p = os.path.join(_REPO_ROOT, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

from dl_protocol import (  # noqa: E402
    DLFold, assert_folds_causal, build_delivery_fold, build_dl_folds, summarize_folds,
)
from dl_search import EarlyKillPolicy, Searcher, enumerate_grid  # noqa: E402
from dl_trainer import SequenceTrainer  # noqa: E402
from feature_store import DEFAULT_FEATURE_DIR  # noqa: E402
from registry import build_adapter, build_cartridge  # noqa: E402
from protocol_config import ProfileKind, Stage  # noqa: E402

# —— SWEEP 两档预算常量（§十一·11.6；改它们 = 改协议预算，慎重） —— #
_SWEEP_SCREEN_FOLDS = 2   # 档1 筛选用最近 2 折（最像老师未来集）
_SWEEP_SCREEN_SEEDS = 2   # 档1 每 combo 2 seed
_DEFAULT_SEQ_LEN = 32     # seq_len 缺省（search_space / defaults 都没给时兜底）


# ================================================================== #
# Orchestrator
# ================================================================== #

class Orchestrator:
    """对一份冻结的 ``RunConfig`` 发起一次 run。"""

    def __init__(
        self,
        run_config,
        *,
        raw_loader: Optional[Callable[[Sequence[int]], object]] = None,
        h5dir: str = "data",
        feature_dir: str = DEFAULT_FEATURE_DIR,
        adapter=None,
        cartridge_factory: Optional[Callable[[], object]] = None,
        resume: bool = False,
        dump_preds: bool = False,
    ):
        self.cfg = run_config
        self.h5dir = h5dir
        self.feature_dir = feature_dir
        # 续跑开关：True 时启动会读回已落盘的 trial / (seed,fold)，跳过已完成的不重算
        # （仅 SWEEP 支持，见 _run_sweep）。默认关 → 与改造前一致：每次 run 从头重写。
        self.resume = bool(resume)
        # 逐票预测落盘开关：True 时各折把 scoring 段逐票预测落 <out_dir>/<run_id>/preds/，
        # 供「DL↔传统预测相关性」离线分析；默认关、零开销（见 _spec 注入 dump_preds_dir）。
        self.dump_preds = bool(dump_preds)
        # 依赖注入：默认从 registry + MeowDataLoader 构建；测试可注入合成 loader / 参考卡带。
        self.adapter = adapter if adapter is not None else build_adapter(run_config.adapter)
        if hasattr(self.adapter, "bind_data_sources"):
            # 让像 FeatureAdapter 这样的实验型适配器拿到当前 run 的 h5/cache 根目录。
            self.adapter.bind_data_sources(h5dir=self.h5dir, feature_dir=self.feature_dir)
        self.cartridge_factory = cartridge_factory or (lambda: build_cartridge(run_config.model))
        self.raw_loader = raw_loader or self._default_raw_loader(h5dir)

    @staticmethod
    def _default_raw_loader(h5dir: str) -> Callable[[Sequence[int]], object]:
        from dl import MeowDataLoader   # src/dl.py
        loader = MeowDataLoader(h5dir)
        return lambda dates: loader.loadDates(list(dates))

    # ---- 派生折 ---- #
    def _build_folds(self) -> List[DLFold]:
        p = self.cfg.protocol
        folds = build_dl_folds(
            p.rolling_start, p.rolling_end,
            mode=p.fold_mode(), val_window=p.val_window, step=p.step, embargo=p.embargo,
            train_window=p.train_window, min_train_days=p.min_train_days,
            earlystop_frac=p.earlystop_frac, max_folds=p.effective_max_folds(),
            fold_select=p.fold_select,
        )
        assert_folds_causal(folds)   # 防泄漏闸：四段时间严格递增、embargo 隔开训练/打分
        return folds

    # ---- 给 SequenceTrainer / FoldResult 的元信息 ---- #
    def _spec(self) -> Dict:
        c = self.cfg
        spec = {
            "experiment_id": c.run_id,
            "model_type": c.model.kind.value,
            "feature_set": c.adapter.kind.value,
            "target_type": c.exec_.target_mode,
            "postprocess_type": "none",
            "notes": f"fp={c.config_fingerprint}",
        }
        # 残差训练模式：不是新模型，而是训练目标构造方式的切换。
        # 这些字段由 SequenceTrainer 读取，用来：
        # 1) 把 label 改成 y - y_trad；
        # 2) 评测时再恢复 pred_trad + pred_dl_res。
        spec["target_mode"] = c.exec_.target_mode
        spec["trad_preds_root"] = c.exec_.trad_preds_root
        spec["trad_pred_col"] = c.exec_.trad_pred_col
        spec["trad_cache_dir"] = c.exec_.trad_cache_dir
        spec["trad_data_dir"] = self.h5dir
        if self.dump_preds:
            # 逐票预测落 <out_dir>/<run_id>/preds/（run() 里 out_dir 同构）。
            spec["dump_preds_dir"] = os.path.join(c.exec_.out_dir, c.run_id, "preds")
        return spec

    # ---- 顶层入口 ---- #
    def run(self) -> dict:
        out_dir = os.path.join(self.cfg.exec_.out_dir, self.cfg.run_id)
        os.makedirs(out_dir, exist_ok=True)
        self.cfg.dump_json(os.path.join(out_dir, "config.json"))

        folds = self._build_folds()
        if not folds:
            summary = {"run_id": self.cfg.run_id, "stage": self.cfg.protocol.stage.value,
                       "status": "no_folds", "note": "日期窗口/min_train_days 不够派生任何折"}
            _dump_json(os.path.join(out_dir, "summary.json"), summary)
            return summary

        if self.cfg.protocol.stage == Stage.SEARCH:
            return self._run_search(folds, out_dir)
        if self.cfg.protocol.stage == Stage.SWEEP:
            return self._run_sweep(folds, out_dir)
        return self._run_validation(folds, out_dir)

    # ---- 海选 ---- #
    def _run_search(self, folds: Sequence[DLFold], out_dir: str) -> dict:
        # search_space 是卡带私有声明：建一个临时卡带读它。
        search_space = dict(getattr(self.cartridge_factory(), "search_space", {}) or {})
        sc = self.cfg.search
        # device 属执行层而非模型层：由 Orchestrator 在派发时注入，避免卡带自己猜。
        defaults = dict(self.cfg.model.hparams)
        defaults.setdefault("device", self.cfg.exec_.device)
        searcher = Searcher(
            spec=self._spec(), adapter=self.adapter, cartridge_factory=self.cartridge_factory,
            raw_loader=self.raw_loader, folds=folds, search_space=search_space,
            n_trials=sc.n_trials, seeds=self.cfg.exec_.seeds,
            defaults=defaults, normalizer_mode=self._normalizer_mode(),
            search_overrides=dict(sc.search_overrides), profile_name="search",
            early_kill=EarlyKillPolicy(enabled=sc.early_kill, warmup_epochs=sc.early_kill_warmup_epochs),
        )
        outcome = searcher.run()

        # 落 trials.csv（hparams 摊平进 hp_ 列，列集取并集）。
        rows = [t.to_row() for t in outcome.trials]
        _dump_csv(os.path.join(out_dir, "trials.csv"), rows)
        # 落 best_config.json（认证档据此冻结，config-lock）。
        best = outcome.best_config_dict()
        _dump_json(os.path.join(out_dir, "best_config.json"), best)

        summary = {
            "run_id": self.cfg.run_id, "stage": "search", "status": "ok",
            "n_trials": len(outcome.trials), "n_folds": len(folds),
            "best": best,
        }
        _dump_json(os.path.join(out_dir, "summary.json"), summary)
        return summary

    # ---- 认证 ---- #
    def _run_validation(self, folds: Sequence[DLFold], out_dir: str) -> dict:
        defaults = dict(self.cfg.model.hparams)
        defaults.setdefault("device", self.cfg.exec_.device)
        seq_len = int(defaults.pop("seq_len", 32))
        cart_hparams = defaults

        # —— 逐折增量落盘：每个 (seed, fold) 一完成立即 append + flush，不再攒到最后一次性写 —— #
        seeds = self.cfg.exec_.seeds
        total = len(seeds) * len(folds)
        prog = _ProgressLog(os.path.join(out_dir, "progress.jsonl"))
        prog.event("validation_start", n_folds=len(folds), n_seeds=len(seeds),
                   seeds=[int(s) for s in seeds], seq_len=seq_len)
        writer = _IncrementalCsvWriter(os.path.join(out_dir, "fold_metrics.csv"))
        corrs: List[float] = []
        done = 0
        try:
            for seed in seeds:
                trainer = SequenceTrainer(
                    self._spec(), self.adapter, self.cartridge_factory, self.raw_loader,
                    seq_len=seq_len, normalizer_mode=self._normalizer_mode(),
                    hparams=cart_hparams, seed=int(seed),
                )
                for fold in folds:
                    t0 = time.time()
                    r = trainer.run_on_dl_fold(fold, profile_name="validation")
                    d = r.to_dict()
                    d["random_seed"] = int(seed)
                    writer.write_row(d)           # ← 落盘 + flush 在此，崩溃也留下已完成折
                    done += 1
                    if r.status == "ok" and np.isfinite(r.val_corr):
                        corrs.append(float(r.val_corr))
                    prog.event("fold_done", seed=int(seed), fold_id=fold.fold_id,
                               val_corr=_num(r.val_corr), status=r.status,
                               elapsed_sec=round(time.time() - t0, 1), done=done, total=total)
        finally:
            writer.close()

        summ = summarize_folds([{"corr": c} for c in corrs])
        summary = {
            "run_id": self.cfg.run_id, "stage": "validation", "status": "ok",
            "seq_len": seq_len, "hparams": cart_hparams,
            "n_folds": len(folds), "n_seeds": len(self.cfg.exec_.seeds),
            "val_corr": summ,   # mean / std / min(最坏折) / max / positive_rate
        }
        _dump_json(os.path.join(out_dir, "summary.json"), summary)
        prog.event("validation_done", status="ok",
                   val_corr_mean=_num(summ["mean"]), val_corr_min=_num(summ["min"]))
        prog.close()
        return summary

    # ---- 一命令两档（SWEEP，§十一·11.6） ---- #
    def _eval_config(self, seq_len: int, cart_hparams: Dict, folds, seeds) -> Dict:
        """一组超参跑 seeds × folds，收 val_corr + best_epoch（档1/档2 共用的最小评估单元）。"""
        corrs: List[float] = []
        best_epochs: List[int] = []
        for seed in seeds:
            trainer = SequenceTrainer(
                self._spec(), self.adapter, self.cartridge_factory, self.raw_loader,
                seq_len=seq_len, normalizer_mode=self._normalizer_mode(),
                hparams=dict(cart_hparams), seed=int(seed),
            )
            for fold in folds:
                r = trainer.run_on_dl_fold(fold, profile_name="sweep_screen")
                if r.status == "ok" and np.isfinite(r.val_corr):
                    corrs.append(float(r.val_corr))
                    best_epochs.append(int(r.best_epoch))
                elif r.status == "error":
                    raise RuntimeError(f"fold {fold.fold_id} 失败: {r.error_msg}")
        return {"corrs": corrs, "best_epoch_mean": float(np.mean(best_epochs)) if best_epochs else 0.0}

    def _run_sweep(self, folds: Sequence[DLFold], out_dir: str) -> dict:
        # prog（进度日志）在外层创建 + try/finally 关闭，保证中断/异常时它的文件句柄也干净落盘
        # （CSV 写器各自在内层 try/finally 关闭；这里专管贯穿两档的 prog）。
        prog = _ProgressLog(os.path.join(out_dir, "progress.jsonl"))
        try:
            return self._run_sweep_impl(folds, out_dir, prog)
        finally:
            prog.close()

    def _run_sweep_impl(self, folds: Sequence[DLFold], out_dir: str, prog: "_ProgressLog") -> dict:
        """
        一命令两档（§十一·11.6），全程同一套 §十一·11.2 忠实协议（锚定扩展 + 倒贴 + embargo）：

        - **档1 筛选**：确定性小网格 × 最近 ``_SWEEP_SCREEN_FOLDS`` 折 × ``_SWEEP_SCREEN_SEEDS``
          seed → **按最坏折(minimax)选冠军**（§十一·11.7 规则 3，不按峰值）。
        - **档2 认证**：冠军 × 全折 × 全 seed（建议 3）→ 落逐折逐 seed 明细 + pooled / 最坏折 /
          R² 双镜头读数（R² 盯老师精度分 1/3，§十一·11.3）。
        """
        ec = self.cfg.exec_
        sc = self.cfg.search
        # 网格 = 卡带 search_space 叠加 overrides 收窄后**确定性**展开（忠实「网格预先声明跑一次」）。
        # 关键：网格确定 → trial_id == 网格下标，续跑时可由网格重建任一 trial 的超参，无需解析 CSV。
        search_space = dict(getattr(self.cartridge_factory(), "search_space", {}) or {})
        grid = enumerate_grid(search_space, dict(sc.search_overrides))
        defaults = dict(self.cfg.model.hparams)
        defaults.setdefault("device", ec.device)

        n_screen = min(_SWEEP_SCREEN_FOLDS, len(folds))
        screen_folds = list(folds[-n_screen:])     # 最近 N 折（fold_select=recent 下 = 最贴 rolling_end）
        screen_seeds = ec.seeds[:_SWEEP_SCREEN_SEEDS] if len(ec.seeds) >= _SWEEP_SCREEN_SEEDS else ec.seeds
        cert_seeds = ec.seeds

        prog.event("sweep_start", grid_size=len(grid), n_screen_folds=n_screen,
                   screen_seeds=[int(s) for s in screen_seeds],
                   cert_seeds=[int(s) for s in cert_seeds], resume=self.resume)

        # —— 档1：网格逐点跑最近折，**逐 trial 增量落盘**，按最坏折排名 —— #
        trials_path = os.path.join(out_dir, "trials.csv")
        done_trials = _read_done_trials(trials_path) if self.resume else {}
        trials_writer = _IncrementalCsvWriter(trials_path, append=self.resume)
        ranked: List[tuple] = []     # (min, mean, tid, seq_len, hparams)
        try:
            for tid, combo in enumerate(grid):
                eff = {**defaults, **combo}
                seq_len = int(eff.pop("seq_len", _DEFAULT_SEQ_LEN))
                cart_hp = eff
                # 续跑：该 trial 已成功落盘 → 由网格重建 cart_hp、从 CSV 读回排名指标，跳过重算。
                prior = done_trials.get(tid)
                if prior is not None and prior.get("status") == "ok" and _coerce_int(prior.get("n_evals")) > 0:
                    smin = _coerce_float(prior.get("screen_corr_min"))
                    smean = _coerce_float(prior.get("screen_corr_mean"))
                    if np.isfinite(smin):
                        ranked.append((smin, smean, tid, seq_len, dict(cart_hp)))
                    prog.event("trial_skipped", trial_id=tid, seq_len=seq_len)
                    continue

                t0 = time.time()
                status, err, res = "ok", "", {"corrs": [], "best_epoch_mean": 0.0}
                try:
                    res = self._eval_config(seq_len, cart_hp, screen_folds, screen_seeds)
                except Exception as e:                   # noqa: BLE001
                    status, err = "error", str(e)[:300]
                summ = summarize_folds([{"corr": c} for c in res["corrs"]])
                row = {"trial_id": tid, "seq_len": seq_len,
                       "screen_corr_mean": summ["mean"], "screen_corr_min": summ["min"],
                       "screen_corr_std": summ["std"], "n_evals": len(res["corrs"]),
                       "best_epoch_mean": res["best_epoch_mean"], "status": status, "error_msg": err}
                for k, v in cart_hp.items():
                    if k != "device":
                        row[f"hp_{k}"] = v
                trials_writer.write_row(row)          # ← 落盘 + flush 在此（逐 trial）
                if status == "ok" and len(res["corrs"]) > 0:
                    ranked.append((summ["min"], summ["mean"], tid, seq_len, dict(cart_hp)))
                prog.event("trial_done", trial_id=tid, seq_len=seq_len,
                           screen_corr_mean=_num(summ["mean"]), screen_corr_min=_num(summ["min"]),
                           n_evals=len(res["corrs"]), status=status,
                           elapsed_sec=round(time.time() - t0, 1))
        finally:
            trials_writer.close()

        if not ranked:
            summary = {"run_id": self.cfg.run_id, "stage": "sweep", "status": "no_champion",
                       "grid_size": len(grid), "note": "档1 全部 trial 无有效评估"}
            _dump_json(os.path.join(out_dir, "summary.json"), summary)
            prog.event("sweep_done", status="no_champion")
            return summary

        # 按最坏折 min 排名（§十一·11.7 规则 3），mean 兜底 tiebreak。
        ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
        champ_min, champ_mean, champ_tid, champ_seq_len, champ_hp = ranked[0]
        prog.event("champion", trial_id=champ_tid, seq_len=champ_seq_len,
                   screen_corr_min=_num(champ_min), screen_corr_mean=_num(champ_mean))

        # 档1 收官即落一份 partial summary（冠军已知、cert 未开），崩在档2 也能知道选了谁。
        champion_block = {"trial_id": champ_tid, "seq_len": champ_seq_len, "hparams": champ_hp,
                          "screen_corr_min": champ_min, "screen_corr_mean": champ_mean}
        total_cert = len(folds) * len(cert_seeds)
        _dump_json(os.path.join(out_dir, "summary.partial.json"), {
            "run_id": self.cfg.run_id, "stage": "sweep", "status": "screening_done",
            "grid_size": len(grid), "n_screen_folds": n_screen,
            "screen_seeds": [int(s) for s in screen_seeds],
            "champion": champion_block,
            "n_folds": len(folds), "cert_seeds": [int(s) for s in cert_seeds],
            "cert_progress": {"done": 0, "total": total_cert},
        })

        # —— 档2：冠军 × 全折 × 全 seed，**逐折增量落盘** + 每折刷新 partial summary —— #
        fm_path = os.path.join(out_dir, "fold_metrics.csv")
        done_folds = _read_done_folds(fm_path) if self.resume else {}
        fold_writer = _IncrementalCsvWriter(fm_path, append=self.resume)
        prog.event("cert_start", n_folds=len(folds), cert_seeds=[int(s) for s in cert_seeds],
                   total=total_cert, resume=self.resume, n_resumed_folds=len(done_folds))
        corrs: List[float] = []
        r2s: List[float] = []
        done = 0
        try:
            for seed in cert_seeds:
                trainer = None      # 懒构造：整组折都已续跑跳过的 seed 不必建 trainer
                for fold in folds:
                    prior = done_folds.get((int(seed), fold.fold_id))
                    if prior is not None:
                        # 续跑：该 (seed, fold) 已落盘 → 从 CSV 回灌 summary 累加量，跳过重算。
                        if prior.get("status") == "ok":
                            vc = _coerce_float(prior.get("val_corr"))
                            vr = _coerce_float(prior.get("val_r2"))
                            if np.isfinite(vc):
                                corrs.append(vc)
                            if np.isfinite(vr):
                                r2s.append(vr)
                        done += 1
                        prog.event("fold_skipped", seed=int(seed), fold_id=fold.fold_id,
                                   done=done, total=total_cert)
                        continue
                    if trainer is None:
                        trainer = SequenceTrainer(
                            self._spec(), self.adapter, self.cartridge_factory, self.raw_loader,
                            seq_len=champ_seq_len, normalizer_mode=self._normalizer_mode(),
                            hparams=dict(champ_hp), seed=int(seed),
                        )
                    t0 = time.time()
                    r = trainer.run_on_dl_fold(fold, profile_name="sweep_cert")
                    d = r.to_dict()
                    d["random_seed"] = int(seed)
                    fold_writer.write_row(d)          # ← 落盘 + flush 在此（逐折）
                    done += 1
                    if r.status == "ok" and np.isfinite(r.val_corr):
                        corrs.append(float(r.val_corr))
                    if r.status == "ok" and np.isfinite(r.val_r2):
                        r2s.append(float(r.val_r2))
                    prog.event("fold_done", seed=int(seed), fold_id=fold.fold_id,
                               val_corr=_num(r.val_corr), val_r2=_num(r.val_r2), status=r.status,
                               elapsed_sec=round(time.time() - t0, 1), done=done, total=total_cert)
                    # 每折刷新 live snapshot：当前已完成折的 pooled / 最坏折 / R² 读数。
                    summ_run = summarize_folds([{"corr": c} for c in corrs])
                    _dump_json(os.path.join(out_dir, "summary.partial.json"), {
                        "run_id": self.cfg.run_id, "stage": "sweep", "status": "certifying",
                        "grid_size": len(grid), "n_screen_folds": n_screen,
                        "screen_seeds": [int(s) for s in screen_seeds],
                        "champion": champion_block,
                        "n_folds": len(folds), "cert_seeds": [int(s) for s in cert_seeds],
                        "cert_progress": {"done": done, "total": total_cert},
                        "val_corr": summ_run,
                        "val_r2_mean": float(np.mean(r2s)) if r2s else 0.0,
                        "val_r2_min": float(np.min(r2s)) if r2s else 0.0,
                    })
        finally:
            fold_writer.close()

        # —— 交付对齐折：冠军定死后只跑 1 seed，只报不选（AGENTS §十一·11.2/11.6） —— #
        delivery = None
        p = self.cfg.protocol
        if p.delivery_eval_end is not None:
            delivery_seed = int(cert_seeds[0]) if cert_seeds else 42
            delivery_fold_id = max(f.fold_id for f in folds) + 1
            delivery_fold = build_delivery_fold(
                p.rolling_start, p.rolling_end, p.delivery_eval_end,
                embargo=p.embargo, earlystop_frac=p.earlystop_frac,
                fold_id=delivery_fold_id,
            )
            delivery = {"status": "pending", "seed": delivery_seed, "fold_id": delivery_fold.fold_id}
            prior_delivery = done_folds.get((delivery_seed, delivery_fold.fold_id)) if self.resume else None
            if prior_delivery is not None:
                # 续跑时若交付折已落盘，直接从 CSV 回灌 summary，不重训、不影响认证排名。
                delivery = _delivery_summary_from_row(prior_delivery, delivery_seed)
                prog.event("delivery_skipped", seed=delivery_seed, fold_id=delivery_fold.fold_id,
                           val_corr=_num(delivery.get("val_corr")))
            else:
                t0 = time.time()
                trainer = SequenceTrainer(
                    self._spec(), self.adapter, self.cartridge_factory, self.raw_loader,
                    seq_len=champ_seq_len, normalizer_mode=self._normalizer_mode(),
                    hparams=dict(champ_hp), seed=delivery_seed,
                )
                r = trainer.run_on_dl_fold(delivery_fold, profile_name="sweep_delivery")
                d = r.to_dict()
                d["random_seed"] = delivery_seed
                delivery_writer = _IncrementalCsvWriter(fm_path, append=True)
                try:
                    delivery_writer.write_row(d)
                finally:
                    delivery_writer.close()
                delivery = _delivery_summary_from_result(r, delivery_seed)
                prog.event("delivery_done", seed=delivery_seed, fold_id=delivery_fold.fold_id,
                           val_corr=_num(r.val_corr), val_r2=_num(r.val_r2), status=r.status,
                           elapsed_sec=round(time.time() - t0, 1))

        summ_cert = summarize_folds([{"corr": c} for c in corrs])
        summary = {
            "run_id": self.cfg.run_id, "stage": "sweep", "status": "ok",
            "grid_size": len(grid), "n_screen_folds": n_screen,
            "screen_seeds": [int(s) for s in screen_seeds],
            "champion": {"trial_id": champ_tid, "seq_len": champ_seq_len, "hparams": champ_hp,
                         "screen_corr_min": champ_min, "screen_corr_mean": champ_mean},
            "n_folds": len(folds), "cert_seeds": [int(s) for s in cert_seeds],
            "val_corr": summ_cert,    # mean / std / min(最坏折) / max / positive_rate
            "val_r2_mean": float(np.mean(r2s)) if r2s else 0.0,
            "val_r2_min": float(np.min(r2s)) if r2s else 0.0,
        }
        if delivery is not None:
            summary["delivery"] = delivery
        _dump_json(os.path.join(out_dir, "summary.json"), summary)
        prog.event("sweep_done", status="ok",
                   val_corr_mean=_num(summ_cert["mean"]), val_corr_min=_num(summ_cert["min"]),
                   val_r2_mean=_num(float(np.mean(r2s)) if r2s else 0.0),
                   delivery_val_corr=_num(delivery.get("val_corr")) if delivery else None)
        return summary

    def _normalizer_mode(self) -> str:
        # RAW_CHANNELS 已在 adapter 做语义归一，但脊柱仍跑 zscore 统计白化（职责分明，规格 §2.3）；
        # 模式可由 model.hparams["normalizer_mode"] 覆盖（如 identity）。
        return str(self.cfg.model.hparams.get("normalizer_mode", "zscore"))


# ================================================================== #
# 落盘小工具（不引 pandas，标准库即可）
# ================================================================== #

def _dump_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True, default=str)


def _dump_csv(path: str, rows: List[dict]) -> None:
    """一次性把若干行写成 CSV（保留给非长跑路径；长跑路径改用 _IncrementalCsvWriter 逐行落）。"""
    if not rows:
        # 仍写一个空文件占位，便于 resume/审计看到 run 跑过。
        open(path, "w", encoding="utf-8").close()
        return
    fieldnames: List[str] = []
    for r in rows:                       # 列集取并集，保插入序
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ------------------------------------------------------------------ #
# 增量落盘基础设施（防长跑中断打水漂）
# ------------------------------------------------------------------ #

def _now_iso() -> str:
    """秒级本地时间戳（progress.jsonl 用，便于人读"跑到哪一步、花了多久"）。"""
    return datetime.now().isoformat(timespec="seconds")


def _num(x):
    """把一个数清洗成"可写进合法 JSON"的值：有限数 → float，nan/inf/非数 → None。

    仅用于 progress.jsonl 这种希望严格合法 JSONL 的日志；summary*.json 沿用旧 _dump_json
    的行为（允许 NaN，与改造前逐字段一致）。
    """
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _coerce_int(x) -> int:
    """CSV 读回的字符串 → int（失败给 0，仅用于续跑判定"是否已完成"）。"""
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return 0


def _coerce_float(x) -> float:
    """CSV 读回的字符串 → float（失败给 nan，配合 np.isfinite 过滤）。"""
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


class _IncrementalCsvWriter:
    """
    逐行增量落盘的 CSV writer：每写一行立即 flush（+ fsync），崩溃时已写行不留在缓冲区里丢。

    与一次性 ``_dump_csv`` 的产物**逐字段一致**（前提：所有行同一组 key——本项目 trials.csv /
    fold_metrics.csv 均满足，列由确定性网格 / 固定 FoldResult schema 决定），区别只在落盘时机：
    一行一落，而非攒到最后一次性写。

    - **表头**：全新文件时由**首行的 key 顺序**确定（等价于旧 ``_dump_csv`` 在"所有行同 schema"
      下的并集列序），写一次；``append`` 续写时从已有表头读列序、不再重写表头。
    - **续写**（``append=True`` 且文件已存在且非空）：以 ``a`` 模式打开，沿用已有表头列序，
      新行追加在后面——配合 Orchestrator 的 resume 跳过逻辑实现"崩溃续跑不重复"。
    - **零行兜底**：全程没写任何行时，``close()`` 留一个空文件占位（对齐旧 ``_dump_csv([])``）。
    """

    def __init__(self, path: str, *, append: bool = False, fsync: bool = True):
        self.path = path
        self.fsync = fsync
        self.fieldnames: Optional[List[str]] = None
        self._fh = None
        self._writer = None
        self._wrote_any = False
        # 续写：已有非空文件 → 读回表头列序，open 'a' 接着追加。
        if append and os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, "r", encoding="utf-8", newline="") as f:
                header = next(csv.reader(f), [])
            if header:
                self.fieldnames = header
                self._fh = open(path, "a", encoding="utf-8", newline="")
                self._writer = csv.DictWriter(self._fh, fieldnames=self.fieldnames)

    def write_row(self, row: dict) -> None:
        if self._writer is None:
            # 全新文件：用首行的 key 顺序定表头，立即写表头。
            self.fieldnames = list(row.keys())
            self._fh = open(self.path, "w", encoding="utf-8", newline="")
            self._writer = csv.DictWriter(self._fh, fieldnames=self.fieldnames)
            self._writer.writeheader()
        else:
            # 防御：万一某行冒出表头未涵盖的新字段（schema 理论上固定，不该发生），
            # 记一条 stderr 警告但不抛——优先保证"已写不丢、长跑不崩"。
            extra = [k for k in row.keys() if k not in self.fieldnames]
            if extra:
                print(f"[incr-dump] 警告: {os.path.basename(self.path)} 出现表头外字段 {extra}，已忽略",
                      file=sys.stderr)
        self._writer.writerow({k: row.get(k) for k in self.fieldnames})
        self._wrote_any = True
        self._fh.flush()
        if self.fsync:
            os.fsync(self._fh.fileno())

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            if self.fsync:
                os.fsync(self._fh.fileno())
            self._fh.close()
            self._fh = None
        elif not self._wrote_any:
            # 从未打开过文件（全新且零行）→ 留空文件占位，与旧 _dump_csv([]) 行为一致。
            open(self.path, "w", encoding="utf-8").close()


class _ProgressLog:
    """
    逐事件 append 的 JSONL 进度日志：每行一个事件、落盘即 flush，可 ``tail -f`` 实时盯进度。

    定位：在 trials.csv / fold_metrics.csv 之外，提供一条**人读友好的时间线**——每个 trial /
    每折何时完成、耗时多久、当前读数多少、跑到第几 / 共几步。崩溃后它是"卡在哪一步"的权威记录。
    以 ``a`` 模式打开（续跑时接着记，不覆盖历史 session）。
    """

    def __init__(self, path: str, *, fsync: bool = True):
        self.path = path
        self.fsync = fsync
        self._fh = open(path, "a", encoding="utf-8")

    def event(self, name: str, **fields) -> None:
        rec = {"ts": _now_iso(), "event": name, **fields}
        # allow_nan=False + 上游已用 _num 清洗 → 保证每行都是合法 JSON。
        self._fh.write(json.dumps(rec, ensure_ascii=False, allow_nan=False, default=str) + "\n")
        self._fh.flush()
        if self.fsync:
            os.fsync(self._fh.fileno())

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            if self.fsync:
                os.fsync(self._fh.fileno())
            self._fh.close()
            self._fh = None


def _read_done_trials(path: str) -> Dict[int, dict]:
    """读已存在的 trials.csv → ``{trial_id: row_dict}``（续跑跳过判定用）。文件缺失/空 → {}。"""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {}
    out: Dict[int, dict] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                out[int(row["trial_id"])] = row
            except (KeyError, ValueError, TypeError):
                continue
    return out


def _read_done_folds(path: str) -> Dict[tuple, dict]:
    """读已存在的 fold_metrics.csv → ``{(seed, fold_id): row_dict}``（续跑跳过判定用）。"""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {}
    out: Dict[tuple, dict] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                out[(int(row["random_seed"]), int(row["fold_id"]))] = row
            except (KeyError, ValueError, TypeError):
                continue
    return out


def _delivery_summary_from_result(r, seed: int) -> dict:
    """把交付折 FoldResult 压成 summary.delivery；只读报告，不参与 SWEEP 排名。"""
    return {
        "status": r.status,
        "seed": int(seed),
        "fold_id": int(r.fold_id),
        "train_start": int(r.train_start),
        "train_end": int(r.train_end),
        "val_start": int(r.val_start),
        "val_end": int(r.val_end),
        "n_train_days": int(r.n_train_days),
        "n_val_days": int(r.n_val_days),
        "val_corr": float(r.val_corr),
        "val_mse": float(r.val_mse),
        "val_r2": float(r.val_r2),
        "daily_corr_mean": float(r.daily_corr_mean),
        "daily_corr_std": float(r.daily_corr_std),
        "best_epoch": int(r.best_epoch),
        "runtime_sec": float(r.runtime_sec),
        "error_msg": r.error_msg,
    }


def _delivery_summary_from_row(row: dict, seed: int) -> dict:
    """从已落盘的 delivery 行回灌 summary.delivery，供 --resume 跳过重训。"""
    return {
        "status": row.get("status", ""),
        "seed": int(seed),
        "fold_id": _coerce_int(row.get("fold_id")),
        "train_start": _coerce_int(row.get("train_start")),
        "train_end": _coerce_int(row.get("train_end")),
        "val_start": _coerce_int(row.get("val_start")),
        "val_end": _coerce_int(row.get("val_end")),
        "n_train_days": _coerce_int(row.get("n_train_days")),
        "n_val_days": _coerce_int(row.get("n_val_days")),
        "val_corr": _coerce_float(row.get("val_corr")),
        "val_mse": _coerce_float(row.get("val_mse")),
        "val_r2": _coerce_float(row.get("val_r2")),
        "daily_corr_mean": _coerce_float(row.get("daily_corr_mean")),
        "daily_corr_std": _coerce_float(row.get("daily_corr_std")),
        "best_epoch": _coerce_int(row.get("best_epoch")),
        "runtime_sec": _coerce_float(row.get("runtime_sec")),
        "error_msg": row.get("error_msg", ""),
    }


# ================================================================== #
# CLI —— 组装一份 RunConfig 并 run（torch-free smoke：参考卡带 + 真实/抽样数据）
# ================================================================== #

def _build_run_config(args):
    from model_config import ModelKind, ModelConfig
    from adapter_config import AdapterKind, AdapterConfig
    from protocol_config import ProtocolConfig
    from search_config import SearchConfig
    from exec_config import ExecConfig
    from run_config import assemble_run_config

    stage = Stage(args.stage)
    profile = ProfileKind.SINGLE_SPLIT if stage == Stage.SEARCH else ProfileKind.EXPANDING
    max_folds = 1 if stage == Stage.SEARCH else args.max_folds

    model = ModelConfig(ModelKind(args.model), hparams=_parse_hparams(args.hparams))
    adapter_kind = AdapterKind(args.adapter)
    adapter = (AdapterConfig(adapter_kind, columns=tuple(args.columns.split(",")))
               if adapter_kind == AdapterKind.IDENTITY and args.columns
               else AdapterConfig(adapter_kind))
    protocol = ProtocolConfig(
        stage, profile, args.start, args.end,
        val_window=args.val_window, step=args.step, embargo=args.embargo,
        min_train_days=args.min_train_days,
        max_folds=max_folds, fold_select=args.fold_select,
        delivery_eval_end=args.delivery_eval_end,
    )
    search = SearchConfig(n_trials=args.trials, search_overrides=_parse_grid_overrides(args))
    exec_ = ExecConfig(
        seeds=tuple(int(s) for s in args.seeds.split(",")),
        device=args.device, out_dir=args.out_dir,
        target_mode=args.target_mode,
        trad_preds_root=args.trad_preds_root,
        trad_pred_col=args.trad_pred_col,
        trad_cache_dir=args.trad_cache_dir,
    )
    return assemble_run_config(args.run_id, model, adapter, protocol, search, exec_)


def _parse_grid_overrides(args) -> dict:
    """从 --grid-* 收窄各结构旋钮的 SWEEP 网格（空 = 用卡带 search_space 全集展开）。"""
    overrides: dict = {}
    for knob, raw in (("seq_len", args.grid_seq_len),
                      ("hidden_size", args.grid_hidden),
                      ("num_layers", args.grid_layers)):
        if raw:
            overrides[knob] = {"type": "choice", "values": [int(x) for x in raw.split(",")]}
    return overrides


def _parse_hparams(s: str) -> dict:
    """解析 ``k=v,k2=v2``（v 尝试 int→float→str）。"""
    out: dict = {}
    if not s:
        return out
    for kv in s.split(","):
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        for cast in (int, float):
            try:
                out[k.strip()] = cast(v); break
            except ValueError:
                continue
        else:
            out[k.strip()] = v
    return out


def _wrap_max_symbols(h5dir: str, max_symbols: int):
    """包一层 raw_loader：抽前 N 个 symbol，控参考卡带 gather_all 的内存（CLI smoke 用）。"""
    from dl import MeowDataLoader
    loader = MeowDataLoader(h5dir)

    def _load(dates):
        df = loader.loadDates(list(dates))
        keep = sorted(df["symbol"].unique())[:max_symbols]
        return df[df["symbol"].isin(keep)].copy()
    return _load


def main(argv=None):
    ap = argparse.ArgumentParser(description="DL Orchestrator（torch-free smoke）")
    ap.add_argument("--run-id", default="20260531_search_refpool_smoke_v1")
    ap.add_argument("--stage", default="search", choices=["search", "validation", "sweep"],
                    help="sweep=一命令两档（主路径，§十一·11.6）；search/validation 为旧两段调试用")
    ap.add_argument("--model", default="reference_pool",
                    help="reference_pool（torch-free smoke）/ gru（433 时序基线）/ xsection（截面主攻）/ reference_zero|last")
    ap.add_argument("--adapter", default="raw_channels",
                    help="raw_channels / feature_433（gru/xsection 必须）/ identity")
    ap.add_argument("--columns", default="", help="identity adapter 的列（逗号分隔）")
    ap.add_argument("--start", type=int, default=20230601)
    ap.add_argument("--end", type=int, default=20230731)
    ap.add_argument("--delivery-eval-end", type=int, default=None,
                    help="SWEEP 交付对齐折 eval 末日；例如 --end 20231130 --delivery-eval-end 20231229")
    ap.add_argument("--val-window", type=int, default=5)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--min-train-days", type=int, default=20)
    ap.add_argument("--embargo", type=int, default=1,
                    help="训练区与打分区之间禁飞天数（日内标签下 1 日已等价 purge）")
    ap.add_argument("--fold-select", default="first", choices=["first", "recent"],
                    help="recent=最近若干折倒贴 rolling_end（新协议主路径）；first=最早若干折（调试）")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="PyTorch 卡带训练设备；整晚 GPU 跑传 cuda")
    ap.add_argument("--max-folds", type=int, default=3, help="validation / sweep 选型折数（新协议主路径为 3）")
    ap.add_argument("--trials", type=int, default=4, help="仅 search 用")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--hparams", default="", help="k=v,k2=v2（如 dropout=0.2,weight_decay=0.001,max_epochs=30）")
    ap.add_argument("--grid-seq-len", default="", help="SWEEP 收窄 seq_len 网格，逗号分隔（如 16,32）")
    ap.add_argument("--grid-hidden", default="", help="SWEEP 收窄 hidden_size 网格（如 32,64）")
    ap.add_argument("--grid-layers", default="", help="SWEEP 收窄 num_layers 网格（如 1,2）")
    ap.add_argument("--out-dir", default="results/dl")
    ap.add_argument("--h5dir", default="data")
    ap.add_argument("--feature-dir", default=DEFAULT_FEATURE_DIR,
                    help="实验链特征缓存根目录（feature_433 优先从这里读取）")
    ap.add_argument("--max-symbols", type=int, default=20,
                    help="抽前 N 个 symbol 控内存（参考卡带 gather_all 用）；<=0 = 不抽样")
    ap.add_argument("--resume", action="store_true",
                    help="续跑：复用同 run-id 目录下已落盘的 trial/(seed,fold)，跳过已完成的不重算（SWEEP）")
    ap.add_argument("--dump-preds", action="store_true",
                    help="把每折 scoring 段逐票预测落 <out_dir>/<run_id>/preds/（date,symbol,interval,label,pred），"
                         "供「DL↔传统预测相关性」等离线分析；默认关、零开销")
    ap.add_argument("--target-mode", default="raw", choices=["raw", "residual_trad"],
                    help="训练目标模式：raw=直接学 fret12；residual_trad=学传统代表残差 y-pred_trad")
    ap.add_argument("--trad-preds-root", default="",
                    help="target-mode=residual_trad 时必填：传统逐票预测根目录（例如 results/trad_dl_protocol/.../run_id）")
    ap.add_argument("--trad-pred-col", default="pred_blend",
                    help="传统预测列名；当前锁定融合代表默认用 pred_blend")
    ap.add_argument("--trad-cache-dir", default="results/dl/_trad_residual_cache",
                    help="残差训练时训练区传统预测缓存目录（gitignore，下次同窗复用）")
    args = ap.parse_args(argv)

    rc = _build_run_config(args)
    raw_loader = None
    if args.max_symbols and args.max_symbols > 0:
        raw_loader = _wrap_max_symbols(args.h5dir, args.max_symbols)
    orch = Orchestrator(
        rc,
        raw_loader=raw_loader,
        h5dir=args.h5dir,
        feature_dir=args.feature_dir,
        resume=args.resume,
        dump_preds=args.dump_preds,
    )
    summary = orch.run()
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return summary


if __name__ == "__main__":
    main()
