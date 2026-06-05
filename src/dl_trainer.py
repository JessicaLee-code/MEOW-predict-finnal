"""
SequenceTrainer —— DL 序列模型的 Trainer 骨架（脊柱编排层，torch-free）

接 ``trainer.BaseTrainer`` 扩展点（与传统 ``TabularTrainer`` 同构），把"可换卡带"的
data pipeline + model cartridge 编排成一次完整的 fold 训练-评测，产出与 leaderboard
兼容的 ``FoldResult``。

编排顺序（规格 §2.2）：

    raw_loader(dates) → InputAdapter.build（逐日）→ SequenceArrays
      → Normalizer.fit-on-train → SequenceDataset（惰性 [B,L,C]）
      → ModelCartridge.fit(train_core, earlystop) / predict(scoring)
      → dl_protocol 4 指标 → FoldResult

解耦策略：本层**只靠鸭子类型**认 adapter（有 ``channels`` / ``build``）与 cartridge
（有 ``fit`` / ``predict``），不 import ``models/`` / ``config/`` 具体类——保持 src 层
torch-free 且不反向依赖卡带目录。具体卡带/适配器由 Orchestrator 用 registry 构造后注入。

防泄漏的三道物理保证全部落在本编排里：
1. **窗口不跨日不跨票**（WindowIndexer，在 SequenceDataset 内）；
2. **标签因果对齐窗末**（同上）；
3. **Normalizer 只用训练区 fit**（本层显式只喂 train 区特征，再套到 scoring）。
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from trainer import BaseTrainer, FoldData, FoldResult
from sequence_dataset import (
    Normalizer,
    SequenceArrays,
    SequenceDataset,
    build_sequence_arrays,
    subset_by_dates,
)
from dl_protocol import DLFold, corr_gap, evaluate_prediction_bundle, split_train_earlystop


class SequenceTrainer(BaseTrainer):
    """
    序列模型 Trainer。两条入口：

    - ``run_on_dl_fold(fold: DLFold)``：DL 主路径，吃 protocol 切好的四段折（含 earlystop）。
    - ``run_fold(fold_data: FoldData)``：BaseTrainer 兼容入口，用 ``earlystop_frac`` 从
      ``train_dates`` 尾段切 earlystop 再委托（让调度层可像传统 trainer 一样调它）。
    """

    def __init__(
        self,
        spec: Dict,
        adapter,
        cartridge_factory: Callable[[], object],
        raw_loader: Callable[[Sequence[int]], pd.DataFrame],
        seq_len: int,
        normalizer_mode: str = "zscore",
        hparams: Optional[Dict] = None,
        seed: int = 42,
        earlystop_frac: float = 0.15,
    ):
        """
        - ``spec``: 给 FoldResult 的元信息（experiment_id / model_type / feature_set /
          target_type / postprocess_type / notes）。
        - ``adapter``: 已构造的 InputAdapter 实例。
        - ``cartridge_factory``: 无参可调用，**每折新建**一个 ModelCartridge（避免折间状态串）。
        - ``raw_loader``: ``dates -> raw DataFrame``（含 META + raw 列 + fret12）。
        - ``seq_len`` / ``normalizer_mode`` / ``hparams`` / ``seed`` / ``earlystop_frac``: 见上。
        """
        super().__init__(spec)
        self.adapter = adapter
        self.cartridge_factory = cartridge_factory
        self.raw_loader = raw_loader
        self.seq_len = int(seq_len)
        self.normalizer_mode = normalizer_mode
        self.hparams = dict(hparams or {})
        self.seed = int(seed)
        self.earlystop_frac = float(earlystop_frac)

    # ---- 数据 ---- #
    def _build_arrays(self, dates: Sequence[int]) -> SequenceArrays:
        """
        构造某段日期的 ``SequenceArrays``。

        默认路径仍是：
        ``raw_loader(dates) -> build_sequence_arrays(raw_df, adapter)``。

        但对 ``FeatureAdapter`` 这类已经能**按日期直接从磁盘特征缓存取数**的适配器，
        优先走 ``adapter.load_sequence_arrays(dates)``，原因有二：
        1. rolling / sweep 多折高度重叠时，433 特征是最贵的 CPU 开销，不该每折重算；
        2. 若仍先把整段 raw 读进来，再让适配器自己忽略 raw 去读缓存，会产生“双份 IO +
           双份内存”的纯浪费。

        兜底策略：
        - 若缓存入口不存在，或本机尚未建好 ``data/features/``，则回退到旧的 raw 现算链；
        - 这样测试/提交桥接仍可继续工作，但正式实验只要缓存一旦备好，就自然走快路径。
        """
        load_cached = getattr(self.adapter, "load_sequence_arrays", None)
        if callable(load_cached):
            try:
                return load_cached(dates)
            except FileNotFoundError:
                # 仅把“缓存尚未准备好”视为可兜底情形；其它异常（列错位、manifest 损坏、
                # 读文件失败）都应该原样抛出，避免 silently 跑回慢路径掩盖问题。
                pass
        raw = self.raw_loader(list(dates))
        return build_sequence_arrays(raw, self.adapter)

    # ---- DL 主入口 ---- #
    def run_on_dl_fold(self, fold: DLFold, profile_name: str = "dl") -> FoldResult:
        start_ts = time.time()
        try:
            # 分阶段计时：把每折 wall-clock 拆成「数据准备(GPU 空) vs fit(GPU 忙)」，
            # 一行打到 stderr（flush，不被重定向缓冲），用来定位"GPU 起来一段又长时间
            # 沉默"到底沉默在哪——是读盘 / 归一化 / 建集 / 还是算指标。
            _t: Dict[str, float] = {}
            _m = time.time()
            # 1) 直接按 core / earlystop / scoring 三段**分别**现算，避免"先建整训练帧再 subset"
            #    造成的 14GB + 12GB 双份共存（119 天全票折会因此 OOM）。三段日期互斥，
            #    core+es 并集 = 原训练区，统计口径不变、无泄漏。
            core_arrays = self._build_arrays(fold.train_core_dates)
            # earlystop 可能为空：用空切分得到形状对的空 arrays（_build_arrays 不接受空日期）。
            es_arrays = (self._build_arrays(fold.earlystop_dates)
                         if fold.earlystop_dates else subset_by_dates(core_arrays, ()))
            score_arrays = self._build_arrays(fold.scoring_dates)
            _t["load"] = time.time() - _m; _m = time.time()
            # 2) Normalizer 只用训练区(core+es)统计，**分块 fit、不 concatenate、不物化整张
            #    副本**；scoring 不参与统计，无泄漏。
            normalizer = Normalizer(self.normalizer_mode).fit_chunked(
                [core_arrays.features, es_arrays.features])
            _t["norm"] = time.time() - _m; _m = time.time()

            # 3) 三个 dataset 各自**原地白化**(own_features=True)：本折独占这三份 arrays，
            #    不再每个 dataset 另造一份 [N,C] _feats 副本。
            train_core_ds = SequenceDataset(core_arrays, self.seq_len, normalizer, own_features=True)
            earlystop_ds = SequenceDataset(es_arrays, self.seq_len, normalizer, own_features=True)
            scoring_ds = SequenceDataset(score_arrays, self.seq_len, normalizer, own_features=True)
            _t["build_ds"] = time.time() - _m; _m = time.time()

            # 3) 每折新建卡带，fit（earlystop 看尾段、绝不碰 scoring），predict。
            cartridge = self.cartridge_factory()
            record = cartridge.fit(train_core_ds, earlystop_ds, self.hparams, self.seed)
            _t["fit_GPU"] = time.time() - _m; _m = time.time()
            pred_val = cartridge.predict(scoring_ds)
            pred_train = cartridge.predict(train_core_ds)
            _t["predict"] = time.time() - _m; _m = time.time()

            # 4) 4 指标（脊柱在 predict 之后算一次；scoring 在此前一次不碰）。
            val_lf = scoring_ds.label_frame()        # date/symbol/interval/label，行序与 pred_val 对齐
            vm = evaluate_prediction_bundle(val_lf, pred_val)
            tm = evaluate_prediction_bundle(train_core_ds.label_frame(), pred_train)
            _t["metrics"] = time.time() - _m

            # 4b) 逐票预测落盘（仅当 spec 带 dump_preds_dir；默认不落，零开销）。供「DL↔传统
            #     预测相关性」等离线分析；GRU/截面卡带都把预测映射回 SequenceDataset 窗口序，
            #     故此处对两种卡带统一：val_lf 的 (date,symbol,interval,label) 与 pred_val 行对齐。
            dump_dir = self.spec.get("dump_preds_dir")
            if dump_dir:
                self._dump_fold_preds(dump_dir, fold.fold_id, profile_name, val_lf, pred_val)
            _gpu_busy = _t["fit_GPU"] + _t["predict"]
            _total = sum(_t.values()) or 1.0
            print(
                f"[fold {fold.fold_id} timing] "
                + " ".join(f"{k}={v:.1f}s" for k, v in _t.items())
                + f" | GPU忙占比≈{100.0 * _gpu_busy / _total:.0f}%"
                + f" (n_train_days={len(fold.train_dates)})",
                file=sys.stderr, flush=True,
            )

            notes = self.spec.get("notes", "")
            best_epoch = getattr(record, "best_epoch", 0)
            n_epochs = getattr(record, "n_epochs", 0)
            # 曲线序列化为 JSON 字符串（保留 6 位小数，节省空间）
            train_curve_json = json.dumps(
                [round(v, 6) for v in (record.train_curve or [])]) if hasattr(record, "train_curve") else "[]"
            es_curve_json = json.dumps(
                [round(v, 6) for v in (record.earlystop_curve or [])]) if hasattr(record, "earlystop_curve") else "[]"
            return FoldResult(
                profile_name=profile_name,
                fold_id=fold.fold_id,
                experiment_id=self.spec.get("experiment_id", "dl_run"),
                feature_set=self.spec.get("feature_set", "dl_channels"),
                model_type=self.spec.get("model_type", "sequence"),
                target_type=self.spec.get("target_type", "raw"),
                postprocess_type=self.spec.get("postprocess_type", "none"),
                train_corr=float(tm["corr"]), val_corr=float(vm["corr"]),
                train_mse=float(tm["mse"]), val_mse=float(vm["mse"]),
                train_r2=float(tm["r2"]), val_r2=float(vm["r2"]),
                daily_corr_mean=float(vm["daily_corr_mean"]),
                daily_corr_std=float(vm["daily_corr_std"]),
                train_val_corr_gap=corr_gap(tm, vm),
                runtime_sec=float(time.time() - start_ts),
                train_start=fold.train_start, train_end=fold.train_end,
                val_start=fold.val_start, val_end=fold.val_end,
                n_train_days=len(fold.train_dates), n_val_days=len(fold.scoring_dates),
                status="ok", error_msg="",
                notes=notes,
                best_epoch=best_epoch,
                n_epochs=n_epochs,
                train_curve=train_curve_json,
                earlystop_curve=es_curve_json,
            )
        except Exception as e:  # 单折失败不拖垮整轮，记 error 供 resume（同 TabularTrainer）。
            nan = float("nan")
            return FoldResult(
                profile_name=profile_name,
                fold_id=fold.fold_id,
                experiment_id=self.spec.get("experiment_id", "dl_run"),
                feature_set=self.spec.get("feature_set", "dl_channels"),
                model_type=self.spec.get("model_type", "sequence"),
                target_type=self.spec.get("target_type", "raw"),
                postprocess_type=self.spec.get("postprocess_type", "none"),
                train_corr=nan, val_corr=nan, train_mse=nan, val_mse=nan,
                train_r2=nan, val_r2=nan, daily_corr_mean=nan, daily_corr_std=nan,
                train_val_corr_gap=nan, runtime_sec=float(time.time() - start_ts),
                train_start=fold.train_start, train_end=fold.train_end,
                val_start=fold.val_start, val_end=fold.val_end,
                n_train_days=len(fold.train_dates), n_val_days=len(fold.scoring_dates),
                status="error", error_msg=str(e)[:500],
                notes=self.spec.get("notes", ""),
            )

    # ---- 逐票预测落盘（离线分析用，默认关） ---- #
    def _dump_fold_preds(
        self,
        dump_dir: str,
        fold_id: int,
        profile_name: str,
        val_lf: pd.DataFrame,
        pred_val: np.ndarray,
    ) -> None:
        """
        把某折 scoring 段的逐票预测落成 CSV：``date,symbol,interval,label,pred``。

        - 键 ``(date,symbol,interval)`` 用整数编码（与提交链 ``MeowEngine`` 同源），
          供离线 join 传统侧预测算「DL↔传统」相关性。
        - 行序 = ``SequenceDataset`` 窗口序（GRU/截面卡带的 predict 均映射回此序），
          故 ``pred_val`` 与 ``val_lf`` 严格行对齐。
        - 失败不抛（落盘是旁路诊断，绝不拖垮训练）：打 stderr 警告即返回。
        """
        try:
            os.makedirs(dump_dir, exist_ok=True)
            out = val_lf.copy()
            # 标签列统一改名为 ``label``（label_frame 用 TARGET_COL=fret12），与传统侧
            # dump_trad_preds.py 落盘列名对齐，分析脚本可直接按 (date,symbol,interval) join。
            meta = ("date", "symbol", "interval")
            out = out.rename(columns={c: "label" for c in out.columns if c not in meta})
            out["pred"] = np.asarray(pred_val, dtype=np.float64).reshape(-1)
            fname = f"preds_{profile_name}_fold{fold_id}_seed{self.seed}.csv"
            out.to_csv(os.path.join(dump_dir, fname), index=False)
        except Exception as e:  # 旁路诊断失败不影响主流程
            print(f"[dump-preds] 警告: 落盘失败 fold{fold_id} seed{self.seed}: {e}",
                  file=sys.stderr, flush=True)

    # ---- BaseTrainer 兼容入口 ---- #
    def run_fold(self, fold_data: FoldData) -> FoldResult:
        core, es = split_train_earlystop(fold_data.train_dates, self.earlystop_frac)
        fold = DLFold(
            fold_id=fold_data.fold_id,
            train_core_dates=tuple(core),
            earlystop_dates=tuple(es),
            embargo_dates=tuple(),                 # FoldData 生成时已在 train/val 间留 embargo
            scoring_dates=tuple(fold_data.val_dates),
        )
        return self.run_on_dl_fold(fold, profile_name=fold_data.profile_name)
