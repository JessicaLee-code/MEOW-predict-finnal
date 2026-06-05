"""
传统残差目标辅助模块 —— 给 DL 训练提供「传统预测残差」标签。

目标
----
把当前 DL 训练目标从原始 ``fret12`` 改成：

    residual = fret12 - y_trad

其中 ``y_trad`` 来自锁定传统代表（当前是 ``pred_blend``）。模块职责只做三件事：

1. 读取**已落盘的传统 OOS 逐票预测**（cert / delivery）并按 ``(date,symbol,interval)`` 建索引；
2. 对某个 DL 折，若训练区没有现成 OOS 传统预测，则**一次性**重训传统代表并缓存训练区逐票预测；
3. 把传统预测按键对齐回 ``SequenceArrays``，生成：
   - 训练用残差标签 ``label_res = label - pred_trad``；
   - 评测 / 落盘用最终预测 ``pred_final = pred_trad + pred_dl_res``。

为什么要单独成模块
------------------
残差训练不是新模型结构，而是「目标构造 + 对齐」问题。把这块逻辑放在独立模块里，
可以保证：

- ``SequenceTrainer`` 只负责调用，不被大量 IO/缓存/merge 代码淹没；
- 现有 ``xsection_raw`` / ``gru`` 卡带完全不动；
- 将来如果把传统代表从 ``pred_blend`` 换成别的列，只改一个地方。

关于训练区传统预测的口径
----------------------
当前仓库已有传统 **scoring 区** 的无泄漏 OOS 预测（`results/trad_dl_protocol/...`），
但没有覆盖每个 DL 折训练区所有日期的 OOS 预测。为了先验证「残差训练」这条路线是否
值得继续，这里采用一个**最小可行**口径：

- **评分区**：严格使用已有 OOS 传统预测（完全无泄漏）；
- **训练区 / earlystop 区**：按该折训练窗口重训一次传统代表，并在同一训练窗口内出
  逐票预测后缓存，用它构造残差标签。

这意味着训练区传统预测是 in-sample 分解，不是 OOS；但它只用于**训练目标构造**，
不参与最终评分。最终 cert / delivery 指标仍用 OOS 传统预测恢复 ``pred_final`` 后评估。
若这条路线跑出明显增益，再决定是否补更严格的全时段 OOF 传统预测地基。
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from dl_protocol import DLFold
from sequence_dataset import META_COLS, SequenceArrays


_META_COLS = list(META_COLS)


@dataclass(frozen=True)
class ResidualBundle:
    """某一段数组对齐后的残差标签与传统预测。"""

    arrays_residual: SequenceArrays
    trad_window: np.ndarray


class TraditionalResidualProvider:
    """
    传统残差目标提供器。

    用法：
    1. 初始化时给定传统结果根目录（例如 ``results/trad_dl_protocol/.../v2``）；
    2. 对某个 ``DLFold`` 调 ``load_fold_payload``；
    3. 返回 train / earlystop / scoring 三段各自对齐好的传统预测 DataFrame。

    目录约定
    --------
    ``trad_preds_root`` 形如：

    ``results/trad_dl_protocol/<run_id>/``
        └─ preds/
           ├─ fold0/fold0_preds.csv
           ├─ fold1/fold1_preds.csv
           ├─ fold2/fold2_preds.csv
           └─ delivery/delivery_preds.csv

    评分区读取这些**现成 OOS 文件**；训练区预测缓存落到 ``trad_cache_dir`` 下。
    """

    def __init__(
        self,
        trad_preds_root: str,
        trad_pred_col: str = "pred_blend",
        trad_cache_dir: str = "results/dl/_trad_residual_cache",
        data_dir: str = "data",
    ):
        self.trad_preds_root = Path(trad_preds_root).resolve()
        self.trad_pred_col = trad_pred_col
        self.trad_cache_dir = Path(trad_cache_dir).resolve()
        self.data_dir = str(Path(data_dir).resolve())
        self._score_index = self._build_score_index()
        self._engine = None
        os.makedirs(self.trad_cache_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 对外主入口
    # ------------------------------------------------------------------
    def load_fold_payload(self, fold: DLFold) -> Dict[str, pd.DataFrame]:
        """
        读取某折残差训练所需的传统预测。

        返回：
        - ``train``：覆盖 ``fold.train_dates`` 的传统预测（训练区缓存，可含 in-sample）
        - ``train_core``：train 过滤到 core 日期
        - ``earlystop``：train 过滤到 earlystop 日期
        - ``scoring``：严格 OOS 的评分区传统预测
        """
        train_all = self._load_train_preds(fold)
        scoring = self._load_scoring_preds(fold)
        return {
            "train": train_all,
            "train_core": self._filter_dates(train_all, fold.train_core_dates),
            "earlystop": self._filter_dates(train_all, fold.earlystop_dates),
            "scoring": scoring,
        }

    def build_residual_bundle(
        self,
        arrays: SequenceArrays,
        trad_df: pd.DataFrame,
    ) -> ResidualBundle:
        """
        把传统预测按键对齐到 ``SequenceArrays`` 上，生成残差标签。

        返回：
        - ``arrays_residual``：与原 arrays 同结构，但 ``labels`` 已替换为 ``label - trad_pred``
        - ``trad_window``：与 arrays 行序对齐的传统逐行预测，供窗口末行 gather
        """
        key_df = pd.DataFrame({
            "date": arrays.dates,
            "symbol": arrays.symbols,
            "interval": arrays.intervals,
        })
        merged = key_df.merge(
            trad_df.loc[:, _META_COLS + [self.trad_pred_col]],
            on=_META_COLS,
            how="left",
            sort=False,
            validate="one_to_one",
        )
        if merged[self.trad_pred_col].isna().any():
            miss = int(merged[self.trad_pred_col].isna().sum())
            raise ValueError(
                "传统预测与 DL 样本键未完全对齐："
                f"缺失 {miss} 行（请检查折边界或传统预测目录）"
            )
        trad_pred = merged[self.trad_pred_col].to_numpy(dtype=np.float32, copy=False)
        residual_labels = arrays.labels.astype(np.float32, copy=False) - trad_pred
        arrays_residual = SequenceArrays(
            features=arrays.features,
            labels=residual_labels.astype(np.float32, copy=False),
            dates=arrays.dates,
            symbols=arrays.symbols,
            intervals=arrays.intervals,
            channels=list(arrays.channels),
            has_label=arrays.has_label,
        )
        return ResidualBundle(arrays_residual=arrays_residual, trad_window=trad_pred)

    # ------------------------------------------------------------------
    # 评分区：严格读取已落盘 OOS 传统预测
    # ------------------------------------------------------------------
    def _build_score_index(self) -> Dict[Tuple[int, int], Path]:
        """
        扫描 ``preds/*_wide.csv``，按 ``(min_date, max_date)`` 建索引。

        之所以不用 fold_id 建索引，是因为 recent 2 折与 recent 3 折的 fold 编号会变化；
        真正稳定的是评分窗口本身。

        这里只认 ``*_wide.csv``，不读 ``*_long.csv``，原因有两个：

        1. residual 训练只需要「每个 (date,symbol,interval) 一行」的 blend 预测，
           ``wide`` 表正好就是这个形态；
        2. ``long`` 表是一成员一行，体量通常是 ``wide`` 的数倍，且与 ``wide`` 混读会把
           同一批日期重复放大，既浪费内存，也会在后续 merge 时制造重复键。
        """
        preds_dir = self.trad_preds_root / "preds"
        if not preds_dir.exists():
            raise FileNotFoundError(f"传统预测目录不存在：{preds_dir}")
        index: Dict[Tuple[int, int], Path] = {}
        for path in preds_dir.rglob("*_wide.csv"):
            frame = pd.read_csv(path, usecols=["date"])
            if frame.empty:
                continue
            key = (int(frame["date"].min()), int(frame["date"].max()))
            index[key] = path
        if not index:
            raise FileNotFoundError(f"传统预测目录下未找到可用 CSV：{preds_dir}")
        return index

    def _load_scoring_preds(self, fold: DLFold) -> pd.DataFrame:
        """
        评分区传统预测读取策略。

        旧版假设“评分窗口必须与某个已落盘传统预测文件的起止日完全一致”，这只适用于
        20×20 大块 cert / delivery。现在要支持更密的 OOF 小块库（如 5×5），因此改成：

        - 取所有与 ``fold.scoring_dates`` 有交集的传统预测文件；
        - 按日期精确过滤到评分日期集合；
        - 拼接后校验键唯一 / 覆盖完整。

        这样 residual 评测既能吃原来的 20 日 cert 大块，也能吃后续更密的 OOF 小块库。
        """
        score_dates = set(int(d) for d in fold.scoring_dates)
        frames = []
        for (win_start, win_end), path in sorted(self._score_index.items()):
            if win_end < min(score_dates) or win_start > max(score_dates):
                continue
            frame = pd.read_csv(path, usecols=_META_COLS + ["label", self.trad_pred_col])
            self._validate_pred_frame(frame, path)
            part = frame[frame["date"].isin(score_dates)].copy()
            if not part.empty:
                frames.append(part.loc[:, _META_COLS + ["label", self.trad_pred_col]])
        if not frames:
            raise FileNotFoundError(
                "未找到与当前评分日期有交集的传统预测："
                f"score_dates=({min(score_dates)}, {max(score_dates)})"
            )
        out = pd.concat(frames, axis=0, ignore_index=True)
        self._validate_pred_frame(out, "<concat_scoring_preds>")
        return out

    # ------------------------------------------------------------------
    # 训练区：按折训练窗口重训一次传统代表，落缓存
    # ------------------------------------------------------------------
    def _load_train_preds(self, fold: DLFold) -> pd.DataFrame:
        """
        训练区传统预测优先级：

        1. **优先**复用现成 OOS 传统预测（更严谨，也更快）：
           把 ``trad_preds_root/preds/*.csv`` 中与 ``fold.train_dates`` 有交集的窗口拼起来，
           再过滤到训练日期交集；
        2. 若训练区一行 OOS 传统预测都没有（例如最早的 fold0），才退回到
           `_load_or_build_train_preds_fallback` 走“该训练窗重训一次传统代表”的慢路径。

        这样最近两折 residual 实验就能直接吃现成 OOS 传统预测，既符合用户原意，也避开
        每折首轮构造缓存的重 CPU 开销。
        """
        train_oos = self._load_train_preds_from_oos(fold)
        if not train_oos.empty:
            return train_oos
        return self._load_or_build_train_preds_fallback(fold)

    def _load_train_preds_from_oos(self, fold: DLFold) -> pd.DataFrame:
        train_dates = set(int(d) for d in fold.train_dates)
        if not train_dates:
            return pd.DataFrame(columns=_META_COLS + ["label", self.trad_pred_col])
        frames = []
        for (win_start, win_end), path in sorted(self._score_index.items()):
            # 任一评分窗口只要与训练日期有交集，就读入并按日期精确过滤。
            if win_end < min(train_dates) or win_start > max(train_dates):
                continue
            frame = pd.read_csv(path, usecols=_META_COLS + ["label", self.trad_pred_col])
            self._validate_pred_frame(frame, path)
            part = frame[frame["date"].isin(train_dates)].copy()
            if not part.empty:
                frames.append(part.loc[:, _META_COLS + ["label", self.trad_pred_col]])
        if not frames:
            return pd.DataFrame(columns=_META_COLS + ["label", self.trad_pred_col])
        out = pd.concat(frames, axis=0, ignore_index=True)
        self._validate_pred_frame(out, "<concat_train_oos_preds>")
        return out

    def _load_or_build_train_preds_fallback(self, fold: DLFold) -> pd.DataFrame:
        cache_path = self._cache_path(fold.train_start, fold.train_end)
        if cache_path.exists():
            frame = pd.read_csv(cache_path)
            self._validate_pred_frame(frame, cache_path)
            return frame.loc[:, _META_COLS + ["label", self.trad_pred_col]].copy()

        frame = self._build_train_preds(fold)
        frame.to_csv(cache_path, index=False)
        return frame.loc[:, _META_COLS + ["label", self.trad_pred_col]].copy()

    def _cache_path(self, train_start: int, train_end: int) -> Path:
        return self.trad_cache_dir / f"trad_train_preds_{train_start}_{train_end}.csv"

    def _build_train_preds(self, fold: DLFold) -> pd.DataFrame:
        """
        用该折训练窗口重训传统代表，并对训练区自身出预测。

        注意：这里是训练目标构造用，因此允许 in-sample；最终 cert / delivery 评分
        依旧使用 `_load_scoring_preds` 读取的 OOS 传统预测。
        """
        engine = self._ensure_engine()
        engine.fit(fold.train_start, fold.train_end)
        frames = engine._build_window_frames(list(fold.train_dates))
        xdf, ydf = frames["xdf"], frames["ydf"]
        pred = np.asarray(engine.predict(xdf), dtype=np.float64).reshape(-1)
        meta = ["date", "symbol", "interval"]
        label_col = [c for c in ydf.columns if c not in meta]
        out = ydf[meta].copy()
        out["label"] = ydf[label_col[0]].to_numpy() if label_col else np.full(len(out), np.nan)
        out[self.trad_pred_col] = pred.astype(np.float32)
        self._validate_pred_frame(out, "<generated_train_preds>")
        return out

    def _ensure_engine(self):
        if self._engine is not None:
            return self._engine
        repo_root = Path(__file__).resolve().parent.parent
        meow_dir = repo_root / "meow"
        src_dir = repo_root / "src"
        import sys
        # 顺序必须与 `dump_trad_preds_folds.py` 保持一致：meow/ 在最前，保证 `meow.py`
        # 内部裸 import 的 `dl/feat/mdl` 优先解析到 meow/ 目录；src/ 只给 `dl_protocol`
        # 等 src 独有模块兜底。若把 src 放前面，会把 `from dl import MeowDataLoader`
        # 误指到 `src/dl.py`，从而缺失 countDate 等提交链专属接口。
        if str(meow_dir) not in sys.path:
            sys.path.insert(0, str(meow_dir))
        if str(src_dir) not in sys.path:
            sys.path.insert(1, str(src_dir))
        # 关键：当前 Python 进程里很可能已经先 import 过 `src/dl.py`，名字就叫 `dl`。
        # 若不把这些同名模块从缓存中清掉，后续 `meow.py` 内部 `from dl import ...`
        # 仍会复用错模块，完全无视我们刚调整好的 sys.path 顺序。
        for mod_name in ("dl", "feat", "mdl"):
            sys.modules.pop(mod_name, None)
        spec = importlib.util.spec_from_file_location(
            "meow_entry_trad_residual", str(meow_dir / "meow.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._engine = module.MeowEngine(h5dir=self.data_dir, cacheDir=None)
        return self._engine

    # ------------------------------------------------------------------
    # 小工具
    # ------------------------------------------------------------------
    @staticmethod
    def _filter_dates(frame: pd.DataFrame, dates: Sequence[int]) -> pd.DataFrame:
        if not dates:
            return frame.iloc[0:0].copy()
        keep = set(int(d) for d in dates)
        out = frame[frame["date"].isin(keep)].copy()
        return out

    def _validate_pred_frame(self, frame: pd.DataFrame, path) -> None:
        need = set(_META_COLS + ["label", self.trad_pred_col])
        miss = [c for c in need if c not in frame.columns]
        if miss:
            raise KeyError(f"传统预测文件缺少必要列 {miss}: {path}")
        dup = frame.duplicated(_META_COLS).sum()
        if dup:
            raise ValueError(f"传统预测文件键重复 {dup} 行：{path}")


def build_label_frame_from_arrays(arrays: SequenceArrays, label_rows: np.ndarray) -> pd.DataFrame:
    """
    用原始 ``SequenceArrays`` + 窗口末行号恢复评测标签帧。

    残差训练时，`SequenceDataset.label_frame()` 会拿到残差标签；而最终评测必须回到
    原始 ``fret12``。这个小工具专门做这个还原，避免为了评测再临时造一个“raw dataset”。
    """
    lr = np.asarray(label_rows, dtype=np.int64)
    return pd.DataFrame({
        "date": arrays.dates[lr],
        "symbol": arrays.symbols[lr],
        "interval": arrays.intervals[lr],
        "fret12": arrays.labels[lr],
    })


def gather_trad_window(trad_row_pred: np.ndarray, label_rows: np.ndarray) -> np.ndarray:
    """
    把逐行传统预测 gather 到窗口末行序。

    ``trad_row_pred`` 与 ``SequenceArrays`` 行序一一对应；DL 评测 / 落盘需要的是
    “每个合法窗口末行”的传统预测，与 ``SequenceDataset.predict`` 输出同序，所以这里只取
    ``label_rows`` 位置即可。
    """
    lr = np.asarray(label_rows, dtype=np.int64)
    return np.asarray(trad_row_pred, dtype=np.float32)[lr]
