"""
序列数据地基 —— Window Indexer + Normalizer + SequenceDataset（脊柱组件，torch-free）

本模块是 DL 主线"固定脊柱"里负责把**原始数值数组**变成**窗口张量 [B, L, C]**
的那一层。它对"通道含义"一无所知（那是 InputAdapter 的职责），只做纯粹的、
与语义无关的时间索引与统计白化：

- ``SequenceArrays``：多日多票的扁平特征 + 标签 + meta（date/symbol/interval）容器，
  行序统一锁在 ``(date, symbol, interval)``。
- ``WindowIndexer``：把扁平行索引成滑动窗口。**绝不跨日、绝不跨票**（窗口只在
  "同一票同一天"的日内 interval 序列里滑），标签因果对齐在窗口末端，warmup
  不足窗的样本丢弃。
- ``Normalizer``：fit-on-train 的 per-channel 统计白化（zscore），可配成 identity；
  写成"不假设只有一张张量"的形态，为将来多张量/多分辨率输入留缝（见规格 §8）。
- ``SequenceDataset``：把以上三者组装成可惰性 gather ``[B, L, C]`` 的数据集，并暴露
  ``label_frame()`` 供协议层把预测对齐回 ``(date, symbol, interval)`` 算 4 指标。

设计要点（与 ``docs/specs/DL实验设计规格.md`` §2.2 / §3 / §8 对齐）：
- PyTorch 在本层完全不存在（脊柱 torch-free）；产出是 numpy ``float32``。
- 防泄漏的物理保证就落在 ``WindowIndexer``：窗口不跨日不跨票 + 标签在窗末
  + Normalizer 只用训练统计量，三条任一破则参考模型会打高分（见单测）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# 与主链一致的 meta / 目标列口径（提交链 src/submission_pipeline.py 同源）。
META_COLS: Tuple[str, ...] = ("date", "symbol", "interval")
TARGET_COL: str = "fret12"


# ================================================================== #
# 扁平数组容器
# ================================================================== #

@dataclass
class SequenceArrays:
    """
    多日多票的扁平特征 + 标签 + meta。

    行序约定（**强约束**）：已按 ``(date, symbol, interval)`` 稳定排序。
    由此带来一个关键性质——同一 ``(date, symbol)`` 的行在数组里是**连续块**
    （date 是最高排序键、symbol 次之），这让 ``WindowIndexer`` 能用纯向量化
    的"段内位置"逻辑切窗口，且天然不跨日不跨票。

    字段：
    - ``features``: ``[N, C]`` float32，C 维布局即通道布局（由 InputAdapter 定义）
    - ``labels``:   ``[N]``    float32，原始 ``fret12`` 量纲（serve 无标签时为全 0）
    - ``dates`` / ``symbols`` / ``intervals``: ``[N]``，逐行 meta（int）
    - ``channels``: 通道名列表，长度 == C
    - ``has_label``: 标签是否真实可用（serve 推理路径为 False）
    """

    features: np.ndarray
    labels: np.ndarray
    dates: np.ndarray
    symbols: np.ndarray
    intervals: np.ndarray
    channels: List[str]
    has_label: bool = True

    def __post_init__(self) -> None:
        n = self.features.shape[0]
        # 形状一致性是后续所有索引的前提，组装那刻就校验，别留到 gather 时炸。
        if not (len(self.labels) == len(self.dates) == len(self.symbols) == len(self.intervals) == n):
            raise ValueError(
                "SequenceArrays 各数组长度不一致: "
                f"features={n}, labels={len(self.labels)}, dates={len(self.dates)}, "
                f"symbols={len(self.symbols)}, intervals={len(self.intervals)}"
            )
        if self.features.ndim != 2:
            raise ValueError(f"features 必须是 2D [N, C]，实际 ndim={self.features.ndim}")
        if self.features.shape[1] != len(self.channels):
            raise ValueError(
                f"features 通道数 {self.features.shape[1]} 与 channels 长度 {len(self.channels)} 不一致"
            )

    @property
    def n_rows(self) -> int:
        return self.features.shape[0]

    @property
    def n_channels(self) -> int:
        return self.features.shape[1]


def build_sequence_arrays(
    raw_df: pd.DataFrame,
    adapter,
    target_col: str = TARGET_COL,
    meta_cols: Sequence[str] = META_COLS,
) -> SequenceArrays:
    """
    用 InputAdapter 把多日 raw 现算成 ``SequenceArrays``。

    职责切分（与规格 §3.1 一致）：
    - **InputAdapter 只吃一天**：本函数逐 ``date`` 分块、对每块单独调 ``adapter.build``，
      绝不把多日 raw 一次性喂进去（避免 EMA/rolling 类 builder 跨日串值）。
    - **行序由本函数统一锁定**：先按 ``(date, symbol, interval)`` 稳定排序，再切块；
      ``adapter.build`` 约定返回与传入块**同序**的 ``[n_rows_day, C]``。meta 与标签都
      从同一排序结果取，三者行序天然对齐，不依赖 adapter 内部排序细节。

    参数：
    - ``adapter``: 实现了 ``channels`` 属性与 ``build(day_raw) -> np.ndarray`` 的 InputAdapter。
    - ``target_col``: 标签列；若 raw 中不存在（serve 推理），labels 置 0 且 ``has_label=False``。
    """
    meta_cols = list(meta_cols)
    raw_sorted = raw_df.sort_values(meta_cols, kind="mergesort").reset_index(drop=True)

    has_label = target_col in raw_sorted.columns
    feats_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    date_parts: List[np.ndarray] = []
    sym_parts: List[np.ndarray] = []
    itv_parts: List[np.ndarray] = []

    for _, day_block in raw_sorted.groupby("date", sort=True):
        day_block = day_block.reset_index(drop=True)
        day_feats = np.asarray(adapter.build(day_block), dtype=np.float32)
        if day_feats.shape[0] != len(day_block):
            raise ValueError(
                "adapter.build 返回行数与输入单日 raw 不一致："
                f"{day_feats.shape[0]} vs {len(day_block)}（行序契约被破坏）"
            )
        feats_parts.append(day_feats)
        if has_label:
            y = (
                pd.to_numeric(day_block[target_col], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .to_numpy(dtype=np.float32)
            )
        else:
            y = np.zeros(len(day_block), dtype=np.float32)
        y_parts.append(y)
        date_parts.append(day_block["date"].to_numpy())
        sym_parts.append(day_block["symbol"].to_numpy())
        itv_parts.append(day_block["interval"].to_numpy())

    if not feats_parts:
        # 空输入：返回零长数组但保留通道布局，下游 len()==0 自然短路。
        c = len(adapter.channels)
        empty_i = np.array([], dtype=np.int64)
        return SequenceArrays(
            features=np.zeros((0, c), dtype=np.float32),
            labels=np.array([], dtype=np.float32),
            dates=empty_i, symbols=empty_i, intervals=empty_i,
            channels=list(adapter.channels), has_label=has_label,
        )

    return SequenceArrays(
        features=np.concatenate(feats_parts, axis=0),
        labels=np.concatenate(y_parts, axis=0),
        dates=np.concatenate(date_parts, axis=0),
        symbols=np.concatenate(sym_parts, axis=0),
        intervals=np.concatenate(itv_parts, axis=0),
        channels=list(adapter.channels),
        has_label=has_label,
    )


def build_sequence_arrays_from_frames(
    xdf: pd.DataFrame,
    ydf: Optional[pd.DataFrame],
    channels: Sequence[str],
    target_col: str = TARGET_COL,
    meta_cols: Sequence[str] = META_COLS,
) -> SequenceArrays:
    """
    用“已加载好的特征表 + 标签表”直接组装 ``SequenceArrays``。

    这个入口是给**实验链的磁盘特征缓存**准备的：
    - 传统评测 / 新 DL 实验都可以先把 433 特征落到 ``data/features/``；
    - 之后按日期从 ``FeatureLoader`` 直接读 ``xdf/ydf``，这里负责把它们收口成
      DL 脊柱统一消费的 ``SequenceArrays``；
    - 这样做能避免每个 fold 再从 raw 重算 433 特征，同时又不碰正式提交链
      ``meow.py`` 的“raw 现算”契约。

    约束与校验：
    - ``xdf`` / ``ydf`` 都必须遵守主链统一行序：``(date, symbol, interval)`` 稳定排序。
    - ``xdf`` 里必须完整包含 ``channels`` 指定的列，且按传入顺序锁定 C 维布局。
    - ``ydf`` 可为 ``None``；这代表 serve/无标签场景，函数会按既有口径补全 0 标签并把
      ``has_label`` 记为 ``False``。
    """
    meta_cols = list(meta_cols)
    missing_meta = [col for col in meta_cols if col not in xdf.columns]
    if missing_meta:
        raise KeyError(f"xdf 缺少必要 meta 列: {missing_meta}")

    missing_channels = [col for col in channels if col not in xdf.columns]
    if missing_channels:
        raise KeyError(
            "xdf 缺少指定通道列，无法组装 SequenceArrays；"
            f"例如: {missing_channels[:5]}"
        )

    x_sorted = (
        xdf.loc[:, meta_cols + list(channels)]
        .sort_values(meta_cols, kind="mergesort")
        .reset_index(drop=True)
        .copy()
    )
    features = x_sorted.loc[:, list(channels)].to_numpy(dtype=np.float32, copy=True)
    dates = x_sorted["date"].to_numpy()
    symbols = x_sorted["symbol"].to_numpy()
    intervals = x_sorted["interval"].to_numpy()

    if ydf is None:
        labels = np.zeros(len(x_sorted), dtype=np.float32)
        has_label = False
    else:
        missing_y_meta = [col for col in meta_cols if col not in ydf.columns]
        if missing_y_meta:
            raise KeyError(f"ydf 缺少必要 meta 列: {missing_y_meta}")
        if target_col not in ydf.columns:
            raise KeyError(f"ydf 缺少目标列: {target_col}")

        y_sorted = (
            ydf.loc[:, meta_cols + [target_col]]
            .sort_values(meta_cols, kind="mergesort")
            .reset_index(drop=True)
            .copy()
        )
        if len(y_sorted) != len(x_sorted):
            raise ValueError(
                "xdf / ydf 行数不一致，无法安全对齐；"
                f"xdf={len(x_sorted)} ydf={len(y_sorted)}"
            )
        if not x_sorted.loc[:, meta_cols].equals(y_sorted.loc[:, meta_cols]):
            raise ValueError("xdf / ydf 的 (date,symbol,interval) 行序不一致，疑似特征缓存错位")
        labels = (
            pd.to_numeric(y_sorted[target_col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
        has_label = True

    return SequenceArrays(
        features=features,
        labels=labels,
        dates=dates,
        symbols=symbols,
        intervals=intervals,
        channels=list(channels),
        has_label=has_label,
    )


def subset_by_dates(arrays: SequenceArrays, dates: Sequence[int]) -> SequenceArrays:
    """
    按交易日子集切 ``SequenceArrays``（保持原行序）。

    用于 trainer 把"训练区" arrays 拆成 train_core / earlystop 两份而**不重算特征**：
    原 arrays 已按 ``(date, symbol, interval)`` 排序，按 date 掩码后行序不变、
    ``(date, symbol)`` 段仍连续，``WindowIndexer`` 依旧正确。
    """
    keep = set(int(d) for d in dates)
    mask = np.isin(arrays.dates, list(keep))
    return SequenceArrays(
        features=arrays.features[mask],
        labels=arrays.labels[mask],
        dates=arrays.dates[mask],
        symbols=arrays.symbols[mask],
        intervals=arrays.intervals[mask],
        channels=list(arrays.channels),
        has_label=arrays.has_label,
    )


# ================================================================== #
# 窗口索引器（不跨日 / 不跨票 / 因果对齐 / warmup 丢弃）
# ================================================================== #

class WindowIndexer:
    """
    把 ``SequenceArrays`` 的扁平行索引成滑动窗口。

    一个窗口 = 某 ``(date, symbol)`` 段内连续的 ``L`` 行；标签取窗口**末行**
    （因果对齐：用 ``[t-L+1, t]`` 的特征预测 ``t`` 时刻标签，不偷看未来）。

    返回 ``label_rows``：所有合法窗口的**末行全局行号**数组。窗口本身就是
    ``[label_row - L + 1, label_row]`` 这段连续行，因此只需记末行即可重建。

    防泄漏的物理保证：
    - **不跨日 / 不跨票**：只在同一 ``(date, symbol)`` 连续段内滑，段首前 ``L-1``
      行（warmup 不足窗）直接丢弃。
    - 由于 ``SequenceArrays`` 行序锁 ``(date, symbol, interval)``，同段行必连续，
      段内位置 ``>= L-1`` 即合法末行——纯向量化、零 Python 循环。
    """

    def __init__(self, seq_len: int):
        if seq_len < 1:
            raise ValueError(f"seq_len 必须 >= 1，实际 {seq_len}")
        self.seq_len = int(seq_len)

    def build_index(self, arrays: SequenceArrays) -> np.ndarray:
        """返回合法窗口末行的全局行号数组 ``label_rows``（int64, 升序）。"""
        n = arrays.n_rows
        L = self.seq_len
        if n == 0:
            return np.array([], dtype=np.int64)

        dates = arrays.dates
        syms = arrays.symbols
        idx = np.arange(n, dtype=np.int64)

        # 段边界：date 或 symbol 相对上一行发生变化，即新段起点。
        is_start = np.empty(n, dtype=bool)
        is_start[0] = True
        is_start[1:] = (dates[1:] != dates[:-1]) | (syms[1:] != syms[:-1])

        seg_id = np.cumsum(is_start) - 1          # 每行所属段号（从 0 起）
        seg_first = idx[is_start]                 # 各段首行的全局行号
        pos_in_seg = idx - seg_first[seg_id]      # 每行在其段内的位置（从 0 起）

        # 段内位置 >= L-1 的行才凑得齐一整窗（warmup 自动丢弃）。
        label_rows = idx[pos_in_seg >= (L - 1)]
        return label_rows


# ================================================================== #
# 归一化器（fit-on-train，可 identity，留多张量缝）
# ================================================================== #

class Normalizer:
    """
    per-channel 统计白化，**只用训练区统计量** fit、套到 train/val 全部。

    防泄漏关键：``fit`` 必须只喂训练区特征，``transform`` 再套到验证区——
    协议层负责保证这一点（用 train 的 ``SequenceArrays.features`` fit 一次，
    同一个 normalizer 实例同时交给 train_ds 与 val_ds）。

    模式：
    - ``"zscore"``：``(x - mean) / std``，逐通道；``std < eps`` 的常量通道不缩放。
    - ``"identity"``：恒等，用于 InputAdapter 已在语义层做完归一化（如 DeepLOB
      的价格相对归一）的场景（规格 §2.3）。

    多张量留缝（规格 §8）：当前锁"单张 ``[N, C]``"。内部把单张统计逻辑收在
    ``_fit_single`` / ``_transform_single``，将来 InputAdapter 放宽成"张量字典"
    时，只需在外层对每个 key 各调一次，协议/指标/HPO 不动。
    """

    def __init__(self, mode: str = "zscore", eps: float = 1e-8):
        if mode not in ("zscore", "identity"):
            raise ValueError(f"未知 Normalizer mode: {mode}，应为 'zscore' / 'identity'")
        self.mode = mode
        self.eps = float(eps)
        self._mean: Optional[np.ndarray] = None   # [C]
        self._std: Optional[np.ndarray] = None    # [C]
        self._fitted = False

    # 分块步长：fit/transform 都按这个行数切块处理，把任一中间副本的峰值钉在
    # ``CHUNK_ROWS × C`` 量级（百 MB），与训练集总行数（百万级）解耦。
    CHUNK_ROWS = 200_000

    # ---- 单张张量的核心实现（多张量时对每个 key 复用） ---- #
    def _fit_chunks_single(self, feature_chunks) -> None:
        """
        **分块累计** per-channel mean/std，绝不物化整张 nan_to_num 副本。

        旧实现 ``nan_to_num(整张[N,C])`` 会再造一份 ~14GB（119 天全票折）副本，直接把
        内存打穿。这里改为逐块（``CHUNK_ROWS`` 行）nan_to_num + float64 累加 count/sum/
        sumsq，最后合成 mean/std——统计值与整块计算逐位等价，但中间副本只有百 MB 级。

        ``feature_chunks``：一组 ``[n_i, C]`` 数组（如 [core.features, es.features]），
        被当作"训练区拼起来"一并统计，但**不真的 concatenate**，省掉一份全量副本。
        """
        C = None
        count = 0
        ssum = None
        sqsum = None
        for arr in feature_chunks:
            if arr is None:
                continue
            a = np.asarray(arr)
            if a.shape[0] == 0:
                continue
            if C is None:
                C = a.shape[1]
                ssum = np.zeros(C, dtype=np.float64)
                sqsum = np.zeros(C, dtype=np.float64)
            for s in range(0, a.shape[0], self.CHUNK_ROWS):
                blk = np.nan_to_num(
                    a[s:s + self.CHUNK_ROWS].astype(np.float32, copy=False),
                    nan=0.0, posinf=0.0, neginf=0.0,
                )
                count += blk.shape[0]
                ssum += blk.sum(axis=0, dtype=np.float64)
                sqsum += np.square(blk, dtype=np.float64).sum(axis=0)
        if C is None or count == 0:
            # 无任何训练数据：留空 mean/std，transform 时退化为只 nan_to_num、不缩放。
            self._mean = None
            self._std = None
            return
        mean = ssum / count
        var = np.maximum(sqsum / count - mean ** 2, 0.0)
        std = np.sqrt(var)
        # 常量/近常量通道：std 置 1，等价于"只去均值不缩放"，避免除零爆炸。
        std = np.where(std < self.eps, 1.0, std)
        self._mean = mean
        self._std = std

    def _transform_single(self, features: np.ndarray, inplace: bool = False) -> np.ndarray:
        """
        逐块原地白化。

        ``inplace=True``（训练热路径用）：直接在传入的 ``features`` 缓冲上改写，
        **不再产生第二份 [N,C] 副本**——这是把 119 天折峰值从"好几份 14GB"压到"一份"
        的关键。调用方需保证这份 ``features`` 之后不再以原始值复用（脊柱已满足）。
        """
        x = np.asarray(features, dtype=np.float32)
        if not inplace:
            x = x.copy()
        do_scale = self.mode == "zscore" and self._mean is not None and self._mean.shape[0] == x.shape[1]
        mean_f32 = self._mean.astype(np.float32) if do_scale else None
        std_f32 = self._std.astype(np.float32) if do_scale else None
        # 逐块：块内 nan_to_num(copy=False 原地) + 仿射原地，全程不产生整张临时。
        for s in range(0, x.shape[0], self.CHUNK_ROWS):
            blk = x[s:s + self.CHUNK_ROWS]
            np.nan_to_num(blk, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            if do_scale:
                blk -= mean_f32
                blk /= std_f32
        return x

    # ---- 对外接口 ---- #
    def fit(self, features: np.ndarray) -> "Normalizer":
        if self.mode == "zscore":
            self._fit_chunks_single([features])
        self._fitted = True
        return self

    def fit_chunked(self, feature_chunks) -> "Normalizer":
        """对一组特征块（不 concatenate）统一 fit，省掉一份全量训练副本。"""
        if self.mode == "zscore":
            self._fit_chunks_single(list(feature_chunks))
        self._fitted = True
        return self

    def transform(self, features: np.ndarray, inplace: bool = False) -> np.ndarray:
        if self.mode == "zscore" and not self._fitted:
            raise RuntimeError("Normalizer(zscore) 未 fit 就 transform —— 可能泄漏或用错顺序")
        return self._transform_single(features, inplace=inplace)

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        return self.fit(features).transform(features)


# ================================================================== #
# 序列数据集（惰性 gather [B, L, C]）
# ================================================================== #

class SequenceDataset:
    """
    把 ``SequenceArrays`` + ``WindowIndexer`` + ``Normalizer`` 组装成可迭代数据集。

    - 归一化在构建时对整张 ``[N, C]`` 特征做**一次**（per-channel 仿射对整块或对
      窗口切片结果完全等价），gather 时只做切片——既无泄漏又省算力。
    - ``__getitem__`` / ``iter_batches`` 惰性产出 ``[L, C]`` / ``[B, L, C]``。
    - ``label_frame()`` 暴露每个窗口末行的 ``(date, symbol, interval, fret12)``，
      供协议层把 ``predict`` 输出对齐回截面、按 date 算 daily-IC。
    """

    def __init__(
        self,
        arrays: SequenceArrays,
        seq_len: int,
        normalizer: Optional[Normalizer] = None,
        own_features: bool = False,
    ):
        self.arrays = arrays
        self.seq_len = int(seq_len)
        self._indexer = WindowIndexer(seq_len)
        self.label_rows = self._indexer.build_index(arrays)
        # normalizer 默认 identity（未 fit 也能用），实战由协议层注入已 fit 的 zscore。
        self.normalizer = normalizer if normalizer is not None else Normalizer(mode="identity")
        # own_features=True：本 dataset 独占这份 arrays，可**原地**白化、不再另造一份 [N,C]
        # 副本（训练热路径用，把 119 天全票折的内存峰从"好几份 14GB"压到"一份"）。
        # 默认 False：拷贝白化，保持旧语义（合成测试可能复用同一 arrays、不容就地改）。
        self._feats = self.normalizer.transform(arrays.features, inplace=own_features)  # [N, C]，已白化
        # 窗口内相对偏移：[-(L-1), ..., 0]，加在末行号上即整窗行号。
        self._offsets = np.arange(-(self.seq_len - 1), 1, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.label_rows.shape[0])

    @property
    def n_channels(self) -> int:
        return self.arrays.n_channels

    @property
    def channels(self) -> List[str]:
        return list(self.arrays.channels)

    def __getitem__(self, k: int) -> Tuple[np.ndarray, float]:
        """惰性取第 k 个窗口：返回 ``(X[L, C] float32, y float)``。"""
        lr = int(self.label_rows[k])
        X = self._feats[lr - self.seq_len + 1: lr + 1]
        y = float(self.arrays.labels[lr])
        return X, y

    def _gather_rows(self, label_rows: np.ndarray) -> np.ndarray:
        """由窗口末行号批量构造整窗行号矩阵 ``[B, L]``。"""
        return label_rows[:, None] + self._offsets[None, :]

    # ---- GPU-gather 快路径所需的「原料」访问器（脊柱仍 torch-free） ---- #
    # 背景：小模型（GRU hidden 32~128）在 GPU 上算一拍就完，若每个 batch 还在 CPU 上
    # 用 numpy fancy-index 现拼 [B, L, C] 再 H2D 拷上卡，GPU 会被 CPU 拼数据饿死、利用率
    # 长期个位数。解法是把**整张归一化特征矩阵 + 窗口索引**一次性交给卡带，由卡带搬上
    # GPU、在 GPU 上做窗口 gather。这里只负责暴露 numpy 原料，绝不引入 torch——torch 仍
    # 封死在卡带内（规格 §2.3「脊柱 torch-free」）。
    def feature_matrix(self) -> np.ndarray:
        """已白化的整张特征矩阵 ``[N, C]`` float32（与 gather 出来的窗口同源）。"""
        return self._feats

    def window_labels(self) -> np.ndarray:
        """每个合法窗口末行的标签 ``[num_windows]`` float32，与 ``label_rows`` 同序。"""
        return self.arrays.labels[self.label_rows].astype(np.float32, copy=False)

    def window_offsets(self) -> np.ndarray:
        """窗内相对偏移 ``[-(L-1), ..., 0]``，加在 ``label_rows`` 上即整窗行号。"""
        return self._offsets

    def gather_all(self) -> Tuple[np.ndarray, np.ndarray]:
        """一次性取全部窗口：``(X[B, L, C] float32, y[B] float32)``。小数据/参考模型用。"""
        if len(self) == 0:
            return (
                np.zeros((0, self.seq_len, self.n_channels), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )
        gr = self._gather_rows(self.label_rows)          # [B, L]
        X = self._feats[gr]                              # [B, L, C]
        y = self.arrays.labels[self.label_rows]          # [B]
        return X.astype(np.float32), y.astype(np.float32)

    def iter_batches(
        self,
        batch_size: int,
        shuffle: bool = False,
        seed: Optional[int] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        惰性按批产出 ``(X[b, L, C], y[b])``。

        训练（shuffle=True）按种子打乱窗口顺序；评测（shuffle=False）保持窗口
        自然顺序，使预测与 ``label_frame()`` 行序一致、便于对齐算指标。
        """
        n = len(self)
        order = np.arange(n)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(order)
        for start in range(0, n, batch_size):
            sel = order[start: start + batch_size]
            rows = self.label_rows[sel]
            gr = self._gather_rows(rows)                 # [b, L]
            X = self._feats[gr].astype(np.float32)
            y = self.arrays.labels[rows].astype(np.float32)
            yield X, y

    def label_frame(self) -> pd.DataFrame:
        """
        每个窗口末行的 meta + 真实标签，行序与窗口顺序（=gather_all/iter_batches
        非 shuffle 顺序）严格一致。协议层据此把 predict 接上 ``forecast`` 列算指标。
        """
        lr = self.label_rows
        return pd.DataFrame({
            "date": self.arrays.dates[lr],
            "symbol": self.arrays.symbols[lr],
            "interval": self.arrays.intervals[lr],
            TARGET_COL: self.arrays.labels[lr],
        })


# ================================================================== #
# 截面索引器 + 截面数据集（按 (date,interval) 聚票，规格 §8.2）
# ================================================================== #

class CrossSectionIndexer:
    """
    把扁平行的**合法窗口末行**按 ``(date, interval)`` 聚成"截面快照"。

    一个快照 = 同一 ``(date, interval)`` 时刻、所有"凑得齐一整窗"的在场票，每票带它
    自己 ``[t-L+1, t]`` 那段日内窗（窗口口径完全复用 ``WindowIndexer``，因此**不跨日、
    不跨票、因果对齐、warmup 丢弃**等物理保证一字不改，只是把窗口换一种分组方式聚起来）。

    返回 ``snapshots``：一个列表，每个元素是某 ``(date, interval)`` 快照里各在场票的
    **窗口末行全局行号**数组（按 ``symbol`` 升序），快照之间按 ``(date, interval)`` 升序。
    变长 N（每个截面在场票数不同）由下游 ``CrossSectionDataset`` 在成批时 pad + mask。

    与逐票 ``WindowIndexer`` 的关系：两者末行集合**完全相同**（都来自
    ``WindowIndexer.build_index``），只是 CrossSection 把它们按截面重新分组。因此截面
    卡带产出的逐票预测，可无损映射回逐票 ``label_rows`` 顺序（升序）做指标对齐。
    """

    def __init__(self, seq_len: int):
        self.seq_len = int(seq_len)
        self._win = WindowIndexer(seq_len)

    def build_index(self, arrays: SequenceArrays) -> List[np.ndarray]:
        label_rows = self._win.build_index(arrays)      # 合法窗口末行（按 (date,sym,itv) 升序）
        if label_rows.size == 0:
            return []
        d = arrays.dates[label_rows]
        itv = arrays.intervals[label_rows]
        sym = arrays.symbols[label_rows]
        # label_rows 原序是 (date, symbol, interval)：同一 (date,interval) 的不同票**不连续**
        # （symbol 夹在中间），必须按 (date, interval, symbol) 重排把同截面的票聚拢。
        order = np.lexsort((sym, itv, d))               # 主键 date，其次 interval，再 symbol
        lr = label_rows[order]
        dd = d[order]
        ii = itv[order]
        # (date,interval) 变化处即新快照起点。
        change = np.empty(lr.shape[0], dtype=bool)
        change[0] = True
        change[1:] = (dd[1:] != dd[:-1]) | (ii[1:] != ii[:-1])
        starts = np.flatnonzero(change)
        # 按起点切片成各快照（np.split 用内部边界 starts[1:]）。
        return np.split(lr, starts[1:])


class CrossSectionDataset:
    """
    截面快照数据集：把 ``SequenceArrays`` 按 ``(date, interval)`` 聚成快照流。

    与 ``SequenceDataset`` 共享同一套"白化特征矩阵 + 窗口偏移"的惰性 gather 机制，
    只是**样本单元从"一个窗 [L,C]"换成"一个截面 [N,L,C] + mask[N] + y[N]"**（规格
    §8.2 张量契约泛化）。瘦内存纪律保持：特征矩阵只白化一次（可原地）、快照按需
    gather、不一次性物化整期所有快照。

    两种构造入口：
    - ``__init__(arrays, seq_len, normalizer, own_features)``：标准入口（与
      ``SequenceDataset`` 同签名），自带白化，单测/独立使用走它。
    - ``from_whitened(feats, arrays, seq_len)``：复用**已白化**的特征矩阵（如 trainer
      已为 ``SequenceDataset`` 原地白化过的那张），**不再重算白化、不再复制 [N,C]**——
      截面卡带内部包装 trainer 传来的 ``SequenceDataset`` 时走它，避免双份内存。
    """

    def __init__(
        self,
        arrays: SequenceArrays,
        seq_len: int,
        normalizer: Optional[Normalizer] = None,
        own_features: bool = False,
    ):
        self.arrays = arrays
        self.seq_len = int(seq_len)
        self.normalizer = normalizer if normalizer is not None else Normalizer(mode="identity")
        self._feats = self.normalizer.transform(arrays.features, inplace=own_features)  # [N,C] 已白化
        self._offsets = np.arange(-(self.seq_len - 1), 1, dtype=np.int64)
        self.snapshots: List[np.ndarray] = CrossSectionIndexer(seq_len).build_index(arrays)

    @classmethod
    def from_whitened(cls, feats: np.ndarray, arrays: SequenceArrays, seq_len: int) -> "CrossSectionDataset":
        """用已白化矩阵 ``feats`` + 原始 ``arrays``（取 meta/labels）直接组装，零再白化、零复制。"""
        obj = cls.__new__(cls)
        obj.arrays = arrays
        obj.seq_len = int(seq_len)
        obj.normalizer = Normalizer(mode="identity")
        obj._feats = feats
        obj._offsets = np.arange(-(int(seq_len) - 1), 1, dtype=np.int64)
        obj.snapshots = CrossSectionIndexer(seq_len).build_index(arrays)
        return obj

    def __len__(self) -> int:
        """快照数（= 有多少个 (date,interval) 截面）。"""
        return len(self.snapshots)

    @property
    def n_channels(self) -> int:
        return self.arrays.n_channels

    @property
    def channels(self) -> List[str]:
        return list(self.arrays.channels)

    def n_tickers_total(self) -> int:
        """所有快照在场票总数（= 逐票预测/label_frame 行数）。"""
        return int(sum(len(s) for s in self.snapshots))

    def snapshot_sizes(self) -> np.ndarray:
        """各快照在场票数 ``[num_snapshots]``。"""
        return np.array([len(s) for s in self.snapshots], dtype=np.int64)

    def feature_matrix(self) -> np.ndarray:
        """已白化的整张特征矩阵 ``[N, C]`` float32（与窗口 gather 同源）。"""
        return self._feats

    def window_offsets(self) -> np.ndarray:
        return self._offsets

    def flat_label_rows(self) -> np.ndarray:
        """按快照顺序拼接的全部窗口末行号（snapshot-flatten 序）；空则返回空数组。"""
        if not self.snapshots:
            return np.array([], dtype=np.int64)
        return np.concatenate(self.snapshots).astype(np.int64, copy=False)

    def gather_snapshot(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """取第 k 个快照：``(X[N, L, C] float32, y[N] float32)``，票按 symbol 升序。"""
        lr = self.snapshots[k]
        gr = lr[:, None] + self._offsets[None, :]        # [N, L]
        X = self._feats[gr].astype(np.float32)           # [N, L, C]
        y = self.arrays.labels[lr].astype(np.float32)    # [N]
        return X, y

    def iter_batches(
        self,
        batch_size: int,
        shuffle: bool = False,
        seed: Optional[int] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        按"快照数"成批（``batch_size`` = 一批多少个截面，**不是多少票**），pad 到批内
        最大在场票数 ``maxN``，惰性产 ``(X[B,maxN,L,C], y[B,maxN], mask[B,maxN])``（纯 numpy）。

        - pad 位特征/标签置 0，``mask=False``；下游损失/注意力只在 mask 有效位上算。
        - ``shuffle=True`` 按种子打乱**快照顺序**（训练）；``shuffle=False`` 保持
          ``(date,interval)`` 升序（评测，便于把逐票预测拼回 ``flat_label_rows`` 序）。
        """
        n = len(self)
        if n == 0:
            return
        order = np.arange(n)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(order)
        L, C = self.seq_len, self.n_channels
        for start in range(0, n, batch_size):
            sel = order[start: start + batch_size]
            snaps = [self.snapshots[i] for i in sel]
            sizes = [len(s) for s in snaps]
            maxN = max(sizes) if sizes else 0
            B = len(snaps)
            X = np.zeros((B, maxN, L, C), dtype=np.float32)
            y = np.zeros((B, maxN), dtype=np.float32)
            mask = np.zeros((B, maxN), dtype=bool)
            for bi, lr in enumerate(snaps):
                nk = len(lr)
                gr = lr[:, None] + self._offsets[None, :]    # [nk, L]
                X[bi, :nk] = self._feats[gr]
                y[bi, :nk] = self.arrays.labels[lr]
                mask[bi, :nk] = True
            yield X, y, mask

    def label_frame(self) -> pd.DataFrame:
        """
        按快照顺序（``(date,interval)`` 升序、票内 symbol 升序）展平的逐票 meta + 标签。

        注意：行序是 **snapshot-flatten 序**，与 ``SequenceDataset.label_frame()`` 的
        ``(date,symbol,interval)`` 窗口序**不同**。截面卡带对外 predict 时会把预测按
        ``flat_label_rows`` 升序映射回逐票窗口序，故指标对齐仍用 SequenceDataset 那份。
        本方法主要供独立使用 / 单测核对聚票正确性。
        """
        lr = self.flat_label_rows()
        return pd.DataFrame({
            "date": self.arrays.dates[lr],
            "symbol": self.arrays.symbols[lr],
            "interval": self.arrays.intervals[lr],
            TARGET_COL: self.arrays.labels[lr],
        })
