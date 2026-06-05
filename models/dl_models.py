"""
卡带层 —— InputAdapter 接口 + 适配器实现 + ModelCartridge 接口 + numpy 参考模型

这是规格 §3 两个接口契约的落地，也是"可换卡带"那一列：

- ``InputAdapter``（输入适配卡带）：唯一理解原始数据语义的地方，吐出通道布局固定的
  ``[n_rows_day, C]`` 干净数组；下游全是不关心含义的张量数学。
    - ``IdentityAdapter``：直接把指定 raw 数值列当通道（调试 / 防泄漏合成测试）。
    - ``FeatureAdapter``：包装现有 433 特征管线（``SubmissionFeaturePipeline``），零新特征公式（D1/LSTM）。
    - ``RawChannelAdapter``：~59 原始微结构通道做**最小语义归一**（价相对 mid / midpx 日内对数收益 /
      量 log1p），让网络自己学交互——刻意不手搓 imbalance/OFI（D2/TCN，规格 §3.1/§8.0）。
- ``ModelCartridge``（模型卡带，**唯一允许出现 torch 的地方**——但本文件只放 torch-free
  的 numpy 参考模型；真正的 LSTM/TCN 卡带等 4060 + PyTorch 就绪后再加，脊柱不动）。
    - ``ReferenceZeroCartridge``：恒 0，corr 基线 sanity。
    - ``ReferenceLastCartridge``：末步通道线性回归——**防泄漏探测器**：干净因果管线下
      只能拿到窗末特征、对"依赖未来的标签"打低分；一旦窗口/对齐/归一化漏了未来信息，
      它立刻能吃到并涨分（规格 §5.5）。
    - ``ReferencePoolCartridge``：窗口均值池化 + numpy 线性；**声明 ``STRUCTURE_SEARCH_SPACE``**，
      作 HPO Searcher 的 torch-free 被测对象 + 未来 TCN 卡带的 search_space 模板。

全部 torch-free、Mac CPU 可跑。
"""

from __future__ import annotations

import queue
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, List, Optional

import numpy as np

from adapter_config import AdapterKind
from model_config import ModelKind
from registry import register_adapter, register_model
# 损失/重标定来自脊柱侧 src/dl_losses.py（torch-free import；torch 仅在被调用时惰性导）。
from dl_losses import fit_linear_rescale_numpy, make_loss


# ================================================================== #
# InputAdapter 接口 + 实现
# ================================================================== #

class InputAdapter(ABC):
    """
    输入适配卡带接口（规格 §3.1）。

    - ``channels``: 通道名列表，顺序即 C 维布局（固定）。
    - ``build(raw_day_df)``: 吃**一天** raw，返回 ``[n_rows_day, C]`` float32；
      行序约定 = 输入按 ``(date, symbol, interval)`` 稳定排序后的序（与
      ``sequence_dataset.build_sequence_arrays`` 的取 meta 口径一致）。
    """

    channels: List[str]

    @classmethod
    @abstractmethod
    def from_config(cls, adapter_config) -> "InputAdapter":
        ...

    @abstractmethod
    def build(self, raw_day_df) -> np.ndarray:
        ...


@register_adapter(AdapterKind.IDENTITY)
class IdentityAdapter(InputAdapter):
    """把指定 raw 数值列**原样**当通道。用于调试与防泄漏合成测试（无任何语义加工）。"""

    def __init__(self, columns):
        if not columns:
            raise ValueError("IdentityAdapter 需要非空 columns（要把哪些 raw 列当通道）")
        self.channels = list(columns)

    @classmethod
    def from_config(cls, adapter_config) -> "IdentityAdapter":
        return cls(columns=adapter_config.columns)

    def build(self, raw_day_df) -> np.ndarray:
        # 按 (symbol, interval) 稳定排序（单日内 date 恒定），对齐统一行序契约。
        day = raw_day_df.sort_values(["symbol", "interval"], kind="mergesort")
        return day[self.channels].to_numpy(dtype=np.float32)


@register_adapter(AdapterKind.FEATURE_433)
class FeatureAdapter(InputAdapter):
    """
    包装现有 433 特征管线：把正式提交特征列当通道（规格 §3.1「首个实现」）。

    设计成**双通路**：
    1. **实验 / DL 训练默认优先走 ``data/features/`` 磁盘缓存**：
       通过 ``FeatureLoader`` 直接按日期取已经构建好的 433 特征，避免 rolling /
       sweep 每折都把同一批特征从 raw 重算一遍。
    2. **提交 / 缺缓存兜底仍可 raw 现算**：
       复用 ``SubmissionFeaturePipeline.build_feature_frames``，保证一旦本机还没建缓存、
       或单测只给了原始单日 DataFrame，仍能沿既有口径跑通。

    这样做的边界很明确：
    - **实验链**：追求速度，优先缓存；
    - **提交链**：追求独立可交付，继续 raw 现算；
    - 两条路共享同一份 433 列定义（``self.channels``），避免列集合/顺序漂移。
    """

    def __init__(self, groups=None):
        # 延迟 import：FeatureAdapter 用到时才拉特征管线（src 在 path）。
        from feature_store import DEFAULT_FEATURE_DIR
        from submission_pipeline import SubmissionFeaturePipeline, DEFAULT_SUBMISSION_GROUPS
        groups = tuple(groups) if groups else DEFAULT_SUBMISSION_GROUPS
        self.groups = tuple(groups)
        self._pipeline = SubmissionFeaturePipeline(groups=groups)
        self._feature_dir = DEFAULT_FEATURE_DIR
        self._h5dir = "data"
        self._feature_loader = None
        self.channels = list(self._pipeline.feature_names())
        # —— 按「交易日」**有界 LRU** 缓存逐日 numpy 结果（性能命脉，见 load_sequence_arrays 注释） —— #
        # rolling/sweep 里同一组日期会被几十次 fit 反复读：档1 4 网格×2 折×2 seed 把同样
        # 2 折读 ~16 遍，档2 5 折×3 seed 再读。而单日 433 特征只取决于「日期 + 本适配器固定
        # 列集」，与 hparams/seed 无关。故按日缓存：命中即跳过整套 h5 读 + pickle select。
        #
        # **但缓存必须有界**：单日全票 433 特征 ~120MB，若无限缓存 144 天 = ~17GB，叠上
        # 每折 ~16GB 的训练矩阵会把 34GB 内存打穿（119 天全票折硬化前实测 24GB+/折）。
        #
        # 策略 = **蓄水式保留"最早装入的共享基段"**（不是 LRU）：所有 expanding 折都从
        # Jun1 起、共享同一段早期日期（Jun–Jul base），这段被每折反复读；而各折只在末端
        # 不同。故缓存装满 ``_day_cache_cap`` 天后**不再淘汰、也不再新增**——自然把最早装入
        # 的共享基段钉住、跨 fit 命中率最高；末端差异日每折从磁盘 pickle 重读（~0.3s/天，
        # 便宜）。默认 32 天 ≈ 3.8GB，配合预分配流式装填，把 119 天折峰值压到 ~22GB。
        self._day_cache: "OrderedDict[int, tuple]" = OrderedDict()
        self._day_cache_cap = 32
        # 逐日行数台账（极小、不设上限）：load_sequence_arrays 据此**预分配输出、逐日流式
        # 填充**，避免 np.concatenate 把"所有日块 + 输出"两份 14GB 同时压在内存里。
        self._day_nrows: Dict[int, int] = {}

    @classmethod
    def from_config(cls, adapter_config) -> "FeatureAdapter":
        return cls(groups=adapter_config.groups or None)

    def bind_data_sources(self, h5dir: str, feature_dir: str) -> None:
        """
        由 Orchestrator 在 run 发起时补齐数据源路径。

        适配器本身只知道“我要 433 哪些列”，不知道本轮实验的数据根目录在哪里；
        路径应由顶层编排器注入，而不是在适配器里写死仓库相对路径。
        """
        self._h5dir = str(h5dir)
        self._feature_dir = str(feature_dir)
        # 路径一旦变化，旧 loader 可能指向了别处缓存，必须强制失效重建。
        self._feature_loader = None
        # 数据源换了，按日缓存/行数台账里的旧数据也作废，清空避免读到别处的特征。
        self._day_cache = OrderedDict()
        self._day_nrows = {}

    def _feature_store_ready(self) -> bool:
        """
        判断本机的磁盘特征缓存是否具备可读条件。

        这里只做**保守探测**：
        - manifest 存在，说明 FeatureStore 至少已经初始化过；
        - 之后真正读取某天某 stage 时若缺文件，由 ``FeatureLoader`` 再抛精确错误。
        """
        manifest = Path(self._feature_dir) / "manifest.json"
        return manifest.exists()

    def _get_feature_loader(self):
        """
        懒加载 ``FeatureLoader``。

        原因：
        - 大多数 torch-free 合成测试压根用不到它；
        - 只有实验链真的选择“按日期读缓存”时才需要；
        - 延迟构造也让 `bind_data_sources()` 可以先覆盖 h5/feature 根目录。
        """
        if self._feature_loader is None:
            from feature_loader import FeatureLoader
            self._feature_loader = FeatureLoader(
                h5dir=self._h5dir,
                feature_dir=self._feature_dir,
            )
        return self._feature_loader

    def load_sequence_arrays(self, dates, target_col="fret12", meta_cols=("date", "symbol", "interval")):
        """
        按日期直接从磁盘特征缓存组装 ``SequenceArrays``。

        这是 DL 训练真正想走的快路径：
        - ``SequenceTrainer`` 传入本折需要的日期；
        - 适配器用 ``FeatureLoader`` 一次把这些日期的 433 特征读出来；
        - 再交给 ``build_sequence_arrays_from_frames`` 收口成脊柱统一格式。

        若本机尚未准备好 ``data/features/``，这里明确抛 ``FileNotFoundError``，
        让上层决定是回退 raw 现算、还是先构建缓存；绝不在底层静默吞掉。
        """
        if not self._feature_store_ready():
            raise FileNotFoundError(
                f"未发现可用特征缓存 manifest: {Path(self._feature_dir) / 'manifest.json'}"
            )

        from sequence_dataset import SequenceArrays

        loader = self._get_feature_loader()
        normalized_dates = loader._normalize_dates(dates)
        resolved = loader.registry.resolve_groups(self.groups)
        stage_order = [
            stage_name
            for stage_name in loader.registry.topo_order(include_archived=False)
            if stage_name in resolved
        ]

        # —— 预分配 + 逐日流式填充：避免 np.concatenate 把"所有日块 + 输出"两份 14GB 同时
        #    压在内存里（119 天全票折就是被这一下打穿的）。先确定每天行数（已知免读、未知
        #    现读一遍并记账），按总行数一次性预分配输出，再逐日把单日块写进对应切片。 ——
        C = len(self.channels)
        nrows: List[int] = []
        for date in normalized_dates:
            d = int(date)
            n = self._day_nrows.get(d)
            if n is None:
                day = self._load_day(loader, stage_order, resolved, d, target_col)
                n = int(day[0].shape[0])   # _load_day 已顺带写 _day_nrows
            nrows.append(n)
        total = int(sum(nrows))

        features = np.empty((total, C), dtype=np.float32)
        labels = np.empty(total, dtype=np.float32)
        dates_arr = symbols_arr = intervals_arr = None
        off = 0
        for date, n in zip(normalized_dates, nrows):
            df, dl, dd, dsym, dint = self._load_day(loader, stage_order, resolved, int(date), target_col)
            if dates_arr is None:   # 用首日块的 dtype 定 meta 列类型，预分配其余三列
                dates_arr = np.empty(total, dtype=dd.dtype)
                symbols_arr = np.empty(total, dtype=dsym.dtype)
                intervals_arr = np.empty(total, dtype=dint.dtype)
            features[off:off + n] = df
            labels[off:off + n] = dl
            dates_arr[off:off + n] = dd
            symbols_arr[off:off + n] = dsym
            intervals_arr[off:off + n] = dint
            off += n

        return SequenceArrays(
            features=features,
            labels=labels,
            dates=dates_arr if dates_arr is not None else np.zeros(0, dtype=np.int64),
            symbols=symbols_arr if symbols_arr is not None else np.zeros(0, dtype=np.int64),
            intervals=intervals_arr if intervals_arr is not None else np.zeros(0, dtype=np.int64),
            channels=list(self.channels),
            has_label=True,
        )

    def _load_day(self, loader, stage_order, resolved, date, target_col):
        """
        读单个交易日的逐日 numpy 原料并**缓存**：返回
        ``(day_features[n,C] f32, labels[n] f32, dates[n], symbols[n], intervals[n])``。

        缓存键只取 ``date``——因为本适配器实例的列集（``self.channels`` / ``groups``）
        在一次 run 内固定，单日特征只随日期变化。命中缓存即跳过全部 h5 读 + pandas
        select，把「同一天被几十次 fit 反复重算」这条最大浪费一次性掐掉。

        缓存**蓄水式有界**（``_day_cache_cap`` 天）：装满后不淘汰也不新增，保留最早装入
        的共享基段（见 __init__）。返回的数组消费方均产新副本/只读切片填充，不就地改缓存
        内容，故可安全把缓存引用直接交出去。
        """
        cached = self._day_cache.get(date)
        if cached is not None:
            return cached

        meta_target_df = loader._load_meta_target_frame(date)
        day_feature_parts = []
        for stage_name in stage_order:
            stage_df = loader._read_stage_frame(stage_name, date)
            loader._assert_stage_alignment(stage_name, date, stage_df, meta_target_df)
            selected = loader._select_stage_columns(
                stage_name=stage_name,
                stage_df=stage_df,
                requested_columns=resolved[stage_name],
            )
            day_feature_parts.append(selected.to_numpy(dtype=np.float32, copy=False))

        if day_feature_parts:
            day_features = np.concatenate(day_feature_parts, axis=1).astype(np.float32, copy=False)
        else:
            day_features = np.zeros((len(meta_target_df), 0), dtype=np.float32)

        result = (
            day_features,
            meta_target_df[target_col].to_numpy(dtype=np.float32, copy=True),
            meta_target_df["date"].to_numpy(copy=True),
            meta_target_df["symbol"].to_numpy(copy=True),
            meta_target_df["interval"].to_numpy(copy=True),
        )
        self._day_nrows[date] = int(day_features.shape[0])   # 行数台账永久保留（极小）
        # 蓄水式有界：未满才装；装满后保留最早的共享基段、新日不再入缓存（封死内存峰）。
        if self._day_cache_cap is None or len(self._day_cache) < self._day_cache_cap:
            self._day_cache[date] = result
        return result

    def build(self, raw_day_df) -> np.ndarray:
        # build_feature_frames 内部按 (date,symbol,interval) 排序后逐日算，单日块行序
        # = (symbol,interval)，与统一行序契约一致。groups=None 即用构造时锁定的 groups；
        # 再强制按 self.channels 取列、定死列顺序（C 维布局固定）。
        xdf, _ = self._pipeline.build_feature_frames(raw_day_df, groups=None)
        return xdf[self.channels].to_numpy(dtype=np.float32)


# —— RawChannelAdapter 的通道分桶（按 raw 列语义固定，规格 §8.0；C 维布局即下面顺序） —— #
# ① 价类：相对 mid 归一 (p/midpx - 1)，价<=0（该 interval 无成交）时置 0（中性）。
_PRICE_REL_COLS = (
    "lastpx", "open", "high", "low",
    "bid0", "ask0", "bid4", "ask4", "bid9", "ask9", "bid19", "ask19",
    "tradeBuyHigh", "tradeBuyLow", "buyVwad",
    "tradeSellHigh", "tradeSellLow", "sellVwad",
    "addBuyHigh", "addBuyLow", "addSellHigh", "addSellLow",
    "cxlBuyHigh", "cxlBuyLow", "cxlSellHigh", "cxlSellLow",
)
# ② 量类：log1p 驯厚尾（非负：聚合 size / 笔数 / 量 / 额）。
_LOG_VOLUME_COLS = (
    "bsize0", "asize0", "bsize0_4", "asize0_4", "bsize5_9", "asize5_9", "bsize10_19", "asize10_19",
    "nTradeBuy", "tradeBuyQty", "tradeBuyTurnover",
    "nTradeSell", "tradeSellQty", "tradeSellTurnover",
    "nAddBuy", "addBuyQty", "addBuyTurnover",
    "nAddSell", "addSellQty", "addSellTurnover",
    "nCxlBuy", "cxlBuyQty", "cxlBuyTurnover",
    "nCxlSell", "cxlSellQty", "cxlSellTurnover",
)
# ③ 比率类：已是各档 turnover ratio（量纲驯过），直接透传，交脊柱 zscore 统计白化。
_RATIO_COLS = ("btr0_4", "atr0_4", "btr5_9", "atr5_9", "btr10_19", "atr10_19")
# ④ 价格路径：midpx 的日内对数收益（按 symbol 组内、首步 0、因果不跨日跨票），单独成一通道。
_MIDPX_COL = "midpx"


@register_adapter(AdapterKind.RAW_CHANNELS)
class RawChannelAdapter(InputAdapter):
    """
    把 ~59 个原始微结构通道做**最小语义归一**当通道（规格 §3.1 / §8.0；D2/TCN）。

    设计哲学：走 raw 这条线的意义就是**让网络自己学通道间交互**（对照 ``FeatureAdapter``
    喂 433 手工特征）。所以这里**只做"平稳化 + 量纲驯服"**，刻意不手搓 imbalance/OFI：

    - **价 → 相对 mid**（``_PRICE_REL_COLS``）：``p/midpx - 1``，行内无状态；价<=0（无成交）置 0。
    - **midpx → 日内对数收益**（``midpx_logret``）：按 symbol 组内 ``log(midpx_t)-log(midpx_{t-1})``，
      段首（symbol 变化）置 0——这是最关键的收益路径通道，**因果、不跨日跨票**。
    - **量 → log1p**（``_LOG_VOLUME_COLS``）：非负厚尾，``log1p`` 驯尾。
    - **比率 → 透传**（``_RATIO_COLS``）：各档 turnover ratio 已驯过量纲，交脊柱 zscore。

    通道布局固定（C 维顺序）：``midpx_logret`` → 价类(rel) → 量类(log) → 比率类。
    全程 numpy、行内/组内无状态变换，无跨日跨票泄漏风险。
    """

    EPS: float = 1e-12

    def __init__(self):
        # 通道名定死顺序即 C 维布局：路径 + 价(rel) + 量(log) + 比率。
        self.channels = (
            ["midpx_logret"]
            + [f"{c}_rel" for c in _PRICE_REL_COLS]
            + [f"{c}_log" for c in _LOG_VOLUME_COLS]
            + list(_RATIO_COLS)
        )
        self._required_raw = (_MIDPX_COL,) + _PRICE_REL_COLS + _LOG_VOLUME_COLS + _RATIO_COLS

    @classmethod
    def from_config(cls, adapter_config) -> "RawChannelAdapter":
        # RAW_CHANNELS 用固定的 59 通道，忽略 AdapterConfig.groups/columns（通道布局须固定）。
        return cls()

    def build(self, raw_day_df) -> np.ndarray:
        # 缺列即报错（通道布局须固定，不静默降级）。
        missing = [c for c in self._required_raw if c not in raw_day_df.columns]
        if missing:
            raise KeyError(f"RawChannelAdapter 缺原始列: {missing[:8]}{'...' if len(missing) > 8 else ''}")

        # 与统一行序契约一致：单日内按 (symbol, interval) 稳定排序。
        day = raw_day_df.sort_values(["symbol", "interval"], kind="mergesort").reset_index(drop=True)
        n = len(day)
        mid = day[_MIDPX_COL].to_numpy(dtype=np.float64)
        safe_mid = np.where(np.abs(mid) < self.EPS, np.nan, mid)   # 防除零

        cols: List[np.ndarray] = []

        # ④ midpx 日内对数收益（按 symbol 段内 diff，段首置 0，因果不跨票）
        syms = day["symbol"].to_numpy()
        logmid = np.log(np.where(mid > 0, mid, np.nan))
        logret = np.zeros(n, dtype=np.float64)
        if n > 1:
            logret[1:] = logmid[1:] - logmid[:-1]
        is_new_sym = np.empty(n, dtype=bool)
        is_new_sym[0] = True if n > 0 else False
        if n > 1:
            is_new_sym[1:] = syms[1:] != syms[:-1]
        logret[is_new_sym] = 0.0
        cols.append(np.nan_to_num(logret, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))

        # ① 价相对 mid（价<=0 置 0）
        for c in _PRICE_REL_COLS:
            p = day[c].to_numpy(dtype=np.float64)
            rel = np.where(p > 0, p / safe_mid - 1.0, 0.0)
            cols.append(np.nan_to_num(rel, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))

        # ② 量 log1p（负值截 0 再 log1p）
        for c in _LOG_VOLUME_COLS:
            v = day[c].to_numpy(dtype=np.float64)
            lv = np.log1p(np.where(v > 0, v, 0.0))
            cols.append(np.nan_to_num(lv, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))

        # ③ 比率透传
        for c in _RATIO_COLS:
            r = day[c].to_numpy(dtype=np.float64)
            cols.append(np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))

        return np.stack(cols, axis=1).astype(np.float32)   # [n_rows_day, C]


# ================================================================== #
# ModelCartridge 接口 + numpy 参考模型
# ================================================================== #

# ---- TCN 搜索空间（原始微结构通道，深层卷积有意义，num_layers 可到 4） ----
STRUCTURE_SEARCH_SPACE: Dict = {
    "seq_len":     {"type": "choice", "values": [16, 32, 48, 64]},
    "hidden_size": {"type": "choice", "values": [32, 64, 128]},
    "num_layers":  {"type": "int", "low": 1, "high": 4},
}

# ---- GRU 搜索空间（433 工程特征输入，特征已含多步期信息，浅层即可） ----
# - seq_len：去掉 64，聚焦 16/32/48（TCN expanding 已证 48 对原始通道负，但特征版值得试）
# - num_layers：上限压 2——特征已预编码多步期，深层 GRU 只增噪声和过拟合风险
GRU_SEARCH_SPACE: Dict = {
    "seq_len":     {"type": "choice", "values": [16, 32, 48]},
    "hidden_size": {"type": "choice", "values": [32, 64, 128]},
    "num_layers":  {"type": "int", "low": 1, "high": 2},
}

# ---- 截面模型搜索空间（结构旋钮；λ 走 --hparams 不进此空间，见 §Q1） ----
# 时序腿沿用 GRU 浅层；截面腿 set-attention 少头低维，重正则换跨时间稳（§11.5）。
XSECTION_SEARCH_SPACE: Dict = {
    "seq_len":     {"type": "choice", "values": [16, 32, 48]},
    "hidden_size": {"type": "choice", "values": [32, 64]},
    "num_layers":  {"type": "int", "low": 1, "high": 2},
}


def _require_torch():
    """
    按需导入 torch。

    约束：``dl_models.py`` 作为卡带注册中心会被脊柱频繁 import；若在模块顶层直接
    import torch，会把“torch-free 脊柱”这条纪律打破，也会让没装 torch 的环境在
    仅使用参考模型时直接炸掉。因此这里统一走懒加载：真正进入 TCN/LSTM 卡带时才导。
    """
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError(
            "当前环境未安装 PyTorch，无法使用 TCN/LSTM 卡带。"
            "如只想跑 torch-free 脊柱，请继续使用 reference_* 模型。"
        ) from exc
    return torch, nn, F


def _resolve_torch_device(torch, requested: str) -> str:
    """
    解析本次训练实际使用的 device。

    口径：
    - 显式要求 ``cuda`` 但本机不可用 → 立即报错，避免用户误以为已经上卡。
    - ``auto`` / 空值 → 有卡用 cuda，否则退 cpu。
    - 其余显式值直接透传给 torch（如 ``cpu`` / ``cuda:0``）。
    """
    req = str(requested or "auto").strip().lower()
    if req in ("", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if req.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("请求使用 CUDA，但当前 torch.cuda.is_available() 为 False。")
    return requested


class _GpuWindowSource:
    """
    把一个 ``SequenceDataset`` 的「归一化特征矩阵 + 窗口索引」喂成 device 上的
    ``[B, L, C]`` batch 流，核心目标：**让 GPU 不再干等 CPU 拼数据**。

    本机现实（决定了下面三条路怎么选）：
    - 内存充足（~34GB），整张 ``[N, C]`` 特征矩阵舒服待在内存，**全量不砍**；
    - 显存小（8G 卡，可用 ~6.5G），而 433 通道 × 百万行的训练矩阵动辄 7–15GB，
      **塞不进显存**——所以"整张驻留显存"只在小抽样 / 筛选早折偶尔成立。

    三条路（按 device + 是否真装得下显存自动选）：
    1. **resident（纯显存快路）**：仅当整张矩阵确实装得进显存预算时才走（**事前按
       真实空闲显存判断、绝不盲分配再接 OOM**）；窗口 gather 全在显存带宽上做。
    2. **prefetched（预取流式，正式大跑的主路）**：装不下显存时——后台线程在 GPU
       算第 i 批时，提前把第 i+1 批的 ``[B,L,C]`` 在 CPU gather 好、落到 pinned 内存，
       主线程再 ``non_blocking`` 异步拷上卡。CPU 备料与 GPU 计算重叠 → GPU 不再干等；
       在途同时只有 ``_PREFETCH`` 个 batch（~百 MB 级），**内存有界、永不 OOM**。
    3. **cpu**：device 为 cpu（单测 / Mac）时退化为纯 CPU gather，不涉及 pin/异步。

    torch 仍只在本卡带层出现，脊柱保持 torch-free；CPU 单测路径完全不受影响。
    """

    # 预取队列深度：在途同时最多这么多 batch，限制 pinned 内存占用。
    _PREFETCH = 3
    # resident 只在「估算字节 < 此比例 × 当前空闲显存」时才走，留足碎片 / 激活余量。
    _VRAM_FRAC = 0.6

    def __init__(self, ds, torch, device: str):
        self._torch = torch
        self._ds = ds
        self.device = str(device)
        self.seq_len = int(ds.seq_len)
        self.n_windows = int(len(ds))
        self.mode = "cpu"            # cpu / resident / prefetched
        # resident 用的 device 句柄
        self.feats = self.label_rows = self.offsets = self.labels = None
        # CPU 侧原料：from_numpy 零拷贝共享 ds 内已有的大矩阵（不再复制一份 14GB）。
        self._feats_cpu = torch.from_numpy(np.ascontiguousarray(ds.feature_matrix()))
        self._label_rows_cpu = torch.from_numpy(
            np.ascontiguousarray(ds.label_rows).astype(np.int64, copy=False))
        self._offsets_cpu = torch.from_numpy(
            np.ascontiguousarray(ds.window_offsets()).astype(np.int64, copy=False))
        self._labels_cpu = torch.from_numpy(
            np.ascontiguousarray(ds.window_labels()).astype(np.float32, copy=False))

        if self.n_windows == 0 or not self.device.startswith("cuda"):
            return

        # ---- 事前显存预算判断：只有确实装得下才 resident，绝不盲分配 14GB 再接 OOM ---- #
        feat_bytes = self._feats_cpu.numel() * 4
        idx_bytes = (self._label_rows_cpu.numel() + self._offsets_cpu.numel()) * 8 \
            + self._labels_cpu.numel() * 4
        need = feat_bytes + idx_bytes
        try:
            free_bytes, _total = torch.cuda.mem_get_info()
        except Exception:
            free_bytes = 0
        if need < self._VRAM_FRAC * free_bytes:
            try:
                self.feats = self._feats_cpu.to(self.device, dtype=torch.float32)
                self.label_rows = self._label_rows_cpu.to(self.device)
                self.offsets = self._offsets_cpu.to(self.device)
                self.labels = self._labels_cpu.to(self.device)
                self.mode = "resident"
            except RuntimeError as exc:
                # 二重保险：预算判断已放行仍 OOM（碎片等），也绝不上抛——退预取流式。
                if "out of memory" not in str(exc).lower():
                    raise
                self.feats = self.label_rows = self.offsets = self.labels = None
                torch.cuda.empty_cache()
                self.mode = "prefetched"
        else:
            self.mode = "prefetched"

    def __len__(self) -> int:
        return self.n_windows

    @property
    def gpu_resident(self) -> bool:   # 兼容旧字段名
        return self.mode == "resident"

    def _make_order(self, shuffle: bool, seed: Optional[int]):
        """生成窗口遍历顺序（CPU 张量）；shuffle 用可复现 Generator。"""
        torch = self._torch
        n = self.n_windows
        if shuffle:
            g = torch.Generator()
            if seed is not None:
                g.manual_seed(int(seed))
            return torch.randperm(n, generator=g)
        return torch.arange(n)

    def iter_batches(self, batch_size: int, shuffle: bool = False, seed: Optional[int] = None):
        if self.n_windows == 0:
            return
        if self.mode == "resident":
            yield from self._iter_resident(batch_size, shuffle, seed)
        elif self.mode == "prefetched":
            yield from self._iter_prefetched(batch_size, shuffle, seed)
        else:
            yield from self._iter_cpu(batch_size, shuffle, seed)

    def _iter_resident(self, batch_size: int, shuffle: bool, seed: Optional[int]):
        """整张驻留 device 的快路径：窗口 gather 全在 device 上做。"""
        order = self._make_order(shuffle, seed).to(self.device)
        n = self.n_windows
        for start in range(0, n, batch_size):
            sel = order[start: start + batch_size]
            rows = self.label_rows[sel][:, None] + self.offsets[None, :]   # [b, L]
            yield self.feats[rows], self.labels[sel]                       # [b,L,C], [b]

    def _gather_cpu(self, sel):
        """在 CPU 上 gather 一个 batch 的 ``([b,L,C], [b])``（torch 索引，GIL 在大拷贝时释放）。"""
        rows = self._label_rows_cpu[sel][:, None] + self._offsets_cpu[None, :]   # [b, L]
        return self._feats_cpu[rows], self._labels_cpu[sel]

    def _iter_cpu(self, batch_size: int, shuffle: bool, seed: Optional[int]):
        """device 为 cpu 时的纯 CPU 路径（单测 / Mac）：直接产 CPU 张量。"""
        order = self._make_order(shuffle, seed)
        n = self.n_windows
        for start in range(0, n, batch_size):
            sel = order[start: start + batch_size]
            yield self._gather_cpu(sel)

    def _iter_prefetched(self, batch_size: int, shuffle: bool, seed: Optional[int]):
        """
        预取流式（正式大跑主路）：后台线程 CPU gather→pinned，主线程异步拷卡。

        生命周期保证：``x_pin``/``y_pin`` 由本生成器帧在 ``yield`` 处持活——直到消费方
        下次 ``next()`` 才释放，因此 ``non_blocking`` 拷贝期间 pinned 源不会被回收
        （与 PyTorch DataLoader 的 pin_memory 路径同一条保证）。
        """
        order = self._make_order(shuffle, seed)
        n = self.n_windows
        q: "queue.Queue" = queue.Queue(maxsize=self._PREFETCH)
        stop = threading.Event()
        _STOP = object()

        def producer():
            try:
                for start in range(0, n, batch_size):
                    if stop.is_set():
                        break
                    sel = order[start: start + batch_size]
                    x_cpu, y_cpu = self._gather_cpu(sel)
                    item = (x_cpu.pin_memory(), y_cpu.pin_memory())
                    # 带超时的 put：消费方提前停了也能周期性检查 stop、不会永久阻塞。
                    while not stop.is_set():
                        try:
                            q.put(item, timeout=0.5)
                            break
                        except queue.Full:
                            continue
            except Exception as exc:   # 异常回传主线程，避免静默卡死
                q.put(exc)
            finally:
                q.put(_STOP)

        t = threading.Thread(target=producer, daemon=True)
        t.start()
        try:
            while True:
                item = q.get()
                if item is _STOP:
                    break
                if isinstance(item, Exception):
                    raise item
                x_pin, y_pin = item
                x = x_pin.to(self.device, non_blocking=True)
                y = y_pin.to(self.device, non_blocking=True)
                yield x, y
        finally:
            # 消费方提前退出时：置 stop + 排空队列解阻塞 producer，再带超时 join，绝不卡死。
            stop.set()
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass
            t.join(timeout=5.0)

    def free(self) -> None:
        """显式释放 device / CPU 句柄，给下一段 dataset 腾资源。"""
        self.feats = self.label_rows = self.offsets = self.labels = None
        self._feats_cpu = self._label_rows_cpu = self._offsets_cpu = self._labels_cpu = None


class _GpuCrossSectionSource:
    """
    把一个 ``CrossSectionDataset`` 的「白化特征矩阵 + 各快照窗口索引」喂成 device 上的
    ``(X[B,maxN,L,C], y[B,maxN], mask[B,maxN])`` batch 流（``B`` = 一批多少个 (date,interval)
    快照）。与 ``_GpuWindowSource`` 同款瘦内存纪律，只是 gather 形状从逐窗 ``[b,L,C]`` 升到
    逐截面 ``[B,maxN,L,C]``，故显存预算更保守。

    两态（按 device + 显存预算自动选；省去后台预取线程，每批瞬时有界、内存安全）：
    1. **resident**：整张白化矩阵装得进显存预算时驻留 device，按快照 gather 全在卡上做；
    2. **cpu**：device=cpu（单测）或显存装不下时——每批在 CPU gather 好 ``[B,maxN,L,C]``
       再整批 ``.to(device)``，在途只有当前一批，**内存有界、永不 OOM**。

    pad 位（``mask=False``）的窗口末行号填 0（合法行、gather 不越界），其特征/标签经
    时序腿后由 mask 在损失/收集处屏蔽，不污染结果。
    """

    # 截面 batch 的 [B,maxN,L,C] 瞬时也吃显存，故 resident 阈值比逐窗源更保守。
    _VRAM_FRAC = 0.45

    def __init__(self, xs_ds, torch, device: str):
        self._torch = torch
        self._xs = xs_ds
        self.device = str(device)
        self.seq_len = int(xs_ds.seq_len)
        self.snapshots = xs_ds.snapshots
        self.n_snaps = len(self.snapshots)
        self._offsets_cpu = torch.from_numpy(
            np.ascontiguousarray(xs_ds.window_offsets()).astype(np.int64, copy=False))
        self._feats_cpu = torch.from_numpy(np.ascontiguousarray(xs_ds.feature_matrix()))
        self._labels_cpu = torch.from_numpy(
            np.ascontiguousarray(xs_ds.arrays.labels).astype(np.float32, copy=False))
        self.mode = "cpu"
        self.feats_gpu = self.offsets_gpu = self.labels_gpu = None

        if self.n_snaps == 0 or not self.device.startswith("cuda"):
            return
        # 事前显存预算：只有整张特征矩阵确实装得下才 resident，绝不盲分配再接 OOM。
        feat_bytes = self._feats_cpu.numel() * 4
        try:
            free_bytes, _total = torch.cuda.mem_get_info()
        except Exception:
            free_bytes = 0
        if feat_bytes < self._VRAM_FRAC * free_bytes:
            try:
                self.feats_gpu = self._feats_cpu.to(self.device, dtype=torch.float32)
                self.offsets_gpu = self._offsets_cpu.to(self.device)
                self.labels_gpu = self._labels_cpu.to(self.device)
                self.mode = "resident"
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                self.feats_gpu = self.offsets_gpu = self.labels_gpu = None
                torch.cuda.empty_cache()
                self.mode = "cpu"

    def __len__(self) -> int:
        return self.n_snaps

    @staticmethod
    def _pack(snaps):
        """把一批变长快照打成 ``(lr_pad[B,maxN] int64, mask[B,maxN] bool)``（pad 行号填 0）。"""
        sizes = [len(s) for s in snaps]
        maxN = max(sizes) if sizes else 0
        B = len(snaps)
        lr_pad = np.zeros((B, maxN), dtype=np.int64)
        mask = np.zeros((B, maxN), dtype=bool)
        for bi, s in enumerate(snaps):
            nk = len(s)
            lr_pad[bi, :nk] = s
            mask[bi, :nk] = True
        return lr_pad, mask

    def iter_batches(self, snap_batch: int, shuffle: bool = False, seed: Optional[int] = None):
        if self.n_snaps == 0:
            return
        torch = self._torch
        order = np.arange(self.n_snaps)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(order)
        for start in range(0, self.n_snaps, snap_batch):
            sel = order[start: start + snap_batch]
            snaps = [self.snapshots[i] for i in sel]
            lr_pad_np, mask_np = self._pack(snaps)
            if self.mode == "resident":
                lr_pad = torch.from_numpy(lr_pad_np).to(self.device)
                rows = lr_pad[:, :, None] + self.offsets_gpu[None, None, :]   # [B,maxN,L]
                X = self.feats_gpu[rows]                                       # [B,maxN,L,C]
                y = self.labels_gpu[lr_pad]                                    # [B,maxN]
                mask = torch.from_numpy(mask_np).to(self.device)
            else:
                lr_pad = torch.from_numpy(lr_pad_np)
                rows = lr_pad[:, :, None] + self._offsets_cpu[None, None, :]   # [B,maxN,L]（CPU）
                X = self._feats_cpu[rows].to(self.device)                      # 整批一次搬卡
                y = self._labels_cpu[lr_pad].to(self.device)
                mask = torch.from_numpy(mask_np).to(self.device)
            yield X, y, mask

    def free(self) -> None:
        self.feats_gpu = self.offsets_gpu = self.labels_gpu = None
        self._feats_cpu = self._labels_cpu = self._offsets_cpu = None


class _CausalConvBlock:
    """
    极薄的因果卷积包装。

    这里不用 ``padding='same'``，因为 same padding 会在左右两侧同时补零；那样时间 t
    的输出会看到未来位置信息，直接破坏“因果卷积”这一条根纪律。正确做法是只在左侧补。
    """

    def __init__(self, conv, pad: int, torch):
        self.conv = conv
        self.pad = int(pad)
        self._torch = torch

    def __call__(self, x):
        if self.pad > 0:
            x = self._torch.nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)


def _build_tcn_module(input_channels: int, hidden_size: int, num_layers: int, dropout: float):
    """
    构造一个轻量 TCN 回归头。

    结构选择刻意保守：
    - 残差块 + 膨胀卷积，覆盖 226 步日内序列足够；
    - 全局平均池化后接线性头，输出单个 ``fret12`` 标量；
    - 不在这里搞花哨结构，先把“接线 + 因果 + 可训 + 可搜”打通。
    """
    torch, nn, _ = _require_torch()

    class ResidualBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, dilation: int):
            super().__init__()
            kernel_size = 3
            pad = dilation * (kernel_size - 1)
            self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation)
            self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, dilation=dilation)
            self.conv1_wrap = _CausalConvBlock(self.conv1, pad, torch)
            self.conv2_wrap = _CausalConvBlock(self.conv2, pad, torch)
            self.norm1 = nn.BatchNorm1d(out_ch)
            self.norm2 = nn.BatchNorm1d(out_ch)
            self.act = nn.GELU()
            self.dropout = nn.Dropout(float(dropout))
            self.skip = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, kernel_size=1)

        def forward(self, x):
            residual = self.skip(x)
            out = self.conv1_wrap(x)
            out = self.norm1(out)
            out = self.act(out)
            out = self.dropout(out)
            out = self.conv2_wrap(out)
            out = self.norm2(out)
            out = self.act(out)
            out = self.dropout(out)
            return out + residual

    class TCNRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            in_ch = int(input_channels)
            for i in range(int(num_layers)):
                dilation = 2 ** i
                layers.append(ResidualBlock(in_ch, int(hidden_size), dilation))
                in_ch = int(hidden_size)
            self.backbone = nn.ModuleList(layers)
            self.head = nn.Linear(int(hidden_size), 1)

        def forward(self, x):
            # 输入约定：[B, L, C]，Conv1d 期望 [B, C, L]
            out = x.transpose(1, 2)
            for block in self.backbone:
                out = block(out)
            # 只用时间维平均池化，保持回归头极薄，避免再引入额外泄漏面。
            pooled = out.mean(dim=-1)
            return self.head(pooled).squeeze(-1)

    return TCNRegressor()

@dataclass
class TrainRecord:
    """
    卡带 ``fit`` 的单向上报（规格 §3.2/§4）：逐 epoch 曲线 + best epoch + 用时。

    供脊柱判过拟合（train vs earlystop gap / 曲线）与 HPO 早杀整个 trial。
    **单向**——脊柱只读不写，绝不反过来控制卡带内层 epoch 循环。
    """
    train_curve: List[float] = field(default_factory=list)
    earlystop_curve: List[float] = field(default_factory=list)
    best_epoch: int = 0
    n_epochs: int = 0
    fit_seconds: float = 0.0
    extra: Dict = field(default_factory=dict)


class ModelCartridge(ABC):
    """
    模型卡带接口（规格 §3.2）。唯一允许 import torch 的层（参考模型 torch-free）。

    - ``search_space``: 本模型私有的超参声明（不进 config/ 全局枚举）。
    - ``required_adapter``: 类型化引用，``None`` = 不挑适配器（参考模型）。
    - ``fit(train_ds, earlystop_ds, hparams, seed) -> TrainRecord``：内含完整训练循环 +
      早停（看 earlystop_ds，不碰 scoring）。
    - ``predict(ds) -> np.ndarray``：输出留在 ``fret12`` 量纲、与 ds 窗口顺序对齐。
    """

    search_space: ClassVar[Dict] = {}
    required_adapter: ClassVar[Optional[AdapterKind]] = None

    @classmethod
    @abstractmethod
    def from_config(cls, model_config) -> "ModelCartridge":
        ...

    @abstractmethod
    def fit(self, train_ds, earlystop_ds, hparams: Dict, seed: int) -> TrainRecord:
        ...

    @abstractmethod
    def predict(self, ds) -> np.ndarray:
        ...


@register_model(ModelKind.REFERENCE_ZERO)
class ReferenceZeroCartridge(ModelCartridge):
    """恒 0 预测。corr 恒约 0，作 sanity 基线（注意：恒 0 无法探测泄漏，探测靠末步线性）。"""

    required_adapter = None   # 不挑适配器

    @classmethod
    def from_config(cls, model_config) -> "ReferenceZeroCartridge":
        return cls()

    def fit(self, train_ds, earlystop_ds, hparams: Dict, seed: int) -> TrainRecord:
        return TrainRecord(train_curve=[0.0], n_epochs=1, fit_seconds=0.0, extra={"kind": "reference_zero"})

    def predict(self, ds) -> np.ndarray:
        return np.zeros(len(ds), dtype=np.float32)


@register_model(ModelKind.REFERENCE_LAST)
class ReferenceLastCartridge(ModelCartridge):
    """
    **防泄漏探测器**：用窗口**末步**各通道做一个 numpy 线性回归预测标签。

    机理：因果窗口下末步只含 ``t`` 时刻信息——若标签依赖未来 ``t+k``，它学不到、打低分；
    一旦 WindowIndexer 跨界 / 标签对齐错位 / Normalizer 用了 val 统计而漏进未来信息，
    它立刻能吃到并涨分。所以"它在干净管线上打低分"是 D0 的核心验收闸（规格 §5.5）。
    """

    required_adapter = None   # 不挑适配器，可在合成 identity 通道或真实 433 特征上探测

    def __init__(self):
        self._w: Optional[np.ndarray] = None   # [C]
        self._b: float = 0.0

    @classmethod
    def from_config(cls, model_config) -> "ReferenceLastCartridge":
        return cls()

    def fit(self, train_ds, earlystop_ds, hparams: Dict, seed: int) -> TrainRecord:
        t0 = time.time()
        X, y = train_ds.gather_all()              # X[B,L,C], y[B]
        if X.shape[0] == 0:
            self._w = np.zeros(train_ds.n_channels, dtype=np.float64)
            self._b = 0.0
            return TrainRecord(train_curve=[0.0], n_epochs=1, fit_seconds=time.time() - t0,
                               extra={"kind": "reference_last", "empty": True})
        x_last = X[:, -1, :].astype(np.float64)   # [B, C] 窗口末步
        # 最小二乘线性回归（含 bias），torch-free。
        A = np.concatenate([x_last, np.ones((x_last.shape[0], 1))], axis=1)  # [B, C+1]
        coef, *_ = np.linalg.lstsq(A, y.astype(np.float64), rcond=None)
        self._w = coef[:-1]
        self._b = float(coef[-1])
        pred = A @ coef
        train_mse = float(np.mean((pred - y) ** 2))
        return TrainRecord(train_curve=[train_mse], n_epochs=1, fit_seconds=time.time() - t0,
                           extra={"kind": "reference_last"})

    def predict(self, ds) -> np.ndarray:
        X, _ = ds.gather_all()
        if X.shape[0] == 0:
            return np.zeros(0, dtype=np.float32)
        x_last = X[:, -1, :].astype(np.float64)
        return (x_last @ self._w + self._b).astype(np.float32)


@register_model(ModelKind.REFERENCE_POOL)
class ReferencePoolCartridge(ModelCartridge):
    """
    窗口**均值池化** + numpy 线性回归（torch-free）。

    用途有三：① 比 ReferenceLast 多看整窗（池化）的 sanity 对照；② **声明
    ``STRUCTURE_SEARCH_SPACE``**，当 HPO Searcher 的 torch-free 被测对象——不同
    ``seq_len`` 会改变池化窗口与样本数、产出不同 val_corr，让搜索/排名机制可在 Mac
    上端到端验证；③ 给未来 ``TCNCartridge`` 当 search_space + fit/predict 形态模板。

    说明：本参考模型用不上 ``hidden_size`` / ``num_layers``（numpy 线性无此结构），
    仅接收并原样记进 TrainRecord——真正消费它们的是等 4060 的 torch 卡带。
    """

    search_space: ClassVar[Dict] = STRUCTURE_SEARCH_SPACE
    required_adapter = None   # 不挑适配器

    def __init__(self):
        self._w: Optional[np.ndarray] = None   # [C]
        self._b: float = 0.0

    @classmethod
    def from_config(cls, model_config) -> "ReferencePoolCartridge":
        return cls()

    @staticmethod
    def _pool(ds):
        """整窗对 L 轴均值池化：``[B, L, C] -> [B, C]``。"""
        X, y = ds.gather_all()
        if X.shape[0] == 0:
            return np.zeros((0, ds.n_channels), dtype=np.float64), y
        return X.mean(axis=1).astype(np.float64), y

    def fit(self, train_ds, earlystop_ds, hparams: Dict, seed: int) -> TrainRecord:
        t0 = time.time()
        Xp, y = self._pool(train_ds)
        if Xp.shape[0] == 0:
            self._w = np.zeros(train_ds.n_channels, dtype=np.float64)
            self._b = 0.0
            return TrainRecord(train_curve=[0.0], n_epochs=1, fit_seconds=time.time() - t0,
                               extra={"kind": "reference_pool", "empty": True})
        A = np.concatenate([Xp, np.ones((Xp.shape[0], 1))], axis=1)   # [B, C+1]
        coef, *_ = np.linalg.lstsq(A, y.astype(np.float64), rcond=None)
        self._w = coef[:-1]
        self._b = float(coef[-1])
        train_mse = float(np.mean((A @ coef - y) ** 2))
        return TrainRecord(train_curve=[train_mse], n_epochs=1, fit_seconds=time.time() - t0,
                           extra={"kind": "reference_pool", "hparams": dict(hparams)})

    def predict(self, ds) -> np.ndarray:
        Xp, _ = self._pool(ds)
        if Xp.shape[0] == 0:
            return np.zeros(0, dtype=np.float32)
        return (Xp @ self._w + self._b).astype(np.float32)


@register_model(ModelKind.TCN)
class TCNCartridge(ModelCartridge):
    """
    主攻卡带：TCN-on-原始微结构。

    设计边界严格贴规格：
    - **唯一 torch 层**：nn.Module / optimizer / autograd / device 全封在这里；
    - **required_adapter 固定 RAW_CHANNELS**：让组装期就拦住“TCN 却喂错输入”的配置错误；
    - **search_space 复用 STRUCTURE_SEARCH_SPACE**：序列长由 trainer 开窗，hidden/layers 在
      这里真正消费。

    训练策略先取最小可用闭环，不在第一版里把复杂度堆上去：
    - loss = MSE（与最终回归目标同量纲）；
    - early stopping 看训练区尾段的 ``earlystop_ds``，绝不碰 scoring；
    - 如果 earlystop 为空，就退化成看训练损失，至少保持接口闭环不炸。
    """

    search_space: ClassVar[Dict] = STRUCTURE_SEARCH_SPACE
    required_adapter = AdapterKind.RAW_CHANNELS

    def __init__(self):
        self._torch = None
        self._model = None
        self._device = "cpu"
        self._fallback_bias = 0.0
        self._fitted = False

    @classmethod
    def from_config(cls, model_config) -> "TCNCartridge":
        return cls()

    @staticmethod
    def _default_hparams(hparams: Dict) -> Dict:
        """
        合并本卡带的默认训练超参与搜索得到的结构超参。

        结构三旋钮（seq_len / hidden_size / num_layers）里，本卡带只真正消费后两者；
        ``seq_len`` 仍由 Searcher/Trainer 决定开窗，不在此重复处理。
        """
        merged = {
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.10,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 256,
            "max_epochs": 20,
            "patience": 4,
            "grad_clip": 1.0,
            "device": "auto",
        }
        merged.update(dict(hparams or {}))
        return merged

    def _eval_loss(self, ds, loss_fn) -> float:
        """按自然顺序评估一个数据集的平均损失。"""
        torch = self._torch
        if len(ds) == 0:
            return float("inf")
        total = 0.0
        count = 0
        self._model.eval()
        with torch.no_grad():
            for xb, yb in ds.iter_batches(batch_size=512, shuffle=False):
                x = torch.as_tensor(xb, device=self._device, dtype=torch.float32)
                y = torch.as_tensor(yb, device=self._device, dtype=torch.float32)
                pred = self._model(x)
                loss = loss_fn(pred, y)
                total += float(loss.item()) * len(yb)
                count += len(yb)
        return total / max(count, 1)

    def fit(self, train_ds, earlystop_ds, hparams: Dict, seed: int) -> TrainRecord:
        torch, nn, _ = _require_torch()
        cfg = self._default_hparams(hparams)
        self._torch = torch
        self._device = _resolve_torch_device(torch, cfg.get("device", "auto"))

        t0 = time.time()
        if len(train_ds) == 0:
            self._model = None
            self._fitted = True
            self._fallback_bias = 0.0
            return TrainRecord(
                train_curve=[0.0],
                earlystop_curve=[0.0] if len(earlystop_ds) else [],
                best_epoch=0,
                n_epochs=1,
                fit_seconds=time.time() - t0,
                extra={"kind": "tcn", "empty": True, "device": self._device},
            )

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

        self._model = _build_tcn_module(
            input_channels=train_ds.n_channels,
            hidden_size=int(cfg["hidden_size"]),
            num_layers=int(cfg["num_layers"]),
            dropout=float(cfg["dropout"]),
        ).to(self._device)

        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=float(cfg["lr"]),
            weight_decay=float(cfg["weight_decay"]),
        )
        loss_fn = nn.MSELoss()
        batch_size = int(cfg["batch_size"])
        max_epochs = int(cfg["max_epochs"])
        patience = int(cfg["patience"])
        grad_clip = float(cfg["grad_clip"])

        best_metric = float("inf")
        best_epoch = 0
        best_state = None
        no_improve = 0
        train_curve: List[float] = []
        early_curve: List[float] = []

        # 兜底偏置：只需要“训练标签的均值”，不需要把整份 ``X[B,L,C]`` 一次性物化进内存。
        # 这里改走 label_frame 只取 ``fret12`` 一列，避免海选单折在 40 天训练窗上额外造一个
        # 数 GB 级的窗口张量副本；训练主路径仍保持 iter_batches 流式。
        train_labels = train_ds.label_frame()["fret12"].to_numpy(dtype=np.float32)
        self._fallback_bias = float(np.mean(train_labels)) if len(train_labels) else 0.0

        for epoch in range(1, max_epochs + 1):
            self._model.train()
            total = 0.0
            count = 0
            for xb, yb in train_ds.iter_batches(batch_size=batch_size, shuffle=True, seed=int(seed) + epoch):
                x = torch.as_tensor(xb, device=self._device, dtype=torch.float32)
                y = torch.as_tensor(yb, device=self._device, dtype=torch.float32)
                optimizer.zero_grad(set_to_none=True)
                pred = self._model(x)
                loss = loss_fn(pred, y)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=grad_clip)
                optimizer.step()
                total += float(loss.item()) * len(yb)
                count += len(yb)

            train_loss = total / max(count, 1)
            train_curve.append(train_loss)

            if len(earlystop_ds) > 0:
                metric = self._eval_loss(earlystop_ds, loss_fn)
                early_curve.append(metric)
            else:
                metric = train_loss

            if metric + 1e-8 < best_metric:
                best_metric = metric
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in self._model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)

        self._fitted = True
        return TrainRecord(
            train_curve=train_curve,
            earlystop_curve=early_curve,
            best_epoch=best_epoch,
            n_epochs=len(train_curve),
            fit_seconds=time.time() - t0,
            extra={
                "kind": "tcn",
                "device": self._device,
                "hidden_size": int(cfg["hidden_size"]),
                "num_layers": int(cfg["num_layers"]),
            },
        )

    def predict(self, ds) -> np.ndarray:
        if len(ds) == 0:
            return np.zeros(0, dtype=np.float32)
        if not self._fitted or self._model is None:
            return np.full(len(ds), self._fallback_bias, dtype=np.float32)

        torch = self._torch
        # 预测同样把窗口源驻留 device（OOM 自动降级流式）：窗口 gather 在卡上做、
        # 零逐 batch CPU 拼接 + 零逐 batch H2D，与 fit/eval 同一条快路径。
        src = _GpuWindowSource(ds, torch, self._device)
        preds: List[np.ndarray] = []
        self._model.eval()
        try:
            with torch.no_grad():
                # shuffle=False：保持窗口原序，预测结果与 ds 行严格对齐。
                for xb, _ in src.iter_batches(batch_size=4096, shuffle=False):
                    pred = self._model(xb).detach().cpu().numpy().astype(np.float32)
                    preds.append(pred)
        finally:
            src.free()
        return np.concatenate(preds, axis=0) if preds else np.zeros(0, dtype=np.float32)


def _build_gru_module(input_channels: int, hidden_size: int, num_layers: int, dropout: float):
    """
    构造轻量 GRU 回归头。

    设计选择与 TCN 对称保守：
    - nn.GRU（batch_first=True）接收 [B, L, C]，取**末步输出**（等价于最后层最后时刻隐藏态）；
    - 末步天然因果：GRU 从左到右顺序处理，t=L 时刻已积累全窗上下文，不跨越标签时刻；
    - 不加 LayerNorm / Attention 等附件——FeatureAdapter 已经过脊柱 zscore 白化，量纲一致；
    - dropout 仅在 num_layers > 1 时有效（PyTorch 限制，单层 dropout 静默忽略）。
    """
    torch, nn, _ = _require_torch()

    class GRURegressor(nn.Module):
        def __init__(self):
            super().__init__()
            gru_drop = float(dropout) if int(num_layers) > 1 else 0.0
            self.gru = nn.GRU(
                input_size=int(input_channels),
                hidden_size=int(hidden_size),
                num_layers=int(num_layers),
                batch_first=True,
                dropout=gru_drop,
            )
            self.head = nn.Linear(int(hidden_size), 1)

        def forward(self, x):
            # x: [B, L, C]；GRU 输出 (output[B,L,H], h_n[num_layers,B,H])
            # output[:, -1, :] == h_n[-1]（最后层、最后时刻），语义更直观。
            output, _ = self.gru(x)
            last = output[:, -1, :]          # [B, H]
            return self.head(last).squeeze(-1)  # [B]

    return GRURegressor()


@register_model(ModelKind.GRU)
class GRUCartridge(ModelCartridge):
    """
    第二卡带：GRU-on-433工程特征。

    与 TCNCartridge 形成"架构 × 输入"对照：
    - TCN 喂 59 个原始微结构通道，让网络自学交互；
    - GRU 喂 433 个工程特征，把传统特征工程的先验直接注入 DL。

    required_adapter 固定 FEATURE_433：
    - 用传统侧已沉淀的截面归一化特征（cross-z / cross-rank / OFI / 动量等）；
    - FeatureAdapter 复用 SubmissionFeaturePipeline，train/serve 同口径，无 skew 风险。

    训练策略与 TCNCartridge 完全一致（MSE + AdamW + 早停 + 梯度裁剪），降低对照噪声。
    """

    search_space: ClassVar[Dict] = GRU_SEARCH_SPACE
    required_adapter = AdapterKind.FEATURE_433

    def __init__(self):
        self._torch = None
        self._model = None
        self._device = "cpu"
        self._fallback_bias = 0.0
        self._fitted = False
        # 全局线性 rescale a·ŷ+b（§定稿 第 2 条，预测阶段永远套）：默认恒等，fit 后被
        # 训练段 OLS 标定覆盖。a=1/b=0 时 predict 等价不 rescale（空训练兜底安全）。
        self._rescale_a = 1.0
        self._rescale_b = 0.0

    @classmethod
    def from_config(cls, model_config) -> "GRUCartridge":
        return cls()

    @staticmethod
    def _default_hparams(hparams: Dict) -> Dict:
        merged = {
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.10,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            # 损失对齐杠杆（§定稿/§Q1）：loss = MSE + λ·(1−批内 Pearson)，λ 经 --hparams 切
            # 0.0 / 0.3，不进 search_space。λ=0 即纯 MSE（消融下界、保护原基线口径）。
            "lambda_corr": 0.0,
            # batch_size 调大到 1024：窗口 gather 已移到 GPU、模型又小，大 batch 才能把
            # GPU 算力喂饱、同时把每 epoch 的 Python 循环步数压下来（4060 8G 完全容得下
            # [1024, L, 433] 的中间激活）。想更省显存可经 hparams 显式下调。
            "batch_size": 1024,
            # 评估/预测用更大 batch（无反传），减少循环步数；4096 兼顾预取 pinned 内存占用。
            "eval_batch_size": 4096,
            "max_epochs": 20,
            "patience": 4,
            "grad_clip": 1.0,
            "device": "auto",
        }
        merged.update(dict(hparams or {}))
        return merged

    def _eval_loss(self, src, loss_fn, eval_batch_size: int) -> float:
        """在 GPU 驻留窗口源上算平均 loss；loss 全程在 device 上累加，只在最后同步一次。"""
        torch = self._torch
        if len(src) == 0:
            return float("inf")
        running = torch.zeros((), device=self._device)
        count = 0
        self._model.eval()
        with torch.no_grad():
            for xb, yb in src.iter_batches(batch_size=eval_batch_size, shuffle=False):
                pred = self._model(xb)
                running = running + loss_fn(pred, yb).detach() * yb.shape[0]
                count += int(yb.shape[0])
        return float(running.item()) / max(count, 1)

    def fit(self, train_ds, earlystop_ds, hparams: Dict, seed: int) -> TrainRecord:
        torch, nn, _ = _require_torch()
        cfg = self._default_hparams(hparams)
        self._torch = torch
        self._device = _resolve_torch_device(torch, cfg.get("device", "auto"))

        t0 = time.time()
        if len(train_ds) == 0:
            self._model = None
            self._fitted = True
            self._fallback_bias = 0.0
            return TrainRecord(
                train_curve=[0.0],
                earlystop_curve=[0.0] if len(earlystop_ds) else [],
                best_epoch=0, n_epochs=1,
                fit_seconds=time.time() - t0,
                extra={"kind": "gru", "empty": True, "device": self._device},
            )

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

        self._model = _build_gru_module(
            input_channels=train_ds.n_channels,
            hidden_size=int(cfg["hidden_size"]),
            num_layers=int(cfg["num_layers"]),
            dropout=float(cfg["dropout"]),
        ).to(self._device)

        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=float(cfg["lr"]),
            weight_decay=float(cfg["weight_decay"]),
        )
        # 组合损失 = MSE + λ·(1−批内 Pearson)。GRU 逐票批：整批当一个截面算 1D Pearson
        # （§3 粗代理）；λ=0 时工厂内部短路 corr、等价纯 MSE。loss_fn(pred, yb) 接口
        # 与原 nn.MSELoss 调用点一致（mask 默认 None），_eval_loss 同样复用。
        lambda_corr = float(cfg.get("lambda_corr", 0.0))
        loss_fn = make_loss(lambda_corr=lambda_corr)
        batch_size = int(cfg["batch_size"])
        eval_batch_size = int(cfg.get("eval_batch_size", 8192))
        max_epochs = int(cfg["max_epochs"])
        patience = int(cfg["patience"])
        grad_clip = float(cfg["grad_clip"])

        best_metric = float("inf")
        best_epoch = 0
        best_state = None
        no_improve = 0
        train_curve: List[float] = []
        early_curve: List[float] = []

        # 兜底偏置：只取窗口标签均值，不物化整窗 X[B,L,C]（window_labels 已是纯 numpy）。
        train_labels = train_ds.window_labels()
        self._fallback_bias = float(np.mean(train_labels)) if len(train_labels) else 0.0

        # 训练/早停数据一次性驻留 GPU：之后每 epoch 的窗口 gather 全在卡上，CPU 不再拼数据。
        train_src = _GpuWindowSource(train_ds, torch, self._device)
        es_src = _GpuWindowSource(earlystop_ds, torch, self._device)
        try:
            for epoch in range(1, max_epochs + 1):
                self._model.train()
                running = torch.zeros((), device=self._device)   # loss 在 device 上累加
                count = 0
                for xb, yb in train_src.iter_batches(batch_size=batch_size, shuffle=True, seed=int(seed) + epoch):
                    optimizer.zero_grad(set_to_none=True)
                    pred = self._model(xb)
                    loss = loss_fn(pred, yb)
                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=grad_clip)
                    optimizer.step()
                    running = running + loss.detach() * yb.shape[0]
                    count += int(yb.shape[0])

                # 每 epoch 只在此处同步一次（取代过去逐 batch .item() 的 GPU↔CPU 阻塞）。
                train_loss = float(running.item()) / max(count, 1)
                train_curve.append(train_loss)

                if len(es_src) > 0:
                    metric = self._eval_loss(es_src, loss_fn, eval_batch_size)
                    early_curve.append(metric)
                else:
                    metric = train_loss

                if metric + 1e-8 < best_metric:
                    best_metric = metric
                    best_epoch = epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in self._model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        break
        finally:
            # 训练窗口源用完即释放 device 显存；predict 时再按需重新驻留。
            train_src.free()
            es_src.free()

        if best_state is not None:
            self._model.load_state_dict(best_state)

        self._fitted = True
        # 用 best 权重在**训练区**现算原始预测，OLS 标定全局 rescale a·ŷ+b（§定稿 第 2 条）。
        # fit-on-train / apply-on-val：a,b 只用训练段拟合，predict 时套到 val/test，零泄漏。
        raw_train = self._predict_raw(train_ds)
        y_train = train_ds.window_labels()
        self._rescale_a, self._rescale_b = fit_linear_rescale_numpy(raw_train, y_train)

        return TrainRecord(
            train_curve=train_curve,
            earlystop_curve=early_curve,
            best_epoch=best_epoch,
            n_epochs=len(train_curve),
            fit_seconds=time.time() - t0,
            extra={
                "kind": "gru",
                "device": self._device,
                "hidden_size": int(cfg["hidden_size"]),
                "num_layers": int(cfg["num_layers"]),
                "lambda_corr": lambda_corr,
                "rescale_a": float(self._rescale_a),
                "rescale_b": float(self._rescale_b),
            },
        )

    def _predict_raw(self, ds) -> np.ndarray:
        """模型原始输出（**未套 rescale**），与 ds 窗口顺序严格对齐。fit 内标定 OLS 用它。"""
        if len(ds) == 0:
            return np.zeros(0, dtype=np.float32)
        if not self._fitted or self._model is None:
            return np.full(len(ds), self._fallback_bias, dtype=np.float32)

        torch = self._torch
        # 预测同样把窗口源驻留 device（OOM 自动降级流式）：窗口 gather 在卡上做、
        # 零逐 batch CPU 拼接 + 零逐 batch H2D，与 fit/eval 同一条快路径。
        src = _GpuWindowSource(ds, torch, self._device)
        preds: List[np.ndarray] = []
        self._model.eval()
        try:
            with torch.no_grad():
                # shuffle=False：保持窗口原序，预测结果与 ds 行严格对齐。
                for xb, _ in src.iter_batches(batch_size=4096, shuffle=False):
                    pred = self._model(xb).detach().cpu().numpy().astype(np.float32)
                    preds.append(pred)
        finally:
            src.free()
        return np.concatenate(preds, axis=0) if preds else np.zeros(0, dtype=np.float32)

    def predict(self, ds) -> np.ndarray:
        """对外预测：原始输出套全局线性 rescale（§定稿 第 2 条，永远开启）→ R²=corr²≥0。"""
        raw = self._predict_raw(ds)
        if raw.size == 0:
            return raw
        return (self._rescale_a * raw + self._rescale_b).astype(np.float32)


def _safe_n_heads(hidden_size: int, requested: int) -> int:
    """注意力头数必须整除 embed_dim：取 ≤requested 且整除 hidden_size 的最大头数（兜底 1）。"""
    h = max(1, int(requested))
    while h > 1 and (int(hidden_size) % h != 0):
        h -= 1
    return h


def _build_xsection_module(
    input_channels: int, hidden_size: int, num_layers: int,
    dropout: float, n_heads: int, attn_dropout: float,
    cross_z: bool = False,
):
    """
    截面模型：因子化两腿 + 零初门控残差（规格 §8.2）。

        每票日内窗 [L,C] ──共享时序腿(GRU，所有票/interval 同一套权重)──▶ h_i ∈ R^d
        {h_i}(同一 (date,interval) 在场票) ──截面腿(MultiheadAttention 1 块，无位置/无 ID，
                                              带 padding mask)──▶ Δ_i
        z_i = h_i + γ·Δ_i  (γ 初始化 0：截面腿一开始零贡献 → 退化回纯时序 GRU 基线)
        ŷ_i = Linear(z_i)  (per-票头，所有票共享)

    - **置换等变 + 零身份**：时序腿逐票独立、截面腿无位置编码无股票 ID、头逐票共享 →
      打乱在场票顺序，输出随之同序置换、值不变（换池子免费保险，§11.5）。
    - **零初残差**：γ 初值 0，学不到截面增益时 Δ 贡献被门到 0、等价 GRU-on-433 那条 ③，
      天然是截面模型的消融下界。
    """
    torch, nn, F = _require_torch()

    class XSectionModel(nn.Module):
        def __init__(self):
            super().__init__()
            # 截面内输入归一（cross-z）开关：True 时 forward 第一层把每个 (date,interval)
            # 快照内、各 (lag,channel) 沿在场票做 z-score——正是传统 0.0776 主力那一维
            # cross-z/rank 的显式输入端供给（规格 §8.2.1 落地修订的「日后专门实验」）。
            # rescale 永远开（§定稿 第 2 条）→ cross-z 抹绝对量纲也不威胁 R²≥0。
            self.cross_z = bool(cross_z)
            self._cz_eps = 1e-5
            gru_drop = float(dropout) if int(num_layers) > 1 else 0.0
            # 共享时序腿：与 GRU 卡带同构，取末步隐藏态作每票日内路径编码。
            self.gru = nn.GRU(
                input_size=int(input_channels),
                hidden_size=int(hidden_size),
                num_layers=int(num_layers),
                batch_first=True,
                dropout=gru_drop,
            )
            # 截面腿：单块自注意力（query=key=value=h），无位置编码、无股票嵌入 → 置换等变。
            self.attn = nn.MultiheadAttention(
                embed_dim=int(hidden_size),
                num_heads=int(n_heads),
                dropout=float(attn_dropout),
                batch_first=True,
            )
            self.attn_drop = nn.Dropout(float(dropout))
            # 零初始化门控标量：init 0 → 初始 z=h（纯时序腿）；可学着放大截面贡献。
            self.gamma = nn.Parameter(torch.zeros(1))
            self.head = nn.Linear(int(hidden_size), 1)

        def forward(self, x, mask):
            # x: [B, N, L, C]；mask: [B, N]（True=在场票，False=pad）
            B, N, L, C = x.shape
            # 截面内 cross-z（可选，默认关）：每个快照、每个 (lag,channel) 沿在场票 z-score。
            # 只用 mask 内真实票算统计（pad 不污染），算完把 pad 位重新置 0。置换等变保持
            # （统计是对称聚合、逐票同一变换），故「打乱在场票顺序输出同序置换」不变。
            if self.cross_z:
                m = mask[:, :, None, None].to(x.dtype)            # [B,N,1,1]
                cnt = m.sum(dim=1, keepdim=True).clamp_min(1.0)   # [B,1,1,1] 各快照在场票数
                mean = (x * m).sum(dim=1, keepdim=True) / cnt     # [B,1,L,C]
                var = ((x - mean) ** 2 * m).sum(dim=1, keepdim=True) / cnt
                x = (x - mean) / var.clamp_min(self._cz_eps).sqrt()
                x = x * m                                          # pad 位归零，避免假值进 GRU
            # 时序腿：把 (B,N) 摊平成一批序列，逐票独立编码（不跨票）。
            out, _ = self.gru(x.reshape(B * N, L, C))
            h = out[:, -1, :].reshape(B, N, -1)              # [B, N, d]
            # 截面腿：key_padding_mask 屏蔽 pad 票（True 处被忽略）；每个 batch 元素=一个
            # 截面，注意力只在该截面在场票内做（每元素恒 ≥1 真实票，softmax 不会全 -inf）。
            key_padding = ~mask                              # [B, N]
            delta, _ = self.attn(h, h, h, key_padding_mask=key_padding, need_weights=False)
            z = h + self.gamma * self.attn_drop(delta)       # 零初残差：γ=0 时 z=h 逐位精确
            return self.head(z).squeeze(-1)                  # [B, N]

    return XSectionModel()


@register_model(ModelKind.XSECTION)
class CrossSectionCartridge(ModelCartridge):
    """
    截面模型卡带（主攻，规格 §8.2）。吃 trainer 现成的 ``SequenceDataset``，内部按
    ``(date,interval)`` 重组成截面快照（``CrossSectionDataset.from_whitened`` 零再白化、
    零复制），故**脊柱/trainer 零改动**。

    与评分天然联姻（§3）：一次预测整快照 → 损失里的 corr 项 = 真·截面内 Pearson =
    老师 pooled corr / 每日截面 IC 那一维被打分的量（逐票 GRU 只能拿批内粗代理）。

    口径：``loss = MSE + λ·(1−截面内 Pearson)``（masked 逐快照），λ 经 hparams（§Q1）；
    预测永远套训练段 OLS rescale（§定稿 第 2 条）保 R²=corr²≥0；重正则（小 GRU、少头、
    dropout/weight_decay/早停，§11.5）。required_adapter 固定 FEATURE_433。
    """

    search_space: ClassVar[Dict] = XSECTION_SEARCH_SPACE
    required_adapter = AdapterKind.FEATURE_433

    def __init__(self):
        self._torch = None
        self._model = None
        self._device = "cpu"
        self._fallback_bias = 0.0
        self._fitted = False
        self._rescale_a = 1.0
        self._rescale_b = 0.0
        self._snap_batch = 4

    @classmethod
    def from_config(cls, model_config) -> "CrossSectionCartridge":
        return cls()

    @staticmethod
    def _default_hparams(hparams: Dict) -> Dict:
        merged = {
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.20,            # 重正则（§11.5）
            "attn_dropout": 0.10,
            "n_heads": 4,
            "lr": 1e-3,
            "weight_decay": 1e-3,       # 比 GRU 更狠：截面腿容量更大、宁可欠拟合换稳
            "snap_batch": 4,            # 一批多少个截面快照（控制 [B,maxN,L,C] 瞬时显存）
            "eval_snap_batch": 8,
            "max_epochs": 30,
            "patience": 5,
            "grad_clip": 1.0,
            "device": "auto",
            # 截面卡带默认带 corr 项（它才是真截面 IC）；λ=0 即纯 MSE 消融、走 --hparams 切。
            "lambda_corr": 0.3,
            # 截面内输入 cross-z 归一（默认关=当前基线语义）；=1 打开做三档消融，走 --hparams 切。
            "cross_z": 0,
        }
        merged.update(dict(hparams or {}))
        return merged

    def _wrap(self, ds):
        """把 trainer 传来的 SequenceDataset（已白化）零复制包成截面快照集。"""
        from sequence_dataset import CrossSectionDataset
        return CrossSectionDataset.from_whitened(ds.feature_matrix(), ds.arrays, ds.seq_len)

    def _eval_loss(self, src, loss_fn, snap_batch: int) -> float:
        """在截面源上算平均组合损失（device 上累加，按有效票数加权），早停用。"""
        torch = self._torch
        if len(src) == 0:
            return float("inf")
        running = torch.zeros((), device=self._device)
        count = 0
        self._model.eval()
        with torch.no_grad():
            for xb, yb, mb in src.iter_batches(snap_batch, shuffle=False):
                pred = self._model(xb, mb)
                nvalid = int(mb.sum().item())
                running = running + loss_fn(pred, yb, mask=mb).detach() * nvalid
                count += nvalid
        return float(running.item()) / max(count, 1)

    def fit(self, train_ds, earlystop_ds, hparams: Dict, seed: int) -> TrainRecord:
        torch, nn, _ = _require_torch()
        cfg = self._default_hparams(hparams)
        self._torch = torch
        self._device = _resolve_torch_device(torch, cfg.get("device", "auto"))
        self._snap_batch = int(cfg["snap_batch"])
        t0 = time.time()

        train_xs = self._wrap(train_ds)
        es_xs = self._wrap(earlystop_ds)
        # 兜底偏置：全训练窗标签均值（与逐票口径一致，不物化整窗）。
        train_labels = train_ds.window_labels()
        self._fallback_bias = float(np.mean(train_labels)) if len(train_labels) else 0.0

        if len(train_xs) == 0:
            self._model = None
            self._fitted = True
            return TrainRecord(
                train_curve=[0.0],
                earlystop_curve=[0.0] if len(es_xs) else [],
                best_epoch=0, n_epochs=1, fit_seconds=time.time() - t0,
                extra={"kind": "xsection", "empty": True, "device": self._device},
            )

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

        n_heads = _safe_n_heads(int(cfg["hidden_size"]), int(cfg["n_heads"]))
        cross_z = bool(int(cfg.get("cross_z", 0)))
        self._model = _build_xsection_module(
            input_channels=train_xs.n_channels,
            hidden_size=int(cfg["hidden_size"]),
            num_layers=int(cfg["num_layers"]),
            dropout=float(cfg["dropout"]),
            n_heads=n_heads,
            attn_dropout=float(cfg["attn_dropout"]),
            cross_z=cross_z,
        ).to(self._device)

        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=float(cfg["lr"]),
            weight_decay=float(cfg["weight_decay"]),
        )
        lambda_corr = float(cfg.get("lambda_corr", 0.3))
        loss_fn = make_loss(lambda_corr=lambda_corr)   # masked 截面口径：传 mask 即走真截面 Pearson
        snap_batch = int(cfg["snap_batch"])
        eval_snap_batch = int(cfg.get("eval_snap_batch", snap_batch))
        max_epochs = int(cfg["max_epochs"])
        patience = int(cfg["patience"])
        grad_clip = float(cfg["grad_clip"])

        best_metric = float("inf")
        best_epoch = 0
        best_state = None
        no_improve = 0
        train_curve: List[float] = []
        early_curve: List[float] = []

        train_src = _GpuCrossSectionSource(train_xs, torch, self._device)
        es_src = _GpuCrossSectionSource(es_xs, torch, self._device)
        try:
            for epoch in range(1, max_epochs + 1):
                self._model.train()
                running = torch.zeros((), device=self._device)
                count = 0
                for xb, yb, mb in train_src.iter_batches(snap_batch, shuffle=True, seed=int(seed) + epoch):
                    optimizer.zero_grad(set_to_none=True)
                    pred = self._model(xb, mb)
                    loss = loss_fn(pred, yb, mask=mb)
                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=grad_clip)
                    optimizer.step()
                    nvalid = int(mb.sum().item())
                    running = running + loss.detach() * nvalid
                    count += nvalid

                train_loss = float(running.item()) / max(count, 1)
                train_curve.append(train_loss)

                if len(es_src) > 0:
                    metric = self._eval_loss(es_src, loss_fn, eval_snap_batch)
                    early_curve.append(metric)
                else:
                    metric = train_loss

                if metric + 1e-8 < best_metric:
                    best_metric = metric
                    best_epoch = epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in self._model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        break
        finally:
            train_src.free()
            es_src.free()

        if best_state is not None:
            self._model.load_state_dict(best_state)

        self._fitted = True
        # 训练段 OLS 标定全局 rescale（§定稿 第 2 条；fit-on-train / apply-on-val，零泄漏）。
        raw_train = self._predict_raw(train_ds)
        y_train = train_ds.window_labels()
        self._rescale_a, self._rescale_b = fit_linear_rescale_numpy(raw_train, y_train)

        return TrainRecord(
            train_curve=train_curve,
            earlystop_curve=early_curve,
            best_epoch=best_epoch,
            n_epochs=len(train_curve),
            fit_seconds=time.time() - t0,
            extra={
                "kind": "xsection",
                "device": self._device,
                "hidden_size": int(cfg["hidden_size"]),
                "num_layers": int(cfg["num_layers"]),
                "n_heads": int(n_heads),
                "lambda_corr": lambda_corr,
                "cross_z": cross_z,
                "rescale_a": float(self._rescale_a),
                "rescale_b": float(self._rescale_b),
            },
        )

    def _predict_raw(self, ds) -> np.ndarray:
        """
        模型原始逐票输出（未套 rescale），**对齐到 ds（SequenceDataset）的 label_rows 升序**。

        内部按截面快照前向，逐快照收集在场票预测（snapshot-flatten 序），再用
        ``argsort(flat_label_rows)`` 映射回升序——因截面快照末行集合与 ds.label_rows
        完全相同，升序后即 ds.label_frame() 行序，指标对齐无损。
        """
        if len(ds) == 0:
            return np.zeros(0, dtype=np.float32)
        if not self._fitted or self._model is None:
            return np.full(len(ds), self._fallback_bias, dtype=np.float32)

        torch = self._torch
        xs = self._wrap(ds)
        flat_lr = xs.flat_label_rows()
        if flat_lr.size == 0:
            return np.zeros(0, dtype=np.float32)
        src = _GpuCrossSectionSource(xs, torch, self._device)
        chunks: List[np.ndarray] = []
        self._model.eval()
        try:
            with torch.no_grad():
                # shuffle=False：快照按 (date,interval) 升序，逐行取有效位 → 与 flat_lr 同序。
                for xb, _yb, mb in src.iter_batches(self._snap_batch, shuffle=False):
                    out = self._model(xb, mb).detach().cpu().numpy().astype(np.float32)
                    m = mb.detach().cpu().numpy()
                    for bi in range(out.shape[0]):
                        chunks.append(out[bi][m[bi]])
        finally:
            src.free()
        flat_pred = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        # 升序映射：argsort(flat_lr) 把 snapshot-flatten 序排成升序行号序（= ds.label_rows）。
        order = np.argsort(flat_lr, kind="mergesort")
        return flat_pred[order].astype(np.float32)

    def predict(self, ds) -> np.ndarray:
        """对外预测：原始输出套全局线性 rescale（§定稿 第 2 条，永远开启）→ R²=corr²≥0。"""
        raw = self._predict_raw(ds)
        if raw.size == 0:
            return raw
        return (self._rescale_a * raw + self._rescale_b).astype(np.float32)


# ================================================================== #
# 截面架构 + raw 通道（2026-06-03：补上"好架构 + 好输入"那一格）
# ================================================================== #
@register_model(ModelKind.XSECTION_RAW)
class CrossSectionRawCartridge(CrossSectionCartridge):
    """
    截面模型架构 + RawChannelAdapter 原始微结构通道输入。

    动机（2026-06-03 诊断）：前三轮从没把"好架构"和"好输入"合在一起——
    - TCN-on-raw：吃 raw 59 通道（好输入），但 [B,L,C] 一次一票、无截面（坏架构）→ 0.009；
    - XSECTION / GRU：带截面（好架构），但只喂 433 手工摘要（被压缩的输入）→ 0.062。

    本卡带 = ``CrossSectionCartridge`` 的完整架构（共享 GRU 时序腿 + set-attention 截面腿 +
    零初门控残差 + per-票头 + masked 截面 Pearson 损失 + 训练段 OLS rescale）**原样复用**，
    唯一区别 = 输入绑定从 ``FEATURE_433`` 换成 ``RAW_CHANNELS``——即喂 59 个原始微结构通道
    （含挂撤单、深档盘口、成交明细，规格 §8.0 的 ``RawChannelAdapter``），让网络自学通道交互，
    而不是吃 433 个人工摘要。

    实现零架构改动：父类 ``fit`` 里 module 的输入维度是 ``input_channels=train_xs.n_channels``
    （数据驱动），喂 RAW_CHANNELS 时自动按 59 通道建网；故这里**只需覆盖 required_adapter**
    （组装期强制喂 raw_channels、喂错即拦）。search_space / 超参 / fit / predict 全继承父类。
    """

    required_adapter = AdapterKind.RAW_CHANNELS
