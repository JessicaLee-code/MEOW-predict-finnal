"""
特征注册表

本模块是 PE1 重构的第一块基石，职责有三类：

1. 统一注册每个特征 stage 的 builder、依赖、状态与 group 归属。
2. 提供 DAG 相关查询能力，供后续 FeatureStore / FeatureLoader 使用。
3. 在尚未构建 manifest 的阶段，也能通过 builder 推断列名，保证 M1 就可测试。

设计原则：
- builder 函数是真相源，后续磁盘缓存只是开发加速层。
- group 解析保持与旧版 FeatureBuilder.select_groups 兼容。
- 只暴露 stage 级抽象，不引入 DSL / recipe / 额外配置文件。
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Set

import numpy as np
import pandas as pd


EPS = 1e-8
META_COLS = ["date", "symbol", "interval"]
TARGET_COL = "fret12"
STAGE_STATUSES = {"promoted", "candidate", "archived"}


def _unique_preserve_order(columns: Iterable[str]) -> List[str]:
    """按首次出现顺序去重，避免 group 合并后列重复。"""
    seen: Set[str] = set()
    ordered: List[str] = []
    for col in columns:
        if col not in seen:
            seen.add(col)
            ordered.append(col)
    return ordered


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    """统一的安全除法，保持与旧版 feat_engine 完全一致。"""
    return a / (b.abs() + EPS)


def _sanitize_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    对 stage 输出做统一清洗。

    这里刻意只处理特征列，不碰 meta / target。
    这样后续写 parquet 时，所有 stage 都满足：
    - 只含特征列
    - 数值型列统一为 float32
    - inf / nan 已转成 0.0
    """
    out = df.copy()
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).astype(np.float32)
    return out


def _sort_raw_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """所有 builder 共享的原始排序口径，保证行序稳定。"""
    return raw.sort_values(META_COLS, kind="mergesort").reset_index(drop=True).copy()


def _concat_inputs(raw: pd.DataFrame, *frames: pd.DataFrame) -> pd.DataFrame:
    """
    按列横向拼接 raw 与上游 stage 输出。

    约束：
    - 调用方必须保证所有输入行序一致。
    - 重名列按首次出现保留，避免上游传入冗余列时覆盖 raw。
    """
    merged = pd.concat([raw, *frames], axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated()]
    return merged


def _lag_columns() -> List[str]:
    """旧版 lag group 的精确列名。"""
    cols: List[str] = []
    for lag in [1, 3, 5, 10, 20, 30]:
        cols.extend(
            [
                f"mid_ret_lag_{lag}",
                f"obi0_lag_{lag}",
                f"trade_imb_lag_{lag}",
                f"spread_lag_{lag}",
            ]
        )
    return cols


def _roll_columns() -> List[str]:
    """旧版 roll stage 的精确列名。"""
    cols: List[str] = []
    for window in [3, 5, 10, 20, 30]:
        for col in ["mid_ret1_raw", "obi0", "trade_imb", "order_pressure", "spread"]:
            cols.extend([f"{col}_rm{window}", f"{col}_rs{window}"])
    return cols


def _patch_columns() -> List[str]:
    """旧版 patch stage 的精确列名。"""
    cols: List[str] = []
    for col in ["mid_ret1_raw", "obi0", "trade_imb", "spread", "order_pressure"]:
        for window in [6, 12, 24, 60]:
            cols.extend(
                [
                    f"{col}_patch{window}_mean",
                    f"{col}_patch{window}_std",
                    f"{col}_patch{window}_max",
                    f"{col}_patch{window}_min",
                    f"{col}_patch{window}_range",
                    f"{col}_patch{window}_slope",
                    f"{col}_patch{window}_last",
                ]
            )
    return cols


def _ofi_stage_columns() -> List[str]:
    """旧版 ofi stage 的全部列名，用于精确 group 映射。"""
    cols: List[str] = []
    for suffix in ["0", "4", "9", "19"]:
        cols.extend(
            [
                f"bid_ofi_{suffix}",
                f"ask_ofi_{suffix}",
                f"ofi_{suffix}",
                f"ofi_{suffix}_depth",
                f"ofi_{suffix}_div_depth",
                f"ofi_{suffix}_div_turnover",
            ]
        )
    cols.extend(
        [
            "ofi_total",
            "ofi_total_depth",
            "ofi_abs",
            "ofi_sign",
            "ofi_div_total_depth",
            "ofi_div_turnover",
        ]
    )
    for window in [3, 6, 12, 24]:
        cols.extend(
            [
                f"ofi_total_ema{window}",
                f"ofi_total_sum{window}",
                f"ofi_total_mean{window}",
                f"ofi_total_z{window}",
            ]
        )
    return cols


def _trade_impact_stage_columns() -> List[str]:
    """旧版 trade_impact stage 的全部列名。"""
    cols = [
        "signed_trade_qty",
        "signed_trade_turnover",
        "trade_pressure_qty",
        "trade_pressure_turnover",
        "trade_intensity",
        "avg_trade_size",
        "avg_trade_turnover",
        "trade_pressure_x_spread",
        "trade_pressure_x_order_pressure",
        "trade_pressure_x_ofi",
    ]
    for window in [3, 6, 12, 24]:
        for col in [
            "trade_pressure_qty",
            "trade_pressure_turnover",
            "trade_intensity",
            "avg_trade_size",
            "avg_trade_turnover",
        ]:
            cols.extend(
                [
                    f"{col}_ema{window}",
                    f"{col}_sum{window}",
                    f"{col}_mean{window}",
                    f"{col}_z{window}",
                ]
            )
    return cols


def _conditional_momentum_columns() -> List[str]:
    """旧版 conditional_momentum stage 的全部列名。"""
    cols: List[str] = []
    for window in [1, 3, 6, 12, 24]:
        cols.extend(
            [
                f"lagret{window}_raw",
                f"lagret{window}",
                f"lagret{window}_abs",
                f"lagret{window}_sign",
                f"lagret{window}_x_trade_pressure",
                f"lagret{window}_x_ofi",
                f"lagret{window}_x_spread",
                f"lagret{window}_x_vol",
            ]
        )
    cols.extend(["momentum_state", "reversal_state", "conditional_momentum_rank"])
    return cols


def _cross_source_columns() -> List[str]:
    """cross stage 做横截面标准化时使用的基底列。"""
    return [
        "midpx",
        "spread",
        "obi0",
        "obi4",
        "trade_imb",
        "order_pressure",
        "trade_activity",
        "ofi_total",
        "ofi_0",
        "ofi_4",
        "ofi_9",
        "ofi_19",
        "trade_pressure_qty",
        "trade_pressure_turnover",
        "trade_intensity",
        "avg_trade_size",
        "avg_trade_turnover",
    ]


def _cross_z_columns() -> List[str]:
    """cross_z / norm_core 在 cross stage 上对应的精确列名。"""
    return [f"{col}_cs_z" for col in _cross_source_columns()]


def _cross_rank_columns() -> List[str]:
    """cross_rank / 若干跨 stage group 在 cross stage 上对应的精确列名。"""
    return [f"{col}_cs_rank" for col in _cross_source_columns()]


def _regime_columns() -> List[str]:
    """旧版 regime stage 的精确列名。"""
    return [
        "regime_score",
        "regime_low",
        "regime_mid",
        "regime_high",
        "state_momentum",
        "state_reversal",
        "state_pressure",
        "state_liquidity",
        "state_vol_cs",
        "state_spread_cs",
        "state_activity_cs",
    ]


BASE_CORE_COLS = [
    "spread",
    "mid_ret1_raw",
    "obi0",
    "obi4",
    "obi9",
    "trade_imb",
    "trade_turnover_imb",
    "add_imb",
    "cxl_imb",
    "qty_add_imb",
    "qty_cxl_imb",
    "buy_vwad_gap",
    "sell_vwad_gap",
    "trade_activity",
    "order_pressure",
]

LEGACY_COLS = [
    "ob_imb0",
    "ob_imb4",
    "ob_imb9",
    "trade_imb",
    "trade_imbema5",
    "lagret12",
]


RAW_SCHEMA_COLUMNS = [
    "date",
    "symbol",
    "interval",
    "fret12",
    "midpx",
    "bid0",
    "ask0",
    "bid4",
    "ask4",
    "bid9",
    "ask9",
    "bid19",
    "ask19",
    "bsize0",
    "asize0",
    "bsize0_4",
    "asize0_4",
    "bsize5_9",
    "asize5_9",
    "bsize10_19",
    "asize10_19",
    "nTradeBuy",
    "tradeBuyQty",
    "tradeBuyTurnover",
    "nTradeSell",
    "tradeSellQty",
    "tradeSellTurnover",
    "nAddBuy",
    "addBuyQty",
    "nAddSell",
    "addSellQty",
    "nCxlBuy",
    "cxlBuyQty",
    "nCxlSell",
    "cxlSellQty",
    "buyVwad",
    "sellVwad",
]


def _make_schema_probe_raw() -> pd.DataFrame:
    """
    生成一个极小但足够覆盖全部 builder 路径的假原始数据。

    用途：
    - 在 manifest 尚不存在时，推断各 stage 的输出列名。
    - 单元测试中验证 registry 与旧管道的列口径是否一致。

    结构选择：
    - 2 个交易日 × 3 个股票 × 24 个 interval
    - 同一 `(date, interval)` 下有多个 symbol，确保横截面特征可计算
    - 同一 `(date, symbol)` 下有足够多 interval，确保 lag / rolling / patch 都能出列
    """
    rows: List[dict] = []
    dates = [20230601, 20230602]
    symbols = ["AAA", "BBB", "CCC"]
    intervals = [93000000 + i * 10000 for i in range(24)]
    for date_idx, date in enumerate(dates):
        for symbol_idx, symbol in enumerate(symbols):
            for interval_idx, interval in enumerate(intervals):
                base = 100.0 + date_idx * 1.5 + symbol_idx * 0.8 + interval_idx * 0.02
                qty_base = 50.0 + symbol_idx * 3.0 + interval_idx
                row = {
                    "date": date,
                    "symbol": symbol,
                    "interval": interval,
                    "fret12": np.float32((symbol_idx - 1) * 0.001 + interval_idx * 0.0001),
                    "midpx": np.float32(base + 0.01 * ((interval_idx % 5) - 2)),
                    "bid0": np.float32(base - 0.01),
                    "ask0": np.float32(base + 0.01),
                    "bid4": np.float32(base - 0.015),
                    "ask4": np.float32(base + 0.015),
                    "bid9": np.float32(base - 0.02),
                    "ask9": np.float32(base + 0.02),
                    "bid19": np.float32(base - 0.03),
                    "ask19": np.float32(base + 0.03),
                    "bsize0": np.float32(qty_base + 5.0),
                    "asize0": np.float32(qty_base + 7.0),
                    "bsize0_4": np.float32(qty_base + 15.0),
                    "asize0_4": np.float32(qty_base + 18.0),
                    "bsize5_9": np.float32(qty_base + 22.0),
                    "asize5_9": np.float32(qty_base + 24.0),
                    "bsize10_19": np.float32(qty_base + 30.0),
                    "asize10_19": np.float32(qty_base + 33.0),
                    "nTradeBuy": np.float32(2 + (interval_idx % 4) + symbol_idx),
                    "tradeBuyQty": np.float32(qty_base + 10.0),
                    "tradeBuyTurnover": np.float32((qty_base + 10.0) * (base + 0.02)),
                    "nTradeSell": np.float32(1 + ((interval_idx + symbol_idx) % 4)),
                    "tradeSellQty": np.float32(qty_base + 8.0),
                    "tradeSellTurnover": np.float32((qty_base + 8.0) * (base - 0.02)),
                    "nAddBuy": np.float32(3 + (interval_idx % 3)),
                    "addBuyQty": np.float32(qty_base + 4.0),
                    "nAddSell": np.float32(2 + ((interval_idx + 1) % 3)),
                    "addSellQty": np.float32(qty_base + 3.0),
                    "nCxlBuy": np.float32(1 + (interval_idx % 2)),
                    "cxlBuyQty": np.float32(qty_base + 2.0),
                    "nCxlSell": np.float32(1 + ((interval_idx + symbol_idx) % 2)),
                    "cxlSellQty": np.float32(qty_base + 1.0),
                    "buyVwad": np.float32(base + 0.03),
                    "sellVwad": np.float32(base - 0.03),
                }
                rows.append(row)
    return pd.DataFrame(rows, columns=RAW_SCHEMA_COLUMNS)


@dataclass
class StageDefinition:
    """单个特征 stage 的注册元信息。"""
    name: str
    deps: List[str]
    groups: List[str]
    group_columns: Dict[str, List[str]]
    status: str
    builder: Callable
    registration_order: int


class FeatureRegistry:
    """
    特征注册表。

    manifest_path:
    - 后续由 FeatureStore 写入 `data/features/manifest.json`
    - M1 阶段可以为空，此时 resolve_groups 会退化为“builder 推断列名”
    """

    def __init__(self, manifest_path: Optional[str] = None):
        self.manifest_path = Path(manifest_path) if manifest_path else None
        self._stages: Dict[str, StageDefinition] = {}
        self._registration_order: List[str] = []
        self._schema_stage_columns_cache: Optional[Dict[str, List[str]]] = None

    def set_manifest_path(self, manifest_path: Optional[str]) -> None:
        """允许后续 FeatureStore/Loader 在运行时绑定 manifest 位置。"""
        self.manifest_path = Path(manifest_path) if manifest_path else None

    def stage(
        self,
        name: str,
        deps: List[str],
        groups: List[str],
        group_columns: Optional[Dict[str, List[str]]] = None,
        status: str = "promoted",
    ) -> Callable:
        """
        stage 装饰器。

        这里显式保存 group_columns，而不是把列规则散落在别处，
        目的是保证“新增一个特征组只改一处”的设计宪法。
        """
        if status not in STAGE_STATUSES:
            raise ValueError(f"Unsupported stage status: {status}")
        group_columns = group_columns or {}

        def decorator(func: Callable) -> Callable:
            if name in self._stages:
                raise ValueError(f"Duplicated stage name: {name}")
            definition = StageDefinition(
                name=name,
                deps=list(deps),
                groups=list(groups),
                group_columns={k: list(v) for k, v in group_columns.items()},
                status=status,
                builder=func,
                registration_order=len(self._registration_order),
            )
            self._stages[name] = definition
            self._registration_order.append(name)
            # stage 注册变化后，列推断缓存必须失效。
            self._schema_stage_columns_cache = None
            return func

        return decorator

    def topo_order(self, include_archived: bool = False) -> List[str]:
        """返回稳定的拓扑序，默认排除 archived stage。"""
        active_names = [
            name
            for name in self._registration_order
            if include_archived or self._stages[name].status != "archived"
        ]
        indegree: Dict[str, int] = {name: 0 for name in active_names}
        graph: Dict[str, List[str]] = {name: [] for name in active_names}
        active_set = set(active_names)
        for name in active_names:
            for dep in self._stages[name].deps:
                if dep not in self._stages:
                    raise ValueError(f"Stage {name} depends on unknown stage {dep}")
                if dep not in active_set:
                    raise ValueError(f"Stage {name} depends on archived stage {dep}")
                graph[dep].append(name)
                indegree[name] += 1

        queue = deque([name for name in active_names if indegree[name] == 0])
        order: List[str] = []
        while queue:
            current = queue.popleft()
            order.append(current)
            for child in graph[current]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if len(order) != len(active_names):
            raise ValueError("Feature stage graph contains a cycle")
        return order

    def downstream(self, stage_name: str) -> Set[str]:
        """返回指定 stage 的所有下游 stage 传递闭包。"""
        if stage_name not in self._stages:
            raise KeyError(stage_name)
        children: Dict[str, List[str]] = defaultdict(list)
        for name, stage in self._stages.items():
            for dep in stage.deps:
                children[dep].append(name)
        out: Set[str] = set()
        queue = deque(children[stage_name])
        while queue:
            current = queue.popleft()
            if current in out:
                continue
            out.add(current)
            queue.extend(children[current])
        return out

    def get_builder(self, stage_name: str) -> Callable:
        """返回 stage builder 函数引用。"""
        return self._stages[stage_name].builder

    def get_deps(self, stage_name: str) -> List[str]:
        """返回 stage 的直接依赖列表。"""
        return list(self._stages[stage_name].deps)

    def get_status(self, stage_name: str) -> str:
        """返回 stage 当前状态。"""
        return self._stages[stage_name].status

    def get_stage_groups(self, stage_name: str) -> List[str]:
        """返回 stage 声明过的 group 名称列表。"""
        return list(self._stages[stage_name].groups)

    def code_hash(self, stage_name: str) -> str:
        """对 builder 源码做 md5，供后续脏检测使用。"""
        source = inspect.getsource(self.get_builder(stage_name))
        return hashlib.md5(source.encode("utf-8")).hexdigest()

    def _load_manifest_columns(self) -> Dict[str, List[str]]:
        """从 manifest 读取各 stage 的真实输出列名；缺失时返回空 dict。"""
        if self.manifest_path is None or not self.manifest_path.exists():
            return {}
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        stages = payload.get("stages", {})
        return {
            stage_name: list(stage_info.get("columns", []))
            for stage_name, stage_info in stages.items()
        }

    def _infer_stage_columns(self) -> Dict[str, List[str]]:
        """
        在 manifest 不可用时，通过执行 builder 链推断列名。

        这是 M1 里最关键的兜底逻辑：
        - 让 resolve_groups 在“尚未 build 特征缓存”的阶段就可用
        - 让单元测试可以直接验证新旧 group 口径一致
        """
        if self._schema_stage_columns_cache is not None:
            return self._schema_stage_columns_cache
        raw = _sort_raw_frame(_make_schema_probe_raw())
        outputs: Dict[str, pd.DataFrame] = {}
        for stage_name in self.topo_order(include_archived=False):
            builder = self.get_builder(stage_name)
            deps = {dep: outputs[dep] for dep in self.get_deps(stage_name)}
            outputs[stage_name] = builder(raw, **deps)
        self._schema_stage_columns_cache = {
            stage_name: list(df.columns)
            for stage_name, df in outputs.items()
        }
        return self._schema_stage_columns_cache

    def stage_columns(self, stage_name: str) -> List[str]:
        """
        获取 stage 的完整输出列。

        优先级：
        1. manifest 中的真实列名（M2 build 后正式来源）
        2. builder 推断列名（M1/M2 早期兜底）
        """
        manifest_columns = self._load_manifest_columns()
        if stage_name in manifest_columns and manifest_columns[stage_name]:
            return list(manifest_columns[stage_name])
        inferred = self._infer_stage_columns()
        return list(inferred.get(stage_name, []))

    def resolve_groups(self, group_names: Optional[Iterable[str]]) -> Dict[str, List[str]]:
        """
        将 group 名称解析为 `{stage_name: [columns...]}`。

        兼容旧版规则：
        - group 可以横跨多个 stage（如 norm_core / ofi_rank / trade_impact_dyn）
        - 未声明精确列名的 group，默认取该 stage 的全部输出列
        - 默认排除 archived stage
        """
        if group_names is None:
            return {
                stage_name: self.stage_columns(stage_name)
                for stage_name in self.topo_order(include_archived=False)
            }
        if isinstance(group_names, str):
            group_names = [group_names]
        normalized = [group for group in group_names if group]
        if not normalized or normalized == ["full"]:
            return {
                stage_name: self.stage_columns(stage_name)
                for stage_name in self.topo_order(include_archived=False)
            }

        resolved: Dict[str, List[str]] = defaultdict(list)
        for group in normalized:
            matched = False
            for stage_name in self.topo_order(include_archived=False):
                stage = self._stages[stage_name]
                if group not in stage.groups:
                    continue
                matched = True
                if group in stage.group_columns:
                    columns = list(stage.group_columns[group])
                else:
                    columns = self.stage_columns(stage_name)
                resolved[stage_name].extend(columns)
            if not matched:
                raise KeyError(f"Unknown feature group: {group}")
        return {
            stage_name: _unique_preserve_order(columns)
            for stage_name, columns in resolved.items()
        }


def build_base(raw: pd.DataFrame) -> pd.DataFrame:
    """base stage：迁移自旧版 `_add_base_features`。"""
    df = _sort_raw_frame(raw)
    out = pd.DataFrame(index=df.index)
    out["spread"] = df["ask0"] - df["bid0"]
    out["mid_ret1_raw"] = df.groupby(["date", "symbol"], sort=False)["midpx"].pct_change().fillna(0.0)
    out["obi0"] = _safe_div(df["asize0"] - df["bsize0"], df["asize0"] + df["bsize0"])
    out["obi4"] = _safe_div(df["asize0_4"] - df["bsize0_4"], df["asize0_4"] + df["bsize0_4"])
    out["obi9"] = _safe_div(df["asize5_9"] - df["bsize5_9"], df["asize5_9"] + df["bsize5_9"])
    out["trade_imb"] = _safe_div(df["tradeBuyQty"] - df["tradeSellQty"], df["tradeBuyQty"] + df["tradeSellQty"])
    out["trade_imbema5"] = out.groupby(df["symbol"], sort=False)["trade_imb"].transform(
        lambda s: s.ewm(halflife=5, adjust=False).mean()
    ).fillna(0.0)
    out["trade_turnover_imb"] = _safe_div(
        df["tradeBuyTurnover"] - df["tradeSellTurnover"],
        df["tradeBuyTurnover"] + df["tradeSellTurnover"],
    )
    out["add_imb"] = _safe_div(df["nAddBuy"] - df["nAddSell"], df["nAddBuy"] + df["nAddSell"])
    out["cxl_imb"] = _safe_div(df["nCxlBuy"] - df["nCxlSell"], df["nCxlBuy"] + df["nCxlSell"])
    out["qty_add_imb"] = _safe_div(df["addBuyQty"] - df["addSellQty"], df["addBuyQty"] + df["addSellQty"])
    out["qty_cxl_imb"] = _safe_div(df["cxlBuyQty"] - df["cxlSellQty"], df["cxlBuyQty"] + df["cxlSellQty"])
    out["buy_vwad_gap"] = df["buyVwad"] - df["midpx"]
    out["sell_vwad_gap"] = df["sellVwad"] - df["midpx"]
    out["bret12"] = df.groupby("symbol", sort=False)["midpx"].pct_change(12).fillna(0.0)
    cxbret = pd.DataFrame(
        {
            "interval": df["interval"],
            "bret12": out["bret12"],
        }
    ).groupby("interval", sort=False)[["bret12"]].transform("mean")
    out["lagret12"] = out["bret12"] - cxbret["bret12"]
    out["ob_imb0"] = out["obi0"]
    out["ob_imb4"] = out["obi4"]
    out["ob_imb9"] = out["obi9"]
    out["trade_activity"] = (
        df["nTradeBuy"].fillna(0.0)
        + df["nTradeSell"].fillna(0.0)
        + df["tradeBuyQty"].fillna(0.0)
        + df["tradeSellQty"].fillna(0.0)
    )
    out["order_pressure"] = out["obi0"] + 0.5 * out["obi4"] + 0.5 * out["trade_imb"]
    return _sanitize_feature_frame(out)


def build_lag(raw: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """lag stage：迁移自旧版 `_add_lag_features`。"""
    df = _concat_inputs(_sort_raw_frame(raw), base)
    out = pd.DataFrame(index=df.index)
    group = df.groupby(["date", "symbol"], sort=False)
    for lag in [1, 3, 5, 10, 20, 30]:
        out[f"mid_ret_lag_{lag}"] = group["midpx"].pct_change(lag).fillna(0.0)
        out[f"obi0_lag_{lag}"] = group["obi0"].shift(lag).fillna(0.0)
        out[f"trade_imb_lag_{lag}"] = group["trade_imb"].shift(lag).fillna(0.0)
        out[f"spread_lag_{lag}"] = group["spread"].shift(lag).fillna(0.0)
    return _sanitize_feature_frame(out)


def build_roll(raw: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """roll stage：迁移自旧版 `_add_roll_features`。"""
    df = _concat_inputs(_sort_raw_frame(raw), base)
    out = pd.DataFrame(index=df.index)
    for window in [3, 5, 10, 20, 30]:
        for col in ["mid_ret1_raw", "obi0", "trade_imb", "order_pressure", "spread"]:
            group = df.groupby(["date", "symbol"], sort=False)[col]
            out[f"{col}_rm{window}"] = group.transform(
                lambda s: s.rolling(window=window, min_periods=1).mean()
            ).fillna(0.0)
            out[f"{col}_rs{window}"] = group.transform(
                lambda s: s.rolling(window=window, min_periods=1).std(ddof=0)
            ).fillna(0.0)
    return _sanitize_feature_frame(out)


def build_patch(raw: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """patch stage：迁移自旧版 `_add_patch_summary_features`。"""
    df = _concat_inputs(_sort_raw_frame(raw), base)
    out = pd.DataFrame(index=df.index)
    group = df.groupby(["date", "symbol"], sort=False)
    for col in ["mid_ret1_raw", "obi0", "trade_imb", "spread", "order_pressure"]:
        series = group[col]
        for window in [6, 12, 24, 60]:
            out[f"{col}_patch{window}_mean"] = series.transform(
                lambda s: s.rolling(window=window, min_periods=1).mean()
            ).fillna(0.0)
            out[f"{col}_patch{window}_std"] = series.transform(
                lambda s: s.rolling(window=window, min_periods=1).std(ddof=0)
            ).fillna(0.0)
            out[f"{col}_patch{window}_max"] = series.transform(
                lambda s: s.rolling(window=window, min_periods=1).max()
            ).fillna(0.0)
            out[f"{col}_patch{window}_min"] = series.transform(
                lambda s: s.rolling(window=window, min_periods=1).min()
            ).fillna(0.0)
            out[f"{col}_patch{window}_range"] = (
                out[f"{col}_patch{window}_max"] - out[f"{col}_patch{window}_min"]
            ).fillna(0.0)
            out[f"{col}_patch{window}_slope"] = series.transform(
                lambda s: s.diff().rolling(window=window, min_periods=1).mean()
            ).fillna(0.0)
            out[f"{col}_patch{window}_last"] = df[col].fillna(0.0)
    return _sanitize_feature_frame(out)


def build_ofi(raw: pd.DataFrame) -> pd.DataFrame:
    """ofi stage：迁移自旧版 `_add_ofi_features`，保持“独立于 base”的设计。"""
    df = _sort_raw_frame(raw)
    out = pd.DataFrame(index=df.index)
    group = df.groupby(["date", "symbol"], sort=False)
    level_specs = [
        ("0", "bid0", "ask0", "bsize0", "asize0"),
        ("4", "bid4", "ask4", "bsize0_4", "asize0_4"),
        ("9", "bid9", "ask9", "bsize5_9", "asize5_9"),
        ("19", "bid19", "ask19", "bsize10_19", "asize10_19"),
    ]
    ofi_cols: List[str] = []
    depth_cols: List[str] = []
    turnover = (df["tradeBuyTurnover"].fillna(0.0) + df["tradeSellTurnover"].fillna(0.0)).astype(np.float32)
    for suffix, bid_px, ask_px, bid_sz, ask_sz in level_specs:
        prev_bid_px = group[bid_px].shift(1)
        prev_ask_px = group[ask_px].shift(1)
        prev_bid_sz = group[bid_sz].shift(1)
        prev_ask_sz = group[ask_sz].shift(1)
        cur_bid_px = df[bid_px]
        cur_ask_px = df[ask_px]
        cur_bid_sz = df[bid_sz]
        cur_ask_sz = df[ask_sz]
        bid_ofi = np.where(
            cur_bid_px > prev_bid_px,
            cur_bid_sz,
            np.where(cur_bid_px == prev_bid_px, cur_bid_sz - prev_bid_sz, -prev_bid_sz),
        )
        ask_ofi = np.where(
            cur_ask_px < prev_ask_px,
            -cur_ask_sz,
            np.where(cur_ask_px == prev_ask_px, -(cur_ask_sz - prev_ask_sz), prev_ask_sz),
        )
        bid_ofi = pd.Series(bid_ofi, index=df.index).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
        ask_ofi = pd.Series(ask_ofi, index=df.index).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
        level_ofi = (bid_ofi + ask_ofi).astype(np.float32)
        depth = (cur_bid_sz + cur_ask_sz).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
        out[f"bid_ofi_{suffix}"] = bid_ofi
        out[f"ask_ofi_{suffix}"] = ask_ofi
        out[f"ofi_{suffix}"] = level_ofi
        out[f"ofi_{suffix}_depth"] = depth
        out[f"ofi_{suffix}_div_depth"] = _safe_div(level_ofi, depth)
        out[f"ofi_{suffix}_div_turnover"] = _safe_div(level_ofi, turnover)
        ofi_cols.append(f"ofi_{suffix}")
        depth_cols.append(f"ofi_{suffix}_depth")
    out["ofi_total"] = out[ofi_cols].sum(axis=1)
    out["ofi_total_depth"] = out[depth_cols].sum(axis=1)
    out["ofi_abs"] = out["ofi_total"].abs()
    out["ofi_sign"] = np.sign(out["ofi_total"]).astype(np.float32)
    out["ofi_div_total_depth"] = _safe_div(out["ofi_total"], out["ofi_total_depth"])
    out["ofi_div_turnover"] = _safe_div(out["ofi_total"], turnover)
    ofi_group = out["ofi_total"].groupby(df["symbol"], sort=False)
    for window in [3, 6, 12, 24]:
        out[f"ofi_total_ema{window}"] = ofi_group.transform(
            lambda s: s.ewm(halflife=window, adjust=False).mean()
        ).fillna(0.0)
        out[f"ofi_total_sum{window}"] = ofi_group.transform(
            lambda s: s.rolling(window=window, min_periods=1).sum()
        ).fillna(0.0)
        out[f"ofi_total_mean{window}"] = ofi_group.transform(
            lambda s: s.rolling(window=window, min_periods=1).mean()
        ).fillna(0.0)
        out[f"ofi_total_z{window}"] = ofi_group.transform(
            lambda s: (s - s.rolling(window=window, min_periods=1).mean())
            / (s.rolling(window=window, min_periods=1).std(ddof=0) + EPS)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return _sanitize_feature_frame(out)


def build_trade_impact(raw: pd.DataFrame, base: pd.DataFrame, ofi: pd.DataFrame) -> pd.DataFrame:
    """trade_impact stage：重构后显式依赖 base + ofi。"""
    df = _concat_inputs(_sort_raw_frame(raw), base, ofi)
    out = pd.DataFrame(index=df.index)
    turnover = (df["tradeBuyTurnover"].fillna(0.0) + df["tradeSellTurnover"].fillna(0.0)).astype(np.float32)
    qty = (df["tradeBuyQty"].fillna(0.0) + df["tradeSellQty"].fillna(0.0)).astype(np.float32)
    trades = (df["nTradeBuy"].fillna(0.0) + df["nTradeSell"].fillna(0.0)).astype(np.float32)
    signed_qty = (df["tradeBuyQty"].fillna(0.0) - df["tradeSellQty"].fillna(0.0)).astype(np.float32)
    signed_turnover = (df["tradeBuyTurnover"].fillna(0.0) - df["tradeSellTurnover"].fillna(0.0)).astype(np.float32)
    out["signed_trade_qty"] = signed_qty
    out["signed_trade_turnover"] = signed_turnover
    out["trade_pressure_qty"] = _safe_div(signed_qty, qty)
    out["trade_pressure_turnover"] = _safe_div(signed_turnover, turnover)
    out["trade_intensity"] = trades
    out["avg_trade_size"] = _safe_div(qty, trades)
    out["avg_trade_turnover"] = _safe_div(turnover, trades)
    out["trade_pressure_x_spread"] = out["trade_pressure_qty"] * df["spread"].fillna(0.0)
    out["trade_pressure_x_order_pressure"] = out["trade_pressure_qty"] * df["order_pressure"].fillna(0.0)
    out["trade_pressure_x_ofi"] = out["trade_pressure_qty"] * df["ofi_total"].fillna(0.0)
    for window in [3, 6, 12, 24]:
        for col in [
            "trade_pressure_qty",
            "trade_pressure_turnover",
            "trade_intensity",
            "avg_trade_size",
            "avg_trade_turnover",
        ]:
            col_group = out[col].groupby(df["symbol"], sort=False)
            out[f"{col}_ema{window}"] = col_group.transform(
                lambda s: s.ewm(halflife=window, adjust=False).mean()
            ).fillna(0.0)
            out[f"{col}_sum{window}"] = col_group.transform(
                lambda s: s.rolling(window=window, min_periods=1).sum()
            ).fillna(0.0)
            out[f"{col}_mean{window}"] = col_group.transform(
                lambda s: s.rolling(window=window, min_periods=1).mean()
            ).fillna(0.0)
            out[f"{col}_z{window}"] = col_group.transform(
                lambda s: (s - s.rolling(window=window, min_periods=1).mean())
                / (s.rolling(window=window, min_periods=1).std(ddof=0) + EPS)
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return _sanitize_feature_frame(out)


def build_cross(raw: pd.DataFrame, base: pd.DataFrame, ofi: pd.DataFrame, trade_impact: pd.DataFrame) -> pd.DataFrame:
    """cross stage：迁移自旧版 `_add_cross_section_features`。"""
    df = _concat_inputs(_sort_raw_frame(raw), base, ofi, trade_impact)
    out = pd.DataFrame(index=df.index)
    cross_group = df.groupby(["date", "interval"], sort=False)
    for col in _cross_source_columns():
        if col not in df.columns:
            continue
        mean = cross_group[col].transform("mean")
        std = cross_group[col].transform("std").replace(0.0, np.nan)
        out[f"{col}_cs_z"] = ((df[col] - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out[f"{col}_cs_rank"] = cross_group[col].rank(pct=True, method="average").fillna(0.0)
    out["interval_pos"] = df.groupby("date", sort=False)["interval"].rank(pct=True, method="average").fillna(0.0)
    out["interval_norm"] = df["interval"].astype(np.float32) / 1e8
    out["is_morning"] = (df["interval"] < 113000000).astype(np.float32)
    out["is_afternoon"] = (df["interval"] >= 130000000).astype(np.float32)
    return _sanitize_feature_frame(out)


def build_conditional_momentum(
    raw: pd.DataFrame,
    base: pd.DataFrame,
    ofi: pd.DataFrame,
    trade_impact: pd.DataFrame,
) -> pd.DataFrame:
    """conditional_momentum stage：显式依赖 base + ofi + trade_impact。"""
    df = _concat_inputs(_sort_raw_frame(raw), base, ofi, trade_impact)
    out = pd.DataFrame(index=df.index)
    day_symbol = df.groupby(["date", "symbol"], sort=False)
    for window in [1, 3, 6, 12, 24]:
        raw_ret = day_symbol["midpx"].transform(lambda s: s.pct_change(window)).fillna(0.0)
        cross_mean = raw_ret.groupby(df["interval"], sort=False).transform("mean")
        lagret = raw_ret - cross_mean
        out[f"lagret{window}_raw"] = raw_ret.astype(np.float32)
        out[f"lagret{window}"] = lagret.astype(np.float32)
        out[f"lagret{window}_abs"] = lagret.abs().astype(np.float32)
        out[f"lagret{window}_sign"] = np.sign(lagret).astype(np.float32)
        out[f"lagret{window}_x_trade_pressure"] = lagret * df["trade_pressure_qty"].fillna(0.0)
        out[f"lagret{window}_x_ofi"] = lagret * df["ofi_total"].fillna(0.0)
        out[f"lagret{window}_x_spread"] = lagret * df["spread"].fillna(0.0)
        out[f"lagret{window}_x_vol"] = lagret * day_symbol["mid_ret1_raw"].transform(
            lambda s: s.rolling(window=min(10, window + 2), min_periods=1).std(ddof=0)
        ).fillna(0.0)
    if "lagret12" in out.columns:
        out["momentum_state"] = (out["lagret12"] > 0).astype(np.float32)
        out["reversal_state"] = (out["lagret12"] < 0).astype(np.float32)
        out["conditional_momentum_rank"] = out["lagret12"].groupby(df["date"], sort=False).transform(
            lambda s: s.rank(pct=True, method="average")
        ).fillna(0.0)
    else:
        out["momentum_state"] = np.float32(0.0)
        out["reversal_state"] = np.float32(0.0)
        out["conditional_momentum_rank"] = np.float32(0.0)
    return _sanitize_feature_frame(out)


def build_regime(raw: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """regime stage：迁移自旧版 `_add_regime_features`。"""
    df = _concat_inputs(_sort_raw_frame(raw), base)
    out = pd.DataFrame(index=df.index)
    day_symbol = df.groupby(["date", "symbol"], sort=False)
    intraday = df.groupby(["date"], sort=False)
    ret1 = df["mid_ret1_raw"].fillna(0.0)
    vol10 = day_symbol["mid_ret1_raw"].transform(
        lambda s: s.rolling(window=10, min_periods=1).std(ddof=0)
    ).fillna(0.0)
    imbalance = df["order_pressure"].fillna(0.0)
    spread = df["spread"].fillna(0.0)
    activity = df["trade_activity"].fillna(0.0)
    cxl = df["cxl_imb"].fillna(0.0)
    vol_cs = intraday["mid_ret1_raw"].transform(
        lambda s: s.rolling(window=10, min_periods=1).std(ddof=0)
    ).fillna(0.0)
    spread_cs = intraday["spread"].transform("mean").fillna(0.0)
    activity_cs = intraday["trade_activity"].transform("mean").fillna(0.0)
    regime_score = (
        0.35 * (vol10 / (vol10.groupby(df["date"], sort=False).transform("mean").replace(0.0, np.nan).fillna(1.0)))
        + 0.25 * spread
        + 0.20 * imbalance.abs()
        + 0.10 * activity
        + 0.10 * cxl.abs()
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    score_q1 = regime_score.groupby(df["date"], sort=False).transform(lambda s: s.quantile(0.33))
    score_q2 = regime_score.groupby(df["date"], sort=False).transform(lambda s: s.quantile(0.66))
    out["regime_score"] = regime_score
    out["regime_low"] = (regime_score <= score_q1).astype(np.float32)
    out["regime_mid"] = ((regime_score > score_q1) & (regime_score <= score_q2)).astype(np.float32)
    out["regime_high"] = (regime_score > score_q2).astype(np.float32)
    out["state_momentum"] = (
        (ret1 > 0).astype(np.float32)
        * (vol10 < vol10.groupby(df["date"], sort=False).transform("median").fillna(0.0)).astype(np.float32)
    ).fillna(0.0)
    out["state_reversal"] = (
        (ret1 < 0).astype(np.float32)
        * (spread > spread.groupby(df["date"], sort=False).transform("median").fillna(0.0)).astype(np.float32)
    ).fillna(0.0)
    out["state_pressure"] = (
        imbalance.abs() > imbalance.groupby(df["date"], sort=False).transform("median").abs().fillna(0.0)
    ).astype(np.float32)
    out["state_liquidity"] = (
        activity < activity_cs.groupby(df["date"], sort=False).transform("median").fillna(0.0)
    ).astype(np.float32)
    out["state_vol_cs"] = vol_cs
    out["state_spread_cs"] = spread_cs
    out["state_activity_cs"] = activity_cs
    return _sanitize_feature_frame(out)


registry = FeatureRegistry()

registry.stage(
    name="base",
    deps=[],
    groups=["legacy", "base", "norm_core"],
    status="promoted",
    group_columns={
        "legacy": LEGACY_COLS,
        "base": BASE_CORE_COLS,
        "norm_core": BASE_CORE_COLS,
    },
)(build_base)

registry.stage(
    name="ofi",
    deps=[],
    groups=["ofi", "ofi_raw", "ofi_dynamic", "ofi_safe"],
    status="promoted",
    group_columns={
        "ofi_raw": [
            "bid_ofi_0",
            "ask_ofi_0",
            "ofi_0",
            "bid_ofi_4",
            "ask_ofi_4",
            "ofi_4",
            "bid_ofi_9",
            "ask_ofi_9",
            "ofi_9",
            "bid_ofi_19",
            "ask_ofi_19",
            "ofi_19",
            "ofi_total",
        ],
        "ofi_dynamic": [
            f"ofi_total_ema{window}" for window in [3, 6, 12, 24]
        ] + [
            f"ofi_total_sum{window}" for window in [3, 6, 12, 24]
        ] + [
            f"ofi_total_mean{window}" for window in [3, 6, 12, 24]
        ] + [
            f"ofi_total_z{window}" for window in [3, 6, 12, 24]
        ],
        "ofi": _ofi_stage_columns(),
        "ofi_safe": _ofi_stage_columns(),
    },
)(build_ofi)

registry.stage(
    name="lag",
    deps=["base"],
    groups=["lag", "lag_short", "lag_mid", "lag_long"],
    status="promoted",
    group_columns={
        "lag": _lag_columns(),
        "lag_short": [col for col in _lag_columns() if any(token in col for token in ["_lag_1", "_lag_3", "_lag_5"])],
        "lag_mid": [col for col in _lag_columns() if "_lag_10" in col],
        "lag_long": [col for col in _lag_columns() if any(token in col for token in ["_lag_20", "_lag_30"])],
    },
)(build_lag)

registry.stage(
    name="roll",
    deps=["base"],
    groups=["roll", "roll_short", "roll_mid", "roll_long"],
    status="promoted",
    group_columns={
        "roll": _roll_columns(),
        "roll_short": [col for col in _roll_columns() if any(token in col for token in ["_rm3", "_rm5", "_rs3", "_rs5"])],
        "roll_mid": [col for col in _roll_columns() if any(token in col for token in ["_rm10", "_rs10"])],
        "roll_long": [col for col in _roll_columns() if any(token in col for token in ["_rm20", "_rm30", "_rs20", "_rs30"])],
    },
)(build_roll)

registry.stage(
    name="patch",
    deps=["base"],
    groups=["patch", "patch_summary"],
    status="promoted",
    group_columns={
        "patch": _patch_columns(),
        "patch_summary": _patch_columns(),
    },
)(build_patch)

registry.stage(
    name="trade_impact",
    deps=["base", "ofi"],
    groups=["trade_impact", "trade_impact_dyn", "trade_impact_interaction", "trade_impact_safe"],
    status="promoted",
    group_columns={
        "trade_impact": _trade_impact_stage_columns(),
        "trade_impact_dyn": [
            col
            for col in _trade_impact_stage_columns()
            if any(token in col for token in ["_ema", "_sum", "_mean", "_z"])
            and (
                col.startswith("trade_pressure_")
                or col.startswith("trade_intensity")
                or col.startswith("avg_trade_")
            )
        ],
        "trade_impact_interaction": [
            "trade_pressure_x_spread",
            "trade_pressure_x_order_pressure",
            "trade_pressure_x_ofi",
        ],
        "trade_impact_safe": _trade_impact_stage_columns(),
    },
)(build_trade_impact)

registry.stage(
    name="cross",
    deps=["base", "ofi", "trade_impact"],
    groups=[
        "cross_z",
        "cross_rank",
        "cross_rank_features",
        "norm_core",
        "cross",
        "ofi",
        "ofi_dynamic",
        "ofi_rank",
        "ofi_safe",
        "trade_impact",
        "trade_impact_dyn",
        "trade_impact_safe",
    ],
    status="promoted",
    group_columns={
        "cross_z": _cross_z_columns(),
        "cross_rank": _cross_rank_columns(),
        "cross_rank_features": _cross_rank_columns(),
        "norm_core": _cross_z_columns(),
        "cross": _cross_z_columns() + _cross_rank_columns() + ["interval_pos", "interval_norm", "is_morning", "is_afternoon"],
        "ofi": [col for col in (_cross_z_columns() + _cross_rank_columns()) if col.startswith("ofi_")],
        "ofi_dynamic": [col for col in _cross_z_columns() if col.startswith("ofi_")],
        "ofi_rank": [col for col in _cross_rank_columns() if col.startswith("ofi_")],
        "ofi_safe": [col for col in (_cross_z_columns() + _cross_rank_columns()) if col.startswith("ofi_")],
        "trade_impact": [
            col
            for col in (_cross_z_columns() + _cross_rank_columns())
            if col.startswith("trade_pressure_")
            or col.startswith("trade_intensity")
            or col.startswith("avg_trade_")
        ],
        "trade_impact_dyn": [
            col
            for col in (_cross_z_columns() + _cross_rank_columns())
            if col.startswith("trade_pressure_")
            or col.startswith("trade_intensity")
            or col.startswith("avg_trade_")
        ],
        "trade_impact_safe": [
            col
            for col in (_cross_z_columns() + _cross_rank_columns())
            if col.startswith("trade_pressure_")
            or col.startswith("trade_intensity")
            or col.startswith("avg_trade_")
        ],
    },
)(build_cross)

registry.stage(
    name="conditional_momentum",
    deps=["base", "ofi", "trade_impact"],
    groups=["conditional_momentum", "conditional_momentum_interaction", "conditional_momentum_safe"],
    status="promoted",
    group_columns={
        "conditional_momentum_interaction": [
            col
            for col in _conditional_momentum_columns()
            if col.startswith("lagret") and "_x_" in col
        ],
        "conditional_momentum_safe": _conditional_momentum_columns(),
    },
)(build_conditional_momentum)

registry.stage(
    name="regime",
    deps=["base"],
    groups=["regime", "regime_tree"],
    status="promoted",
    group_columns={
        "regime": _regime_columns(),
        # regime_tree（P4 树专用）：去掉 state_spread_cs / state_activity_cs 两个
        # “一天一值、当天全 symbol 相同”的广播常量——树会拿它们当“日期身份”乱切、
        # 稀释重要性。state_vol_cs 是日内 rolling、非广播常量，保留。
        "regime_tree": [
            col for col in _regime_columns()
            if col not in ("state_spread_cs", "state_activity_cs")
        ],
    },
)(build_regime)


__all__ = [
    "EPS",
    "META_COLS",
    "TARGET_COL",
    "FeatureRegistry",
    "StageDefinition",
    "registry",
    "build_base",
    "build_lag",
    "build_roll",
    "build_patch",
    "build_ofi",
    "build_trade_impact",
    "build_cross",
    "build_conditional_momentum",
    "build_regime",
    "_make_schema_probe_raw",
]
