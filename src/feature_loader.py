"""
特征加载层

M3 的职责：
1. 从 `data/features/` 读取已经构建好的 stage artifact。
2. 根据 registry 的 group 解析结果，按需选列并拼接多天数据。
3. 从原始 H5 读取 meta/target，避免把非特征列混入 stage artifact。
4. 在加载阶段做最基本的行对齐校验，尽早暴露错位问题。

设计约束：
- Loader 本身保持无状态，不做跨调用缓存，避免再次引入 OOM 根因。
- 存储协议复用 `feature_store.py` 的读写辅助函数，保证单点定义。
- 为后续 `resolved_columns.json` 快照预留结构化记录接口。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from dl import MeowDataLoader
from feature_registry import META_COLS, TARGET_COL, FeatureRegistry, registry as default_registry
from feature_store import DEFAULT_FEATURE_DIR, detect_storage_backend, read_feature_frame


class FeatureLoader:
    """
    无状态特征加载器。

    参数说明：
    - h5dir: 原始 H5 根目录。Loader 只从这里读取 meta/target，不做特征计算。
    - feature_dir: FeatureStore 的产物根目录。
    - registry: FeatureRegistry 实例，用于 group → stage/columns 解析。
    - loader_cls: 允许测试时注入 FakeLoader，避免依赖真实 H5。
    - storage_backend: 可显式指定；若为空则优先从 manifest 推断，再回退自动检测。
    """

    def __init__(
        self,
        h5dir: str,
        feature_dir: str = DEFAULT_FEATURE_DIR,
        registry: FeatureRegistry = default_registry,
        loader_cls=MeowDataLoader,
        storage_backend: Optional[str] = None,
        feature_dtype: Optional[str] = "float32",
    ):
        self.h5dir = Path(h5dir)
        self.feature_dir = Path(feature_dir)
        self.registry = registry
        self.loader = loader_cls(h5dir=str(self.h5dir))
        self.manifest_path = self.feature_dir / "manifest.json"
        # 让 registry 在运行时优先使用当前 feature_dir 的 manifest 列信息。
        self.registry.set_manifest_path(str(self.manifest_path))
        self.storage_backend = storage_backend or self._resolve_storage_backend()
        # #16：默认在 loader 返回前把特征列压成 float32，直接降低后续拼接与
        # worker 进程内 DataFrame 常驻内存；同时保留可切回 float64 的开关，
        # 供数值对照验收复用，避免靠手改代码做一次性实验。
        self.feature_dtype = self._normalize_feature_dtype(feature_dtype)
        # 记录最近一次 load 的解析结果，供后续 eval_protocol 写 resolved_columns.json。
        self._last_load_info: Dict[str, object] = {}

    # -----------------------------------------------------------------
    # 基础工具
    # -----------------------------------------------------------------

    def _resolve_storage_backend(self) -> str:
        """
        决定当前 loader 使用的存储 backend。

        优先级：
        1. manifest 中记录的 storage_backend
        2. 当前环境自动检测结果
        """
        if self.manifest_path.exists():
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            backend = payload.get("storage_backend")
            if backend:
                return str(backend)
        return detect_storage_backend()

    def _normalize_feature_dtype(self, feature_dtype: Optional[str]) -> Optional[np.dtype]:
        """
        统一解析特征 dtype 配置。

        约束：
        - None 表示保持 stage artifact 原始 dtype，不额外强转
        - 当前只允许 float32 / float64，避免有人误传整数或 object 破坏训练口径
        """
        if feature_dtype is None:
            return None
        dtype = np.dtype(feature_dtype)
        if dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError(
                "feature_dtype 仅支持 float32 / float64 / None"
            )
        return dtype

    def stage_file(self, stage_name: str, date: int) -> Path:
        """返回某个 stage 的单日 artifact 路径。"""
        return self.feature_dir / stage_name / f"{int(date)}.parquet"

    def _normalize_dates(self, dates: Sequence[int]) -> Tuple[int, ...]:
        """
        规范化日期输入，但保留调用方传入的时序。

        这里不做排序，因为 rolling fold 的训练/验证日期已经是上层确定好的时间顺序。
        """
        normalized = tuple(int(date) for date in dates)
        if not normalized:
            raise ValueError("FeatureLoader.load() 至少需要一个交易日")
        return normalized

    def _load_meta_target_frame(self, date: int) -> pd.DataFrame:
        """
        从原始 H5 读取 meta + target，并统一排序口径。

        这样做的原因：
        - stage artifact 只保存特征列，职责更干净。
        - 训练/评估真正需要的 `fret12` 仍然由 loader 统一提供。
        - raw 读几列 meta/target 的开销很低，不值得单独缓存。
        """
        raw = self.loader.loadDate(int(date))
        frame = (
            raw[META_COLS + [TARGET_COL]]
            .sort_values(META_COLS, kind="mergesort")
            .reset_index(drop=True)
            .copy()
        )
        frame[TARGET_COL] = pd.to_numeric(frame[TARGET_COL], errors="coerce").fillna(0.0).astype("float32")
        return frame

    def _read_stage_frame(self, stage_name: str, date: int) -> pd.DataFrame:
        """读取某个 stage 某天的 artifact，不存在时抛出明确错误。"""
        path = self.stage_file(stage_name, date)
        if not path.exists():
            raise FileNotFoundError(f"缺少 stage artifact: {path}")
        return read_feature_frame(path, backend=self.storage_backend)

    def _assert_stage_alignment(
        self,
        stage_name: str,
        date: int,
        stage_df: pd.DataFrame,
        meta_target_df: pd.DataFrame,
    ) -> None:
        """
        校验 stage 输出与原始 meta/target 的行对齐。

        当前 stage artifact 不保存 key 列，因此可验证的信息主要有两类：
        1. 行数必须一致
        2. DataFrame index 必须与 raw 排序后的 index 一致

        在 pickle_fallback backend 下，若有人意外打乱行顺序并保留原 index，
        这里可以直接抓到。对于 parquet backend，index 会被抹平为 RangeIndex，
        这条校验退化为“至少保证行数一致且没有显式 index 异常”。
        """
        if len(stage_df) != len(meta_target_df):
            raise ValueError(
                f"stage={stage_name} date={date} 行数不匹配: "
                f"stage={len(stage_df)} raw={len(meta_target_df)}"
            )
        if not stage_df.index.equals(meta_target_df.index):
            raise ValueError(
                f"stage={stage_name} date={date} 行顺序不匹配: "
                "stage index 与 raw meta index 不一致"
            )

    def _select_stage_columns(
        self,
        stage_name: str,
        stage_df: pd.DataFrame,
        requested_columns: Sequence[str],
    ) -> pd.DataFrame:
        """
        从单个 stage 输出中提取所需列，并对缺列报明确错误。

        这里不做“默默跳过缺列”，因为那会把缓存损坏或 manifest 失配的问题吞掉。
        """
        missing = [col for col in requested_columns if col not in stage_df.columns]
        if missing:
            raise KeyError(
                f"stage={stage_name} 缺少 {len(missing)} 个请求列，"
                f"例如: {missing[:5]}"
            )
        selected = stage_df.loc[:, list(requested_columns)].copy()
        if self.feature_dtype is None or selected.empty:
            return selected
        # 这里统一在 loader 层做 dtype 收口，而不是等到各训练分支各自 to_numpy 时
        # 再隐式转换，便于：
        # 1. 更早释放一半内存；
        # 2. 让串行 / 并行 / 提交桥接都走同一份口径；
        # 3. 支持 float32 vs float64 的显式数值对照验收。
        for col in selected.columns:
            selected[col] = pd.to_numeric(selected[col], errors="coerce").fillna(0.0).astype(self.feature_dtype)
        return selected

    # -----------------------------------------------------------------
    # 对外接口
    # -----------------------------------------------------------------

    def last_load_info(self) -> Dict[str, object]:
        """
        返回最近一次 `load()` 的解析信息副本。

        该结构主要给后续 `eval_protocol` 产出 `resolved_columns.json` 使用，
        当前阶段先保证接口稳定，避免后面再改调用约定。
        """
        return dict(self._last_load_info)

    def load(
        self,
        dates: Sequence[int],
        groups: Optional[Iterable[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        加载指定日期范围 + 指定特征组的数据。

        返回：
        - xdf: `meta + feature columns`
        - ydf: `meta + target`
        """
        normalized_dates = self._normalize_dates(dates)
        resolved = self.registry.resolve_groups(groups)
        stage_order = [
            stage_name
            for stage_name in self.registry.topo_order(include_archived=False)
            if stage_name in resolved
        ]

        x_parts: List[pd.DataFrame] = []
        y_parts: List[pd.DataFrame] = []

        for date in normalized_dates:
            meta_target_df = self._load_meta_target_frame(date)
            stage_feature_parts: List[pd.DataFrame] = []

            for stage_name in stage_order:
                stage_df = self._read_stage_frame(stage_name, date)
                self._assert_stage_alignment(stage_name, date, stage_df, meta_target_df)
                selected = self._select_stage_columns(
                    stage_name=stage_name,
                    stage_df=stage_df,
                    requested_columns=resolved[stage_name],
                )
                stage_feature_parts.append(selected)

            day_x = pd.concat([meta_target_df[META_COLS].copy(), *stage_feature_parts], axis=1)
            # 多 group 合并后再次防御性去重，避免未来有人在 registry 中误配重复列。
            day_x = day_x.loc[:, ~day_x.columns.duplicated()].copy()
            day_y = meta_target_df[META_COLS + [TARGET_COL]].copy()

            x_parts.append(day_x)
            y_parts.append(day_y)

        xdf = pd.concat(x_parts, ignore_index=True)
        ydf = pd.concat(y_parts, ignore_index=True)

        resolved_columns: List[str] = [
            col
            for stage_name in stage_order
            for col in resolved[stage_name]
        ]
        self._last_load_info = {
            "dates": list(normalized_dates),
            "groups": list(groups) if groups is not None and not isinstance(groups, str) else (
                [groups] if isinstance(groups, str) else None
            ),
            "resolved_stage_columns": {
                stage_name: list(resolved[stage_name])
                for stage_name in stage_order
            },
            "resolved_columns": list(resolved_columns),
            "stages_used": {
                stage_name: self.registry.code_hash(stage_name)
                for stage_name in stage_order
            },
            "storage_backend": self.storage_backend,
            "feature_dtype": None if self.feature_dtype is None else self.feature_dtype.name,
        }
        return xdf, ydf
