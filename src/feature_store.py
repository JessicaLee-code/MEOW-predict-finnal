"""
特征缓存存储层

M2 的职责：
1. 管理 `data/features/` 目录下的 stage artifact 与 manifest。
2. 做脏检测：common_version / code_hash / 缺失文件 / 下游传递。
3. 提供 `build` / `status` 两个 CLI 入口。

实现说明：
- 规格优先使用 parquet；当前环境若缺少 parquet 引擎，则自动降级到 pickle-backend。
- 文件后缀仍保持 `.parquet`，这样目录结构与规格一致；manifest 会记录实际 backend。
- 后续 FeatureLoader 会复用本模块的读写辅助函数，保证存储协议单点定义。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

from dl import MeowDataLoader
from feature_registry import META_COLS, FeatureRegistry, registry as default_registry
from log import log
from tradingcalendar import Calendar


FEATURE_COMMON_VERSION = "common-v1"
DEFAULT_FEATURE_DIR = "data/features"
SUPPORTED_STORAGE_BACKENDS = {"parquet", "pickle_fallback"}


def detect_storage_backend() -> str:
    """
    检测当前 Python 环境可用的列式存储 backend。

    优先级：
    1. pyarrow / fastparquet 可用 → 真 parquet
    2. 否则退化为 pickle_fallback

    这样做的原因是当前仓库运行环境里没有 parquet engine，
    若此处直接硬失败，M2 的 build/status 无法独立验证。
    """
    if importlib.util.find_spec("pyarrow") or importlib.util.find_spec("fastparquet"):
        return "parquet"
    return "pickle_fallback"


def write_feature_frame(df: pd.DataFrame, path: Path, backend: Optional[str] = None) -> None:
    """
    将单个 stage 的单日特征写入磁盘。

    path 后缀统一使用 `.parquet`，以保持目录结构稳定；
    真实编码格式由 backend 决定，并写入 manifest。
    """
    backend = backend or detect_storage_backend()
    path.parent.mkdir(parents=True, exist_ok=True)
    if backend == "parquet":
        df.to_parquet(path, index=False)
        return
    if backend == "pickle_fallback":
        df.to_pickle(path)
        return
    raise ValueError(f"Unsupported storage backend: {backend}")


def read_feature_frame(path: Path, backend: Optional[str] = None) -> pd.DataFrame:
    """读取单个 stage 的单日特征文件，与 `write_feature_frame` 成对使用。"""
    backend = backend or detect_storage_backend()
    if backend == "parquet":
        return pd.read_parquet(path)
    if backend == "pickle_fallback":
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported storage backend: {backend}")


class FeatureStore:
    """
    管理 stage artifact、manifest 与增量构建。

    参数尽量保持显式：
    - h5dir: 原始 H5 根目录
    - feature_dir: stage artifact 根目录
    - registry: 特征注册表
    - common_version: 公共函数版本号，便于测试和后续手动升级
    """

    def __init__(
        self,
        h5dir: str,
        feature_dir: str = DEFAULT_FEATURE_DIR,
        registry: FeatureRegistry = default_registry,
        common_version: str = FEATURE_COMMON_VERSION,
        loader_cls=MeowDataLoader,
        calendar: Optional[Calendar] = None,
        storage_backend: Optional[str] = None,
    ):
        self.h5dir = Path(h5dir)
        self.feature_dir = Path(feature_dir)
        self.registry = registry
        self.common_version = common_version
        self.loader = loader_cls(h5dir=str(self.h5dir))
        self.calendar = calendar or Calendar()
        self.storage_backend = storage_backend or detect_storage_backend()
        if self.storage_backend not in SUPPORTED_STORAGE_BACKENDS:
            raise ValueError(f"Unsupported storage backend: {self.storage_backend}")

        self.feature_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.feature_dir / "manifest.json"
        self.registry.set_manifest_path(str(self.manifest_path))

    # -----------------------------------------------------------------
    # manifest 与路径工具
    # -----------------------------------------------------------------

    def stage_dir(self, stage_name: str) -> Path:
        """返回某个 stage 的目录。"""
        return self.feature_dir / stage_name

    def stage_file(self, stage_name: str, date: int) -> Path:
        """返回某个 stage 的单日 artifact 路径。"""
        return self.stage_dir(stage_name) / f"{int(date)}.parquet"

    def load_manifest(self) -> dict:
        """读取 manifest；不存在时返回默认骨架。"""
        if not self.manifest_path.exists():
            return {
                "common_version": None,
                "storage_backend": self.storage_backend,
                "stages": {},
            }
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        payload.setdefault("common_version", None)
        payload.setdefault("storage_backend", self.storage_backend)
        payload.setdefault("stages", {})
        return payload

    def save_manifest(self, manifest: dict) -> None:
        """原子性需求不高，直接覆盖写入即可。"""
        self.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def available_dates(self) -> List[int]:
        """
        枚举原始 H5 目录中可构建的交易日。

        这里只看真实文件名，不假设 calendar 中的每个交易日都已下载到本地。
        """
        if not self.h5dir.exists():
            return []
        dates: List[int] = []
        for path in self.h5dir.glob("*.h5"):
            try:
                date = int(path.stem)
            except ValueError:
                continue
            if self.calendar.isTradingDay(date):
                dates.append(date)
        return sorted(set(dates))

    def normalize_dates(self, dates: Optional[Iterable[int]] = None, include_all: bool = False) -> List[int]:
        """把外部输入的日期统一转成升序 int 列表。"""
        if include_all or dates is None:
            out = self.available_dates()
        else:
            out = sorted({int(x) for x in dates})
        if not out:
            raise ValueError("没有可构建的交易日")
        return out

    def stage_existing_dates(self, stage_name: str) -> List[int]:
        """列出某个 stage 当前已存在的单日文件。"""
        folder = self.stage_dir(stage_name)
        if not folder.exists():
            return []
        out: List[int] = []
        for path in folder.glob("*.parquet"):
            try:
                out.append(int(path.stem))
            except ValueError:
                continue
        return sorted(set(out))

    def read_stage_frame(self, stage_name: str, date: int, backend: Optional[str] = None) -> pd.DataFrame:
        """读取某 stage 某天的特征文件，不存在则报明确错误。"""
        path = self.stage_file(stage_name, date)
        if not path.exists():
            raise FileNotFoundError(f"缺少 stage artifact: {path}")
        return read_feature_frame(path, backend=backend or self.storage_backend)

    # -----------------------------------------------------------------
    # 脏检测
    # -----------------------------------------------------------------

    def _selected_stage_closure(self, stages: Optional[Sequence[str]]) -> Optional[Set[str]]:
        """
        将 CLI 指定的 stage 展开为“自身 + 全部下游”。

        这样做更安全：
        - 用户显式要求重建某 stage 时，不会留下已知 stale 的下游产物。
        - 与“改一个上游 stage 必须让依赖它的下游失效”这一设计宪法保持一致。
        """
        if not stages:
            return None
        selected: Set[str] = set()
        for stage_name in stages:
            if stage_name not in self.registry.topo_order(include_archived=True):
                raise KeyError(f"未知 stage: {stage_name}")
            selected.add(stage_name)
            selected.update(self.registry.downstream(stage_name))
        return selected

    def find_dirty_reasons(
        self,
        dates: Optional[Sequence[int]] = None,
        force: bool = False,
        stages: Optional[Sequence[str]] = None,
    ) -> Dict[str, str]:
        """
        返回 `{stage_name: reason}`，只包含 dirty stage。

        触发条件：
        1. `force=True`
        2. common_version 不一致
        3. manifest 缺 stage
        4. code_hash 不一致
        5. 请求日期范围内缺失 artifact
        6. 上游 stage dirty，向下游传播
        """
        dates = list(dates or self.available_dates())
        manifest = self.load_manifest()
        manifest_stages = manifest.get("stages", {})
        selected_closure = self._selected_stage_closure(stages)
        topo = self.registry.topo_order(include_archived=False)
        dirty: Dict[str, str] = {}

        if force:
            for stage_name in topo:
                if selected_closure is None or stage_name in selected_closure:
                    dirty[stage_name] = "forced rebuild"
            return dirty

        common_mismatch = manifest.get("common_version") != self.common_version
        if common_mismatch:
            old = manifest.get("common_version")
            for stage_name in topo:
                if selected_closure is None or stage_name in selected_closure:
                    dirty[stage_name] = f"common version changed (old: {old} -> new: {self.common_version})"
            return dirty

        for stage_name in topo:
            if selected_closure is not None and stage_name not in selected_closure:
                continue
            stage_meta = manifest_stages.get(stage_name)
            if stage_meta is None:
                dirty[stage_name] = "missing from manifest"
                continue
            current_hash = self.registry.code_hash(stage_name)
            old_hash = stage_meta.get("code_hash")
            if old_hash != current_hash:
                dirty[stage_name] = f"hash changed (old: {old_hash} -> new: {current_hash})"
                continue
            missing_dates = [
                int(date)
                for date in dates
                if not self.stage_file(stage_name, date).exists()
            ]
            if missing_dates:
                dirty[stage_name] = f"missing {len(missing_dates)} day files"

        # 这里做一次拓扑传播，确保上游改动会把所有下游一起打脏。
        dirty_stage_names = set(dirty.keys())
        for stage_name in topo:
            if selected_closure is not None and stage_name not in selected_closure:
                continue
            if stage_name in dirty_stage_names:
                continue
            for dep in self.registry.get_deps(stage_name):
                if dep in dirty_stage_names:
                    dirty[stage_name] = f"upstream {dep} is dirty"
                    dirty_stage_names.add(stage_name)
                    break
        return dirty

    def find_dirty_stages(
        self,
        dates: Optional[Sequence[int]] = None,
        force: bool = False,
        stages: Optional[Sequence[str]] = None,
    ) -> Set[str]:
        """只返回 dirty stage 名称集合，供 build 流程直接使用。"""
        return set(self.find_dirty_reasons(dates=dates, force=force, stages=stages).keys())

    # -----------------------------------------------------------------
    # 状态查询
    # -----------------------------------------------------------------

    def status_rows(
        self,
        dates: Optional[Sequence[int]] = None,
        force: bool = False,
        stages: Optional[Sequence[str]] = None,
    ) -> List[dict]:
        """生成 `status` CLI 所需的结构化输出。"""
        manifest = self.load_manifest()
        dirty_reasons = self.find_dirty_reasons(dates=dates, force=force, stages=stages)
        rows: List[dict] = []
        for stage_name in self.registry.topo_order(include_archived=False):
            stage_meta = manifest.get("stages", {}).get(stage_name, {})
            columns = stage_meta.get("columns") or self.registry.stage_columns(stage_name)
            row = {
                "stage": stage_name,
                "state": "dirty" if stage_name in dirty_reasons else "ok",
                "reason": dirty_reasons.get(stage_name, ""),
                "code_hash": stage_meta.get("code_hash", self.registry.code_hash(stage_name)),
                "n_days": int(stage_meta.get("n_days", len(self.stage_existing_dates(stage_name)))),
                "n_columns": len(columns),
            }
            rows.append(row)
        return rows

    def format_status_lines(self, rows: List[dict]) -> List[str]:
        """把结构化状态行渲染成易读的纯文本。"""
        lines: List[str] = []
        for row in rows:
            short_hash = row["code_hash"][:8] if row["code_hash"] else "unknown"
            if row["state"] == "ok":
                line = (
                    f"{row['stage']:<22} ok     "
                    f"hash={short_hash}  {row['n_days']} days  {row['n_columns']} cols"
                )
            else:
                line = (
                    f"{row['stage']:<22} dirty  "
                    f"{row['reason']}"
                )
            lines.append(line)
        return lines

    # -----------------------------------------------------------------
    # 构建
    # -----------------------------------------------------------------

    def _build_stage_for_date(
        self,
        stage_name: str,
        date: int,
        raw: pd.DataFrame,
        built_outputs: Dict[str, pd.DataFrame],
        dirty_stages: Set[str],
    ) -> pd.DataFrame:
        """
        构建某 stage 某一天的输出。

        built_outputs 只保存当前日期内已经构建出的上游结果，
        这样既能避免同一天重复读取，又不会把多天数据常驻内存。
        """
        deps = {}
        for dep in self.registry.get_deps(stage_name):
            if dep in built_outputs:
                deps[dep] = built_outputs[dep]
            else:
                deps[dep] = self.read_stage_frame(dep, date)
        builder = self.registry.get_builder(stage_name)
        frame = builder(raw, **deps)
        write_feature_frame(frame, self.stage_file(stage_name, date), backend=self.storage_backend)
        return frame

    def _refresh_stage_manifest_entry(self, manifest: dict, stage_name: str) -> None:
        """
        用磁盘现状刷新单个 stage 的 manifest。

        刷新字段：
        - code_hash
        - built_at
        - n_days
        - columns
        """
        stage_dates = self.stage_existing_dates(stage_name)
        columns = self.registry.stage_columns(stage_name)
        if stage_dates:
            sample = self.read_stage_frame(stage_name, stage_dates[0])
            columns = list(sample.columns)
        manifest.setdefault("stages", {})[stage_name] = {
            "code_hash": self.registry.code_hash(stage_name),
            "built_at": datetime.now().isoformat(timespec="seconds"),
            "n_days": len(stage_dates),
            "columns": columns,
        }

    def build(
        self,
        dates: Optional[Sequence[int]] = None,
        force: bool = False,
        stages: Optional[Sequence[str]] = None,
        include_all: bool = False,
    ) -> dict:
        """
        按日期范围构建脏 stage。

        流程与规格一致：
        1. 解析日期
        2. 脏检测
        3. 对每个日期按拓扑顺序构建 dirty stage
        4. 刷新 manifest
        """
        normalized_dates = self.normalize_dates(dates=dates, include_all=include_all)
        dirty_stages = self.find_dirty_stages(
            dates=normalized_dates,
            force=force,
            stages=stages,
        )
        topo = self.registry.topo_order(include_archived=False)
        dirty_order = [stage for stage in topo if stage in dirty_stages]

        if not dirty_order:
            manifest = self.load_manifest()
            manifest["common_version"] = self.common_version
            manifest["storage_backend"] = self.storage_backend
            self.save_manifest(manifest)
            return {
                "dates": normalized_dates,
                "dirty_stages": [],
                "built_days": 0,
                "storage_backend": self.storage_backend,
            }

        for date in normalized_dates:
            log.inf(f"[FeatureStore] building date={date}, stages={','.join(dirty_order)}")
            raw = self.loader.loadDate(int(date))
            raw = raw.sort_values(META_COLS, kind="mergesort").reset_index(drop=True)
            built_outputs: Dict[str, pd.DataFrame] = {}
            for stage_name in dirty_order:
                built_outputs[stage_name] = self._build_stage_for_date(
                    stage_name=stage_name,
                    date=int(date),
                    raw=raw,
                    built_outputs=built_outputs,
                    dirty_stages=dirty_stages,
                )

        manifest = self.load_manifest()
        manifest["common_version"] = self.common_version
        manifest["storage_backend"] = self.storage_backend
        for stage_name in dirty_order:
            self._refresh_stage_manifest_entry(manifest, stage_name)
        self.save_manifest(manifest)
        return {
            "dates": normalized_dates,
            "dirty_stages": dirty_order,
            "built_days": len(normalized_dates),
            "storage_backend": self.storage_backend,
        }


def parse_dates_arg(value: Optional[str]) -> Optional[List[int]]:
    """解析 CLI 传入的 `20230601,20230602` 形式日期列表。"""
    if value is None or value.strip() == "":
        return None
    return [int(token.strip()) for token in value.split(",") if token.strip()]


def parse_stages_arg(value: Optional[str]) -> Optional[List[str]]:
    """解析 CLI 传入的 `ofi,trade_impact` 形式 stage 列表。"""
    if value is None or value.strip() == "":
        return None
    return [token.strip() for token in value.split(",") if token.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    """构建 `python -m feature_store ...` 的 CLI。"""
    parser = argparse.ArgumentParser(description="Feature store CLI")
    parser.add_argument("command", choices=["build", "status"], help="要执行的命令")
    parser.add_argument("--h5dir", default="data", help="原始 H5 根目录")
    parser.add_argument("--feature-dir", default=DEFAULT_FEATURE_DIR, help="特征缓存根目录")
    parser.add_argument("--all", action="store_true", help="构建全部可用交易日")
    parser.add_argument("--dates", default=None, help="逗号分隔的交易日列表，例如 20230601,20230602")
    parser.add_argument("--force", action="store_true", help="忽略脏检测，强制重建")
    parser.add_argument("--stages", default=None, help="仅重建指定 stage（自动包含全部下游）")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 主入口。"""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    dates = parse_dates_arg(args.dates)
    stages = parse_stages_arg(args.stages)
    store = FeatureStore(
        h5dir=args.h5dir,
        feature_dir=args.feature_dir,
    )

    if args.command == "status":
        rows = store.status_rows(dates=dates, force=args.force, stages=stages)
        for line in store.format_status_lines(rows):
            print(line)
        return 0

    if args.command == "build":
        result = store.build(
            dates=dates,
            force=args.force,
            stages=stages,
            include_all=args.all,
        )
        built_stages = ",".join(result["dirty_stages"]) if result["dirty_stages"] else "(none)"
        print(
            "[FeatureStore] build finished: "
            f"backend={result['storage_backend']}, "
            f"days={result['built_days']}, stages={built_stages}"
        )
        rows = store.status_rows(dates=dates, force=False, stages=stages)
        for line in store.format_status_lines(rows):
            print(line)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
