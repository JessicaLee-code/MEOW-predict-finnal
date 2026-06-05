#!/usr/bin/env python3
"""
M2 回归测试：FeatureStore

覆盖点：
1. 首次 build 会生成 manifest 与 9 个 stage 文件。
2. 第二次检查无代码变更时应全部 clean。
3. 篡改上游 stage 的 hash 后，dirty 能传递到全部下游。
4. common_version 变化会使全部 stage 失效。
5. `stages=ofi` 会自动展开到 ofi + 全部下游，而不是只重建单点。

测试策略：
- 不依赖真实 H5 内容，使用 FakeLoader 注入可控的日级 DataFrame。
- 真正落盘到临时目录，直接覆盖存储协议与 manifest 行为。
"""

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ_ROOT / "src"))

from feature_registry import _make_schema_probe_raw, registry
from feature_store import FeatureStore
from tradingcalendar import Calendar


class FakeLoader:
    """返回可重复的单日 raw DataFrame，用于替代真实 H5 IO。"""

    def __init__(self, h5dir: str):
        self.h5dir = h5dir
        self._template = _make_schema_probe_raw()

    def loadDate(self, date: int) -> pd.DataFrame:
        frame = self._template[self._template["date"] == 20230601].copy()
        frame.loc[:, "date"] = int(date)
        return frame


def _touch(path: Path) -> None:
    """创建一个空文件，用于模拟本地 H5 存在。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def run_tests():
    checks = []

    def ok(name, cond, detail=""):
        tag = "✓ PASS" if cond else "✗ FAIL"
        line = f"  {tag}  {name}"
        if detail:
            line += f"  [{detail}]"
        checks.append((cond, line))

    with tempfile.TemporaryDirectory(prefix="feature-store-test-") as tmp:
        root = Path(tmp)
        h5dir = root / "data"
        feature_dir = root / "features"
        dates = [20230601, 20230602]
        for date in dates:
            _touch(h5dir / f"{date}.h5")

        store = FeatureStore(
            h5dir=str(h5dir),
            feature_dir=str(feature_dir),
            registry=registry,
            loader_cls=FakeLoader,
            storage_backend="pickle_fallback",
        )

        # ── T1: 首次全量构建 ───────────────────────────────────────────
        result = store.build(dates=dates)
        manifest = store.load_manifest()
        stage_names = store.registry.topo_order()
        ok("T1  首次 build 覆盖全部 9 个 stage", len(result["dirty_stages"]) == 9, f"实际={len(result['dirty_stages'])}")
        ok("T1  manifest 已生成", store.manifest_path.exists())
        ok("T1  manifest 记录 common_version", manifest["common_version"] == store.common_version, manifest["common_version"])
        ok("T1  manifest 记录 storage_backend", manifest["storage_backend"] == "pickle_fallback", manifest["storage_backend"])
        ok(
            "T1  每个 stage 都有 2 天 artifact",
            all(len(store.stage_existing_dates(stage)) == 2 for stage in stage_names),
            ",".join(f"{stage}:{len(store.stage_existing_dates(stage))}" for stage in stage_names),
        )

        # ── T2: 无改动时应全部 clean ───────────────────────────────────
        clean_dirty = store.find_dirty_stages(dates=dates)
        ok("T2  二次检查无脏 stage", clean_dirty == set(), f"实际={sorted(clean_dirty)}")

        # ── T3: base hash 改动应传递到所有下游 ─────────────────────────
        hacked = store.load_manifest()
        hacked["stages"]["base"]["code_hash"] = "outdated-hash"
        store.save_manifest(hacked)
        dirty_reasons = store.find_dirty_reasons(dates=dates)
        expected_base_chain = {"base", "lag", "roll", "patch", "trade_impact", "cross", "conditional_momentum", "regime"}
        ok(
            "T3  base hash 改动会打脏 base 及全部下游",
            expected_base_chain.issubset(set(dirty_reasons.keys())),
            f"实际={sorted(dirty_reasons.keys())}",
        )
        ok("T3  ofi 不依赖 base，不应被误打脏", "ofi" not in dirty_reasons, f"实际={sorted(dirty_reasons.keys())}")

        # 恢复正确 manifest，便于后续测试
        store.build(dates=dates, force=True)

        # ── T4: common_version 变化会让全部 stage 失效 ────────────────
        newer_store = FeatureStore(
            h5dir=str(h5dir),
            feature_dir=str(feature_dir),
            registry=registry,
            loader_cls=FakeLoader,
            storage_backend="pickle_fallback",
            common_version="common-v2",
        )
        all_dirty = newer_store.find_dirty_stages(dates=dates)
        ok("T4  common_version 变化会打脏全部 9 个 stage", len(all_dirty) == 9, f"实际={len(all_dirty)}")

        # ── T5: 指定 ofi 时会自动包含全部下游 ────────────────────────
        force_result = store.build(dates=dates, force=True, stages=["ofi"])
        ok(
            "T5  stages=ofi 会自动展开到自身与下游",
            force_result["dirty_stages"] == ["ofi", "trade_impact", "cross", "conditional_momentum"],
            f"实际={force_result['dirty_stages']}",
        )
        rows = store.status_rows(dates=dates)
        ok(
            "T5  定向重建后整体状态恢复 clean",
            all(row["state"] == "ok" for row in rows),
            json.dumps(rows, ensure_ascii=False),
        )

    print("\n" + "=" * 60)
    for _, line in checks:
        print(line)
    print("=" * 60)

    all_pass = all(cond for cond, _ in checks)
    if all_pass:
        print("  全部通过 ✓")
    else:
        fail_count = sum(1 for cond, _ in checks if not cond)
        print(f"  {fail_count} 项失败 ✗")
    return all_pass


if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)
