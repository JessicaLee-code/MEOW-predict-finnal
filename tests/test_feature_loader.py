# -*- coding: utf-8 -*-
"""
feature_loader 的 dtype 收口单元测试。

覆盖目标：
1. #16 默认必须把特征列压成 float32，直接降低后续训练/并发常驻内存。
2. 元信息列与目标列口径不能被误伤；目标仍沿用现有 float32 规则。
3. 必须保留显式切回 float64 / 原样保留的开关，供同折数值对照验收复用。
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from feature_loader import FeatureLoader  # noqa: E402
from feature_registry import META_COLS, TARGET_COL, FeatureRegistry  # noqa: E402
from feature_store import write_feature_frame  # noqa: E402


class _FakeRawLoader:
    """
    最小 raw loader：只提供 FeatureLoader 所需的 `loadDate`。

    测试目标是 loader 的 dtype 与拼接行为，不依赖真实 H5 数据。
    """

    def __init__(self, h5dir: str):
        self.h5dir = h5dir

    def loadDate(self, date: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "symbol": ["000001", "000002"],
                "date": [date, date],
                "interval": [93000000, 93000000],
                "fret12": [0.12, -0.34],
            }
        )


class TestFeatureLoaderDType(unittest.TestCase):
    """锁住 #16 的 loader 层 dtype 收口，避免后续重构把口径改丢。"""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.feature_dir = Path(self.tempdir.name)
        self.registry = FeatureRegistry()

        @self.registry.stage(
            name="demo_stage",
            deps=[],
            groups=["demo"],
            group_columns={"demo": ["feature_a", "feature_b"]},
        )
        def _demo_stage_builder(df, deps=None):
            # 测试只关心 registry 的列解析，不真正执行 builder。
            return df

        demo_stage_dir = self.feature_dir / "demo_stage"
        demo_stage_dir.mkdir(parents=True, exist_ok=True)
        write_feature_frame(
            pd.DataFrame(
                {
                    "feature_a": np.asarray([1.25, np.nan], dtype=np.float64),
                    "feature_b": np.asarray([2.5, 3.5], dtype=np.float64),
                }
            ),
            demo_stage_dir / "20230601.parquet",
            backend="pickle_fallback",
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def _make_loader(self, feature_dtype="float32"):
        return FeatureLoader(
            h5dir="dummy-data",
            feature_dir=str(self.feature_dir),
            registry=self.registry,
            loader_cls=_FakeRawLoader,
            storage_backend="pickle_fallback",
            feature_dtype=feature_dtype,
        )

    def test_default_feature_dtype_is_float32(self):
        """默认返回的特征列必须是 float32，且 NaN 要按既有口径补 0。"""
        loader = self._make_loader()
        xdf, ydf = loader.load([20230601], groups=["demo"])

        self.assertEqual(list(xdf.columns[:3]), META_COLS)
        self.assertEqual(list(ydf.columns), META_COLS + [TARGET_COL])
        self.assertEqual(xdf["feature_a"].dtype, np.dtype("float32"))
        self.assertEqual(xdf["feature_b"].dtype, np.dtype("float32"))
        self.assertEqual(ydf[TARGET_COL].dtype, np.dtype("float32"))
        self.assertAlmostEqual(float(xdf.loc[1, "feature_a"]), 0.0, places=6)
        self.assertEqual(loader.last_load_info()["feature_dtype"], "float32")

    def test_can_force_float64_for_numeric_audit(self):
        """数值对照验收需要显式切回 float64，保证不是一次性手改代码。"""
        loader = self._make_loader(feature_dtype="float64")
        xdf, _ = loader.load([20230601], groups=["demo"])
        self.assertEqual(xdf["feature_a"].dtype, np.dtype("float64"))
        self.assertEqual(xdf["feature_b"].dtype, np.dtype("float64"))
        self.assertEqual(loader.last_load_info()["feature_dtype"], "float64")

    def test_none_keeps_original_stage_dtype(self):
        """传 None 时保持 stage artifact 原始 dtype，便于排查 loader 外的数值漂移。"""
        loader = self._make_loader(feature_dtype=None)
        xdf, _ = loader.load([20230601], groups=["demo"])
        self.assertEqual(xdf["feature_a"].dtype, np.dtype("float64"))
        self.assertEqual(xdf["feature_b"].dtype, np.dtype("float64"))
        self.assertIsNone(loader.last_load_info()["feature_dtype"])

    def test_invalid_feature_dtype_rejected(self):
        """非法 dtype 必须尽早失败，避免 object/int 悄悄混进训练链。"""
        with self.assertRaises(ValueError):
            self._make_loader(feature_dtype="int64")


if __name__ == "__main__":
    unittest.main(verbosity=2)
