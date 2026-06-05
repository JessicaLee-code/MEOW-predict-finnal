# -*- coding: utf-8 -*-
"""
experiment_runner 训练标签 winsorize 单元测试。

覆盖目标：
  - 默认配置应落在 P0.5 扫描后锁定的 P1 / P99
  - 开关关闭时，不得偷偷改训练目标
  - 打开时，必须按训练集分位对目标两侧裁剪
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from experiment_runner import ExperimentRunner  # noqa: E402


class _DummyLoader:
    """最小占位 loader：测试只关心 runner 初始化，不触发真实数据读取。"""

    def __init__(self, h5dir):
        self.h5dir = h5dir


class _DummyFeatureLoader:
    """最小占位 feature loader：避免测试依赖磁盘特征目录。"""

    def __init__(self, h5dir, feature_dir, loader_cls, feature_dtype="float32"):
        self.h5dir = h5dir
        self.feature_dir = feature_dir
        self.loader_cls = loader_cls
        self.feature_dtype = feature_dtype


class TestTargetWinsorize(unittest.TestCase):
    """锁住 #15 的核心约束，避免后续重构把训练标签口径改丢。"""

    def _make_runner(self, config=None, ridge_alpha=2.0):
        return ExperimentRunner(
            h5dir="dummy-data",
            feature_dir="dummy-features",
            loader_cls=_DummyLoader,
            feature_loader_cls=_DummyFeatureLoader,
            target_winsorize_config=config,
            ridge_alpha=ridge_alpha,
        )

    def test_default_config_matches_agents(self):
        """默认口径必须是开启 + P1 / P99。"""
        runner = self._make_runner()
        cfg = runner.get_target_winsorize_config()
        self.assertTrue(cfg["enabled"])
        self.assertAlmostEqual(cfg["lower_quantile"], 0.01)
        self.assertAlmostEqual(cfg["upper_quantile"], 0.99)
        self.assertAlmostEqual(runner.get_ridge_alpha(), 2.0)

    def test_disabled_keeps_targets_unchanged(self):
        """用户显式关闭时，训练目标应原样返回。"""
        runner = self._make_runner({"enabled": False})
        values = np.asarray([-10.0, -1.0, 0.0, 1.0, 10.0], dtype=np.float32)
        clipped, info = runner._apply_target_winsorize(values)
        self.assertIsNone(info)
        np.testing.assert_allclose(clipped, values)

    def test_enabled_clips_to_train_quantiles(self):
        """打开时必须真的按训练集分位数裁剪，而不是只记录配置不生效。"""
        runner = self._make_runner({
            "enabled": True,
            "lower_quantile": 0.2,
            "upper_quantile": 0.8,
        })
        values = np.asarray([-100.0, -1.0, 0.0, 1.0, 100.0], dtype=np.float32)
        clipped, info = runner._apply_target_winsorize(values)
        self.assertIsNotNone(info)
        self.assertAlmostEqual(info["lower_bound"], -20.8, places=5)
        self.assertAlmostEqual(info["upper_bound"], 20.8, places=5)
        np.testing.assert_allclose(
            clipped,
            np.asarray([-20.8, -1.0, 0.0, 1.0, 20.8], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_invalid_ridge_alpha_fails_fast(self):
        """ridge alpha 必须是正数，避免扫参时把非法值带进真实训练。"""
        with self.assertRaises(ValueError):
            self._make_runner(ridge_alpha=0.0)


class TestTrainSubsample(unittest.TestCase):
    """锁住 P4 训练行降采样口径：仅树族生效、验证集不参与、x/y 同位、可复现、全交易日保留。"""

    def _make_runner(self, frac=None, seed=42):
        return ExperimentRunner(
            h5dir="dummy-data",
            feature_dir="dummy-features",
            loader_cls=_DummyLoader,
            feature_loader_cls=_DummyFeatureLoader,
            train_subsample_frac=frac,
            train_subsample_seed=seed,
        )

    def _make_xy(self, n=9000, days=30):
        import pandas as pd
        x = pd.DataFrame({"a": np.arange(n), "b": np.arange(n) * 2.0})
        y = pd.DataFrame({"date": np.arange(n) // (n // days), "fret12": np.arange(n) * 0.1})
        return x, y

    def test_normalize_disables_on_edge_values(self):
        """None / 1.0 / 0 / 负 / >1 一律视为关闭（None）；(0,1) 内才保留。"""
        self.assertIsNone(self._make_runner(None).get_train_subsample_frac())
        self.assertIsNone(self._make_runner(1.0).get_train_subsample_frac())
        self.assertIsNone(self._make_runner(0.0).get_train_subsample_frac())
        self.assertIsNone(self._make_runner(-0.5).get_train_subsample_frac())
        self.assertIsNone(self._make_runner(1.5).get_train_subsample_frac())
        self.assertAlmostEqual(self._make_runner(0.33).get_train_subsample_frac(), 0.33)

    def test_tree_model_is_subsampled(self):
        """树族模型按比例无放回降采样：行数对、x/y 同位、保序。"""
        runner = self._make_runner(0.33)
        x, y = self._make_xy()
        xs, ys = runner._subsample_train_rows(x, y, model_name="tree_shallow")
        self.assertEqual(len(xs), int(len(x) * 0.33))
        self.assertEqual(list(xs.index), list(ys.index))           # x/y 必须同位
        self.assertEqual(list(xs.index), sorted(xs.index))         # 采样后保序

    def test_linear_model_keeps_full_rows(self):
        """线性模型（闭式解、秒级）始终全量，不付保真代价。"""
        runner = self._make_runner(0.33)
        x, y = self._make_xy()
        for m in ("ridge", "elasticnet", "huber"):
            xs, ys = runner._subsample_train_rows(x, y, model_name=m)
            self.assertEqual(len(xs), len(x), f"线性 {m} 不应被采样")

    def test_reproducible_and_all_days_retained(self):
        """同种子结果可复现；全块均匀采样后所有交易日仍被保留（regime 覆盖不丢）。"""
        runner = self._make_runner(0.33, seed=42)
        x, y = self._make_xy()
        xs1, _ = runner._subsample_train_rows(x, y, model_name="histgb_shallow")
        xs2, _ = runner._subsample_train_rows(x, y, model_name="histgb_shallow")
        self.assertEqual(list(xs1.index), list(xs2.index))         # 可复现
        _, ys = runner._subsample_train_rows(x, y, model_name="tree_shallow")
        self.assertEqual(set(ys["date"]), set(y["date"]))          # 全交易日保留

    def test_disabled_frac_is_noop(self):
        """frac 关闭时，即便树族模型也原样返回。"""
        runner = self._make_runner(None)
        x, y = self._make_xy()
        xs, ys = runner._subsample_train_rows(x, y, model_name="tree_shallow")
        self.assertEqual(len(xs), len(x))


if __name__ == "__main__":
    unittest.main(verbosity=2)
