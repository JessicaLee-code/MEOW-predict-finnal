# -*- coding: utf-8 -*-
"""
DL serve 腿（焊接）单元测试 —— 只测不需 GPU/数据的"胶水逻辑"：
1. 融合对齐：传统 × DL 按 (date,symbol,interval) 等权融合，DL warmup 缺行用纯传统填、不引入 NaN；
2. 防御降级：DL 不可用 / fit 失败 → predict 返回 None、available=False、绝不抛异常拖崩提交链。

DL 的真实训练（torch + raw 数据）由 serve 端到端验证跑覆盖，不在单测里（单测保持快、无依赖）。
"""

import os
import sys
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# dl_serve 在 meow/ 下，且其 import 的 log 也在 meow/；把 meow/ 放最前。
sys.path.insert(0, os.path.join(REPO_ROOT, "meow"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dl_serve import DLServe, fuse_traditional_with_dl  # noqa: E402


def _xdf(rows):
    """构造只含 meta 的特征帧（融合只用 meta，特征列无所谓）。"""
    return pd.DataFrame(rows, columns=["date", "symbol", "interval"])


class TestFusion(unittest.TestCase):
    """等权融合 + warmup 填的对齐正确性。"""

    def test_fuse_partial_dl_fills_warmup_with_traditional(self):
        # 4 行：DL 只覆盖第 0、2 行（第 1、3 行模拟 warmup 缺预测）。
        xdf = _xdf([
            [20230601, 1, 100],
            [20230601, 1, 101],   # warmup：DL 无预测
            [20230601, 2, 100],
            [20230601, 2, 101],   # warmup：DL 无预测
        ])
        trad = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        dl_df = pd.DataFrame({
            "date": [20230601, 20230601],
            "symbol": [1, 2],
            "interval": [100, 100],
            "pred_dl": [10.0, 30.0],
        })
        fused = fuse_traditional_with_dl(xdf, trad, dl_df, weight_dl=0.5)
        # 第 0、2 行等权融合；第 1、3 行（warmup）保持纯传统。
        np.testing.assert_allclose(fused, [5.5, 2.0, 16.5, 4.0], atol=1e-6)
        self.assertTrue(np.isfinite(fused).all())

    def test_fuse_preserves_xdf_row_order(self):
        # 故意让 dl_df 顺序与 xdf 相反，验证融合后仍对齐回 xdf 行序。
        xdf = _xdf([
            [20230601, 5, 10],
            [20230601, 3, 20],
            [20230601, 9, 30],
        ])
        trad = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        dl_df = pd.DataFrame({
            "date": [20230601, 20230601, 20230601],
            "symbol": [9, 3, 5],
            "interval": [30, 20, 10],
            "pred_dl": [9.0, 3.0, 5.0],
        })
        fused = fuse_traditional_with_dl(xdf, trad, dl_df, weight_dl=0.5)
        # 每行 = 0.5*1 + 0.5*该行 symbol 对应的 dl → [0.5+2.5, 0.5+1.5, 0.5+4.5]。
        np.testing.assert_allclose(fused, [3.0, 2.0, 5.0], atol=1e-6)

    def test_fuse_none_returns_traditional(self):
        xdf = _xdf([[20230601, 1, 100], [20230601, 1, 101]])
        trad = np.array([7.0, 8.0], dtype=np.float32)
        out = fuse_traditional_with_dl(xdf, trad, None)
        np.testing.assert_allclose(out, [7.0, 8.0], atol=1e-6)

    def test_fuse_empty_dl_returns_traditional(self):
        xdf = _xdf([[20230601, 1, 100]])
        trad = np.array([7.0], dtype=np.float32)
        empty = pd.DataFrame({"date": [], "symbol": [], "interval": [], "pred_dl": []})
        out = fuse_traditional_with_dl(xdf, trad, empty)
        np.testing.assert_allclose(out, [7.0], atol=1e-6)


class TestDefensiveFallback(unittest.TestCase):
    """DL 不可用 / fit 失败时绝不拖崩提交链。"""

    def test_predict_none_when_unavailable(self):
        serve = DLServe(raw_loader=lambda dates: None)
        self.assertFalse(serve.available)
        self.assertIsNone(serve.predict([20230601]))

    def test_fit_failure_sets_unavailable_and_does_not_raise(self):
        # raw_loader 一调用就抛 → fit 必须吞掉、置 available=False、不向上抛。
        def _boom(dates):
            raise RuntimeError("模拟 raw 读取失败")

        serve = DLServe(raw_loader=_boom)
        try:
            serve.fit([20230601, 20230602, 20230605, 20230606, 20230607,
                       20230608, 20230609, 20230612, 20230613, 20230614])
        except Exception as e:  # 绝不应该抛
            self.fail("DLServe.fit 不应向上抛异常，但抛了: {!r}".format(e))
        self.assertFalse(serve.available)
        self.assertIsNone(serve.predict([20230601]))

    def test_seeds_env_override(self):
        # MEOW_DL_SEEDS 覆盖 seed 数（serve 测试提速用）。
        old = os.environ.get("MEOW_DL_SEEDS")
        try:
            os.environ["MEOW_DL_SEEDS"] = "42,43"
            serve = DLServe(raw_loader=lambda dates: None)
            self.assertEqual(serve.seeds, (42, 43))
        finally:
            if old is None:
                os.environ.pop("MEOW_DL_SEEDS", None)
            else:
                os.environ["MEOW_DL_SEEDS"] = old


if __name__ == "__main__":
    unittest.main(verbosity=2)
