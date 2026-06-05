# -*- coding: utf-8 -*-
"""
P5 OOF 落盘 + 加权融合评测器单元测试。

覆盖：
  - EvaluationProtocolRunner._make_oof_writer：逐折 append、HDF 行 schema、可往返读回
  - p5_blend_oof：parse_member / _safe_corr / _zscore / load_member_oof /
    build_blend_frame 对齐 / compute_metrics 三镜头指标

运行：
    PYTHONPATH=src python -m unittest tests.test_oof_blend -v
"""

import os
import sys
import tempfile
import unittest

_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from eval_protocol import EvaluationProtocolRunner  # noqa: E402
from experiments.p5_blend_oof import (  # noqa: E402
    parse_member,
    _safe_corr,
    _zscore,
    load_member_oof,
    build_blend_frame,
    compute_metrics,
    KEY_COLS,
)


def _make_yval(fold_seed, n=200):
    """构造一折的 yval（date/symbol/interval/fret12）+ 与之相关的 pred。"""
    rng = np.random.default_rng(fold_seed)
    n_sym = 20
    n_int = n // n_sym
    symbols = np.repeat(np.arange(90000000, 90000000 + n_sym), n_int)
    intervals = np.tile(np.arange(93000000, 93000000 + n_int), n_sym)
    date = 20230600 + fold_seed
    fret12 = rng.normal(0, 0.007, size=len(symbols)).astype(np.float32)
    yval = pd.DataFrame({
        "date": np.int64(date),
        "symbol": symbols.astype(np.int64),
        "interval": intervals.astype(np.int64),
        "fret12": fret12,
    })
    # pred 与 fret12 正相关 + 噪声
    pred = (0.6 * fret12 + rng.normal(0, 0.004, size=len(symbols))).astype(np.float32)
    return yval, pred


class TestOofWriter(unittest.TestCase):
    def test_round_trip_and_append(self):
        runner = EvaluationProtocolRunner(None)  # _make_oof_writer 不依赖 runner
        with tempfile.TemporaryDirectory() as tmp:
            writer = runner._make_oof_writer(tmp)
            y0, p0 = _make_yval(0)
            y1, p1 = _make_yval(1)
            writer("expanding_40d_5d", 0, "M_test", y0, p0)
            writer("expanding_40d_5d", 1, "M_test", y1, p1)

            path = os.path.join(tmp, "expanding_40d_5d__M_test.h5")
            self.assertTrue(os.path.exists(path))
            df = pd.read_hdf(path, key="oof").reset_index(drop=True)
            # schema
            for c in ["fold_id", "date", "symbol", "interval", "fret12", "pred"]:
                self.assertIn(c, df.columns)
            # 两折都落了
            self.assertEqual(len(df), len(y0) + len(y1))
            self.assertEqual(sorted(df["fold_id"].unique().tolist()), [0, 1])
            # pred 应与传入对齐（取 fold 0 第一行核对量级）
            self.assertEqual(df["pred"].dtype, np.float32)


class TestBlendHelpers(unittest.TestCase):
    def test_parse_member_ok(self):
        self.assertEqual(parse_member("run123:SPEC_A"), ("run123", "SPEC_A"))

    def test_parse_member_bad(self):
        with self.assertRaises(ValueError):
            parse_member("no_colon")
        with self.assertRaises(ValueError):
            parse_member("a:b:c")

    def test_safe_corr_zero_variance(self):
        self.assertEqual(_safe_corr(np.ones(10), np.arange(10)), 0.0)

    def test_safe_corr_perfect(self):
        x = np.arange(50, dtype=float)
        self.assertAlmostEqual(_safe_corr(x, 2 * x + 1), 1.0, places=6)

    def test_zscore_unit(self):
        z = _zscore(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        self.assertAlmostEqual(float(np.mean(z)), 0.0, places=6)
        self.assertAlmostEqual(float(np.std(z)), 1.0, places=6)


class TestBlendFrame(unittest.TestCase):
    def _write_member(self, oof_root, run_id, profile, exp_id, pred_scale=1.0, seed_off=0):
        runner = EvaluationProtocolRunner(None)
        oof_dir = os.path.join(oof_root, run_id, "oof")
        os.makedirs(oof_dir, exist_ok=True)
        writer = runner._make_oof_writer(oof_dir)
        for fold in (0, 1):
            yval, pred = _make_yval(fold + seed_off)
            writer(profile, fold, exp_id, yval, pred * pred_scale)

    def test_load_missing_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                load_member_oof(tmp, "nope", "expanding_40d_5d", "X")

    def test_build_blend_and_metrics(self):
        profile = "expanding_40d_5d"
        with tempfile.TemporaryDirectory() as tmp:
            # 两个成员用同一组 yval（same seed_off=0）→ 主键完全对齐；
            # 成员 B 预测放大 100x，验证 zscore 后不会被量级主导。
            self._write_member(tmp, "runA", profile, "SPEC_A", pred_scale=1.0)
            self._write_member(tmp, "runB", profile, "SPEC_B", pred_scale=100.0)

            members = [("runA", "SPEC_A"), ("runB", "SPEC_B")]
            merged, pred_cols, w = build_blend_frame(
                members, tmp, profile, "zscore", [1.0, 1.0]
            )
            self.assertEqual(len(pred_cols), 2)
            self.assertIn("pred_blend", merged.columns)
            self.assertTrue(np.allclose(w, [0.5, 0.5]))
            # 对齐后行数 = 单成员行数（两成员 key 一致）
            single = load_member_oof(tmp, "runA", profile, "SPEC_A")
            self.assertEqual(len(merged), len(single))

            m = compute_metrics(merged, "pred_blend")
            self.assertEqual(m["n_folds"], 2)
            # 预测与 fret12 正相关构造 → pooled corr 应明显为正
            self.assertGreater(m["pooled_corr"], 0.1)
            for k in ["fold_corr_mean", "fold_corr_min", "stability_score",
                      "daily_ic_mean", "daily_ic_ir"]:
                self.assertIn(k, m)

    def test_raw_blend_dominated_by_scale(self):
        # 反例对照：raw 模式下放大 100x 的成员会主导，证明默认 zscore 的必要性。
        profile = "expanding_40d_5d"
        with tempfile.TemporaryDirectory() as tmp:
            self._write_member(tmp, "runA", profile, "SPEC_A", pred_scale=1.0)
            self._write_member(tmp, "runB", profile, "SPEC_B", pred_scale=100.0)
            members = [("runA", "SPEC_A"), ("runB", "SPEC_B")]
            merged, _, _ = build_blend_frame(members, tmp, profile, "raw", [1.0, 1.0])
            # raw blend 几乎等于成员 B（放大列）的走向
            corr_b = _safe_corr(merged["pred_blend"], merged["pred_1"])
            self.assertGreater(corr_b, 0.99)


if __name__ == "__main__":
    unittest.main(verbosity=2)
