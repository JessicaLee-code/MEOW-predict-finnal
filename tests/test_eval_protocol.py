# -*- coding: utf-8 -*-
"""
eval_protocol 评测协议单元测试。

覆盖：
  - build_leaderboard 每日 IC / IC-IR 字段（任务 #12）
  - make_decision 硬契约（任务 #10，AGENTS §4.6）

运行：
    PYTHONPATH=src python -m unittest tests.test_eval_protocol -v
  或直接：
    python tests/test_eval_protocol.py
"""

import os
import sys
import unittest

# 把 src/ 与仓库根注入路径：src 供评测模块直接 import，仓库根供 experiments 命名空间包 import
_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from eval_protocol import (  # noqa: E402
    EvaluationProtocolRunner,
    BASELINE_ID,
    make_decision,
)

from experiments.p0_eval_protocol import (  # noqa: E402
    build_daily_specs,
    RIDGE_SPECS,
)


def _decision_baseline():
    """promote 判定用的固定基线行"""
    return {
        "protocol_corr_mean": 0.050,
        "protocol_stability_score": 0.040,
        "protocol_daily_ic_mean": 0.020,
        "short_corr_mean": 0.050, "short_corr_min": 0.020,
        "long_corr_mean": 0.050, "long_corr_min": 0.020,
        "expanding_corr_mean": 0.050, "expanding_corr_min": 0.020,
    }


def _good_candidate():
    """delta_corr=+0.006，且全部 promote 附加门槛都满足的候选"""
    return {
        "protocol_corr_mean": 0.056,
        "protocol_stability_score": 0.045,
        "protocol_daily_ic_mean": 0.022,
        "short_corr_mean": 0.056, "short_corr_min": 0.022,
        "long_corr_mean": 0.052, "long_corr_min": 0.022,
        "expanding_corr_mean": 0.056, "expanding_corr_min": 0.022,
    }


def _psum_row(eid, corr_mean, daily_mean, daily_std,
              stability=None, corr_min=None):
    """构造一行 profile_summary 记录（仅含 build_leaderboard 会读到的列）"""
    return {
        "experiment_id": eid,
        "rolling_corr_mean": corr_mean,
        "rolling_corr_std": 0.01,
        "rolling_corr_min": corr_min if corr_min is not None else corr_mean - 0.02,
        "rolling_corr_median": corr_mean,
        "stability_score": stability if stability is not None else corr_mean - 0.01,
        "positive_fold_rate": 1.0,
        "n_folds": 10,
        "rolling_mse_mean": 1.0,
        "rolling_r2_mean": 0.01,
        "daily_corr_mean": daily_mean,
        "daily_corr_std": daily_std,
        "model_type": "ridge",
        "feature_set": "x",
        "target_type": "raw",
    }


class TestLeaderboardDailyIC(unittest.TestCase):
    """任务 #12：每日 IC / IC-IR 必须并进 leaderboard 主视图"""

    def _build(self):
        short = pd.DataFrame([
            _psum_row(BASELINE_ID, 0.050, 0.020, 0.040),
            _psum_row("CAND", 0.060, 0.025, 0.040),
        ])
        exp = pd.DataFrame([
            _psum_row(BASELINE_ID, 0.050, 0.020, 0.040),
            _psum_row("CAND", 0.060, 0.030, 0.050),
        ])
        return EvaluationProtocolRunner(None).build_leaderboard(
            {"short_8d_2d": short, "expanding_40d_5d": exp},
            baseline_id=BASELINE_ID,
        )

    def test_protocol_daily_ic_columns_present(self):
        lb = self._build()
        self.assertIn("protocol_daily_ic_mean", lb.columns)
        self.assertIn("protocol_daily_ic_ir", lb.columns)

    def test_per_profile_daily_ic_columns_present(self):
        lb = self._build()
        self.assertIn("short_daily_corr_mean", lb.columns)
        self.assertIn("expanding_daily_ic_ir", lb.columns)

    def test_daily_ic_ir_value(self):
        lb = self._build()
        cand = lb[lb["experiment_id"] == "CAND"].iloc[0]
        # short: daily_corr_mean=0.025 / daily_corr_std=0.040 = 0.625
        self.assertAlmostEqual(float(cand["short_daily_ic_ir"]), 0.025 / 0.040, places=6)
        # expanding: 0.030 / 0.050 = 0.600
        self.assertAlmostEqual(float(cand["expanding_daily_ic_ir"]), 0.030 / 0.050, places=6)


class TestMakeDecision(unittest.TestCase):
    """任务 #10：make_decision 硬契约（AGENTS §4.6）"""

    def test_reject_below_floor(self):
        # delta_corr = 0.001 < 0.003 → reject
        row = _good_candidate()
        row["protocol_corr_mean"] = 0.051
        decision, _ = make_decision(row, _decision_baseline())
        self.assertEqual(decision, "reject")

    def test_review_at_boundary(self):
        # 强制断言 1：delta_corr = 0.004 ∈ [0.003,0.005) 必须返回 review
        row = _good_candidate()
        row["protocol_corr_mean"] = 0.054
        decision, _ = make_decision(row, _decision_baseline())
        self.assertEqual(decision, "review")

    def test_missing_expanding_cannot_promote(self):
        # 强制断言 2：达标但缺 expanding，最高只能 review，绝不能 promote
        row = _good_candidate()
        del row["expanding_corr_mean"]
        del row["expanding_corr_min"]
        decision, _ = make_decision(row, _decision_baseline())
        self.assertNotEqual(decision, "promote")
        self.assertEqual(decision, "review")

    def test_strong_negative_fold_cannot_promote(self):
        # 强制断言 3：出现新强负折（corr_min < -0.01）必须不能 promote
        row = _good_candidate()
        row["short_corr_min"] = -0.02
        decision, _ = make_decision(row, _decision_baseline())
        self.assertNotEqual(decision, "promote")

    def test_promote_happy_path(self):
        # 正例：全门槛通过 → promote
        decision, _ = make_decision(_good_candidate(), _decision_baseline())
        self.assertEqual(decision, "promote")


class TestBuildDailySpecs(unittest.TestCase):
    """--spec-ids：daily 快车道候选选择（P1–P3 候选入口）"""

    def test_default_falls_back_to_ridge(self):
        # 不给 spec_ids → 回退默认 RIDGE_SPECS，历史行为不变
        specs = build_daily_specs(None, BASELINE_ID)
        self.assertEqual(
            [s["experiment_id"] for s in specs],
            [s["experiment_id"] for s in RIDGE_SPECS],
        )

    def test_candidates_include_baseline_first(self):
        # 给候选 → baseline 置首位，候选随后，用于同表算 delta
        specs = build_daily_specs(
            ["O1_R02_plus_ofi_raw", "O2_R02_plus_ofi_dynamic"], BASELINE_ID
        )
        ids = [s["experiment_id"] for s in specs]
        self.assertEqual(ids[0], BASELINE_ID)
        self.assertEqual(
            ids, [BASELINE_ID, "O1_R02_plus_ofi_raw", "O2_R02_plus_ofi_dynamic"]
        )

    def test_baseline_not_duplicated(self):
        # 候选里已含 baseline → 不重复
        specs = build_daily_specs([BASELINE_ID, "O1_R02_plus_ofi_raw"], BASELINE_ID)
        ids = [s["experiment_id"] for s in specs]
        self.assertEqual(ids, [BASELINE_ID, "O1_R02_plus_ofi_raw"])

    def test_unknown_id_raises(self):
        # 未知 id → 直接报错，不静默吞
        with self.assertRaises(ValueError):
            build_daily_specs(["NOT_A_REAL_SPEC"], BASELINE_ID)


if __name__ == "__main__":
    unittest.main(verbosity=2)
