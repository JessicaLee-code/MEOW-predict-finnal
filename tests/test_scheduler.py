# -*- coding: utf-8 -*-
"""
scheduler 调度层单元测试。

覆盖：
  - #17 heavy profile 必须按成本均衡切组，而不是连续切块
  - 均衡切组后两组总成本应接近，避免 expanding 后段全部堆到同一 worker
"""

import os
import sys
import unittest
from dataclasses import dataclass

# 把 src/ 注入路径，保证可以直接导入 scheduler 模块。
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from scheduler import _build_cost_balanced_fold_groups  # noqa: E402


@dataclass
class DummyFold:
    """测试用 fold：只保留调度切组需要的最小字段。"""

    fold_id: int
    train_dates: tuple
    val_dates: tuple


class TestBalancedHeavyFoldGroups(unittest.TestCase):
    """#17：heavy profile 不允许继续使用连续切块。"""

    def test_groups_are_cost_balanced_and_non_contiguous(self):
        # 人工构造一个“fold 越靠后训练窗口越长”的场景，
        # 模拟 expanding/long profile 的真实成本分布。
        folds = [
            DummyFold(
                fold_id=fold_id,
                train_dates=tuple(range(fold_id + 1)),
                val_dates=(1000 + fold_id,),
            )
            for fold_id in range(6)
        ]

        groups = _build_cost_balanced_fold_groups(
            "expanding_40d_5d",
            folds,
            n_groups=2,
        )

        self.assertEqual(len(groups), 2)

        # 逐组回收 fold id 与“训练窗口长度成本”。
        group_fold_ids = []
        group_costs = []
        for group in groups:
            fold_ids = [meta.fold_id for meta in group.fold_metas]
            group_fold_ids.append(fold_ids)
            group_costs.append(sum(len(meta.train_dates) for meta in group.fold_metas))

        # 至少有一组不是连续切片，证明已经不是旧逻辑的 [0..k][k+1..n]。
        self.assertTrue(
            any(
                any((curr - prev) > 1 for prev, curr in zip(fold_ids, fold_ids[1:]))
                for fold_ids in group_fold_ids
            )
        )

        # 两组总成本应接近；这里用一个宽松但足以挡回退的阈值。
        self.assertLessEqual(abs(group_costs[0] - group_costs[1]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
