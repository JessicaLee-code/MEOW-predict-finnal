"""
P0: 扩展 rolling 评测 + 训练窗口敏感性分析

目标：
  1. 以 R02_ridge_legacy_plus_norm_core 为基线，在标准三层评测体系下复现
  2. 扫描 max_train_days ∈ {4, 8, 10, 20, 40, 80, None(expanding)} 找最优训练窗口
  3. 输出统一指标：rolling_corr_mean / std / min，stability_score

运行方式：
  cd MEOW--predict
  PYTHONPATH=src python experiments/p0_rolling_audit.py

输出：
  results/p0_rolling_audit_results.csv
  results/p0_rolling_audit_folds.csv
"""

import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT / "src"))

from experiment_runner import ExperimentRunner, SplitConfig


DATA_DIR = str(PROJ_ROOT / "data")
OUTPUT_CSV = str(PROJ_ROOT / "results" / "p0_rolling_audit_results.csv")
OUTPUT_FOLDS = str(PROJ_ROOT / "results" / "p0_rolling_audit_folds.csv")

# 三层评测：第一层 rolling 作为内部选模型主依据
SPLIT_CONFIG = SplitConfig(
    train_start=20230601,
    train_end=20231031,
    val_start=20231101,   # 第二层：11月 holdout
    val_end=20231130,
    test_start=20231201,  # 第三层：12月 final holdout（尽量少看）
    test_end=20231229,
)

# R02 基线 spec（与 experiment_runner 内部格式一致）
R02_SPEC = {
    "experiment_id": "R02_ridge_legacy_plus_norm_core",
    "type": "standard",
    "model": "ridge",
    "target_mode": "raw",
    "groups": ["legacy", "norm_core"],
    "notes": "P0 baseline: R02 backbone, training window sensitivity scan",
    "collect_coefs": False,
}

# 训练窗口扫描范围（None = expanding window）
TRAIN_WINDOWS = [4, 8, 10, 20, 40, 80, None]


def main():
    print(f"[P0] 数据目录: {DATA_DIR}")
    print(f"[P0] 输出文件: {OUTPUT_CSV}")
    print(f"[P0] 扫描训练窗口: {TRAIN_WINDOWS}")

    runner = ExperimentRunner(DATA_DIR)

    summary, fold_df = runner.run_train_window_sensitivity_suite(
        split_config=SPLIT_CONFIG,
        spec=R02_SPEC,
        train_windows=TRAIN_WINDOWS,
        train_window=80,   # 构建 rolling fold 时允许的最大 train window
        val_window=2,
        step=10,
        max_folds=None,    # 使用全部可用 fold，不再截断为 5
        embargo=1,
    )

    print("\n[P0] 训练窗口敏感性结果：")
    print(summary.to_string(index=False))

    summary.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    fold_df.to_csv(OUTPUT_FOLDS, index=False, encoding="utf-8-sig")
    print(f"\n[P0] 汇总已保存到 {OUTPUT_CSV}")
    print(f"[P0] 逐折明细已保存到 {OUTPUT_FOLDS}")


if __name__ == "__main__":
    main()
