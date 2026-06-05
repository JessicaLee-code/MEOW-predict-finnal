# -*- coding: utf-8 -*-
"""
P0.5：锁定标准 ridge alpha 与训练标签 winsorize 口径。

目标：
  1. 只用当前基线 R02（禁止混入其它 spec）
  2. 只跑 short + medium（AGENTS §7.7 的一次性标定例外）
  3. 在 alpha × winsorize 组合上批量扫，统一输出汇总表

设计取舍：
  - 这里不重建特征，只复用现有 feature store 产物
  - 这里不尝试自动“拍板”平台中心，只负责把结果收集完整；
    真正锁值仍由本轮 agent 结合结果人工写回文档与代码默认值
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT / "src"))

from eval_protocol import BASELINE_ID, EvaluationProtocolRunner, ROLLING_PROFILES, ALL_SPECS  # noqa: E402
from experiment_runner import ExperimentRunner  # noqa: E402


DATA_DIR = str(PROJ_ROOT / "data")
OUTPUT_DIR = PROJ_ROOT / "results" / "p05_alpha_winsorize"
FEATURE_DIR = str(PROJ_ROOT / "data" / "features")

ROLLING_START = 20230601
ROLLING_END = 20231031

ALPHAS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
WINSOR_OPTIONS = [
    {
        "label": "clip_p005_p995",
        "config": {"enabled": True, "lower_quantile": 0.005, "upper_quantile": 0.995},
    },
    {
        "label": "clip_p01_p99",
        "config": {"enabled": True, "lower_quantile": 0.01, "upper_quantile": 0.99},
    },
    {
        "label": "clip_off",
        "config": {"enabled": False, "lower_quantile": 0.005, "upper_quantile": 0.995},
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="P0.5 ridge alpha + winsorize 扫描")
    parser.add_argument("--h5dir", type=str, default=DATA_DIR, help="原始 h5 数据目录")
    parser.add_argument("--feature-dir", type=str, default=FEATURE_DIR, help="特征缓存目录")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR), help="扫描结果输出目录")
    parser.add_argument("--n-workers", type=int, default=4, help="并行 worker 数（short+medium 默认 4）")
    parser.add_argument("--run-id", type=str, default=None, help="固定本次扫描 run_id")
    parser.add_argument("--max-folds", type=int, default=None, help="限制每个 profile 的 fold 数（脚本冒烟用）")
    parser.add_argument(
        "--alphas",
        nargs="*",
        type=float,
        default=None,
        help="只跑指定 alpha 子集（默认跑内置全量列表）",
    )
    parser.add_argument(
        "--winsor-labels",
        nargs="*",
        default=None,
        help="只跑指定 winsor 方案标签子集（默认跑内置全量列表）",
    )
    return parser.parse_args()


def resolve_baseline_spec():
    """只解析当前唯一允许进入本次扫描的基线 spec。"""
    for spec in ALL_SPECS:
        if spec["experiment_id"] == BASELINE_ID:
            return spec
    raise ValueError(f"未找到 baseline spec: {BASELINE_ID}")


def resolve_profiles():
    """按 AGENTS §7.7 只返回 short + medium。"""
    profile_map = {profile.profile_name: profile for profile in ROLLING_PROFILES}
    return [
        profile_map["short_8d_2d"],
        profile_map["medium_20d_5d"],
    ]


def run_single_combo(
    *,
    h5dir: str,
    feature_dir: str,
    output_dir: Path,
    n_workers: int,
    max_folds: int | None,
    combo_run_id: str,
    ridge_alpha: float,
    winsor_option: Dict[str, object],
    baseline_spec: Dict[str, object],
    profiles,
) -> Dict[str, object]:
    """
    运行单个 alpha × winsorize 组合，并提取 leaderboard 唯一一行汇总。

    这里故意每个组合各跑一个独立 run_id：
    - 便于回看 manifest / config / fold_metrics
    - 即使中途某个组合失败，已完成组合的结果也不会丢
    """
    runner = ExperimentRunner(
        h5dir=h5dir,
        feature_dir=feature_dir,
        target_winsorize_config=winsor_option["config"],
        feature_dtype="float32",
        ridge_alpha=ridge_alpha,
    )
    protocol = EvaluationProtocolRunner(runner)
    result = protocol.run_full_protocol(
        rolling_start=ROLLING_START,
        rolling_end=ROLLING_END,
        specs=[baseline_spec],
        profiles=profiles,
        max_folds=max_folds,
        baseline_id=baseline_spec["experiment_id"],
        n_workers=n_workers,
        output_dir=str(output_dir),
        run_id=combo_run_id,
    )
    leaderboard = result["leaderboard"]
    if leaderboard.empty:
        raise RuntimeError(f"{combo_run_id} 未产出 leaderboard")
    row = leaderboard.iloc[0].to_dict()
    row["scan_run_id"] = combo_run_id
    row["ridge_alpha"] = float(ridge_alpha)
    row["winsor_label"] = str(winsor_option["label"])
    row["target_winsorize_enabled"] = bool(winsor_option["config"]["enabled"])
    row["target_winsorize_lower_q"] = float(winsor_option["config"]["lower_quantile"])
    row["target_winsorize_upper_q"] = float(winsor_option["config"]["upper_quantile"])
    return row


def main():
    args = parse_args()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_spec = resolve_baseline_spec()
    profiles = resolve_profiles()
    selected_alphas = args.alphas or ALPHAS
    selected_winsor_options = [
        option
        for option in WINSOR_OPTIONS
        if args.winsor_labels is None or option["label"] in args.winsor_labels
    ]
    if not selected_winsor_options:
        raise ValueError("未匹配到任何 winsor 方案，请检查 --winsor-labels")

    manifest = {
        "run_id": run_id,
        "baseline_spec_id": baseline_spec["experiment_id"],
        "profiles": [profile.profile_name for profile in profiles],
        "alphas": selected_alphas,
        "winsor_options": selected_winsor_options,
        "feature_dtype": "float32",
        "n_workers": args.n_workers,
        "max_folds": args.max_folds,
        "note": "P0.5 一次性扫描：只跑 R02 + short/medium，不重建特征。",
    }
    with open(output_dir / f"{run_id}_manifest.json", "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)

    summary_rows: List[Dict[str, object]] = []
    for winsor_option in selected_winsor_options:
        for ridge_alpha in selected_alphas:
            combo_run_id = f"{run_id}_{winsor_option['label']}_a{str(ridge_alpha).replace('.', 'p')}"
            print(
                "[P0.5] running combo: winsor={winsor}, alpha={alpha}, run_id={run_id}".format(
                    winsor=winsor_option["label"],
                    alpha=ridge_alpha,
                    run_id=combo_run_id,
                )
            )
            row = run_single_combo(
                h5dir=args.h5dir,
                feature_dir=args.feature_dir,
                output_dir=output_dir,
                n_workers=args.n_workers,
                max_folds=args.max_folds,
                combo_run_id=combo_run_id,
                ridge_alpha=ridge_alpha,
                winsor_option=winsor_option,
                baseline_spec=baseline_spec,
                profiles=profiles,
            )
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            ["protocol_corr_mean", "protocol_daily_ic_ir", "protocol_stability_score"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    summary_path = output_dir / f"{run_id}_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[P0.5] 扫描完成，汇总已写入: {summary_path}")
    if not summary_df.empty:
        display_cols = [
            "winsor_label",
            "ridge_alpha",
            "protocol_corr_mean",
            "protocol_stability_score",
            "protocol_daily_ic_mean",
            "protocol_daily_ic_ir",
            "protocol_corr_min",
            "scan_run_id",
        ]
        display_cols = [col for col in display_cols if col in summary_df.columns]
        print(summary_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
