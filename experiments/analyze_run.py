#!/usr/bin/env python3
"""
多角度结果分析脚本（落实 AGENTS §4.6/§4.7：指标是诊断仪表盘，不是自动判卷）。

给定一个 eval_protocol run 的 run-id 与基线 spec，逐 profile、逐候选打印一组镜头：
- pooled `val_corr` 均值 delta（对齐老师评分的"期望分"代理；老师 = 全行 pooled Pearson）
- 配对每折 Δval（候选减基线、同折比）的 均值 / 标准误 / t / 折向 → 抓"是不是真的"
- 训练-验证 gap 变化 → 抓过拟合
- corr_min（最坏折）、val_corr std → 看鲁棒性/最坏情况
- daily_corr_mean delta → 选股 vs 大盘

用法：
  PYTHONPATH=src python experiments/analyze_run.py --run-id <RUN_ID> [--baseline <SPEC_ID>]

这是只读分析，不改任何结果；输出供人（现在是 Claude，带专业判断）观察后下结论，
绝不能只看一个标签自动判卷。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# 仓库根目录（本脚本在 experiments/ 下）
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE = "R02_ridge_legacy_plus_norm_core"


def build_arg_parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""
    p = argparse.ArgumentParser(description="eval_protocol run 的多角度诊断分析")
    p.add_argument("--run-id", required=True, help="results/eval_protocol/<run-id>")
    p.add_argument("--baseline", default=DEFAULT_BASELINE, help="基线 spec id")
    return p


def _fmt_signed(x: float, nd: int = 4) -> str:
    """带正负号的定点格式。"""
    return f"{x:+.{nd}f}"


def analyze(run_id: str, baseline: str) -> None:
    """读取一个 run 的 fold 级与汇总结果，逐 profile/候选打印诊断镜头。"""
    run_dir = ROOT / "results" / "eval_protocol" / run_id
    fm_path = run_dir / "fold_metrics.csv"
    lb_path = run_dir / "leaderboard.csv"

    if not fm_path.exists():
        print(f"[analyze] 未找到 fold_metrics：{fm_path}（run 可能还没产出/已死）")
        return

    fm = pd.read_csv(fm_path)
    print(f"==== run_id={run_id}  baseline={baseline} ====")

    # 汇总层（若 leaderboard 已生成）：打印官方 protocol 指标与分诊标签，仅作参考
    if lb_path.exists():
        lb = pd.read_csv(lb_path)
        cols = [c for c in [
            "experiment_id", "protocol_corr_mean", "protocol_stability_score",
            "protocol_corr_min", "baseline_delta_corr", "decision", "reason",
        ] if c in lb.columns]
        print("\n[汇总 leaderboard（分诊参考，非判决）]")
        with pd.option_context("display.width", 200, "display.max_columns", 50):
            print(lb[cols].to_string(index=False))
    else:
        print("\n[leaderboard 尚未生成 → run 未正常收尾，只看 fold 级]")

    if baseline not in set(fm["experiment_id"]):
        print(f"\n[analyze] fold_metrics 里没有基线 {baseline}，无法算配对 delta。")
        return

    cands = [e for e in fm["experiment_id"].unique() if e != baseline]

    # 逐 profile 打印配对诊断
    for prof in fm["profile_name"].unique():
        sub = fm[fm["profile_name"] == prof]
        b = sub[sub["experiment_id"] == baseline].set_index("fold_id")
        if b.empty:
            continue
        bv = b["val_corr"]
        n = len(bv)
        se_b = bv.std() / np.sqrt(n) if n > 1 else float("nan")
        print(
            f"\n[{prof}] 折数={n}  基线 val_corr: mean={_fmt_signed(bv.mean())} "
            f"std={bv.std():.4f} SE={se_b:.4f} min={bv.min():+.4f}"
        )
        print(
            f"   {'候选':32s} {'Δval':>9s} {'ΔSE':>7s} {'t':>6s} "
            f"{'折向':>8s} {'Δgap(tr-val)':>13s} {'Δcorr_min':>10s} {'ΔdailyIC':>9s}"
        )
        for c in cands:
            cc = sub[sub["experiment_id"] == c].set_index("fold_id")
            if cc.empty:
                continue
            d = (cc["val_corr"] - bv).dropna()
            if len(d) == 0:
                continue
            se = d.std() / np.sqrt(len(d)) if len(d) > 1 else float("nan")
            t = d.mean() / se if se and se > 0 else float("nan")
            signs = f"{int((d > 0).sum())}+/{int((d < 0).sum())}-"
            dgap = float("nan")
            if "train_val_corr_gap" in cc.columns:
                dgap = (cc["train_val_corr_gap"] - b["train_val_corr_gap"]).dropna().mean()
            dmin = cc["val_corr"].min() - bv.min()
            ddic = float("nan")
            if "daily_corr_mean" in cc.columns:
                ddic = (cc["daily_corr_mean"] - b["daily_corr_mean"]).dropna().mean()
            print(
                f"   {c[:32]:32s} {_fmt_signed(d.mean()):>9s} {se:>7.4f} {t:>+6.2f} "
                f"{signs:>8s} {_fmt_signed(dgap):>13s} {_fmt_signed(dmin):>10s} {_fmt_signed(ddic):>9s}"
            )

    print(
        "\n[读法] Δval 的 t 小（|t|<2）或折向对半 → 与噪声不可分；"
        "Δgap 明显为正 → 过拟合（训练吃进去、验证没传导）；"
        "Δcorr_min 为负 → 制造了更坏的折。结论由人看完综合判断，别只认标签。"
    )


def main() -> None:
    """入口。"""
    args = build_arg_parser().parse_args()
    analyze(args.run_id, args.baseline)


if __name__ == "__main__":
    main()
