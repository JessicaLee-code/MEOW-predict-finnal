# -*- coding: utf-8 -*-
"""
P4 树特征重要性扫描（回答 Fork A 的"反偏颇"那一问）。

背景（2026-05-27 拍板）：
  P4 树特征集走"大集让树自筛"路线（`eval_protocol.P4_TREE_GROUPS`），第一版**含手工
  交互**（trade_pressure_x_* / lagret_x_*），不预剪。本脚本在 Dev 训练窗上 fit 一棵浅
  ExtraTrees（与初筛同款 `tree_shallow`），取 `feature_importances_` 排序 + 按特征族聚
  合，一次回答两件事：
    1. 树是否真用得上手工交互？（交互族重要性 ≈ 0 → 树确实不需要，可在下一版剪掉）
    2. 反偏颇/反错杀：线性被基线吃掉、被低估的原始族（ofi / trade 原始信号、Q3 交互源）
       在树里是否冒头？（冒头 = 线性错杀，对树有价值）

定位：
  - 这是**诊断脚本**，只 fit 一棵代表性的树、不做 fold 矩阵、不碰任何 holdout；
    与初筛 leaderboard 互补、互不阻塞。
  - 不重建特征，只复用现有 feature store 产物（与其它 experiments/*.py 一致）。
  - winsorize 沿用 runner 锁定默认（训练标签 P1/P99 开）；重要性是特征侧排序，与之无关。

跑法（§5.1 硬约束：挂内存看门狗 + 日志写 logs/）：
  PYTHONPATH=src caffeinate -i python experiments/run_with_memory_guard.py \
    --rss-limit-gb 9 --rss-hard-limit-gb 11 \
    --log-file logs/memory_guard_20260527_p4_tree_importance_v1.log \
    -- python experiments/p4_tree_importance.py \
       --run-id 20260527_p4_tree_importance_v1 \
       > logs/20260527_p4_tree_importance_v1.log 2>&1
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

from eval_protocol import P4_TREE_GROUPS  # noqa: E402
from experiment_runner import ExperimentRunner  # noqa: E402


DATA_DIR = str(PROJ_ROOT / "data")
FEATURE_DIR = str(PROJ_ROOT / "data" / "features")
OUTPUT_DIR = PROJ_ROOT / "results" / "p4_tree_importance"

# 默认训练窗：取 Dev 后段 ~2 个月（与初筛 long_40d_5d 训练窗量级相当，够稳又不撑爆 16GB）。
# 想用全 Dev（20230601–20231031）做更稳的重要性可经 --train-start/--train-end 覆盖。
DEFAULT_TRAIN_START = 20230901
DEFAULT_TRAIN_END = 20231031


def classify_family(col: str) -> str:
    """把特征列归到一个族（用于按族聚合重要性）。

    顺序敏感：手工交互优先（这是 Fork A 的核心问题）；cross-z/cross-rank 要在
    ofi/trade 之前判（`ofi_total_cs_z` 也含 'ofi'，但它属"截面归一"族、是树自己做不出的）。
    """
    if "_x_" in col:
        return "interaction(手工交互)"
    if col.endswith("_cs_rank"):
        return "cross_rank(截面排名)"
    if col.endswith("_cs_z"):
        return "cross_z(截面归一)"
    if "regime" in col or col.startswith("state_"):
        return "regime"
    if col.startswith("lagret") or "_lag_" in col or col.startswith("lag"):
        return "momentum/lag(动量-滞后)"
    if "_rm" in col or "_rs" in col or "roll" in col:
        return "roll(滚动统计)"
    if "ofi" in col:
        return "ofi"
    if col.startswith("trade") or col.startswith("avg_trade") or "trade_" in col:
        return "trade(成交冲击)"
    if "patch" in col:
        return "patch"
    return "base/legacy(原始基础)"


def parse_args():
    parser = argparse.ArgumentParser(description="P4 树特征重要性扫描（Fork A 反偏颇诊断）")
    parser.add_argument("--h5dir", type=str, default=DATA_DIR, help="原始 h5 数据目录")
    parser.add_argument("--feature-dir", type=str, default=FEATURE_DIR, help="特征缓存目录")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR), help="结果输出目录")
    parser.add_argument("--run-id", type=str, default=None, help="固定本次 run_id")
    parser.add_argument("--train-start", type=int, default=DEFAULT_TRAIN_START, help="训练窗起始日 (YYYYMMDD)")
    parser.add_argument("--train-end", type=int, default=DEFAULT_TRAIN_END, help="训练窗结束日 (YYYYMMDD)")
    parser.add_argument("--max-depth", type=int, default=5, help="浅树深度（默认 5，初筛网格中心）")
    parser.add_argument("--n-estimators", type=int, default=300, help="树棵数（默认 300）")
    parser.add_argument("--top", type=int, default=40, help="终端打印 top-N 特征")
    return parser.parse_args()


def main():
    args = parse_args()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 构造 runner（内建 calendar + FeatureLoader；winsorize/alpha 走锁定默认）。
    runner = ExperimentRunner(
        h5dir=args.h5dir,
        feature_dir=args.feature_dir,
        feature_dtype="float32",
    )

    # 2. 加载树大集训练数据。
    dates = runner.calendar.range(args.train_start, args.train_end)
    print(f"[P4-imp] 训练窗 {args.train_start}–{args.train_end}（{len(dates)} 交易日）；特征族 = {P4_TREE_GROUPS}")
    xtrain, ytrain = runner._load_group_split(dates, groups=P4_TREE_GROUPS)
    xtrain = xtrain.loc[:, ~xtrain.columns.duplicated()].copy()
    print(f"[P4-imp] 训练 shape = {xtrain.shape}（行=样本，列含 meta）")

    # 3. fit 浅 ExtraTrees（与初筛 tree_shallow 同款，仅按 args 覆盖 depth/棵数）。
    model_params = {"max_depth": args.max_depth, "n_estimators": args.n_estimators}
    print(f"[P4-imp] fit tree_shallow，model_params={model_params} …")
    model, feature_cols, _ = runner.fit_model(
        "tree_shallow", xtrain, ytrain, target_mode="raw", model_params=model_params
    )

    # 4. 取重要性、排序、标族。
    imp = runner._extract_tree_importance(model, feature_cols)
    if imp is None:
        raise RuntimeError("未能从模型取出 feature_importances_（模型类型不符？）")
    imp = imp.rename(columns={"coef": "importance"}).drop(columns=["abs_coef"])
    imp["family"] = imp["feature"].map(classify_family)
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
    imp["rank"] = imp.index + 1

    # 5. 按族聚合（重要性求和 + 计数 + top1），直接回答"哪个族撑分 / 交互是否为零"。
    family_agg = (
        imp.groupby("family")
        .agg(importance_sum=("importance", "sum"), n_features=("feature", "count"), top_importance=("importance", "max"))
        .sort_values("importance_sum", ascending=False)
        .reset_index()
    )
    family_agg["importance_share"] = family_agg["importance_sum"] / family_agg["importance_sum"].sum()

    # 6. 落盘 + 打印。
    feat_csv = output_dir / f"{run_id}_feature_importance.csv"
    fam_csv = output_dir / f"{run_id}_family_importance.csv"
    imp.to_csv(feat_csv, index=False, encoding="utf-8-sig")
    family_agg.to_csv(fam_csv, index=False, encoding="utf-8-sig")
    manifest = {
        "run_id": run_id,
        "train_start": args.train_start,
        "train_end": args.train_end,
        "n_dates": len(dates),
        "n_rows": int(xtrain.shape[0]),
        "n_features": len(feature_cols),
        "model": "tree_shallow",
        "model_params": model_params,
        "groups": P4_TREE_GROUPS,
        "note": "P4 Fork A 反偏颇诊断：树是否用得上手工交互 + 线性被低估的原始族是否冒头。",
    }
    with open(output_dir / f"{run_id}_manifest.json", "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)

    print("\n========== 按族聚合重要性（占比降序）==========")
    print(family_agg.to_string(index=False))
    print(f"\n========== Top-{args.top} 特征 ==========")
    print(imp.head(args.top)[["rank", "feature", "importance", "family"]].to_string(index=False))
    print(f"\n[P4-imp] 明细已写：{feat_csv}")
    print(f"[P4-imp] 按族汇总已写：{fam_csv}")


if __name__ == "__main__":
    main()
