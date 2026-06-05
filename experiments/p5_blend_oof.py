"""
P5 加权融合评测器（OOF blend）

读取若干已落盘的逐行 OOF 验证预测（由 p0_eval_protocol.py --dump-oof 产出），
按 (fold_id, date, symbol, interval) 对齐 → 加权平均 → 在同一套 fold 上
用双镜头口径评测融合后的预测：
  - pooled corr：所有验证行整体相关（对齐老师 pooled corr 的代理指标）
  - per-fold corr：逐折相关的均值 / 最坏折 / 稳定度（minimax 镜头）
  - daily IC：逐日截面相关的均值 / 标准差 / IR

设计要点：
  - **默认 zscore 融合**：Ridge 与 LightGBM 的原始输出方差量级不同，直接对原始预测做
    加权和会被方差大的模型主导。先把每个成员在 pooled 行上标准化到零均值单位方差，
    再加权平均，等价于“等信息量”融合。需要原始口径时用 --blend-mode raw；
    需要纯排序融合时用 --blend-mode rank。
  - 同时打印每个成员单独的 pooled / fold / daily 指标作参照，便于判断“融合是否真增益”。

用法：
  PYTHONPATH=src python experiments/p5_blend_oof.py \
    --profile expanding_40d_5d \
    --member 20260528_p43_gate_x1:X1_R02_plus_ofi_safe_condmom_interaction \
    --member 20260528_p43_gate_lgbm_d4:M_lgbm_d4 \
    --member 20260528_p43_gate_tree_d8:M_tree_d8 \
    [--weights 1,1,1] [--blend-mode zscore] \
    [--out results/eval_protocol/<blend_id>/blend_summary.csv]

成员写法：<run_id>:<experiment_id>，对应文件
  <oof-root>/<run_id>/oof/<profile>__<experiment_id>.h5
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OOF_ROOT = str(PROJ_ROOT / "results" / "eval_protocol")

# OOF 行的对齐主键（不含 fret12/pred）
KEY_COLS = ["fold_id", "date", "symbol", "interval"]


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """零方差/长度不足时回退 0，避免 np.corrcoef 报 nan 警告。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _zscore(x: np.ndarray) -> np.ndarray:
    """整体标准化到零均值单位方差；零方差时原样返回（全 0 不致命）。"""
    x = np.asarray(x, dtype=np.float64)
    sd = np.std(x)
    if sd < 1e-12:
        return x - np.mean(x)
    return (x - np.mean(x)) / sd


def parse_member(token: str):
    """解析 <run_id>:<experiment_id> → (run_id, experiment_id)。"""
    parts = token.split(":")
    if len(parts) != 2:
        raise ValueError(f"--member 格式应为 <run_id>:<experiment_id>，收到: {token}")
    run_id, exp_id = parts[0].strip(), parts[1].strip()
    if not run_id or not exp_id:
        raise ValueError(f"--member 解析为空: {token}")
    return run_id, exp_id


def load_member_oof(oof_root: str, run_id: str, profile: str, exp_id: str) -> pd.DataFrame:
    """读取单个成员的 OOF；列 = KEY_COLS + fret12 + pred。"""
    safe = f"{profile}__{exp_id}".replace("/", "_")
    path = os.path.join(oof_root, run_id, "oof", f"{safe}.h5")
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 OOF 文件: {path}（确认该 run 用了 --dump-oof）")
    df = pd.read_hdf(path, key="oof").reset_index(drop=True)
    need = set(KEY_COLS + ["fret12", "pred"])
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path} 缺列: {missing}")
    return df[KEY_COLS + ["fret12", "pred"]].copy()


def compute_metrics(frame: pd.DataFrame, pred_col: str) -> dict:
    """对单列预测算 pooled / per-fold / daily IC 三镜头指标。"""
    y = frame["fret12"].to_numpy(dtype=np.float64)
    p = frame[pred_col].to_numpy(dtype=np.float64)

    pooled = _safe_corr(p, y)

    fold_corrs = []
    for _, g in frame.groupby("fold_id", sort=True):
        fold_corrs.append(_safe_corr(g[pred_col].to_numpy(), g["fret12"].to_numpy()))
    fold_corrs = np.asarray(fold_corrs, dtype=np.float64)

    daily_corrs = []
    for _, g in frame.groupby("date", sort=True):
        daily_corrs.append(_safe_corr(g[pred_col].to_numpy(), g["fret12"].to_numpy()))
    daily_corrs = np.asarray(daily_corrs, dtype=np.float64)

    fold_mean = float(np.mean(fold_corrs)) if fold_corrs.size else float("nan")
    fold_std = float(np.std(fold_corrs, ddof=0)) if fold_corrs.size > 1 else 0.0
    fold_min = float(np.min(fold_corrs)) if fold_corrs.size else float("nan")
    daily_mean = float(np.mean(daily_corrs)) if daily_corrs.size else float("nan")
    daily_std = float(np.std(daily_corrs, ddof=0)) if daily_corrs.size > 1 else 0.0

    return {
        "pooled_corr": pooled,
        "fold_corr_mean": fold_mean,
        "fold_corr_std": fold_std,
        "fold_corr_min": fold_min,
        "fold_positive_rate": float(np.mean(fold_corrs > 0)) if fold_corrs.size else float("nan"),
        "stability_score": fold_mean - 0.7 * fold_std,
        "n_folds": int(fold_corrs.size),
        "daily_ic_mean": daily_mean,
        "daily_ic_std": daily_std,
        "daily_ic_ir": (daily_mean / daily_std) if daily_std > 1e-12 else float("nan"),
        "n_rows": int(len(frame)),
    }


def build_blend_frame(members, oof_root, profile, blend_mode, weights):
    """读取所有成员 → 按主键对齐 → 生成各成员预测列 + blended 列。"""
    merged = None
    pred_cols = []
    for i, (run_id, exp_id) in enumerate(members):
        df = load_member_oof(oof_root, run_id, profile, exp_id)
        col = f"pred_{i}"
        df = df.rename(columns={"pred": col})
        if merged is None:
            merged = df
        else:
            # 在主键 + fret12 上做 inner 对齐；fret12 应跨成员一致（同一批验证行）。
            merged = merged.merge(df, on=KEY_COLS + ["fret12"], how="inner")
        pred_cols.append(col)

    if merged is None or merged.empty:
        raise RuntimeError("对齐后无任何公共行，检查各成员是否跑的同一 profile / 同一 fold 区间")

    # 各成员预测转换：zscore（默认）/ raw / rank
    transformed = []
    for col in pred_cols:
        x = merged[col].to_numpy(dtype=np.float64)
        if blend_mode == "zscore":
            transformed.append(_zscore(x))
        elif blend_mode == "rank":
            transformed.append(_zscore(pd.Series(x).rank(pct=True).to_numpy()))
        elif blend_mode == "raw":
            transformed.append(x)
        else:
            raise ValueError(f"未知 blend-mode: {blend_mode}")

    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    blended = np.zeros(len(merged), dtype=np.float64)
    for wi, t in zip(w, transformed):
        blended += wi * t
    merged["pred_blend"] = blended
    return merged, pred_cols, w


def main():
    parser = argparse.ArgumentParser(description="P5 OOF 加权融合评测")
    parser.add_argument("--profile", default="expanding_40d_5d", help="OOF 来源 profile（默认 expanding_40d_5d）")
    parser.add_argument("--member", action="append", required=True,
                        help="成员 <run_id>:<experiment_id>，可重复多次")
    parser.add_argument("--weights", default=None,
                        help="逗号分隔权重（与 --member 顺序一致），默认等权")
    parser.add_argument("--blend-mode", default="zscore", choices=["zscore", "raw", "rank"],
                        help="融合前各成员预测变换：zscore（默认，等信息量）/ raw / rank")
    parser.add_argument("--oof-root", default=DEFAULT_OOF_ROOT, help="OOF 根目录（默认 results/eval_protocol）")
    parser.add_argument("--out", default=None, help="可选：把对比表写到该 CSV 路径")
    args = parser.parse_args()

    members = [parse_member(m) for m in args.member]
    if args.weights:
        weights = [float(x) for x in args.weights.split(",")]
        if len(weights) != len(members):
            raise ValueError(f"--weights 个数 {len(weights)} 与成员数 {len(members)} 不一致")
    else:
        weights = [1.0] * len(members)

    print(f"[P5] profile={args.profile} blend_mode={args.blend_mode}")
    print(f"[P5] 成员（共 {len(members)}）:")
    for (run_id, exp_id), w in zip(members, weights):
        print(f"    - {exp_id:50s} weight={w:.3f}  ({run_id})")

    merged, pred_cols, norm_w = build_blend_frame(
        members, args.oof_root, args.profile, args.blend_mode, weights
    )
    print(f"[P5] 对齐后公共行数: {len(merged)}, fold 数: {merged['fold_id'].nunique()}")

    # 逐成员单独指标（用原始 pred 列，scale 不影响单列 corr）
    rows = []
    for (run_id, exp_id), col in zip(members, pred_cols):
        m = compute_metrics(merged, col)
        m["name"] = exp_id
        m["role"] = "member"
        rows.append(m)
    # 融合指标
    mb = compute_metrics(merged, "pred_blend")
    mb["name"] = f"BLEND[{args.blend_mode}] w={','.join(f'{x:.2f}' for x in norm_w)}"
    mb["role"] = "blend"
    rows.append(mb)

    out_df = pd.DataFrame(rows)[[
        "role", "name", "pooled_corr",
        "fold_corr_mean", "fold_corr_min", "fold_corr_std", "stability_score",
        "fold_positive_rate", "daily_ic_mean", "daily_ic_ir", "n_folds", "n_rows",
    ]]

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    print("\n" + "=" * 100)
    print("融合对比（pooled=整体相关；fold_*=逐折双镜头；daily_ic_*=逐日截面）")
    print("=" * 100)
    print(out_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        out_df.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"\n[P5] 已写出对比表: {args.out}")


if __name__ == "__main__":
    main()
