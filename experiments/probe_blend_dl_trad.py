#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DL-on-raw × 传统模型 融合分析（机器1 用第二台机器推来的 DL 预测做）。

目的
----
第二台机器跑的 DL-on-raw（XSECTION_RAW 卡带，直接吃 raw 59 通道）在三个选型折 + 交付折落了
逐票预测；本脚本把它和机器1 的传统 OOS 逐票预测按 (date,symbol,interval) inner-join，
在**同一交集行**上算（**全部复刻老师 `meow/eval.py` 口径：fillna(0) 后 Pearson / R² / MSE**）：
  - 传统单独 (Pearson, R², MSE)
  - DL 单独   (Pearson, R², MSE)
  - DL↔传统去相关度 ρ（两预测之间的 Pearson）
  - 等权融合 blend_raw = 0.5*dl + 0.5*trad —— **交付口径**（零自由参数、量纲留在 fret12，保 R²/MSE）
  - blend_z = z(dl) + z(trad) —— 纯方向融合对照（对量纲差异稳健，但会破 fret12 量纲、不能直接交付）
  - 理论最优合并相关：R = sqrt((i1²+i2²-2ρ·i1·i2)/(1-ρ²))
  - 权重扫描 w·dl+(1-w)·trad 的 Pearson —— 报最优 w（**in-sample 过拟合上界、仅诊断**）
  - **seed 平均**：cert 折有 seed42+43 两个 DL 预测，额外报"两 seed 平均后的 DL/融合"——
    这是交付真正会用的形态（焊进 meow.py 后训 K seed 平均），seed 集成是免费杠杆。

数据源（2026-06-04 更新到 3 折 run）
-----------------------------------
DL = `20260604_xsection_raw_3fold_2seed`（机器2 推来）。**折号**：
  cert fold0=Aug31–Sep27、cert fold1=Sep28–Nov02、cert fold2=Nov03–Nov30、delivery fold3=Dec04–Dec29。
传统 = 机器1 dump 的**新传统**（含 rx_micro 两腿）四折预测，落 `_blend_dl_trad/`：
  trad_fold0/1/2_newfeat_preds.csv（各选型窗 OOS）+ trad_dec_newfeat_preds.csv（Dec sanity 0.0812）。
→ **四窗完全同口径**（传统腿都含 rx_micro），可直接横比"融合 vs 传统"增益是否三折全稳。

为什么三项指标都要算
--------------------
老师评分 = pooled Pearson + R² + MSE 各 1/3。只看 Pearson 会漏掉"融合后量纲是否还健康"。
DL 腿已在训练段做 OLS rescale → 量纲落在 fret12，故等权 raw 融合后量纲应仍健康；但必须把
R²/MSE 也报出来核对、不能假设。

为什么要在"同一交集行"上算
--------------------------
DL 因序列 warmup 比传统少若干行（且 DL delivery 从 Dec4 起）；inner-join 后传统/DL/融合都在
同一行集上，融合 vs 传统的增益才可直接比较。

只读 CSV、不训练、不碰 GPU。结果打印成表并落 results/dl/_blend_dl_trad/summary.json。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DL_DIR = REPO / "results/dl/20260604_xsection_raw_3fold_2seed/preds"   # 新 3 折 run
TRAD_P2_DIR = REPO / "results/dl/_p2_trad_folds"          # 旧 P2 传统（不含 rx_micro，仅兜底）
BLEND_DIR = REPO / "results/dl/_blend_dl_trad"            # 新传统四折预测落在这里
OUT_DIR = BLEND_DIR

# 每个窗口：DL 预测（可多 seed，列表第一个为 seed42）+ 新/旧传统预测路径。
# 新 3 折 run 折号：fold0=Aug31–Sep27、fold1=Sep28–Nov02、fold2=Nov03–Nov30、delivery=fold3=Dec04–Dec29。
WINDOWS = {
    "fold0_Aug31_Sep27": {
        "dl_seeds": [
            DL_DIR / "preds_sweep_cert_fold0_seed42.csv",
            DL_DIR / "preds_sweep_cert_fold0_seed43.csv",
        ],
        "trad_new": BLEND_DIR / "trad_fold0_newfeat_preds.csv",
        "trad_old": None,                       # P2 没 dump fold0，无兜底（trad_new 必在）
    },
    "fold1_Sep28_Nov02": {
        "dl_seeds": [
            DL_DIR / "preds_sweep_cert_fold1_seed42.csv",
            DL_DIR / "preds_sweep_cert_fold1_seed43.csv",
        ],
        "trad_new": BLEND_DIR / "trad_fold1_newfeat_preds.csv",
        "trad_old": TRAD_P2_DIR / "trad_preds_fold1_20230928_20231102.csv",
    },
    "fold2_Nov03_Nov30": {
        "dl_seeds": [
            DL_DIR / "preds_sweep_cert_fold2_seed42.csv",
            DL_DIR / "preds_sweep_cert_fold2_seed43.csv",
        ],
        "trad_new": BLEND_DIR / "trad_fold2_newfeat_preds.csv",
        "trad_old": TRAD_P2_DIR / "trad_preds_fold2_20231103_20231130.csv",
    },
    "delivery_Dec04_Dec29": {
        # delivery 折当前被代码写死只跑 seed42 → 这里只有单 seed（交付多 seed 是下一步代码改动）。
        "dl_seeds": [DL_DIR / "preds_sweep_delivery_fold3_seed42.csv"],
        "trad_new": BLEND_DIR / "trad_dec_newfeat_preds.csv",
        "trad_old": TRAD_P2_DIR / "trad_preds_delivery_20231204_20231229.csv",
    },
}

META = ["date", "symbol", "interval"]


def meow_metrics(p: np.ndarray, y: np.ndarray) -> tuple:
    """
    复刻老师 `meow/eval.py` 的三指标口径：
    - 先把预测/真值的 ±inf 换 NaN、再 fillna(0)（老师 eval 前置处理）。
    - Pearson = pandas .corr()（皮尔逊）。
    - R² = 1 - SS_res / var(y, ddof=1) / N（老师写法，带 ddof=1 的 var 乘 N，N 大时≈标准 R²）。
    - MSE = SS_res / N。
    返回 (pearson, r2, mse)。
    """
    p = pd.Series(np.asarray(p, dtype=np.float64)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = pd.Series(np.asarray(y, dtype=np.float64)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    n = len(y)
    pcor = float(p.corr(y))
    sse = float(((p - y) ** 2).sum())
    yvar = float(y.var())  # pandas 默认 ddof=1
    r2 = float(1.0 - sse / yvar / n) if (yvar > 0 and n > 0) else float("nan")
    mse = float(sse / n) if n > 0 else float("nan")
    return pcor, r2, mse


def zscore(x: np.ndarray) -> np.ndarray:
    """整列标准化（减均值除标准差）；std 退化时退回仅去均值。"""
    x = np.asarray(x, dtype=np.float64)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd < 1e-15:
        return x - mu
    return (x - mu) / sd


def theory_optimal(i1: float, i2: float, rho: float) -> float:
    """两去相关信号线性合并可达的相关上界 R=sqrt((i1²+i2²-2ρ i1 i2)/(1-ρ²))。"""
    denom = 1.0 - rho * rho
    if denom <= 1e-12:
        return float("nan")
    val = (i1 * i1 + i2 * i2 - 2.0 * rho * i1 * i2) / denom
    return float(np.sqrt(val)) if val > 0 else float("nan")


def weight_sweep(pdl: np.ndarray, ptr: np.ndarray, y: np.ndarray) -> dict:
    """
    扫描融合权重 w∈[0,1]（blend=w·dl+(1-w)·trad）的 Pearson，找最优 w。

    诚实声明：最优 w 是**在当窗挑出来的 in-sample 过拟合上界**，不可作为交付权重；
    交付一律用等权（w=0.5，零自由参数）。这里仅用于回答"等权离最优有多远"。
    """
    ws = np.linspace(0.0, 1.0, 21)
    best_w, best_ic = 0.5, -1.0
    curve = []
    for w in ws:
        ic, _, _ = meow_metrics(w * pdl + (1.0 - w) * ptr, y)
        curve.append((round(float(w), 2), round(float(ic), 5)))
        if ic > best_ic:
            best_ic, best_w = ic, float(w)
    return {"best_w": round(best_w, 3), "best_ic": round(best_ic, 5), "curve": curve}


def analyze_one(name: str, cfg: dict) -> dict:
    """
    对单个窗口做 inner-join 并算全套读数。

    DL 腿支持多 seed：列表第一个为 seed42（主判决用，四窗可比）；若 >1 seed 额外算
    "全 seed 平均"的 DL/融合（交付真正会用的形态）。传统腿优先新传统（含 rx_micro）。
    """
    # —— 传统腿：优先新传统，缺则回退旧 P2，记录用了哪个 —— #
    trad_new, trad_old = cfg.get("trad_new"), cfg.get("trad_old")
    if trad_new is not None and trad_new.exists():
        tr_path, trad_tag = trad_new, "new(rx_micro)"
    elif trad_old is not None and trad_old.exists():
        tr_path, trad_tag = trad_old, "old(P2,no rx_micro)"
    else:
        return {"window": name, "error": f"传统预测缺失: {trad_new} / {trad_old}"}

    dl_paths = [p for p in cfg["dl_seeds"] if p.exists()]
    if not dl_paths:
        return {"window": name, "error": f"DL 预测缺失: {cfg['dl_seeds']}"}

    # —— 读传统 + 逐 seed 读 DL，全部按 META inner-join 对齐到同一行集 —— #
    tr = pd.read_csv(tr_path, usecols=META + ["label", "pred"]).rename(
        columns={"pred": "pred_tr", "label": "label_tr"}
    )
    base = None
    seed_cols = []
    for i, p in enumerate(dl_paths):
        col = f"pred_dl_{i}"
        d = pd.read_csv(p, usecols=META + ["label", "pred"]).rename(
            columns={"pred": col, "label": "label_dl"}
        )
        seed_cols.append(col)
        if base is None:
            base = d
        else:
            d = d.drop(columns=["label_dl"])  # 各 seed label 相同，只留一份
            base = base.merge(d, on=META, how="inner")
    m = base.merge(tr, on=META, how="inner")
    n = len(m)
    if n == 0:
        return {"window": name, "error": "inner-join 交集为空（meta 对不齐）", "trad_tag": trad_tag}

    # 真值一致性核对：DL 侧 label 与传统侧 label 应是同一 fret12。
    label_max_abs_diff = float(np.nanmax(np.abs(m["label_dl"].to_numpy() - m["label_tr"].to_numpy())))
    y = m["label_dl"].to_numpy(dtype=np.float64)
    ptr = m["pred_tr"].to_numpy(dtype=np.float64)
    seed_mat = np.column_stack([m[c].to_numpy(dtype=np.float64) for c in seed_cols])
    pdl_s42 = seed_mat[:, 0]            # 第一个 seed = seed42（主判决，四窗可比）
    pdl_avg = seed_mat.mean(axis=1)     # 全 seed 平均（交付会用的形态）
    n_seeds = len(dl_paths)

    # —— 主判决（seed42 口径，四窗一致可横比） —— #
    ic_tr, r2_tr, mse_tr = meow_metrics(ptr, y)
    ic_dl, r2_dl, mse_dl = meow_metrics(pdl_s42, y)
    rho, _, _ = meow_metrics(pdl_s42, ptr)
    ic_braw, r2_braw, mse_braw = meow_metrics(0.5 * pdl_s42 + 0.5 * ptr, y)
    ic_bz, r2_bz, mse_bz = meow_metrics(zscore(pdl_s42) + zscore(ptr), y)
    theo = theory_optimal(ic_tr, ic_dl, rho)
    sweep = weight_sweep(pdl_s42, ptr, y)

    # —— seed 平均（>1 seed 才有意义；delivery 单 seed 时与 seed42 相同） —— #
    if n_seeds > 1:
        ic_dl_avg, _, _ = meow_metrics(pdl_avg, y)
        rho_avg, _, _ = meow_metrics(pdl_avg, ptr)
        ic_braw_avg, r2_braw_avg, mse_braw_avg = meow_metrics(0.5 * pdl_avg + 0.5 * ptr, y)
    else:
        ic_dl_avg, rho_avg = ic_dl, rho
        ic_braw_avg, r2_braw_avg, mse_braw_avg = ic_braw, r2_braw, mse_braw

    return {
        "window": name,
        "trad_tag": trad_tag,
        "n_join": int(n),
        "n_dl_rows": int(len(base)),
        "n_trad_rows": int(len(tr)),
        "n_dl_seeds": int(n_seeds),
        "label_max_abs_diff": label_max_abs_diff,
        # 三指标（老师口径，seed42）
        "trad": {"pearson": ic_tr, "r2": r2_tr, "mse": mse_tr},
        "dl": {"pearson": ic_dl, "r2": r2_dl, "mse": mse_dl},
        "blend_raw_mean": {"pearson": ic_braw, "r2": r2_braw, "mse": mse_braw},  # 交付口径(seed42)
        "blend_zscore": {"pearson": ic_bz, "r2": r2_bz, "mse": mse_bz},          # 仅方向对照
        # 去相关 & 增益（seed42）
        "rho_dl_trad": rho,
        "theory_optimal_pearson": theo,
        "blend_raw_pearson_vs_trad": ic_braw - ic_tr,
        # seed 平均（交付会用的形态）
        "dl_avg": {"pearson": ic_dl_avg},
        "blend_raw_mean_avg": {"pearson": ic_braw_avg, "r2": r2_braw_avg, "mse": mse_braw_avg},
        "blend_avg_pearson_vs_trad": ic_braw_avg - ic_tr,
        "rho_avg_dl_trad": rho_avg,
        # 权重扫描（诊断、过拟合上界）
        "weight_sweep": sweep,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [analyze_one(name, cfg) for name, cfg in WINDOWS.items()]

    # —— 主表：seed42 口径，四窗一致可横比（回答"三折 + 交付是否全稳正增益"） —— #
    print("\n===================== DL-on-raw × 新传统 融合(seed42,同交集行,老师口径,四窗同口径) =====================")
    print("{:<22} {:<16} {:>8} {:>8} {:>9} {:>9} {:>6} {:>9} {:>10} {:>7}".format(
        "window", "trad_leg", "ic_trad", "ic_dl", "ic_blend", "Δic", "rho", "r2_blend", "mse_blend", "best_w"))
    for r in rows:
        if "error" in r:
            print("{:<22} ERROR: {}".format(r["window"], r["error"]))
            continue
        print("{:<22} {:<16} {:>8.4f} {:>8.4f} {:>9.4f} {:>+9.4f} {:>6.3f} {:>9.5f} {:>10.2e} {:>7.2f}".format(
            r["window"], r["trad_tag"],
            r["trad"]["pearson"], r["dl"]["pearson"], r["blend_raw_mean"]["pearson"],
            r["blend_raw_pearson_vs_trad"], r["rho_dl_trad"],
            r["blend_raw_mean"]["r2"], r["blend_raw_mean"]["mse"], r["weight_sweep"]["best_w"]))
    print("=" * 110)
    print("ic_blend=等权 raw 融合(交付口径)；Δic=融合相对传统单独增益；rho=DL↔传统去相关度；")
    print("r2_blend/mse_blend=等权融合的 R²/MSE(老师口径)；best_w=当窗最优权(过拟合上界,仅诊断,~0.5 即等权≈最优)。")

    # —— 副表：seed 平均（cert 折两 seed 平均后的免费杠杆；delivery 当前单 seed 不变） —— #
    print("\n----- seed 平均(cert 折 seed42+43；交付真正会用的形态) -----")
    print("{:<22} {:>7} {:>9} {:>11} {:>11}".format("window", "n_seed", "ic_dl_avg", "ic_blend_avg", "Δic_avg"))
    for r in rows:
        if "error" in r:
            continue
        print("{:<22} {:>7d} {:>9.4f} {:>11.4f} {:>+11.4f}".format(
            r["window"], r["n_dl_seeds"], r["dl_avg"]["pearson"],
            r["blend_raw_mean_avg"]["pearson"], r["blend_avg_pearson_vs_trad"]))
    print("注：delivery 折当前被代码写死只跑 seed42(n_seed=1) → 其 avg 列=seed42，非真多 seed；交付多 seed 是下一步代码改动。")

    out = OUT_DIR / "summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "windows": rows,
                "dl_run": "20260604_xsection_raw_3fold_2seed",
                "primary_seed": 42,
                "metric_convention": "复刻 meow/eval.py：fillna(0) 后 Pearson / R²(1-SSres/var(ddof=1)/N) / MSE(SSres/N)",
                "trad_leg": "四窗均为新传统(含 rx_micro 两腿)，完全同口径；delivery 自检 pooled Pearson≈0.0812",
                "delivery_note": "delivery 折 DL 当前只有 seed42(写死)；blend_raw_mean 即交付口径(等权,零自由参数)",
                "judgment_rule": "三折 + 交付 Δic 是否全为正(融合稳赢传统) = GO；任一折 Δic≤0 则不焊、回落传统",
            },
            f, ensure_ascii=False, indent=2,
        )
    print(f"\n结果已落：{out}")


if __name__ == "__main__":
    main()
