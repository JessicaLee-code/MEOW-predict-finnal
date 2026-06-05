"""
DL↔传统 预测相关性 + 集成增益分析（线 A 的判决脚本）。

输入两份逐票预测 CSV（列含 date,symbol,interval,label,pred）：
- DL 侧：``run_dl --dump-preds`` 落的 ``preds_*_fold*_seed*.csv``（可传多份，按行拼接）；
- 传统侧：``dump_trad_preds.py`` 落的 ``trad_preds_*.csv``。

按 ``(date,symbol,interval)`` inner-join 后回答三个问题：
1. **DL 和传统预测有多去相关**（整体 Pearson + 逐日 IC 的均值）——决定集成有没有肉；
2. 各自 vs 标签的 corr（整体 + 逐日 IC），核对与已知读数（传统 ~0.0776/0.0803）是否一致；
3. **等权 / 最优线性 / 逐日最优 集成后 vs 标签的 corr**——看叠 DL 能否把传统抬过 0.0776。

判读口径：
- 老师精度分含 pooled corr 与 daily-IC，本脚本两者都报；**daily-IC 均值是主镜头**。
- corr 尺度无关，集成前对每个预测做 z-score 再线性组合（raw_mean 另报一行供交付口径对照）。

用法：
    python experiments/analyze_dl_trad_corr.py \
        --dl results/dl/<run_id>/preds/preds_*.csv \
        --trad results/dl/_corr_probe/trad_preds_20231201_20231229.csv \
        --out results/dl/_corr_probe/corr_report.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

KEYS = ["date", "symbol", "interval"]


def _read_many(patterns):
    """把若干 glob 模式命中的预测 CSV 按行拼接（多折/多 seed 时去重取均值）。"""
    paths = []
    for p in patterns:
        paths.extend(sorted(glob.glob(p)))
    if not paths:
        raise FileNotFoundError(f"无匹配预测文件：{patterns}")
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    # 同一 (date,symbol,interval) 若被多折/多 seed 重复预测，取 pred 均值（label 取首个）。
    agg = df.groupby(KEYS, as_index=False).agg(pred=("pred", "mean"), label=("label", "first"))
    return agg, paths


def _pearson(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2:
        return float("nan")
    c = np.corrcoef(a[m], b[m])[0, 1]
    return float(c) if np.isfinite(c) else float("nan")


def _daily_ic(df, pred_col, label_col="label"):
    """逐日 IC：按 date 分组算组内 Pearson，返回均值/标准差/最坏日。"""
    ics = []
    for _, g in df.groupby("date"):
        c = _pearson(g[pred_col], g[label_col])
        if np.isfinite(c):
            ics.append(c)
    if not ics:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "n_days": 0}
    ics = np.array(ics)
    return {"mean": float(ics.mean()), "std": float(ics.std()),
            "min": float(ics.min()), "n_days": int(len(ics))}


def _zscore(x):
    x = np.asarray(x, dtype=np.float64)
    s = np.nanstd(x)
    return (x - np.nanmean(x)) / s if s > 0 else x - np.nanmean(x)


def main(argv=None):
    ap = argparse.ArgumentParser(description="DL↔传统 相关性 + 集成增益分析")
    ap.add_argument("--dl", nargs="+", required=True, help="DL 预测 CSV（可 glob，多份按行拼接）")
    ap.add_argument("--trad", nargs="+", required=True, help="传统预测 CSV（可 glob）")
    ap.add_argument("--out", default="results/dl/_corr_probe/corr_report.json")
    args = ap.parse_args(argv)

    dl, dl_paths = _read_many(args.dl)
    trad, trad_paths = _read_many(args.trad)
    print(f"[corr] DL 文件 {len(dl_paths)} 份 / {len(dl)} 行；传统 {len(trad_paths)} 份 / {len(trad)} 行", flush=True)

    # inner-join：只在「DL 有预测」的 (date,symbol,interval) 上比较（DL 因 warmup 丢前 L-1 行）。
    j = dl.merge(trad[KEYS + ["pred"]], on=KEYS, how="inner", suffixes=("_dl", "_trad"))
    if j.empty:
        raise RuntimeError("inner-join 为空：DL 与传统的 (date,symbol,interval) 键不重叠，检查编码口径。")
    print(f"[corr] join 后 {len(j)} 行，覆盖 {j['date'].nunique()} 天", flush=True)

    # 集成预测（z-score 后等权；raw_mean 另算供交付口径对照）。
    z_dl, z_trad = _zscore(j["pred_dl"]), _zscore(j["pred_trad"])
    j["ens_z_equal"] = 0.5 * z_dl + 0.5 * z_trad
    j["ens_raw_equal"] = 0.5 * j["pred_dl"] + 0.5 * j["pred_trad"]
    # 全局最优静态权重（在 z 空间扫 w·trad+(1-w)·dl，按 daily-IC 均值挑），只作上界参考。
    best = {"w_trad": None, "daily_ic_mean": -np.inf}
    for w in np.linspace(0.0, 1.0, 21):
        blend = w * z_trad + (1.0 - w) * z_dl
        tmp = j.assign(_b=blend)
        ic = _daily_ic(tmp, "_b")["mean"]
        if np.isfinite(ic) and ic > best["daily_ic_mean"]:
            best = {"w_trad": float(w), "daily_ic_mean": float(ic)}

    report = {
        "n_rows": int(len(j)),
        "n_days": int(j["date"].nunique()),
        "dl_files": dl_paths,
        "trad_files": trad_paths,
        # ① DL↔传统去相关程度（主判：越低集成越有肉）
        "corr_dl_trad_overall": _pearson(j["pred_dl"], j["pred_trad"]),
        "corr_dl_trad_daily_ic": _daily_ic(j.assign(_t=j["pred_trad"]), "pred_dl", "_t"),
        # ② 各自 vs 标签（核对已知读数）
        "dl_vs_label": {"overall": _pearson(j["pred_dl"], j["label"]),
                        "daily_ic": _daily_ic(j, "pred_dl")},
        "trad_vs_label": {"overall": _pearson(j["pred_trad"], j["label"]),
                          "daily_ic": _daily_ic(j, "pred_trad")},
        # ③ 集成 vs 标签（看能否破传统）
        "ens_z_equal_vs_label": {"overall": _pearson(j["ens_z_equal"], j["label"]),
                                 "daily_ic": _daily_ic(j, "ens_z_equal")},
        "ens_raw_equal_vs_label": {"overall": _pearson(j["ens_raw_equal"], j["label"]),
                                   "daily_ic": _daily_ic(j, "ens_raw_equal")},
        "ens_best_static_w": best,   # 上界参考（同窗挑权，乐观）
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 人读摘要
    def _ic(d):
        return f"mean={d['daily_ic']['mean']:.4f} min={d['daily_ic']['min']:.4f} (pooled={d['overall']:.4f})"
    print("\n================ DL↔传统 相关性 + 集成增益 ================")
    print(f"join 行数 {report['n_rows']} / {report['n_days']} 天")
    print(f"① DL↔传统去相关：pooled corr={report['corr_dl_trad_overall']:.4f} "
          f"/ daily-IC mean={report['corr_dl_trad_daily_ic']['mean']:.4f}  "
          f"(越低集成越有肉)")
    print(f"② DL   vs 标签：{_ic(report['dl_vs_label'])}")
    print(f"   传统 vs 标签：{_ic(report['trad_vs_label'])}")
    print(f"③ 等权(z)  集成 vs 标签：{_ic(report['ens_z_equal_vs_label'])}")
    print(f"   等权(raw) 集成 vs 标签：{_ic(report['ens_raw_equal_vs_label'])}")
    print(f"   最优静态权(同窗挑,乐观上界)：w_trad={best['w_trad']} daily-IC={best['daily_ic_mean']:.4f}")
    print(f"\n报告落盘：{args.out}")
    return report


if __name__ == "__main__":
    main()
