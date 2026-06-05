# -*- coding: utf-8 -*-
"""
serve 交付端到端验证 —— 走和提交件 `python meow.py` **完全相同**的代码路径
（`MeowEngine.fit` 现训传统 + 3-seed DL → `model.predict` → `dl_serve.predict` →
`fuse_traditional_with_dl`），但额外把「传统单独 / DL 单独 / 融合」三者按**老师 eval.py
口径**（pooled Pearson、R²=1-SSE/var(y)/n、MSE=SSE/n，评前 inf/nan→0）拆开报，并扫
最优融合权重、落盘预测，回答两件事：
1. 提交件端到端能否在本机不崩地跑出 3-seed 融合交付分；
2. 融合相对纯传统是否真净增（三指标一起好），等权 w=0.5 是否≈最优。

用法（默认 = 老师 `python meow.py` 的窗口：train Jun–Nov / eval Dec1–Dec29）：
    python experiments/_serve_delivery_eval.py
"""
import os
import sys
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "meow"))   # MeowEngine / dl_serve / log
sys.path.append(str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from meow import MeowEngine  # noqa: E402
from dl_serve import fuse_traditional_with_dl  # noqa: E402

YCOL = "fret12"


def teacher_metrics(forecast, y):
    """逐字复刻 src/eval.py：评前 inf/nan→0，再算 pooled Pearson / R² / MSE。"""
    df = pd.DataFrame({"forecast": np.asarray(forecast, dtype=np.float64),
                       YCOL: np.asarray(y, dtype=np.float64)})
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    pcor = df[["forecast", YCOL]].corr().to_numpy()[0, 1]
    n = df.shape[0]
    sse = ((df["forecast"] - df[YCOL]) ** 2).sum()
    r2 = 1 - sse / df[YCOL].var() / n
    mse = sse / n
    return {"pearson": float(pcor), "r2": float(r2), "mse": float(mse), "n": int(n)}


def main():
    h5dir = os.environ.get("MEOW_DATA_DIR", str((REPO / "data").resolve()))
    train_start = int(os.environ.get("MEOW_TRAIN_START", "20230601"))
    train_end = int(os.environ.get("MEOW_TRAIN_END", "20231130"))
    eval_start = int(os.environ.get("MEOW_EVAL_START", "20231201"))
    eval_end = int(os.environ.get("MEOW_EVAL_END", "20231229"))
    outdir = REPO / "results" / "dl" / "_serve_delivery"
    outdir.mkdir(parents=True, exist_ok=True)

    print("[serve-eval] train={}~{} eval={}~{}".format(train_start, train_end, eval_start, eval_end))
    engine = MeowEngine(h5dir=h5dir, cacheDir=None)

    # —— 与提交件同路径：fit 现训传统 + 3-seed DL —— #
    engine.fit(train_start, train_end)
    dl_ok = engine.dl_serve is not None and engine.dl_serve.available
    print("[serve-eval] DL available={} seeds={}".format(
        dl_ok, engine.dl_serve.seeds if engine.dl_serve else None))

    # —— eval 窗：构 xdf/ydf（与 engine.eval 同一 _build_window_frames）—— #
    eval_dates = engine.calendar.range(eval_start, eval_end)
    frames = engine._build_window_frames(eval_dates)
    xdf, ydf = frames["xdf"], frames["ydf"]
    y = ydf[YCOL].to_numpy(dtype=np.float64)

    trad_pred = np.asarray(engine.model.predict(xdf), dtype=np.float64).reshape(-1)
    dl_df = engine.dl_serve.predict(eval_dates) if dl_ok else None
    fused = np.asarray(fuse_traditional_with_dl(xdf, trad_pred, dl_df), dtype=np.float64).reshape(-1)

    # —— 三指标：传统单独 / 融合（= 提交件实际输出）—— #
    m_trad = teacher_metrics(trad_pred, y)
    m_fused = teacher_metrics(fused, y)
    print("\n[serve-eval] === 老师口径三指标 ===")
    print("  传统单独 : Pearson={pearson:.4f} R2={r2:.5f} MSE={mse:.3e} n={n}".format(**m_trad))
    print("  融合(交付): Pearson={pearson:.4f} R2={r2:.5f} MSE={mse:.3e} n={n}".format(**m_fused))
    print("  Δic(融合-传统) = {:+.4f}".format(m_fused["pearson"] - m_trad["pearson"]))

    result = {"train": [train_start, train_end], "eval": [eval_start, eval_end],
              "dl_available": bool(dl_ok),
              "dl_seeds": list(engine.dl_serve.seeds) if engine.dl_serve else [],
              "trad": m_trad, "fused_w0.5": m_fused}

    # —— DL 覆盖行上的诊断：DL-alone、去相关 ρ、最优权扫描 —— #
    if dl_df is not None and len(dl_df) > 0:
        key = xdf[["date", "symbol", "interval"]].copy()
        key["_row"] = np.arange(len(key), dtype=np.int64)
        merged = key.merge(dl_df, on=["date", "symbol", "interval"], how="left").sort_values("_row")
        dl_aligned = merged["pred_dl"].to_numpy(dtype=np.float64)
        has_dl = np.isfinite(dl_aligned)
        ndl = int(has_dl.sum())
        print("\n[serve-eval] DL 覆盖 {}/{} 行（{:.1%}），warmup 缺 {} 行用纯传统".format(
            ndl, len(y), ndl / len(y), len(y) - ndl))

        t_c, d_c, y_c = trad_pred[has_dl], dl_aligned[has_dl], y[has_dl]
        m_dl_only = teacher_metrics(d_c, y_c)
        m_trad_on_cov = teacher_metrics(t_c, y_c)
        rho = float(np.corrcoef(t_c, d_c)[0, 1])
        print("  [仅DL覆盖行] 传统={:.4f} / DL单独={:.4f} / 去相关ρ(trad,dl)={:.3f}".format(
            m_trad_on_cov["pearson"], m_dl_only["pearson"], rho))

        # 最优等量纲权扫描（仅覆盖行，纯诊断；交付实际用 w=0.5）。
        best_w, best_ic = 0.5, -1.0
        for w in np.linspace(0, 1, 21):
            ic = teacher_metrics((1 - w) * t_c + w * d_c, y_c)["pearson"]
            if ic > best_ic:
                best_ic, best_w = ic, float(w)
        print("  [仅DL覆盖行] best_w={:.2f} best_ic={:.4f}（w=0.5 等权 ic={:.4f}）".format(
            best_w, best_ic, teacher_metrics(0.5 * t_c + 0.5 * d_c, y_c)["pearson"]))

        result.update({"dl_coverage": ndl, "n_total": int(len(y)), "rho": rho,
                       "dl_only_on_cov": m_dl_only, "trad_on_cov": m_trad_on_cov,
                       "best_w_on_cov": best_w, "best_ic_on_cov": best_ic})

        # 落盘预测（供后续深挖；results/ 已 gitignore）。
        dump = xdf[["date", "symbol", "interval"]].copy()
        dump["y"] = y
        dump["trad"] = trad_pred
        dump["dl"] = dl_aligned
        dump["fused"] = fused
        dump.to_csv(outdir / "serve_delivery_preds.csv", index=False)
        print("[serve-eval] 预测已落盘: {}".format(outdir / "serve_delivery_preds.csv"))

    (outdir / "serve_delivery_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[serve-eval] summary: {}".format(outdir / "serve_delivery_summary.json"))
    print("[serve-eval] ===== DONE =====")


if __name__ == "__main__":
    main()
