"""验证：给 ridge 成员也加 rx_micro 新特征后，ridge+lgbm 融合是否真正受益。

背景（2026-06-03）
------------------
我们最强的传统代表 = ridge(X1) + lgbm(M_lgbm_d4) **等权 raw_mean 融合**（≈0.0803）。
此前已把 rx_micro 49 列新特征只挂到 lgbm 成员（三折全票 lgbm 单腿 +0.0034），但融合是等权 →
**只有 lgbm 半边吃了新特征、增益被 ridge 半边稀释**。本脚本补测“ridge 也加新特征”后融合的真实变化。

一次对比三种融合（全部 raw_mean 等权、与交付口径一致）：
  - blend_base       = ridge(旧 108) + lgbm(旧 413)        ← 接入前基线
  - blend_lgbm_only  = ridge(旧 108) + lgbm(新 462)        ← 当前已接入状态
  - blend_both       = ridge(新 157) + lgbm(新 462)        ← 目标：两腿都加新特征

口径
----
- 折：与 DL / P2 / lgbm 探针同 `build_dl_folds` 协议（expanding / val_window=20 / step=20 /
  embargo=1 / min_train_days=40 / max_folds=3 / fold_select=recent）。
- 时间紧 → 默认只跑 **fold2**（最强折 0.0904 + 最近、最像交付）做方向探查；
  环境变量 `PROBE_FOLDS`（逗号分隔 fold_id，如 "0,1,2"）可指定跑哪几折。
- 传统单 seed 确定性（ridge 无随机；lgbm random_state=42 固定）→ 无多 seed 可减，一次即准。
- ridge alpha=2.0、lgbm 参数 = M_lgbm_d4，winsorize 走 ExperimentRunner 默认，全部与交付一致。
- 评分 = pooled Pearson（与 meow/eval.py、experiment_runner 同口径）。
- 特征一次现算 482 列并集（含 rx_micro），按列名切各成员子集；逐模型 fit、用完即释放，控内存峰值。

用法：python experiments/probe_ridge_blend_newfeat.py
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# meow 置最前：保证裸 import 的 dl 解析到 meow/dl.py（带 loadDate / countDate / date 列）。
for d in ("config", "models", "src", "meow"):
    sys.path.insert(0, os.path.join(REPO, d))

from dl import MeowDataLoader                                  # meow/dl.py
from submission_pipeline import (
    SubmissionFeaturePipeline,
    DEFAULT_SUBMISSION_GROUPS,
    DEFAULT_SUBMISSION_MEMBERS,
)
from experiment_runner import ExperimentRunner
from dl_protocol import build_dl_folds

META = ["date", "symbol", "interval"]


def _pooled_pearson(pred, y):
    """pooled Pearson：全样本拉平算一个相关系数（与 meow/eval.py 同口径）。"""
    p = np.asarray(pred, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    m = np.isfinite(p) & np.isfinite(yy)
    return float(np.corrcoef(p[m], yy[m])[0, 1]) if m.sum() > 1 else float("nan")


def _build_482(pipeline, loader, dates):
    """预分配 + 逐日填充 → (X[float32, 482列], y[float64], feat_names, meta_df)。

    绕开整窗 concat 的内存尖峰：预读每日行数、预分配单份总矩阵、逐日 build_feature_frames(单日) 填入。
    pipeline 已是含 rx_micro 的 482 列并集，因此这里直接拿全部成员所需的列，后续按列名切子集。
    """
    dates = [int(d) for d in dates]
    per_day = [int(loader.countDate(d)) for d in dates]
    total = int(sum(per_day))
    raw0 = loader.loadDate(dates[0])
    x0, _ = pipeline.build_feature_frames(raw0)
    feat = [c for c in x0.columns if c not in META]
    del raw0, x0
    gc.collect()
    X = np.empty((total, len(feat)), dtype=np.float32)
    y = np.empty(total, dtype=np.float64)
    md = np.empty(total, np.int64); ms = np.empty(total, np.int64); mi = np.empty(total, np.int64)
    r = 0
    for d, n in zip(dates, per_day):
        raw_d = loader.loadDate(d)
        x_d, y_d = pipeline.build_feature_frames(raw_d)
        m = x_d.merge(y_d, on=META)
        if len(m) != n:
            raise RuntimeError(f"行数不一致 @ {d}: merge {len(m)} vs countDate {n}")
        X[r:r + n, :] = m[feat].to_numpy(np.float32)
        y[r:r + n] = m["fret12"].to_numpy(np.float64)
        md[r:r + n] = m["date"].to_numpy(); ms[r:r + n] = m["symbol"].to_numpy(); mi[r:r + n] = m["interval"].to_numpy()
        r += n
        del raw_d, x_d, y_d, m
    gc.collect()
    meta_df = pd.DataFrame({"date": md, "symbol": ms, "interval": mi})
    return X, y, feat, meta_df


def _resolve_col_sets():
    """解析 4 个特征子集的列名（全部基于交付链单点定义，避免漂移）。"""
    ridge_member = DEFAULT_SUBMISSION_MEMBERS[0]   # X1 ridge
    lgbm_member = DEFAULT_SUBMISSION_MEMBERS[1]     # M_lgbm_d4（已含 rx_micro）
    ridge_old = SubmissionFeaturePipeline(groups=ridge_member.groups).feature_names()
    lgbm_new = SubmissionFeaturePipeline(groups=lgbm_member.groups).feature_names()
    rx_cols = SubmissionFeaturePipeline(groups=("rx_micro",)).feature_names()
    rx_set = set(rx_cols)
    ridge_new = list(ridge_old) + [c for c in rx_cols if c not in set(ridge_old)]
    lgbm_old = [c for c in lgbm_new if c not in rx_set]
    return {
        "ridge_old": list(ridge_old),
        "ridge_new": list(ridge_new),
        "lgbm_old": list(lgbm_old),
        "lgbm_new": list(lgbm_new),
        "ridge_alpha": float(ridge_member.model_params.get("alpha", 2.0)),
        "lgbm_params": dict(lgbm_member.model_params),
    }


def _fit_one(runner, kind, X_all, col_idx, cols, meta_df, y, params):
    """切子集 → fit 单个底模；ridge/lgbm 走 _fit_model_core，口径与交付一致。"""
    idx = [col_idx[c] for c in cols]
    x_sub = X_all[:, idx]                      # float32 子集（copy）
    ytr = meta_df.copy()
    ytr["fret12"] = y.astype(np.float32)
    model, feature_cols, _ = runner._fit_model_core(
        kind, x_sub, cols, ytr, target_mode="raw", model_params=params
    )
    del x_sub, ytr
    gc.collect()
    return model, feature_cols


def _predict_one(model, kind, X_all, col_idx, cols):
    """切打分子集 → predict；ridge 转 float64（对齐 fit），lgbm float32。"""
    idx = [col_idx[c] for c in cols]
    dtype = np.float64 if kind == "ridge" else np.float32
    x_sub = X_all[:, idx].astype(dtype, copy=False)
    pred = np.asarray(model.predict(x_sub), dtype=np.float64)
    del x_sub
    gc.collect()
    return pred


def main():
    h5dir = os.environ.get("MEOW_DATA_DIR", os.path.join(REPO, "data"))
    out_dir = os.path.join(REPO, "results", "dl", "_p3_ridge_blend")
    os.makedirs(out_dir, exist_ok=True)
    which = os.environ.get("PROBE_FOLDS", "2")
    want_folds = {int(s) for s in which.split(",") if s.strip() != ""}
    t0 = time.time()

    folds = build_dl_folds(20230601, 20231130, mode="expanding", val_window=20, step=20,
                           embargo=1, min_train_days=40, max_folds=3, fold_select="recent")
    folds = [f for f in folds if f.fold_id in want_folds]
    print(f"[blend] 跑折 = {sorted(want_folds)}（共 {len(folds)} 折）；传统单 seed 确定性", flush=True)

    cs = _resolve_col_sets()
    print(f"[blend] 列数：ridge_old={len(cs['ridge_old'])} ridge_new={len(cs['ridge_new'])} "
          f"lgbm_old={len(cs['lgbm_old'])} lgbm_new={len(cs['lgbm_new'])} "
          f"| ridge_alpha={cs['ridge_alpha']} lgbm={cs['lgbm_params']}", flush=True)

    loader = MeowDataLoader(h5dir=h5dir)
    pipeline = SubmissionFeaturePipeline(groups=DEFAULT_SUBMISSION_GROUPS)   # 482 列并集
    runner = ExperimentRunner(h5dir=h5dir)

    results = []
    for fold in folds:
        tf = time.time()
        print(f"\n[blend] === fold{fold.fold_id} 训练 {fold.train_start}-{fold.train_end}"
              f"（{len(fold.train_dates)}日）→ 打分 {fold.val_start}-{fold.val_end}"
              f"（{len(fold.scoring_dates)}日）全票 ===", flush=True)

        # ---- 训练集特征（482 列，一次现算）----
        print(f"[blend] fold{fold.fold_id} build 训练特征 ...", flush=True)
        Xtr, ytr, feat, meta_tr = _build_482(pipeline, loader, fold.train_dates)
        col_idx = {c: i for i, c in enumerate(feat)}
        print(f"[blend]   训练 {Xtr.shape[0]} 行 × {Xtr.shape[1]} 列", flush=True)

        # ---- 逐个底模 fit（顺序经内存优化）----
        # 关键：fold2 整窗 ~7M 行 × 482 列 ≈ 14GB；若 lgbm fit 时仍与整窗并存会逼近 31.6GB OOM。
        # 故：ridge 两腿先吃整窗（子集小、float64 + copy=False），随后切出 lgbm 子集并**立即释放整窗**，
        # 使后续两个 lgbm fit 不再与整窗并存，把峰值压到 ~25GB 安全区。
        ytr_df = meta_tr.copy()
        ytr_df["fret12"] = ytr.astype(np.float32)
        models = {}
        models["ridge_old"], _ = _fit_one(runner, "ridge", Xtr, col_idx, cs["ridge_old"], meta_tr, ytr, {"alpha": cs["ridge_alpha"]})
        print(f"[blend]   ridge_old fit 完", flush=True)
        models["ridge_new"], _ = _fit_one(runner, "ridge", Xtr, col_idx, cs["ridge_new"], meta_tr, ytr, {"alpha": cs["ridge_alpha"]})
        print(f"[blend]   ridge_new fit 完", flush=True)
        # 切出 lgbm_new(462) 子集后立即释放整窗 Xtr。
        idx_new = [col_idx[c] for c in cs["lgbm_new"]]
        X_lgbm = Xtr[:, idx_new]
        del Xtr
        gc.collect()
        models["lgbm_new"], _, _ = runner._fit_model_core(
            "lgbm", X_lgbm, cs["lgbm_new"], ytr_df, target_mode="raw", model_params=cs["lgbm_params"])
        print(f"[blend]   lgbm_new fit 完", flush=True)
        # lgbm_old = lgbm_new 去掉 49 个 rx 列（在 lgbm_new 内的相对位置切，无需回到整窗）。
        pos = {c: i for i, c in enumerate(cs["lgbm_new"])}
        idx_old = [pos[c] for c in cs["lgbm_old"]]
        X_lgbm_old = X_lgbm[:, idx_old]
        del X_lgbm
        gc.collect()
        models["lgbm_old"], _, _ = runner._fit_model_core(
            "lgbm", X_lgbm_old, cs["lgbm_old"], ytr_df, target_mode="raw", model_params=cs["lgbm_params"])
        print(f"[blend]   lgbm_old fit 完", flush=True)
        del X_lgbm_old, ytr, meta_tr, ytr_df
        gc.collect()

        # ---- 打分集特征 + 各模型 predict ----
        print(f"[blend] fold{fold.fold_id} build 打分特征 + predict ...", flush=True)
        Xsc, ysc, feat_s, _ = _build_482(pipeline, loader, fold.scoring_dates)
        col_idx_s = {c: i for i, c in enumerate(feat_s)}
        p_ridge_old = _predict_one(models["ridge_old"], "ridge", Xsc, col_idx_s, cs["ridge_old"])
        p_ridge_new = _predict_one(models["ridge_new"], "ridge", Xsc, col_idx_s, cs["ridge_new"])
        p_lgbm_old = _predict_one(models["lgbm_old"], "lgbm", Xsc, col_idx_s, cs["lgbm_old"])
        p_lgbm_new = _predict_one(models["lgbm_new"], "lgbm", Xsc, col_idx_s, cs["lgbm_new"])
        del Xsc
        gc.collect()

        # ---- 单腿 + 三种 raw_mean 融合的 pooled Pearson ----
        rec = {
            "fold_id": fold.fold_id,
            "val_start": int(fold.val_start), "val_end": int(fold.val_end),
            "ridge_old": _pooled_pearson(p_ridge_old, ysc),
            "ridge_new": _pooled_pearson(p_ridge_new, ysc),
            "lgbm_old": _pooled_pearson(p_lgbm_old, ysc),
            "lgbm_new": _pooled_pearson(p_lgbm_new, ysc),
            "blend_base": _pooled_pearson((p_ridge_old + p_lgbm_old) / 2.0, ysc),
            "blend_lgbm_only": _pooled_pearson((p_ridge_old + p_lgbm_new) / 2.0, ysc),
            "blend_both": _pooled_pearson((p_ridge_new + p_lgbm_new) / 2.0, ysc),
        }
        rec["ridge_delta"] = rec["ridge_new"] - rec["ridge_old"]
        rec["blend_both_vs_base"] = rec["blend_both"] - rec["blend_base"]
        rec["blend_both_vs_lgbmonly"] = rec["blend_both"] - rec["blend_lgbm_only"]
        results.append(rec)
        del p_ridge_old, p_ridge_new, p_lgbm_old, p_lgbm_new, models
        gc.collect()

        print(f"[blend] fold{fold.fold_id} 用时 {time.time()-tf:.0f}s  →", flush=True)
        for k in ("ridge_old", "ridge_new", "lgbm_old", "lgbm_new",
                  "blend_base", "blend_lgbm_only", "blend_both"):
            print(f"         {k:18s} = {rec[k]:.4f}", flush=True)
        print(f"         ridge Δ(新-旧)      = {rec['ridge_delta']:+.4f}", flush=True)
        print(f"         融合 都加-基线       = {rec['blend_both_vs_base']:+.4f}", flush=True)
        print(f"         融合 都加-只lgbm     = {rec['blend_both_vs_lgbmonly']:+.4f}", flush=True)

    summary = {
        "folds": results,
        "note": "传统单 seed 确定性；ridge alpha=2.0 / lgbm=M_lgbm_d4；raw_mean 等权融合；pooled Pearson",
        "total_sec": time.time() - t0,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[blend] 完成，用时 {summary['total_sec']:.0f}s，结果 → {out_dir}/summary.json", flush=True)


if __name__ == "__main__":
    main()
