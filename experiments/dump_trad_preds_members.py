"""传统提交链「逐成员」Dec 预测落盘（供 DL↔传统相关性分析）。

背景（2026-06-02 诊断）：本机（34GB Win，可用 ~23GB）上，正式提交融合的
**X1 ridge 成员**整窗 fit 会数值失败——433 特征里存在 ~1e15 量级的未归一化列，
float32 窗口下 X^TX 病态、cholesky 失败 → 回退 svd → svd 需 ~30GB U 矩阵 OOM →
ridge 系数 NaN → 等权融合 `(NaN+lgbm)/2` 全 NaN。
（lgbm 成员树模型尺度无关、内存友好，整窗 fit 正常。换机迁移前 0.0803 sanity
能跑通，是因为那台机器内存/数据条件不同。）

因此这里不取**融合输出**（会被 NaN ridge 拖垮），改为**逐成员**落盘：
- 健康的 `M_lgbm_d4` 成员 → 作为本机可靠的传统信号，写 `trad_preds_<s>_<e>.csv`
  （列 date,symbol,interval,label,pred，与 DL 侧 --dump-preds 同源，可直接 join）；
- `X1_...` ridge 成员 → 单独写 `trad_preds_ridge_<s>_<e>.csv` 留档（预期 NaN，实证用）。

不改 engine / submission 任何代码：只读 `engine.model.runtime` 的已训练成员，
复用其 `_predict_with_baseline`（与正式 predict 完全同一条 winsorize/baseline 路径）。
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
MEOW_DIR = REPO_ROOT / "meow"
SRC_DIR = REPO_ROOT / "src"

# lgbm 成员（健康）与 ridge 成员（本机 NaN）的 experiment_id，照 submission_pipeline 定义。
LGBM_ID = "M_lgbm_d4"
RIDGE_ID = "X1_R02_plus_ofi_safe_condmom_interaction"


def _load_meow_engine():
    sys.path.insert(0, str(SRC_DIR))
    sys.path.insert(0, str(MEOW_DIR))
    spec = importlib.util.spec_from_file_location("meow_entry_dump_members", str(MEOW_DIR / "meow.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MeowEngine


def _member_pred(runtime, xpred, exp_id):
    """复用正式 predict 的逐成员路径，对单个成员产出预测（不参与融合）。"""
    member = next(m for m in runtime.spec.members if m.experiment_id == exp_id)
    member_xpred = runtime._member_xdf(xpred, member)
    pred = runtime.runner._predict_with_baseline(
        runtime.models[exp_id],
        member_xpred,
        runtime.member_feature_cols[exp_id],
        ydf=None,
        baseline=runtime.member_baselines.get(exp_id),
        target_mode=member.target_mode,
    )
    return np.asarray(pred, dtype=np.float64).reshape(-1)


def _dump(out_path, meta_df, label, pred):
    out = meta_df.copy()
    out["label"] = np.asarray(label, dtype=np.float64).reshape(-1)
    out["pred"] = np.asarray(pred, dtype=np.float64).reshape(-1)
    out.to_csv(out_path, index=False)
    m = np.isfinite(out["label"]) & np.isfinite(out["pred"])
    overall = float(np.corrcoef(out["pred"][m], out["label"][m])[0, 1]) if m.sum() > 1 else float("nan")
    print(f"[dump-members] 落盘 {out_path}  行数={len(out)}  有限pred={int(m.sum())}  整体Pearson(自检)={overall:.4f}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="传统提交链逐成员 Dec 预测落盘（相关性分析用）")
    ap.add_argument("--out", default="results/dl/_corr_probe")
    ap.add_argument("--data-dir", default=os.environ.get("MEOW_DATA_DIR", str((REPO_ROOT / "data").resolve())))
    ap.add_argument("--train-start", type=int, default=int(os.environ.get("MEOW_TRAIN_START", "20230601")))
    ap.add_argument("--train-end", type=int, default=int(os.environ.get("MEOW_TRAIN_END", "20231130")))
    ap.add_argument("--eval-start", type=int, default=int(os.environ.get("MEOW_EVAL_START", "20231201")))
    ap.add_argument("--eval-end", type=int, default=int(os.environ.get("MEOW_EVAL_END", "20231229")))
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    print(f"[dump-members] 训练 {args.train_start}-{args.train_end} / 评测 {args.eval_start}-{args.eval_end}", flush=True)

    MeowEngine = _load_meow_engine()
    engine = MeowEngine(h5dir=args.data_dir, cacheDir=None)

    print("[dump-members] fit ...（ridge 成员预期数值失败成 NaN，lgbm 成员正常）", flush=True)
    engine.fit(args.train_start, args.train_end)

    print("[dump-members] 构造评测帧 ...", flush=True)
    dates = engine.calendar.range(args.eval_start, args.eval_end)
    frames = engine._build_window_frames(dates)
    xdf, ydf = frames["xdf"], frames["ydf"]

    runtime = engine.model.runtime
    xpred = runtime._normalize_meow_frame(xdf, require_target=False)
    meta = ["date", "symbol", "interval"]
    meta_df = xpred[meta].reset_index(drop=True)
    label_col = [c for c in ydf.columns if c not in meta]
    label = ydf[label_col[0]].to_numpy() if label_col else np.full(len(meta_df), np.nan)

    # 健康成员（lgbm）→ 规范传统信号文件名（analyze 脚本默认读这个）。
    print("[dump-members] predict lgbm 成员 ...", flush=True)
    lgbm_pred = _member_pred(runtime, xpred, LGBM_ID)
    _dump(os.path.join(args.out, f"trad_preds_{args.eval_start}_{args.eval_end}.csv"), meta_df, label, lgbm_pred)

    # ridge 成员（本机预期 NaN）→ 单独留档实证。
    print("[dump-members] predict ridge 成员（实证 NaN）...", flush=True)
    try:
        ridge_pred = _member_pred(runtime, xpred, RIDGE_ID)
        _dump(os.path.join(args.out, f"trad_preds_ridge_{args.eval_start}_{args.eval_end}.csv"), meta_df, label, ridge_pred)
    except Exception as e:
        print(f"[dump-members] ridge 成员预测异常（符合预期）: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
