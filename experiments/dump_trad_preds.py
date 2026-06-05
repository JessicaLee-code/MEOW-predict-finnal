"""
传统提交链逐票预测落盘（供「DL↔传统预测相关性」离线分析）。

复用 ``meow/meow.py`` 的 ``MeowEngine``：
1. ``fit(train_start, train_end)``  默认 Jun–Nov（与交付折训练窗一致）；
2. 对 eval 窗口（默认 Dec）逐票 ``predict``，落
   ``<out>/trad_preds_<eval_start>_<eval_end>.csv``（列 date,symbol,interval,label,pred）。

**不改 engine 代码**：只调用其公开 ``fit`` + ``predict`` + 内部 ``_build_window_frames``
（与 ``engine.eval`` 完全同一套帧构造，保证传统侧预测口径与 0.0803 sanity 一致）。

键 ``(date,symbol,interval)`` 为整数编码，与 DL 侧 ``run_dl --dump-preds`` 落盘同源，
可被 ``analyze_dl_trad_corr.py`` 直接 inner-join。

环境变量（与 run_submission_full_window 同名，便于复用）：
    MEOW_DATA_DIR        数据目录（默认 <repo>/data）
    MEOW_TRAIN_START/END 训练窗口（默认 20230601 / 20231130）
    MEOW_EVAL_START/END  评测窗口（默认 20231201 / 20231229）
用法：
    python experiments/dump_trad_preds.py [--out results/dl/_corr_probe]
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


def _load_meow_engine():
    """按文件路径加载 meow/meow.py 的 MeowEngine（meow/ 置于 sys.path 最前，
    保证裸 import 的 dl/feat/mdl 解析到 meow/ 下带 countDate 的版本）。"""
    sys.path.insert(0, str(SRC_DIR))
    sys.path.insert(0, str(MEOW_DIR))
    spec = importlib.util.spec_from_file_location("meow_entry_dump_preds", str(MEOW_DIR / "meow.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MeowEngine


def main(argv=None):
    ap = argparse.ArgumentParser(description="传统提交链逐票预测落盘（相关性分析用）")
    ap.add_argument("--out", default="results/dl/_corr_probe", help="落盘目录")
    ap.add_argument("--data-dir", default=os.environ.get("MEOW_DATA_DIR", str((REPO_ROOT / "data").resolve())))
    ap.add_argument("--train-start", type=int, default=int(os.environ.get("MEOW_TRAIN_START", "20230601")))
    ap.add_argument("--train-end", type=int, default=int(os.environ.get("MEOW_TRAIN_END", "20231130")))
    ap.add_argument("--eval-start", type=int, default=int(os.environ.get("MEOW_EVAL_START", "20231201")))
    ap.add_argument("--eval-end", type=int, default=int(os.environ.get("MEOW_EVAL_END", "20231229")))
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    print(f"[trad-dump] 训练 {args.train_start}-{args.train_end} / 评测 {args.eval_start}-{args.eval_end}", flush=True)

    MeowEngine = _load_meow_engine()
    engine = MeowEngine(h5dir=args.data_dir, cacheDir=None)

    print("[trad-dump] fit ...", flush=True)
    engine.fit(args.train_start, args.train_end)

    print("[trad-dump] 构造评测帧 + predict ...", flush=True)
    dates = engine.calendar.range(args.eval_start, args.eval_end)
    frames = engine._build_window_frames(dates)        # 与 engine.eval 同一套帧构造
    xdf, ydf = frames["xdf"], frames["ydf"]
    pred = np.asarray(engine.predict(xdf), dtype=np.float64).reshape(-1)

    # ydf 的非 meta 列即标签列（meow.py 用 ycol）；统一改名为 label 对齐 DL 侧。
    meta = ["date", "symbol", "interval"]
    label_col = [c for c in ydf.columns if c not in meta]
    out = ydf[meta].copy()
    out["label"] = ydf[label_col[0]].to_numpy() if label_col else np.nan
    out["pred"] = pred

    path = os.path.join(args.out, f"trad_preds_{args.eval_start}_{args.eval_end}.csv")
    out.to_csv(path, index=False)
    # 自检：整体 Pearson 应与 Dec sanity 0.0803 量级一致，作为口径未漂移的护栏。
    m = np.isfinite(out["label"]) & np.isfinite(out["pred"])
    overall = float(np.corrcoef(out["pred"][m], out["label"][m])[0, 1]) if m.sum() > 1 else float("nan")
    print(f"[trad-dump] 落盘 {path}  行数={len(out)}  整体Pearson(自检)={overall:.4f}", flush=True)
    return path


if __name__ == "__main__":
    main()
