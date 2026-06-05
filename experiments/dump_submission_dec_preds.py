#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dump 新传统（含 rx_micro 两腿）的 Dec 逐票预测，供与第二台 DL-on-raw 融合。

为什么需要这个脚本
------------------
机器1 手上的传统 delivery 逐票预测（`results/dl/_p2_trad_folds/`）是 P2 阶段产物、
**不含 rx_micro**；而最新交付传统是含 rx_micro 两腿的（Dec sanity pooled Pearson=0.0812）。
要算**真正交付口径**的「传统×DL」融合分，必须先 dump 含 rx_micro 的新传统 Dec 逐票预测。

流程
----
1. 用最新交付管线 `MeowEngine.fit(Jun1–Nov30)` —— 逐成员现算+fit（ridge+lgbm 都含 rx_micro），
   内存峰值 ~22GB（已验证、不 OOM）。
2. 对 Dec（20231201–20231229）逐日现算并集特征 + predict，落 (date,symbol,interval,label,pred) CSV。
3. 输出 `results/dl/_blend_dl_trad/trad_dec_newfeat_preds.csv`，
   供 `probe_blend_dl_trad.py` 用新传统重算 delivery 融合（交付口径）。

只 CPU、不碰 GPU。embargo 自然处理：DL delivery 是 Dec4–Dec29、本脚本 predict 整个 Dec，
融合时按 (date,symbol,interval) inner-join 取交集（= Dec4–Dec29）。
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
# 三目录平铺 import 约定；meow/ 放最前（meow.py 内部裸 import dl/feat/mdl）。
for sub in ("src", "config", "models"):
    p = str(REPO / sub)
    if p not in sys.path:
        sys.path.insert(0, p)
# 清掉可能被 src 同名模块占用的缓存，强制从 meow/ 解析 dl/feat/mdl。
for name in ("dl", "feat", "mdl"):
    sys.modules.pop(name, None)
sys.path.insert(0, str(REPO / "meow"))

_spec = importlib.util.spec_from_file_location("meow_entry_dump", str(REPO / "meow" / "meow.py"))
_meow = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_meow)
MeowEngine = _meow.MeowEngine

DATA = os.environ.get("MEOW_DATA_DIR", str((REPO / "data").resolve()))
TRAIN_START = int(os.environ.get("MEOW_TRAIN_START", "20230601"))
TRAIN_END = int(os.environ.get("MEOW_TRAIN_END", "20231130"))
EVAL_START = int(os.environ.get("MEOW_EVAL_START", "20231201"))
EVAL_END = int(os.environ.get("MEOW_EVAL_END", "20231229"))
# 输出路径可经 MEOW_OUT 覆盖：默认 dump delivery(Dec)；复用同一逐成员 fit 路径
# dump 选型折(fold1/fold2)新传统 expanding 预测时，传不同的 train/eval 窗 + MEOW_OUT 即可。
OUT = Path(os.environ.get("MEOW_OUT", str(REPO / "results/dl/_blend_dl_trad/trad_dec_newfeat_preds.csv")))


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(
        "[dump] 数据={} train={}-{} eval={}-{}".format(DATA, TRAIN_START, TRAIN_END, EVAL_START, EVAL_END),
        flush=True,
    )
    engine = MeowEngine(h5dir=DATA, cacheDir=None)
    print("[dump] 开始 fit（逐成员现算+fit，ridge+lgbm 均含 rx_micro）...", flush=True)
    engine.fit(TRAIN_START, TRAIN_END)
    print("[dump] fit 完成，开始 predict Dec 并落盘...", flush=True)
    dates = engine.calendar.range(EVAL_START, EVAL_END)
    frames = engine._build_window_frames(dates)  # union 482 列（predict 路径不变）
    xdf = frames["xdf"]
    ydf = frames["ydf"]
    forecast = engine.predict(xdf)
    out = ydf[["date", "symbol", "interval", "fret12"]].copy()
    out = out.rename(columns={"fret12": "label"})
    out["pred"] = np.asarray(forecast, dtype=np.float64)
    out.to_csv(OUT, index=False)
    # 自检 pooled Pearson（应 ≈ 0.0812，和 Dec sanity 对得上，确认 dump 口径正确）。
    y = out["label"].to_numpy(np.float64)
    p = out["pred"].to_numpy(np.float64)
    mask = np.isfinite(y) & np.isfinite(p)
    ic = float(np.corrcoef(p[mask], y[mask])[0, 1])
    print("[dump] 落盘 {} 行 → {}".format(len(out), OUT), flush=True)
    print("[dump] 自检 pooled Pearson={:.4f}（delivery 应≈0.0812；fold1/fold2 为各自选型窗 OOS）".format(ic), flush=True)


if __name__ == "__main__":
    main()
