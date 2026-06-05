"""P2 地基 —— 传统提交链「对齐 DL 折」的无泄漏滚动 OOS 预测落盘。

为什么需要这个脚本（与单窗 dump_trad_preds.py 的区别）
--------------------------------------------------------
单窗版 `dump_trad_preds.py` 只在一个 Dec 窗 fit→predict，且历史上探针被迫退到
lgbm 单腿（彼时 ridge 会 NaN）。但「DL+传统融合」要下判决，必须满足两件事：

1. **同协议**：传统和 DL 必须在 *完全相同的折边界、相同 embargo、相同样本点*
   上各自出预测——否则两套不同评测协议下的历史数字（传统 expanding 0.0776 vs
   DL 新协议 0.0624）根本不是同一批样本，不能比、不能逐票融合。
2. **真代表**：P1 修好 ridge 的内存依赖型 NaN 后，这里落的是 *锁定的融合代表*
   （X1 ridge + lgbm_d4 等权 raw_mean），不再是退化的 lgbm 单腿。

本脚本因此 **直接 import 与 DL 同一个 `build_dl_folds`/`build_delivery_fold`**
（同函数、同参数 → 折边界与 DL SWEEP 逐字节一致），对 3 个选型折 + 交付折逐折
传统 `fit(train)→predict(scoring)`，embargo 焊在折边界里、天然无泄漏，落
`date,symbol,interval,label,pred`，可被 `analyze_dl_trad_corr.py` 与 DL 同折
逐票 inner-join。

产物（默认落 results/dl/_p2_trad_folds/，gitignore 不入库）
----------------------------------------------------------
- `trad_preds_fold{k}_{val_start}_{val_end}.csv`  每个选型折一份；
- `trad_preds_delivery_{val_start}_{val_end}.csv`  交付折一份；
- `folds_index.json`  每折边界 + 自检 Pearson + pred NaN 计数（核对无泄漏对齐
  + 免费护栏：传统在新折读数应与 expanding ~0.0776 量级吻合，差太远=协议搬运有 bug）。

无泄漏 / 对齐口径
-----------------
- 每折传统 `fit(fold.train_start, fold.train_end)` 用满该折训练区（= DL 的
  train_core + earlystop，DL 只是把 earlystop 尾段留作早停监控，训练区是同一批历史）；
- `predict(fold.scoring_dates)` 只打分打分区，embargo 那几日既不训也不评 → 与 DL
  同训练区、同禁飞区；
- 不改 engine 代码：只调用其公开 `fit` + `predict` + 内部 `_build_window_frames`
  （与 `engine.eval` 同一套帧构造，保证传统侧口径与 0.0803/0.0775 sanity 一致）。

用法（CPU、可后台；4 个全量 fit 约 1.5–2h）
--------------------------------------------
    python experiments/dump_trad_preds_folds.py
    # 参数默认 = CLAUDE.md 的 SWEEP 长跑命令；如需对齐其它 run，显式覆盖即可。
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
MEOW_DIR = REPO_ROOT / "meow"
SRC_DIR = REPO_ROOT / "src"

# 路径顺序与 dump_trad_preds.py 一致：meow/ 置最前，保证裸 import 的 dl/feat/mdl 解析到
# meow/ 下带 countDate 的版本；src/ 紧随其后，供 import dl_protocol（src 独有模块）。
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(MEOW_DIR))

# 与 DL 同源的折构造（src/dl_protocol.py）——这是「同协议对齐」的根：传统折直接由
# 与 run_dl.py 完全相同的函数 + 参数派生，逐字节同边界。
from dl_protocol import (  # noqa: E402  (须在 sys.path 设置之后 import)
    assert_folds_causal,
    build_delivery_fold,
    build_dl_folds,
)


def _load_meow_engine():
    """按文件路径加载 meow/meow.py 的 MeowEngine（与单窗版同一套加载方式）。"""
    spec = importlib.util.spec_from_file_location("meow_entry_dump_folds", str(MEOW_DIR / "meow.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MeowEngine


def _dump_fold(engine, fold, out_dir, tag):
    """对单折跑传统 fit→predict→落盘，返回该折的边界 + 自检摘要 dict。"""
    train_start, train_end = fold.train_start, fold.train_end
    val_start, val_end = fold.val_start, fold.val_end
    t0 = time.time()
    print(
        f"[{tag}] fit 训练区 {train_start}-{train_end}"
        f"（{len(fold.train_dates)} 交易日）→ predict 打分区 {val_start}-{val_end}"
        f"（{len(fold.scoring_dates)} 交易日，embargo {len(fold.embargo_dates)} 日）...",
        flush=True,
    )
    # 每折重训锁定融合代表（_fit_impl 起手 self.models={}，干净覆盖，不残留上折模型）。
    engine.fit(train_start, train_end)

    # 与 engine.eval 同一套帧构造，仅在打分区上 predict。
    dates = list(fold.scoring_dates)
    frames = engine._build_window_frames(dates)
    xdf, ydf = frames["xdf"], frames["ydf"]
    pred = np.asarray(engine.predict(xdf), dtype=np.float64).reshape(-1)

    meta = ["date", "symbol", "interval"]
    label_col = [c for c in ydf.columns if c not in meta]
    out = ydf[meta].copy()
    out["label"] = ydf[label_col[0]].to_numpy() if label_col else np.full(len(out), np.nan)
    out["pred"] = pred

    path = os.path.join(out_dir, f"trad_preds_{tag}_{val_start}_{val_end}.csv")
    out.to_csv(path, index=False)

    # 自检：① pred NaN 计数（验 P1 修复在多折训练区上同样稳、不再内存依赖型 NaN）；
    #       ② 整体 Pearson（护栏：应与 expanding ~0.0776 量级吻合，差太远=协议搬运 bug）。
    finite = np.isfinite(out["label"].to_numpy()) & np.isfinite(out["pred"].to_numpy())
    overall = (
        float(np.corrcoef(out["pred"].to_numpy()[finite], out["label"].to_numpy()[finite])[0, 1])
        if finite.sum() > 1 else float("nan")
    )
    n_pred_nan = int((~np.isfinite(out["pred"].to_numpy())).sum())
    n_rows = int(len(out))
    print(
        f"[{tag}] 落盘 {path}  行={n_rows}  pred_nan={n_pred_nan}  "
        f"自检Pearson={overall:.4f}  耗时={time.time() - t0:.0f}s",
        flush=True,
    )

    rec = {
        "tag": tag,
        "fold_id": int(fold.fold_id),
        "train_start": int(train_start),
        "train_end": int(train_end),
        "n_train_days": len(fold.train_dates),
        "embargo_days": len(fold.embargo_dates),
        "val_start": int(val_start),
        "val_end": int(val_end),
        "n_scoring_days": len(fold.scoring_dates),
        "n_rows": n_rows,
        "pred_nan": n_pred_nan,
        "selfcheck_pearson": overall,
        "path": os.path.relpath(path, REPO_ROOT),
    }

    # 主动释放，压低折间内存基线（下一折又会预分配整窗矩阵）。
    del frames, xdf, ydf, pred, out
    gc.collect()
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description="P2: 传统提交链对齐 DL 折的无泄漏 OOS 预测落盘")
    ap.add_argument("--out", default="results/dl/_p2_trad_folds", help="落盘目录")
    ap.add_argument("--data-dir", default=os.environ.get("MEOW_DATA_DIR", str((REPO_ROOT / "data").resolve())))
    # 以下默认 = CLAUDE.md 的 SWEEP 长跑命令；改对齐别的 run 时显式覆盖。
    ap.add_argument("--start", type=int, default=20230601, help="rolling_start（训练区锚点）")
    ap.add_argument("--end", type=int, default=20231130, help="rolling_end（选型截止）")
    ap.add_argument("--delivery-eval-end", type=int, default=20231229, help="交付折 eval 末日；<=0 关闭交付折")
    ap.add_argument("--val-window", type=int, default=20)
    ap.add_argument("--step", type=int, default=20)
    ap.add_argument("--embargo", type=int, default=1)
    ap.add_argument("--min-train-days", type=int, default=40)
    ap.add_argument("--max-folds", type=int, default=3)
    ap.add_argument("--fold-select", default="recent", choices=["recent", "first"])
    ap.add_argument("--earlystop-frac", type=float, default=0.15)
    ap.add_argument("--dry-run", action="store_true",
                    help="只派生+打印折计划+过无泄漏闸就退出，不 fit（验证折边界对齐用）")
    args = ap.parse_args(argv)

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    # ---- 1) 派生与 DL 完全同源的折（同函数同参数 → 逐字节同边界）---- #
    folds = build_dl_folds(
        args.start, args.end,
        mode="expanding", val_window=args.val_window, step=args.step, embargo=args.embargo,
        train_window=None, min_train_days=args.min_train_days, earlystop_frac=args.earlystop_frac,
        max_folds=args.max_folds, fold_select=args.fold_select,
    )
    if not folds:
        raise SystemExit("未派生出任何选型折，检查 start/end/min-train-days/val-window。")

    delivery_fold = None
    if args.delivery_eval_end and args.delivery_eval_end > 0:
        delivery_fold_id = max(f.fold_id for f in folds) + 1
        delivery_fold = build_delivery_fold(
            args.start, args.end, args.delivery_eval_end,
            embargo=args.embargo, earlystop_frac=args.earlystop_frac, fold_id=delivery_fold_id,
        )

    # 无泄漏闸：四段时间严格递增、embargo 真的隔开训练/打分（与 DL 同一道校验）。
    assert_folds_causal(folds + ([delivery_fold] if delivery_fold else []))

    print("=" * 64, flush=True)
    print("P2 折计划（传统侧，与 DL SWEEP 同边界）：", flush=True)
    for f in folds:
        print(f"  选型 fold{f.fold_id}: 训练 {f.train_start}-{f.train_end}"
              f"（{len(f.train_dates)}日）| embargo {len(f.embargo_dates)}日 | "
              f"打分 {f.val_start}-{f.val_end}（{len(f.scoring_dates)}日）", flush=True)
    if delivery_fold:
        f = delivery_fold
        print(f"  交付 fold{f.fold_id}: 训练 {f.train_start}-{f.train_end}"
              f"（{len(f.train_dates)}日）| embargo {len(f.embargo_dates)}日 | "
              f"打分 {f.val_start}-{f.val_end}（{len(f.scoring_dates)}日）", flush=True)
    print(f"传统代表 = 锁定融合 [X1 ridge + lgbm_d4] 等权 raw_mean（ridge 已 P1 修 float64）", flush=True)
    print("=" * 64, flush=True)

    if args.dry_run:
        print("（--dry-run：折计划已打印、无泄漏闸已过，不执行 fit，退出。）", flush=True)
        return None

    # ---- 2) 单 engine 循环每折 fit→predict→落盘 ---- #
    MeowEngine = _load_meow_engine()
    engine = MeowEngine(h5dir=args.data_dir, cacheDir=None)

    index = []
    for f in folds:
        rec = _dump_fold(engine, f, out_dir, tag=f"fold{f.fold_id}")
        index.append(rec)
    if delivery_fold:
        rec = _dump_fold(engine, delivery_fold, out_dir, tag="delivery")
        index.append(rec)

    # ---- 3) 写折索引 + 汇总自检 ---- #
    index_path = os.path.join(out_dir, "folds_index.json")
    with open(index_path, "w", encoding="utf-8") as fp:
        json.dump({
            "params": {
                "start": args.start, "end": args.end, "delivery_eval_end": args.delivery_eval_end,
                "val_window": args.val_window, "step": args.step, "embargo": args.embargo,
                "min_train_days": args.min_train_days, "max_folds": args.max_folds,
                "fold_select": args.fold_select, "earlystop_frac": args.earlystop_frac,
            },
            "representative": "locked_fusion[X1_ridge+lgbm_d4]_raw_mean",
            "folds": index,
        }, fp, ensure_ascii=False, indent=2)

    print("\n" + "=" * 64, flush=True)
    print(f"P2 完成。折索引：{index_path}", flush=True)
    sel = [r for r in index if r["tag"].startswith("fold")]
    if sel:
        ps = [r["selfcheck_pearson"] for r in sel if np.isfinite(r["selfcheck_pearson"])]
        nan_total = sum(r["pred_nan"] for r in index)
        print(f"选型 3 折自检 Pearson：{[round(r['selfcheck_pearson'], 4) for r in sel]}"
              f"  均值={np.mean(ps):.4f}  最坏={np.min(ps):.4f}" if ps else "选型折自检 Pearson 全 NaN（异常！）", flush=True)
        print(f"全折 pred NaN 总数：{nan_total}（应为 0 → 验证 P1 修复在多折训练区上同样稳）", flush=True)
    print("护栏：传统新折读数应与 expanding ~0.0776 量级吻合；差太远=协议搬运有 bug。", flush=True)
    print("=" * 64, flush=True)
    return index


if __name__ == "__main__":
    main()
