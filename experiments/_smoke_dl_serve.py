# -*- coding: utf-8 -*-
"""
DLServe 焊接「快速 smoke」——只验 fit/predict 拆分这段新代码路径能端到端跑通，
不追求精度（短窗 + 1 seed + 小 epoch）。目的：在委托全窗长跑前 ~2 分钟内暴露
新代码的契约 bug（normalizer 复用 / 多 seed 循环 / label_frame 对齐 / 防御降级）。
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "meow"))  # dl_serve + log 在 meow/ 下
sys.path.append(str(REPO / "src"))

import numpy as np  # noqa: E402

from dl import MeowDataLoader  # noqa: E402
from dl_serve import DLServe, fuse_traditional_with_dl  # noqa: E402
from tradingcalendar import Calendar  # noqa: E402


def main():
    h5dir = str((REPO / "data").resolve())
    cal = Calendar()
    # 短窗：训练 ~15 天、评测紧随其后 ~5 天（够构出截面 + 序列窗、又快）。
    train_dates = cal.range(20230601, 20230626)
    eval_dates = cal.range(20230627, 20230704)
    print("[smoke] train_dates={} eval_dates={}".format(len(train_dates), len(eval_dates)))

    loader = MeowDataLoader(h5dir=h5dir)
    # 1 seed + 小 epoch，纯为验代码路径（精度无意义）。
    serve = DLServe(
        raw_loader=loader.loadDates,
        seeds=(42,),
        hparams={
            "dropout": 0.2, "hidden_size": 32, "num_layers": 1,
            "lambda_corr": 0.3, "max_epochs": 2, "patience": 2, "weight_decay": 0.001,
        },
    )
    serve.fit(train_dates)
    print("[smoke] after fit: available={}".format(serve.available))
    assert serve.available, "smoke: DLServe.fit 未成功（available=False），看上面报错"

    dl_df = serve.predict(eval_dates)
    assert dl_df is not None and len(dl_df) > 0, "smoke: predict 返回空"
    print("[smoke] dl_df rows={} cols={}".format(len(dl_df), list(dl_df.columns)))
    print("[smoke] pred_dl 统计: mean={:.3e} std={:.3e} finite={}".format(
        float(dl_df["pred_dl"].mean()),
        float(dl_df["pred_dl"].std()),
        bool(np.isfinite(dl_df["pred_dl"].to_numpy()).all()),
    ))

    # 验融合胶水：构一个假 xdf（含 eval 全部 meta）+ 假传统预测，跑 fuse 看 warmup 填补。
    raw_eval = loader.loadDates(list(eval_dates)).sort_values(["date", "symbol", "interval"])
    xdf = raw_eval[["date", "symbol", "interval"]].reset_index(drop=True)
    trad = np.zeros(len(xdf), dtype=np.float32)  # 假传统=0，融合后非 warmup 行应=0.5*pred_dl
    fused = fuse_traditional_with_dl(xdf, trad, dl_df, weight_dl=0.5)
    n_dl = int(np.isfinite(dl_df.set_index(["date", "symbol", "interval"])
                           .reindex(xdf.set_index(["date", "symbol", "interval"]).index)["pred_dl"]
                           .to_numpy()).sum())
    print("[smoke] fuse: xdf_rows={} 用上DL行={} fused_finite={}".format(
        len(xdf), n_dl, bool(np.isfinite(fused).all())))
    assert np.isfinite(fused).all(), "smoke: 融合结果含 NaN（warmup 填补出问题）"
    print("[smoke] ===== PASS =====")


if __name__ == "__main__":
    main()
