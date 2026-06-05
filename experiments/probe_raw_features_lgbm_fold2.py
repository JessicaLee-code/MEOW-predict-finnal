"""快速验证：把"被欠用的 raw 数据"摘成归一化特征加进强传统 lgbm，看 fold2 上涨不涨。

背景（2026-06-03）：诊断发现 raw 是富 LOB+订单流，但 433 特征严重欠用——挂撤单 20 列
只压成 4 个静态比率、盘口只做不平衡摘要、成交明细几乎没用。本脚本**不碰交付特征管线**，
独立验证：在 fold2（与 DL/P2 同边界）上，比

    A = lgbm(纯 433)                  ← 复现强传统 lgbm 腿
    B = lgbm(433 + 摘取的新特征)       ← 加上挂撤单/盘口/成交明细的归一化特征

的 pooled Pearson + delta。

口径对齐：
- lgbm 走 `ExperimentRunner._fit_model_core("lgbm", ...)`，参数 = 提交链 M_lgbm_d4
  （max_depth=4, num_leaves=15），target/winsorize 与交付完全一致；
- 评分 = pooled Pearson（与 meow/eval.py、experiment_runner.evaluate_predictions 同口径）；
- 折由 `build_dl_folds` 派生（与 DL/传统 P2 逐字节同边界）。

**新特征全部归一化/比率/相对量 → 跨票跨时间可比、平稳 → 泛化优先，绝不喂原始量纲值。**

用法：python experiments/probe_raw_features_lgbm_fold2.py
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
# meow 置最前：保证裸 import 的 dl 解析到 meow/dl.py（带 loadDate/date 列）。
for d in ("config", "models", "src", "meow"):
    sys.path.insert(0, os.path.join(REPO, d))

from dl import MeowDataLoader                       # meow/dl.py
from submission_pipeline import SubmissionFeaturePipeline, DEFAULT_SUBMISSION_GROUPS
from experiment_runner import ExperimentRunner
from dl_protocol import build_dl_folds

META = ["date", "symbol", "interval"]
EPS = 1e-9


def _safe_div(a, b):
    return np.asarray(a, dtype=np.float64) / (np.asarray(b, dtype=np.float64) + EPS)


def build_new_features(raw: pd.DataFrame) -> pd.DataFrame:
    """从 raw 摘取被 433 欠用的归一化特征（挂撤单为主 + 盘口形状 + 成交明细）。

    全部为比率/不平衡/相对量/归一化统计 → 跨票跨时间可比、平稳。
    时序项（EMA/变化率）按 (date,symbol) 组内、日内因果、不跨日跨票。
    """
    # 新特征统一加 rx_ 前缀，避免与 433 既有列（如 add_imb/cxl_imb/buy_vwad_gap）重名。
    df = raw.sort_values(["date", "symbol", "interval"], kind="mergesort").reset_index(drop=True)
    out = pd.DataFrame({c: df[c].to_numpy() for c in META})
    mid = df["midpx"].to_numpy(dtype=np.float64)
    safe_mid = np.where(np.abs(mid) < EPS, np.nan, mid)

    # add_imb / cxl_imb 已在 433（build_base），这里只作中间量算"新"信号，不重复输出。
    add_imb = _safe_div(df["nAddBuy"] - df["nAddSell"], df["nAddBuy"] + df["nAddSell"])
    cxl_imb = _safe_div(df["nCxlBuy"] - df["nCxlSell"], df["nCxlBuy"] + df["nCxlSell"])

    # ===== ① 挂撤单（最欠用：20 列只做过 4 比率）=====
    out["rx_cxl_rate_cnt"] = _safe_div(df["nCxlBuy"] + df["nCxlSell"], df["nAddBuy"] + df["nAddSell"])  # 撤单率（毒性）
    out["rx_cxl_rate_qty"] = _safe_div(df["cxlBuyQty"] + df["cxlSellQty"], df["addBuyQty"] + df["addSellQty"])
    out["rx_add_imb_to"] = _safe_div(df["addBuyTurnover"] - df["addSellTurnover"], df["addBuyTurnover"] + df["addSellTurnover"])
    out["rx_cxl_imb_to"] = _safe_div(df["cxlBuyTurnover"] - df["cxlSellTurnover"], df["cxlBuyTurnover"] + df["cxlSellTurnover"])
    out["rx_net_order_press"] = add_imb - cxl_imb   # 挂买多+撤买少=真实买压
    trd = df["nTradeBuy"].to_numpy(dtype=np.float64) + df["nTradeSell"].to_numpy(dtype=np.float64)
    out["rx_add_vs_trade"] = np.log1p(_safe_div(df["nAddBuy"] + df["nAddSell"], trd))
    out["rx_cxl_vs_trade"] = np.log1p(_safe_div(df["nCxlBuy"] + df["nCxlSell"], trd))

    # ===== ② 盘口形状（433 只做了不平衡摘要 obi/ofi，没建形状）=====
    asz = df["asize0"].to_numpy(dtype=np.float64); bsz = df["bsize0"].to_numpy(dtype=np.float64)
    asz04 = df["asize0_4"].to_numpy(dtype=np.float64); bsz04 = df["bsize0_4"].to_numpy(dtype=np.float64)
    asz59 = df["asize5_9"].to_numpy(dtype=np.float64); bsz59 = df["bsize5_9"].to_numpy(dtype=np.float64)
    asz1019 = df["asize10_19"].to_numpy(dtype=np.float64); bsz1019 = df["bsize10_19"].to_numpy(dtype=np.float64)
    obi0 = _safe_div(asz - bsz, asz + bsz)
    obi4 = _safe_div(asz04 - bsz04, asz04 + bsz04)
    obi9 = _safe_div(asz59 - bsz59, asz59 + bsz59)
    obi19 = _safe_div(asz1019 - bsz1019, asz1019 + bsz1019)   # 最深档（433 的 obi 没到这档）
    out["rx_obi19"] = obi19
    out["rx_obi_weighted"] = 0.4 * obi0 + 0.3 * obi4 + 0.2 * obi9 + 0.1 * obi19
    out["rx_depth_nf_bid"] = np.log1p(_safe_div(bsz04, bsz1019))   # 近/远档深度比
    out["rx_depth_nf_ask"] = np.log1p(_safe_div(asz04, asz1019))
    spr0 = df["ask0"].to_numpy(dtype=np.float64) - df["bid0"].to_numpy(dtype=np.float64)
    spr4 = df["ask4"].to_numpy(dtype=np.float64) - df["bid4"].to_numpy(dtype=np.float64)
    out["rx_spread_deep_rel"] = _safe_div(spr4 - spr0, safe_mid)   # 深档价差相对收紧

    # ===== ③ 成交明细（buyVwad/sellVwad/High/Low 几乎没用）=====
    out["rx_vwad_gap_buy"] = _safe_div(df["buyVwad"].to_numpy(dtype=np.float64) - mid, safe_mid)
    out["rx_vwad_gap_sell"] = _safe_div(df["sellVwad"].to_numpy(dtype=np.float64) - mid, safe_mid)
    out["rx_trade_range_buy"] = _safe_div(df["tradeBuyHigh"].to_numpy(dtype=np.float64) - df["tradeBuyLow"].to_numpy(dtype=np.float64), safe_mid)
    out["rx_trade_range_sell"] = _safe_div(df["tradeSellHigh"].to_numpy(dtype=np.float64) - df["tradeSellLow"].to_numpy(dtype=np.float64), safe_mid)

    # 静态项清 NaN/inf（无成交导致）
    out = out.replace([np.inf, -np.inf], np.nan)
    for c in out.columns:
        if c not in META:
            out[c] = out[c].fillna(0.0)

    # ===== ④ 时序项（日内因果，按 (date,symbol) 组内）=====
    out["_add_imb_tmp"] = add_imb   # 临时列供组内 diff，算完即丢
    g = out.groupby(["date", "symbol"], sort=False)
    out["rx_net_press_ema5"] = g["rx_net_order_press"].transform(lambda s: s.ewm(halflife=5, adjust=False).mean()).fillna(0.0)
    out["rx_cxl_rate_ema5"] = g["rx_cxl_rate_cnt"].transform(lambda s: s.ewm(halflife=5, adjust=False).mean()).fillna(0.0)
    out["rx_obi_w_ema5"] = g["rx_obi_weighted"].transform(lambda s: s.ewm(halflife=5, adjust=False).mean()).fillna(0.0)
    out["rx_add_imb_chg"] = g["_add_imb_tmp"].transform(lambda s: s.diff()).fillna(0.0)
    out = out.drop(columns=["_add_imb_tmp"])

    # ============================================================ #
    # 第二批（rx2_）：盘口微观结构深挖 + 多时间尺度 + 波动率 + 交互
    # ============================================================ #
    # 微观价格偏离：对侧量加权 microprice 减 mid —— 经典短期方向信号（卖压大→micro 偏 bid）
    micro = _safe_div(df["bid0"].to_numpy(np.float64) * asz + df["ask0"].to_numpy(np.float64) * bsz, asz + bsz)
    out["rx2_microprice_dev"] = _safe_div(micro - mid, safe_mid)
    # 全档量不平衡 + 流动性集中度（最优档量占总深度的比）
    tot_b = bsz + bsz04 + bsz59 + bsz1019
    tot_a = asz + asz04 + asz59 + asz1019
    out["rx2_depth_imb_all"] = _safe_div(tot_b - tot_a, tot_b + tot_a)
    out["rx2_conc_bid"] = _safe_div(bsz, tot_b)
    out["rx2_conc_ask"] = _safe_div(asz, tot_a)
    out["rx2_obi_term"] = obi19 - obi0                                   # 盘口不平衡的深浅期限结构
    out["rx2_spread_rel"] = _safe_div(df["ask0"].to_numpy(np.float64) - df["bid0"].to_numpy(np.float64), safe_mid)
    # 成交不平衡（中间量）→ 与净挂撤压力的交互（订单流一致性）
    trade_imb = _safe_div(df["tradeBuyQty"].to_numpy(np.float64) - df["tradeSellQty"].to_numpy(np.float64),
                          df["tradeBuyQty"].to_numpy(np.float64) + df["tradeSellQty"].to_numpy(np.float64))
    out["rx2_press_x_tradeimb"] = out["rx_net_order_press"].to_numpy(np.float64) * trade_imb

    out = out.replace([np.inf, -np.inf], np.nan)
    for c in [c for c in out.columns if c.startswith("rx2_")]:
        out[c] = out[c].fillna(0.0)

    # 多时间尺度时序（z-score = 当前值相对近窗的标准化异常；长窗 EMA；已实现波动）
    out["_mid_tmp"] = mid
    g = out.groupby(["date", "symbol"], sort=False)

    def _z(col, w):
        return g[col].transform(
            lambda s: (s - s.rolling(w, min_periods=2).mean()) / (s.rolling(w, min_periods=2).std(ddof=0) + EPS)
        ).fillna(0.0)

    out["rx2_net_press_z12"] = _z("rx_net_order_press", 12)
    out["rx2_obi_w_z12"] = _z("rx_obi_weighted", 12)
    out["rx2_cxl_rate_z12"] = _z("rx_cxl_rate_cnt", 12)
    out["rx2_micro_dev_z12"] = _z("rx2_microprice_dev", 12)
    out["rx2_net_press_ema12"] = g["rx_net_order_press"].transform(lambda s: s.ewm(halflife=12, adjust=False).mean()).fillna(0.0)
    # 已实现波动（mid 日内对数收益的 rolling std）+ 与撤单率交互
    out["_logret_tmp"] = g["_mid_tmp"].transform(lambda s: np.log(s.clip(lower=EPS)).diff()).fillna(0.0)
    g2 = out.groupby(["date", "symbol"], sort=False)
    out["rx2_rvol12"] = g2["_logret_tmp"].transform(lambda s: s.rolling(12, min_periods=2).std(ddof=0)).fillna(0.0)
    out["rx2_cxl_x_rvol"] = out["rx_cxl_rate_cnt"].to_numpy(np.float64) * out["rx2_rvol12"].to_numpy(np.float64)
    out = out.drop(columns=["_mid_tmp", "_logret_tmp"])

    # ============================================================ #
    # 第三批（rx3_）：补传统 433+rx+rx2 漏掉的 18 列原始字段
    #   OHLC(lastpx/open/high/low) + 分档成交额比率(btr/atr) + 挂撤单价格区间(add/cxl High/Low)
    #   全部相对 mid 归一化 / 比率，跨票跨时间可比；无成交无挂撤导致的 NaN 填 0。
    # ============================================================ #
    def _rel(col):
        return _safe_div(df[col].to_numpy(np.float64) - mid, safe_mid)

    # OHLC（日内多稀疏，相对 mid 偏离 / 振幅 / 累计收益）
    out["rx3_lastpx_dev"] = _rel("lastpx")
    out["rx3_hl_range"] = _safe_div(df["high"].to_numpy(np.float64) - df["low"].to_numpy(np.float64), safe_mid)
    op = df["open"].to_numpy(np.float64)
    out["rx3_oc_ret"] = _safe_div(mid - op, np.where(np.abs(op) < EPS, np.nan, op))
    # 分档成交额比率（btr/atr 各档）→ 买卖不平衡 + 近远档比
    btr04 = df["btr0_4"].to_numpy(np.float64); atr04 = df["atr0_4"].to_numpy(np.float64)
    btr59 = df["btr5_9"].to_numpy(np.float64); atr59 = df["atr5_9"].to_numpy(np.float64)
    btr1019 = df["btr10_19"].to_numpy(np.float64); atr1019 = df["atr10_19"].to_numpy(np.float64)
    out["rx3_to_imb_04"] = _safe_div(btr04 - atr04, btr04 + atr04)
    out["rx3_to_imb_59"] = _safe_div(btr59 - atr59, btr59 + atr59)
    out["rx3_to_imb_1019"] = _safe_div(btr1019 - atr1019, btr1019 + atr1019)
    out["rx3_to_imb_all"] = _safe_div((btr04 + btr59 + btr1019) - (atr04 + atr59 + atr1019),
                                      btr04 + btr59 + btr1019 + atr04 + atr59 + atr1019)
    out["rx3_to_nf_b"] = np.log1p(_safe_div(btr04, btr1019))
    out["rx3_to_nf_a"] = np.log1p(_safe_div(atr04, atr1019))
    # 挂撤单价格区间/价位（相对 mid）：挂买单挂多高、撤单价位、价格分散度
    out["rx3_addbuy_range"] = _safe_div(df["addBuyHigh"].to_numpy(np.float64) - df["addBuyLow"].to_numpy(np.float64), safe_mid)
    out["rx3_addsell_range"] = _safe_div(df["addSellHigh"].to_numpy(np.float64) - df["addSellLow"].to_numpy(np.float64), safe_mid)
    out["rx3_addbuy_lvl"] = _rel("addBuyHigh")
    out["rx3_addsell_lvl"] = _rel("addSellLow")
    out["rx3_cxlbuy_lvl"] = _rel("cxlBuyHigh")
    out["rx3_cxlsell_lvl"] = _rel("cxlSellLow")

    out = out.replace([np.inf, -np.inf], np.nan)
    for c in [c for c in out.columns if c.startswith("rx3_")]:
        out[c] = out[c].fillna(0.0)

    return out.astype({c: np.float32 for c in out.columns if c not in META})


def _build_X(loader, pipeline, dates, with_new):
    """预分配 + 逐日填充 → (X[float32], y[float64], feat_names, meta_df)。

    绕开 build_feature_frames 整窗 concat 的内存尖峰（全票 708万行×433 concat 峰值 ~22GB OOM），
    改为预读每日行数、预分配单份总矩阵、逐日 build_feature_frames(单日) 填入 → 峰值≈单份(~12-14GB)，
    支持**全票**。with_new=False 只 433；True 拼上 rx/rx2/rx3 新特征。
    A、B 各自独立调用（不在内存里同时持有两份大矩阵），峰值可控。
    """
    dates = [int(d) for d in dates]
    per_day = [int(loader.countDate(d)) for d in dates]
    total = int(sum(per_day))
    # 首日定列名/列序（保证全窗一致）
    raw0 = loader.loadDate(dates[0])
    x0, _ = pipeline.build_feature_frames(raw0)
    feat = [c for c in x0.columns if c not in META]
    if with_new:
        xn0 = build_new_features(raw0)
        feat = feat + [c for c in xn0.columns if c not in META]
        del xn0
    del raw0, x0
    gc.collect()
    X = np.empty((total, len(feat)), dtype=np.float32)
    y = np.empty(total, dtype=np.float64)
    md = np.empty(total, np.int64); ms = np.empty(total, np.int64); mi = np.empty(total, np.int64)
    r = 0
    for d, n in zip(dates, per_day):
        raw_d = loader.loadDate(d)
        x_d, y_d = pipeline.build_feature_frames(raw_d)
        if with_new:
            xn_d = build_new_features(raw_d)
            m = x_d.merge(xn_d, on=META).merge(y_d, on=META)
            del xn_d
        else:
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


def _fit_lgbm(runner, X, feat, meta_df, y, lgbm_params):
    """走 _fit_model_core("lgbm")，口径 = 提交链 M_lgbm_d4（同 target/winsorize）。"""
    ytr = meta_df.copy()
    ytr["fret12"] = y.astype(np.float32)
    model, _, _ = runner._fit_model_core("lgbm", X, feat, ytr, target_mode="raw", model_params=lgbm_params)
    return model


def _pooled_pearson(pred, y):
    p = np.asarray(pred, dtype=np.float64); yy = np.asarray(y, dtype=np.float64)
    m = np.isfinite(p) & np.isfinite(yy)
    return float(np.corrcoef(p[m], yy[m])[0, 1]) if m.sum() > 1 else float("nan")


def main():
    h5dir = os.environ.get("MEOW_DATA_DIR", os.path.join(REPO, "data"))
    out_dir = os.path.join(REPO, "results", "dl", "_p3_trad_newfeat")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    # ---- 与 DL/P2 完全同协议：三折全票 ----
    folds = build_dl_folds(20230601, 20231130, mode="expanding", val_window=20, step=20,
                           embargo=1, min_train_days=40, max_folds=3, fold_select="recent")
    print(f"[p3] 三折全票完整验证（与 DL/P2 同 build_dl_folds 边界）：{len(folds)} 折", flush=True)

    loader = MeowDataLoader(h5dir=h5dir)
    pipeline = SubmissionFeaturePipeline(groups=DEFAULT_SUBMISSION_GROUPS)
    runner = ExperimentRunner(h5dir=h5dir)
    lgbm_params = {"max_depth": 4, "num_leaves": 15, "n_jobs": 8}   # = M_lgbm_d4

    results = []
    last_new_imp = []
    for fi, fold in enumerate(folds):
        tf = time.time()
        print(f"\n[p3] === fold{fold.fold_id} 训练 {fold.train_start}-{fold.train_end}"
              f"（{len(fold.train_dates)}日）→ 打分 {fold.val_start}-{fold.val_end}"
              f"（{len(fold.scoring_dates)}日）全票 ===", flush=True)

        # A: 纯 433（独立 build，峰值≈单份）
        print(f"[p3] fold{fold.fold_id} build+fit A(纯433) ...", flush=True)
        Xa, ya, feat433, meta_a = _build_X(loader, pipeline, fold.train_dates, with_new=False)
        print(f"[p3]   训练 {Xa.shape[0]} 行 × {Xa.shape[1]} 列(433)", flush=True)
        model_a = _fit_lgbm(runner, Xa, feat433, meta_a, ya, lgbm_params)
        del Xa, ya, meta_a; gc.collect()

        # B: 433+新（独立 build）
        print(f"[p3] fold{fold.fold_id} build+fit B(433+新) ...", flush=True)
        Xb, yb, feat_all, meta_b = _build_X(loader, pipeline, fold.train_dates, with_new=True)
        newcols = [c for c in feat_all if c not in feat433]
        print(f"[p3]   训练 {Xb.shape[0]} 行 × {Xb.shape[1]} 列(433+{len(newcols)}新)", flush=True)
        model_b = _fit_lgbm(runner, Xb, feat_all, meta_b, yb, lgbm_params)
        del Xb, yb, meta_b; gc.collect()

        # 打分窗（build all 一次，A 切前 433 列）
        print(f"[p3] fold{fold.fold_id} build 打分窗 + predict ...", flush=True)
        Xsc, ysc, feat_all_s, _ = _build_X(loader, pipeline, fold.scoring_dates, with_new=True)
        k = len(feat433)
        pa = _pooled_pearson(model_a.predict(Xsc[:, :k]), ysc)
        pb = _pooled_pearson(model_b.predict(Xsc), ysc)
        del Xsc, ysc; gc.collect()

        imp = np.asarray(model_b.feature_importances_, dtype=np.float64)
        imp = imp / (imp.sum() + EPS)
        new_imp = sorted([(feat_all[i], imp[i]) for i in range(len(feat_all)) if feat_all[i] in newcols],
                         key=lambda kv: -kv[1])
        last_new_imp = new_imp
        results.append({"fold_id": int(fold.fold_id), "val_start": int(fold.val_start),
                        "val_end": int(fold.val_end), "A_433": pa, "B_433plus": pb,
                        "delta": pb - pa, "new_imp_sum": float(sum(w for _, w in new_imp))})
        print(f"[p3] fold{fold.fold_id}: A(纯433)={pa:.4f}  B(433+新)={pb:.4f}  Δ={pb-pa:+.4f}"
              f"  新特征重要性={sum(w for _, w in new_imp)*100:.1f}%  耗时{time.time()-tf:.0f}s", flush=True)
        del model_a, model_b; gc.collect()

    # ---- 三折汇总 ----
    dels = [r["delta"] for r in results]
    a_mean = float(np.mean([r["A_433"] for r in results]))
    b_mean = float(np.mean([r["B_433plus"] for r in results]))
    summary = {"folds": results, "A_mean": a_mean, "B_mean": b_mean,
               "delta_mean": float(np.mean(dels)), "delta_min": float(np.min(dels)),
               "delta_max": float(np.max(dels)), "n_new_feats": len(last_new_imp),
               "total_sec": time.time() - t0}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 64, flush=True)
    print("传统 lgbm 三折全票完整验证（433 vs 433+新特征，新协议）", flush=True)
    for r in results:
        print(f"  fold{r['fold_id']} {r['val_start']}-{r['val_end']}: "
              f"A={r['A_433']:.4f}  B={r['B_433plus']:.4f}  Δ={r['delta']:+.4f}", flush=True)
    print(f"  ── 均值: A={a_mean:.4f}  B={b_mean:.4f}  Δ均值={np.mean(dels):+.4f}  Δ最坏={np.min(dels):+.4f}", flush=True)
    print(f"  新特征(fold{results[-1]['fold_id']})重要性 top:", flush=True)
    for name, w in last_new_imp[:10]:
        print(f"     {name:22s} {w*100:5.2f}%", flush=True)
    print(f"  汇总落: {os.path.join(out_dir, 'summary.json')}   总耗时 {time.time()-t0:.0f}s", flush=True)
    print("=" * 64, flush=True)


if __name__ == "__main__":
    main()
