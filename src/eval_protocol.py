"""
评测协议模块 - Rolling Evaluation Protocol

实现三层评测体系：
  第一层：Dev Rolling    - 模型开发、特征筛选的主要依据
  第二层：Review Holdout - 候选模型复核（11月）
  第三层：Final Holdout  - 最终提交前模拟（12月，尽量少碰）

提供四个 rolling profile 横向对比，输出统一可复现 leaderboard。

使用方式：
  from eval_protocol import EvaluationProtocolRunner, ROLLING_PROFILES, ALL_SPECS, BASELINE_ID
"""

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd

from experiment_runner import ExperimentRunner, SplitConfig, RollingFold

try:
    from scheduler import ParallelScheduler
except ImportError:
    ParallelScheduler = None


# ================================================================== #
# Rolling Profile 配置
# ================================================================== #

@dataclass
class RollingProfile:
    """Rolling 评测 profile 配置"""
    profile_name: str       # 唯一名称
    val_window: int         # 验证窗口（交易日）
    step: int               # 滚动步长（交易日）
    embargo: int            # 禁飞区（交易日）
    mode: str               # "sliding" 或 "expanding"
    train_window: Optional[int] = None  # 固定训练窗口（sliding 模式）
    min_train_days: int = 10            # 最小训练天数


# 四个标准 profile
ROLLING_PROFILES: List[RollingProfile] = [
    RollingProfile(
        profile_name="short_8d_2d",
        train_window=8,
        min_train_days=8,
        val_window=2,
        step=5,
        embargo=1,
        mode="sliding",
    ),
    RollingProfile(
        profile_name="medium_20d_5d",
        train_window=20,
        min_train_days=20,
        val_window=5,
        step=5,
        embargo=1,
        mode="sliding",
    ),
    RollingProfile(
        profile_name="long_40d_5d",
        train_window=40,
        min_train_days=40,
        val_window=5,
        step=10,
        embargo=1,
        mode="sliding",
    ),
    RollingProfile(
        profile_name="expanding_40d_5d",
        train_window=None,
        min_train_days=40,
        val_window=5,
        step=5,
        embargo=1,
        mode="expanding",
    ),
]

# protocol_stability_score 的加权权重
PROFILE_WEIGHTS: Dict[str, float] = {
    "short_8d_2d": 0.25,
    "medium_20d_5d": 0.35,
    "long_40d_5d": 0.25,
    "expanding_40d_5d": 0.15,
}


# ================================================================== #
# Fold Manifest（带 embargo 信息）
# ================================================================== #

@dataclass
class FoldManifestEntry:
    """单个 fold 的完整日期切法，含 embargo 区间"""
    profile_name: str
    fold_id: int
    train_start: int
    train_end: int
    embargo_start: int  # embargo 起（如 embargo=0 则等于 train_end）
    embargo_end: int    # embargo 末
    val_start: int
    val_end: int
    n_train_days: int
    n_val_days: int


# ================================================================== #
# Baseline 与历史实验 Specs
# ================================================================== #

BASELINE_ID = "R02_ridge_legacy_plus_norm_core"

BASELINE_SPEC = {
    "experiment_id": BASELINE_ID,
    "type": "standard",
    "model": "ridge",
    "target_mode": "raw",
    "groups": ["legacy", "norm_core"],
    "notes": "current stable ridge baseline",
}

# P4 选模型（§4.9）特征集口径：
#   - 树喂“大集”让树自筛（含手工交互、保留 cross-z/cross-rank、regime 清广播脏列）；
#     Fork A（2026-05-27 拍板）= 第一版含交互，跑重要性扫描一次回答“树是否真用得上交互 + 反偏颇”。
#   - 线性候选（EN/Huber）沿用 X1 集做苹果对苹果；ridge-on-X1 即既有 X1 spec，不重复。
P4_TREE_GROUPS = ["legacy", "norm_core", "ofi_safe", "trade_impact", "conditional_momentum", "lag", "roll", "patch_summary", "cross_rank", "regime_tree"]
# P4-2b 精简版：按 P4-2' 重要性扫描结论剪掉 conditional_momentum（手工交互，0.90%≈0，树自建交互不需要）
P4_TREE_GROUPS_V2 = ["legacy", "norm_core", "ofi_safe", "trade_impact", "lag", "roll", "patch_summary", "cross_rank", "regime_tree"]
P4_LINEAR_GROUPS = ["legacy", "norm_core", "ofi_safe", "conditional_momentum_interaction"]  # = X1 集

# 全部历史实验（含 baseline）
ALL_SPECS: List[Dict] = [
    # R 系列：Ridge backbone 变体（最优基线对比）
    {"experiment_id": "R00_ridge_legacy", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy"], "notes": "ridge legacy only"},
    {"experiment_id": "R01_ridge_legacy_plus_core", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "base", "lag", "roll", "cross"], "notes": "ridge legacy plus core features"},
    {"experiment_id": "R02_ridge_legacy_plus_norm_core", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "norm_core"], "notes": "current stable ridge baseline"},
    {"experiment_id": "R03_ridge_legacy_plus_patch_summary", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "patch_summary"], "notes": "ridge legacy plus patch summary"},
    {"experiment_id": "R04_ridge_legacy_plus_cross_rank", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "cross_rank_features"], "notes": "ridge legacy plus cross-sectional ranks"},
    # B 系列：结构性 backbone
    {"experiment_id": "B6_common_residual", "type": "common_residual", "notes": "formal common residual branch"},
    {"experiment_id": "B7_soft_regime", "type": "soft_regime", "notes": "formal soft regime ensemble"},
    {"experiment_id": "B8_ridge_legacy_plus_core", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "base", "lag", "roll", "cross"], "notes": "formal ridge legacy plus core"},
    # 注（2026-05-26 清理）：各族 `_safe` group 当前在 registry 里 = 该 stage 的“全部列”，
    # “safe” 命名有误导（实为 “all/full”）。原 O6 / T4 / C3 三个 spec 已删除，因为它们与
    # O4 / T1 / C1 逐位重复：
    #   - ofi_safe ⊇ ofi_rank，故 O6(ofi_safe+ofi_rank) ≡ O4(ofi_safe)；
    #   - trade_impact_safe ≡ trade_impact（同为 stage 全列），故 T4 ≡ T1；
    #   - conditional_momentum 未在 group_columns 显式声明→默认取 stage 全列，等于
    #     conditional_momentum_safe，故 C3 ≡ C1。
    # 后续若要测“全族最大集”，直接用 O4 / T1 / C1，别再加 *_safe / *_all 重复 spec。
    # （group 重命名 _safe→_all 属更大的 registry 重构，留待单独处理。）
    # O 系列：OFI 动态订单流（O4 = OFI 最大集）
    {"experiment_id": "O1_R02_plus_ofi_raw", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "norm_core", "ofi_raw"], "notes": "R02 plus raw OFI"},
    {"experiment_id": "O2_R02_plus_ofi_dynamic", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "norm_core", "ofi_dynamic"], "notes": "R02 plus dynamic OFI"},
    {"experiment_id": "O3_R02_plus_ofi_rank", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "norm_core", "ofi_rank", "ofi_raw"], "notes": "R02 plus OFI cross ranks"},
    {"experiment_id": "O4_R02_plus_ofi_safe", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "norm_core", "ofi_safe"], "notes": "R02 plus 全部 OFI（ofi_safe=ofi stage 全列+cross ofi cs_z/cs_rank，OFI 最大集）"},
    {"experiment_id": "O5_R02_plus_ofi_raw_dynamic", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "norm_core", "ofi_raw", "ofi_dynamic"], "notes": "R02 plus raw and dynamic OFI"},
    # T 系列：成交冲击（T1 = 成交冲击最大集）
    {"experiment_id": "T1_R02_plus_trade_impact", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "norm_core", "trade_impact"], "notes": "R02 plus 全部成交冲击（trade_impact=stage 全列，最大集）"},
    {"experiment_id": "T2_R02_plus_trade_impact_dyn", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "norm_core", "trade_impact_dyn"], "notes": "R02 plus trade impact dynamic"},
    {"experiment_id": "T3_R02_plus_trade_impact_interaction", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "norm_core", "trade_impact_interaction"], "notes": "R02 plus trade impact interactions"},
    # C 系列：条件动量/反转（C1 = 条件动量最大集）
    {"experiment_id": "C1_R02_plus_conditional_momentum", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "norm_core", "conditional_momentum"], "notes": "R02 plus 全部条件动量（默认取 stage 全列，最大集）"},
    {"experiment_id": "C2_R02_plus_conditional_momentum_interaction", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "norm_core", "conditional_momentum_interaction"], "notes": "R02 plus conditional momentum interactions"},
    # X 系列：跨族组合（§4.5 组合不可加，必须当新 spec 重跑）。X1 = 两个最干净的小而真信号叠加：O4 的 ofi_safe + C2 的 conditional_momentum_interaction
    {"experiment_id": "X1_R02_plus_ofi_safe_condmom_interaction", "type": "standard", "model": "ridge", "target_mode": "raw", "groups": ["legacy", "norm_core", "ofi_safe", "conditional_momentum_interaction"], "notes": "R02 plus OFI safe + conditional momentum interactions (§4.5 cross-family combo of best sub-floor signals)"},
    # M 系列：P4 选模型（§4.9，模型为变量、各自最佳集；预钉小网格防多重比较）。
    # 初筛走 long-only（--profiles long_40d_5d）；决赛 2–3 个才上 expanding，打 X1 expanding 0.0668 靶子。
    # 线性候选（用 X1 集；ridge-on-X1 = 既有 X1，不重复）：
    {"experiment_id": "M_en_X1", "type": "standard", "model": "elasticnet", "target_mode": "raw", "groups": P4_LINEAR_GROUPS, "notes": "P4 model select: ElasticNet on X1 set"},
    {"experiment_id": "M_huber_X1", "type": "standard", "model": "huber", "target_mode": "raw", "groups": P4_LINEAR_GROUPS, "notes": "P4 model select: Huber on X1 set"},
    # 浅 ExtraTrees（树大集，depth 网格；leaf=500 当下限非约束，depth 是主正则器）：
    # n_jobs=8：ExtraTrees 在模型内(L2 线程级)并行建树，共享同一份训练数据、不复制内存。
    # 因 experiment_runner 把 LOKY_MAX_CPU_COUNT 钉成 1，自动探核失效，故必须显式给 n_jobs。
    {"experiment_id": "M_tree_d4", "type": "standard", "model": "tree_shallow", "target_mode": "raw", "groups": P4_TREE_GROUPS, "model_params": {"max_depth": 4, "n_jobs": 8}, "notes": "P4 model select: shallow ExtraTrees depth=4 on tree set"},
    {"experiment_id": "M_tree_d5", "type": "standard", "model": "tree_shallow", "target_mode": "raw", "groups": P4_TREE_GROUPS, "model_params": {"max_depth": 5, "n_jobs": 8}, "notes": "P4 model select: shallow ExtraTrees depth=5 on tree set"},
    {"experiment_id": "M_tree_d6", "type": "standard", "model": "tree_shallow", "target_mode": "raw", "groups": P4_TREE_GROUPS, "model_params": {"max_depth": 6, "n_jobs": 8}, "notes": "P4 model select: shallow ExtraTrees depth=6 on tree set"},
    # 浅 HistGB（树大集，depth/lr 网格；多轮 boosting 补浅 depth）：
    {"experiment_id": "M_histgb_d3", "type": "standard", "model": "histgb_shallow", "target_mode": "raw", "groups": P4_TREE_GROUPS, "model_params": {"max_depth": 3}, "notes": "P4 model select: shallow HistGB depth=3 on tree set"},
    {"experiment_id": "M_histgb_d4", "type": "standard", "model": "histgb_shallow", "target_mode": "raw", "groups": P4_TREE_GROUPS, "model_params": {"max_depth": 4}, "notes": "P4 model select: shallow HistGB depth=4 on tree set"},
    {"experiment_id": "M_histgb_d4_lr03", "type": "standard", "model": "histgb_shallow", "target_mode": "raw", "groups": P4_TREE_GROUPS, "model_params": {"max_depth": 4, "learning_rate": 0.03}, "notes": "P4 model select: shallow HistGB depth=4 lr=0.03 on tree set"},
    # P4-2b 精炼轮（2026-05-28）：补深度 d7/d8 + LightGBM，均用精简 V2 特征集（剪手工交互）
    # ExtraTrees 补深度（d4<d5<d6 单调递增还没到头，看拐点）：
    {"experiment_id": "M_tree_d7", "type": "standard", "model": "tree_shallow", "target_mode": "raw", "groups": P4_TREE_GROUPS_V2, "model_params": {"max_depth": 7, "n_jobs": 8}, "notes": "P4-2b: ExtraTrees depth=7 on trimmed V2 set"},
    {"experiment_id": "M_tree_d8", "type": "standard", "model": "tree_shallow", "target_mode": "raw", "groups": P4_TREE_GROUPS_V2, "model_params": {"max_depth": 8, "n_jobs": 8}, "notes": "P4-2b: ExtraTrees depth=8 on trimmed V2 set"},
    # LightGBM（GBDT 正牌选手，老师方案推荐；浅深度 + 保守正则防过拟合）：
    {"experiment_id": "M_lgbm_d4", "type": "standard", "model": "lgbm", "target_mode": "raw", "groups": P4_TREE_GROUPS_V2, "model_params": {"max_depth": 4, "num_leaves": 15, "n_jobs": 8}, "notes": "P4-2b: LightGBM depth=4 leaves=15 on trimmed V2 set"},
    {"experiment_id": "M_lgbm_d6", "type": "standard", "model": "lgbm", "target_mode": "raw", "groups": P4_TREE_GROUPS_V2, "model_params": {"max_depth": 6, "num_leaves": 31, "n_jobs": 8}, "notes": "P4-2b: LightGBM depth=6 leaves=31 on trimmed V2 set"},
]

# Ridge baseline 子集（快速复现用）
RIDGE_SPECS: List[Dict] = [s for s in ALL_SPECS if s["experiment_id"].startswith("R")]


# ================================================================== #
# 工具函数
# ================================================================== #

def _weighted_avg(profile_scores: Dict[str, Dict], key: str) -> float:
    """按 PROFILE_WEIGHTS 对各 profile 的指定指标加权平均"""
    total_w = 0.0
    total_v = 0.0
    for pname, scores in profile_scores.items():
        w = PROFILE_WEIGHTS.get(pname, 0.0)
        v = scores.get(key, np.nan)
        if not np.isnan(float(v)) and w > 0:
            total_v += w * float(v)
            total_w += w
    return float(total_v / total_w) if total_w > 0 else np.nan


def make_decision(row: Dict, baseline: Dict) -> Tuple[str, str]:
    """
    promote / review / reject 自动判定，严格实现 AGENTS §4.6 硬契约。

    coding agent 在循环里只认本函数返回的标签，所以判定表必须固化在这里、由单测锁住，
    不能让“散文比代码严”的地方往松处漂。

    判定表：
      delta_corr < 0.003                          → reject
      0.003 ≤ delta_corr < 0.005                  → review（边界增益，禁止直接 promote）
      delta_corr ≥ 0.005 且通过全部 promote 附加门槛 → promote
      其它                                          → review
    缺 expanding 结果时最高只能 review，不得 promote。

    依赖字段（均来自 build_leaderboard 的一行 row / baseline）：
      protocol_corr_mean / protocol_stability_score / protocol_daily_ic_mean
      {short,long,expanding}_corr_mean / {short,long,expanding}_corr_min
    """
    def _safe(d, k):
        v = d.get(k, np.nan)
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan

    def _has(d, k):
        return k in d and not np.isnan(_safe(d, k))

    base_corr = _safe(baseline, "protocol_corr_mean")
    corr = _safe(row, "protocol_corr_mean")
    if np.isnan(corr) or np.isnan(base_corr):
        return "unknown", "缺少 protocol_corr_mean"

    delta_corr = corr - base_corr

    # —— 安全地板：低于 +0.003 直接拒（§4.3：低于一个标准误，与噪声不可分）——
    if delta_corr < 0.003:
        return "reject", f"corr 提升不足（Δ={delta_corr:+.4f} < 0.003）"

    # —— 边界区间 [0.003, 0.005)：只能 review，禁止直接 promote ——
    if delta_corr < 0.005:
        return "review", f"边界增益（Δ={delta_corr:+.4f} ∈ [0.003,0.005)），送复核"

    # —— delta_corr ≥ 0.005：逐项核 promote 附加门槛，任一不过则降级为 review ——
    # (1) expanding 必须已单独跑过，否则最高 review（硬约束，不得 promote）
    if not _has(row, "expanding_corr_mean"):
        return "review", f"Δ={delta_corr:+.4f} 达标但缺 expanding 结果，最高只能 review"

    fails = []
    exp_corr = _safe(row, "expanding_corr_mean")

    # expanding 不为负
    if exp_corr < 0:
        fails.append(f"expanding 为负（{exp_corr:+.4f}）")

    # (2) short 与 expanding 相对基线同向为正
    short_delta = _safe(row, "short_corr_mean") - _safe(baseline, "short_corr_mean")
    exp_delta = exp_corr - _safe(baseline, "expanding_corr_mean")
    if not (short_delta > 0 and exp_delta > 0):
        fails.append(f"short/expanding 未同向为正（short Δ={short_delta:+.4f}, exp Δ={exp_delta:+.4f}）")

    # (3) long 不显著翻负：long 的 delta_corr ≥ -0.006（约一个均值标准误，§4.3）
    if _has(row, "long_corr_mean") and _has(baseline, "long_corr_mean"):
        long_delta = _safe(row, "long_corr_mean") - _safe(baseline, "long_corr_mean")
        if long_delta < -0.006:
            fails.append(f"long 显著翻负（Δ={long_delta:+.4f} < -0.006）")

    # (4) 每日 IC 不恶化（两边都有该字段时才核；#12 已保证生产环境存在）
    if _has(row, "protocol_daily_ic_mean") and _has(baseline, "protocol_daily_ic_mean"):
        dic_delta = _safe(row, "protocol_daily_ic_mean") - _safe(baseline, "protocol_daily_ic_mean")
        if dic_delta < -1e-6:
            fails.append(f"每日 IC 恶化（Δ={dic_delta:+.4f}）")

    # (5) stability 不降
    if _has(row, "protocol_stability_score") and _has(baseline, "protocol_stability_score"):
        stab_delta = _safe(row, "protocol_stability_score") - _safe(baseline, "protocol_stability_score")
        if stab_delta < -1e-6:
            fails.append(f"stability 下降（Δ={stab_delta:+.4f}）")

    # (6) 没有新的强负折：某 profile 候选 corr_min < -0.01，或由非负转负（§4.6 量化定义）
    neg_fold_hits = []
    for prefix in ("short", "long", "expanding"):
        ck = f"{prefix}_corr_min"
        if not _has(row, ck):
            continue
        cand_min = _safe(row, ck)
        base_min = _safe(baseline, ck) if _has(baseline, ck) else np.nan
        if cand_min < -0.01:
            neg_fold_hits.append(f"{prefix}(min={cand_min:+.4f}<-0.01)")
        elif cand_min < 0 and not np.isnan(base_min) and base_min >= 0:
            neg_fold_hits.append(f"{prefix}(min 由非负转负 {base_min:+.4f}→{cand_min:+.4f})")
    if neg_fold_hits:
        fails.append("出现新强负折：" + ", ".join(neg_fold_hits))

    if fails:
        return "review", f"Δ={delta_corr:+.4f} 达标但未过 promote 门槛：" + "；".join(fails)

    return "promote", f"稳定超基线（Δ={delta_corr:+.4f}，expanding={exp_corr:+.4f}，已过全部门槛）"


# ================================================================== #
# EvaluationProtocolRunner
# ================================================================== #

class EvaluationProtocolRunner:
    """
    滚动评测协议执行器。

    封装 ExperimentRunner，提供多 profile 横向对比、holdout 评测和 leaderboard 生成。
    """

    def __init__(self, experiment_runner: ExperimentRunner):
        self.runner = experiment_runner

    # ---------------------------------------------------------------- #
    # Fold 构造
    # ---------------------------------------------------------------- #

    def build_folds_for_profile(
        self,
        profile: RollingProfile,
        rolling_start: int,
        rolling_end: int,
        max_folds: Optional[int] = None,
    ) -> Tuple[List[RollingFold], List[FoldManifestEntry]]:
        """
        为指定 profile 构建 rolling folds 列表和 fold manifest。

        严格保证：train_end < embargo_start <= embargo_end < val_start
        """
        all_dates = self.runner.calendar.range(rolling_start, rolling_end)
        if not all_dates:
            return [], []

        embargo = max(0, profile.embargo)
        folds: List[RollingFold] = []
        manifest: List[FoldManifestEntry] = []
        fold_id = 0

        if profile.mode == "sliding":
            assert profile.train_window is not None, "sliding 模式必须设置 train_window"
            # cursor 指向 val_start 的索引
            cursor = profile.train_window + embargo
            while cursor + profile.val_window <= len(all_dates):
                train_end_idx = cursor - embargo        # train_dates 右边界（不含）
                train_dates = all_dates[max(0, train_end_idx - profile.train_window):train_end_idx]
                embargo_dates = all_dates[train_end_idx:train_end_idx + embargo] if embargo > 0 else []
                val_dates = all_dates[cursor:cursor + profile.val_window]

                if len(train_dates) >= profile.min_train_days and len(val_dates) > 0:
                    folds.append(RollingFold(fold_id=fold_id, train_dates=tuple(train_dates), val_dates=tuple(val_dates)))
                    manifest.append(FoldManifestEntry(
                        profile_name=profile.profile_name,
                        fold_id=fold_id,
                        train_start=train_dates[0],
                        train_end=train_dates[-1],
                        embargo_start=embargo_dates[0] if embargo_dates else train_dates[-1],
                        embargo_end=embargo_dates[-1] if embargo_dates else train_dates[-1],
                        val_start=val_dates[0],
                        val_end=val_dates[-1],
                        n_train_days=len(train_dates),
                        n_val_days=len(val_dates),
                    ))
                    fold_id += 1

                cursor += profile.step

        elif profile.mode == "expanding":
            # expanding：训练集从头扩张，最少 min_train_days 天
            cursor = profile.min_train_days   # cursor = 训练集长度（右边界，不含）
            start_idx = 0
            while cursor + embargo + profile.val_window <= len(all_dates):
                train_dates = all_dates[start_idx:cursor]
                embargo_dates = all_dates[cursor:cursor + embargo] if embargo > 0 else []
                val_dates = all_dates[cursor + embargo:cursor + embargo + profile.val_window]

                if len(train_dates) >= profile.min_train_days and len(val_dates) > 0:
                    folds.append(RollingFold(fold_id=fold_id, train_dates=tuple(train_dates), val_dates=tuple(val_dates)))
                    manifest.append(FoldManifestEntry(
                        profile_name=profile.profile_name,
                        fold_id=fold_id,
                        train_start=train_dates[0],
                        train_end=train_dates[-1],
                        embargo_start=embargo_dates[0] if embargo_dates else train_dates[-1],
                        embargo_end=embargo_dates[-1] if embargo_dates else train_dates[-1],
                        val_start=val_dates[0],
                        val_end=val_dates[-1],
                        n_train_days=len(train_dates),
                        n_val_days=len(val_dates),
                    ))
                    fold_id += 1

                cursor += profile.step

        else:
            raise ValueError(f"未知 profile.mode: {profile.mode}，应为 'sliding' 或 'expanding'")

        if max_folds is not None:
            folds = folds[:max_folds]
            manifest = manifest[:max_folds]

        return folds, manifest

    # ---------------------------------------------------------------- #
    # 单 profile 运行
    # ---------------------------------------------------------------- #

    def _make_oof_writer(self, oof_dir: str):
        """
        构造逐行 OOF 落盘回调（P5 融合用）。

        每个 (profile, experiment_id) 一个 HDF 文件，各折 append；
        行 schema = fold_id / date / symbol / interval / fret12 / pred，全为数值列。
        用 pytables（项目已有依赖）落 table 格式，append 友好、低内存。
        """
        def _writer(profile_name, fold_id, experiment_id, yval, pred_val):
            meta_cols = ["date", "symbol", "interval", "fret12"]
            frame = yval[meta_cols].copy()
            frame.insert(0, "fold_id", int(fold_id))
            frame["pred"] = np.asarray(pred_val, dtype=np.float32)
            frame = frame.reset_index(drop=True)
            safe = f"{profile_name}__{experiment_id}".replace("/", "_")
            path = os.path.join(oof_dir, f"{safe}.h5")
            frame.to_hdf(
                path, key="oof", mode="a", append=True,
                format="table", complevel=5, complib="zlib",
            )
        return _writer

    def run_profile(
        self,
        profile: RollingProfile,
        rolling_start: int,
        rolling_end: int,
        specs: List[Dict],
        max_folds: Optional[int] = None,
        oof_writer=None,
    ) -> Tuple[List[FoldManifestEntry], pd.DataFrame]:
        """
        在指定 profile 下运行所有 specs，返回 (manifest, fold_metrics_df)。

        fold_metrics_df 含 profile_name / fold 日期 / 各指标列。

        oof_writer：可选回调 (profile_name, fold_id, experiment_id, yval, pred_val) -> None，
        用于把每折逐行 OOF 预测落盘（P5 融合用）。为 None 时完全不触发，行为同旧版。
        """
        folds, manifest = self.build_folds_for_profile(profile, rolling_start, rolling_end, max_folds=max_folds)
        if not folds:
            return [], pd.DataFrame()

        rows = []
        for fold in folds:
            fold_split = SplitConfig(
                train_start=fold.train_dates[0],
                train_end=fold.train_dates[-1],
                val_start=fold.val_dates[0],
                val_end=fold.val_dates[-1],
                test_start=fold.val_dates[0],
                test_end=fold.val_dates[-1],
            )
            for spec in specs:
                try:
                    bundle = self.runner._evaluate_spec_on_fold(fold_split, spec)
                    row = self.runner._fold_metric_row(
                        fold_id=fold.fold_id,
                        experiment_id=spec["experiment_id"],
                        feature_set=bundle["feature_set"],
                        target_type=bundle["target_type"],
                        model_type=bundle["model_type"],
                        postprocess_type=bundle["postprocess_type"],
                        train_metrics=bundle["train_metrics"],
                        val_metrics=bundle["val_metrics"],
                        runtime_sec=bundle["runtime_sec"],
                        notes=spec.get("notes", ""),
                    )
                    row["profile_name"] = profile.profile_name
                    row["train_start"] = fold.train_dates[0]
                    row["train_end"] = fold.train_dates[-1]
                    row["val_start"] = fold.val_dates[0]
                    row["val_end"] = fold.val_dates[-1]
                    row["n_train_days"] = len(fold.train_dates)
                    row["n_val_days"] = len(fold.val_dates)
                    rows.append(row)
                    # P5 OOF 落盘：仅 standard spec 且 result 同时带 pred_val/yval 时触发。
                    if oof_writer is not None:
                        result = bundle.get("result", {})
                        pred_val = result.get("pred_val")
                        yval = result.get("yval")
                        if pred_val is not None and yval is not None:
                            oof_writer(
                                profile.profile_name,
                                fold.fold_id,
                                spec["experiment_id"],
                                yval,
                                pred_val,
                            )
                except Exception as e:
                    # 记录失败而不中断，便于调试
                    rows.append({
                        "profile_name": profile.profile_name,
                        "fold_id": fold.fold_id,
                        "experiment_id": spec["experiment_id"],
                        "train_start": fold.train_dates[0],
                        "train_end": fold.train_dates[-1],
                        "val_start": fold.val_dates[0],
                        "val_end": fold.val_dates[-1],
                        "n_train_days": len(fold.train_dates),
                        "n_val_days": len(fold.val_dates),
                        "val_corr": np.nan,
                        "val_mse": np.nan,
                        "val_r2": np.nan,
                        "notes": f"ERROR: {str(e)[:200]}",
                    })

        return manifest, pd.DataFrame(rows)

    # ---------------------------------------------------------------- #
    # Profile 汇总
    # ---------------------------------------------------------------- #

    def summarize_profile(self, fold_df: pd.DataFrame, profile_name: str) -> pd.DataFrame:
        """聚合单个 profile 下每个实验的汇总指标"""
        if fold_df.empty:
            return pd.DataFrame()

        summary_rows = []
        for experiment_id, group in fold_df.groupby("experiment_id", sort=False):
            group = group.sort_values("fold_id")
            val_corrs = group["val_corr"].dropna().tolist()
            val_mses = group["val_mse"].dropna().tolist() if "val_mse" in group.columns else []
            val_r2s = group["val_r2"].dropna().tolist() if "val_r2" in group.columns else []
            n_folds = len(group)

            row: Dict[str, Any] = {
                "profile_name": profile_name,
                "experiment_id": experiment_id,
                "model_type": group["model_type"].iloc[0] if "model_type" in group.columns else "",
                "feature_set": group["feature_set"].iloc[0] if "feature_set" in group.columns else "",
                "target_type": group["target_type"].iloc[0] if "target_type" in group.columns else "",
                "n_folds": n_folds,
            }

            if val_corrs:
                row["rolling_corr_mean"] = float(np.mean(val_corrs))
                row["rolling_corr_std"] = float(np.std(val_corrs, ddof=0)) if len(val_corrs) > 1 else 0.0
                row["rolling_corr_min"] = float(np.min(val_corrs))
                row["rolling_corr_median"] = float(np.median(val_corrs))
                row["positive_fold_rate"] = float(sum(c > 0 for c in val_corrs) / len(val_corrs))
                row["stability_score"] = row["rolling_corr_mean"] - 0.7 * row["rolling_corr_std"]
            else:
                for k in ["rolling_corr_mean", "rolling_corr_std", "rolling_corr_min",
                          "rolling_corr_median", "positive_fold_rate", "stability_score"]:
                    row[k] = np.nan

            if val_mses:
                row["rolling_mse_mean"] = float(np.mean(val_mses))
                row["rolling_mse_std"] = float(np.std(val_mses, ddof=0)) if len(val_mses) > 1 else 0.0
            else:
                row["rolling_mse_mean"] = np.nan
                row["rolling_mse_std"] = np.nan

            if val_r2s:
                row["rolling_r2_mean"] = float(np.mean(val_r2s))
                row["rolling_r2_min"] = float(np.min(val_r2s))
            else:
                row["rolling_r2_mean"] = np.nan
                row["rolling_r2_min"] = np.nan

            if "daily_corr_mean" in group.columns:
                row["daily_corr_mean"] = float(group["daily_corr_mean"].mean())
            if "daily_corr_std" in group.columns:
                row["daily_corr_std"] = float(group["daily_corr_std"].mean())
            if "train_val_corr_gap" in group.columns:
                row["train_val_corr_gap_mean"] = float(group["train_val_corr_gap"].dropna().mean()) if group["train_val_corr_gap"].notna().any() else np.nan
            if "runtime_sec" in group.columns:
                row["runtime_sec_sum"] = float(group["runtime_sec"].sum())

            summary_rows.append(row)

        return pd.DataFrame(summary_rows)

    # ---------------------------------------------------------------- #
    # Holdout 评测
    # ---------------------------------------------------------------- #

    def run_holdout(
        self,
        train_start: int,
        train_end: int,
        holdout_start: int,
        holdout_end: int,
        specs: List[Dict],
        holdout_name: str = "holdout",
    ) -> pd.DataFrame:
        """
        单次 holdout 评测（不参与 rolling 汇总和 protocol_stability_score 计算）。

        holdout_name: "review"（11月）或 "final"（12月）
        """
        split_config = SplitConfig(
            train_start=train_start,
            train_end=train_end,
            val_start=holdout_start,
            val_end=holdout_end,
            test_start=holdout_start,
            test_end=holdout_end,
        )
        rows = []
        for spec in specs:
            start_ts = time.time()
            try:
                bundle = self.runner._evaluate_spec_on_fold(split_config, spec)
                rows.append({
                    "holdout_name": holdout_name,
                    "experiment_id": spec["experiment_id"],
                    "model_type": bundle["model_type"],
                    "feature_set": bundle["feature_set"],
                    "target_type": bundle["target_type"],
                    "train_start": train_start,
                    "train_end": train_end,
                    "holdout_start": holdout_start,
                    "holdout_end": holdout_end,
                    "holdout_corr": bundle["val_metrics"]["corr"],
                    "holdout_mse": bundle["val_metrics"]["mse"],
                    "holdout_r2": bundle["val_metrics"]["r2"],
                    "runtime_sec": float(time.time() - start_ts),
                    "notes": spec.get("notes", ""),
                })
            except Exception as e:
                rows.append({
                    "holdout_name": holdout_name,
                    "experiment_id": spec["experiment_id"],
                    "holdout_corr": np.nan,
                    "holdout_mse": np.nan,
                    "holdout_r2": np.nan,
                    "notes": f"ERROR: {str(e)[:200]}",
                })
        return pd.DataFrame(rows)

    # ---------------------------------------------------------------- #
    # Leaderboard 构建
    # ---------------------------------------------------------------- #

    def build_leaderboard(
        self,
        profile_summaries: Dict[str, pd.DataFrame],
        baseline_id: str = BASELINE_ID,
        review_holdout_df: Optional[pd.DataFrame] = None,
        final_holdout_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """构建跨 profile 加权综合 leaderboard，附带 baseline delta 和自动 decision"""
        # 收集所有实验 ID
        all_ids: List[str] = []
        seen: set = set()
        for df in profile_summaries.values():
            if df.empty or "experiment_id" not in df.columns:
                continue
            for eid in df["experiment_id"].tolist():
                if eid not in seen:
                    all_ids.append(eid)
                    seen.add(eid)

        if not all_ids:
            return pd.DataFrame()

        rows = []
        for experiment_id in all_ids:
            row: Dict[str, Any] = {"experiment_id": experiment_id}
            profile_scores: Dict[str, Dict] = {}

            # 各 profile 指标
            for pname, pdf in profile_summaries.items():
                if pdf.empty or "experiment_id" not in pdf.columns:
                    continue
                exp_rows = pdf[pdf["experiment_id"] == experiment_id]
                if exp_rows.empty:
                    continue
                r = exp_rows.iloc[0]
                # 取 profile 名称前缀（short/medium/long/expanding）
                prefix = pname.split("_")[0]
                for metric in ["corr_mean", "corr_std", "corr_min", "stability_score",
                               "n_folds", "positive_fold_rate", "daily_corr_mean", "daily_corr_std"]:
                    src_key = f"rolling_{metric}" if metric in ["corr_mean", "corr_std", "corr_min"] else metric
                    row[f"{prefix}_{metric}"] = r.get(src_key, np.nan)
                # 每日截面 IC-IR（每日 IC 的 均值÷波动）：区分“稳的选股钱”与“虚胖”，防除零
                _dic_m = float(r.get("daily_corr_mean", np.nan))
                _dic_s = float(r.get("daily_corr_std", np.nan))
                row[f"{prefix}_daily_ic_ir"] = (
                    _dic_m / _dic_s
                    if not (np.isnan(_dic_m) or np.isnan(_dic_s)) and abs(_dic_s) > 1e-12
                    else np.nan
                )

                profile_scores[pname] = {
                    "corr_mean": r.get("rolling_corr_mean", np.nan),
                    "stability": r.get("stability_score", np.nan),
                    "corr_min": r.get("rolling_corr_min", np.nan),
                    "mse_mean": r.get("rolling_mse_mean", np.nan),
                    "r2_mean": r.get("rolling_r2_mean", np.nan),
                    "positive_fold_rate": r.get("positive_fold_rate", np.nan),
                    "daily_corr_mean": r.get("daily_corr_mean", np.nan),
                    "daily_corr_std": r.get("daily_corr_std", np.nan),
                }
                # 顺便从第一个匹配的 profile 取元信息
                if "model_type" not in row:
                    row["model_type"] = r.get("model_type", "")
                    row["feature_set"] = r.get("feature_set", "")
                    row["target_type"] = r.get("target_type", "")

            # 加权综合指标
            row["protocol_corr_mean"] = _weighted_avg(profile_scores, "corr_mean")
            row["protocol_stability_score"] = _weighted_avg(profile_scores, "stability")
            row["protocol_corr_min"] = _weighted_avg(profile_scores, "corr_min")
            row["protocol_mse_mean"] = _weighted_avg(profile_scores, "mse_mean")
            row["protocol_r2_mean"] = _weighted_avg(profile_scores, "r2_mean")
            row["protocol_positive_fold_rate"] = _weighted_avg(profile_scores, "positive_fold_rate")
            # 每日截面 IC 一等指标：均值 + IC-IR（均值÷波动），与池化 corr 并列进主视图
            row["protocol_daily_ic_mean"] = _weighted_avg(profile_scores, "daily_corr_mean")
            _pic_std = _weighted_avg(profile_scores, "daily_corr_std")
            row["protocol_daily_ic_ir"] = (
                row["protocol_daily_ic_mean"] / _pic_std
                if not np.isnan(_pic_std) and abs(_pic_std) > 1e-12 else np.nan
            )

            # holdout 结果（不参与 protocol_stability_score）
            if review_holdout_df is not None and not review_holdout_df.empty:
                rv = review_holdout_df[review_holdout_df["experiment_id"] == experiment_id]
                if not rv.empty:
                    row["review_holdout_corr"] = rv.iloc[0].get("holdout_corr", np.nan)
                    row["review_holdout_mse"] = rv.iloc[0].get("holdout_mse", np.nan)
                    row["review_holdout_r2"] = rv.iloc[0].get("holdout_r2", np.nan)

            if final_holdout_df is not None and not final_holdout_df.empty:
                fv = final_holdout_df[final_holdout_df["experiment_id"] == experiment_id]
                if not fv.empty:
                    row["final_holdout_corr"] = fv.iloc[0].get("holdout_corr", np.nan)
                    row["final_holdout_mse"] = fv.iloc[0].get("holdout_mse", np.nan)
                    row["final_holdout_r2"] = fv.iloc[0].get("holdout_r2", np.nan)

            rows.append(row)

        lb = pd.DataFrame(rows)

        # baseline delta
        baseline_rows = lb[lb["experiment_id"] == baseline_id]
        if not baseline_rows.empty:
            baseline = baseline_rows.iloc[0]
            base_corr = float(baseline.get("protocol_corr_mean", np.nan))
            base_stab = float(baseline.get("protocol_stability_score", np.nan))
            base_mse = float(baseline.get("protocol_mse_mean", np.nan))
            base_r2 = float(baseline.get("protocol_r2_mean", np.nan))

            lb["baseline_delta_corr"] = lb["protocol_corr_mean"].astype(float) - base_corr
            lb["baseline_delta_stability"] = lb["protocol_stability_score"].astype(float) - base_stab
            if not np.isnan(base_mse) and abs(base_mse) > 1e-12:
                lb["baseline_delta_mse_pct"] = (lb["protocol_mse_mean"].astype(float) - base_mse) / abs(base_mse)
            lb["baseline_delta_r2"] = lb["protocol_r2_mean"].astype(float) - base_r2

        # 自动 decision
        if not baseline_rows.empty:
            baseline_dict = baseline_rows.iloc[0].to_dict()
            decisions, reasons = [], []
            for _, r in lb.iterrows():
                d, rsn = make_decision(r.to_dict(), baseline_dict)
                decisions.append(d)
                reasons.append(rsn)
            lb["decision"] = decisions
            lb["reason"] = reasons
            # 基线本身标记
            lb.loc[lb["experiment_id"] == baseline_id, "decision"] = "baseline"
            lb.loc[lb["experiment_id"] == baseline_id, "reason"] = "当前稳定基线"

        # 头条按 protocol_corr_mean 降序（§4.6：对齐老师评分；stability 作并排守门指标，不再当第一排序键）
        if "protocol_corr_mean" in lb.columns:
            lb = lb.sort_values("protocol_corr_mean", ascending=False).reset_index(drop=True)

        return lb

    # ---------------------------------------------------------------- #
    # 主入口
    # ---------------------------------------------------------------- #

    def run_full_protocol(
        self,
        rolling_start: int,
        rolling_end: int,
        specs: List[Dict],
        profiles: Optional[List[RollingProfile]] = None,
        max_folds: Optional[int] = None,
        include_review_holdout: bool = False,
        review_train_start: Optional[int] = None,
        review_train_end: Optional[int] = None,
        review_holdout_start: Optional[int] = None,
        review_holdout_end: Optional[int] = None,
        include_final_holdout: bool = False,
        final_train_start: Optional[int] = None,
        final_train_end: Optional[int] = None,
        final_holdout_start: Optional[int] = None,
        final_holdout_end: Optional[int] = None,
        baseline_id: str = BASELINE_ID,
        n_workers: int = 1,
        resume: bool = False,
        output_dir: Optional[str] = None,
        run_id: Optional[str] = None,
        dump_oof: bool = False,
    ) -> Dict[str, Any]:
        """
        主入口：运行完整三层评测协议。

        流程：
          1. 对每个 profile 运行所有 specs → fold_metrics
          2. 聚合每个 profile 的 summary
          3. 可选：review holdout（11月）
          4. 可选：final holdout（12月，尽量少跑）
          5. 构建 leaderboard（含 baseline delta 和 decision）
          6. 保存所有输出到 output_dir/<run_id>/

        返回包含所有结果 DataFrame 的字典。
        """
        if profiles is None:
            profiles = ROLLING_PROFILES
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # P5 OOF 落盘 writer：仅串行路径支持（并行 scheduler 只回传 metric 行，拿不到逐行预测）。
        oof_writer = None
        if dump_oof:
            if n_workers > 1:
                print("[Protocol][OOF] 警告：--dump-oof 仅支持串行（--n-workers 1），当前并行模式跳过落盘")
            elif not output_dir:
                print("[Protocol][OOF] 警告：未指定 output_dir，跳过 OOF 落盘")
            else:
                oof_dir = os.path.join(output_dir, run_id, "oof")
                os.makedirs(oof_dir, exist_ok=True)
                oof_writer = self._make_oof_writer(oof_dir)
                print(f"[Protocol][OOF] 开启逐行 OOF 落盘 → {oof_dir}")

        print(f"\n[Protocol] run_id={run_id}")
        print(f"[Protocol] rolling 范围：{rolling_start} ~ {rolling_end}")
        print(f"[Protocol] profiles: {[p.profile_name for p in profiles]}")
        print(f"[Protocol] specs 数量: {len(specs)}")
        if max_folds:
            print(f"[Protocol] max_folds={max_folds}（调试模式）")

        all_manifests: List[FoldManifestEntry] = []
        all_fold_metrics: List[pd.DataFrame] = []
        profile_summaries: Dict[str, pd.DataFrame] = {}
        _parallel_wrote_fold_metrics = False  # 并行模式下 scheduler 已增量落盘，末尾跳过重写

        if n_workers > 1 and ParallelScheduler is not None:
            # ── 并行路径 ─────────────────────────────────────────────────────
            # 1. 主进程统一构建所有 fold manifest
            profiles_with_folds = []
            for profile in profiles:
                folds, manifest = self.build_folds_for_profile(
                    profile, rolling_start, rolling_end, max_folds=max_folds
                )
                profiles_with_folds.append((profile, folds))
                all_manifests.extend(manifest)

            # 2. 提前创建输出目录，配置 scheduler 落盘路径（供 resume 使用）
            _run_dir_parallel = os.path.join(output_dir, run_id) if output_dir else None
            if _run_dir_parallel:
                os.makedirs(_run_dir_parallel, exist_ok=True)
            _fold_metrics_csv = (
                os.path.join(_run_dir_parallel, "fold_metrics.csv")
                if _run_dir_parallel else None
            )

            # 3. 并发执行
            h5dir = self.runner.loader.h5dir
            feature_dir = getattr(self.runner, "feature_dir", "data/features")
            scheduler = ParallelScheduler(
                h5dir,
                feature_dir=feature_dir,
                n_workers=n_workers,
                target_winsorize_config=self.runner.get_target_winsorize_config(),
                feature_dtype=getattr(self.runner, "feature_dtype", "float32"),
                ridge_alpha=self.runner.get_ridge_alpha(),
                train_subsample_frac=self.runner.get_train_subsample_frac(),
            )
            if _fold_metrics_csv:
                scheduler.set_output_path(_fold_metrics_csv)
                _parallel_wrote_fold_metrics = True

            merged_fold_df = scheduler.run(profiles_with_folds, specs, resume=resume)

            # 4. 按 profile 拆分，分别做 summarize_profile
            if not merged_fold_df.empty:
                all_fold_metrics.append(merged_fold_df)
            for profile in profiles:
                pname = profile.profile_name
                profile_df = (
                    merged_fold_df[merged_fold_df["profile_name"] == pname].copy()
                    if not merged_fold_df.empty else pd.DataFrame()
                )
                if not profile_df.empty:
                    n_folds = profile_df["fold_id"].nunique()
                    n_exp = profile_df["experiment_id"].nunique()
                    print(f"  [Profile] {pname}: {n_folds} folds × {n_exp} experiments = {len(profile_df)} rows")
                else:
                    print(f"  [Profile] {pname}: 无有效结果")
                summary = self.summarize_profile(profile_df, pname)
                profile_summaries[pname] = summary

        else:
            # ── 串行路径（原有逻辑，完全不变）────────────────────────────────
            for profile in profiles:
                print(f"\n[Profile] {profile.profile_name} (mode={profile.mode})")
                manifest, fold_df = self.run_profile(
                    profile, rolling_start, rolling_end, specs, max_folds=max_folds,
                    oof_writer=oof_writer,
                )
                all_manifests.extend(manifest)
                if not fold_df.empty:
                    all_fold_metrics.append(fold_df)
                    n_folds = fold_df["fold_id"].nunique()
                    n_exp = fold_df["experiment_id"].nunique()
                    print(f"  → {n_folds} folds × {n_exp} experiments = {len(fold_df)} rows")
                else:
                    print("  → 无有效 fold（日期范围不足）")

                summary = self.summarize_profile(fold_df, profile.profile_name)
                profile_summaries[profile.profile_name] = summary

        # 组装汇总表
        fold_manifest_df = (
            pd.DataFrame([vars(m) for m in all_manifests])
            if all_manifests else pd.DataFrame()
        )
        fold_metrics_df = (
            pd.concat(all_fold_metrics, ignore_index=True)
            if all_fold_metrics else pd.DataFrame()
        )
        profile_summary_df = (
            pd.concat(
                [df for df in profile_summaries.values() if not df.empty],
                ignore_index=True,
            )
            if any(not df.empty for df in profile_summaries.values())
            else pd.DataFrame()
        )

        # Review holdout（11月）
        review_holdout_df: Optional[pd.DataFrame] = None
        if include_review_holdout and all(
            x is not None for x in [review_train_start, review_train_end, review_holdout_start, review_holdout_end]
        ):
            print(f"\n[Holdout] review：train {review_train_start}~{review_train_end}, holdout {review_holdout_start}~{review_holdout_end}")
            review_holdout_df = self.run_holdout(
                review_train_start, review_train_end,
                review_holdout_start, review_holdout_end,
                specs, holdout_name="review",
            )

        # Final holdout（12月）
        final_holdout_df: Optional[pd.DataFrame] = None
        if include_final_holdout and all(
            x is not None for x in [final_train_start, final_train_end, final_holdout_start, final_holdout_end]
        ):
            print(f"\n[Holdout] final：train {final_train_start}~{final_train_end}, holdout {final_holdout_start}~{final_holdout_end}")
            final_holdout_df = self.run_holdout(
                final_train_start, final_train_end,
                final_holdout_start, final_holdout_end,
                specs, holdout_name="final",
            )

        # Leaderboard
        print("\n[Protocol] 构建 leaderboard...")
        leaderboard_df = self.build_leaderboard(
            profile_summaries,
            baseline_id=baseline_id,
            review_holdout_df=review_holdout_df,
            final_holdout_df=final_holdout_df,
        )

        # 保存输出
        if output_dir:
            run_dir = os.path.join(output_dir, run_id)
            os.makedirs(run_dir, exist_ok=True)

            config_data = {
                "run_id": run_id,
                "rolling_start": rolling_start,
                "rolling_end": rolling_end,
                "profiles": [vars(p) for p in profiles],
                "specs": [s["experiment_id"] for s in specs],
                "max_folds": max_folds,
                "baseline_id": baseline_id,
                "target_winsorize_config": self.runner.get_target_winsorize_config(),
                "feature_dtype": getattr(self.runner, "feature_dtype", "float32"),
                "ridge_alpha": self.runner.get_ridge_alpha(),
                "include_review_holdout": include_review_holdout,
                "include_final_holdout": include_final_holdout,
                "review_train_start": review_train_start,
                "review_train_end": review_train_end,
                "review_holdout_start": review_holdout_start,
                "review_holdout_end": review_holdout_end,
                "final_train_start": final_train_start,
                "final_train_end": final_train_end,
                "final_holdout_start": final_holdout_start,
                "final_holdout_end": final_holdout_end,
            }
            with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2, default=str)

            if not fold_manifest_df.empty:
                fold_manifest_df.to_csv(os.path.join(run_dir, "fold_manifest.csv"), index=False, encoding="utf-8-sig")
            if not fold_metrics_df.empty and not _parallel_wrote_fold_metrics:
                fold_metrics_df.to_csv(os.path.join(run_dir, "fold_metrics.csv"), index=False, encoding="utf-8-sig")
            if not profile_summary_df.empty:
                profile_summary_df.to_csv(os.path.join(run_dir, "profile_summary.csv"), index=False, encoding="utf-8-sig")
            if not leaderboard_df.empty:
                leaderboard_df.to_csv(os.path.join(run_dir, "leaderboard.csv"), index=False, encoding="utf-8-sig")
            if review_holdout_df is not None and not review_holdout_df.empty:
                review_holdout_df.to_csv(os.path.join(run_dir, "review_holdout.csv"), index=False, encoding="utf-8-sig")
            if final_holdout_df is not None and not final_holdout_df.empty:
                final_holdout_df.to_csv(os.path.join(run_dir, "final_holdout.csv"), index=False, encoding="utf-8-sig")

            # —— 复现审计快照：特征 manifest + 本次解析的特征列清单 ——
            # 放在主结果落盘之后；快照失败只告警，绝不影响已保存的 leaderboard 等主结果。
            try:
                fl = getattr(self.runner, "feature_loader", None)
                manifest_path = getattr(fl, "manifest_path", None) if fl is not None else None
                if manifest_path is not None and os.path.exists(str(manifest_path)):
                    with open(str(manifest_path), "r", encoding="utf-8") as f:
                        manifest_payload = json.load(f)
                    with open(os.path.join(run_dir, "manifest_snapshot.json"), "w", encoding="utf-8") as f:
                        json.dump(manifest_payload, f, ensure_ascii=False, indent=2, default=str)

                registry = getattr(fl, "registry", None) if fl is not None else None
                if registry is not None:
                    all_groups = sorted({g for s in specs for g in s.get("groups", [])})
                    resolved = registry.resolve_groups(all_groups) if all_groups else {}
                    resolved_payload = {
                        "feature_dir": str(getattr(self.runner, "feature_dir", "")),
                        "manifest_path": str(manifest_path) if manifest_path is not None else "",
                        "all_groups": list(all_groups),
                        "resolved_stage_columns": {k: list(v) for k, v in resolved.items()},
                        "specs_groups": {s["experiment_id"]: list(s.get("groups", [])) for s in specs},
                    }
                    try:
                        resolved_payload["stage_code_hash"] = {
                            st: registry.code_hash(st) for st in resolved.keys()
                        }
                    except Exception:
                        pass
                    with open(os.path.join(run_dir, "resolved_columns.json"), "w", encoding="utf-8") as f:
                        json.dump(resolved_payload, f, ensure_ascii=False, indent=2, default=str)
            except Exception as e:
                print(f"[Protocol] 警告：特征快照写盘失败（不影响主结果）：{e}")

            print(f"\n[Protocol] 输出已保存至: {run_dir}")

        return {
            "run_id": run_id,
            "fold_manifest": fold_manifest_df,
            "fold_metrics": fold_metrics_df,
            "profile_summary": profile_summary_df,
            "profile_summaries": profile_summaries,
            "leaderboard": leaderboard_df,
            "review_holdout": review_holdout_df,
            "final_holdout": final_holdout_df,
        }
