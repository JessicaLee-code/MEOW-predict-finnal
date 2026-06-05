"""
DL 损失模块 —— 可微 Pearson + 截面内 Pearson + ``MSE + λ·(1−corr)`` 组合损失

这是 DL 主线"与评分对齐"的关键一环（AGENT_TASK §3 / §定稿）。老师精度分 =
⅓ MSE + ⅓ Pearson + ⅓ R²，其中 **Pearson / 每日截面 IC 那一维**恰好是截面模型
一次预测整快照时、跨票方向的相关系数——所以在损失里直接放一个"截面内 Pearson"
项，就等于在直接优化被打分的量。这是逐票模型做不到、截面模型独有的杠杆。

设计要点：
- **脊柱 torch-free**：本模块可被脊柱安全 import（顶层不 import torch）；torch 只在
  组合损失/可微 Pearson **被调用时**惰性导入（``_require_torch``），与 ``dl_models``
  里卡带的 torch 边界一致。
- **数值稳定**：中心化 + 分母里加 eps 防 sqrt(0)/除零；样本 < 2 或零方差时退化为 0
  且保持计算图（梯度为 0，不产生 NaN）。
- **numpy 对拍参考**：``pearson_numpy`` 是 torch-free 的金标准实现，单测用它对拍
  torch 版本的数值正确性。

损失口径（§定稿 第 1 条，焊死、无开关）：
    loss = MSE + λ·(1 − 截面内 Pearson)，  λ ∈ {0, 0.3}
- λ=0 即纯 MSE（消融下界）；
- 量纲项固定 MSE（**不做 MSE/Huber 开关**，避免出错）；
- corr 项 = 截面内 Pearson：GRU 逐票卡带用"整批当一个截面"的粗代理（1D Pearson），
  截面卡带用"每 (date,interval) 快照一个截面"的真口径（masked 2D Pearson）。
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


def _require_torch():
    """惰性导入 torch（脊柱可 torch-free import 本模块；只有真正算可微损失时才需要）。"""
    try:
        import torch  # noqa: WPS433
    except ImportError as e:  # pragma: no cover - 环境无 torch 时给出明确指引
        raise ImportError(
            "dl_losses 的可微损失需要 PyTorch；请在卡带运行环境安装 torch 后重试。"
        ) from e
    return torch


# ================================================================== #
# numpy 金标准（torch-free，单测对拍用）
# ================================================================== #

def pearson_numpy(pred, target, eps: float = 1e-8) -> float:
    """
    Pearson 相关系数的 numpy 参考实现（torch-free）。

    口径与下方 ``pearson_torch`` 完全一致：中心化后 cov / (std_p·std_t)，用
    sum-of-squares 形式（尺度无关，sum 与 mean 不影响比值）。样本 < 2 或任一侧
    零方差（常量）→ 返回 0（相关无定义时的安全退化）。
    """
    p = np.asarray(pred, dtype=np.float64).ravel()
    t = np.asarray(target, dtype=np.float64).ravel()
    if p.size < 2:
        return 0.0
    pc = p - p.mean()
    tc = t - t.mean()
    ss_p = float((pc * pc).sum())
    ss_t = float((tc * tc).sum())
    denom = np.sqrt(ss_p * ss_t)
    if denom < eps:
        return 0.0
    return float((pc * tc).sum() / denom)


def fit_linear_rescale_numpy(raw_pred, y, eps: float = 1e-12):
    """
    OLS 全局线性重标定 ``a·ŷ + b``：在训练段最小化 ``||y − (a·raw + b)||²`` 求 (a, b)。

    §定稿 第 2 条：预测阶段**永远**套这个 rescale（fit-on-train / apply-on-val，零泄漏）。
    机理：OLS 重标定后训练段上 ``R² = corr²``——故只要 corr>0 就自动满足 R²≥0 硬底，
    且 corr 尺度/平移不变、一分不掉。于是选 λ 只按 max corr，corr 与 R² 不再冲突。

    退化保护：raw 近常量（零方差）时斜率无定义 → 返回 ``a=0, b=mean(y)``，即预测训练
    均值（R²=0，不为负），安全不爆。
    """
    raw = np.asarray(raw_pred, dtype=np.float64).ravel()
    yy = np.asarray(y, dtype=np.float64).ravel()
    if raw.size < 2:
        return 1.0, 0.0
    rm = float(raw.mean())
    ym = float(yy.mean())
    var = float(((raw - rm) ** 2).sum())
    if var < eps:
        return 0.0, ym
    a = float(((raw - rm) * (yy - ym)).sum() / var)
    b = float(ym - a * rm)
    return a, b


# ================================================================== #
# 可微 Pearson（torch）
# ================================================================== #

def pearson_torch(pred, target, eps: float = 1e-8):
    """
    单组可微 Pearson（1D tensor）。

    用于 GRU 逐票卡带：把整个 minibatch 当成一个"截面"算 corr（§3 说明这是粗代理，
    minibatch 混了随机 (票,时刻)、不是真截面 IC；真口径在截面卡带的 masked 版）。

    数值稳定：分母 ``sqrt(ss_p·ss_t + eps)`` 同时防 sqrt(0) 的无穷梯度与除零；常量/
    样本不足时 num=0 → corr=0，梯度有限不 NaN。返回 0-d tensor（保留计算图）。
    """
    if pred.numel() < 2:
        return pred.sum() * 0.0
    pc = pred - pred.mean()
    tc = target - target.mean()
    num = (pc * tc).sum()
    ss_p = (pc * pc).sum()
    ss_t = (tc * tc).sum()
    torch = _require_torch()
    denom = torch.sqrt(ss_p * ss_t + eps)
    return num / denom


def masked_cross_section_pearson_torch(pred, target, mask, eps: float = 1e-8):
    """
    masked 截面内 Pearson（2D ``[B, N]`` + ``mask[B, N]``），向量化、无 Python 循环。

    语义（截面卡带真口径）：每一行 = 同一 ``(date, interval)`` 快照的 N 只在场票
    （pad 位由 mask=False 屏蔽）；逐行算"跨票" Pearson（即被老师打分的每日截面 IC
    那一维），再对**有效票数 ≥ 2 的行**求平均（每个截面等权，对齐 pooled/daily-IC）。

    - 均值/中心化都只在 mask 有效位上做（pad 不污染统计）；
    - 行内有效票 < 2 → 该行 corr 无定义、不计入平均；
    - 全部行都无效 → 返回 0（保留计算图）。
    """
    torch = _require_torch()
    m = mask.to(pred.dtype)                              # [B, N]，1=在场票 0=pad
    cnt = m.sum(dim=1, keepdim=True)                     # [B, 1] 每行有效票数
    safe_cnt = cnt.clamp(min=1.0)
    pm = (pred * m).sum(dim=1, keepdim=True) / safe_cnt  # 行均值（仅有效位）
    tm = (target * m).sum(dim=1, keepdim=True) / safe_cnt
    pc = (pred - pm) * m                                 # 中心化后再屏蔽 pad（pad→0 不进 sum）
    tc = (target - tm) * m
    num = (pc * tc).sum(dim=1)                           # [B]
    ss_p = (pc * pc).sum(dim=1)
    ss_t = (tc * tc).sum(dim=1)
    corr = num / torch.sqrt(ss_p * ss_t + eps)           # [B] 逐行 corr
    valid = cnt.squeeze(1) >= 2.0                        # 有效票 ≥2 的行才算 corr
    if bool(valid.any()):
        return corr[valid].mean()
    return pred.sum() * 0.0


# ================================================================== #
# 组合损失工厂
# ================================================================== #

def make_loss(lambda_corr: float = 0.0, eps: float = 1e-8) -> Callable:
    """
    构造组合损失 ``loss = MSE + λ·(1 − 截面内 Pearson)``（§定稿 第 1 条，无开关）。

    返回的 ``loss_fn(pred, target, mask=None)`` 两种用法：
    - **逐票批（GRU）**：``mask=None``、``pred/target`` 为 1D ``[B]``——量纲项 = 批 MSE，
      corr 项 = 整批 1D Pearson（粗代理）。
    - **截面批（XSection）**：传 ``mask[B,N]``、``pred/target`` 为 2D ``[B,N]``——量纲项 =
      仅有效位的 MSE，corr 项 = masked 逐快照 Pearson 平均（真口径）。

    λ=0 时直接返回纯量纲项（短路 corr 计算，等价 ``MSELoss``，消融下界）。
    """
    lam = float(lambda_corr)

    def loss_fn(pred, target, mask=None):
        if mask is not None:
            # 截面批：量纲项 MSE 也只在有效位上算（pad 不进分子分母）。
            m = mask.to(pred.dtype)
            denom = m.sum().clamp(min=1.0)
            dim_term = ((pred - target) ** 2 * m).sum() / denom
            if lam <= 0.0:
                return dim_term
            corr = masked_cross_section_pearson_torch(pred, target, mask, eps)
            return dim_term + lam * (1.0 - corr)
        # 逐票批：标准 MSE。
        dim_term = ((pred - target) ** 2).mean()
        if lam <= 0.0:
            return dim_term
        corr = pearson_torch(pred, target, eps)
        return dim_term + lam * (1.0 - corr)

    return loss_fn
