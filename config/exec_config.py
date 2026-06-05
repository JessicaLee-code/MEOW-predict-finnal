"""
执行配置块 —— ExecConfig(frozen)

对应规格 §7.2 归属表里 Orchestrator「独占」的执行级配置：种子、device、重训/resume、
输出目录、worker 数。

关键：规格 §7.7「是否重新训练」拆成**两个独立字段**，别混成一个 bool：
- ``resume``: 续跑被中断的 run（按 (profile, fold, experiment_id) 查 status 接着跑）；
- ``reuse_checkpoint``: 复用已训权重（按 (config_fingerprint, fold, seed) 训过就 load 跳过）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class ExecConfig:
    """本次 run 的执行级配置（全部 Orchestrator 独占）。"""
    seeds: Tuple[int, ...] = (42,)   # 海选少种子(1-2)、认证多种子(3)，规格 §6.5
    device: str = "cpu"              # "cpu" / "cuda"；D0 全程 cpu 可测
    resume: bool = False             # (a) 续跑中断的 run
    reuse_checkpoint: bool = False   # (b) 复用 checkpoint 跳过已训 (config,fold,seed)
    out_dir: str = "results/dl"      # 输出根目录（run 产物落 out_dir/<run_id>/）
    n_workers: int = 1               # 进程级并行度；Mac 16GB 恒为 1（AGENTS §5.1）
    target_mode: str = "raw"         # 训练目标模式：raw=直接学 fret12；residual_trad=学传统预测残差
    trad_preds_root: str = ""        # residual_trad 时传统逐票预测根目录（含 preds/fold*/...）
    trad_pred_col: str = "pred_blend"  # 传统目标列：默认锁定代表融合列 pred_blend
    trad_cache_dir: str = "results/dl/_trad_residual_cache"  # 训练区传统预测缓存（gitignore）

    def __post_init__(self) -> None:
        if len(self.seeds) == 0:
            raise ValueError("seeds 不能为空")
        if self.device not in ("cpu", "cuda"):
            raise ValueError(f"device 必须是 'cpu' / 'cuda'，实际 {self.device}")
        if self.target_mode not in ("raw", "residual_trad"):
            raise ValueError(
                f"target_mode 必须是 'raw' / 'residual_trad'，实际 {self.target_mode}"
            )
        if self.target_mode == "residual_trad" and not self.trad_preds_root:
            raise ValueError("target_mode=residual_trad 时必须提供 trad_preds_root")

    def to_dict(self) -> dict:
        return {
            "seeds": list(self.seeds),
            "device": self.device,
            "resume": self.resume,
            "reuse_checkpoint": self.reuse_checkpoint,
            "out_dir": self.out_dir,
            "n_workers": self.n_workers,
            "target_mode": self.target_mode,
            "trad_preds_root": self.trad_preds_root,
            "trad_pred_col": self.trad_pred_col,
            "trad_cache_dir": self.trad_cache_dir,
        }
