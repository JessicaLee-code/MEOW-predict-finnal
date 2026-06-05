"""
超参搜索配置块 —— SearchConfig(frozen)

对应规格 §6（省算力搜索策略）与 §7.2 归属表里"搜索预算"那一行（Orchestrator 独占）。

这里**只管"本次 run 怎么搜"**：搜几个 trial、早杀开不开、训练行采样多少、对卡带
私有 ``search_space`` 的收窄/冻结覆盖。``search_space`` 本体（哪些旋钮、各自分布）
是卡带私有声明，**不在这里**（规格 §7.3）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class SearchConfig:
    """
    本次 run 的搜索预算与对 search_space 的收窄。

    - ``n_trials``: 随机搜索 trial 数（海选大、认证为 1 = 不再搜）。
    - ``early_kill``: 早杀开关。D0 先留钩子（卡带上报 TrainRecord），实现推后（规格 §4/§11）。
    - ``early_kill_warmup_epochs``: 早杀前至少观察的 epoch 数（钩子参数，实现推后）。
    - ``train_subsample_frac``: 训练行采样比例（类比树侧 0.33，降单 fit 成本，规格 §6.4）。
    - ``search_overrides``: 对卡带 ``search_space`` 的收窄/冻结（如把 seq_len 限定到 {16,32}）；
      只搜结构 3 旋钮（seq_len / hidden_size / num_layers）。frozen + MappingProxy 不可变。
    """
    n_trials: int = 1
    early_kill: bool = False
    early_kill_warmup_epochs: int = 3
    train_subsample_frac: float = 1.0
    search_overrides: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "search_overrides", MappingProxyType(dict(self.search_overrides)))
        if not (0.0 < self.train_subsample_frac <= 1.0):
            raise ValueError(f"train_subsample_frac 必须在 (0, 1]，实际 {self.train_subsample_frac}")
        if self.n_trials < 1:
            raise ValueError(f"n_trials 必须 >= 1，实际 {self.n_trials}")

    def to_dict(self) -> dict:
        return {
            "n_trials": self.n_trials,
            "early_kill": self.early_kill,
            "early_kill_warmup_epochs": self.early_kill_warmup_epochs,
            "train_subsample_frac": self.train_subsample_frac,
            "search_overrides": dict(self.search_overrides),
        }
