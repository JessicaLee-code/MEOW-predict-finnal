"""
模型配置块 —— ModelKind(枚举) + ModelConfig(frozen)

按规格 §7.3/§7.4「三层枚举拆分」与「每块单独成文件 + 枚举进块文件顶部」：
- 文件顶部放**合法词表**（``ModelKind`` 枚举，封闭选项集，集中维护）；
- 下方放本块的 **schema**（``ModelConfig`` frozen dataclass，"本次 run 选了哪个值"）。

注意区分（规格 §6/§7.3）：
- ``search_space``（某模型私有的超参声明 + 范围/分布）是**卡带私有**，跟 ``ModelCartridge``
  走，**不在这里**（不属于全局枚举、不进 config/）；
- 这里的 ``hparams`` 是"本次 run 钉死的固定超参"（非搜索时直接用，或搜索的默认基底）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


class ModelKind(Enum):
    """模型卡带合法词表（封闭集；新增卡带先在这里登记一个枚举值）。"""
    REFERENCE_ZERO = "reference_zero"   # numpy 参考模型：恒 0，sanity 基线
    REFERENCE_LAST = "reference_last"   # numpy 参考模型：末步通道线性，防泄漏探测器
    REFERENCE_POOL = "reference_pool"   # numpy 参考模型：窗口均值池化 + 线性；声明 STRUCTURE_SEARCH_SPACE，当 HPO/TCN 的 torch-free 模板
    TCN = "tcn"                         # 【否决·2026-06-01】TCN-on-raw 截面盲区（详见规格 §8.0 / NOTE.md）；保留为历史词位
    GRU = "gru"                         # 时序基线 + 保底主跑（433 工程特征，绑 FEATURE_433）；编码每票日内路径，也是截面模型(规格 §8.2)的时序腿
    XSECTION = "xsection"               # 【主攻】截面模型：共享 GRU 时序腿 + set-attention 截面腿 + 零初门控残差 + per-票头（规格 §8.2，绑 FEATURE_433）
    XSECTION_RAW = "xsection_raw"       # 【2026-06-03 新】截面模型架构 + RAW_CHANNELS 输入：把 XSECTION 的输入从 433 摘要换成 RawChannelAdapter 的 59 原始微结构通道（含挂撤单/深档盘口/成交明细）；架构完全复用、input_channels 数据驱动自适应。验证"好架构(截面)+好输入(raw)"这一格（前三轮从没合过：TCN 用 raw 但无截面、截面/GRU 有截面但只喂 433）
    XSECTION_RAW_DEEPLOB = "xsection_raw_deeplob"  # 【2026-06-03 新】并行模型：RAW_CHANNELS 输入 + DeepLOB 风格卷积前端 + 现有 XSECTION 截面层；专门验证“更强 raw 前端”能否把 xsection_raw 从 0.08x 再往上推，且绝不覆盖原 XSECTION_RAW 基线
    LSTM = "lstm"                       # 占位，待定（GRU 已覆盖 D1 角色，此槽保留供未来对照）
    DEEPLOB = "deeplob"                 # 【退役】数据无连续 LOB（规格 §8.0），保留为历史词位，不实现


@dataclass(frozen=True)
class ModelConfig:
    """
    本次 run 的模型选择 + 钉死的固定超参。

    - ``kind``: 选哪个卡带（registry 据此实例化对应 ``ModelCartridge``）。
    - ``hparams``: 本次钉死的固定超参（如 ``{"seq_len": 32, "hidden_size": 64}``）；
      只搜结构 3 旋钮时，未搜的旋钮从这里取默认。frozen + MappingProxy 双保险，
      组装后不可变（config-lock 机械实现，规格 §7.5）。
    """
    kind: ModelKind
    hparams: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # frozen dataclass 内改字段需走 object.__setattr__；把 dict 冻成只读视图。
        object.__setattr__(self, "hparams", MappingProxyType(dict(self.hparams)))

    def to_dict(self) -> dict:
        """可序列化视图（枚举→value、MappingProxy→dict），供 RunConfig dump JSON。"""
        return {"kind": self.kind.value, "hparams": dict(self.hparams)}
