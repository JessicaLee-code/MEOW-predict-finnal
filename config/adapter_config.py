"""
输入适配配置块 —— AdapterKind(枚举) + AdapterConfig(frozen)

InputAdapter 是唯一"理解原始数据语义"的卡带（规格 §3.1）。这里只声明：
- 顶部词表 ``AdapterKind``：有哪些适配器可选（封闭集）；
- ``AdapterConfig``：本次 run 选哪个 + 该适配器需要的配置（如特征组 / 通道列）。

适配器吐出的**通道名 ``channels``** 由适配器实现自己声明（跟实现走），不在这里枚举。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class AdapterKind(Enum):
    """输入适配器合法词表。"""
    FEATURE_433 = "feature_433"     # 包装现有 433 特征管线（SubmissionFeaturePipeline），特征列当通道（D1/LSTM）
    RAW_CHANNELS = "raw_channels"   # ~59 原始微结构通道做最小语义归一当通道（D2/TCN，规格 §3.1/§8.0）
    IDENTITY = "identity"           # 直接把指定 raw 数值列当通道（调试 / 防泄漏合成测试用）


@dataclass(frozen=True)
class AdapterConfig:
    """
    本次 run 的输入适配选择。

    - ``kind``: 选哪个适配器。
    - ``groups``: 仅 ``FEATURE_433`` 用——选哪些正式特征组（空 = 提交默认并集）。
    - ``columns``: 仅 ``IDENTITY`` 用——把哪些 raw 列当通道（空 = 由适配器报错提示必填）。
    """
    kind: AdapterKind
    groups: Tuple[str, ...] = ()
    columns: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {"kind": self.kind.value, "groups": list(self.groups), "columns": list(self.columns)}
