"""
实现注册表 —— 枚举值 → 卡带/适配器类（规格 §7.3 第②层「实现注册」）

三层枚举拆分里，这是"实现注册"层：把 ``ModelKind`` / ``AdapterKind`` 枚举值映射到
具体实现类。注册放在**和实现挨着**的地方——``dl_models.py`` 里各类用本文件的
``@register_model`` / ``@register_adapter`` 装饰器把自己登记进来。

为避免 ``config`` ↔ ``models`` 顶层循环：本文件只 import 枚举（轻），实现类的加载
延迟到 ``build_* / required_adapter_for`` 首次调用时（``_ensure_impls_loaded``）。
"""

from __future__ import annotations

from typing import Dict, Optional, Type

from adapter_config import AdapterKind
from model_config import ModelKind

# 枚举值 → 实现类（由 dl_models.py 的装饰器填充）。
ADAPTER_REGISTRY: Dict[AdapterKind, Type] = {}
MODEL_REGISTRY: Dict[ModelKind, Type] = {}

_IMPLS_LOADED = False


def register_adapter(kind: AdapterKind):
    """类装饰器：把某 InputAdapter 实现登记到 ``kind``。"""
    def deco(cls):
        ADAPTER_REGISTRY[kind] = cls
        return cls
    return deco


def register_model(kind: ModelKind):
    """类装饰器：把某 ModelCartridge 实现登记到 ``kind``。"""
    def deco(cls):
        MODEL_REGISTRY[kind] = cls
        return cls
    return deco


def _ensure_impls_loaded() -> None:
    """延迟 import 实现模块，触发其装饰器注册（只做一次）。"""
    global _IMPLS_LOADED
    if not _IMPLS_LOADED:
        import dl_models  # noqa: F401  （import 即注册）
        _IMPLS_LOADED = True


def build_adapter(adapter_config):
    """按 AdapterConfig 实例化 InputAdapter。"""
    _ensure_impls_loaded()
    if adapter_config.kind not in ADAPTER_REGISTRY:
        raise KeyError(f"未注册的 AdapterKind: {adapter_config.kind}")
    return ADAPTER_REGISTRY[adapter_config.kind].from_config(adapter_config)


def build_cartridge(model_config):
    """按 ModelConfig 实例化 ModelCartridge。"""
    _ensure_impls_loaded()
    if model_config.kind not in MODEL_REGISTRY:
        raise KeyError(f"未注册的 ModelKind: {model_config.kind}")
    return MODEL_REGISTRY[model_config.kind].from_config(model_config)


def required_adapter_for(model_kind: ModelKind) -> Optional[AdapterKind]:
    """
    查某模型卡带要求的适配器种类（类型化引用，RunConfig 组装期据此校验匹配）。

    返回 ``None`` 表示该卡带不挑适配器（如 numpy 参考模型，可配任意 adapter）。
    """
    _ensure_impls_loaded()
    if model_kind not in MODEL_REGISTRY:
        raise KeyError(f"未注册的 ModelKind: {model_kind}")
    return MODEL_REGISTRY[model_kind].required_adapter
