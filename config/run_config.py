"""
中央组装 —— RunConfig(frozen) + assemble_run_config(组装 + 跨块校验 + fingerprint)

规格 §7.1/§7.5/§7.6 的落地：Orchestrator 不写各块旋钮，只**组装**——把五个配置块
拼成一份 ``RunConfig`` → 跨块校验 → 冻结 → 算 ``config_fingerprint``。

四性质（§7.5）：
1. 不可变 + 冻结（frozen dataclass）= config-lock 机械实现；
2. 可序列化（``to_dict`` / ``dump_json``）= 每次 run 落 JSON 可复现可审计；
3. 组装期校验（``assemble_run_config`` 一刻全报错）；
4. 分层嵌套（不平铺成大 dict）。

run_id 口径（§7.6）：手工语义命名（``<日期>_<阶段>_<模型>_<意图>_<版本>``）；
``config_fingerprint`` 是语义内容哈希，唯一用途 = resume/复用时比对防漂移。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from adapter_config import AdapterConfig
from exec_config import ExecConfig
from model_config import ModelConfig
from protocol_config import ProfileKind, ProtocolConfig, Stage
from search_config import SearchConfig


@dataclass(frozen=True)
class RunConfig:
    """一次 DL run 的完整冻结配置（分层嵌套，不平铺）。"""
    run_id: str
    model: ModelConfig
    adapter: AdapterConfig
    protocol: ProtocolConfig
    search: SearchConfig
    exec_: ExecConfig
    config_fingerprint: str = field(default="")

    def __post_init__(self) -> None:
        # fingerprint 只依赖语义内容（不含自身），构造期算一次写回（frozen 走 setattr）。
        if not self.config_fingerprint:
            object.__setattr__(self, "config_fingerprint", self._compute_fingerprint())

    def _semantic_dict(self) -> dict:
        """参与 fingerprint 的语义内容（刻意排除 run_id 与执行级噪声字段）。"""
        ex = self.exec_.to_dict()
        # resume / out_dir / n_workers / reuse_checkpoint 是执行细节，改它们不算"配置漂移"。
        for k in ("resume", "out_dir", "n_workers", "reuse_checkpoint"):
            ex.pop(k, None)
        return {
            "model": self.model.to_dict(),
            "adapter": self.adapter.to_dict(),
            "protocol": self.protocol.to_dict(),
            "search": self.search.to_dict(),
            "exec": ex,
        }

    def _compute_fingerprint(self) -> str:
        canonical = json.dumps(self._semantic_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "config_fingerprint": self.config_fingerprint,
            "model": self.model.to_dict(),
            "adapter": self.adapter.to_dict(),
            "protocol": self.protocol.to_dict(),
            "search": self.search.to_dict(),
            "exec": self.exec_.to_dict(),
        }

    def dump_json(self, path: str) -> None:
        """落 JSON 到输出目录旁（可复现可审计，规格 §7.5.2）。"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2, sort_keys=True)


def assemble_run_config(
    run_id: str,
    model: ModelConfig,
    adapter: AdapterConfig,
    protocol: ProtocolConfig,
    search: SearchConfig,
    exec_: ExecConfig,
) -> RunConfig:
    """
    组装 + 跨块校验 + 冻结。任何不一致在**组装这一刻**报错（规格 §7.5.3）：

    1. ``run_id`` 非空；
    2. **required_adapter 匹配**：模型卡带要求的适配器种类必须等于本次选的适配器
       （延迟 import registry，查卡带类的 ``required_adapter``——类型化引用，拼错当场炸）；
    3. **stage / profile 搭配**：认证（VALIDATION）必须用 EXPANDING（少折判官）；
       海选（SEARCH）建议 SINGLE_SPLIT（也允许 EXPANDING 小折调试）；
    4. **数据窗口合法**：rolling_start <= rolling_end。
    """
    if not run_id or not run_id.strip():
        raise ValueError("run_id 不能为空（手工语义命名，规格 §7.6）")

    # (4) 数据窗口
    if protocol.rolling_start > protocol.rolling_end:
        raise ValueError(
            f"protocol 数据窗口非法：rolling_start={protocol.rolling_start} > rolling_end={protocol.rolling_end}"
        )
    if protocol.delivery_eval_end is not None and protocol.delivery_eval_end <= protocol.rolling_end:
        raise ValueError(
            "delivery_eval_end 必须晚于 rolling_end："
            f"delivery_eval_end={protocol.delivery_eval_end}, rolling_end={protocol.rolling_end}"
        )

    # (3) stage / profile 搭配：认证 / 一命令两档都要多折判官（EXPANDING）
    if protocol.stage in (Stage.VALIDATION, Stage.SWEEP) and protocol.profile != ProfileKind.EXPANDING:
        raise ValueError(
            f"{protocol.stage.value} 阶段必须用 EXPANDING profile（多折判官），实际 {protocol.profile}"
        )

    # (2) required_adapter 匹配（延迟 import，避免 config <-> models 顶层循环）
    from registry import required_adapter_for
    required = required_adapter_for(model.kind)
    # required 为 None 表示该卡带不挑适配器（如 numpy 参考模型），跳过匹配校验。
    if required is not None and required != adapter.kind:
        raise ValueError(
            f"适配器不匹配：模型 {model.kind.value} 要求 adapter={required.value}，"
            f"本次选的是 {adapter.kind.value}"
        )

    return RunConfig(
        run_id=run_id,
        model=model,
        adapter=adapter,
        protocol=protocol,
        search=search,
        exec_=exec_,
    )
