"""
评测协议配置块 —— Stage / ProfileKind(枚举) + ProtocolConfig(frozen)

对应规格 §5（两阶段评测）与 §7.2（折结构归属）。两个词表：
- ``Stage``：海选（search）/ 认证（validation），决定预算与看不看 expanding；
- ``ProfileKind``：单切分（海选/调试便宜车道）/ expanding 少折（认证判官）。

``ProtocolConfig`` 把"本次 run 用哪个 profile + 哪段日期 + 折参数"收口，并派生出
``dl_protocol.build_dl_folds`` 需要的 ``mode`` / ``max_folds``，让脊柱协议引擎照单切折。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Stage(Enum):
    """评测阶段（规格 §5.2；SWEEP 见 AGENTS §十一·11.6）。"""
    SEARCH = "search"           # 海选：单切分 + 早杀，便宜搜超参（旧两段法，保留调试用）
    VALIDATION = "validation"   # 认证：expanding 少折 + 多种子，定参确认（旧两段法，保留调试用）
    SWEEP = "sweep"             # 一命令两档（主路径）：档1 小网格×近2折×2seed→按最坏折选冠军；档2 冠军×全折×3seed


class ProfileKind(Enum):
    """折结构 profile（规格 §5.6：砍 short/medium，加单切分迭代档）。"""
    SINGLE_SPLIT = "single_split"   # 单时间切分（海选 / 调试）
    EXPANDING = "expanding"         # expanding walk-forward 少折（认证）


@dataclass(frozen=True)
class ProtocolConfig:
    """
    本次 run 的协议选择 + 折参数。

    - ``stage`` / ``profile``: 阶段与折结构（组装期会校验二者搭配合理，见 run_config）。
    - ``rolling_start`` / ``rolling_end``: 本次 run 的数据窗口（Orchestrator 拥有区间，
      协议据此派生折，规格 §7.2）。
    - 折参数沿用 eval_protocol 口径：``val_window`` / ``step`` / ``embargo`` / ``min_train_days``
      / ``train_window``（None=expanding）。
    - ``earlystop_frac``: 训练区尾段切出 earlystop-val 的比例（规格 §5.3）。
    - ``max_folds``: SINGLE_SPLIT 派生为 1；EXPANDING 认证档建议 3 折。
    """
    stage: Stage
    profile: ProfileKind
    rolling_start: int
    rolling_end: int
    val_window: int = 5
    step: int = 5
    embargo: int = 1
    min_train_days: int = 40
    train_window: Optional[int] = None
    earlystop_frac: float = 0.15
    max_folds: Optional[int] = None
    fold_select: str = "first"      # "first"=最早若干折（默认/调试）；"recent"=最近若干折倒贴 rolling_end（§十一·11.2 新协议）
    delivery_eval_end: Optional[int] = None   # SWEEP 交付对齐折 eval 末日；None=不跑交付折（调试/旧路径）

    def fold_mode(self) -> str:
        """派生 build_dl_folds 的 mode：两种 profile 都走 expanding 切法（单切分=取 1 折）。"""
        return "expanding"

    def effective_max_folds(self) -> Optional[int]:
        """SINGLE_SPLIT 强制只取首折；EXPANDING 用配置的 max_folds（None=不限）。"""
        if self.profile == ProfileKind.SINGLE_SPLIT:
            return 1
        return self.max_folds

    def to_dict(self) -> dict:
        return {
            "stage": self.stage.value,
            "profile": self.profile.value,
            "rolling_start": self.rolling_start,
            "rolling_end": self.rolling_end,
            "val_window": self.val_window,
            "step": self.step,
            "embargo": self.embargo,
            "min_train_days": self.min_train_days,
            "train_window": self.train_window,
            "earlystop_frac": self.earlystop_frac,
            "max_folds": self.max_folds,
            "fold_select": self.fold_select,
            "delivery_eval_end": self.delivery_eval_end,
        }
