"""
HPO Searcher —— 结构超参随机搜索 + 早杀钩子 + 海选排名（脊柱基础设施，torch-free）

对应规格 §6（省算力搜索）与 §2.1 控制层级里 Orchestrator 与 Protocol Engine 之间那一格。
本层**不碰 torch**：它只负责"采一组超参 → 用 SequenceTrainer 跑折 → 收 val_corr → 排名"，
真正的 epoch 循环/optimizer 全封在卡带 ``fit`` 内。

三块能力：

1. **采样器**（``sample_hparams`` / ``_sample_one``）：读卡带私有 ``search_space`` 的迷你
   spec（``choice`` / ``int`` / ``uniform`` 三型）随机采样；``SearchConfig.search_overrides``
   按 knob 整体替换/收窄（如把 ``seq_len`` 限到 ``{16,32}``）。

2. **早杀钩子**（``EarlyKillPolicy``）：D0 是**可独立测试但默认关**的桩——真正"跑几个 epoch
   明显落后即杀整个 trial"需要卡带 ``fit`` 逐 epoch 回调（torch 卡带就绪后接），见 §6/§11。

3. **Searcher**：跑 ``n_trials`` 个 trial，每 trial 采一组结构超参 → 海选档（单切分）× 1–2
   种子跑折 → ``summarize_folds`` 取 pooled 均值 + 最坏折 → 按 ``val_corr`` 均值排名 →
   产 ``SearchOutcome``（best + 全 trial 明细，供 Orchestrator 落 trials.csv / best_config.json）。

**关键边界**：``seq_len`` 是 **trainer 级**旋钮（决定 ``SequenceDataset`` 开窗、要重建数据集，
由本层喂 ``SequenceTrainer(seq_len=...)``）；``hidden_size`` / ``num_layers`` 是 **卡带级** hparams
（进 ``cartridge.fit``）。这条拆分写死在 ``Searcher.run`` 里。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from dl_protocol import DLFold, summarize_folds
from dl_trainer import SequenceTrainer

# seq_len 缺省值（search_space / defaults 都没给时兜底）。
_DEFAULT_SEQ_LEN = 32


# ================================================================== #
# 采样器
# ================================================================== #

def _sample_one(spec: Dict, rng: np.random.Generator):
    """按迷你 spec 采一个值（choice / int / uniform）。"""
    t = spec.get("type")
    if t == "choice":
        vals = list(spec["values"])
        if not vals:
            raise ValueError("choice spec 的 values 为空")
        return vals[int(rng.integers(0, len(vals)))]   # 用索引采，保留原生类型（int/str 混合也安全）
    if t == "int":
        lo, hi = int(spec["low"]), int(spec["high"])
        return int(rng.integers(lo, hi + 1))           # 闭区间 [lo, hi]
    if t == "uniform":
        return float(rng.uniform(float(spec["low"]), float(spec["high"])))
    raise ValueError(f"未知 search_space spec type: {t!r}（应为 choice/int/uniform）")


def sample_hparams(
    search_space: Dict,
    rng: np.random.Generator,
    overrides: Optional[Dict] = None,
) -> Dict:
    """
    从卡带 ``search_space`` 采一组超参；``overrides`` 按 knob 整体替换该 knob 的 spec（收窄/冻结）。

    返回 dict（knob -> 采样值）。``search_space`` 为空（如参考模型）即返回空 dict——
    此时 Searcher 退化为"只用 defaults 跑 n_trials 次同配置"（仍可借多种子看稳定性）。
    """
    overrides = overrides or {}
    space = dict(search_space)
    for knob, spec in overrides.items():
        space[knob] = spec
    return {knob: _sample_one(spec, rng) for knob, spec in space.items()}


def enumerate_grid(search_space: Dict, overrides: Optional[Dict] = None) -> List[Dict]:
    """
    把卡带 ``search_space``（叠加 ``overrides`` 收窄）展开成**确定性网格**（笛卡尔积）。

    与 ``sample_hparams`` 随机采样相对：SWEEP 档1 用确定性小网格，忠实「HPO 网格预先声明、
    跑一次」（AGENTS §十一·11.7 护栏 1）。逐 knob 展开：
    - ``choice``  → 全部 values；
    - ``int``     → 闭区间 ``[low, high]`` 全部整数；
    - ``uniform`` → 取 ``[low, high]`` 两端点（连续旋钮本不该进网格，仅兜底）。

    返回 hparam dict 列表（每个 = 一组完整超参，knob 顺序稳定）；空空间返回 ``[{}]``（单点）。
    """
    overrides = overrides or {}
    space = dict(search_space)
    for knob, spec in overrides.items():
        space[knob] = spec
    if not space:
        return [{}]
    knobs = list(space.keys())
    axes: List[List] = []
    for knob in knobs:
        spec = space[knob]
        t = spec.get("type")
        if t == "choice":
            axis = list(spec["values"])
        elif t == "int":
            axis = list(range(int(spec["low"]), int(spec["high"]) + 1))
        elif t == "uniform":
            axis = [float(spec["low"]), float(spec["high"])]
        else:
            raise ValueError(f"未知 search_space spec type: {t!r}（应为 choice/int/uniform）")
        if not axis:
            raise ValueError(f"knob {knob!r} 展开为空")
        axes.append(axis)
    return [dict(zip(knobs, combo)) for combo in itertools.product(*axes)]


# ================================================================== #
# 早杀钩子（D0 桩，可独立测试、默认关）
# ================================================================== #

@dataclass
class EarlyKillPolicy:
    """
    HPO 级早杀策略（杀整个 trial，规格 §6.3）。

    D0 现状：**钩子就位、默认关**。``should_kill`` 的判据逻辑本身可独立测，但真正"在
    训练途中省算力"需要卡带 ``fit`` 逐 epoch 回调本策略（torch 卡带就绪后接，§11）；
    numpy 参考模型无 epoch 可杀，故 Searcher 里此钩子默认 inert。

    判据（``enabled=True`` 时）：观察 earlystop 曲线（**约定 lower-is-better**，如 val MSE），
    warmup 之后的最优值仍比 ``best_so_far``（已知最好 trial 的同口径最优值）差超过
    ``rel_margin`` 比例，即判该 trial 没希望、可杀。
    """
    enabled: bool = False
    warmup_epochs: int = 3
    rel_margin: float = 0.5          # 落后 best_so_far 超 50% 即杀

    def should_kill(self, earlystop_curve: Sequence[float], best_so_far: Optional[float]) -> bool:
        if not self.enabled:
            return False
        curve = [c for c in earlystop_curve if np.isfinite(c)]
        if len(curve) < self.warmup_epochs or best_so_far is None or not np.isfinite(best_so_far):
            return False
        best_here = min(curve[self.warmup_epochs - 1:])   # warmup 后的最优（loss 越小越好）
        if best_so_far <= 0:
            return best_here > best_so_far
        return best_here > best_so_far * (1.0 + self.rel_margin)


# ================================================================== #
# Trial / 搜索结果容器
# ================================================================== #

@dataclass
class TrialResult:
    """单个 trial 的汇总（跨 seed × fold 的 val_corr 双镜头）。"""
    trial_id: int
    seq_len: int
    hparams: Dict                 # 卡带级 hparams（不含 seq_len）
    val_corr_mean: float          # pooled 均值
    val_corr_min: float           # 最坏折/种子（minimax 镜头）
    val_corr_std: float
    n_evals: int                  # 参与汇总的 (seed, fold) 数
    status: str = "ok"            # "ok" | "error"
    error_msg: str = ""
    best_epoch_mean: float = 0.0  # 平均 best_epoch（诊断 early stopping 是否提前触发）
    n_epochs_mean: float = 0.0    # 平均实际跑了多少 epoch（判断是否撞 max_epochs 天花板）

    def to_row(self) -> dict:
        """落 trials.csv 的一行（hparams 摊平进列，前缀 hp_）。"""
        row = {
            "trial_id": self.trial_id,
            "seq_len": self.seq_len,
            "val_corr_mean": self.val_corr_mean,
            "val_corr_min": self.val_corr_min,
            "val_corr_std": self.val_corr_std,
            "n_evals": self.n_evals,
            "best_epoch_mean": self.best_epoch_mean,
            "n_epochs_mean": self.n_epochs_mean,
            "status": self.status,
            "error_msg": self.error_msg,
        }
        for k, v in self.hparams.items():
            row[f"hp_{k}"] = v
        return row


@dataclass
class SearchOutcome:
    """搜索产物：全 trial 明细 + 冠军（按 val_corr 均值排名）。"""
    trials: List[TrialResult] = field(default_factory=list)
    best: Optional[TrialResult] = None

    def best_config_dict(self) -> dict:
        """冠军的可序列化视图，供 Orchestrator 落 best_config.json（认证档据此冻结）。"""
        if self.best is None:
            return {"found": False}
        return {
            "found": True,
            "trial_id": self.best.trial_id,
            "seq_len": self.best.seq_len,
            "hparams": dict(self.best.hparams),
            "val_corr_mean": self.best.val_corr_mean,
            "val_corr_min": self.best.val_corr_min,
            "n_evals": self.best.n_evals,
        }


# ================================================================== #
# Searcher
# ================================================================== #

class Searcher:
    """
    海选搜索器：随机采结构超参 → 跑折 → 排名。

    依赖注入（不反向依赖 registry / config 具体类，与 SequenceTrainer 同策略）：
    - ``adapter`` / ``cartridge_factory`` / ``raw_loader`` 由 Orchestrator 用 registry 构造后注入；
    - ``folds`` 由 Orchestrator 按 protocol 派生（海选档通常单切分 = 1 折）。
    """

    def __init__(
        self,
        *,
        spec: Dict,
        adapter,
        cartridge_factory: Callable[[], object],
        raw_loader: Callable[[Sequence[int]], object],
        folds: Sequence[DLFold],
        search_space: Dict,
        n_trials: int,
        seeds: Sequence[int] = (42,),
        defaults: Optional[Dict] = None,
        normalizer_mode: str = "zscore",
        search_overrides: Optional[Dict] = None,
        profile_name: str = "search",
        early_kill: Optional[EarlyKillPolicy] = None,
        sampling_seed: int = 12345,
    ):
        self.spec = dict(spec)
        self.adapter = adapter
        self.cartridge_factory = cartridge_factory
        self.raw_loader = raw_loader
        self.folds = list(folds)
        self.search_space = dict(search_space or {})
        self.n_trials = int(n_trials)
        self.seeds = tuple(seeds)
        self.defaults = dict(defaults or {})
        self.normalizer_mode = normalizer_mode
        self.search_overrides = dict(search_overrides or {})
        self.profile_name = profile_name
        # 早杀钩子默认关；D0 不在 run() 里真正用（无 epoch 回调），留作 torch 卡带就绪后的接点。
        self.early_kill = early_kill or EarlyKillPolicy(enabled=False)
        self.sampling_seed = int(sampling_seed)

    def _eval_trial(self, seq_len: int, cart_hparams: Dict) -> Dict:
        """一组超参跑 seeds × folds，收集 val_corr + epoch 诊断信息。"""
        corrs: List[float] = []
        best_epochs: List[int] = []
        n_epochs_list: List[int] = []
        for seed in self.seeds:
            trainer = SequenceTrainer(
                self.spec, self.adapter, self.cartridge_factory, self.raw_loader,
                seq_len=seq_len, normalizer_mode=self.normalizer_mode,
                hparams=cart_hparams, seed=int(seed),
            )
            for fold in self.folds:
                r = trainer.run_on_dl_fold(fold, profile_name=self.profile_name)
                if r.status == "ok" and np.isfinite(r.val_corr):
                    corrs.append(float(r.val_corr))
                    best_epochs.append(int(r.best_epoch))
                    n_epochs_list.append(int(r.n_epochs))
                elif r.status == "error":
                    # 单折失败不拖垮 trial，但记一次让上层 status 反映异常。
                    raise RuntimeError(f"fold {fold.fold_id} 失败: {r.error_msg}")
        return {
            "corrs": corrs,
            "best_epoch_mean": float(np.mean(best_epochs)) if best_epochs else 0.0,
            "n_epochs_mean": float(np.mean(n_epochs_list)) if n_epochs_list else 0.0,
        }

    def run(self) -> SearchOutcome:
        rng = np.random.default_rng(self.sampling_seed)
        trials: List[TrialResult] = []

        for t in range(self.n_trials):
            sampled = sample_hparams(self.search_space, rng, self.search_overrides)
            # defaults 打底、采样覆盖；seq_len 抽出来归 trainer，其余归卡带 hparams。
            effective = {**self.defaults, **sampled}
            seq_len = int(effective.pop("seq_len", _DEFAULT_SEQ_LEN))
            cart_hparams = effective

            status, err = "ok", ""
            trial_data: Dict = {"corrs": [], "best_epoch_mean": 0.0, "n_epochs_mean": 0.0}
            try:
                trial_data = self._eval_trial(seq_len, cart_hparams)
            except Exception as e:                       # noqa: BLE001
                status, err = "error", str(e)[:500]

            corrs = trial_data["corrs"]
            summ = summarize_folds([{"corr": c} for c in corrs])
            trials.append(TrialResult(
                trial_id=t, seq_len=seq_len, hparams=dict(cart_hparams),
                val_corr_mean=summ["mean"], val_corr_min=summ["min"],
                val_corr_std=summ["std"], n_evals=len(corrs),
                status=status, error_msg=err,
                best_epoch_mean=trial_data["best_epoch_mean"],
                n_epochs_mean=trial_data["n_epochs_mean"],
            ))

        ranked = sorted(
            [tr for tr in trials if tr.status == "ok" and tr.n_evals > 0],
            key=lambda x: x.val_corr_mean, reverse=True,
        )
        return SearchOutcome(trials=trials, best=(ranked[0] if ranked else None))
