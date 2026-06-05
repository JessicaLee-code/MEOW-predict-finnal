"""
调度层 — 并发执行 rolling fold 任务

ParallelScheduler:   将 (profile × fold) 任务展平，ProcessPoolExecutor 并发执行
_fold_group_worker:  模块级 worker 函数（multiprocessing 'spawn' 模式可 pickle）

M4 调整后的并发策略：
  - worker 不再依赖 ExperimentRunner 的 `_daily_feature_cache/_split_cache`
  - 每个任务显式创建 `FeatureLoader`，直接从磁盘 stage artifact 读取数据
  - fold 分组仍然保留，目的是减少任务调度开销，而不是复用旧内存缓存
  - long / expanding 视为重任务批次，进入该阶段后总并发硬限制为 2，
    避免 long_g0 + expanding_g0 + expanding_g1 这类三重任务同时在飞
"""

import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from feature_store import DEFAULT_FEATURE_DIR

# ================================================================== #
# 元数据结构
# ================================================================== #

@dataclass
class FoldMeta:
    """单个 fold 的轻量元数据（可跨进程 pickle）"""
    profile_name: str
    fold_id: int
    train_dates: tuple
    val_dates: tuple


@dataclass
class FoldGroup:
    """一组 fold，作为单个 worker 任务的输入单元"""
    group_id: str              # "{profile_name}_g{n}"，用于日志
    fold_metas: List[FoldMeta]


# ================================================================== #
# Worker 函数（模块级，multiprocessing 'spawn' 可序列化）
# ================================================================== #

def _fold_group_worker(args: tuple) -> List[dict]:
    """
    进程池 worker。

    每个 worker 在进程内创建独立 ExperimentRunner + FeatureLoader，
    按顺序处理组内各 fold。数据加载直接走 FeatureLoader，不再清理旧 cache。

    args: (h5dir, feature_dir, target_winsorize_config, feature_dtype, ridge_alpha, fold_group, specs, completed_keys, train_subsample_frac)
    返回: list[FoldResult.to_dict()]
    """
    # macOS 'spawn' 模式：确保 src/ 在 sys.path 中
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    if _this_dir not in sys.path:
        sys.path.insert(0, _this_dir)

    from experiment_runner import ExperimentRunner
    from feature_loader import FeatureLoader
    from trainer import FoldData, TabularTrainer

    h5dir, feature_dir, target_winsorize_config, feature_dtype, ridge_alpha, fold_group, specs, completed_keys, train_subsample_frac = args

    runner = ExperimentRunner(
        h5dir,
        feature_dir=feature_dir,
        target_winsorize_config=target_winsorize_config,
        feature_dtype=feature_dtype,
        ridge_alpha=ridge_alpha,
        train_subsample_frac=train_subsample_frac,
    )
    loader = FeatureLoader(h5dir=h5dir, feature_dir=feature_dir, feature_dtype=feature_dtype)
    results: List[dict] = []

    for meta in fold_group.fold_metas:
        fold_data = FoldData(
            profile_name=meta.profile_name,
            fold_id=meta.fold_id,
            train_dates=meta.train_dates,
            val_dates=meta.val_dates,
        )
        for spec in specs:
            key = (meta.profile_name, meta.fold_id, spec["experiment_id"])
            if key in completed_keys:
                continue
            trainer = TabularTrainer(spec, runner, loader)
            result = trainer.run_fold(fold_data)
            results.append(result.to_dict())

    return results


# ================================================================== #
# 辅助：fold 列表切分
# ================================================================== #

def _build_fold_groups(
    profile_name: str,
    folds: list,
    target_group_size: int = 7,
) -> List[FoldGroup]:
    """
    把单个 profile 的 fold 列表切成若干**连续**子组。

    target_group_size: 每组目标 fold 数。
    相邻 fold 的 train_dates 重叠越多，同一组内的 cache 复用率越高。
    """
    n = len(folds)
    if n == 0:
        return []
    n_groups = max(1, math.ceil(n / target_group_size))
    size = math.ceil(n / n_groups)
    groups = []
    for i in range(0, n, size):
        chunk = folds[i: i + size]
        metas = [
            FoldMeta(
                profile_name=profile_name,
                fold_id=f.fold_id,
                train_dates=f.train_dates,
                val_dates=f.val_dates,
            )
            for f in chunk
        ]
        groups.append(FoldGroup(
            group_id=f"{profile_name}_g{i // size}",
            fold_metas=metas,
        ))
    return groups


def _estimate_fold_cost(fold) -> int:
    """
    估算单个 fold 的训练成本。

    这里直接用 train_dates 长度近似：
    - long / expanding 的主要成本来自训练窗口越来越长；
    - 比只看 fold_id 更稳，因为它直接对应真实载入天数。
    """
    return len(getattr(fold, "train_dates", ()) or ())


def _build_cost_balanced_fold_groups(
    profile_name: str,
    folds: list,
    n_groups: int = 2,
) -> List[FoldGroup]:
    """
    按 fold 成本做贪心均衡切分。

    用途：
    - #17 关口提速要求 heavy profile 不再按连续区间切组；
    - 后段 expanding fold 最重，若连续切分会把最重折堆在同一 worker；
    - 这里改成“先放最重折，再贪心塞给当前总成本最低的组”，
      让 2 个 worker 的负载更接近。
    """
    if not folds:
        return []
    actual_groups = max(1, min(int(n_groups), len(folds)))
    buckets = [
        {"cost": 0, "folds": []}
        for _ in range(actual_groups)
    ]
    indexed_folds = list(enumerate(folds))
    indexed_folds.sort(
        key=lambda item: (_estimate_fold_cost(item[1]), getattr(item[1], "fold_id", item[0])),
        reverse=True,
    )
    for original_index, fold in indexed_folds:
        target_bucket = min(
            buckets,
            key=lambda bucket: (
                bucket["cost"],
                len(bucket["folds"]),
            ),
        )
        target_bucket["folds"].append((original_index, fold))
        target_bucket["cost"] += _estimate_fold_cost(fold)

    groups: List[FoldGroup] = []
    for group_idx, bucket in enumerate(buckets):
        if not bucket["folds"]:
            continue
        # 组内仍按原始 fold 顺序执行，避免日志和结果顺序过于跳跃。
        ordered_pairs = sorted(bucket["folds"], key=lambda item: item[0])
        metas = [
            FoldMeta(
                profile_name=profile_name,
                fold_id=fold.fold_id,
                train_dates=fold.train_dates,
                val_dates=fold.val_dates,
            )
            for _, fold in ordered_pairs
        ]
        groups.append(
            FoldGroup(
                group_id=f"{profile_name}_g{group_idx}",
                fold_metas=metas,
            )
        )
    return groups


def _is_heavy_group(group: FoldGroup) -> bool:
    """
    判断 group 是否属于重任务 profile。

    当前只按 profile_name 做最小规则判断：
    - long_40d_5d
    - expanding_40d_5d

    不引入更复杂的预算模型，先用稳定的硬限制挡住 OOM。
    """
    if not group.fold_metas:
        return False
    profile_name = group.fold_metas[0].profile_name
    return profile_name.startswith("long_") or profile_name.startswith("expanding_")


def _is_heavy_profile_name(profile_name: str) -> bool:
    """把 heavy profile 判定单独抽出来，便于建组前决定策略。"""
    return profile_name.startswith("long_") or profile_name.startswith("expanding_")


# ================================================================== #
# ParallelScheduler
# ================================================================== #

class ParallelScheduler:
    """
    并发执行所有 (profile × fold × spec) 任务。

    - fold 按 profile 分组，相邻 fold 落同一 worker，最大化进程内 cache 复用。
    - 支持断点续跑：启动时读取已有 fold_metrics.csv，跳过已完成 job。
    - 实时落盘：每个 worker 完成后立即 append 到 fold_metrics.csv，
      主进程崩溃不丢已完成结果。
    """

    def __init__(
        self,
        h5dir: str,
        feature_dir: str = DEFAULT_FEATURE_DIR,
        n_workers: int = 4,
        heavy_max_workers: int = 2,
        target_winsorize_config: Optional[Dict[str, object]] = None,
        feature_dtype: str = "float32",
        ridge_alpha: float = 2.0,
        train_subsample_frac: Optional[float] = None,
    ):
        self.h5dir = h5dir
        self.feature_dir = feature_dir
        self.n_workers = n_workers
        self.heavy_max_workers = heavy_max_workers
        # P4 训练行降采样比例（None=关闭）。同 winsorize/dtype，必须显式透传给 worker，
        # 否则并行真实跑数会悄悄退回全量训练，与串行口径不一致。
        self.train_subsample_frac = train_subsample_frac
        # worker 内会重新实例化 ExperimentRunner/FeatureLoader，
        # 这里必须显式透传 feature_dtype，避免主进程与子进程口径漂移。
        self.feature_dtype = feature_dtype
        # 标准 ridge alpha 也必须显式透传给 worker，
        # 否则主进程扫参和子进程真实训练会用不同值。
        self.ridge_alpha = float(ridge_alpha)
        # worker 会在独立进程里重新实例化 ExperimentRunner，
        # 这里必须显式保存 winsorize 配置并一并透传，
        # 否则真实并行跑数时会悄悄退回默认值，形成口径不一致。
        self.target_winsorize_config = dict(target_winsorize_config or {})
        self._fold_metrics_path: Optional[str] = None

    def set_output_path(self, fold_metrics_path: str):
        """设置增量写入路径（resume 和实时落盘共用）"""
        self._fold_metrics_path = fold_metrics_path

    # ---------------------------------------------------------------- #
    # Resume 支持
    # ---------------------------------------------------------------- #

    def _load_completed_keys(self) -> FrozenSet[Tuple]:
        """
        从已有 fold_metrics.csv 读取已完成的 (profile_name, fold_id, experiment_id)。
        只有 status == "ok" 的行才算完成，error 行会被重跑。
        """
        if not self._fold_metrics_path:
            return frozenset()
        if not os.path.exists(self._fold_metrics_path):
            return frozenset()
        try:
            df = pd.read_csv(self._fold_metrics_path, encoding="utf-8-sig")
            if "status" in df.columns:
                df = df[df["status"] == "ok"]
            keys = frozenset(
                zip(
                    df["profile_name"].tolist(),
                    df["fold_id"].astype(int).tolist(),
                    df["experiment_id"].tolist(),
                )
            )
            return keys
        except Exception as e:
            print(f"[Scheduler] 读取 resume 记录失败（将从头开始）: {e}")
            return frozenset()

    # ---------------------------------------------------------------- #
    # 落盘
    # ---------------------------------------------------------------- #

    def _append_results(self, rows: List[dict]) -> None:
        """将一批 FoldResult 增量 append 到 fold_metrics.csv（主进程写，线程安全）"""
        if not self._fold_metrics_path or not rows:
            return
        df = pd.DataFrame(rows)
        write_header = not os.path.exists(self._fold_metrics_path)
        df.to_csv(
            self._fold_metrics_path,
            mode="a",
            header=write_header,
            index=False,
            encoding="utf-8-sig",
        )

    # ---------------------------------------------------------------- #
    # 主入口
    # ---------------------------------------------------------------- #

    def _run_group_batch(
        self,
        groups: List[FoldGroup],
        specs: List[dict],
        completed_keys: FrozenSet[Tuple],
        max_workers: int,
        t0: float,
        completed_groups: int,
        total_groups: int,
        new_rows: List[dict],
    ) -> int:
        """
        执行一批 group，并返回累计完成的 group 数。

        这里故意按批次跑：
        - light 批次：保持原 n_workers
        - heavy 批次：硬限制到 heavy_max_workers

        这样可以在不改训练链路的前提下，直接阻止重任务阶段出现 >2 个并发。
        """
        if not groups:
            return completed_groups

        # 至少保留 1 个 worker，避免外部误传 0 导致进程池报错。
        actual_workers = max(1, min(max_workers, self.n_workers))
        worker_args = [
            (
                self.h5dir,
                self.feature_dir,
                self.target_winsorize_config,
                self.feature_dtype,
                self.ridge_alpha,
                group,
                specs,
                completed_keys,
                self.train_subsample_frac,
            )
            for group in groups
        ]

        with ProcessPoolExecutor(max_workers=actual_workers) as pool:
            futures = {
                # args[5] 才是 FoldGroup；前面新增了 feature_dtype / ridge_alpha 两个透传参数。
                pool.submit(_fold_group_worker, args): args[5].group_id
                for args in worker_args
            }
            for future in as_completed(futures):
                group_id = futures[future]
                try:
                    rows = future.result()
                    self._append_results(rows)
                    new_rows.extend(rows)
                    completed_groups += 1
                    elapsed = time.time() - t0
                    ok_cnt = sum(1 for r in rows if r.get("status") == "ok")
                    err_cnt = len(rows) - ok_cnt
                    status_str = f"{ok_cnt} ok" + (f", {err_cnt} err" if err_cnt else "")
                    print(
                        f"  [✓] {group_id}  {status_str}"
                        f"  {completed_groups}/{total_groups} groups"
                        f"  {elapsed:.0f}s elapsed"
                    )
                except Exception as e:
                    completed_groups += 1
                    print(f"  [✗] {group_id} worker 进程失败: {e}")

        return completed_groups

    def run(
        self,
        profiles_with_folds: List[Tuple],   # [(RollingProfile, List[RollingFold])]
        specs: List[dict],
        resume: bool = True,
    ) -> pd.DataFrame:
        """
        并发执行所有 fold × spec 任务，实时落盘，支持 resume。

        返回：本次执行后 fold_metrics.csv 的完整 DataFrame
              （含之前已完成 + 本次新完成，供上层 summarize_profile 使用）。
        """
        completed_keys = self._load_completed_keys() if resume else frozenset()
        n_completed = len(completed_keys)

        # 展平所有 profile → FoldGroup 列表
        all_groups: List[FoldGroup] = []
        for profile, folds in profiles_with_folds:
            if _is_heavy_profile_name(profile.profile_name):
                groups = _build_cost_balanced_fold_groups(
                    profile.profile_name,
                    folds,
                    n_groups=2,
                )
            else:
                groups = _build_fold_groups(profile.profile_name, folds)
            all_groups.extend(groups)

        total_jobs = sum(len(g.fold_metas) for g in all_groups) * len(specs)
        pending_jobs = total_jobs - n_completed
        print(
            f"\n[Scheduler] {len(all_groups)} fold-groups，{total_jobs} 个 job"
            f"（{n_completed} 已完成，{pending_jobs} 待执行，n_workers={self.n_workers}）"
        )

        if pending_jobs <= 0:
            print("[Scheduler] 全部 job 已完成，直接读取历史结果。")
            return self._read_all_results()

        # 将重任务单独拆批，进入 long/expanding 阶段后硬降总并发到 2。
        light_groups = [group for group in all_groups if not _is_heavy_group(group)]
        heavy_groups = [group for group in all_groups if _is_heavy_group(group)]
        heavy_worker_cap = max(1, min(self.heavy_max_workers, self.n_workers))
        print(
            f"[Scheduler] 调度批次：light={len(light_groups)} groups @ {self.n_workers} workers，"
            f"heavy={len(heavy_groups)} groups @ {heavy_worker_cap} workers"
        )

        t0 = time.time()
        completed_groups = 0
        new_rows: List[dict] = []
        completed_groups = self._run_group_batch(
            groups=light_groups,
            specs=specs,
            completed_keys=completed_keys,
            max_workers=self.n_workers,
            t0=t0,
            completed_groups=completed_groups,
            total_groups=len(all_groups),
            new_rows=new_rows,
        )
        completed_groups = self._run_group_batch(
            groups=heavy_groups,
            specs=specs,
            completed_keys=completed_keys,
            max_workers=heavy_worker_cap,
            t0=t0,
            completed_groups=completed_groups,
            total_groups=len(all_groups),
            new_rows=new_rows,
        )

        total_elapsed = time.time() - t0
        print(f"\n[Scheduler] 完成，耗时 {total_elapsed:.1f}s，新增 {len(new_rows)} 条结果。")

        return self._read_all_results()

    def _read_all_results(self) -> pd.DataFrame:
        """读取 fold_metrics.csv 的完整内容（含历史 + 本次新增）"""
        if not self._fold_metrics_path or not os.path.exists(self._fold_metrics_path):
            return pd.DataFrame()
        try:
            return pd.read_csv(self._fold_metrics_path, encoding="utf-8-sig")
        except Exception as e:
            print(f"[Scheduler] 读取 fold_metrics.csv 失败: {e}")
            return pd.DataFrame()
