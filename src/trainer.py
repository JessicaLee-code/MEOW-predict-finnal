"""
Trainer 层 — 定义实验执行的标准接口

FoldData:      单个 rolling fold 的元数据容器
FoldResult:    统一输出格式（列名与 fold_metrics.csv 完全兼容）
BaseTrainer:   抽象基类，DL 接入点
TabularTrainer: 封装现有传统 ML 逻辑，委托给 ExperimentRunner
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ================================================================== #
# 数据容器
# ================================================================== #

@dataclass
class FoldData:
    """
    Rolling fold 的元数据，Trainer 执行的输入单元。

    传统 ML（TabularTrainer）只使用 train_dates / val_dates，
    数据加载由内部 ExperimentRunner 负责。
    DL Trainer 可在预处理阶段将张量填入可选字段，避免重复构建。
    """
    profile_name: str
    fold_id: int
    train_dates: tuple   # 交易日 int 元组，已按时间升序排列
    val_dates: tuple

    # 以下字段仅供 DL Trainer 使用，传统 ML 保持 None
    xtrain: Optional[pd.DataFrame] = None
    ytrain: Optional[pd.DataFrame] = None
    xval:   Optional[pd.DataFrame] = None
    yval:   Optional[pd.DataFrame] = None


@dataclass
class FoldResult:
    """
    单个 (fold, spec) 的统一执行结果。

    字段与 fold_metrics.csv 保持兼容，额外增加 status / error_msg 用于 resume。
    """
    profile_name: str
    fold_id: int
    experiment_id: str
    feature_set: str
    model_type: str
    target_type: str
    postprocess_type: str

    train_corr: float
    val_corr: float
    train_mse: float
    val_mse: float
    train_r2: float
    val_r2: float
    daily_corr_mean: float
    daily_corr_std: float
    train_val_corr_gap: float
    runtime_sec: float

    # fold 日期信息（来自 FoldData，供 fold_metrics.csv 记录）
    train_start: int = 0
    train_end: int = 0
    val_start: int = 0
    val_end: int = 0
    n_train_days: int = 0
    n_val_days: int = 0

    # 状态字段（resume 判断依据）
    status: str = "ok"       # "ok" | "error"
    error_msg: str = ""

    notes: str = ""

    def to_dict(self) -> dict:
        """
        输出与旧版 fold_metrics.csv 列名兼容的字典。

        rolling_* / stability_score / random_seed 保留占位（fold 级别均为 NaN），
        与 summarize_profile 期望的列结构一致。
        """
        nan = float("nan")
        return {
            # 核心标识
            "profile_name":       self.profile_name,
            "fold_id":            self.fold_id,
            "experiment_id":      self.experiment_id,
            # 模型信息
            "feature_set":        self.feature_set,
            "model_type":         self.model_type,
            "target_type":        self.target_type,
            "postprocess_type":   self.postprocess_type,
            # 指标
            "train_corr":         self.train_corr,
            "val_corr":           self.val_corr,
            "train_mse":          self.train_mse,
            "val_mse":            self.val_mse,
            "train_r2":           self.train_r2,
            "val_r2":             self.val_r2,
            "daily_corr_mean":    self.daily_corr_mean,
            "daily_corr_std":     self.daily_corr_std,
            # fold 汇总占位（由 summarize_profile 填充）
            "rolling_corr_mean":  nan,
            "rolling_corr_std":   nan,
            "rolling_corr_min":   nan,
            "rolling_mse_mean":   nan,
            "rolling_r2_mean":    nan,
            "stability_score":    nan,
            # 其他
            "train_val_corr_gap": self.train_val_corr_gap,
            "runtime_sec":        self.runtime_sec,
            "random_seed":        42,
            "notes":              self.notes,
            # 日期信息
            "train_start":        self.train_start,
            "train_end":          self.train_end,
            "val_start":          self.val_start,
            "val_end":            self.val_end,
            "n_train_days":       self.n_train_days,
            "n_val_days":         self.n_val_days,
            # resume 状态
            "status":             self.status,
            "error_msg":          self.error_msg,
        }


# ================================================================== #
# Trainer 接口
# ================================================================== #

class BaseTrainer(ABC):
    """
    所有 Trainer 的抽象基类。

    DL 接入时只需实现此接口，其余层（调度、协议、leaderboard）一字不改：

        class SequenceTrainer(BaseTrainer):
            def run_fold(self, fold_data: FoldData) -> FoldResult:
                # 处理 epoch / early stopping / GPU / checkpoint
                ...
    """

    def __init__(self, spec: dict):
        self.spec = spec

    @abstractmethod
    def run_fold(self, fold_data: FoldData) -> FoldResult:
        """
        在单个 rolling fold 上执行完整的训练-验证流程。

        实现负责：特征加载/过滤、模型训练、预测、指标计算。
        不负责：fold 切分、结果落盘、并发调度。
        """
        ...


class TabularTrainer(BaseTrainer):
    """
    封装现有传统 ML 执行逻辑（Ridge / Tree / ElasticNet 等）。

    完全委托给 ExperimentRunner._evaluate_spec_on_fold，不重复任何模型逻辑。
    runner 由调度层在 worker 进程内创建，跨同一 worker 的多个 fold 共享，
    以复用同一份 FeatureLoader / 模型执行上下文。
    """

    def __init__(self, spec: dict, runner, loader):
        """
        spec:   实验配置 dict（与 ALL_SPECS 中的格式一致）
        runner: 进程内的 ExperimentRunner 实例
        loader: FeatureLoader 实例。M4 起由调度层显式注入，避免 Trainer
                再隐式依赖 runner 内部的旧缓存加载逻辑。
        """
        super().__init__(spec)
        self.runner = runner
        self.loader = loader

    def run_fold(self, fold_data: FoldData) -> FoldResult:
        from experiment_runner import SplitConfig

        fold_split = SplitConfig(
            train_start=fold_data.train_dates[0],
            train_end=fold_data.train_dates[-1],
            val_start=fold_data.val_dates[0],
            val_end=fold_data.val_dates[-1],
            test_start=fold_data.val_dates[0],
            test_end=fold_data.val_dates[-1],
        )
        start_ts = time.time()
        try:
            bundle = self.runner._evaluate_spec_on_fold(
                fold_split,
                self.spec,
                loader=self.loader,
            )
            vm = bundle["val_metrics"]
            tm = bundle["train_metrics"]
            return FoldResult(
                profile_name=fold_data.profile_name,
                fold_id=fold_data.fold_id,
                experiment_id=self.spec["experiment_id"],
                feature_set=bundle["feature_set"],
                model_type=bundle["model_type"],
                target_type=bundle["target_type"],
                postprocess_type=bundle["postprocess_type"],
                train_corr=float(tm["corr"]),
                val_corr=float(vm["corr"]),
                train_mse=float(tm["mse"]),
                val_mse=float(vm["mse"]),
                train_r2=float(tm["r2"]),
                val_r2=float(vm["r2"]),
                daily_corr_mean=float(vm.get("daily_corr_mean", 0.0)),
                daily_corr_std=float(vm.get("daily_corr_std", 0.0)),
                train_val_corr_gap=float(tm["corr"] - vm["corr"]),
                runtime_sec=float(time.time() - start_ts),
                train_start=fold_data.train_dates[0],
                train_end=fold_data.train_dates[-1],
                val_start=fold_data.val_dates[0],
                val_end=fold_data.val_dates[-1],
                n_train_days=len(fold_data.train_dates),
                n_val_days=len(fold_data.val_dates),
                status="ok",
                error_msg="",
                notes=self.spec.get("notes", ""),
            )
        except Exception as e:
            nan = float("nan")
            return FoldResult(
                profile_name=fold_data.profile_name,
                fold_id=fold_data.fold_id,
                experiment_id=self.spec["experiment_id"],
                feature_set="",
                model_type=self.spec.get("model", "unknown"),
                target_type=self.spec.get("target_mode", "raw"),
                postprocess_type="none",
                train_corr=nan, val_corr=nan,
                train_mse=nan,  val_mse=nan,
                train_r2=nan,   val_r2=nan,
                daily_corr_mean=nan, daily_corr_std=nan,
                train_val_corr_gap=nan,
                runtime_sec=float(time.time() - start_ts),
                train_start=fold_data.train_dates[0],
                train_end=fold_data.train_dates[-1],
                val_start=fold_data.val_dates[0],
                val_end=fold_data.val_dates[-1],
                n_train_days=len(fold_data.train_dates),
                n_val_days=len(fold_data.val_dates),
                status="error",
                error_msg=str(e)[:500],
                notes=self.spec.get("notes", ""),
            )
