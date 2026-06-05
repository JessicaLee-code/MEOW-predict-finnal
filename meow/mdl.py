from pathlib import Path

import numpy as np

from experiment_runner import DEFAULT_TARGET_WINSORIZE
from log import log
from submission_pipeline import SubmissionModelPipeline, SubmissionSpec


class MeowModel(object):
    """模型包装层：保留 fit/predict 外部方法，内部委托 src/ 正式核心（winsorize/模型参数与实验链一致）。"""

    def __init__(self, cacheDir, h5dir):
        self.cacheDir = cacheDir
        # 提交通道不依赖持久化特征缓存；feature_dir 故意指到不存在路径，误接 FeatureLoader 会立刻暴露。
        unused_feature_dir = str(
            Path(h5dir).resolve().parent / "__meow_submission_no_cache__"
        )
        self.runtime = SubmissionModelPipeline(
            h5dir=h5dir,
            feature_dir=unused_feature_dir,
            spec=SubmissionSpec(),
            target_winsorize_config=DEFAULT_TARGET_WINSORIZE,
        )

    def fit(self, xdf, ydf):
        """非破坏式训练入口（训练后仍要复用入参时用）。"""
        self.runtime.fit(xdf, ydf)
        log.inf("Done fitting formal submission model")

    def fit_window(self, frames):
        """整窗消费式训练入口：接收 holder 并交出所有权，委托提交链在末位成员训练前释放整窗源帧、压低峰值。"""
        self.runtime.fit_window(frames)
        log.inf("Done fitting formal submission model")

    def member_specs(self):
        """返回正式提交成员列表，供 meow 入口编排「逐成员现算+fit」压内存。"""
        return self.runtime.member_specs()

    def begin_fit(self):
        """逐成员流式训练起始：重置成员训练态。"""
        self.runtime.begin_fit()

    def fit_one_member(self, member, frames):
        """训练单个成员（消费式）：frames 只含该成员 groups 特征+目标，逐成员现算→fit→释放把峰值压到单成员级（治 ridge 157 列 OOM）。"""
        self.runtime.fit_one_member(member, frames)

    def end_fit(self):
        """逐成员流式训练收尾：打印完成日志，与整窗路径日志口径一致。"""
        log.inf("Done fitting formal submission model")

    def predict(self, xdf):
        """用正式提交模型推理，强制输出 float32 一维数组。"""
        return np.asarray(self.runtime.predict(xdf), dtype=np.float32)
