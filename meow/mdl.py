from pathlib import Path

import numpy as np

from experiment_runner import DEFAULT_TARGET_WINSORIZE
from log import log
from submission_pipeline import SubmissionModelPipeline, SubmissionSpec


class MeowModel(object):
    """
    老师 `meow.py` 入口使用的模型包装层。

    这里保留 `MeowModel.fit/predict` 这组外部方法，
    但内部训练与推理全部委托给 `src/` 中的正式核心实现，
    从而保证：
    - winsorize 与实验链一致
    - 模型参数与实验链一致
    - 提交通道不再维护第二套 baseline 训练逻辑
    """

    def __init__(self, cacheDir, h5dir):
        self.cacheDir = cacheDir
        # 提交通道明确声明“不依赖持久化特征缓存”。
        # 因此这里故意把 feature_dir 指到一个本来就不存在的路径：
        # - 若后续有人误把提交链接回 FeatureLoader / data/features，运行时会立刻暴露
        # - 当前正式实现只复用训练/推理核心，不会真正去读这个目录
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
        """
        非破坏式训练入口（训练后仍要复用入参时用）。

        这里显式保留日志，是为了老师直接跑 `python meow.py` 时，
        能从终端看到提交链已经进入我们的正式训练实现，而不是样例 baseline。
        """

        self.runtime.fit(xdf, ydf)
        log.inf("Done fitting formal submission model")

    def fit_window(self, frames):
        """
        整窗消费式训练入口：接收 `{"xdf","ydf"}` holder 并交出所有权，
        委托正式提交链在末位成员训练前释放整窗源帧、压低内存峰值。
        """

        self.runtime.fit_window(frames)
        log.inf("Done fitting formal submission model")

    def predict(self, xdf):
        """
        用正式提交模型推理，并强制输出 float32 数组。

        这样做的原因：
        - 保证 `forecast` 列是干净的一维数值结果
        - 和实验主链的预测 dtype 保持一致
        """

        return np.asarray(self.runtime.predict(xdf), dtype=np.float32)
