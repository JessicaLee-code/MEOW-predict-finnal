import pandas as pd
from log import log

from submission_pipeline import DEFAULT_SUBMISSION_GROUPS, SubmissionFeaturePipeline


class MeowFeatureGenerator(object):
    """
    老师 `meow.py` 入口使用的特征包装层。

    外部类名 / 方法名保持老师样例风格不变，
    但内部不再维护独立 baseline 特征公式，而是直接调用 `src/` 中的正式特征管线。
    """

    @classmethod
    def featureNames(cls):
        """
        返回当前正式提交 spec 的特征列名。

        注意这里不再手写列名列表，而是直接从共享特征管线解析，
        防止实验链和提交通道因为“列名手工维护”出现漂移。
        """

        return SubmissionFeaturePipeline(groups=DEFAULT_SUBMISSION_GROUPS).feature_names()

    def __init__(self, cacheDir):
        self.cacheDir = cacheDir
        self.ycol = "fret12"
        self.mcols = ["date", "symbol", "interval"]
        self.pipeline = SubmissionFeaturePipeline(groups=DEFAULT_SUBMISSION_GROUPS)

    def genFeatures(self, df):
        """
        从老师传入的 raw DataFrame 现算正式提交特征。

        关键约束：
        - 不依赖 `data/features/` 持久化特征缓存
        - 返回仍保持老师样例喜欢的 `MultiIndex` 形态
        - 但内部真正使用的特征公式与实验主链完全共用
        """

        log.inf(
            "Generating {} formal submission features from raw data...".format(
                len(self.featureNames())
            )
        )
        xdf, ydf = self.pipeline.build_feature_frames(df, groups=DEFAULT_SUBMISSION_GROUPS)
        xdf = xdf.set_index(self.mcols)
        ydf = ydf.set_index(self.mcols)
        return xdf.fillna(0.0), ydf.fillna(0.0)
