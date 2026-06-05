import pandas as pd
from log import log

from submission_pipeline import DEFAULT_SUBMISSION_GROUPS, SubmissionFeaturePipeline


class MeowFeatureGenerator(object):
    """特征包装层：类名/方法名保持老师样例风格，内部直接调用 src/ 正式特征管线（不维护第二套 baseline 公式）。"""

    @classmethod
    def featureNames(cls, groups=None):
        """返回提交特征列名（groups=None 取全部成员并集；从共享管线解析、不手写列名以防漂移）。"""
        resolved_groups = groups if groups is not None else DEFAULT_SUBMISSION_GROUPS
        return SubmissionFeaturePipeline(groups=resolved_groups).feature_names()

    def __init__(self, cacheDir):
        self.cacheDir = cacheDir
        self.ycol = "fret12"
        self.mcols = ["date", "symbol", "interval"]
        self.pipeline = SubmissionFeaturePipeline(groups=DEFAULT_SUBMISSION_GROUPS)

    def genFeatures(self, df, groups=None):
        """从 raw DataFrame 现算提交特征（不依赖持久化缓存，返回 MultiIndex 形态；公式与实验主链共用）。

        groups=None 现算全部成员并集；传入某成员 groups 则只算该成员，供交付链逐成员把 fit 峰值压到单成员级。
        """
        resolved_groups = groups if groups is not None else DEFAULT_SUBMISSION_GROUPS
        log.inf(
            "Generating {} formal submission features from raw data...".format(
                len(self.featureNames(resolved_groups))
            )
        )
        xdf, ydf = self.pipeline.build_feature_frames(df, groups=resolved_groups)
        xdf = xdf.set_index(self.mcols)
        ydf = ydf.set_index(self.mcols)
        return xdf.fillna(0.0), ydf.fillna(0.0)
