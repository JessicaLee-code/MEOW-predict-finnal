# -*- coding: utf-8 -*-
"""
正式提交通道桥接层单元测试。

覆盖目标：
1. 从 raw 现算正式提交特征时，列集合要和 spec 解析结果一致。
2. 训练 / 预测桥接层可以在不依赖磁盘特征缓存的情况下跑通。
"""

import importlib.util
import os
import sys
import tempfile
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from feature_registry import META_COLS, TARGET_COL, _make_schema_probe_raw  # noqa: E402
from submission_pipeline import (  # noqa: E402
    DEFAULT_SUBMISSION_GROUPS,
    SubmissionFeaturePipeline,
    SubmissionMemberSpec,
    SubmissionModelPipeline,
    SubmissionSpec,
)


class TestSubmissionFeaturePipeline(unittest.TestCase):
    """保证正式提交特征桥接层不会和 registry/group 解析漂移。"""

    def test_feature_names_match_built_columns(self):
        raw = _make_schema_probe_raw()
        pipeline = SubmissionFeaturePipeline(groups=DEFAULT_SUBMISSION_GROUPS)
        xdf, ydf = pipeline.build_feature_frames(raw)
        self.assertEqual(list(xdf.columns[:3]), META_COLS)
        self.assertEqual(list(ydf.columns), META_COLS + [TARGET_COL])
        self.assertEqual(
            list(xdf.columns[3:]),
            pipeline.feature_names(),
        )


class TestSubmissionModelPipeline(unittest.TestCase):
    """保证正式提交训练/推理桥接层可以独立跑通。"""

    def test_fit_predict_smoke(self):
        raw = _make_schema_probe_raw()
        feature_pipeline = SubmissionFeaturePipeline(groups=DEFAULT_SUBMISSION_GROUPS)
        xdf, ydf = feature_pipeline.build_feature_frames(raw)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_pipeline = SubmissionModelPipeline(h5dir=tmpdir)
            model_pipeline.fit(xdf, ydf)
            pred = model_pipeline.predict(xdf)
            self.assertEqual(pred.shape[0], xdf.shape[0])
            self.assertTrue(np.isfinite(pred).all())

    def _blend_fixture(self, blend_mode):
        """构造一个跨两天、两成员的小样本，返回 (frame, member_preds, pipeline)。"""
        frame = _make_schema_probe_raw().iloc[:4].copy()
        frame.loc[:, "date"] = np.array([20230601, 20230601, 20230602, 20230602], dtype=int)
        frame = frame.reset_index(drop=True)
        member_a = np.array([1.0, 2.0, 10.0, 20.0], dtype=np.float32)
        member_b = np.array([2.0, 4.0, 30.0, 10.0], dtype=np.float32)
        spec = SubmissionSpec(
            members=(
                SubmissionMemberSpec("member_a", ("legacy",), "ridge"),
                SubmissionMemberSpec("member_b", ("legacy",), "ridge"),
            ),
            blend_mode=blend_mode,
        )
        return frame, {"member_a": member_a, "member_b": member_b}, spec

    def test_blend_raw_mean_is_default(self):
        """
        提交默认融合 = raw_mean：直接等权平均、输出留在原始量纲（保 MSE/R²）。

        member_a=[1,2,10,20], member_b=[2,4,30,10] → 等权平均 [1.5,3,20,15]。
        同时断言默认 spec 的 blend_mode 就是 raw_mean，防止有人误改回 zscore。
        """
        self.assertEqual(SubmissionSpec().blend_mode, "raw_mean")
        frame, member_preds, spec = self._blend_fixture("raw_mean")
        with tempfile.TemporaryDirectory() as tmpdir:
            model_pipeline = SubmissionModelPipeline(h5dir=tmpdir, spec=spec)
            pred = model_pipeline._blend_member_predictions(frame, member_preds)
        expected = np.array([1.5, 3.0, 20.0, 15.0], dtype=np.float32)
        np.testing.assert_allclose(pred, expected, atol=1e-6)

    def test_fit_window_matches_fit(self):
        """
        整窗消费式 `fit_window` 与非破坏式 `fit` 必须训出等价模型、给出一致预测。

        这是内存优化（末位成员训练前释放整窗源帧 + 直接喂 numpy）的安全网：
        只压内存、不改训练数学。
        """
        raw = _make_schema_probe_raw()
        feature_pipeline = SubmissionFeaturePipeline(groups=DEFAULT_SUBMISSION_GROUPS)
        xdf, ydf = feature_pipeline.build_feature_frames(raw)
        with tempfile.TemporaryDirectory() as tmpdir:
            p_fit = SubmissionModelPipeline(h5dir=tmpdir)
            p_fit.fit(xdf, ydf)
            pred_fit = p_fit.predict(xdf)

            # fit_window 会消费并释放入参 → 喂独立拷贝；随后仍用原 xdf 推理。
            p_win = SubmissionModelPipeline(h5dir=tmpdir)
            p_win.fit_window({"xdf": xdf.copy(), "ydf": ydf.copy()})
            pred_win = p_win.predict(xdf)
        np.testing.assert_allclose(pred_fit, pred_win, rtol=1e-5, atol=1e-5)

    def test_predict_blends_per_day_zscore(self):
        """
        保证 per_day_zscore_mean 模式是“按天截面标准化后再平均”，而非全局 pooled zscore。

        这个模式现在只作诊断对照，但仍卡住口径里最容易漂移的点：
        - 标准化必须用“当天数据自身”
        - 不能偷用训练期统计量、不能把所有日期混在一起 pooled zscore
        """
        frame, member_preds, spec = self._blend_fixture("per_day_zscore_mean")
        with tempfile.TemporaryDirectory() as tmpdir:
            model_pipeline = SubmissionModelPipeline(h5dir=tmpdir, spec=spec)
            pred = model_pipeline._blend_member_predictions(frame, member_preds)
        expected = np.array([-1.0, 1.0, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_allclose(pred, expected, atol=1e-6)


def _load_meow_engine_class():
    """
    按文件路径加载 meow/meow.py 的 MeowEngine。

    注意：src/ 下存在同名 `dl`/`mdl` 模块（实验 driver 用），若测试已先 import 过 src
    侧实现，`dl` 会被缓存成 src 版本（无 countDate）。生产环境老师从 meow/ 启动 `python
    meow.py` 时 `dl` 先被 meow/dl.py 缓存、不受影响；这里只为测试复现 meow 入口，
    故 exec 前把这些名字从 sys.modules 清掉、并把 meow_dir 置于 path 最前，强制从 meow/ 解析。
    """
    meow_dir = os.path.join(REPO_ROOT, "meow")
    sys.path.insert(0, meow_dir)  # meow 内部裸 import（dl/feat/mdl…）优先从 meow/ 解析
    for name in ("dl", "feat", "mdl"):
        sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(
        "meow_entry_for_test", os.path.join(meow_dir, "meow.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MeowEngine


def _meow_data_available(dates):
    data_dir = os.path.join(REPO_ROOT, "data")
    return all(os.path.exists(os.path.join(data_dir, "{}.h5".format(d))) for d in dates)


class TestMeowWindowBuild(unittest.TestCase):
    """保证 meow 整窗「预分配流式填充」与朴素 concat 结果逐元素一致（只压内存、不改数值）。"""

    DATES = [20230601, 20230602]

    @unittest.skipUnless(
        _meow_data_available([20230601, 20230602]),
        "缺少 data/2023060{1,2}.h5，跳过整窗构造等价性测试",
    )
    def test_streaming_build_matches_naive_concat(self):
        MeowEngine = _load_meow_engine_class()
        data_dir = os.path.join(REPO_ROOT, "data")
        engine = MeowEngine(h5dir=data_dir, cacheDir=None)
        dates = list(self.DATES)

        # 朴素参照：逐日 genFeatures 后整体 concat。
        x_parts, y_parts = [], []
        for d in dates:
            raw = engine.dloader.loadDate(d)
            xday, yday = engine.featGenerator.genFeatures(raw)
            x_parts.append(xday.reset_index())
            y_parts.append(yday.reset_index())
        xref = pd.concat(x_parts, ignore_index=True)
        yref = pd.concat(y_parts, ignore_index=True)

        frames = engine._build_window_frames(dates)
        xdf, ydf = frames["xdf"], frames["ydf"]

        # 行数与 meta 完全一致。
        self.assertEqual(len(xdf), len(xref))
        for col in ["date", "symbol", "interval"]:
            np.testing.assert_array_equal(
                xdf[col].to_numpy(), xref[col].to_numpy()
            )
        # 每个特征列逐元素一致（按列名对齐）。
        feat_cols = engine.featGenerator.featureNames()
        for col in feat_cols:
            np.testing.assert_allclose(
                xdf[col].to_numpy(dtype=np.float32),
                xref[col].to_numpy(dtype=np.float32),
                rtol=0,
                atol=0,
                err_msg="特征列不一致: {}".format(col),
            )
        # 标签一致。
        np.testing.assert_allclose(
            ydf["fret12"].to_numpy(dtype=np.float32),
            yref["fret12"].to_numpy(dtype=np.float32),
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
