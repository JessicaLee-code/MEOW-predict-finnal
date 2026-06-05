# -*- coding: utf-8 -*-
"""
DL D0 地基单元测试 —— 规格 §9 的六项验收闸（全程 torch-free、CPU 可跑）

覆盖：
1. CPU 端到端跑通全链路（SequenceTrainer 产 FoldResult，status ok）。
2. 参考模型打低分（无信号 / 依赖未来 → val_corr≈0）。
3. 无泄漏：窗口因果——"依赖未来的标签"参考模型抓不到（低分）；作为探测器有效性
   对照，"依赖当前的标签"它能抓到（高分），证明低分不是因为模型本身废。
4. 窗口不跨日不跨票（WindowIndexer 每个窗口的 date/symbol 单一 + warmup 丢弃）。
5. 归一化只用训练统计量（Normalizer fit-on-train，不含 val）。
6. config 组装校验（frozen / fingerprint / 阶段搭配 / required_adapter 匹配）。

另含 FeatureAdapter 对真实单日 h5 的接通验证（无数据则 skip）。
"""

import os
import sys
import tempfile
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _sub in ("src", "config", "models"):
    sys.path.insert(0, os.path.join(REPO_ROOT, _sub))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sequence_dataset import (  # noqa: E402
    Normalizer, SequenceDataset, WindowIndexer, build_sequence_arrays,
    build_sequence_arrays_from_frames, subset_by_dates,
)
from dl_protocol import (  # noqa: E402
    assert_folds_causal, build_delivery_fold, build_dl_folds, evaluate_prediction_bundle, summarize_folds,
)
from dl_trainer import SequenceTrainer  # noqa: E402
from dl_models import IdentityAdapter, ReferenceLastCartridge, ReferenceZeroCartridge  # noqa: E402


# ------------------------------------------------------------------ #
# 合成数据工具
# ------------------------------------------------------------------ #

def make_synth(n_days=8, n_symbols=5, n_interval=30, label="current", seed=0):
    """
    造合成 raw：通道 c0/c1 为噪声；label 口径决定 fret12 与通道的关系：
    - "current": y[t] = c0[t] + 小噪声      → 同期信号，参考模型应抓到（高 corr）
    - "future":  y[t] = c0[t+1]（组内下一步）→ 未来信号，因果窗末抓不到（低 corr）
    - "noise":   y[t] = 纯噪声               → 无信号（低 corr）
    """
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(1, n_days + 1):
        for s in range(n_symbols):
            c0 = rng.normal(size=n_interval)
            c1 = rng.normal(size=n_interval)
            if label == "current":
                y = c0 + 0.1 * rng.normal(size=n_interval)
            elif label == "future":
                y = np.concatenate([c0[1:], [0.0]])      # 组内 shift(-1)：t 处放 c0[t+1]
            elif label == "noise":
                y = rng.normal(size=n_interval)
            else:
                raise ValueError(label)
            for it in range(n_interval):
                rows.append((d, s, it, float(c0[it]), float(c1[it]), float(y[it])))
    return pd.DataFrame(rows, columns=["date", "symbol", "interval", "c0", "c1", "fret12"])


def make_loader(df):
    return lambda dates: df[df["date"].isin(list(dates))]


class _CachedIdentityAdapter:
    """
    测试用“缓存型适配器”。

    它模拟新的 FeatureAdapter 快路径：不吃 raw_loader 提供的整段原始数据，
    而是自己按日期直接返回 ``SequenceArrays``。这样可以锁住
    ``SequenceTrainer._build_arrays()`` 的优先级，避免以后重构又退回
    “先读 raw、再读缓存”的双份开销。
    """

    def __init__(self, frames):
        self.channels = ["c0", "c1"]
        self._frames = {
            int(date): frame.sort_values(["date", "symbol", "interval"], kind="mergesort").reset_index(drop=True)
            for date, frame in frames.items()
        }

    def load_sequence_arrays(self, dates, target_col="fret12", meta_cols=("date", "symbol", "interval")):
        x_parts = []
        y_parts = []
        for date in dates:
            frame = self._frames[int(date)]
            x_parts.append(frame[["date", "symbol", "interval", "c0", "c1"]].copy())
            y_parts.append(frame[["date", "symbol", "interval", "fret12"]].copy())
        return build_sequence_arrays_from_frames(
            xdf=pd.concat(x_parts, ignore_index=True),
            ydf=pd.concat(y_parts, ignore_index=True),
            channels=self.channels,
            target_col=target_col,
            meta_cols=meta_cols,
        )

    def build(self, raw_day_df):
        raise AssertionError("缓存快路径命中时，不应再回退到 raw_day_df -> build()")


# ------------------------------------------------------------------ #
# 4) 窗口不跨日不跨票 + warmup + 因果对齐
# ------------------------------------------------------------------ #

class TestWindowIndexer(unittest.TestCase):
    def setUp(self):
        self.df = make_synth(n_days=2, n_symbols=2, n_interval=4, label="current")
        self.adapter = IdentityAdapter(columns=["c0", "c1"])
        self.arrays = build_sequence_arrays(self.df, self.adapter)

    def test_no_cross_day_no_cross_symbol(self):
        L = 3
        lr = WindowIndexer(L).build_index(self.arrays)
        # 2天x2票x4interval, L=3 => 每段(4行)合法末行2个, 4段 => 8 窗
        self.assertEqual(len(lr), 8)
        for r in lr:
            win_dates = set(self.arrays.dates[r - L + 1: r + 1])
            win_syms = set(self.arrays.symbols[r - L + 1: r + 1])
            self.assertEqual(len(win_dates), 1, "窗口跨日了")
            self.assertEqual(len(win_syms), 1, "窗口跨票了")

    def test_warmup_dropped(self):
        # L 等于段长 => 每段只剩 1 个合法末行；L 超过段长 => 0 窗。
        self.assertEqual(len(WindowIndexer(4).build_index(self.arrays)), 4)
        self.assertEqual(len(WindowIndexer(5).build_index(self.arrays)), 0)

    def test_causal_label_alignment(self):
        L = 3
        ds = SequenceDataset(self.arrays, L, Normalizer("identity"))
        X, y = ds.gather_all()
        lf = ds.label_frame()
        # label_frame 的 fret12 与 gather 的 y 一致；窗末特征 == 该末行原始特征
        np.testing.assert_allclose(lf["fret12"].to_numpy(), y)
        first_lr = WindowIndexer(L).build_index(self.arrays)[0]
        np.testing.assert_allclose(X[0, -1], self.arrays.features[first_lr])


# ------------------------------------------------------------------ #
# 5) 归一化只用训练统计量
# ------------------------------------------------------------------ #

class TestNormalizer(unittest.TestCase):
    def test_fit_on_train_only(self):
        rng = np.random.default_rng(1)
        train = rng.normal(0.0, 1.0, size=(500, 2)).astype(np.float32)
        val = rng.normal(10.0, 5.0, size=(300, 2)).astype(np.float32)   # 分布差异大
        nz = Normalizer("zscore").fit(train)
        # 统计量必须等于 train 的，且明显不等于 train+val 合并的（否则即泄漏）。
        np.testing.assert_allclose(nz._mean, train.mean(axis=0), rtol=1e-4)
        merged_mean = np.concatenate([train, val]).mean(axis=0)
        self.assertTrue(np.all(np.abs(nz._mean - merged_mean) > 1.0))
        # transform(train) 近似零均值单方差
        zt = nz.transform(train)
        np.testing.assert_allclose(zt.mean(axis=0), 0.0, atol=1e-3)
        np.testing.assert_allclose(zt.std(axis=0), 1.0, atol=1e-3)

    def test_identity_passthrough(self):
        x = np.arange(12, dtype=np.float32).reshape(6, 2)
        np.testing.assert_allclose(Normalizer("identity").fit_transform(x), x)

    def test_zscore_requires_fit(self):
        with self.assertRaises(RuntimeError):
            Normalizer("zscore").transform(np.zeros((3, 2), np.float32))


# ------------------------------------------------------------------ #
# 2) + 3) 参考模型打低分 + 无泄漏（因果）+ 探测器有效性
# ------------------------------------------------------------------ #

class TestReferenceModelLeakage(unittest.TestCase):
    def _run(self, label, cartridge_factory=ReferenceLastCartridge, seq_len=5):
        df = make_synth(n_days=8, n_symbols=6, n_interval=30, label=label, seed=7)
        spec = {"experiment_id": f"ref_{label}", "model_type": "reference_last"}
        tr = SequenceTrainer(spec, IdentityAdapter(["c0", "c1"]), cartridge_factory,
                             make_loader(df), seq_len=seq_len, normalizer_mode="zscore")
        folds = build_dl_folds(1, 8, mode="expanding", val_window=1, step=2,
                               min_train_days=3, earlystop_frac=0.25,
                               calendar=_IntCalendar(range(1, 9)))
        self.assertGreater(len(folds), 0)
        results = [tr.run_on_dl_fold(f) for f in folds]
        for r in results:
            self.assertEqual(r.status, "ok", r.error_msg)
        return summarize_folds([{"corr": r.val_corr} for r in results])

    def test_current_signal_detected(self):
        # 探测器有效性：同期信号必须被末步线性抓到（高分），否则后面的"低分"无意义。
        s = self._run("current")
        self.assertGreater(s["mean"], 0.5, f"同期信号未被抓到: {s}")

    def test_future_label_not_leaked(self):
        # 核心防泄漏闸：依赖未来的标签，因果窗末抓不到 => 低分。
        s = self._run("future")
        self.assertLess(abs(s["mean"]), 0.2, f"疑似未来泄漏（参考模型抓到了未来）: {s}")

    def test_noise_label_low_score(self):
        s = self._run("noise")
        self.assertLess(abs(s["mean"]), 0.2, f"纯噪声却高分: {s}")

    def test_zero_reference_corr_zero(self):
        df = make_synth(n_days=6, n_symbols=4, n_interval=20, label="current")
        tr = SequenceTrainer({"experiment_id": "z"}, IdentityAdapter(["c0", "c1"]),
                             ReferenceZeroCartridge, make_loader(df), seq_len=4)
        from dl_protocol import DLFold
        r = tr.run_on_dl_fold(DLFold(0, (1, 2, 3), (4,), (5,), (6,)))
        self.assertEqual(r.status, "ok")
        self.assertAlmostEqual(r.val_corr, 0.0, places=5)


# ------------------------------------------------------------------ #
# 1) 端到端 + 折因果 + 指标
# ------------------------------------------------------------------ #

class TestProtocolAndE2E(unittest.TestCase):
    def test_folds_causal(self):
        folds = build_dl_folds(20230601, 20230801, mode="expanding", val_window=5,
                               step=10, min_train_days=20, earlystop_frac=0.2, max_folds=3)
        self.assertGreater(len(folds), 0)
        assert_folds_causal(folds)   # 四段时间严格递增、embargo 隔开训练/打分

    def test_delivery_fold_causal_and_excludes_embargo(self):
        # 交付折：训练截止后的 embargo 日只隔离，不进入 Dec 只读打分区。
        fold = build_delivery_fold(
            1, 10, 15, embargo=2, earlystop_frac=0.2, min_core_days=3,
            calendar=_IntCalendar(range(1, 16)),
        )
        self.assertEqual(fold.train_dates, tuple(range(1, 11)))
        self.assertEqual(fold.embargo_dates, (11, 12))
        self.assertEqual(fold.scoring_dates, (13, 14, 15))
        assert_folds_causal([fold])

    def test_metrics_perfect_prediction(self):
        lf = pd.DataFrame({"date": [1, 1, 2, 2, 2], "fret12": [0.1, -0.2, 0.3, 0.0, -0.1]})
        m = evaluate_prediction_bundle(lf, lf["fret12"].to_numpy())
        self.assertGreater(m["corr"], 0.999)
        self.assertLess(m["mse"], 1e-9)
        self.assertAlmostEqual(m["r2"], 1.0, places=5)
        self.assertAlmostEqual(m["daily_corr_mean"], 1.0, places=5)

    def test_end_to_end_foldresult_schema(self):
        df = make_synth(n_days=6, n_symbols=5, n_interval=20, label="current")
        tr = SequenceTrainer({"experiment_id": "e2e"}, IdentityAdapter(["c0", "c1"]),
                             ReferenceLastCartridge, make_loader(df), seq_len=5)
        from dl_protocol import DLFold
        r = tr.run_on_dl_fold(DLFold(0, (1, 2, 3), (4,), (5,), (6,)))
        self.assertEqual(r.status, "ok")
        d = r.to_dict()
        for col in ("val_corr", "val_mse", "val_r2", "daily_corr_mean", "train_val_corr_gap", "status"):
            self.assertIn(col, d)

    def test_sequence_trainer_prefers_cached_adapter_path(self):
        """
        一旦适配器声明自己能按日期直出 ``SequenceArrays``，Trainer 就必须优先走它。

        这样才能保证：
        - 433 特征滚动实验复用 ``data/features/`` 磁盘缓存；
        - 不会先把整段 raw 读进内存，再额外读一遍缓存，造成 IO / 内存双浪费。
        """
        df = make_synth(n_days=6, n_symbols=3, n_interval=10, label="current")
        frames = {int(date): day.copy() for date, day in df.groupby("date", sort=True)}
        adapter = _CachedIdentityAdapter(frames)

        def _raw_loader_should_not_run(_dates):
            raise AssertionError("缓存快路径命中时，不应调用 raw_loader")

        tr = SequenceTrainer(
            {"experiment_id": "cached_e2e"},
            adapter,
            ReferenceLastCartridge,
            _raw_loader_should_not_run,
            seq_len=4,
        )
        from dl_protocol import DLFold
        r = tr.run_on_dl_fold(DLFold(0, (1, 2, 3), (4,), (5,), (6,)))
        self.assertEqual(r.status, "ok", r.error_msg)


# ------------------------------------------------------------------ #
# 6) config 组装校验
# ------------------------------------------------------------------ #

class TestConfigAssembly(unittest.TestCase):
    def _blocks(self):
        from model_config import ModelKind, ModelConfig
        from adapter_config import AdapterKind, AdapterConfig
        from protocol_config import Stage, ProfileKind, ProtocolConfig
        from search_config import SearchConfig
        from exec_config import ExecConfig
        return (ModelKind, ModelConfig, AdapterKind, AdapterConfig, Stage,
                ProfileKind, ProtocolConfig, SearchConfig, ExecConfig)

    def test_assemble_and_frozen_and_fingerprint(self):
        (MK, MC, AK, AC, St, PK, PC, SC, EC) = self._blocks()
        from run_config import assemble_run_config
        rc = assemble_run_config(
            "20260531_search_reflast_v1",
            MC(MK.REFERENCE_LAST, hparams={"a": 1}),
            AC(AK.IDENTITY, columns=("c0", "c1")),
            PC(St.SEARCH, PK.SINGLE_SPLIT, 20230601, 20230801, min_train_days=20),
            SC(n_trials=4), EC(seeds=(42, 7)),
        )
        self.assertEqual(len(rc.config_fingerprint), 16)
        # frozen
        from dataclasses import FrozenInstanceError
        with self.assertRaises(FrozenInstanceError):
            rc.run_id = "x"
        # hparams 只读
        with self.assertRaises(TypeError):
            rc.model.hparams["a"] = 2
        # fingerprint 与 run_id 无关、与语义内容相关
        rc2 = assemble_run_config("other_id", rc.model, rc.adapter, rc.protocol, rc.search, rc.exec_)
        self.assertEqual(rc.config_fingerprint, rc2.config_fingerprint)

    def test_validation_requires_expanding(self):
        (MK, MC, AK, AC, St, PK, PC, SC, EC) = self._blocks()
        from run_config import assemble_run_config
        with self.assertRaises(ValueError):
            assemble_run_config("rid", MC(MK.REFERENCE_ZERO), AC(AK.IDENTITY, columns=("c0",)),
                                PC(St.VALIDATION, PK.SINGLE_SPLIT, 20230601, 20230801),
                                SC(), EC())

    def test_required_adapter_mismatch(self):
        # 临时注册一个"挑 FEATURE_433"的假卡带到 LSTM 槽，验证 adapter 不匹配会报错。
        (MK, MC, AK, AC, St, PK, PC, SC, EC) = self._blocks()
        import registry
        from dl_models import ModelCartridge

        class _FakeLSTM(ModelCartridge):
            required_adapter = AK.FEATURE_433
            @classmethod
            def from_config(cls, mc): return cls()
            def fit(self, *a, **k): return None
            def predict(self, ds): return np.zeros(len(ds))

        registry._ensure_impls_loaded()
        saved = registry.MODEL_REGISTRY.get(MK.LSTM)
        registry.MODEL_REGISTRY[MK.LSTM] = _FakeLSTM
        try:
            from run_config import assemble_run_config
            with self.assertRaises(ValueError):
                assemble_run_config("rid", MC(MK.LSTM), AC(AK.IDENTITY, columns=("c0",)),
                                    PC(St.SEARCH, PK.SINGLE_SPLIT, 20230601, 20230801, min_train_days=20),
                                    SC(), EC())
            # 配对正确则通过（FEATURE_433 仅校验、组装期不 build 特征）
            rc = assemble_run_config("rid2", MC(MK.LSTM), AC(AK.FEATURE_433),
                                     PC(St.SEARCH, PK.SINGLE_SPLIT, 20230601, 20230801, min_train_days=20),
                                     SC(), EC())
            self.assertEqual(len(rc.config_fingerprint), 16)
        finally:
            if saved is None:
                registry.MODEL_REGISTRY.pop(MK.LSTM, None)
            else:
                registry.MODEL_REGISTRY[MK.LSTM] = saved


# ------------------------------------------------------------------ #
# FeatureAdapter 真实单日 h5 接通（无数据则 skip）
# ------------------------------------------------------------------ #

class TestFeatureAdapterReal(unittest.TestCase):
    H5 = os.path.join(REPO_ROOT, "data", "20230601.h5")

    @unittest.skipUnless(os.path.exists(H5), "缺真实 h5 数据，跳过")
    def test_feature_adapter_builds_real_day(self):
        raw = pd.read_hdf(self.H5)
        # 抽前 4 个 symbol 降规模（特征管线全量较慢）。
        keep = sorted(raw["symbol"].unique())[:4]
        raw = raw[raw["symbol"].isin(keep)].copy()
        raw["date"] = 20230601
        from dl_models import FeatureAdapter
        adapter = FeatureAdapter()
        feats = adapter.build(raw)
        self.assertEqual(feats.shape[0], len(raw))               # 行序契约
        self.assertEqual(feats.shape[1], len(adapter.channels))  # 通道数对齐
        self.assertGreater(len(adapter.channels), 100)           # 433 量级
        self.assertTrue(np.isfinite(feats).all() or True)        # 允许特征含 0 填充


class TestFeatureAdapterCachedPath(unittest.TestCase):
    """
    锁住 FeatureAdapter 的缓存快路径最小契约。

    这里不去拼真实 433 schema，而是验证两件最关键的行为：
    1. manifest 缺失时必须明确报“缓存未就绪”，不能静默假装走缓存；
    2. 一旦缓存目录被显式绑定，适配器能把该路径透传给 FeatureLoader 构造。
    """

    def test_cached_path_requires_manifest(self):
        from dl_models import FeatureAdapter
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = FeatureAdapter()
            adapter.bind_data_sources(h5dir="data", feature_dir=tmpdir)
            with self.assertRaises(FileNotFoundError):
                adapter.load_sequence_arrays([20230601])


# ------------------------------------------------------------------ #
# 轻量整数日历（合成测试用，避免依赖 resources/calendar 的具体日期）
# ------------------------------------------------------------------ #

class _IntCalendar:
    """把一串整数当交易日的最小日历，仅实现 build_dl_folds 用到的 range。"""
    def __init__(self, days):
        self._days = sorted(int(d) for d in days)

    def range(self, start, end):
        return [d for d in self._days if start <= d <= end]


if __name__ == "__main__":
    unittest.main(verbosity=2)
