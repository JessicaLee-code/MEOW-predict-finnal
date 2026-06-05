# -*- coding: utf-8 -*-
"""
截面改造 + 损失对齐 单元测试（WT-B）

分两块：
- **Phase 1（损失/重标定）**：可微 Pearson 对拍 numpy、masked 截面 Pearson、组合损失
  ``MSE+λ·(1−corr)``、OLS rescale 的 ``R²=corr²`` 恒等、GRU 卡带接 λ + rescale 端到端 smoke。
- **Phase 2/3（截面数据/卡带）**：聚票/pad/mask、置换等变、残差零初退化、截面卡带 smoke。

torch 相关用例用 ``skipUnless`` 守卫；纯 numpy 口径（pearson_numpy / rescale / 聚票索引）
无 torch 也必须过。
"""

import os
import sys
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _sub in ("src", "config", "models"):
    sys.path.insert(0, os.path.join(REPO_ROOT, _sub))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import dl_losses  # noqa: E402
from dl_losses import (  # noqa: E402
    fit_linear_rescale_numpy, make_loss, pearson_numpy,
)

try:
    import torch  # noqa: E402
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

_skip_no_torch = unittest.skipUnless(_HAS_TORCH, "无 torch，跳过可微损失用例")


# ------------------------------------------------------------------ #
# 合成工具
# ------------------------------------------------------------------ #

def masked_xs_pearson_ref(pred, target, mask):
    """masked 截面 Pearson 的 numpy 金标准：逐行取有效位算 Pearson、≥2 才计入、求平均。"""
    corrs = []
    for i in range(pred.shape[0]):
        sel = mask[i].astype(bool)
        if int(sel.sum()) < 2:
            continue
        corrs.append(pearson_numpy(pred[i][sel], target[i][sel]))
    return float(np.mean(corrs)) if corrs else 0.0


def make_synth(n_days=8, n_symbols=5, n_interval=30, label="current", seed=0):
    """造合成 raw（口径同 test_dl_pipeline.make_synth）：c0/c1 噪声通道，y 按 label 决定。"""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(1, n_days + 1):
        for s in range(n_symbols):
            c0 = rng.normal(size=n_interval)
            c1 = rng.normal(size=n_interval)
            if label == "current":
                y = c0 + 0.1 * rng.normal(size=n_interval)
            elif label == "noise":
                y = rng.normal(size=n_interval)
            else:
                raise ValueError(label)
            for it in range(n_interval):
                rows.append((d, s, it, float(c0[it]), float(c1[it]), float(y[it])))
    return pd.DataFrame(rows, columns=["date", "symbol", "interval", "c0", "c1", "fret12"])


# ================================================================== #
# Phase 1 · numpy 口径
# ================================================================== #

class TestPearsonNumpy(unittest.TestCase):
    def test_known_values(self):
        a = np.array([1, 2, 3, 4, 5.0])
        self.assertAlmostEqual(pearson_numpy(a, 2 * a + 7), 1.0, places=6)
        self.assertAlmostEqual(pearson_numpy(a, -3 * a + 1), -1.0, places=6)
        self.assertEqual(pearson_numpy(a, np.ones(5)), 0.0)   # 常量 → 0
        self.assertEqual(pearson_numpy([1.0], [2.0]), 0.0)     # 样本<2 → 0

    def test_matches_corrcoef(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            a = rng.normal(size=50)
            b = a * rng.normal() + rng.normal(size=50)
            self.assertAlmostEqual(pearson_numpy(a, b), float(np.corrcoef(a, b)[0, 1]), places=6)


class TestLinearRescale(unittest.TestCase):
    def test_recovers_affine(self):
        rng = np.random.default_rng(5)
        raw = rng.normal(size=2000)
        y = 2.0 * raw + 3.0 + 0.01 * rng.normal(size=2000)
        a, b = fit_linear_rescale_numpy(raw, y)
        self.assertAlmostEqual(a, 2.0, places=1)
        self.assertAlmostEqual(b, 3.0, places=1)

    def test_constant_raw_degrades_to_mean(self):
        raw = np.full(100, 0.7)
        y = np.linspace(-1, 1, 100)
        a, b = fit_linear_rescale_numpy(raw, y)
        self.assertEqual(a, 0.0)
        self.assertAlmostEqual(b, float(y.mean()), places=6)

    def test_r2_equals_corr_squared_after_rescale(self):
        """§定稿核心保证：OLS 重标定后训练段上 R² == corr²（故 corr>0 ⇒ R²≥0）。"""
        rng = np.random.default_rng(9)
        raw = rng.normal(size=500)
        y = 0.4 * raw + rng.normal(size=500)        # corr 适中、量纲与 raw 不一致
        a, b = fit_linear_rescale_numpy(raw, y)
        pred = a * raw + b
        corr = pearson_numpy(raw, y)
        n = len(y)
        r2 = 1.0 - np.sum((pred - y) ** 2) / (np.var(y) * n)
        self.assertAlmostEqual(r2, corr ** 2, places=4)
        self.assertGreaterEqual(r2, -1e-9)          # 硬底：R² 不为负


# ================================================================== #
# Phase 1 · 可微损失（torch）
# ================================================================== #

@_skip_no_torch
class TestPearsonTorch(unittest.TestCase):
    def test_matches_numpy(self):
        rng = np.random.default_rng(11)
        for _ in range(15):
            a = rng.normal(size=64)
            b = a * 0.5 + rng.normal(size=64)
            ta = torch.tensor(a, dtype=torch.float64)
            tb = torch.tensor(b, dtype=torch.float64)
            got = float(dl_losses.pearson_torch(ta, tb))
            self.assertAlmostEqual(got, pearson_numpy(a, b), places=5)

    def test_differentiable(self):
        a = torch.randn(32, dtype=torch.float64, requires_grad=True)
        b = torch.randn(32, dtype=torch.float64)
        loss = 1.0 - dl_losses.pearson_torch(a, b)
        loss.backward()
        self.assertIsNotNone(a.grad)
        self.assertTrue(torch.isfinite(a.grad).all())

    def test_constant_pred_zero_finite_grad(self):
        a = torch.zeros(16, dtype=torch.float64, requires_grad=True)  # 常量预测
        b = torch.randn(16, dtype=torch.float64)
        corr = dl_losses.pearson_torch(a, b)
        self.assertAlmostEqual(float(corr), 0.0, places=6)
        corr.backward()
        self.assertTrue(torch.isfinite(a.grad).all())     # 零方差不产生 NaN 梯度


@_skip_no_torch
class TestMaskedCrossSectionPearson(unittest.TestCase):
    def test_matches_numpy_reference(self):
        rng = np.random.default_rng(13)
        B, N = 4, 10
        pred = rng.normal(size=(B, N))
        target = rng.normal(size=(B, N))
        mask = (rng.random((B, N)) > 0.3)
        mask[:, 0] = True                                  # 保证每行至少有些有效位
        got = float(dl_losses.masked_cross_section_pearson_torch(
            torch.tensor(pred), torch.tensor(target), torch.tensor(mask)))
        self.assertAlmostEqual(got, masked_xs_pearson_ref(pred, target, mask), places=5)

    def test_row_with_too_few_excluded(self):
        # 第 0 行只有 1 个有效位 → 不计入；其余行正常。
        pred = torch.tensor([[1.0, 2, 3], [1.0, 2, 3]])
        target = torch.tensor([[5.0, 9, 9], [1.0, 2, 3]])
        mask = torch.tensor([[True, False, False], [True, True, True]])
        got = float(dl_losses.masked_cross_section_pearson_torch(pred, target, mask))
        self.assertAlmostEqual(got, 1.0, places=5)         # 仅第 1 行（完美相关）


@_skip_no_torch
class TestMakeLoss(unittest.TestCase):
    def test_lambda_zero_is_pure_mse(self):
        pred = torch.randn(40, dtype=torch.float64)
        target = torch.randn(40, dtype=torch.float64)
        loss = make_loss(lambda_corr=0.0)(pred, target)
        mse = ((pred - target) ** 2).mean()
        self.assertAlmostEqual(float(loss), float(mse), places=8)

    def test_lambda_combines_mse_and_corr_1d(self):
        pred = torch.randn(40, dtype=torch.float64)
        target = torch.randn(40, dtype=torch.float64)
        lam = 0.3
        loss = make_loss(lambda_corr=lam)(pred, target)
        mse = ((pred - target) ** 2).mean()
        corr = dl_losses.pearson_torch(pred, target)
        self.assertAlmostEqual(float(loss), float(mse + lam * (1.0 - corr)), places=6)

    def test_lambda_combines_masked(self):
        B, N = 3, 8
        pred = torch.randn(B, N, dtype=torch.float64)
        target = torch.randn(B, N, dtype=torch.float64)
        mask = torch.ones(B, N, dtype=torch.bool)
        lam = 0.3
        loss = make_loss(lambda_corr=lam)(pred, target, mask=mask)
        m = mask.to(pred.dtype)
        mse = ((pred - target) ** 2 * m).sum() / m.sum()
        corr = dl_losses.masked_cross_section_pearson_torch(pred, target, mask)
        self.assertAlmostEqual(float(loss), float(mse + lam * (1.0 - corr)), places=6)


# ================================================================== #
# Phase 1 · GRU 卡带接 λ + rescale 端到端 smoke（CPU，小数据）
# ================================================================== #

@_skip_no_torch
class TestGruLambdaRescaleSmoke(unittest.TestCase):
    def _run(self, lam):
        from dl_trainer import SequenceTrainer
        from dl_models import GRUCartridge, FeatureAdapter  # noqa: F401
        from dl_models import IdentityAdapter
        from dl_protocol import DLFold
        df = make_synth(n_days=10, n_symbols=8, n_interval=20, label="current", seed=7)
        loader = lambda dates: df[df["date"].isin(list(dates))]
        hp = {"hidden_size": 16, "num_layers": 1, "max_epochs": 3, "patience": 3,
              "batch_size": 256, "eval_batch_size": 512, "device": "cpu",
              "lambda_corr": lam}
        tr = SequenceTrainer({"experiment_id": f"gru_lam{lam}"}, IdentityAdapter(["c0", "c1"]),
                             GRUCartridge, loader, seq_len=5, normalizer_mode="zscore", hparams=hp)
        fold = DLFold(0, tuple(range(1, 7)), (7, 8), (), (9, 10))
        return tr.run_on_dl_fold(fold)

    def test_lambda0_runs_and_rescale_makes_r2_nonneg_on_train(self):
        r = self._run(0.0)
        self.assertEqual(r.status, "ok", r.error_msg)
        # rescale 在训练段上保证 R²==corr²（OLS 恒等），故 train_r2 ≈ train_corr² 且 ≥0。
        self.assertAlmostEqual(r.train_r2, r.train_corr ** 2, places=3)
        self.assertGreaterEqual(r.train_r2, -1e-6)

    def test_lambda03_runs(self):
        r = self._run(0.3)
        self.assertEqual(r.status, "ok", r.error_msg)
        self.assertAlmostEqual(r.train_r2, r.train_corr ** 2, places=3)


# ================================================================== #
# Phase 2 · 截面数据集（聚票 / pad / mask）— torch-free
# ================================================================== #

class TestCrossSectionDataset(unittest.TestCase):
    def _arrays(self, n_days=3, n_symbols=4, n_interval=6):
        df = make_synth(n_days=n_days, n_symbols=n_symbols, n_interval=n_interval, label="current", seed=1)
        from dl_models import IdentityAdapter
        from sequence_dataset import build_sequence_arrays
        return build_sequence_arrays(df, IdentityAdapter(["c0", "c1"]))

    def test_grouping_and_flat_matches_windows(self):
        from sequence_dataset import CrossSectionDataset, SequenceDataset, Normalizer
        arr = self._arrays()
        L = 3
        xs = CrossSectionDataset(arr, L, Normalizer("identity"))
        seq = SequenceDataset(arr, L, Normalizer("identity"))
        # 票总数守恒 + flat 升序 == 逐票窗口 label_rows（映射无损的前提）
        self.assertEqual(xs.n_tickers_total(), len(seq))
        self.assertTrue(np.array_equal(np.sort(xs.flat_label_rows()), seq.label_rows))
        # 每快照 (date,interval) 唯一
        lf = xs.label_frame()
        self.assertEqual(lf.groupby(["date", "interval"]).ngroups, len(xs))

    def test_pad_and_mask(self):
        from sequence_dataset import CrossSectionDataset, Normalizer
        arr = self._arrays(n_symbols=4)
        xs = CrossSectionDataset(arr, 3, Normalizer("identity"))
        for X, y, mask in xs.iter_batches(batch_size=5):
            B, maxN, L, C = X.shape
            self.assertEqual((L, C), (3, 2))
            self.assertEqual(mask.shape, (B, maxN))
            self.assertEqual(y.shape, (B, maxN))
            # 本合成每截面恰 4 票（无 pad）
            self.assertTrue((mask.sum(axis=1) == 4).all())
            break

    def test_from_whitened_no_recompute(self):
        from sequence_dataset import CrossSectionDataset, SequenceDataset, Normalizer
        arr = self._arrays()
        seq = SequenceDataset(arr, 3, Normalizer("zscore").fit(arr.features))
        xs = CrossSectionDataset.from_whitened(seq.feature_matrix(), seq.arrays, seq.seq_len)
        # 共享同一张白化矩阵（零复制）
        self.assertIs(xs.feature_matrix(), seq.feature_matrix())
        self.assertEqual(xs.n_tickers_total(), len(seq))


# ================================================================== #
# Phase 3 · 截面模型结构（置换等变 / 零初残差）— torch
# ================================================================== #

@_skip_no_torch
class TestXSectionModule(unittest.TestCase):
    def _model(self, C=2, d=8, heads=2, gamma=None):
        from dl_models import _build_xsection_module
        torch.manual_seed(0)
        m = _build_xsection_module(input_channels=C, hidden_size=d, num_layers=1,
                                   dropout=0.0, n_heads=heads, attn_dropout=0.0)
        m.eval()
        if gamma is not None:
            with torch.no_grad():
                m.gamma.fill_(float(gamma))
        return m

    def test_permutation_equivariant(self):
        # γ≠0 让截面腿真正参与；打乱在场票顺序，输出应随之同序置换、值不变。
        m = self._model(gamma=1.5)
        torch.manual_seed(3)
        N, L, C = 5, 4, 2
        x = torch.randn(1, N, L, C)
        mask = torch.ones(1, N, dtype=torch.bool)
        with torch.no_grad():
            out = m(x, mask)[0]                      # [N]
            perm = torch.tensor([2, 0, 4, 1, 3])
            out_p = m(x[:, perm], mask)[0]           # 打乱票顺序
        self.assertTrue(torch.allclose(out_p, out[perm], atol=1e-5))

    def test_zero_init_residual_decouples(self):
        # γ=0（初始）：某票输出与"其它票特征"无关（纯时序腿）；γ≠0 后变得相关。
        m0 = self._model(gamma=0.0)
        torch.manual_seed(4)
        N, L, C = 4, 3, 2
        x = torch.randn(1, N, L, C)
        mask = torch.ones(1, N, dtype=torch.bool)
        x2 = x.clone()
        x2[0, 1] = torch.randn(L, C)                 # 只改第 1 只票
        with torch.no_grad():
            o0 = m0(x, mask)[0, 0]                    # 第 0 票输出
            o0b = m0(x2, mask)[0, 0]
        self.assertAlmostEqual(float(o0), float(o0b), places=5)   # γ=0 解耦
        # γ≠0：同样改第 1 票，第 0 票输出应变化（截面耦合生效）
        with torch.no_grad():
            m0.gamma.fill_(2.0)
            o1 = m0(x, mask)[0, 0]
            o1b = m0(x2, mask)[0, 0]
        self.assertGreater(abs(float(o1) - float(o1b)), 1e-6)

    def test_n_heads_divides_hidden(self):
        from dl_models import _safe_n_heads
        self.assertEqual(_safe_n_heads(32, 4), 4)
        self.assertEqual(_safe_n_heads(32, 3), 2)
        self.assertEqual(_safe_n_heads(30, 4), 3)
        self.assertEqual(_safe_n_heads(7, 4), 1)


@_skip_no_torch
class TestXSectionCrossZ(unittest.TestCase):
    """截面内 cross-z 输入归一（线 B：把传统主力那维 cross-z 显式喂进模型）。"""

    def _model(self, cross_z, C=2, d=8, heads=2, gamma=1.0):
        from dl_models import _build_xsection_module
        torch.manual_seed(0)
        m = _build_xsection_module(input_channels=C, hidden_size=d, num_layers=1,
                                   dropout=0.0, n_heads=heads, attn_dropout=0.0, cross_z=cross_z)
        m.eval()
        with torch.no_grad():
            m.gamma.fill_(float(gamma))
        return m

    def test_cross_z_removes_cross_sectional_level(self):
        # cross_z=on：给某 (l,c) 上所有在场票同加常数 → 截面去均值后不变 → 输出不变。
        m = self._model(cross_z=True)
        torch.manual_seed(3)
        N, L, C = 6, 4, 2
        x = torch.randn(1, N, L, C)
        mask = torch.ones(1, N, dtype=torch.bool)
        x_shift = x.clone()
        x_shift[0, :, 2, 1] += 3.7                     # 同一 (l=2,c=1) 全票平移
        with torch.no_grad():
            o, o_shift = m(x, mask), m(x_shift, mask)
        self.assertTrue(torch.allclose(o, o_shift, atol=1e-4))

    def test_cross_z_off_is_sensitive_to_level(self):
        # 对照：cross_z=off 时同样的平移会改变输出（证明不变性确由 cross-z 产生）。
        m = self._model(cross_z=False)
        torch.manual_seed(3)
        N, L, C = 6, 4, 2
        x = torch.randn(1, N, L, C)
        mask = torch.ones(1, N, dtype=torch.bool)
        x_shift = x.clone()
        x_shift[0, :, 2, 1] += 3.7
        with torch.no_grad():
            o, o_shift = m(x, mask), m(x_shift, mask)
        self.assertGreater(float((o - o_shift).abs().max()), 1e-4)

    def test_cross_z_mask_ignores_pad(self):
        # 加一只 mask=False 的 pad 票，不应改变在场票的截面统计 → 在场票输出不变。
        m = self._model(cross_z=True)
        torch.manual_seed(5)
        N, L, C = 5, 3, 2
        x = torch.randn(1, N, L, C)
        mask = torch.ones(1, N, dtype=torch.bool)
        x_pad = torch.cat([x, torch.randn(1, 1, L, C)], dim=1)        # 末尾加一只票
        mask_pad = torch.cat([mask, torch.zeros(1, 1, dtype=torch.bool)], dim=1)  # 该票 mask=False
        with torch.no_grad():
            o = m(x, mask)[0]                            # [N]
            o_pad = m(x_pad, mask_pad)[0, :N]            # 前 N 只在场票
        self.assertTrue(torch.allclose(o, o_pad, atol=1e-4))

    def test_cross_z_permutation_equivariant(self):
        # cross-z 是对称聚合 + 逐票同一变换 → 置换等变保持。
        m = self._model(cross_z=True, gamma=1.5)
        torch.manual_seed(7)
        N, L, C = 5, 4, 2
        x = torch.randn(1, N, L, C)
        mask = torch.ones(1, N, dtype=torch.bool)
        perm = torch.tensor([2, 0, 4, 1, 3])
        with torch.no_grad():
            out = m(x, mask)[0]
            out_p = m(x[:, perm], mask)[0]
        self.assertTrue(torch.allclose(out_p, out[perm], atol=1e-5))


# ================================================================== #
# Phase 3 · 截面卡带端到端 smoke + registry
# ================================================================== #

class TestXSectionRegistry(unittest.TestCase):
    def test_required_adapter_is_feature_433(self):
        import registry
        from model_config import ModelKind
        from adapter_config import AdapterKind
        self.assertEqual(registry.required_adapter_for(ModelKind.XSECTION), AdapterKind.FEATURE_433)


@_skip_no_torch
class TestXSectionCartridgeSmoke(unittest.TestCase):
    def _run(self, lam):
        from dl_trainer import SequenceTrainer
        from dl_models import CrossSectionCartridge, IdentityAdapter
        from dl_protocol import DLFold
        df = make_synth(n_days=10, n_symbols=8, n_interval=20, label="current", seed=7)
        loader = lambda dates: df[df["date"].isin(list(dates))]
        hp = {"hidden_size": 16, "num_layers": 1, "n_heads": 2, "max_epochs": 3,
              "patience": 3, "snap_batch": 4, "eval_snap_batch": 4, "device": "cpu",
              "lambda_corr": lam}
        tr = SequenceTrainer({"experiment_id": f"xs_lam{lam}"}, IdentityAdapter(["c0", "c1"]),
                             CrossSectionCartridge, loader, seq_len=5,
                             normalizer_mode="zscore", hparams=hp)
        fold = DLFold(0, tuple(range(1, 7)), (7, 8), (), (9, 10))
        return tr.run_on_dl_fold(fold)

    def test_lambda03_end_to_end_aligned(self):
        r = self._run(0.3)
        self.assertEqual(r.status, "ok", r.error_msg)
        # rescale 在训练段保证 R²==corr²（预测长度若与 label_frame 不齐，指标会炸/status≠ok）
        self.assertAlmostEqual(r.train_r2, r.train_corr ** 2, places=2)
        self.assertGreaterEqual(r.train_r2, -1e-6)

    def test_lambda0_pure_mse_runs(self):
        r = self._run(0.0)
        self.assertEqual(r.status, "ok", r.error_msg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
