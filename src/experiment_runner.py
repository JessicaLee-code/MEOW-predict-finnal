import argparse
import os
import json
import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dl import MeowDataLoader
from feature_loader import FeatureLoader
from feature_store import DEFAULT_FEATURE_DIR
from log import log
from tradingcalendar import Calendar

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

try:
    from lightgbm import LGBMRegressor
except ImportError:  # LightGBM is optional in this workspace.
    LGBMRegressor = None


EPS = 1e-8
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
DEFAULT_TARGET_WINSORIZE = {
    "enabled": True,
    "lower_quantile": 0.01,
    "upper_quantile": 0.99,
}
DEFAULT_RIDGE_ALPHA = 2.0


@dataclass(frozen=True)
class SplitConfig:
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int


@dataclass(frozen=True)
class RollingFold:
    fold_id: int
    train_dates: tuple
    val_dates: tuple


class ExperimentRunner(object):
    def __init__(
        self,
        h5dir,
        feature_dir=DEFAULT_FEATURE_DIR,
        loader_cls=MeowDataLoader,
        feature_loader=None,
        feature_loader_cls=FeatureLoader,
        target_winsorize_config=None,
        feature_dtype="float32",
        ridge_alpha=DEFAULT_RIDGE_ALPHA,
        train_subsample_frac=None,
        train_subsample_seed=42,
    ):
        """
        实验执行器。

        当前阶段只保留一条正式数据链路：
        - 训练/评估逻辑继续放在 runner 内部；
        - 特征加载统一走 FeatureLoader，不再回退旧版 FeatureBuilder/cache。
        """
        self.calendar = Calendar()
        self.h5dir = h5dir
        self.feature_dir = feature_dir
        # #16：默认特征以 float32 进入训练；保留显式切回 float64 的入口，
        # 供数值一致性验收和调试使用。
        self.feature_dtype = feature_dtype
        # #18：把标准 ridge 主路径的 alpha 收口成 runner 级参数，
        # 便于 P0.5 扫描 / 锁定，而不是继续靠手改硬编码。
        self.ridge_alpha = self._normalize_ridge_alpha(ridge_alpha)
        # 仅保留底层原始数据 loader 引用，供上层读取 h5dir 等环境信息；
        # 正式特征链路仍统一走 FeatureLoader。
        self.loader = loader_cls(h5dir=h5dir)
        self.feature_loader = feature_loader or feature_loader_cls(
            h5dir=h5dir,
            feature_dir=feature_dir,
            loader_cls=loader_cls,
            feature_dtype=feature_dtype,
        )
        # 训练标签 winsorize 作为统一评测口径挂在 runner 级别，
        # 这样主进程、串行评测、并行 worker 都会走同一份配置，
        # 避免“命令行开了，但子进程没生效”的口径漂移。
        self.target_winsorize_config = self._normalize_target_winsorize_config(
            target_winsorize_config
        )
        # P4 选模型提速：仅对“训练行”做降采样（验证集全量不动，保证排名指标不被污染）。
        # 树模型是递归搜索、每行被重扫 depth×n_estimators 遍，砍训练行直接线性减算力 + 减内存；
        # 线性是闭式解、行数近乎免费，故默认 1.0（关闭）不影响任何既有路径。
        # 同 winsorize 一样挂 runner 级，主进程 / 串行 / 并行 worker 走同一份配置，避免口径漂移。
        self.train_subsample_frac = self._normalize_train_subsample_frac(train_subsample_frac)
        self.train_subsample_seed = int(train_subsample_seed)

    def _normalize_train_subsample_frac(self, frac):
        """None / 1.0 / 非正 → 视为关闭（返回 None）；否则裁到 (0, 1]。"""
        if frac is None:
            return None
        frac = float(frac)
        if frac <= 0 or frac >= 1.0:
            return None
        return frac

    def get_train_subsample_frac(self):
        """返回训练行降采样比例（None=关闭），供调度层透传给并行 worker。"""
        return self.train_subsample_frac

    # 仅对“算力贵”的树族模型降采样；线性模型（闭式解、秒级）始终全量，不付保真代价。
    # smoke 实证：同折 0.33 vs 全量，树 corr 仅 -0.0004，而 ridge 达 -0.002 → 线性不值得采样。
    _SUBSAMPLE_MODELS = frozenset({"tree", "tree_big", "tree_shallow", "histgb", "histgb_shallow", "gbdt", "lgbm"})

    def _subsample_train_rows(self, xtrain, ytrain, model_name=None):
        """
        对训练特征/标签按相同行位做无放回降采样（带固定种子，可复现）。
        - 只采训练，调用方不得对验证集调用本函数；
        - 仅树族模型生效，线性模型直接全量返回（见 _SUBSAMPLE_MODELS）；
        - 用全块均匀随机采样：每个交易日按比例变薄但全部保留，regime 覆盖不丢；
        - 采样后保持原行序（排序行位），不依赖下游对顺序的假设。
        """
        frac = self.train_subsample_frac
        if frac is None:
            return xtrain, ytrain
        if model_name is not None and model_name not in self._SUBSAMPLE_MODELS:
            return xtrain, ytrain
        n = len(xtrain)
        k = int(n * frac)
        if k <= 0 or k >= n:
            return xtrain, ytrain
        rng = np.random.default_rng(self.train_subsample_seed)
        idx = np.sort(rng.choice(n, size=k, replace=False))
        log.inf(f"Train subsample: frac={frac:.3f} rows {n} -> {k}（仅训练，验证集不动）")
        return xtrain.iloc[idx], ytrain.iloc[idx]

    def _normalize_groups(self, groups):
        if groups is None:
            return None
        if isinstance(groups, str):
            groups = [groups]
        groups = [g.strip() for g in groups if g and g.strip()]
        if not groups or groups == ["full"]:
            return None
        return groups

    def _normalize_target_winsorize_config(self, config):
        """
        统一清洗训练标签 winsorize 配置。

        约束：
        - 允许完全关闭（enabled=False）
        - 打开时必须满足 0 <= lower < upper <= 1
        - 默认口径在 P0.5 扫描后锁为 P1 / P99
        """
        merged = dict(DEFAULT_TARGET_WINSORIZE)
        if config:
            merged.update(config)
        merged["enabled"] = bool(merged.get("enabled", True))
        merged["lower_quantile"] = float(merged.get("lower_quantile", 0.01))
        merged["upper_quantile"] = float(merged.get("upper_quantile", 0.99))
        lower_q = merged["lower_quantile"]
        upper_q = merged["upper_quantile"]
        if not (0.0 <= lower_q < upper_q <= 1.0):
            raise ValueError(
                "target_winsorize_config 非法：要求 0 <= lower_quantile < upper_quantile <= 1"
            )
        return merged

    def get_target_winsorize_config(self):
        """返回当前 runner 使用的 winsorize 配置副本，供调度层透传给 worker。"""
        return dict(self.target_winsorize_config)

    def _normalize_ridge_alpha(self, ridge_alpha):
        """
        统一校验标准 ridge 主路径的 alpha。

        约束很简单：
        - 必须能转成 float
        - 必须严格大于 0，避免把模型推进到非法或退化状态
        """
        value = float(ridge_alpha)
        if value <= 0.0:
            raise ValueError("ridge_alpha 必须 > 0")
        return value

    def get_ridge_alpha(self):
        """返回当前 runner 的标准 ridge alpha，供协议层写盘和 worker 透传。"""
        return float(self.ridge_alpha)

    def _regime_state(self, xdf):
        state_cols = [c for c in ["regime_low", "regime_mid", "regime_high"] if c in xdf.columns]
        if len(state_cols) != 3:
            raise ValueError("Regime state requires regime_low/regime_mid/regime_high columns")
        state_values = xdf[state_cols].to_numpy(dtype=np.float32)
        return np.argmax(state_values, axis=1)

    def _interval_baseline(self, ydf):
        baseline = (
            ydf.groupby("interval", sort=False)["fret12"]
            .mean()
            .rename("interval_mean")
            .reset_index()
        )
        return baseline

    def _attach_interval_baseline(self, ydf, baseline):
        out = ydf.merge(baseline, on="interval", how="left")
        out["interval_mean"] = out["interval_mean"].fillna(0.0)
        out["fret12_residual"] = out["fret12"] - out["interval_mean"]
        return out

    def _fit_common_component(self, ytrain):
        common_model = Pipeline([
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ("ridge", Ridge(alpha=10.0, fit_intercept=True, random_state=None)),
        ])
        common_x = ytrain[["interval"]].to_numpy()
        common_y = ytrain["fret12"].to_numpy(dtype=np.float32)
        common_model.fit(common_x, common_y)
        return common_model

    def _make_target_series(self, ydf, target_mode):
        if target_mode == "raw":
            return ydf["fret12"].to_numpy(dtype=np.float32), None
        if target_mode == "date_demean":
            series = (
                ydf.groupby("date")["fret12"]
                .transform(lambda s: s - s.mean())
                .to_numpy(dtype=np.float32)
            )
            return series, None
        if target_mode == "interval_demean":
            series = (
                ydf.groupby(["date", "interval"])["fret12"]
                .transform(lambda s: s - s.mean())
                .to_numpy(dtype=np.float32)
            )
            return series, None
        if target_mode == "interval_residual":
            baseline = self._interval_baseline(ydf)
            merged = self._attach_interval_baseline(ydf, baseline)
            return merged["fret12_residual"].to_numpy(dtype=np.float32), baseline
        raise ValueError(f"Unknown target mode: {target_mode}")

    def _apply_target_winsorize(self, target_array):
        """
        仅对训练目标做 winsorize。

        这里故意只接收已经构造好的训练目标数组，而不是直接改 `ytrain` DataFrame：
        - 保证测试 / 提交路径完全不受影响，仍然保留原始 `fret12`
        - 无论 target_mode 是 raw 还是 residual，都是“只改训练时喂给模型的 y”
        - 返回裁剪后的数组 + 裁剪边界，便于日志和真实跑数审计
        """
        cfg = self.target_winsorize_config
        values = np.asarray(target_array, dtype=np.float32)
        if not cfg.get("enabled", True) or values.size == 0:
            return values, None
        lower_q = cfg["lower_quantile"]
        upper_q = cfg["upper_quantile"]
        lower = float(np.quantile(values, lower_q))
        upper = float(np.quantile(values, upper_q))
        clipped = np.clip(values, lower, upper).astype(np.float32, copy=False)
        return clipped, {
            "lower_quantile": lower_q,
            "upper_quantile": upper_q,
            "lower_bound": lower,
            "upper_bound": upper,
        }

    def split_dates(self, split_config):
        train_dates = self.calendar.range(split_config.train_start, split_config.train_end)
        val_dates = self.calendar.range(split_config.val_start, split_config.val_end)
        test_dates = self.calendar.range(split_config.test_start, split_config.test_end)
        return train_dates, val_dates, test_dates

    def _build_sequence_features(self, raw_df, lags):
        raw_df = raw_df.copy()
        raw_df = raw_df.sort_values(["date", "symbol", "interval"], kind="mergesort").reset_index(drop=True)
        seq_cols = [
            "midpx",
            "bid0",
            "ask0",
            "bid4",
            "ask4",
            "bid9",
            "ask9",
            "bsize0",
            "asize0",
            "bsize0_4",
            "asize0_4",
            "bsize5_9",
            "asize5_9",
            "tradeBuyQty",
            "tradeSellQty",
            "tradeBuyTurnover",
            "tradeSellTurnover",
            "nAddBuy",
            "nAddSell",
            "nCxlBuy",
            "nCxlSell",
            "buyVwad",
            "sellVwad",
        ]
        out = raw_df[["date", "symbol", "interval", "fret12"]].copy()
        group = raw_df.groupby(["date", "symbol"], sort=False)
        for col in seq_cols:
            out[col] = raw_df[col].astype(np.float32)
            for lag in lags:
                out[f"{col}_seq_lag_{lag}"] = group[col].shift(lag).fillna(0.0).astype(np.float32)
        xdf = out.drop(columns=["fret12"]).copy()
        ydf = out[["date", "symbol", "interval", "fret12"]].copy()
        xdf = xdf.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        ydf = ydf.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return xdf, ydf

    def _load_group_split(self, dates, groups=None, max_days=None, loader=None):
        """
        统一的数据加载入口。

        优先级：
        1. 调用方显式传入的 loader（供 scheduler / trainer 注入）
        2. runner 自身持有的 FeatureLoader

        旧版 `feat_engine -> load_feature_split` 链路已经归档；
        如果这里拿不到 loader，说明调用方绕开了当前正式入口，应立即失败。
        """
        if max_days is not None:
            dates = dates[:max_days]
        active_loader = loader or self.feature_loader
        if active_loader is None:
            raise RuntimeError("FeatureLoader is required: 旧版 feat_engine 加载链已归档，不再允许回退。")
        return active_loader.load(dates, groups=groups)

    def evaluate_predictions(self, ydf, pred):
        y = ydf["fret12"].to_numpy()
        p = np.asarray(pred)
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        mse = mean_squared_error(y, p)
        corr = np.corrcoef(p, y)[0, 1] if len(y) > 1 else 0.0
        r2 = 1.0 - np.sum((p - y) ** 2) / (np.var(y) * len(y) + EPS)
        return {"mse": float(mse), "corr": float(corr), "r2": float(r2)}

    def evaluate_prediction_bundle(self, ydf, pred):
        metrics = self.evaluate_predictions(ydf, pred)
        daily_corrs = []
        tmp = ydf[["date", "fret12"]].copy()
        tmp["pred"] = np.asarray(pred, dtype=np.float32)
        for _, group in tmp.groupby("date", sort=True):
            if group.shape[0] < 2:
                continue
            corr = np.corrcoef(group["pred"].to_numpy(), group["fret12"].to_numpy())[0, 1]
            if np.isfinite(corr):
                daily_corrs.append(float(corr))
        metrics["daily_corr_mean"] = float(np.mean(daily_corrs)) if daily_corrs else 0.0
        metrics["daily_corr_std"] = float(np.std(daily_corrs)) if daily_corrs else 0.0
        metrics["n_days"] = int(tmp["date"].nunique())
        return metrics

    def _corr_gap(self, train_metrics, val_metrics):
        return float(train_metrics["corr"] - val_metrics["corr"])

    def _safe_rank_pct(self, series):
        return series.rank(pct=True, method="average")

    def _build_common_features(self, xdf):
        cols = [
            "spread",
            "obi0",
            "obi4",
            "trade_imb",
            "trade_turnover_imb",
            "cxl_imb",
            "trade_activity",
            "order_pressure",
            "mid_ret1_raw",
            "regime_score",
            "state_vol_cs",
            "state_spread_cs",
            "state_activity_cs",
            "interval_pos",
            "interval_norm",
            "is_morning",
            "is_afternoon",
        ]
        cols = [c for c in cols if c in xdf.columns]
        agg = xdf[["date", "interval"] + cols].groupby(["date", "interval"], sort=False)[cols].mean().reset_index()
        return agg

    def _extract_linear_coefficients(self, model, feature_cols):
        if isinstance(model, Pipeline):
            inner = model.named_steps.get("model")
            if hasattr(inner, "coef_"):
                coef = np.asarray(inner.coef_, dtype=np.float32)
            else:
                return None
        elif hasattr(model, "coef_"):
            coef = np.asarray(model.coef_, dtype=np.float32)
        else:
            return None
        if coef.ndim > 1:
            coef = coef.ravel()
        if len(coef) != len(feature_cols):
            return None
        return pd.DataFrame({"feature": list(feature_cols), "coef": coef, "abs_coef": np.abs(coef)})

    def _extract_tree_importance(self, model, feature_cols):
        # 树族特征重要性提取（镜像 _extract_linear_coefficients），供 P4-2 反偏颇/重要性扫描。
        # 输出列对齐线性版的 coef/abs_coef，方便下游统一消费：coef=importance、abs_coef=importance。
        inner = model.named_steps.get("model") if isinstance(model, Pipeline) else model
        importance = getattr(inner, "feature_importances_", None)
        if importance is None:
            return None
        importance = np.asarray(importance, dtype=np.float32).ravel()
        if len(importance) != len(feature_cols):
            return None
        return pd.DataFrame({"feature": list(feature_cols), "coef": importance, "abs_coef": importance})

    def _fold_metric_row(self, fold_id, experiment_id, feature_set, target_type, model_type, postprocess_type, train_metrics, val_metrics, runtime_sec, notes, random_seed=42):
        return {
            "fold_id": fold_id,
            "experiment_id": experiment_id,
            "feature_set": feature_set,
            "target_type": target_type,
            "model_type": model_type,
            "postprocess_type": postprocess_type,
            "train_corr": train_metrics["corr"],
            "val_corr": val_metrics["corr"],
            "train_mse": train_metrics["mse"],
            "val_mse": val_metrics["mse"],
            "train_r2": train_metrics["r2"],
            "val_r2": val_metrics["r2"],
            "daily_corr_mean": val_metrics["daily_corr_mean"],
            "daily_corr_std": val_metrics["daily_corr_std"],
            "rolling_corr_mean": np.nan,
            "rolling_corr_std": np.nan,
            "rolling_corr_min": np.nan,
            "rolling_mse_mean": np.nan,
            "rolling_r2_mean": np.nan,
            "stability_score": np.nan,
            "train_val_corr_gap": self._corr_gap(train_metrics, val_metrics),
            "runtime_sec": float(runtime_sec),
            "random_seed": random_seed,
            "notes": notes,
        }

    def _make_common_targets(self, ydf):
        return (
            ydf.groupby(["date", "interval"], sort=False)["fret12"]
            .mean()
            .rename("fret12_common")
            .reset_index()
        )

    def _fit_common_model(self, xtrain, ytrain, model_name="ridge"):
        common_x = self._build_common_features(xtrain)
        common_y = self._make_common_targets(ytrain)
        common_df = common_x.merge(common_y, on=["date", "interval"], how="inner")
        common_feature_cols = [c for c in common_df.columns if c not in ["date", "interval", "fret12_common"]]
        if model_name == "hgb":
            model = HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.05,
                max_iter=200,
                max_depth=6,
                min_samples_leaf=20,
                l2_regularization=0.1,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42,
            )
            model.fit(common_df[common_feature_cols].to_numpy(dtype=np.float32), common_df["fret12_common"].to_numpy(dtype=np.float32))
        else:
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=self.ridge_alpha, fit_intercept=True, random_state=None)),
            ])
            model.fit(
                common_df[common_feature_cols].to_numpy(dtype=np.float32),
                common_df["fret12_common"].to_numpy(dtype=np.float32),
            )
        return model, common_feature_cols

    def _predict_common_component(self, model, feature_cols, xdf):
        common_x = self._build_common_features(xdf)
        common_x["pred_common"] = model.predict(common_x[feature_cols].to_numpy(dtype=np.float32))
        merged = xdf[["date", "interval"]].merge(
            common_x[["date", "interval", "pred_common"]],
            on=["date", "interval"],
            how="left",
        )
        return merged["pred_common"].fillna(0.0).to_numpy(dtype=np.float32)

    def make_rolling_folds(self, start_date, end_date, train_window=40, val_window=10, step=10, min_train_days=30, embargo=0):
        all_dates = self.calendar.range(start_date, end_date)
        folds = []
        fold_id = 0
        if not all_dates:
            return folds
        embargo = max(0, int(embargo))
        cursor = train_window + embargo
        while cursor + val_window <= len(all_dates):
            train_end = cursor - embargo
            train_dates = tuple(all_dates[max(0, train_end - train_window):train_end])
            val_dates = tuple(all_dates[cursor: cursor + val_window])
            if len(train_dates) >= min_train_days and len(val_dates) > 0:
                folds.append(RollingFold(fold_id=fold_id, train_dates=train_dates, val_dates=val_dates))
                fold_id += 1
            cursor += step
        if not folds and len(all_dates) >= 3:
            fallback_val = max(1, min(2, len(all_dates) // 3))
            folds.append(
                RollingFold(
                    fold_id=0,
                    train_dates=tuple(all_dates[:-fallback_val]),
                    val_dates=tuple(all_dates[-fallback_val:]),
                )
            )
        return folds

    def _normalize_by_date_interval_rank(self, frame, pred_col):
        out = frame.copy()
        rank = out.groupby(["date", "interval"], sort=False)[pred_col].transform(self._safe_rank_pct)
        centered = rank - 0.5
        scale = float(out[pred_col].std()) if out.shape[0] > 1 else 0.0
        out["pred_rank_scaled"] = centered.fillna(0.0) * scale * 2.0
        out["pred_group_demean"] = out[pred_col] - out.groupby(["date", "interval"], sort=False)[pred_col].transform("mean")
        return out

    def choose_postprocess_params(self, ydf, pred):
        frame = ydf[["date", "interval", "fret12"]].copy()
        frame["pred"] = np.asarray(pred, dtype=np.float32)
        candidates = []
        for q in [0.001, 0.005, 0.01]:
            candidates.append({"name": f"clip_{q:.3f}", "clip_q": q, "blend_alpha": 1.0, "blend_kind": "none"})
            for alpha in [0.7, 0.8, 0.9]:
                candidates.append({"name": f"clip_rank_{q:.3f}_{alpha:.1f}", "clip_q": q, "blend_alpha": alpha, "blend_kind": "rank"})
                candidates.append({"name": f"clip_neutral_{q:.3f}_{alpha:.1f}", "clip_q": q, "blend_alpha": alpha, "blend_kind": "neutral"})
        best = None
        best_metrics = None
        for params in candidates:
            post = self.apply_postprocess(frame, params)
            metrics = self.evaluate_prediction_bundle(frame[["date", "fret12"]], post)
            score = (metrics["corr"], -metrics["mse"], metrics["r2"])
            if best is None or score > best:
                best = score
                best_metrics = metrics
                best_params = params
        return best_params, best_metrics

    def apply_postprocess(self, frame, params):
        out = frame.copy()
        q = params.get("clip_q", 0.005)
        lower = float(out["pred"].quantile(q))
        upper = float(out["pred"].quantile(1.0 - q))
        out["pred"] = out["pred"].clip(lower=lower, upper=upper)
        out = self._normalize_by_date_interval_rank(out, "pred")
        alpha = float(params.get("blend_alpha", 1.0))
        blend_kind = params.get("blend_kind", "none")
        if blend_kind == "rank":
            return alpha * out["pred"].to_numpy(dtype=np.float32) + (1.0 - alpha) * out["pred_rank_scaled"].to_numpy(dtype=np.float32)
        if blend_kind == "neutral":
            return alpha * out["pred"].to_numpy(dtype=np.float32) + (1.0 - alpha) * out["pred_group_demean"].to_numpy(dtype=np.float32)
        return out["pred"].to_numpy(dtype=np.float32)

    def make_target(self, ydf, target_mode):
        series, _ = self._make_target_series(ydf, target_mode)
        return series

    def fit_model(self, model_name, xtrain, ytrain, target_mode="raw", sample_weight=None, model_params=None):
        # 薄包装：从 DataFrame 抽出特征 numpy 后委托 `_fit_model_core`。
        # 交付链（submission_pipeline）整窗训练时会跳过这层、直接把预抽好的 numpy 喂给 core，
        # 从而避免「pandas 列子集 + to_numpy」两份大矩阵并存的额外内存尖峰。
        feature_cols = [c for c in xtrain.columns if c not in ["date", "symbol", "interval"]]
        x = xtrain[feature_cols].to_numpy(dtype=np.float32)
        return self._fit_model_core(
            model_name, x, feature_cols, ytrain,
            target_mode=target_mode, sample_weight=sample_weight, model_params=model_params,
        )

    def _fit_model_core(self, model_name, x, feature_cols, ytrain, target_mode="raw", sample_weight=None, model_params=None):
        # model_params：来自 spec 的预钉超参覆盖（§4.9 各模型小网格预先钉死），
        # 与各模型的默认参数合并；为 None 时完全走默认，不影响既有线性路径。
        # 约定：x 已是 float32 特征矩阵、feature_cols 与其列严格一一对应（由调用方保证）。
        mp = dict(model_params or {})
        y, baseline = self._make_target_series(ytrain, target_mode=target_mode)
        y, winsor_info = self._apply_target_winsorize(y)
        if winsor_info is None:
            log.inf("Target winsorize: disabled")
        else:
            log.inf(
                "Target winsorize: q=({lq:.3f}, {uq:.3f}) bounds=({lb:.6f}, {ub:.6f})".format(
                    lq=winsor_info["lower_quantile"],
                    uq=winsor_info["upper_quantile"],
                    lb=winsor_info["lower_bound"],
                    ub=winsor_info["upper_bound"],
                )
            )
        if model_name == "ridge":
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=mp.get("alpha", self.ridge_alpha), fit_intercept=True, random_state=None)),
            ])
        elif model_name == "elasticnet":
            en_params = {"alpha": 0.0005, "l1_ratio": 0.1, "fit_intercept": True, "max_iter": 5000, "random_state": 42}
            en_params.update(mp)
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("model", ElasticNet(**en_params)),
            ])
        elif model_name == "huber":
            # P4 候选：Huber 回归（对收益尾部稳健，介于 OLS/分位之间）。
            # epsilon 越小越稳健；alpha 是 L2 正则。预钉默认，model_params 可覆盖。
            hub_params = {"epsilon": 1.35, "alpha": 1e-4, "max_iter": 500, "fit_intercept": True}
            hub_params.update(mp)
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("model", HuberRegressor(**hub_params)),
            ])
        elif model_name == "tree":
            tree_params = {
                "n_estimators": 40, "max_depth": 16, "min_samples_leaf": 50,
                "max_features": 0.3, "bootstrap": True, "max_samples": 0.7,
                "random_state": 42, "n_jobs": 1,
            }
            tree_params.update(mp)
            model = ExtraTreesRegressor(**tree_params)
        elif model_name == "tree_big":
            tree_big_params = {
                "n_estimators": 120, "max_depth": 20, "min_samples_leaf": 20,
                "max_features": 0.5, "bootstrap": True, "max_samples": 0.8,
                "random_state": 42, "n_jobs": 1,
            }
            tree_big_params.update(mp)
            model = ExtraTreesRegressor(**tree_big_params)
        elif model_name == "tree_shallow":
            # P4 浅 ExtraTrees（§六/CLAUDE：depth≤5）。极低信噪比下 depth 是主正则器；
            # min_samples_leaf 在 ~200 万行上基本非约束、仅作下限。网格经 model_params 钉 depth。
            tree_sh_params = {
                "n_estimators": 300, "max_depth": 5, "min_samples_leaf": 500,
                "max_features": 0.4, "bootstrap": True, "max_samples": 0.7,
                "random_state": 42, "n_jobs": 1,
            }
            tree_sh_params.update(mp)
            model = ExtraTreesRegressor(**tree_sh_params)
        elif model_name in {"gbdt", "histgb"}:
            if model_name == "histgb":
                hgb_params = {
                    "loss": "squared_error", "learning_rate": 0.05, "max_iter": 200,
                    "max_depth": 8, "min_samples_leaf": 50, "l2_regularization": 0.1,
                    "early_stopping": True, "validation_fraction": 0.1, "random_state": 42,
                }
                hgb_params.update(mp)
                model = HistGradientBoostingRegressor(**hgb_params)
            else:
                gbdt_params = {
                    "learning_rate": 0.05, "n_estimators": 30, "max_depth": 2,
                    "min_samples_leaf": 200, "subsample": 0.8, "random_state": 42,
                }
                gbdt_params.update(mp)
                model = GradientBoostingRegressor(**gbdt_params)
        elif model_name == "histgb_shallow":
            # P4 浅 HistGB（max_depth≤4 + 多轮 boosting 补深度，强 L2）。网格经 model_params 钉 depth/lr。
            hgb_sh_params = {
                "loss": "squared_error", "learning_rate": 0.05, "max_iter": 300,
                "max_depth": 4, "min_samples_leaf": 200, "l2_regularization": 0.5,
                "early_stopping": True, "validation_fraction": 0.1, "random_state": 42,
            }
            hgb_sh_params.update(mp)
            model = HistGradientBoostingRegressor(**hgb_sh_params)
        elif model_name == "lgbm":
            if LGBMRegressor is None:
                raise ImportError(
                    "lightgbm is not installed. Install it or use model_name='histgb' in this workspace."
                )
            # P4-2b：LightGBM（GBDT 正牌选手）。默认参数偏保守；
            # spec 经 model_params 覆盖 depth/num_leaves/n_jobs 等。
            lgbm_params = {
                "n_estimators": 300,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "max_depth": 6,
                "min_child_samples": 200,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
                "random_state": 42,
                "n_jobs": 1,
                "verbosity": -1,
            }
            lgbm_params.update(mp)
            model = LGBMRegressor(**lgbm_params)
        elif model_name == "mlp":
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("model", MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    alpha=1e-4,
                    learning_rate_init=1e-3,
                    max_iter=10,
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=42,
                )),
            ])
        else:
            raise ValueError(f"Unknown model: {model_name}")
        # 线性成员（ridge/elasticnet/huber/mlp，均为 StandardScaler→model 的 Pipeline）统一上转 float64。
        # 根因：433 提交特征里有 ~1e15 量级未归一化列；StandardScaler.fit 在 float32 下计算
        # 均值/方差时，对百万行的巨值求和会发生灾难性抵消 → 均值是数值垃圾 → 居中/缩放后是
        # 近常数/乱列 → X^TX 近奇异 → ridge cholesky 失败回退 30GB 经济 SVD（内存紧则分配失败
        # → 系数 NaN → 等权融合整列 NaN）。上转 float64 后求和在 float64 累加、均值精确、缩放正常，
        # cholesky 毫秒级成功、内存小，彻底绕开 SVD。树/提升族尺度无关，保持 float32 省内存。
        if isinstance(model, Pipeline):
            x = np.asarray(x, dtype=np.float64)
        try:
            if sample_weight is None:
                model.fit(x, y)
            elif isinstance(model, Pipeline):
                model.fit(x, y, model__sample_weight=np.asarray(sample_weight, dtype=np.float32))
            else:
                model.fit(x, y, sample_weight=np.asarray(sample_weight, dtype=np.float32))
        except PermissionError:
            if model_name not in {"histgb", "histgb_shallow"}:
                raise
            log.yellow("HistGradientBoosting hit a sandbox permission error, falling back to GradientBoostingRegressor.")
            model = GradientBoostingRegressor(
                learning_rate=0.05,
                n_estimators=30,
                max_depth=2,
                min_samples_leaf=200,
                subsample=0.8,
                random_state=42,
            )
            if sample_weight is None:
                model.fit(x, y)
            else:
                model.fit(x, y, sample_weight=np.asarray(sample_weight, dtype=np.float32))
        return model, feature_cols, baseline

    def predict(self, model, xdf, feature_cols):
        # 与 _fit_model_core 的 float64 口径对齐：线性 Pipeline 成员（StandardScaler→model）
        # 取数为 float64，保证预测时 StandardScaler 居中对 ~1e15 量级列同样不丢精度；
        # 树/提升族尺度无关，仍取 float32 省内存。
        dtype = np.float64 if isinstance(model, Pipeline) else np.float32
        x = xdf[feature_cols].to_numpy(dtype=dtype)
        return model.predict(x)

    def _predict_with_baseline(self, model, xdf, feature_cols, ydf=None, baseline=None, target_mode="raw"):
        pred = self.predict(model, xdf, feature_cols)
        if target_mode != "interval_residual":
            return pred
        if ydf is None or baseline is None:
            raise ValueError("interval_residual requires ydf and baseline")
        merged = ydf.merge(baseline, on="interval", how="left")
        common = merged["interval_mean"].fillna(0.0).to_numpy(dtype=np.float32)
        return common + pred

    def run(self, split_config, model_name, max_train_days=None, max_val_days=None, target_mode="raw", model_params=None):
        return self.run_with_groups(
            split_config=split_config,
            model_name=model_name,
            feature_groups=None,
            max_train_days=max_train_days,
            max_val_days=max_val_days,
            target_mode=target_mode,
            model_params=model_params,
        )

    def run_with_groups(
        self,
        split_config,
        model_name,
        feature_groups=None,
        max_train_days=None,
        max_val_days=None,
        target_mode="raw",
        loader=None,
        model_params=None,
    ):
        train_dates, val_dates, test_dates = self.split_dates(split_config)
        log.inf(f"Train dates: {train_dates[0]} -> {train_dates[-1]} ({len(train_dates)})")
        log.inf(f"Val dates: {val_dates[0]} -> {val_dates[-1]} ({len(val_dates)})")
        xtrain, ytrain = self._load_group_split(
            train_dates,
            groups=feature_groups,
            max_days=max_train_days,
            loader=loader,
        )
        xval, yval = self._load_group_split(
            val_dates,
            groups=feature_groups,
            max_days=max_val_days,
            loader=loader,
        )
        # P4 提速：仅对树族模型降采样训练行；验证集 (xval/yval) 全量不动 → 打分/排名指标不被污染。
        xtrain, ytrain = self._subsample_train_rows(xtrain, ytrain, model_name=model_name)
        log.inf(f"Train shape: {xtrain.shape}, Val shape: {xval.shape}")
        model, feature_cols, baseline = self.fit_model(model_name, xtrain, ytrain, target_mode=target_mode, model_params=model_params)
        pred_train = self._predict_with_baseline(
            model,
            xtrain,
            feature_cols,
            ydf=ytrain,
            baseline=baseline,
            target_mode=target_mode,
        )
        pred_val = self._predict_with_baseline(
            model,
            xval,
            feature_cols,
            ydf=yval,
            baseline=baseline,
            target_mode=target_mode,
        )
        train_metrics = self.evaluate_prediction_bundle(ytrain, pred_train)
        val_metrics = self.evaluate_prediction_bundle(yval, pred_val)
        log.inf(
            "Train metrics - corr={corr:.4f}, r2={r2:.5f}, mse={mse:.6f}".format(**train_metrics)
        )
        log.inf(
            "Val metrics   - corr={corr:.4f}, r2={r2:.5f}, mse={mse:.6f}".format(**val_metrics)
        )
        return {
            "model": model,
            "feature_cols": feature_cols,
            "baseline": baseline,
            "pred_train": pred_train,
            "pred_val": pred_val,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "feature_groups": self._normalize_groups(feature_groups),
            "model_name": model_name,
            "target_mode": target_mode,
            # P5 融合：保留验证集 meta（date/symbol/interval/fret12）以便上层把逐行 OOF 预测落盘。
            # 仅是引用、不复制；只有串行 OOF 落盘路径会用到，不影响既有线性/并行路径。
            "yval": yval,
        }

    def run_on_features(self, xtrain, ytrain, xval, yval, model_name, target_mode="raw"):
        log.inf(f"Train shape: {xtrain.shape}, Val shape: {xval.shape}")
        model, feature_cols, baseline = self.fit_model(model_name, xtrain, ytrain, target_mode=target_mode)
        pred_train = self._predict_with_baseline(
            model,
            xtrain,
            feature_cols,
            ydf=ytrain,
            baseline=baseline,
            target_mode=target_mode,
        )
        pred_val = self._predict_with_baseline(
            model,
            xval,
            feature_cols,
            ydf=yval,
            baseline=baseline,
            target_mode=target_mode,
        )
        train_metrics = self.evaluate_prediction_bundle(ytrain, pred_train)
        val_metrics = self.evaluate_prediction_bundle(yval, pred_val)
        log.inf(
            "Train metrics - corr={corr:.4f}, r2={r2:.5f}, mse={mse:.6f}".format(**train_metrics)
        )
        log.inf(
            "Val metrics   - corr={corr:.4f}, r2={r2:.5f}, mse={mse:.6f}".format(**val_metrics)
        )
        return {
            "model": model,
            "feature_cols": feature_cols,
            "feature_count": int(len(feature_cols)),
            "baseline": baseline,
            "pred_train": pred_train,
            "pred_val": pred_val,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }

    def run_common_residual_reconstruction(
        self,
        split_config,
        residual_model_name="tree",
        common_model_name="ridge",
        feature_groups=None,
        max_train_days=None,
        max_val_days=None,
        lambda_grid=None,
        loader=None,
    ):
        train_dates, val_dates, _ = self.split_dates(split_config)
        xtrain, ytrain = self._load_group_split(
            train_dates,
            groups=feature_groups,
            max_days=max_train_days,
            loader=loader,
        )
        xval, yval = self._load_group_split(
            val_dates,
            groups=feature_groups,
            max_days=max_val_days,
            loader=loader,
        )
        common_model, common_feature_cols = self._fit_common_model(xtrain, ytrain, model_name=common_model_name)
        common_train = self._predict_common_component(common_model, common_feature_cols, xtrain)
        common_val = self._predict_common_component(common_model, common_feature_cols, xval)

        ytrain_resid = ytrain.copy()
        ytrain_resid["fret12"] = ytrain_resid["fret12"].to_numpy(dtype=np.float32) - common_train
        residual_model, residual_feature_cols, _ = self.fit_model(
            residual_model_name,
            xtrain,
            ytrain_resid,
            target_mode="raw",
        )
        residual_train = self.predict(residual_model, xtrain, residual_feature_cols)
        residual_val = self.predict(residual_model, xval, residual_feature_cols)
        if lambda_grid is None:
            lambda_grid = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
        best = None
        for lam in lambda_grid:
            pred_train = residual_train + lam * common_train
            pred_val = residual_val + lam * common_val
            val_metrics = self.evaluate_prediction_bundle(yval, pred_val)
            score = (val_metrics["corr"], -val_metrics["mse"], val_metrics["r2"])
            if best is None or score > best[0]:
                best = (
                    score,
                    lam,
                    pred_train.copy(),
                    pred_val.copy(),
                    self.evaluate_prediction_bundle(ytrain, pred_train),
                    val_metrics,
                )
        _, best_lambda, pred_train, pred_val, train_metrics, val_metrics = best
        return {
            "common_model": common_model,
            "common_feature_cols": common_feature_cols,
            "residual_model": residual_model,
            "residual_feature_cols": residual_feature_cols,
            "lambda": float(best_lambda),
            "pred_train": pred_train,
            "pred_val": pred_val,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }

    def run_soft_regime_ensemble(self, split_config, max_train_days=None, max_val_days=None, target_mode="interval_demean", loader=None):
        train_dates, val_dates, _ = self.split_dates(split_config)
        xtrain_full, ytrain = self._load_group_split(
            train_dates,
            groups=None,
            max_days=max_train_days,
            loader=loader,
        )
        xval_full, yval = self._load_group_split(
            val_dates,
            groups=None,
            max_days=max_val_days,
            loader=loader,
        )

        main_result = self.run_with_groups(
            split_config=split_config,
            model_name="tree",
            feature_groups=["base", "lag", "roll", "cross"],
            max_train_days=max_train_days,
            max_val_days=max_val_days,
            target_mode=target_mode,
            loader=loader,
        )
        regime_result = self.run_with_groups(
            split_config=split_config,
            model_name="tree",
            feature_groups=["base", "lag", "roll", "cross", "regime"],
            max_train_days=max_train_days,
            max_val_days=max_val_days,
            target_mode=target_mode,
            loader=loader,
        )

        train_meta = pd.DataFrame({
            "main_pred": np.asarray(main_result["pred_train"], dtype=np.float32),
            "regime_pred": np.asarray(regime_result["pred_train"], dtype=np.float32),
            "regime_score": xtrain_full["regime_score"].to_numpy(dtype=np.float32),
            "regime_low": xtrain_full["regime_low"].to_numpy(dtype=np.float32),
            "regime_mid": xtrain_full["regime_mid"].to_numpy(dtype=np.float32),
            "regime_high": xtrain_full["regime_high"].to_numpy(dtype=np.float32),
        })
        val_meta = pd.DataFrame({
            "main_pred": np.asarray(main_result["pred_val"], dtype=np.float32),
            "regime_pred": np.asarray(regime_result["pred_val"], dtype=np.float32),
            "regime_score": xval_full["regime_score"].to_numpy(dtype=np.float32),
            "regime_low": xval_full["regime_low"].to_numpy(dtype=np.float32),
            "regime_mid": xval_full["regime_mid"].to_numpy(dtype=np.float32),
            "regime_high": xval_full["regime_high"].to_numpy(dtype=np.float32),
        })
        meta_ytrain = ytrain.copy()
        meta_yval = yval.copy()

        candidate_models = [
            (
                "ridge",
                Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", Ridge(alpha=1.0, fit_intercept=True, random_state=None)),
                ]),
            ),
            (
                "tree",
                ExtraTreesRegressor(
                    n_estimators=120,
                    max_depth=4,
                    min_samples_leaf=400,
                    max_features=1.0,
                    bootstrap=True,
                    max_samples=0.8,
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
        best_name = None
        best_model = None
        best_pred_train = None
        best_pred_val = None
        best_metrics = None
        best_train_metrics = None
        for name, meta_model in candidate_models:
            meta_model.fit(train_meta, meta_ytrain["fret12"].to_numpy(dtype=np.float32))
            pred_train = np.asarray(meta_model.predict(train_meta), dtype=np.float32)
            pred_val = np.asarray(meta_model.predict(val_meta), dtype=np.float32)
            train_metrics = self.evaluate_prediction_bundle(meta_ytrain, pred_train)
            val_metrics = self.evaluate_prediction_bundle(meta_yval, pred_val)
            score = (val_metrics["corr"], -val_metrics["mse"], val_metrics["r2"])
            if best_metrics is None or score > (best_metrics["corr"], -best_metrics["mse"], best_metrics["r2"]):
                best_name = name
                best_model = meta_model
                best_pred_train = pred_train
                best_pred_val = pred_val
                best_metrics = val_metrics
                best_train_metrics = train_metrics
        post_params, post_train_metrics = self.choose_postprocess_params(meta_ytrain, best_pred_train)
        train_post_frame = meta_ytrain[["date", "interval", "fret12"]].copy()
        train_post_frame["pred"] = np.asarray(best_pred_train, dtype=np.float32)
        val_post_frame = meta_yval[["date", "interval", "fret12"]].copy()
        val_post_frame["pred"] = np.asarray(best_pred_val, dtype=np.float32)
        final_train_pred = np.asarray(best_pred_train, dtype=np.float32)
        final_val_pred = np.asarray(best_pred_val, dtype=np.float32)
        train_metrics = self.evaluate_prediction_bundle(meta_ytrain, final_train_pred)
        val_metrics = self.evaluate_prediction_bundle(meta_yval, final_val_pred)
        log.inf(
            "Soft meta model selected: {}".format(best_name)
        )
        log.inf(
            "Val metrics   - corr={corr:.4f}, r2={r2:.5f}, mse={mse:.6f}".format(**val_metrics)
        )
        return {
            "model": best_model,
            "feature_cols": list(train_meta.columns),
            "pred_train": final_train_pred,
            "pred_val": final_val_pred,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "main_result": main_result,
            "regime_result": regime_result,
            "postprocess_params": post_params,
            "postprocess_train_metrics": post_train_metrics,
        }

    def _build_rolling_folds(self, split_config, train_window=8, val_window=2, step=10, max_folds=None, embargo=1):
        # max_folds=None 表示使用所有可用 fold，传正整数时截断（调试用）
        train_dates, _, _ = self.split_dates(split_config)
        folds = self.make_rolling_folds(
            train_dates[0],
            train_dates[-1],
            train_window=train_window,
            val_window=val_window,
            step=step,
            min_train_days=train_window,
            embargo=embargo,
        )
        if max_folds is not None:
            folds = folds[:max_folds]
        if not folds:
            raise ValueError("No rolling folds available")
        return folds

    def _evaluate_spec_on_fold(self, fold_split, spec, max_train_days=None, max_val_days=None, loader=None):
        start_ts = time.time()
        coef_df = None
        if spec["type"] == "standard":
            result = self.run_with_groups(
                split_config=fold_split,
                model_name=spec["model"],
                feature_groups=spec["groups"],
                max_train_days=max_train_days,
                max_val_days=max_val_days,
                target_mode=spec.get("target_mode", "raw"),
                loader=loader,
                model_params=spec.get("model_params"),
            )
            train_metrics = result["train_metrics"]
            val_metrics = result["val_metrics"]
            feature_set = json.dumps(result["feature_groups"], ensure_ascii=False)
            target_type = spec.get("target_mode", "raw")
            model_type = spec["model"]
            postprocess_type = "none"
            if spec.get("collect_coefs"):
                xtrain, ytrain = self._load_group_split(
                    self.calendar.range(fold_split.train_start, fold_split.train_end),
                    groups=spec["groups"],
                    max_days=max_train_days,
                    loader=loader,
                )
                xtrain = xtrain.loc[:, ~xtrain.columns.duplicated()].copy()
                model, feature_cols, _ = self.fit_model(spec["model"], xtrain, ytrain, target_mode=spec.get("target_mode", "raw"), model_params=spec.get("model_params"))
                # 线性取系数；树/HGB 无 coef_ 时回退取特征重要性（供 P4-2 反偏颇/重要性扫描复用）。
                coef_df = self._extract_linear_coefficients(model, feature_cols)
                if coef_df is None:
                    coef_df = self._extract_tree_importance(model, feature_cols)
        elif spec["type"] == "common_residual":
            result = self.run_common_residual_reconstruction(
                split_config=fold_split,
                residual_model_name=spec.get("residual_model", "tree"),
                common_model_name=spec.get("common_model", "ridge"),
                feature_groups=spec.get("feature_groups"),
                max_train_days=max_train_days,
                max_val_days=max_val_days,
                loader=loader,
            )
            train_metrics = result["train_metrics"]
            val_metrics = result["val_metrics"]
            feature_set = json.dumps(spec.get("feature_groups") or ["base", "lag", "roll", "cross", "common_aggregate"], ensure_ascii=False)
            target_type = "common_plus_residual"
            model_type = "common_residual"
            postprocess_type = "none"
        elif spec["type"] == "soft_regime":
            result = self.run_soft_regime_ensemble(
                split_config=fold_split,
                max_train_days=max_train_days,
                max_val_days=max_val_days,
                target_mode=spec.get("target_mode", "interval_demean"),
                loader=loader,
            )
            train_metrics = result["train_metrics"]
            val_metrics = result["val_metrics"]
            feature_set = json.dumps(["main_tree", "regime_tree", "regime_score"], ensure_ascii=False)
            target_type = spec.get("target_mode", "interval_demean")
            model_type = "tree_soft_regime"
            postprocess_type = "none"
        else:
            raise ValueError(f"Unknown spec type: {spec['type']}")
        return {
            "result": result,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "feature_set": feature_set,
            "target_type": target_type,
            "model_type": model_type,
            "postprocess_type": postprocess_type,
            "runtime_sec": float(time.time() - start_ts),
            "coef_df": coef_df,
        }
