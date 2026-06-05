import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# 把仓库 `src/` 加入搜索路径，但不改老师要求的 `python meow.py` 入口形式。
# 这样 meow 仍然是正式提交壳层，而真正的核心实现统一收口在 `src/`。
THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from log import log
from dl import MeowDataLoader
from feat import MeowFeatureGenerator
from mdl import MeowModel
from eval import MeowEvaluator
from tradingcalendar import Calendar
from dl_serve import DLServe, fuse_traditional_with_dl


class MeowEngine(object):
    def __init__(self, h5dir, cacheDir):
        self.calendar = Calendar()
        self.h5dir = h5dir
        if not os.path.exists(h5dir):
            raise ValueError("Data directory not exists: {}".format(self.h5dir))
        if not os.path.isdir(h5dir):
            raise ValueError("Invalid data directory: {}".format(self.h5dir))
        # 健壮性：把数据目录里实际存在的交易日并入引擎日历，使 fit/eval 的
        # calendar.range 在老师换成别的时段数据时也能返回正确日期。对 2023 数据为 no-op。
        self.calendar.mergeDataDirDays(h5dir)
        # 这里保留老师样例中的 `cacheDir` 参数形态，便于外部调用保持兼容。
        # 但正式提交实现不依赖持久化特征缓存，真正的核心逻辑统一走 `src/`。
        self.cacheDir = cacheDir
        self.dloader = MeowDataLoader(h5dir=h5dir)
        self.featGenerator = MeowFeatureGenerator(cacheDir=cacheDir)
        self.model = MeowModel(cacheDir=cacheDir, h5dir=h5dir)
        self.evaluator = MeowEvaluator(cacheDir=cacheDir)
        # DL-on-raw serve 腿（截面卡带直吃 raw 59 通道）：fit() 时现训 K seed、predict() 与传统等权融合。
        # 防御式——torch/CUDA/任一 DL 环节出错则 available=False，自动回落纯传统（最坏=传统保底 0.0812，绝不崩）。
        # MEOW_DISABLE_DL=1 可彻底关掉 DL、退回纯传统提交链。
        self.dl_enabled = os.environ.get("MEOW_DISABLE_DL", "").strip() not in ("1", "true", "True")
        self.dl_serve = DLServe(raw_loader=self.dloader.loadDates) if self.dl_enabled else None

    def _build_window_frames(self, dates, groups=None):
        """逐日现算特征 +「预分配整窗 float32 矩阵 + 流式填充」拼成 xdf/ydf（避开 concat 的 2× 内存尖峰）。

        groups=None 现算全部成员并集（predict/eval）；传入某成员 groups 则只算该成员，供 fit 逐成员把峰值压到单成员级。
        返回 holder 字典让调用方可交出源帧所有权、在末位成员训练前释放整窗源帧。
        """

        feat_cols = self.featGenerator.featureNames(groups)
        n_feat = len(feat_cols)
        ycol = self.featGenerator.ycol
        # 预读每日行数（只读 h5 轴元信息），据此预分配整窗缓冲。
        per_day_rows = [self.dloader.countDate(d) for d in dates]
        total = int(sum(per_day_rows))
        log.inf(
            "Preallocating window matrix: {} rows x {} feats (~{:.1f} GB)".format(
                total, n_feat, total * n_feat * 4 / 1e9
            )
        )
        xmat = np.empty((total, n_feat), dtype=np.float32)
        date_arr = np.empty(total, dtype=np.int64)
        symbol_arr = np.empty(total, dtype=np.int64)
        interval_arr = np.empty(total, dtype=np.int64)
        ymat = np.empty(total, dtype=np.float32)

        r = 0
        for date, expected in zip(dates, per_day_rows):
            rawData = self.dloader.loadDate(date)
            xday, yday = self.featGenerator.genFeatures(rawData, groups)
            # genFeatures 返回 (date,symbol,interval) MultiIndex 形态，这里还原成普通列再取数。
            xday = xday.reset_index()
            yday = yday.reset_index()
            n = len(xday)
            if n != expected:
                raise ValueError(
                    "行数预读({})与实算({})不一致 @ {}".format(expected, n, date)
                )
            # 按列名取数：即使列顺序漂移也对齐到 feat_cols。
            xmat[r:r + n, :] = xday[feat_cols].to_numpy(dtype=np.float32)
            date_arr[r:r + n] = xday["date"].to_numpy(dtype=np.int64)
            symbol_arr[r:r + n] = xday["symbol"].to_numpy(dtype=np.int64)
            interval_arr[r:r + n] = xday["interval"].to_numpy(dtype=np.int64)
            ymat[r:r + n] = yday[ycol].to_numpy(dtype=np.float32)
            r += n
            del rawData, xday, yday
        if r != total:
            raise ValueError("累计行数({})与预分配({})不一致".format(r, total))

        # copy=False：直接以预分配的 numpy 缓冲构造 DataFrame，不再复制大矩阵。
        # meta 作为普通列追加；提交链按列名取子集，列顺序无所谓。
        xdf = pd.DataFrame(xmat, columns=feat_cols, copy=False)
        xdf["date"] = date_arr
        xdf["symbol"] = symbol_arr
        xdf["interval"] = interval_arr
        ydf = pd.DataFrame(
            {
                "date": date_arr,
                "symbol": symbol_arr,
                "interval": interval_arr,
                ycol: ymat,
            },
            copy=False,
        )
        return {"xdf": xdf, "ydf": ydf}

    def fit(self, startDate, endDate):
        """全窗训练：逐成员现算各自 groups 特征 → fit → 释放，把内存峰值压到单成员级（避免并集整窗 + ridge float64 叠加 OOM；与整窗 fit 数学等价、有单测护网）。"""

        dates = self.calendar.range(startDate, endDate)
        log.inf("Running model fitting (逐成员现算+fit，压内存峰值到单成员级)...")
        self.model.begin_fit()
        for member in self.model.member_specs():
            log.inf(
                "  逐成员现算并训练成员: {} (groups={})".format(
                    member.experiment_id, list(member.groups)
                )
            )
            # 只现算该成员 groups 的整窗特征；交出 holder 所有权，fit 前即释放、压低峰值。
            member_frames = self._build_window_frames(dates, groups=member.groups)
            self.model.fit_one_member(member, member_frames)
        self.model.end_fit()
        # —— 焊接：传统训练完后，现训 DL-on-raw 腿（K seed）；失败内部已吞、自动回落纯传统 —— #
        if self.dl_serve is not None:
            log.inf("Running DL-on-raw fitting (现训 K seed; 失败自动回落纯传统)...")
            self.dl_serve.fit(dates)

    def predict(self, xdf):
        # 传统预测照旧；若 DL 腿现训成功，则按 (date,symbol,interval) 与传统等权融合，
        # DL 因序列 warmup 缺的行用纯传统填。DL 不可用 → 直接返回纯传统（保底）。
        trad_pred = self.model.predict(xdf)
        if self.dl_serve is None or not self.dl_serve.available:
            return trad_pred
        eval_dates = sorted(int(d) for d in pd.unique(xdf["date"]))
        dl_df = self.dl_serve.predict(eval_dates)
        return fuse_traditional_with_dl(xdf, trad_pred, dl_df)

    def eval(self, startDate, endDate):
        log.inf("Running model evaluation...")
        dates = self.calendar.range(startDate, endDate)
        frames = self._build_window_frames(dates)
        xdf = frames["xdf"]
        ydf = frames["ydf"]
        ydf.loc[:, "forecast"] = self.predict(xdf)
        self.evaluator.eval(ydf)


if __name__ == "__main__":
    # 默认优先读环境变量，便于老师或本地脚本临时覆盖数据路径；
    # 若未设置，则直接回退到仓库根目录下的 `data/`，保证在本仓库里 `python meow.py` 可直接跑。
    default_h5dir = os.environ.get("MEOW_DATA_DIR", str((THIS_DIR.parent / "data").resolve()))
    train_start = int(os.environ.get("MEOW_TRAIN_START", "20230601"))
    train_end = int(os.environ.get("MEOW_TRAIN_END", "20231130"))
    eval_start = int(os.environ.get("MEOW_EVAL_START", "20231201"))
    eval_end = int(os.environ.get("MEOW_EVAL_END", "20231229"))

    # 健壮性兜底：老师评测时只更换数据文件路径、不改入口日期。若上面的默认/配置日期区间
    # 在实际数据目录里一天都不存在（= 老师换成了别的时间段数据），就直接按数据目录里真实
    # 存在的交易日自动切分（末 eval_n 个交易日作评测、其余作训练），避免 `python meow.py`
    # 因日期对不上而崩。对原始 2023 数据该分支不触发，train/eval 区间与之前逐字一致。
    available = sorted(
        int(p.stem) for p in Path(default_h5dir).glob("*.h5") if p.stem.isdigit()
    )
    if available and not any(train_start <= d <= eval_end for d in available):
        eval_n = 20  # 评测窗口交易日数，与原始 12 月评测窗口（约 20 个交易日）对齐
        if len(available) > eval_n:
            train_start, train_end = available[0], available[-eval_n - 1]
            eval_start, eval_end = available[-eval_n], available[-1]
        else:  # 数据极少：留最后 1 个交易日评测、其余训练，保底仍能跑
            train_start, train_end = available[0], available[-2]
            eval_start, eval_end = available[-1], available[-1]
        log.inf(
            "配置日期区间与数据不相交，自动按数据范围切分: "
            "train {}-{} / eval {}-{}".format(train_start, train_end, eval_start, eval_end)
        )

    engine = MeowEngine(h5dir=default_h5dir, cacheDir=None)
    engine.fit(train_start, train_end)
    engine.eval(eval_start, eval_end)
