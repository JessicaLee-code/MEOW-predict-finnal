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


class MeowEngine(object):
    def __init__(self, h5dir, cacheDir):
        self.calendar = Calendar()
        self.h5dir = h5dir
        if not os.path.exists(h5dir):
            raise ValueError("Data directory not exists: {}".format(self.h5dir))
        if not os.path.isdir(h5dir):
            raise ValueError("Invalid data directory: {}".format(self.h5dir))
        # 这里保留老师样例中的 `cacheDir` 参数形态，便于外部调用保持兼容。
        # 但正式提交实现不依赖持久化特征缓存，真正的核心逻辑统一走 `src/`。
        self.cacheDir = cacheDir
        self.dloader = MeowDataLoader(h5dir=h5dir)
        self.featGenerator = MeowFeatureGenerator(cacheDir=cacheDir)
        self.model = MeowModel(cacheDir=cacheDir, h5dir=h5dir)
        self.evaluator = MeowEvaluator(cacheDir=cacheDir)

    def _build_window_frames(self, dates):
        """
        逐日现算特征，并以「预分配 + 流式填充」拼成整窗 `xdf/ydf`。

        为什么不用 `pd.concat(list_of_day_frames)`：
        - concat 需要同时持有「全部日碎片」和「拼接结果」两份 → 整窗下约 2× 内存尖峰。
        这里改为：
        1. 先用 h5 轴元信息廉价拿到每日行数（不加载数据块）、得到整窗总行数；
        2. 预分配一块整窗 float32 特征矩阵 + meta/标签数组；
        3. 逐日把当日特征写进对应行段、当日帧用完即释放。
        峰值因此≈单份整窗矩阵，而非碎片+结果两份。

        返回一个 holder 字典（而非直接返回两张表），是为了让调用方能立刻交出源帧所有权，
        交付训练链（`fit_window`）据此在末位成员训练前释放整窗源帧、进一步压低峰值。
        """

        feat_cols = self.featGenerator.featureNames()
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
            xday, yday = self.featGenerator.genFeatures(rawData)
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
        dates = self.calendar.range(startDate, endDate)
        log.inf("Running model fitting...")
        frames = self._build_window_frames(dates)
        # 交出整窗源帧所有权：消费式训练会在末位成员 fit 前释放它，压低内存峰值。
        self.model.fit_window(frames)

    def predict(self, xdf):
        return self.model.predict(xdf)

    def eval(self, startDate, endDate):
        log.inf("Running model evaluation...")
        dates = self.calendar.range(startDate, endDate)
        frames = self._build_window_frames(dates)
        xdf = frames["xdf"]
        ydf = frames["ydf"]
        ydf.loc[:, "forecast"] = self.predict(xdf)
        self.evaluator.eval(ydf)


if __name__ == "__main__":
    # ========== 老师改这里 ==========
    # h5dir: 所有 .h5 数据文件所在目录（训练+评测共用同一目录，按日期参数区分）
    # fit(start, end): 训练区间
    # eval(start, end): 评测区间
    h5dir = str((THIS_DIR.parent / "data").resolve())  # ← 改成实际数据目录路径
    engine = MeowEngine(h5dir=h5dir, cacheDir=None)
    engine.fit(20230601, 20231130)
    engine.eval(20231201, 20231229)
