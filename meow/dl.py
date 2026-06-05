import os
import pandas as pd
import tables
from tradingcalendar import Calendar
from log import log


class MeowDataLoader(object):
    def __init__(self, h5dir):
        self.h5dir = h5dir
        self.calendar = Calendar()
        # 健壮性：把数据目录实际交易日并入日历，老师换别的时段数据也不会被 isTradingDay 误判而崩（2023 数据 no-op）。
        self.calendar.mergeDataDirDays(h5dir)

    def countDate(self, date):
        # 只读 h5 轴元信息(pandas fixed 格式行索引存于 axis1)拿当日行数，不加载数据块。
        if not self.calendar.isTradingDay(date):
            raise ValueError("Not a trading day: {}".format(date))
        h5File = os.path.join(self.h5dir, "{}.h5".format(date))
        with tables.open_file(h5File, "r") as h:
            for node in h.walk_nodes("/"):
                if getattr(node, "_v_name", None) == "axis1" and hasattr(node, "shape"):
                    return int(node.shape[0])
        raise ValueError("无法从 h5 读取行数（未找到 axis1）: {}".format(h5File))

    def loadDates(self, dates):
        if len(dates) == 0:
            raise ValueError("Dates empty")
        log.inf("Loading data of {} dates from {} to {}...".format(len(dates), min(dates), max(dates)))
        return pd.concat(self.loadDate(x) for x in dates)

    def loadDate(self, date):
        if not self.calendar.isTradingDay(date):
            raise ValueError("Not a trading day: {}".format(date))
        h5File = os.path.join(self.h5dir, "{}.h5".format(date))
        df = pd.read_hdf(h5File)
        df.loc[:, "date"] = date
        precols = ["symbol", "interval", "date"]
        df = df[precols + [x for x in df.columns if x not in precols]] # re-arrange columns
        return df
