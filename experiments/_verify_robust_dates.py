# -*- coding: utf-8 -*-
"""
快速验证「入口日期健壮性」修复（不触发任何训练）：
A. 真实 2023 数据：日历并入 = no-op（天数不变）、isTradingDay 正常、__main__ 自动推日期
   兜底分支【不触发】、推出的 train/eval 区间逐字等于原始 20230601-20231130 / 20231201-20231229。
B. 模拟「老师换成别的时段数据」：临时目录里造几个假的 2024-命名 .h5 → 日历能识别这些日期、
   isTradingDay 为真、__main__ 兜底分支【触发】并按数据范围正确切分。
"""
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# 镜像真实入口 `python meow.py` 的 path 优先级：脚本目录 meow/ 在最前（sys.path[0]），
# src/ 仅作为 `from log import log` 的来源 append 在后。故此处 meow/ 必须在 src/ 之前，
# 确保验证的是入口实际会用到的 meow/tradingcalendar.py（而非 src/ 同名副本）。
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "meow"))

from tradingcalendar import Calendar  # noqa: E402


def derive_dates(h5dir, train_start, train_end, eval_start, eval_end):
    """复刻 meow.py __main__ 的兜底自动推日期逻辑（与源码保持一致）。"""
    available = sorted(int(p.stem) for p in Path(h5dir).glob("*.h5") if p.stem.isdigit())
    triggered = bool(available) and not any(train_start <= d <= eval_end for d in available)
    if triggered:
        eval_n = 20
        if len(available) > eval_n:
            train_start, train_end = available[0], available[-eval_n - 1]
            eval_start, eval_end = available[-eval_n], available[-1]
        else:
            train_start, train_end = available[0], available[-2]
            eval_start, eval_end = available[-1], available[-1]
    return triggered, (train_start, train_end, eval_start, eval_end)


def main():
    # ---------- A. 真实 2023 数据：必须是 no-op ----------
    data_dir = REPO / "data"
    base = Calendar()
    n_before = len(base.tradingDays)
    cal = Calendar()
    cal.mergeDataDirDays(str(data_dir))
    n_after = len(cal.tradingDays)
    print("A. 真实 2023 数据")
    print("   日历天数 并入前={} 并入后={}  -> {}".format(
        n_before, n_after, "no-op ✓" if n_before == n_after else "改变 ✗(异常)"))
    print("   isTradingDay(20230601)={}, isTradingDay(20231229)={}".format(
        cal.isTradingDay(20230601), cal.isTradingDay(20231229)))
    trig, dates = derive_dates(str(data_dir), 20230601, 20231130, 20231201, 20231229)
    ok_A = (not trig) and dates == (20230601, 20231130, 20231201, 20231229)
    print("   兜底触发={} 推出区间={}  -> {}".format(
        trig, dates, "与原始逐字一致 ✓" if ok_A else "✗(异常)"))

    # ---------- B. 模拟别的时段数据（假 2024 文件名） ----------
    with tempfile.TemporaryDirectory() as td:
        fake_days = [20240102, 20240103, 20240104, 20240105, 20240108,
                     20240109, 20240110, 20240111, 20240112, 20240115,
                     20240116, 20240117, 20240118, 20240119, 20240122,
                     20240123, 20240124, 20240125, 20240126, 20240129,
                     20240130, 20240131, 20240201, 20240202, 20240205]
        for d in fake_days:
            (Path(td) / "{}.h5".format(d)).write_bytes(b"")  # 空文件，仅测日期识别
        cal2 = Calendar()
        before2 = cal2.isTradingDay(20240102)
        cal2.mergeDataDirDays(td)
        after2 = cal2.isTradingDay(20240102)
        print("\nB. 模拟别的时段数据（假 2024，25 个交易日）")
        print("   isTradingDay(20240102) 并入前={} 并入后={}  -> {}".format(
            before2, after2, "识别成功 ✓" if (not before2 and after2) else "✗(异常)"))
        trig2, dates2 = derive_dates(td, 20230601, 20231130, 20231201, 20231229)
        # 25 天 > eval_n=20 → train=前5天(0102-0108), eval=后20天(0109-0205)
        exp = (20240102, 20240108, 20240109, 20240205)
        ok_B = trig2 and dates2 == exp
        print("   兜底触发={} 推出区间={}  -> {}".format(
            trig2, dates2, "按数据范围正确切分 ✓" if ok_B else "✗(期望{})".format(exp)))

    print("\n==== 结论 ====")
    if ok_A and ok_B:
        print("全部通过：2023 数据零影响（no-op + 区间逐字一致）；别的时段数据可识别、不崩、自动切分。")
    else:
        print("有失败项，需排查。")


if __name__ == "__main__":
    main()
