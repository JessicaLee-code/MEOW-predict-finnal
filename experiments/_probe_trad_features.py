"""一次性探针：抽样训练窗交易日，检查传统提交特征是否含非有限/极值，
定位 X1 ridge fit 时 cholesky 失败的根因（Inf/极值 vs 纯秩亏）。用完即弃。"""
import sys, importlib.util
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "meow"))
spec = importlib.util.spec_from_file_location("meow_probe", str(REPO / "meow" / "meow.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

eng = m.MeowEngine(h5dir=str((REPO / "data").resolve()), cacheDir=None)
probe_dates = [20230615, 20230717, 20230815, 20230915, 20231016, 20231115, 20231129]

print("=== PROBE START ===", flush=True)
for d in probe_dates:
    try:
        ds = eng.calendar.range(d, d)
        fr = eng._build_window_frames(ds)
        xdf = fr["xdf"]
        Xnum = xdf.select_dtypes("number")
        X = Xnum.to_numpy()
        nfin = ~np.isfinite(X)
        bad_mask = nfin.any(axis=0)
        bad_cols = [c for c, b in zip(Xnum.columns, bad_mask) if b]
        finite_vals = X[np.isfinite(X)]
        mx = float(np.max(np.abs(finite_vals))) if finite_vals.size else float("nan")
        print(f"{d}: rows={X.shape[0]} cols={X.shape[1]} nonfinite_cells={int(nfin.sum())} "
              f"bad_cols={len(bad_cols)} maxabs={mx:.4g}", flush=True)
        if bad_cols:
            print(f"    非有限列: {bad_cols[:15]}", flush=True)
    except Exception as e:
        print(f"{d}: ERROR {type(e).__name__}: {e}", flush=True)
print("=== PROBE END ===", flush=True)
