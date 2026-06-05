# 交付演练 SOP — Windows 全窗口 fit/eval + 内存峰值核对

> 目的：把正式提交链（`X1 ridge + lgbm_d4` 两成员融合）在 **≥32GB 机器**上跑一次
> 全窗口 `fit(Jun–Nov) + eval(Dec)`，核验：① 不 OOM；② 内存峰值符合「中档内存精简」
> 预期（fit 持续峰 ~20GB、lgbm 列抽 numpy 处瞬时尖峰可达 ~28GB）；③ 三指标量纲健康。
>
> 用法定位：**fire-and-forget**。照 Part A 装好环境，跑 Part B **那一条命令**就走开，
> 所有输出自动落到一个固定日志文件。跑完按 Part C 把日志（或末尾「FINAL 汇总」几行）
> 带回，由 Claude 在 Mac 侧核对峰值（Part D）。
>
> ⚠️ 本机 Mac 仅 16GB，跑不动全窗口（会 OOM），故全量峰值**必须**在 Windows（或其它
> ≥32GB 机器）上实测。脚本逻辑已在本机用小窗口（训 3 天/评 1 天）冒烟验证通过。

---

## Part A — 迁移到 Windows

### A1. 拷贝仓库

在 Mac 上打包（排除 Python 缓存和特征缓存）：
```bash
cd ~/code
tar czf MEOW--predict.tar.gz \
  --exclude='__pycache__' \
  --exclude='data/features' \
  MEOW--predict/
```
把生成的 `MEOW--predict.tar.gz` 传到 Windows（U 盘 / 网盘 / scp 均可），解压即用。

包含内容：全部代码 + `.git/` 历史 + `data/*.h5` 原始数据 + `results/` + `logs/` + `.archive/`。
排除内容：`__pycache__`（跨平台无意义）、`data/features/`（特征缓存，交付链不依赖，运行时自动现算）。

### A2. 数据确认
拷贝后 `data/` 目录下应有 144 个 `.h5` 文件，覆盖 `20230601` ~ `20231229` 所有交易日。
缺任何一个交易日的文件，运行会直接报错。

### A3. Python 环境（PowerShell）
```powershell
cd MEOW--predict
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
# 演练脚本额外需要 psutil（采样内存峰值用，不进正式提交依赖）
pip install psutil
```

### A4. 装好自检（可选但建议）
```powershell
python -c "import numpy,pandas,sklearn,lightgbm,tables;print('deps ok')"
```

---

## Part B — 跑（就这一条命令，跑完即走）

在仓库根目录、`.venv` 已激活：
```powershell
python experiments\run_submission_full_window.py
```
- 默认就是全窗口：训练 `20230601–20231130`、评测 `20231201–20231229`。
- 数据若不在 `data/` 下，打开 `meow/meow.py` 底部把 `h5dir = ...` 改成实际路径即可。
- 全程自动：现算特征 → fit 两成员（X1 先、lgbm 末位）→ eval(Dec) → 打印峰值汇总。
- **耗时**：取决于 CPU，整窗 lgbm + ridge 训练大致十几分钟到半小时量级，放着别管。

输出去向（无需手动重定向，脚本已 tee）：
- 控制台实时打印；同时**完整写入** `logs\submission_full_window_<时间戳>.log`。
- 日志含：环境信息、`Preallocating window matrix` 行（整窗行数×特征，可反推矩阵 GB）、
  每 ~20s 一条 `[mem] current=… peak=…` 心跳、`PHASE fit/eval` 标记、eval 三指标、
  以及末尾显眼的「FINAL 汇总」。

---

## Part C — 跑完做什么

把结果带回 Mac 仓库供 Claude 核对，二选一：
1. **带文件**：把 `logs\submission_full_window_<时间戳>.log` 拷到 Mac 仓库的 `logs/` 下
   （`logs/` 已被 git 忽略，不会污染提交），然后告诉 Claude 文件名。
2. **贴尾巴**：日志末尾的「FINAL 汇总」整块 + 几条 `[mem]` 心跳，直接贴给 Claude 即可判断。

至少要让 Claude 看到这几行：
```
FINAL PEAK RSS（全程最高）：__.__ GB
  其中 fit 阶段峰值：__.__ GB
（外加几条 [mem] current=… peak=… 心跳，用来看“持续峰”而非只看瞬时尖峰）
```

---

## Part D — 预期与核对口径（Claude 回来按此判断）

| 观测 | 预期 | 含义 |
|---|---|---|
| `Preallocating window matrix` 行 | 训练 ≈ 8.5M 行 × 433 特征 ≈ **~14.7 GB** | 整窗特征矩阵地板，符合即说明逐日流式填充的预分配正确 |
| `[mem]` **持续心跳峰** | 稳定在 **~15–20 GB** | fit 的持续工作集（整窗矩阵 + 单成员训练）落在中档目标 |
| `FINAL PEAK RSS`（含瞬时） | **≤ ~28 GB** | 允许 lgbm 列抽 numpy 那一刻「整窗源帧 + lgbm numpy」短暂叠加；32GB 可 survive |
| eval 三指标 | Pearson 正、MSE 量级正常、R² 非荒谬负值 | raw_mean 量纲正确（Dec 只当 sanity，**不回灌选型** §4.8） |

判定：
- 持续峰 ~20GB 且全程峰 ≤ ~28GB、无 OOM → **达成中档预期**，交付链内存达标。
- 若全程峰明显 > 28GB / 逼近 32GB / OOM → 回报 Claude，再评估是否上「激进档」
  （按成员预分配两张矩阵、永不物化整窗 union，可压到 ~16–18GB，但改动大、需重核 train/serve 对称）。

> 备注：这次 eval(Dec) 会动用 12 月 Final Holdout——按 `AGENTS §4.8`，**只在最终代表选定后
> 一次性确认、看完不改选型**。本演练把 Dec 分当 sanity bonus，不作为选型依据。
