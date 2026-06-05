# MEOW 金融时序预测

分钟级 A 股订单簿数据的股票收益预测研究项目。目标：预测未来 12 分钟收益 `fret12`。

## 项目定位

- 数据：144 个交易日（2023-06-01 ~ 2023-12-29），每日一个 `.h5` 文件，约 2.45 GB
- 任务：截面预测，评价指标为 `rolling_corr_mean`（IC 均值）和 `stability_score`
- 当前 rolling 结果与基线口径：见 `CLAUDE.md` 和 `docs/实验记录.md`

## 目录结构

```
MEOW--predict/
├── src/                       # 核心代码
│   ├── feature_registry.py    # 特征族注册表（@registry.stage builder）
│   ├── feature_store.py       # 特征构造 + 落盘（python -m feature_store build）
│   ├── feature_loader.py      # 特征加载（resolve_groups + manifest，默认 float32）
│   ├── eval_protocol.py       # 评测协议（Rolling Profiles / Leaderboard / make_decision）
│   ├── experiment_runner.py   # 实验编排（特征/模型/训练/评估核心逻辑）
│   ├── scheduler.py           # 并发调度（ProcessPoolExecutor + resume + 成本均衡切组）
│   ├── submission_pipeline.py # 提交桥接层（正式特征现算 + 复用 runner 训练/推理核心）
│   ├── trainer.py             # BaseTrainer ABC + TabularTrainer（DL 扩展点）
│   ├── feat_legacy.py         # teacher 原始 6 特征（legacy 对照）
│   ├── dl.py / mdl.py / eval.py / tradingcalendar.py / log.py  # 数据/模型/评估/日历/日志
├── experiments/               # 实验入口脚本
│   ├── p0_eval_protocol.py    # P0：Rolling 评测基准（主入口，含 daily/gate/ridge/quick/full suite）
│   ├── p05_lock_ridge_alpha_and_winsorize.py # P0.5：alpha + winsorize 扫锁
│   ├── run_with_memory_guard.py # 通用 RSS 内存看门狗包装器（仅 Unix）
│   ├── run_submission_full_window.py # 交付链全窗口 fit/eval + 内存峰值采样（跨平台，Windows 演练用）
│   └── legacy/p0_rolling_audit.py # 历史 rolling 审计脚本
├── meow/                      # 老师提交目录（python meow.py 入口；正式提交通道）
├── data/features/             # 特征缓存（gitignored；pickle_fallback 后端）
├── data/                      # 原始 .h5 数据（gitignored）
├── results/                   # 实验结果 CSV（gitignored）
├── .archive/                  # 废弃/归档文件（gitignored，不主动追踪）
└── docs/                      # 文档
    ├── 实验记录.md             # 所有实验记录的唯一入口
    ├── specs/                 # 规格文档
    └── archived/              # 历史快照（只读）
```

## Quick Start（老师验收用）

只需三步：

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 打开 meow/meow.py，修改 __main__ 中的 h5dir 指向数据目录
#    所有 .h5 文件（6-12月）放在同一个目录即可，训练/评测按日期参数自动区分

# 3. 运行
cd meow && python meow.py
```

运行后会依次执行：
- `fit(20230601, 20231130)` — 用 6-11 月数据训练（两成员融合：Ridge + LightGBM）
- `eval(20231201, 20231229)` — 用 12 月数据评测，输出 Pearson / R² / MSE

如需修改训练/评测区间，直接改 `meow.py` 底部的日期参数即可。

已完成一次**交付链全窗口自测**（绝对日期 `2026-05-31`）：
- 训练窗口：`20230601–20231130`
- 评测窗口：`20231201–20231229`
- 结果：`Pearson=0.0803`、`R²=0.00465`、`MSE≈2.364e-05`
- 内存峰值：`22.43 GB`

这次自测的定位是**交付链 sanity + 量纲核对**：确认正式提交通道在全窗口下可完整执行、三指标健康、内存落在预期区间。完整解释见 `CLAUDE.md` 与 `docs/实验记录.md`。

## 运行环境

- Python >= 3.8
- 依赖：`pip install -r requirements.txt`（numpy / pandas / scikit-learn / lightgbm / tables）

## 运行方式

```bash
# 从项目根目录运行（确保 src/ 在路径里）
cd MEOW--predict

# 快速验证（2折 × short profile，约 2 分钟）
PYTHONPATH=src python experiments/p0_eval_protocol.py --suite quick

# 日常筛选（默认）：short + long 快车道 + 每日 IC
PYTHONPATH=src python experiments/p0_eval_protocol.py --suite daily

# 提交关口：候选 vs 基线、只跑 expanding（macOS 必须显式 --n-workers 1，见 AGENTS §5.1）
PYTHONPATH=src python experiments/p0_eval_protocol.py \
  --suite gate --candidate-spec-id <候选实验ID> --n-workers 1

# 通用内存看门狗：为任意长任务加 RSS 保护（防休眠用 caffeinate -i）
python experiments/run_with_memory_guard.py \
  --rss-limit-gb 12 --rss-limit-duration-sec 30 --rss-hard-limit-gb 13 \
  --env PYTHONPATH=src \
  -- caffeinate -i python experiments/p0_eval_protocol.py \
    --suite daily --resume --run-id <run_id>

# 正式提交通道（老师入口，正式特征现算，不依赖本地缓存）
cd meow && python meow.py

# 交付链全窗口演练 + 内存峰值核对（≥32GB 机器；Windows 步骤见 docs/交付演练SOP_Windows全窗口.md）
python experiments/run_submission_full_window.py
```

**已锁定的标准口径（勿手改，对照才显式覆写）**：训练标签 winsorize = 开启 + P1/P99；ridge alpha = 2.0；特征 dtype = float32。来源见 AGENTS §7.7 / §7.11。

## 关键文档

| 文档 | 说明 |
|---|---|
| `NOTE.md` | 用户实验笔记：策略讨论、概念、待决问题 |
| `CLAUDE.md` | 当前阶段任务看板、进度、决策 |
| `AGENTS.md` | 开发规范、实验 SOP、禁止事项 |
| `docs/实验记录.md` | 所有实验结果的历史记录 |
| `docs/P0运行耗时监控报告_20260525.md` | 本次 P0 `expanding_40d_5d` 运行的耗时监控与阶段分析报告 |
| `docs/交付演练SOP_Windows全窗口.md` | 交付链迁移到 Windows + 全窗口 fit/eval + 内存峰值核对的演练 SOP |
| `docs/specs/高分实验总方案V2.md` | 整体方案设计 |
| `docs/specs/MEOW金融时序预测V3.3_论文启发稳健冲10_AI执行版.md` | V3.3 执行方案 |
| `docs/specs/实验平台架构设计.md` | 并发实验平台架构（trainer/scheduler/resume） |
| `docs/specs/特征管道重构规格.md` | PE1 特征三件套（registry/store/loader）设计与实现规格 |
| `docs/specs/开跑前编码指导_评测口径与提速.md` | 两速评测口径落地 + expanding 提速的编码实施清单（含 §2c 串/并行口径） |
| `docs/specs/meow提交通道收口规格.md` | 提交通道（meow.py）正式特征现算与训练/推理复用规格 |
| `experiments/run_with_memory_guard.py` | 通用内存看门狗包装器，超过 RSS 阈值自动终止任务 |
