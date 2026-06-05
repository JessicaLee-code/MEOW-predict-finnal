# MEOW 金融时序预测 V3.3 实验方案：论文启发的稳健冲 10% AI 执行版

版本：V3.3  
目标读者：后续执行实验的 AI、队友、代码实现者、汇报整理者  
核心目标：在不违背老师要求、不引入泄漏、不牺牲可运行性的前提下，尽可能冲击 `Pearson Corr > 0.10`。  
当前策略：从 V3.1 的 `soft regime ensemble` 单点冲高，切换为 **稳健 tabular backbone + 横截面目标分解 + 非平稳归一化 + patch 序列摘要 + 轻量序列专家 + 受限融合**。

---

## 0. 方案总判断

当前实验线不是完全失败，而是出现了明确的泛化风险：

- `soft regime ensemble` 在正式 holdout 上最好，`val corr ≈ 0.0852`。
- 但 `train corr ≈ 0.3052`，`val r2 ≈ -0.0655`，训练-验证差距过大。
- rolling 三折约为 `0.0937 / 0.0142 / 0.0169`，说明该方法对时间切分极其敏感。
- `common residual + soft regime` 的简单融合没有带来额外收益，最优权重退化为只用 `soft regime`。
- 复杂 postprocess 没有稳定提升，部分设置反而降低 corr。

因此，后续 AI 不应继续扩大 `soft regime` 的复杂度。V3.3 的执行原则是：

```text
稳定性优先于单点分数；
最终输出必须仍然是原始 fret12 预测；
所有冲 10% 的模块必须经过 purged rolling validation；
soft regime 最多作为小权重 expert，而不是最终主模型。
```

---

## 1. 老师要求硬约束

后续 AI 执行实验时必须首先满足以下约束，冲 10% 只能排在这些约束之后。

### 1.1 预测目标

MUST：最终提交预测值必须对应原始 `fret12`。  
MUST：允许训练中使用 residual、rank、common component 等辅助目标，但最终输出必须映射回原始预测任务。  
FORBIDDEN：只提交 `interval residual`、`rank target` 或任何非原始尺度目标作为最终预测。

### 1.2 评价指标

MUST：每次实验同时记录：

```text
corr
mse
r2
daily_corr_mean
daily_corr_std
rolling_corr_mean
rolling_corr_std
rolling_corr_min
```

MUST：最终模型选择不能只看单次 holdout corr。  
SHOULD：模型选择优先使用 rolling 稳定分数：

```text
stability_score = rolling_corr_mean - 0.7 * rolling_corr_std
```

若两个模型 corr 接近，则选择 `MSE` 更低、`R²` 更高、rolling 最差折更不崩的模型。

### 1.3 代码提交

MUST：最终代码保持类似初始项目风格，入口为：

```bash
python meow.py
```

MUST：最终提交包不得依赖本地缓存、不得包含数据、不得包含模型缓存文件。  
MUST：LightGBM、PyTorch 等非标准依赖只能作为可选增强；若环境没有这些包，代码必须自动退化到 sklearn-only 版本。  
MUST：基础可运行版本必须只依赖 Python >= 3.8 和常见数据科学库。

### 1.4 报告要求

MUST：报告包含以下内容：

1. 原始数据分析与金融数据难点。
2. 特征工程逻辑，不允许只写“生成很多特征”。
3. 模型选择、模型优缺点、失败实验。
4. 创新点：论文启发方法、Agent 因子生成、非平稳处理、目标分解、受限融合。
5. 结果分析：三指标、rolling 稳定性、消融实验。
6. 成员分工：每个人必须有实质贡献，不能只有报告整理或汇报。

---

## 2. 论文启发与落地转化

本方案只吸收论文思想，不盲目照搬大模型。

### 2.1 DeepLOB 启发

DeepLOB 使用卷积提取 order book 空间结构，并用 LSTM 捕捉较长时间依赖。对应本项目，完整 DeepLOB 不适合直接照搬，因为当前数据是分钟聚合微观结构特征，而不是完整逐笔 LOB 快照。

落地转化：

```text
不要做完整 DeepLOB。
做 MLPLOB-lite / ConvLOB-lite：
过去 30~60 个 interval 的核心微观结构特征
→ 简单 Conv1D 或 Flatten-MLP
→ 输出 raw / residual / rank prediction
```

### 2.2 PatchTST 启发

PatchTST 的关键思想是把时间序列切成 patch，降低 token 数并保留局部语义。本项目不直接上 Transformer，而是把 patch 思想转化为 tabular 特征。

落地转化：

```text
过去 60 个 interval
切成若干 patch：6 / 12 / 24 / 60
每个 patch 计算 mean / std / last / slope / range
形成 patch summary features
再喂给 ExtraTrees / HistGradientBoosting / Ridge
```

### 2.3 RevIN 与 Non-stationary Transformer 启发

金融数据非平稳，V3.1 的 rolling 崩坏说明模型对时间分布漂移敏感。RevIN 和 Non-stationary Transformer 的启发是：先让序列更可预测，同时保留必要的非平稳信息。

落地转化：

```text
不再继续加复杂 regime gating。
改为加入 regime-aware / RevIN-style normalization features。
```

具体做法：

```text
rolling_z = (x - rolling_median_past) / (rolling_iqr_past + eps)
cross_section_z = (x - cs_median(date, interval)) / (cs_iqr(date, interval) + eps)
```

### 2.4 Purged / Embargoed Validation 启发

本任务标签为未来 12 分钟收益，普通时间切分仍可能存在相邻样本标签重叠或特征窗口重叠。后续必须使用 purged rolling validation。

落地转化：

```text
embargo_intervals = max(12, max_history_window)
```

如果使用过去 60 interval 特征，则 embargo 至少为 60 interval。若以 date 为单位切分且相邻日期完全不连续，也应在同日内部 rolling 验证时保留 embargo。

---

## 3. 数据泄漏红线

任何后续 AI 执行实验时，必须逐项检查本节。

### 3.1 标签泄漏

FORBIDDEN：验证集或测试集预测时使用真实 `fret12` 的 `date + interval` 均值、rank、std、分位数来修正预测。  
FORBIDDEN：使用未来 `midpx`、未来成交、未来盘口或任何由未来生成的特征。  
FORBIDDEN：用全量数据拟合 scaler、imputer、feature selector、PCA、target encoder。

### 3.2 residual target 使用规则

训练 residual 模型时允许：

```text
y_resid_train = y_train - mean(y_train | date, interval)
```

验证或测试预测时禁止：

```text
pred = pred_resid + mean(y_val | date, interval)   # 禁止
pred = pred_resid + mean(y_test | date, interval)  # 禁止
```

正确做法：

```text
pred_final = w_raw * pred_raw
           + w_resid * pred_resid
           + w_rank * pred_rank_scaled
           + w_common * pred_common
```

其中 `pred_common` 必须由模型根据当前及历史特征预测。

### 3.3 rank target 使用规则

训练 rank 模型时允许：

```text
y_rank_train = percentile_rank(y_train within date + interval)
y_rank_train = 2 * y_rank_train - 1
```

验证/测试预测时禁止使用真实 y rank。rank 模型只能输出自己的预测值，之后可按训练集尺度进行缩放或与其他模型融合。

### 3.4 stacking / blending 泄漏

FORBIDDEN：用同一训练数据上直接 fit 后的预测训练二层模型。  
MUST：若使用二层 Ridge/ElasticNet 融合，必须使用 OOF prediction。  
SHOULD：如果时间不够，就用固定受限权重网格，不做自由 stacking。

### 3.5 postprocess 泄漏

FORBIDDEN：根据 final holdout 反复调 `clip quantile`、`rank alpha`、`neutralization alpha`。  
SHOULD：postprocess 只保留轻微 clipping，除非 rolling 所有折都提升。

---

## 4. V3.3 总体架构

```text
Input data
    ↓
Feature block A: core microstructure features
Feature block B: RevIN-style / regime-aware normalization features
Feature block C: patch summary features
Feature block D: optional Agent-generated factors after IC filtering
    ↓
Target branches:
    raw fret12
    interval residual
    cross-sectional rank target
    common component
    ↓
Model experts:
    Ridge / ElasticNet
    ExtraTrees
    HistGradientBoosting
    optional LightGBM
    optional MLPLOB-lite / ConvLOB-lite
    old soft regime expert with capped weight
    ↓
Constrained blend
    ↓
Light clipping only
    ↓
Final raw fret12 prediction
```

---

## 5. 验证协议

### 5.1 数据切分

MUST：所有模型使用相同切分，避免不可比。  
MUST：保留一个 final holdout，只在最终模型定型后看一次。  
MUST：至少使用 3 折 rolling validation，最好 5 折。

推荐协议：

```text
Fold k:
    train_dates = earlier block
    embargo = max(12, max_history_window)
    val_dates = following block
```

如果切分粒度是 interval，则必须 purge 掉训练集中与验证标签窗口重叠的样本。

### 5.2 模型入选标准

一个模块进入最终融合，必须满足：

```text
MUST: rolling_corr_mean 不低于当前 backbone
MUST: rolling_corr_min 不明显崩坏
MUST: MSE / R² 不显著恶化
SHOULD: 与 backbone 的预测相关性不太高，能提供互补信息
FORBIDDEN: 只因为单次 holdout 分数高就进入最终模型
```

建议保留标准：

```text
rolling_corr_mean 提升 >= 0.002
或 rolling_corr_std 明显下降
或 MSE/R² 明显改善且 corr 不下降
```

---

## 6. 实验阶段设计

### Stage 0：泄漏审计与基准复现

目标：确认当前代码没有泄漏，建立可信的 rolling baseline。

MUST RUN：

```text
B0: Ridge raw, baseline 6 features
B1: Ridge raw, core features
B2: ExtraTrees raw, core features
B3: ExtraTrees residual, core features
B4: HistGradientBoosting raw, core features
B5: HistGradientBoosting residual, core features
B6: common residual branch
B7: current soft regime ensemble, unchanged, only for comparison
```

输出表：

| id | model | target | feature_set | holdout_corr | fold1 | fold2 | fold3 | mean | std | min | mse | r2 | decision |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|

验收：

```text
如果 B7 单点高但 rolling stability_score 低，不得作为主模型。
选择 B2~B6 中稳定性最高者作为 backbone。
```

---

### Stage 1：多目标分支

目标：不再只依赖 residual 或 soft regime，而是构建 raw / residual / rank / common 四类互补目标。

MUST RUN：

```text
T1: ExtraTrees raw
T2: ExtraTrees interval_residual
T3: ExtraTrees rank_target
T4: HGB raw
T5: HGB interval_residual
T6: HGB rank_target
T7: Ridge raw
T8: Ridge residual
T9: Ridge rank_target
T10: common component model
```

目标定义：

```text
y_raw = fret12

y_resid = fret12 - mean(fret12 | date, interval), computed only inside training fold

y_rank = 2 * percentile_rank(fret12 within date + interval) - 1

y_common = mean(fret12 | date, interval), training label only; prediction must use features
```

融合网格：

```text
pred = a * pred_raw + b * pred_resid + c * pred_rank + d * pred_common

Allowed ranges:
a ∈ {0.20, 0.30, 0.40}
b ∈ {0.30, 0.40, 0.50}
c ∈ {0.10, 0.20, 0.30}
d ∈ {0.00, 0.05, 0.10, 0.15, 0.20}
a + b + c + d = 1
```

验收：

```text
rank branch 若单独 corr 不高，但能提升 blend 的 rolling mean，可保留。
common branch 若改善 MSE/R² 且 corr 不降，可保留。
```

---

### Stage 2：RevIN-style / regime-aware normalization features

目标：处理金融非平稳，降低 rolling 方差。

只对核心特征做归一化，不要对所有特征暴力扩张。

核心特征清单：

```text
mid_return_1 / 3 / 6 / 12
spread
book_imbalance_1
book_imbalance_5
trade_imbalance_qty
trade_imbalance_count
cancel_imbalance_qty
vwad_gap
activity
rolling_volatility
```

新增特征：

```text
rolling_z_12
rolling_z_24
rolling_z_60
cross_section_z
cross_section_rank
regime_group_z, optional
```

实验：

```text
N1: Core
N2: Core + rolling_z
N3: Core + cross_section_z
N4: Core + rolling_z + cross_section_z
N5: Full
N6: Full + selected normalized features
```

验收：

```text
若 normalized features 提高 rolling mean 或降低 rolling std，则保留。
若只提高 holdout 但 rolling 变差，则删除。
```

---

### Stage 3：Patch summary features

目标：借鉴 PatchTST 的局部片段思想，但仍用稳健 tabular 模型。

窗口：

```text
history_window = 60 intervals
patch_sizes = {6, 12, 24, 60}
```

每个 patch 计算：

```text
mean
std
last
slope
max_minus_min
last_minus_first
```

仅对核心特征生成 patch summary。禁止对所有 100+ 特征生成 patch，以免维度爆炸。

实验：

```text
P1: backbone + patch_6
P2: backbone + patch_12
P3: backbone + patch_24
P4: backbone + patch_multi
```

验收：

```text
若 patch_multi 维度太高且 rolling 不稳，优先保留 patch_12 或 patch_24。
```

---

### Stage 4：MLPLOB-lite / ConvLOB-lite 轻量序列专家

目标：提供与树模型不同的误差结构，而不是替代主模型。

MAY RUN：只有在 Stage 0~3 已经稳定后才执行。

输入：

```text
window = 30 or 60 intervals
features = 16~32 core normalized features
target = residual or rank_target
```

推荐结构 A：MLPLOB-lite

```text
Flatten
Dense(128)
ReLU
Dropout(0.2)
Dense(64)
ReLU
Dropout(0.1)
Output(1)
Loss = Huber or MSE
Early stopping on rolling validation fold
```

推荐结构 B：ConvLOB-lite

```text
Conv1D(filters=32, kernel_size=3)
ReLU
Conv1D(filters=32, kernel_size=5)
ReLU
GlobalAveragePooling1D
Dense(64)
Output(1)
```

FORBIDDEN：当前阶段不做完整 Transformer、TFT、PatchTST、DeepLOB 大模型、复杂 LSTM 堆叠。  
MUST：如果 PyTorch/TensorFlow 不可用，最终 `meow.py` 自动跳过该 expert。

入选条件：

```text
sequence expert 单独 rolling 不崩；
或者与 tree backbone 预测相关性较低，并在受限融合中稳定提升。
```

---

### Stage 5：受限融合

目标：防止融合权重退化为不稳定 expert。

候选专家：

```text
stable_backbone_raw
stable_backbone_residual
stable_backbone_rank
common_component
patch_summary_expert
optional_sequence_expert
old_soft_regime_expert
```

权重规则：

```text
stable_backbone total weight >= 0.50
rank/residual total weight <= 0.35
common weight <= 0.20
patch expert weight <= 0.20
sequence expert weight <= 0.15
soft_regime weight <= 0.15
```

默认网格：

```text
final_pred =
    w_backbone * pred_backbone
  + w_resid_rank * pred_resid_rank
  + w_common * pred_common
  + w_patch * pred_patch
  + w_seq * pred_sequence
  + w_soft * pred_soft_regime
```

FORBIDDEN：

```text
w_soft_regime = 1.0
w_sequence > 0.15
unconstrained Ridge stacking without OOF
```

验收：

```text
选择 stability_score 最高且 MSE/R² 不崩的融合。
若最佳融合不如 backbone 稳，提交 backbone。
```

---

### Stage 6：后处理

默认只允许轻微 clipping：

```text
clip 0.5% / 99.5%
clip 1.0% / 99.0%
```

MAY：如果 rolling 所有折都提升，可以保留 rank blend。

FORBIDDEN：复杂后处理作为主增益来源。  
FORBIDDEN：在 final holdout 上反复调后处理参数。

---

## 7. Agent 因子挖掘闭环

老师提示可以做 Agent 自动因子挖掘。后续 AI 如要写进报告，必须形成完整闭环。

MUST：保存以下材料：

```text
agent_prompt.md
agent_candidate_factors.csv
factor_ic_report.csv
selected_agent_factors.txt
```

执行流程：

1. Agent 根据字段含义生成 30~50 个候选因子。
2. 人工或规则过滤泄漏因子、重复因子、不可实现因子。
3. 计算 rolling IC：

```text
mean_ic
std_ic
positive_ratio
fold_ic_1 / fold_ic_2 / fold_ic_3
```

4. 只保留：

```text
abs(mean_ic) >= 0.003
positive_ratio >= 0.60
加入 backbone 后 rolling 不下降
```

FORBIDDEN：报告里写“用了 Agent 挖掘因子”，但没有候选因子表和筛选结果。

---

## 8. 实验优先级

### Priority A：必须完成

```text
A1: purged rolling baseline
A2: raw / residual / rank / common 四目标分支
A3: 受限融合
A4: 三指标完整记录
A5: 泄漏审计
A6: 最终 python meow.py 一键运行
```

### Priority B：强烈建议完成

```text
B1: RevIN-style normalized features
B2: patch summary features
B3: Agent factor mining closed loop
B4: soft regime 限权融合
```

### Priority C：有时间再做

```text
C1: MLPLOB-lite
C2: ConvLOB-lite
C3: optional LightGBM expert
```

### Priority D：不建议做

```text
D1: 完整 DeepLOB
D2: 完整 PatchTST / Transformer
D3: 继续加复杂 soft gating
D4: 大规模 postprocess 网格
D5: 无约束 stacking
```

---

## 9. 具体命令建议

后续 AI 可以设计如下入口，保持实验清晰：

```bash
python meow.py --mode v33_audit
python meow.py --mode v33_roll_baseline
python meow.py --mode v33_targets
python meow.py --mode v33_norm_features
python meow.py --mode v33_patch_features
python meow.py --mode v33_blend
python meow.py --mode v33_final
```

最终提交时只保留：

```bash
python meow.py
```

`python meow.py` 应自动运行最终已选方案，而不是跑全部实验。

---

## 10. 结果记录模板

每次实验必须写入 `实验记录.md` 或 `results/v33_results.csv`。

| run_id | date | stage | model | target | features | split | corr | mse | r2 | daily_corr_mean | daily_corr_std | rolling_mean | rolling_std | rolling_min | stability_score | selected | notes |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|

实验结论必须写“保留/删除/待确认”，不能只留数字。

---

## 11. 最终模型选择规则

最终模型按下面顺序选择：

1. 先排除任何有泄漏风险的模型。
2. 再排除无法 `python meow.py` 稳定运行的模型。
3. 在剩余模型中按 `stability_score` 排名。
4. 若最高 `corr` 模型的 `MSE/R²` 明显变差，则降级到三指标更平衡模型。
5. 若冲 10% 模型只在单个 holdout 高，rolling 不稳，不提交。
6. 若所有融合都不如 backbone 稳，提交 backbone。

推荐最终选择指标：

```text
selection_score =
    0.50 * rank(rolling_corr_mean)
  + 0.20 * rank(-rolling_corr_std)
  + 0.15 * rank(-mse)
  + 0.15 * rank(r2)
```

---

## 12. 最终报告主线

报告不要写成“我们试了很多模型”，而要写成如下逻辑：

```text
金融分钟数据存在噪声、非平稳、长尾、横截面共同波动。
因此我们没有单纯追求复杂模型，而是构建：
    1. 微观结构特征体系
    2. raw/residual/rank/common 多目标分解
    3. RevIN-style 非平稳归一化
    4. patch summary 序列局部模式特征
    5. 轻量序列专家探索
    6. 受限融合防止过拟合
最终通过 purged rolling validation 选择稳定模型。
```

创新点表述：

| 创新点 | 说明 | 是否进入最终模型 |
|---|---|---|
| Agent 因子挖掘 | 自动生成候选微观结构因子，并用 rolling IC 筛选 | 视实验结果 |
| 多目标分解 | raw / residual / rank / common 分支兼顾 corr 与 MSE/R² | 是 |
| 非平稳归一化 | 借鉴 RevIN / Non-stationary 思想，构造 rolling/cross-section z-score | 视 rolling 结果 |
| Patch summary | 借鉴 PatchTST，把历史窗口压缩为局部统计特征 | 视 rolling 结果 |
| 轻量序列专家 | 借鉴 DeepLOB，但只做 MLPLOB-lite / ConvLOB-lite | 可作为探索 |
| 受限融合 | 防止 unstable expert 主导最终预测 | 是 |

---

## 13. 失败时降级方案

如果 V3.3 没有冲过 10%，也不要强行提交不稳模型。按下面顺序降级：

```text
Level 1: best constrained blend
Level 2: raw + residual + rank 三分支 blend
Level 3: stable ExtraTrees/HGB residual backbone
Level 4: Ridge/ElasticNet stable baseline + selected factors
```

最终宁可提交 `corr = 0.085~0.095` 且 rolling 稳定、MSE/R² 可解释的模型，也不要提交单点 `corr > 0.10` 但 rolling 崩坏、R² 极差的模型。

---

## 14. 最终交付清单

代码：

```text
meow.py
feature_engineering.py
validation.py
models.py
blend.py
config.py
```

实验记录：

```text
实验记录.md
results/v33_results.csv
factor_ic_report.csv
```

报告材料：

```text
项目报告.pdf / docx
方案汇报.pptx
成员分工表
Agent 因子挖掘附录
论文启发方法说明
```

提交检查：

```text
[ ] python meow.py 可运行
[ ] 不依赖本地缓存
[ ] 不提交数据
[ ] 不提交模型缓存
[ ] 无验证/测试 y 泄漏
[ ] 三指标均记录
[ ] rolling 结果可复现
[ ] 报告解释失败实验
[ ] 成员贡献明确
```

---

## 15. 参考文献与方案依据

1. DeepLOB: Deep Convolutional Neural Networks for Limit Order Books.  
   用于启发轻量 LOB-style sequence expert，而非直接照搬完整模型。

2. PatchTST: A Time Series is Worth 64 Words.  
   用于启发 patch summary features，而非直接上完整 Transformer。

3. RevIN: Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift.  
   用于启发 rolling / cross-section normalization features。

4. Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting.  
   用于支持“非平稳处理 + 避免过度 stationarization”的思路。

5. Purged / Embargoed Cross Validation in Financial Machine Learning.  
   用于支持 `fret12` 这种未来窗口标签下的 purged rolling validation。

---

## 16. 一句话执行摘要

V3.3 不是继续堆 `soft regime`，而是把实验主线改成：

```text
purged rolling validation
+ stable tabular backbone
+ raw/residual/rank/common 多目标分支
+ RevIN-style 非平稳归一化
+ patch summary 局部序列特征
+ optional MLPLOB-lite
+ constrained blend
```

最终目标仍然是冲击 `corr > 0.10`，但任何超过 10% 的结果只有在无泄漏、rolling 稳定、MSE/R² 不崩、代码可一键运行时，才允许作为最终提交模型。
