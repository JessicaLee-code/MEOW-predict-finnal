# MEOW 金融时序预测 V3.1 实验计划：AI 执行验收版

版本：V3.1  
日期：2026-05-14  
目标读者：后续接手本项目的 AI Agent、实验执行器、代码生成模型、报告整理模型。  
文档定位：本文件不是泛泛的实验设想，而是一份**可执行、可审计、可答辩**的实验协议。后续 AI 必须优先满足老师项目要求，其次再冲击 Pearson Correlation > 0.10。

---

## 0. 最重要结论

本项目最终目标不是“预测 residual”，也不是“只优化 Pearson Correlation”，而是：

```text
输入：老师给定的订单簿、成交、撤单等金融分钟级数据
输出：对原始 fret12 的预测值
评价：MSE、Pearson Correlation、R² 综合评价
冲刺目标：在逻辑严谨、无泄漏、可复现的前提下，使 Pearson Correlation 尽量超过 0.10
```

V3.1 的最终推荐主线为：

```text
Full microstructure features
+ interval residual / interval_demean target
+ common component reconstruction
+ regime features / regime-specific experts
+ sklearn-compatible tree models
+ OOF multi-expert fusion
+ corr-oriented but leakage-free postprocess
```

后续 AI 必须遵守以下总原则：

```text
MUST：最终提交预测原始 fret12。
MUST：所有实验同时记录 MSE、Pearson Corr、R²。
MUST：所有目标变换、特征选择、融合、后处理均不得使用验证集/测试集真实 y 信息。
MUST：最终代码能通过 python meow.py 一键运行。
SHOULD：优先冲击 Pearson Corr > 0.10。
SHOULD：用 rolling validation 证明提升不是单一切分偶然结果。
MAY：使用 LightGBM 等可选库，但必须提供 sklearn-only fallback。
FORBIDDEN：为了冲 10% 使用验证集/测试集真实 fret12 的 group mean、ranking、scale、residual 或任何统计量。
```

---

## 1. 老师要求对齐清单

后续 AI 在执行任何实验前，必须先检查本节。若实验设计与本节冲突，以本节为准。

### 1.1 预测目标

老师要求预测 `fret12`，即 12 分钟后的 forward return。项目中的所有中间目标只能作为训练技巧，不能改变最终提交目标。

允许：

```text
训练 residual target
训练 interval_demean target
训练 common component model
训练 regime-specific model
最终 pred_final = pred_residual + λ * pred_common
```

禁止：

```text
最终只提交 residual_pred
最终只提交 interval_demeaned_pred
使用验证/测试真实 y 的 interval mean 还原预测
把预测任务改成分类、rank、方向预测后不还原到 fret12
```

### 1.2 评价指标

老师最终预测评价包含：

```text
MSE
Pearson Correlation
R²
```

因此，后续 AI 不能只报告 `corr`。所有实验日志必须至少包含：

```text
experiment_id
feature_set
target_type
model_type
postprocess_type
train_corr
val_corr
train_mse
val_mse
train_r2
val_r2
daily_corr_mean
daily_corr_std
train_val_corr_gap
runtime_sec
random_seed
notes
```

模型选择的优先级如下：

```text
第一目标：Pearson Corr 尽量超过 0.10。
第二约束：R² 不能明显劣于强 baseline。
第三约束：MSE 不能明显劣于强 baseline。
第四约束：rolling validation 中提升必须有稳定性。
```

建议用于最终筛选的内部综合分数：

```text
internal_selection_score = 0.50 * rank(val_corr)
                         + 0.25 * rank(-val_mse)
                         + 0.25 * rank(val_r2)
```

说明：冲刺阶段可以优先看 `val_corr`；最终提交阶段必须同时参考三指标。

### 1.3 报告要求

最终报告必须能回答老师关心的以下问题：

```text
1. 原始数据有什么金融时序特点？
2. baseline 怎么复现？
3. 每类特征为什么有金融含义？
4. 为什么选择当前模型？
5. 尝试过哪些失败/不稳定方案？
6. 如何防止过拟合和目标泄漏？
7. 最终模型的 MSE、Corr、R² 表现如何？
8. 每位成员是否参与了完整项目流程？
9. 是否有大模型/Agent 辅助因子挖掘或实验设计的创新点？
```

---

## 2. 数据与时间切分协议

### 2.1 数据使用原则

后续 AI 只能使用当前时刻及历史可获得信息构造特征。目标 `fret12` 可以用于训练标签，但不能泄漏到验证/测试特征或后处理。

允许使用：

```text
当前 interval 的订单簿状态
当前及过去 interval 的成交、撤单、价格、盘口统计
同一时刻横截面上的特征统计量
历史滚动统计量
训练集 y 构造的训练标签变换
```

禁止使用：

```text
未来 midpx
未来成交量
未来盘口状态
验证/测试集真实 fret12 的均值、方差、rank、demean 统计量
全数据 fit 的 scaler / imputer / feature selector
全数据训练的 stacking meta-model
```

### 2.2 时间切分

必须使用时间序列切分，不能随机打散。

建议采用三层验证：

```text
Quick Split：快速调试和筛选候选模型
Rolling Splits：判断模块是否稳定有效
Final Holdout：最终确认，只允许评估，不允许再调参
```

推荐格式：

```text
Split A: train dates 1-8,  val dates 9-10
Split B: train dates 2-9,  val dates 10-11
Split C: train dates 3-10, val dates 11-12
...
Final Holdout: 最后一段日期，仅用于最终模型确认
```

如果数据天数不足，后续 AI 可以缩短窗口，但必须保证：

```text
train 时间早于 val 时间
val 时间早于 final holdout 时间
不能随机交叉验证
不能把同一天的未来样本泄漏给过去样本
```

---

## 3. 目标构造与防泄漏规则

本节是 V3.1 最关键的合规部分。后续 AI 必须严格执行。

### 3.1 Raw target

基础目标：

```text
y_raw = fret12
```

Raw target 模型直接预测原始 `fret12`。

用途：

```text
作为 baseline
作为 fusion expert
补充 residual 模型遗漏的市场共同项
```

### 3.2 Interval residual / interval_demean target

训练阶段可以定义：

```text
y_common_train(date, interval) = mean(y_raw_train | date, interval)
y_residual_train = y_raw_train - y_common_train(date, interval)
```

注意：`y_common_train` 只能在训练集内部计算。

验证/测试阶段禁止计算：

```text
y_common_val(date, interval) = mean(y_raw_val | date, interval)
y_common_test(date, interval) = mean(y_raw_test | date, interval)
```

验证/测试阶段只能使用模型预测 common component：

```text
pred_common = common_model.predict(X_common)
pred_residual = residual_model.predict(X_full)
pred_final = pred_residual + λ * pred_common
```

λ 必须在训练/rolling validation 中选择，不能用 final holdout 调参。

### 3.3 Common component model

common component model 的目标是预测同一 `date + interval` 的市场共同收益项。训练目标来自训练集：

```text
y_common_train = group_mean(y_raw_train by date, interval)
```

特征必须只来自当下或历史特征的横截面聚合，例如：

```text
market_mean_order_imbalance
market_mean_trade_imbalance
market_mean_cancel_imbalance
market_mean_spread
market_mean_volatility
market_mean_return_lag_1/3/6/12
market_activity
interval_position
```

禁止将验证/测试真实 y 的任何统计量作为 common 特征。

### 3.4 Stacking / OOF fusion 防泄漏

多专家融合必须使用 out-of-fold prediction。

正确流程：

```text
For each rolling fold:
    Train base experts on fold_train
    Predict fold_val to generate OOF predictions
Train meta_model on all OOF predictions and corresponding y
For final prediction:
    Refit base experts on full training data
    Generate test predictions
    Apply trained meta_model
```

禁止流程：

```text
Train base experts on full training data
Predict same training data
Train meta_model on in-sample base predictions
Report inflated validation corr
```

### 3.5 Postprocess 防泄漏

允许的 postprocess：

```text
prediction clipping
rank blend based on predictions only
partial group neutralization based on predictions only
low-confidence shrinkage based on feature-derived confidence only
```

禁止的 postprocess：

```text
用验证/测试真实 y 计算最优 group mean
用验证/测试真实 y 计算 rank target
用 final holdout 反复调 clip quantile / alpha / λ
```

postprocess 参数必须在 Quick Split 或 Rolling Splits 中选择。Final Holdout 只能用于评估。

---

## 4. 特征工程计划

特征工程必须有金融解释，不能只为了堆维度。

### 4.1 必做特征组 F0：老师 baseline 特征

目的：复现基础模型，证明代码流程正确。

包含：

```text
midpx
spread
order imbalance
trade volume / turnover basic features
basic return lag
basic rolling volatility
```

具体字段根据数据实际列名映射。

### 4.2 必做特征组 F1：盘口微观结构特征

金融逻辑：盘口买卖压力和价格微结构会影响短期收益。

候选：

```text
microprice = (ask0 * bsize0 + bid0 * asize0) / (bsize0 + asize0)
microprice_gap = (microprice - midpx) / midpx
book_imbalance_1 = (bsize0 - asize0) / (bsize0 + asize0)
book_imbalance_5 = (sum_bsize_0_4 - sum_asize_0_4) / (sum_bsize_0_4 + sum_asize_0_4)
depth_ratio = sum_bsize_0_4 / sum_asize_0_4
spread = (ask0 - bid0) / midpx
relative_spread_rank
```

### 4.3 必做特征组 F2：成交主动性特征

金融逻辑：主动买入/主动卖出不平衡代表短期价格压力。

候选：

```text
trade_imbalance_qty = (tradeBuyQty - tradeSellQty) / (tradeBuyQty + tradeSellQty)
trade_imbalance_count = (nTradeBuy - nTradeSell) / (nTradeBuy + nTradeSell)
trade_imbalance_turnover = (tradeBuyTurnover - tradeSellTurnover) / (tradeBuyTurnover + tradeSellTurnover)
trade_activity = log1p(total_trade_qty or total_trade_turnover)
```

### 4.4 必做特征组 F3：撤单压力特征

金融逻辑：撤买单可能削弱支撑，撤卖单可能削弱抛压。

候选：

```text
cxl_imbalance_qty = (cxlBuyQty - cxlSellQty) / (cxlBuyQty + cxlSellQty)
cxl_imbalance_count = (nCxlBuy - nCxlSell) / (nCxlBuy + nCxlSell)
cxl_pressure_ratio = cancel_side_pressure / displayed_depth
```

### 4.5 必做特征组 F4：收益、波动与日内状态

金融逻辑：短期动量/反转依赖市场活跃度、波动率和日内阶段。

候选：

```text
return_lag_1, return_lag_3, return_lag_6, return_lag_12
rolling_vol_6, rolling_vol_12, rolling_vol_24, rolling_vol_48
rolling_mean_return_6/12/24
interval_position
morning/afternoon/session dummy if available
activity_rank_by_interval
volatility_rank_by_interval
```

### 4.6 必做特征组 F5：交互特征

金融逻辑：动量/反转信号在不同订单流状态下符号和强度可能不同。

候选：

```text
return_3 * trade_imbalance_qty
return_6 * book_imbalance_1
return_12 * spread_rank
return_12 * activity_rank
microprice_gap * trade_imbalance_qty
book_imbalance_5 * cxl_imbalance_qty
volatility_rank * abs(order_imbalance)
```

### 4.7 必做特征组 F6：Regime features

金融逻辑：金融分钟数据非平稳，不同市场状态下最优模型不同。

候选：

```text
market_activity_level
market_volatility_level
market_spread_level
market_imbalance_level
liquidity_regime
volatility_regime
trend_or_reversal_regime
```

实现方式可以先用规则分桶：

```text
low / medium / high volatility
low / medium / high activity
tight / normal / wide spread
```

如果时间允许，再尝试 KMeans/GMM 聚类，但聚类模型必须只在训练集 fit。

### 4.8 可选特征组 F7：Agent-generated factors

若报告中要写“大模型/Agent 辅助因子挖掘”，后续 AI 必须生成可审计证据。

最低要求：

```text
1. 记录用于生成候选因子的 prompt 摘要
2. 生成至少 30 个候选因子
3. 人工或规则过滤明显泄漏因子
4. 用 rolling IC 筛选因子
5. 报告展示候选类别、IC 表、最终保留因子
```

筛选标准建议：

```text
mean_IC > 0.005
IC 同号比例 > 60%
加入主模型后 rolling val corr 不下降
```

如果没有完成上述闭环，则报告中只能写“尝试使用大模型辅助生成候选因子”，不能夸大为核心创新。

---

## 5. 模型实验计划

### 5.1 模型依赖原则

最终提交代码必须能在基础 Python + sklearn 环境下运行。

必须支持：

```text
Ridge / ElasticNet
ExtraTreesRegressor
RandomForestRegressor if needed
HistGradientBoostingRegressor
```

可选支持：

```text
LightGBM
XGBoost
CatBoost
```

若可选库不可用，程序必须自动 fallback：

```text
try LightGBM:
    use LGBMRegressor expert
except ImportError:
    use HistGradientBoostingRegressor expert
```

禁止最终代码强依赖不可控第三方库。

### 5.2 必做实验矩阵

#### E00：Baseline 复现

```text
feature_set = F0
model = Ridge
 target = raw fret12
purpose = 验证数据读取、切分、指标、提交格式正确
```

必须输出：

```text
MSE, Corr, R²
```

#### E01：Full features + Ridge

```text
feature_set = F0 + F1 + F2 + F3 + F4 + F5
model = Ridge / ElasticNet
 target = raw fret12
purpose = 判断线性可解释因子上限
```

#### E02：Full features + Tree

```text
feature_set = F0 + F1 + F2 + F3 + F4 + F5
model = ExtraTrees / HistGradientBoosting
 target = raw fret12
purpose = 建立强 tabular baseline
```

#### E03：Interval residual target

```text
feature_set = full features
model = ExtraTrees / HGB / Ridge
 target = interval residual
prediction = residual_pred only for comparison, not final submission
purpose = 验证横截面 residual 是否提高 corr
```

注意：E03 的 residual-only 结果可以作为实验对比，但不能直接作为最终提交。

#### E04：Common + residual reconstruction

```text
residual_model = Tree or Ridge on full features
common_model = Ridge / HGB on market aggregate features
prediction = residual_pred + λ * common_pred
λ ∈ {0.1, 0.2, 0.3, 0.5, 0.8, 1.0}
purpose = 在提升 corr 的同时修复 MSE / R²
```

λ 在 rolling validation 中选择，不能在 final holdout 调。

#### E05：Regime features

```text
feature_set = full features + regime features
model = ExtraTrees / HGB
 target = raw or residual
purpose = 验证非平稳状态特征是否稳定增益
```

#### E06：Regime-specific experts

```text
Train separate experts for volatility/activity/liquidity regimes
Combine predictions by hard rule or soft probability
purpose = 让不同市场状态使用不同预测器
```

若 hard regime 不稳定，优先保留 regime features，而不是强行保留 regime expert。

#### E07：OOF multi-expert fusion

专家池建议：

```text
expert_1 = ExtraTrees raw target
expert_2 = HGB raw target
expert_3 = ExtraTrees residual target
expert_4 = Ridge residual target
expert_5 = common + residual model
expert_6 = regime feature model
optional_expert_7 = LightGBM if available
```

融合器：

```text
RidgeCV / ElasticNetCV / constrained linear blend
```

目标：

```text
利用不同模型误差相关性较低的特点，冲击 corr > 0.10
```

必须使用 OOF predictions，禁止 in-sample stacking。

#### E08：Corr-oriented postprocess

候选后处理：

```text
P1: clip prediction at 0.1/99.9, 0.5/99.5, 1/99 quantiles
P2: rank-normalize prediction within date + interval
P3: final = α * pred + (1 - α) * rank_pred, α ∈ {0.6, 0.7, 0.8, 0.9}
P4: partial group neutralization: final = α * pred + (1 - α) * group_demeaned_pred
P5: low-confidence shrinkage based on spread/activity/volatility features
```

注意：postprocess 只能使用预测值和当前/历史特征，不能使用真实 y。

### 5.3 可选探索实验

以下实验只作为报告探索，不作为第一优先级：

```text
DeepLOB-lite
Patch Transformer
LSTM/GRU sequence model
Learning-based gating network
Complex residual stacking
```

保留条件：

```text
rolling validation 平均 corr 明显高于主线
R²/MSE 不崩
训练稳定
代码复杂度可控
```

如果不满足，则只写入“探索但未采用”的报告部分。

---

## 6. 冲击 Pearson Corr > 0.10 的执行策略

后续 AI 必须按以下优先级冲刺，不要重新发散到大规模复杂模型。

### 6.1 第一优先级：多专家 OOF 融合

最现实破 10 路线：

```text
ExtraTrees raw
HGB raw
ExtraTrees residual
Ridge residual
common + residual
regime feature model
optional LightGBM
=> OOF Ridge / ElasticNet fusion
```

预期增益来源：

```text
单模型 corr 接近 0.095 时，低相关误差融合可能提供 +0.003 ~ +0.008
```

### 6.2 第二优先级：common + residual

原因：

```text
residual target 往往提升横截面排序，即提升 corr
但可能损失原始 y 的共同市场项，影响 MSE/R²
common + residual 能兼顾二者
```

重点实验：

```text
pred_final = pred_residual + λ * pred_common
```

### 6.3 第三优先级：postprocess 网格

优先尝试：

```text
clip + rank blend
clip + partial group neutralization
low-confidence shrinkage
```

示例：

```text
pred_clip = clip(pred, q0.005, q0.995)
pred_rank = rank_normalize(pred_clip within date + interval)
pred_final = 0.8 * pred_clip + 0.2 * pred_rank
```

注意：参数必须来自 rolling validation。

### 6.4 第四优先级：sample weighting

目标：让模型更重视高信噪比样本。

候选权重：

```text
W0 = 1
W1 = 1 + a * abs(order_imbalance_rank)
W2 = 1 + b * trade_activity_rank
W3 = 1 + c * volatility_rank
W4 = W2 but downweight extremely wide spread samples
W5 = W1 + W2 + W4 combined
```

权重也必须通过 rolling validation 判断是否保留。

### 6.5 第五优先级：轻量多尺度树模型

如果前四项还未超过 0.10，再做：

```text
short-window expert: 1/3/6 min features
mid-window expert: 12/24 min features
long-window expert: 48/60 min features
fusion = linear blend or OOF Ridge
```

不要优先做重 Transformer。

---

## 7. 实验保留与淘汰规则

后续 AI 必须用规则判断模块是否进入最终方案。

### 7.1 保留条件

一个模块可以进入最终模型，需要满足至少以下条件：

```text
1. quick split val_corr 有提升；
2. rolling validation 平均 val_corr 有提升；
3. 至少 60% rolling splits 不下降；
4. val_mse 和 val_r2 不明显恶化；
5. 没有任何目标泄漏；
6. 代码复杂度可控，可在 meow.py 中复现。
```

### 7.2 淘汰条件

出现以下任一情况，应淘汰或降级为探索实验：

```text
train_corr 大幅提升但 val_corr 不升或下降
只在单一 split 提升
R² 明显恶化
MSE 明显恶化
需要不可控第三方库且无 fallback
存在潜在 y leakage
运行时间无法接受
难以在报告中解释金融逻辑
```

### 7.3 最终模型选择规则

最终模型不能只选最高单点 corr，而应按：

```text
1. 无泄漏
2. 可一键运行
3. rolling validation 稳定
4. final holdout 表现合理
5. corr 尽可能高
6. MSE/R² 不崩
```

如果两个模型 corr 接近，优先选择：

```text
R² 更高
MSE 更低
结构更简单
rolling 方差更小
代码更稳
```

---

## 8. 代码实现要求

最终提交必须包含 `meow.py`，老师更改数据路径后能够运行：

```bash
python meow.py
```

### 8.1 推荐代码结构

可以是单文件，也可以是模块化文件。若是模块化，`meow.py` 必须作为唯一入口。

推荐结构：

```text
meow.py
src/
  config.py
  data_io.py
  features.py
  targets.py
  models.py
  validation.py
  fusion.py
  postprocess.py
  metrics.py
  report_utils.py
outputs/
  experiments.csv
  predictions.csv
  model_config.json
```

如果老师只允许提交单文件，则将核心逻辑合并到 `meow.py`。

### 8.2 Pipeline 顺序

后续 AI 应按此顺序实现：

```text
1. Load data
2. Sort by date, stock, interval/time
3. Build features using only current and past information
4. Split by time
5. Fit imputers/scalers/encoders on train only
6. Build train targets
7. Train base experts
8. Generate OOF predictions
9. Train fusion model
10. Fit final experts on full training data
11. Predict validation/test
12. Apply fixed postprocess
13. Output raw fret12-scale prediction
14. Log all metrics and configs
```

### 8.3 Determinism

必须设置随机种子：

```text
random_state = 42 or fixed project seed
```

所有实验必须记录 seed。

### 8.4 Missing value handling

缺失值处理必须只在训练集 fit。

允许：

```text
median imputation fit on train
constant imputation fit on train
model-native missing handling if supported
```

禁止：

```text
用全数据 median / mean 填充
用验证/测试分布调整 train scaler
```

---

## 9. 最终报告结构建议

最终报告应按以下逻辑组织，避免显得像“盲目堆模型”。

### 9.1 摘要

说明：

```text
项目目标：预测 fret12
核心难点：噪声、非平稳、长尾、短周期信号弱
最终方案：微观结构特征 + 目标分解 + regime + OOF 融合 + 防泄漏后处理
最终指标：MSE / Corr / R²
```

### 9.2 数据分析

必须展示：

```text
fret12 分布
不同日期/interval 的波动变化
收益长尾
盘口不平衡分布
成交活跃度分布
corr/IC 初步分析
```

### 9.3 特征工程

按金融逻辑分组说明：

```text
盘口压力
成交主动性
撤单压力
波动率与日内状态
动量/反转交互
regime features
common aggregate features
```

### 9.4 模型设计

说明为什么最终使用树模型和融合：

```text
树模型适合非线性特征交互
Ridge 提供低方差稳定专家
residual target 提升横截面排序
common model 修复原始目标尺度
regime model 处理非平稳
OOF fusion 降低单模型误差
```

### 9.5 实验结果

必须包含表格：

```text
Baseline Ridge
Full Ridge
Full Tree
Residual Tree
Common + Residual
Regime Features
OOF Fusion
Postprocess
Final Model
```

每行至少包含：

```text
MSE
Corr
R²
daily_corr_mean
daily_corr_std
```

### 9.6 消融实验

必须包含：

```text
no book features
no trade features
no cancel features
no lag/rolling features
no regime features
raw target vs residual target vs common+residual
single model vs OOF fusion
without postprocess vs with postprocess
```

### 9.7 失败实验

可以写：

```text
MLP / DeepLOB / Transformer 训练集提升但验证集不稳定
复杂 gating 容易过拟合
过多自动生成因子会引入噪声
residual-only 虽提升 corr 但可能损害 MSE/R²
```

失败实验不丢分，反而体现探索度。

### 9.8 防泄漏说明

必须单独成节，说明：

```text
所有 split 按时间划分
scaler/imputer/feature selector 只在 train fit
interval residual 只用 train y 构造
common component 在 val/test 由模型预测
stacking 使用 OOF
postprocess 不使用真实 y
final holdout 不参与调参
```

### 9.9 成员贡献

成员分工不能写成单一流水线。必须体现每个人参与完整项目。

推荐格式：

```text
成员 A：主负责 EDA 与 baseline；共同参与特征讨论、消融分析、报告审阅。
成员 B：主负责盘口/成交/撤单特征；共同参与模型训练和结果解释。
成员 C：主负责目标分解与防泄漏审计；共同参与 baseline、特征筛选和报告撰写。
成员 D：主负责模型训练、OOF 融合和后处理；共同参与 EDA 与最终代码复现。
成员 E：主负责实验日志、可视化和报告整合；共同参与 postprocess、代码验收和答辩材料。
```

---

## 10. AI 执行 Checklist

后续 AI 每完成一个阶段，必须检查以下列表。

### 10.1 合规 Checklist

```text
[ ] 最终输出是原始 fret12 尺度预测。
[ ] 没有使用验证/测试真实 y 的 group mean。
[ ] 没有使用验证/测试真实 y 做 postprocess。
[ ] scaler/imputer/selector 只在 train fit。
[ ] stacking 使用 OOF predictions。
[ ] final holdout 没有参与调参。
[ ] 所有实验记录 MSE、Corr、R²。
[ ] meow.py 可以一键运行。
[ ] LightGBM 等外部库有 fallback。
```

### 10.2 冲 10 Checklist

```text
[ ] ExtraTrees raw 已跑。
[ ] HGB raw 已跑。
[ ] residual target 已跑。
[ ] common + residual 已跑。
[ ] regime features 已跑。
[ ] OOF multi-expert fusion 已跑。
[ ] clip / rank blend / partial neutralization 已跑。
[ ] sample weighting 至少测试 2-3 版。
[ ] rolling validation 已确认不是单 split 偶然提升。
```

### 10.3 报告 Checklist

```text
[ ] 有 EDA 图或表。
[ ] 有 baseline。
[ ] 有特征工程解释。
[ ] 有模型选择理由。
[ ] 有完整实验表。
[ ] 有消融实验。
[ ] 有失败实验。
[ ] 有防泄漏说明。
[ ] 有成员贡献说明。
[ ] 有最终代码运行说明。
```

---

## 11. 最终推荐执行顺序

不要从最复杂模型开始。按以下顺序执行：

```text
Step 1: 复现 baseline，确认数据、指标、切分无误。
Step 2: 构造 full microstructure features。
Step 3: 训练 Ridge / ExtraTrees / HGB raw target。
Step 4: 训练 interval residual target 模型。
Step 5: 训练 common component model，并做 common + residual reconstruction。
Step 6: 加入 regime features，判断是否稳定提升。
Step 7: 生成 OOF predictions，训练 fusion model。
Step 8: 对 fusion prediction 做固定参数 postprocess。
Step 9: rolling validation 复核。
Step 10: final holdout 只评估，不再调参。
Step 11: 固化 meow.py、实验日志、模型参数和最终报告。
```

---

## 12. 最终模型建议

若时间有限，优先提交以下稳健版本：

```text
features = F0 + F1 + F2 + F3 + F4 + F5 + F6
experts = [
    ExtraTrees(raw),
    HistGradientBoosting(raw),
    ExtraTrees(interval_residual),
    Ridge(interval_residual),
    CommonResidualModel,
    RegimeFeatureTree
]
fusion = Ridge/ElasticNet on OOF predictions
postprocess = clip + mild rank blend
prediction_scale = original fret12
```

推荐最终预测形式：

```text
pred_base = fusion_model.predict([
    pred_et_raw,
    pred_hgb_raw,
    pred_et_residual,
    pred_ridge_residual,
    pred_common_residual,
    pred_regime
])

pred_clip = clip(pred_base, selected_quantiles)
pred_rank = rank_normalize(pred_clip within date + interval, based on prediction only)
pred_final = α * pred_clip + (1 - α) * pred_rank
```

其中：

```text
α 建议从 0.7 / 0.8 / 0.9 中选择
clip quantile 建议从 0.005/0.995 或 0.01/0.99 中选择
所有参数必须由 rolling validation 确定
```

---

## 13. 禁止事项汇总

后续 AI 严禁执行以下操作：

```text
FORBIDDEN 1: 使用验证/测试真实 fret12 计算 interval mean 并加入预测。
FORBIDDEN 2: 对全数据 fit scaler、imputer、feature selector、PCA、cluster。
FORBIDDEN 3: 用 validation 反复调 postprocess 后再声称泛化有效。
FORBIDDEN 4: 只报告 corr，不报告 MSE 和 R²。
FORBIDDEN 5: 最终提交 residual-only prediction。
FORBIDDEN 6: 最终代码强依赖 LightGBM 且没有 fallback。
FORBIDDEN 7: 随机划分金融时序数据做交叉验证。
FORBIDDEN 8: 把复杂深度模型作为最终主线但没有 rolling validation 支撑。
FORBIDDEN 9: 在报告中夸大 Agent 因子挖掘，但没有候选因子表和筛选结果。
FORBIDDEN 10: 成员分工写成某人只负责报告、某人只负责跑代码，不能体现共同参与。
```

---

## 14. 最终验收标准

V3.1 的验收不是只看是否超过 10%。后续 AI 应按以下顺序验收：

```text
Level 0: 无泄漏，能运行，输出原始 fret12。
Level 1: baseline 复现，三指标完整。
Level 2: full feature tree 明显超过 baseline。
Level 3: residual/common/regime/fusion 至少一个模块稳定提升。
Level 4: rolling validation 下 corr 接近或超过 0.10，MSE/R² 不崩。
Level 5: final report 能完整解释特征、模型、消融、失败实验、防泄漏和贡献。
```

若最终 corr 未超过 0.10，但满足 Level 0-5 中除 Level 4 的过线条件，方案仍然是合规且高质量的。若 corr 超过 0.10 但存在泄漏或无法运行，则该结果无效。

---

## 15. 一句话给后续 AI

请优先做一个**无泄漏、可复现、输出原始 fret12、三指标完整**的强 tabular + residual/common + regime + OOF fusion 方案；在这个框架内用 postprocess 和模型融合冲击 Pearson Corr > 0.10。不要为了 10% 牺牲老师要求、代码可运行性或实验可信度。
