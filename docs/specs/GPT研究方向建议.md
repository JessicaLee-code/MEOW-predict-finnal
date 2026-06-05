# GPT 研究方向建议（2026-05-22）

来源：项目交接时 GPT 给出的整合判断，已作为后续工作的指导基础。

## 核心结论

当前真正要突破的不是模型复杂度，而是：

> 找到比"静态盘口不平衡"更稳定的新信号，并用更严格的 rolling 证明它不是偶然有效。

## 历史方法分层回顾

### 第一层：原始 baseline
`少量盘口/成交特征 + Ridge`，跑通流程，上限低。

### 第二层：扩展微观结构特征
加入盘口不平衡、成交不平衡、撤单压力、lag、rolling、cross rank 等。
有效信号：`trade_imb / ob_imb / order_pressure / trade_imbema5 / lagret12`

### 第三层：目标变换
`interval_demean` 明显优于直接回归原始 `fret12`。
残差分解可以试，但必须防止前视泄漏。

### 第四层：复杂模型和融合
大多数复杂模型（ExtraTrees、soft regime、fusion）在单次验证上看起来高，
但 rolling 后稳定性不够。当前最可信结果仍是 Ridge backbone。

## 主要缺陷

1. **单次验证误导方向**：必须扩展 rolling（已列为 P0 最高优先）
2. **复杂模型过拟合**：train corr 高，val corr 不稳，rolling std 大
3. **特征同质化**：大量特征都在表达"买卖盘谁强"，新增信息有限
4. **目标变换有用但风险高**：必须防止前视泄漏
5. **缺乏系统信号体检**：没有按月份/市场状态检验特征的跨期稳定性

## 推荐实验路线（P0–P5）

```
P0：扩展 rolling 评测 + 训练窗口敏感性（最高优先）
P1：OFI 动态订单流（已有实现，需 rolling 验证）
P2：成交冲击 trade impact（已有实现，需 rolling 验证）
P3：条件动量 / 条件反转
P4：稳健模型比较（Ridge → ElasticNet → 浅树）
P5：受限融合
```

### P1：OFI 动态订单流

普通盘口不平衡问"现在买卖盘谁多"，OFI 问"这段时间里买卖盘力量怎么变化"。
关键特征：`bid_ofi / ask_ofi / ofi_total / ofi_div_depth / ofi_div_turnover`

### P2：成交冲击

```
trade_pressure = (主动买 - 主动卖) / 总成交量
trade_intensity = 成交笔数
avg_trade_size  = 平均每笔大小
trade_pressure * spread
trade_pressure * depth
trade_pressure * volatility
```

### P3：条件动量 / 条件反转

```
lagret1 / lagret3 / lagret6 / lagret12 / lagret24
lagret * trade_pressure
lagret * OFI
lagret * spread / volatility / depth_imbalance
```

### 模型策略

```
第一优先：Ridge / ElasticNet / HuberRegressor
第二优先：浅 ExtraTrees / 浅 HistGB
禁止：Transformer / LSTM / 复杂 MLP / 自由 stacking
```

新信号必须在 Ridge 验证有效后，再考虑浅树。

## 重要原则

- 业界看 IC/corr 的跨期稳定性，不看单日最高值
- 所有 rolling/EMA/zscore 只用历史信息
- 非平稳归一化：rolling zscore / cross-section rank / stock-level z
- DeepLOB 借思想：用历史 N 个 interval 的盘口/成交做序列摘要，落地成简单的 mean/std/slope/EMA，不要照搬重型 Transformer
