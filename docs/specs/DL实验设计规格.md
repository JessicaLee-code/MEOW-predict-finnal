# DL 实验设计规格

创建日期：2026-05-31
状态：设计定稿（D0 地基与截面卡带已落地；正式长跑前仍需 maxfold 内存演练）
适用范围：MEOW 金融时序预测 —— 深度学习（DL）主线（Windows 4060 / PyTorch）

> 本文是 DL 主线的**设计真相源**。规则正文（红线纪律、两速评测、宽进严出等通用口径）仍以 `AGENTS.md` 为准；本文只补 DL 特有的架构、协议、配置与地基约定。动态进度看 `CLAUDE.md`。

---

## 0. 一句话定位

把**评测协议 / 窗口切分 / 归一化 / 指标 / 配置管理**做成**不可变脊柱**，把**输入适配**与**模型本体**做成**可换卡带**；PyTorch 只在卡带内部存在，脊柱全程 torch-free。目标是**直接冲 0.12**（不做"序列是否有信号"的存在性验证——该结论已是先验），同时保留"不骗自己"的评测纪律。

---

## 1. 战略定位与目标

- **目标分**：直接冲 **0.12**（老师明示有人做到过 0.12、业界 0.2–0.3）。不浪费预算去"验证序列有没有料"。
- **保留的纪律**：评测要能拦住自欺——过拟合探测、时间外推稳定性、防泄漏、config-lock、红线 holdout（见 §5、§10）。
- **算力**：Windows 4060 / PyTorch；与传统侧（Mac）互不抢算力。预算口径已改为 `AGENTS.md §十一·11.6`「一命令两档」——一晚一个结构族 ≈25–30 次 fit / 7–15 小时，单卡一周可比完 2–3 个结构族。
- **与传统侧的关系**：评测体系、数据认知、9-stage 特征工程、提交通道**全可复用**；DL 只是把"模型"这一格换成序列模型 + 把"输入"这一格换成可换的 adapter。

---

## 2. 总架构：固定脊柱 + 可换卡带

### 2.1 控制层级（谁调谁、谁负责什么）

```
Orchestrator (experiments/run_dl.py)          ← 发起一次实验；组装+冻结 RunConfig；管两阶段交接
  │  对一份 RunConfig 启动一次 run
  ▼
HPO Searcher                                   ← 谁负责超参搜索
  │  采样 config / 早杀坏 trial / 海选→认证两段 / 种子重复
  ▼  (对每个候选 config 调一次)
Protocol Engine (src/dl_protocol.py)           ← 谁负责评测协议(窗口切分) + 把模型输出换成统一指标
  │  走查折日期 / 三段切分 / embargo / 跑 k 个种子 / 算 4 指标 / gap·曲线·稳定性判断
  ▼  (对每折调一次)
SequenceTrainer (src/dl_trainer.py)  [纯编排, torch-free]
  ├─ Data Pipeline
  │   ├─ Input Adapter   [可换]  ← 谁负责适配不同模型的 load + 预处理（含价格相对归一）
  │   ├─ Window Indexer  [不变]  ← 惰性出 [B, L, C]，不跨日 + 因果对齐 + warmup
  │   └─ Normalizer      [不变]  ← fit-on-train 的统计白化（可配置成 identity）
  └─ Model Cartridge (models/dl_models.py)  [可换 + 唯一 torch]  ← 谁对接 PyTorch
        fit() / predict() 内含 epoch 循环 / optimizer / GPU 放置 / nn.Module / 早停
```

**两个边界一句话记牢**：
- **横切：脊柱不变 / 卡带可换。** 中间一整列（HPO、协议、窗口、归一化、指标、leaderboard、配置）是脊柱，永不为换模型而改。
- **纵切：PyTorch 封死在卡带。** 所有 torch import / nn.Module / optimizer / device / checkpoint 只出现在 Model Cartridge 内；脊柱从不碰一个 tensor。

### 2.2 数据流

```
老师/本地 raw .h5（逐日）
   │  InputAdapter.build(raw_day_df)         [可换：定义通道含义]
   ▼
每日 [n_interval, C] 干净数值数组（通道布局固定、含义已知）
   │  Window Indexer                         [不变：纯时间索引]
   ▼
窗口批 [B, L, C]（不跨日、标签因果对齐在窗口末端、warmup 丢弃不足窗）
   │  Normalizer（fit-on-train 统计量 → 套全部）[不变：纯张量数学，可关]
   ▼
归一化后的 [B, L, C]  ──────────────►  Model Cartridge.fit/predict()   [可换 + 唯一 torch]
                                              │  predict → np[fret12 量纲, 与样本顺序对齐]
                                              ▼
                                        Protocol Engine 指标模块
                                              │  corr / MSE / R² / daily-IC
                                              ▼
                                        FoldResult（列与 leaderboard 兼容）
```

### 2.3 换模型时：什么不变、什么变

以 `LSTM-on-features`（433 手工特征当通道）→ `TCN-on-原始微结构`（~59 原始微结构通道，见 §8.0）为例：

| | 是否改动 | 说明 |
|---|---|---|
| **InputAdapter** | ✅ 变 | 通道定义 + 语义预处理（特征列 → 原始微结构各通道；价相对 mid / 量 log1p 在此做） |
| **ModelCartridge** | ✅ 变 | nn.Module 结构 + 训练循环（LSTM → TCN） |
| Window Indexer | ❌ 不变 | "不跨日、因果对齐、按 step 滑" 与通道含义无关 |
| Normalizer | ❌ 不变（机制） | 仍是 fit-on-train 白化；adapter 已做语义归一时，statistical 白化照样在脊柱跑（zscore），二者职责分明 |
| 协议（walk-forward / 三段 / embargo） | ❌ 不变 | |
| 4 指标 / leaderboard | ❌ 不变 | |
| HPO 搜索环 / 种子集成 | ❌ 不变 | |
| gap·曲线判过拟合 / walk-forward 稳定性 / config-lock | ❌ 不变 | |

---

## 3. 两个接口契约

### 3.1 InputAdapter（输入适配卡带）

唯一需要"理解原始数据语义"的地方。一旦它吐出通道布局固定的干净数组，下游全是不关心含义的张量数学。

```
class InputAdapter:
    channels: list[str]                     # 通道名（顺序即 C 维布局，固定）
    def build(self, raw_day_df) -> np.ndarray   # 返回单日 [n_interval, C]，float32
```

- **不跨日**：每次只吃"一天"的 raw，禁止把多日 raw 一次性喂进来（避免跨日串值，与 `SubmissionFeaturePipeline` 逐日现算口径一致）。
- **语义归一化归这里**：如原始价按当下 mid-price 做相对归一，属"语义预处理"，放 adapter；统计白化才交给脊柱 Normalizer。
- **两个实现（已落地，torch-free）**：
  - `FeatureAdapter`（D1 用）：包装现有 433 特征管线（`SubmissionFeaturePipeline` / `feature_registry`），把手工特征列当通道，零新特征公式。绑 `LSTM` 卡带。
  - `RawChannelAdapter`（D2 主攻 / TCN 用）：把 ~59 个原始微结构通道（§8.0）做**最小语义归一**当通道，让网络自己学交互：① 价（各档买卖价 / OHLC / 成交高低 / vwad）→ `p/midpx - 1`（行内，价=0 时置 0）；② `midpx` 本身 → 按 `(symbol)` 组内日内对数收益（`log midpx_t - log midpx_{t-1}`，首步 0，因果不跨日跨票）；③ 量 / 笔数 / 额 → `log1p` 驯厚尾。**刻意不在 adapter 里手搓 imbalance/OFI**——那是 FeatureAdapter 的活，走 raw 这条线就是要 TCN 自己学。绑 `TCN` 卡带。

### 3.2 ModelCartridge（模型卡带，唯一 torch）

```
class ModelCartridge:
    search_space: dict                      # 本模型私有的超参声明（不进 config/ 全局枚举）
    required_adapter: AdapterKind           # 类型化引用集中枚举，不写裸字符串
    def fit(self, train_ds, earlystop_ds, hparams, seed) -> TrainRecord
    def predict(self, ds) -> np.ndarray     # np[fret12 量纲, 与 ds 样本顺序对齐]
```

- `fit()` **内部跑完整个 epoch 循环 / optimizer / GPU / 早停**；早停看 `earlystop_ds`（训练区尾段切出，与打分集隔离），可用代理指标（如 val MSE）。
- `fit()` 返回 **TrainRecord**：逐 epoch 的 train / earlystop 曲线、best epoch、用时等——供脊柱判过拟合 + HPO 早杀（**单向上报，不让脊柱控制内层循环**）。
- `predict()` 输出**留在 `fret12` 量纲**（与传统侧 raw_mean 口径一致，保 MSE/R²），脊柱据此算 4 指标。

---

## 4. PyTorch 边界与训练生命周期归属

**内层训练循环归 PyTorch（卡带内），外层协议归框架（脊柱）。**

| 关注点 | 归属 | 说明 |
|---|---|---|
| epoch 循环 / autograd / optimizer / device / checkpoint | **卡带（PyTorch）** | 脊柱不可见、不介入 |
| 早停（看 earlystop_ds 上的代理指标决定何时收手） | **卡带自包含** | 用训练区尾段，**不碰打分集**，避免对打分集过拟合 |
| 官方 4 指标（corr/MSE/R²/daily-IC） | **脊柱**，在 `predict()` 之后算一次 | 打分集在 predict 前一次不碰 |
| 训练过程的可见性（曲线/早杀） | **卡带 → 脊柱单向上报**（TrainRecord / 轻量进度回调） | 用于判过拟合 + HPO 早杀整个 trial；**不是** "这个 epoch 该不该发生" 的逐步控制 |

**结论**：训练**之中**无数据回流脊柱做控制；脊柱只在 `predict()` 边界拿到 numpy 预测、训练**之后**算指标。早杀是 HPO 级（杀整个 trial）的可选加速，D0 先留 TrainRecord 钩子、实现推后。

---

## 5. 评测协议

> ⚠️ **2026-06-01 作废通告 / 2026-06-02 修订**：本节 §5.2 的"海选(单切分+早杀) + expanding 两阶段"口径**已废**，由 `AGENTS.md §十一 DL 评测协议`取代（锚定扩展 walk-forward + purge/embargo + 最坏折/bootstrap + 一命令两档预算）。**2026-06-02 §十一 又修订为：选型折采用 Nov 末倒贴 3 段×20 交易日窗（当前边界 `20230831–20230927`、`20230928–20231102`、`20231103–20231130`，非严格日历月）+ Dec 抽出做"交付对齐折"（方案 A：`train(Jun–Nov)/embargo(Dec1)/eval(Dec4–Dec29)`，1 seed、只报不选、验交付链）**——一切折口径以 `AGENTS.md §十一` 为准。原因见 `NOTE.md`「交付=方法非权重 → 评测协议为什么这么重设计」。下方 §5.3 三段折+embargo、§5.4 四指标双镜头、§5.5 numpy 参考模型**仍有效**，沿用。

### 5.1 walk-forward 复用，不另起炉灶

复用传统侧 `expanding` 走查思想。DL 有两类盲区、配两类工具：
- **单窗口内**自带过拟合探测器：一次 fit 里 train vs earlystop-val 两条曲线就能看出过拟合。
- **walk-forward 多折**探测时间不稳定性（撞运气）。

→ 最优分配 = **单窗口看曲线 + 少量 walk-forward 折看稳定性**，不堆很多折（DL 训练贵、数据有限）。

### 5.2 两阶段预算

| 阶段 | profile | 目的 | 种子 |
|---|---|---|---|
| **海选 search** | 单时间切分 + 早杀 | 快速分诊大量 config，砍掉没希望的 | 1–2 |
| **认证 validation（= expanding）** | 少量 expanding walk-forward 折 | 对 1–2 个幸存者做最终确认 | 3 |

交接 = 海选输出 `best_config.json`，认证（expanding）**消费 + 冻结**（config-lock）。Orchestrator 管这个生命周期（见 §7）。

> **就这两段，没有传统那套"三层 Holdout"**：传统 Dev/Review/Final 里第三层（12 月 Final）历史上从未执行，新协议已用 海选 + expanding 取代分层。防自欺内核见 §10。

### 5.3 三段折结构 + embargo

每折：

```
[ train_core | earlystop-val(训练区尾段) | embargo(1 日) | scoring-val ]
```

- earlystop-val 从训练区切，归卡带早停用；scoring-val 归脊柱打分用，**两者隔离**。
- embargo 沿用现有 1 日（`RollingProfile.embargo`），切断标签前视泄漏。

### 5.4 四指标 + 双镜头判官

- 指标：`corr`（对齐老师 pooled corr）/ `MSE` / `R²` / `daily-IC`。
- 判官 = **均值（pooled corr）+ 鲁棒（最坏折/minimax）双镜头同时看、人工权衡**，不退化成单指标闸刀（沿用 AGENTS §四「双镜头」/ §十一·11.3）。
- 同时看 **train vs earlystop-val gap + 逐 epoch 曲线** 判过拟合，**逐折 corr 方差** 判稳定性。

### 5.5 防泄漏：numpy 参考模型

一个 torch-free 的 **numpy 参考模型**（如恒 0 / 末 interval 线性）走**完整脊柱管线**必须打**低分**。若它打高分 = 脊柱漏了未来信息（窗口/归一化/标签对齐出 bug）。这是 D0 的核心验收闸。

### 5.6 profile 调整

- **砍 short/medium**（8/20 天序列对 DL 不公平）。
- **加单切分迭代档**（调试 / 海选用）。
- 认证沿用 `expanding`-类 walk-forward（少折）。

---

## 6. 超参搜索（省算力的叠加策略）

> ⚠️ **2026-06-01 升级通告**：选型粒度口径以 `AGENTS.md §十一·11.4/11.6/11.7` 为准——**结构族为主**（一命令一族）+ 命令内小 HPO 网格（结构子旋钮 + λ 损失对齐）+ **seed 当探针不优化** + 其余训练超参**固定重正则默认**；一命令两档预算（筛选→认证）取代旧"海选→另起命令 expanding"。下方"只搜 3 旋钮 / 随机搜索 / 早杀 / 训练行采样"作为**实现细节**仍可参考，但范围/阶段以 §十一 为准。

1. **只搜 3 个结构超参**：序列长 `seq_len` / 隐藏维 `hidden_size` / 层数 `num_layers`；其余冻结默认。
2. **随机搜索 ≫ 网格**。
3. **早杀**：跑几个 epoch 后明显落后即弃整个 trial。
4. **训练行采样**（类比树侧 0.33）降单 fit 成本。
5. **海选少种子（1–2）、认证多种子（3）**。

> `search_space` 是模型私有声明（跟卡带走）；"本次 run 搜哪几个、收窄到什么范围、几个 trial"是 `SearchConfig`（跟 run 走，见 §7）。

**实现落地（`src/dl_search.py`，torch-free）**：

- `sample_hparams` 读 `cartridge.search_space` 的迷你 spec（`choice` / `int` / `uniform` 三型）随机采样，`SearchConfig.search_overrides` 按 knob 收窄/替换。
- **`seq_len` 是 trainer 级旋钮、`hidden_size`/`num_layers` 是卡带级 hparams**：`seq_len` 决定 `SequenceDataset` 开窗（要重建数据集），故 Searcher 把采到的 `seq_len` 喂 `SequenceTrainer(seq_len=...)`，其余进 `cartridge.fit(hparams=...)`。这条边界写死在 Searcher 里。
- `Searcher` 跑 trial 环：每 trial 采一组结构超参 → 海选档单切分 × 1–2 种子跑 fold → `summarize_folds` 取 pooled+最坏折 → 按 `val_corr` 排名 → 落 `trials.csv` + `best_config.json`。
- **早杀** = HPO 级（杀整个 trial）的可选加速：`EarlyKillPolicy` 留接口、D0 为 no-op 桩（读 `TrainRecord` 曲线判断的逻辑等海选真撞 GPU 上限再补，规格 §11）。

---

## 7. 配置管理：分布式声明 + 中央组装

### 7.1 原则

- **每层声明属于自己职责的配置块**（谁的活谁定义旋钮）。
- **Orchestrator 只负责组装**：把各块拼成一份 `RunConfig` → 校验一致性 → 冻结 → 派发 → 盖进 run_id；外加它独占的 run 级/执行级配置。
- **反模式警告**：不要把所有配置塞进 Orchestrator 写成 god-object——那会让"换模型不动脊柱"作废。

**实现落地（`experiments/run_dl.py`，torch-free）**：`Orchestrator` 吃一份已组装的 `RunConfig`，按 registry 建 `adapter` + `cartridge_factory`、按 `MeowDataLoader` 建 `raw_loader`（测试可注入合成 loader）、按 `protocol` 派生折；dump `config.json`（含 `config_fingerprint`）到 `out_dir/<run_id>/`；按 `stage` 分派——`SEARCH` → 交 `Searcher`（落 `trials.csv` + `best_config.json`）、`VALIDATION` → 定参跑 expanding 少折 × 多种子（落 `fold_metrics.csv` + `summary.json`）。它不写任何块旋钮，只组装 + 派发 + 落盘。

### 7.2 配置归属表

| 配置块 | 真相源（谁声明） | Orchestrator 的角色 |
|---|---|---|
| 搜索空间（超参 + 范围/分布） | ModelCartridge.`search_space` | 选 cartridge；本 run 收窄/覆盖/冻结部分范围 |
| required_adapter + 通道 | Cartridge 声明需求、Adapter 声明通道 | 绑定 + 校验匹配（不匹配组装期报错） |
| RollingProfile（折结构/embargo/阶段） | dl_protocol / eval_protocol | 选哪个 profile + 哪个阶段 |
| 种子 / 种子数 | — | 独占 |
| 重训 vs resume、run_id、输出目录 | — | 独占 |
| 搜索预算（n_trials/早杀/subsample） | — | 独占 |
| GPU device / worker 数 | — | 独占（执行级） |
| 数据窗口（train/val 日期区间） | Orchestrator 给区间、Protocol 派生折 | 拥有区间 |
| config-lock 冻结 | — | 强制执行 |

### 7.3 三层枚举拆分（避免"枚举字符串散落各处"）

把三件事分开，别糊在一起：

| 层 | 是什么 | 放哪 |
|---|---|---|
| ① 合法词表（枚举） | "blend_mode ∈ {raw_mean, zscore}" 这类封闭选项集 | **集中**：放该配置块文件顶部的 `Enum` 类 |
| ② 实现注册（枚举→类/函数） | `LSTM` 枚举值对应哪个 cartridge 类 | per-kind registry，**和实现挨着**（cartridge 文件把自己注册进去） |
| ③ 本次选择（选了哪个值） | 这次 run 用 raw_mean 还是 zscore | `RunConfig`，Orchestrator 组装 |

- 保留"分布式"的只有 **③ 的选择 + cartridge 私有 search_space**；**① 词表必须集中**。
- cartridge 用 `required_adapter: AdapterKind`（**类型化引用集中枚举**），不写裸字符串——拼错当场报错。
- search_space 是模型私有超参声明，**不算全局枚举、不进 config/**。

### 7.4 config/ 文件结构（每块单独成文件 + 枚举进块文件顶部）

```
config/
  model_config.py     # ModelKind(Enum) + ModelConfig(frozen dataclass)
  adapter_config.py   # AdapterKind(Enum) + AdapterConfig
  protocol_config.py  # Stage / ProfileKind(Enum) + ProtocolConfig
  search_config.py    # SearchConfig（n_trials / 早杀 / 对 search_space 的收窄·冻结）
  exec_config.py      # ExecConfig（seeds / device / resume / retrain / out_dir）
  run_config.py       # RunConfig = 组装以上(frozen) + 组装期校验
```

- 枚举放**各块文件顶部**（高内聚：一个文件同时见到词表 + schema），不放单个 `enums.py` 巨型文件；**真·跨块共享**的才提升到 `config/common.py`。
- 实现注册放对应 registry（如 `models/registry.py`）。

### 7.5 RunConfig 四性质

1. **不可变 + 冻结**（frozen dataclass）：组装后中途改不动——**这就是 config-lock 的机械实现**。
2. **可序列化**：每次 run dump 成 JSON 落输出目录旁，完全可复现可审计。
3. **组装期校验**：required_adapter 对不对、阶段/profile 一不一致、覆盖有没有拧到不存在的旋钮——全在组装那刻报错。
4. **分层嵌套**：`RunConfig = {ModelConfig, AdapterConfig, ProtocolConfig, SearchConfig, ExecConfig}`，不平铺成大 dict。

### 7.6 run_id 口径

- **手工维护、语义更重**（沿用传统侧习惯）：`<日期>_<阶段>_<模型>_<意图>_<版本>`，如 `20260601_search_lstm_seqsweep_v1`、`20260603_valid_lstm_wf3seed_v1`。
- **哈希降级成字段防漂移**：dump 的 RunConfig 里存 `config_fingerprint = hash(语义内容)`；唯一用途 = resume/复用 run_id 时比对，不一致就告警/拒跑。可读性 + 防配置悄悄漂移两头都占。

### 7.7 "是否重新训练"拆两层（都归 Orchestrator/执行级）

- **(a) resume 中断的 run**：部分折跑完接着跑，按 `(profile, fold, experiment_id)` 查 status（同传统侧）。
- **(b) 复用 checkpoint vs 从头重训**：DL 训练贵，支持 `(config_fingerprint, fold, seed)` 训过就 load 权重跳过。
- 两者是 `ExecConfig` 里**两个独立字段**，别混成一个 bool。

---

## 8. 数据喂入泛化性

### 8.0 数据实情（决定路线，接 PyTorch 前查真实 h5 定）

查 `data/20230601.h5`（62 列、~309 票/日、~226 interval/票）得到的硬约束：

- **没有连续 LOB**：盘口只有 4 个稀疏聚合档位（`bid0/4/9/19` + `ask` 同构）+ 聚合 size（`bsize0 / bsize0_4 / bsize5_9 / bsize10_19`）+ turnover ratio（`btr0_4 / atr0_4 …`），**不是 DeepLOB 假设的连续 10 档价量网格**。→ **DeepLOB-on-rawLOB 退役**（无对应输入，照搬 Inception 网格没意义）。
- **能喂的 = ~59 个原始微结构通道**：价（`midpx / lastpx / OHLC` + 4 档买卖价）、量（聚合 size / turnover ratio）、订单流（主动买卖 + 挂单 + 撤单的笔数 / 量 / 额 / 高低 / vwad）。
- **序列 = 某票某日 ~226 步日内路径**，绝不跨日。
- **路线收敛（2026-06-01 更新，替代原 TCN 主攻判断）**：`TCN-on-原始微结构` 海选 + expanding 已跑完、**否决**——真因 = **截面盲区**（`[B,L,C]` 一次只看一只票、RawChannelAdapter 无任何截面归一 → 看不见 cross-z/rank 那一维 alpha，而传统 0.0776 主力恰在此），非 max_epochs；且 raw 与 433 同源、无更富数据、raw-LOB 终局不存在。详见 `NOTE.md`「路线再收敛」+ `CLAUDE.md`。**新主攻 = 截面联合建模**（共享时序腿 + 截面 set-attention，见 §8.2）；`GRU-on-433`（FeatureAdapter）作**时序基线 + 保底**先跑。`TCN` / `DEEPLOB` 枚举保留为历史词位。

### 8.1 张量形态与泛化缝

- 主流序列模型标准输入 = `[B, L, C]`（批 × 序列长 × 通道）。
- 我们的"序列" = **某票某日的日内 interval 序列**，`L`=日内 bar 数、`C`=通道，**绝不跨日**。
- 业界改进与本框架适配能力：

| 方向 | 例子 | 当前契约能否直接适配 |
|---|---|---|
| 换通道内容 | TCN 原始微结构通道 / PatchTST patch | ✅ 直接适配，换 InputAdapter 即可，脊柱零改 |
| 多粒度多分辨率 | 1min+5min+日级多张张量 | ⚠️ 不直接：当前单张 [L,C] 不够 |
| 截面/图结构 | 跨票 attention / 图网络 | ⚠️ 不直接：缺邻接关系输入 |

- **诚实结论**：对"换通道内容"是真泛化（D0 即覆盖 LSTM-on-features 与 DeepLOB）；对"换输入结构种类"（多张量/图）有**清晰扩展缝**——把 InputAdapter 输出从"一张 ndarray"放宽成"张量字典"，仅 Window Indexer + Normalizer 两组件需做一次泛化，协议/指标/HPO 不动。
- **D0 取舍（已拍板）**：先锁单张 `[L,C]`（YAGNI，主线冲 0.12 不需要多张量），但 Window Indexer / Normalizer 写成"不假设只有一张"的形态，留缝。

---

### 8.2 截面联合建模（新主攻，张量契约泛化；设计「为什么」见 `NOTE.md`「截面模型怎么设计」）

**目标**：每个 `(date, interval)` 把当时在场的 ~N 票横喂一个模型，让网络看见"这只票此刻在全市场排第几"——cross-z / cross-rank 那一维 alpha（传统 0.0776 主力、TCN-on-raw 整整瞎掉的维度）。

**结构 = 因子化两腿 + 零初门控残差**（业界 STGNN / 关系排序同构；为什么不用时空联合大注意力、不用 GAT/GCN 图网络见 `NOTE.md`）：

```
每票日内窗 [L, C] ──共享时序腿(GRU，所有票/所有 interval 同一套权重)──▶ h_i ∈ R^d
{h_1..h_N}(同一 (date,interval) 在场票) ──截面腿(set-attention，无位置/无 ID，带 padding mask)──▶ Δ_i
z_i = h_i + γ·Δ_i (γ 初始化 0) ──per-票头(Linear)──▶ ŷ_i  (该票该 interval 的 fret12)
```

- 时序腿权重**跨票 + 跨 interval 共享**（样本效率拉满）；截面腿只在某 `(date,interval)` 的票集合内自注意力。
- set-attention **无位置编码、无股票 ID** → 置换等变 + 零身份（换池子免费保险，§十一·11.5）。
- **零初门控残差**：`γ` 标量初始化 **0** → 训练起点截面腿零贡献、逐位等价纯时序 GRU 基线（= GRU-on-433 那条 ③，③ 天然是 ② 的消融下界）；学得到截面增益再放大 `γ`。

**张量契约泛化（§8.1 扩展缝兑现，唯一动脊柱处）**：

| 组件 | 原（单票序列） | 改（截面快照） |
|---|---|---|
| 样本单元 | 某 (票,日,interval) 一个窗 `[L,C]` | 某 (date,interval) 的票集合 `[N,L,C]` + `mask[N]` + `y[N]` |
| Window Indexer | 按 (票,日) 逐窗 gather | 按 (date,interval) 聚票，各票取前 L 窗 stack；变长 N → pad + mask（不跨日、因果对齐不变） |
| Normalizer | 通道 train-stat 白化 | **v1 沿用逐通道 train-stat 白化、不加 cross-z**（见下「落地修订」） |
| 协议 / 指标 / HPO / Orchestrator / SWEEP / trainer | — | **全不动**（脊柱+卡带架构的价值兑现；卡带内部把 `SequenceDataset` 重组成截面，trainer 零改） |

**卡带**：新 `CrossSectionCartridge`（`ModelKind` 加 `XSECTION`，绑 `FEATURE_433`）；torch（共享 GRU + `nn.MultiheadAttention` 1 块 + 零初门控残差 + Linear 头）；loss = `MSE + λ·(1 − 截面内 Pearson)`，λ∈{0,0.3} 走 SWEEP（λ 经 `--hparams lambda_corr`）。

**与评分联姻**：一次预测整截面 ~N 个 fret12 → corr 项就是**截面内 Pearson** = 老师 pooled corr / daily-IC 那一维，预测一个截面、直接优化它的截面排名。

**风险盒**：截面腿是冲 0.10 主攻一摆；周中若 ③ 基线 + ② 截面都无正信号超传统保底（最坏折 + 均值双镜头），回落保底交付（DL 作增强叠传统核心，§十一·11.8）。

#### 8.2.1 落地修订（2026-06-02，WT-B 合并后；凡与上表冲突以本节为准）

实现已落 `src/sequence_dataset.py`（`CrossSectionIndexer` / `CrossSectionDataset`）、`models/dl_models.py`（`_build_xsection_module` / `_GpuCrossSectionSource` / `CrossSectionCartridge`）、`src/dl_losses.py`、`tests/test_dl_xsection.py`。三处口径较上文收敛：

1. **截面内 cross-z 归一化：v1 默认关；2026-06-02 起加 `cross_z` 开关供消融**（更新上文「v1 不做、且不做开关」）。原 v1 默认关的理由：① 截面那维本想由**学出来的 set-attention 截面腿**承重，不靠输入端 cross-z；② 433 已含 cross-z/cross-rank，恐冗余；③ 输入端跨票 z-score 抹绝对量级、威胁「R²≥0」硬底。**实跑反证（run `20260602_sweep_xsection_v1`，6/9 折 `val_corr.mean=0.0624` ≈ 纯 GRU 0.0585）：截面腿没把这维学出来，理由①未兑现；而理由③已被永远开启的 OLS rescale 兜底、过时。** 故把 cross-z 实现为 `_build_xsection_module(cross_z=...)` 的 forward 第一层 masked 截面 z（同一 `(date,interval)` 快照、各 `(lag,channel)` 沿在场票 z，mask 排 pad），经 `--hparams cross_z=1` 切，**默认 0（保旧基线语义）**。线 B 三档消融 `纯GRU / 截面 cross_z=0 / 截面 cross_z=1` 验它能否把分数顶过 0.06 天花板。单测见 `tests/test_dl_xsection.py::TestXSectionCrossZ`。

2. **损失固定 `MSE + λ·(1−截面内 Pearson)`，不做 MSE/Huber 开关**；corr 项 = masked 逐快照 Pearson（`src/dl_losses.py`，数值稳定可微、向量化、≥2 票才计入、对各快照等权平均）；λ 经 `--hparams lambda_corr`，不进 search_space。

3. **预测永远套全局线性 rescale `a·ŷ+b`**（OLS **fit-on-train / apply-on-val**，零泄漏，GRU + XSECTION 两卡带都套）。机理：OLS 重标定后**训练段** `R²=corr²` 精确成立，故 corr>0 即保 R²≥0；val 上套用 train 系数，受 train→val 尺度漂移影响可微负，这是零泄漏口径下可接受的真实读数。选 λ 仍只按 max val_corr，三指标逐折照报供核对。

**卡带数据缝（关键）**：`CrossSectionCartridge` 吃 trainer 现成的 `SequenceDataset`，内部用 `CrossSectionDataset.from_whitened(ds.feature_matrix(), ds.arrays, ds.seq_len)` **零再白化、零复制**地按 `(date,interval)` 重组成快照；predict 逐快照前向后，用 `argsort(flat_label_rows)` 把逐票预测映射回 `ds.label_rows` 升序（两者末行集合相同，映射无损）→ 指标对齐仍走 `SequenceDataset.label_frame()`。**故 `src/dl_trainer.py`、协议、Orchestrator、SWEEP 全部零改**。

**内存纪律**：`_GpuCrossSectionSource` 复用 `_GpuWindowSource` 同款「事前显存预算判断 → resident / 否则 CPU-gather→整批搬卡」两态（截面 `[B,maxN,L,C]` 瞬时更吃显存，`_VRAM_FRAC` 收紧到 0.45，`snap_batch` 默认 4 控瞬时）；每批瞬时有界、避免整期截面物化。

## 9. D0 地基交付物（torch-free，Mac CPU 即可跑通）

按依赖顺序实现，**泄漏风险最高的先写先测**：

1. `src/sequence_dataset.py` —— Window Indexer：惰性 gather [B,L,C]、不跨日、因果对齐、warmup 处理；Normalizer：fit-on-train、可配 identity。
2. `src/dl_protocol.py` —— Protocol Engine：walk-forward 折日期、三段切分、embargo、4 指标、gap/曲线/稳定性、防泄漏检查入口。
3. `src/dl_trainer.py` —— `SequenceTrainer(BaseTrainer)` 骨架：编排 data pipeline + cartridge，产出 `FoldResult`（已与 leaderboard 兼容，见 `src/trainer.py`）。
4. `models/dl_models.py` —— `InputAdapter` 接口 + `FeatureAdapter`（包装 433 管线）；**numpy 参考模型**（防泄漏探测器）。
5. `config/` —— §7.4 的配置块（frozen dataclass + 枚举）+ `RunConfig` 组装/校验骨架。
6. **单测**：CPU 端到端跑通全链路；**参考模型打低分**；无泄漏告警；窗口不跨日；归一化只用训练统计量。

> 全部 torch-free、Mac 可测；PyTorch 卡带（LSTM 等）等 4060 就绪后再在卡带层加，脊柱不动。

---

## 10. 评测纪律（防自欺：config-lock + 少看 expanding）

DL 评测就两段（§5.2）：**海选（便宜搜超参）→ expanding 认证（walk-forward 判官）**。**不套传统"三层 Holdout"**——第三层（12 月 Final）历史从未执行，新协议已用 海选 + expanding 取代分层。保留的只有防自欺内核：

- **config-lock**：获胜超参在进入最终交付演练（Dec 全窗口 fit/eval）之前冻结——靠 RunConfig frozen 机械保证；演练只当 sanity，绝不回灌选型。
- **少看 expanding**：同一 expanding 窗口反复看会被过拟合、不再可信；海选已用便宜车道滤掉没希望的，expanding 只在少数幸存者上跑。
- **不在最终确认数据上调参**：交付演练那次（最近一段、未用于选型的数据）看完即定，不据其改模型再交。

---

## 11. 未决 / 待办（2026-06-01 路线收敛后刷新）

- **D0 地基 + 接卡前基础设施**：✅ 已落地（§9 六件 + Searcher / Orchestrator / RawChannelAdapter，torch-free / Mac CPU 跑通）。`ModelKind` 含 `TCN`(否决) / `GRU`(已实现) / `DEEPLOB`(退役)；`AdapterKind` 含 `FEATURE_433` / `RAW_CHANNELS` / `IDENTITY`。
- **TCN-on-raw**：❌ **否决**（卡带本体 + 海选 + expanding 均已跑完）。真因 = 截面盲区，非 max_epochs；详见 `NOTE.md`「路线再收敛」、`CLAUDE.md` 看板。
- **新评测协议落代码**：✅ 两档骨架已落地——`Stage.SWEEP`「一命令两档」（档1 小网格 × 近 2 折 × 2 seed → 按**最坏折**选冠军；档2 冠军 × N 折 × 3 seed）+ `build_dl_folds(fold_select="recent")` + `enumerate_grid`（确定性网格）。✅ **2026-06-02 协议修订已落代码**：① 选型折改为 **Nov 末倒贴 3 段×20 交易日窗**（当前边界 `20230831–20230927`、`20230928–20231102`、`20231103–20231130`，非严格日历月）；② 新增显式 `--delivery-eval-end` **交付对齐折**——冠军定死后跑 `train(Jun–Nov)/embargo(Dec1)/eval(Dec4–Dec29)` × 1 seed，写 `summary.json` 的 `delivery` 块、不参与排名。口径权威见 `AGENTS.md §十一`（11.2/11.6/11.8/11.9，2026-06-02 修订段）。
- **GRU-on-433 基线**：✅ 卡带已实现（绑 `FEATURE_433`、MSE + AdamW + 早停）。**`MSE+λ·corr` 损失对齐 + OLS rescale 后处理已落（2026-06-02）**：λ 经 `--hparams lambda_corr`、预测永远套训练段 OLS rescale（§8.2.1）。
- **截面模型卡带（主攻）**：✅ **已落地（2026-06-02，§8.2.1）**——`XSECTION` 卡带（共享 GRU 时序腿 + set-attention 截面腿 + 零初门控残差 + per-票头 + masked 截面 Pearson 损失 + rescale），张量契约经 `CrossSectionIndexer`/`CrossSectionDataset` 按 `(date,interval)` 聚票泛化；测试入口见 `tests/test_dl_xsection.py`。maxfold 内存演练见 `CLAUDE.md`。
- **早杀实现**：`EarlyKillPolicy` 钩子在位（no-op）；GRU 卡带可逐 epoch 回调，撞 GPU 上限再接。
- **多张量/图输入**：暂不实现，按 §8.1 留缝。
- **传统交付 fit/predict 签名核验**：遗留另会话办（见 `CLAUDE.md` 看板），与 DL 地基不冲突。

---

## 12. 运行期资源管理（GPU 搬运 + 瘦内存管线 + 崩溃可排查日志）

> 本节是 4060 实跑的工程沉淀，把"**GPU 怎么喂、内存怎么管、崩了怎么查**"三件事定死。
> 核心纪律一句话：**宁可慢也绝不让长跑因显存/内存溢出崩**。所有"省时"优化的前提都是
> 先满足这条；任何会盲分配大块显存/内存的写法一律否决。
>
> **本机硬件实情（决定下面所有取舍）**：
> - 内存 ~**34GB**（充足，整张 `[N,C]` 特征矩阵舒服待在内存，**全量数据/全量窗口不砍**）；
> - GPU **8GB**（可用 ~6.5GB），而 433 通道 × 百万行训练矩阵 ~7–15GB，**塞不进显存**——
>   所以"整张驻留显存"只在小抽样/早折偶尔成立，正式大跑必须流式喂。

### 12.1 GPU 喂数：三态 `_GpuWindowSource`（`models/dl_models.py`）

把一个 `SequenceDataset` 的「归一化特征矩阵 `[N,C]` + 窗口行索引」喂成 device 上的
`[B,L,C]` batch 流。**目标 = 让 GPU 不再干等 CPU 拼数据**。三条路按 device + 是否真装得下
显存**自动选**：

| 模式 | 触发条件 | 做法 | 内存/显存安全 |
|---|---|---|---|
| **resident**（纯显存快路） | `mem_get_info()` 实测 `need < 0.6 × free` 才走 | 整张矩阵驻显存，窗口 gather 全在显存带宽上做 | 事前按真实空闲显存判断，**绝不盲分配再接 OOM**；放行后仍 OOM（碎片）→ 二重保险退预取 |
| **prefetched**（预取流式，**正式大跑主路**） | 装不下显存 | 后台线程在 GPU 算第 i 批时，提前把第 i+1 批 `[B,L,C]` 在 CPU gather→落 pinned 内存，主线程 `non_blocking` 异步拷卡；CPU 备料与 GPU 计算重叠 | 在途同时只有 `_PREFETCH=3` 个 batch（~百 MB 级），**内存有界、永不 OOM**；实测稳态 GPU 利用率 ~**79%**（够好，不再投资） |
| **cpu** | device 为 cpu（单测 / Mac） | 纯 CPU gather，不涉及 pin/异步 | —— |

关键工程细节：
- **零拷贝共享**：CPU 侧原料用 `torch.from_numpy(ds.feature_matrix())`，直接共享 dataset
  里已有的大矩阵，**不再复制一份 14GB**。
- **pinned 生命周期保证**：预取生成器在 `yield` 处持活 `x_pin`/`y_pin`，消费方下次 `next()`
  才释放——`non_blocking` 拷贝期间 pinned 源不会被回收（与 PyTorch DataLoader 同条保证）。
- **不卡死**：producer/consumer 用 `threading.Event` stop + 带超时 `put`/排空队列 + `join(timeout=5)`，
  消费方提前退出（早停/异常）也能干净收尾。
- **显式释放**：`predict()` 用 `try/finally: src.free()` 释放 device/CPU 句柄，给下一段腾资源。
- torch 仍**只在卡带层**出现，脊柱保持 torch-free；CPU 单测路径完全不受影响。

### 12.2 瘦内存管线（把每折峰值从 ~24GB 压到 ~12GB）

硬化前每折会同时压着 **~7 份全量副本**（无限日缓存 + concatenate 双份 + fit 的 nan_to_num
整张临时 + subset 副本 + 三个 dataset 各一份 transform 副本），119 天全票折直接打穿。逐一干掉：

1. **逐日缓存有界**（`FeatureAdapter._day_cache`）：单日全票 433 特征 ~120MB，无限缓存 144 天
   = ~17GB。改为**蓄水式保留共享基段**（非 LRU）——所有 expanding 折都从 Jun1 起、共享早期
   日期前缀；装满 `_day_cache_cap=32` 天（≈3.8GB）后**不淘汰也不新增**，自然钉住最高命中率
   的共享基段，末端差异日每折从磁盘 pickle 重读（~0.3s/天，便宜）。
2. **预分配 + 逐日流式填充**（`load_sequence_arrays`）：先据 `_day_nrows` 台账定总行数、一次性
   `np.empty((total,C))`，再逐日把单日块写进切片——**杜绝 `np.concatenate` 把"所有日块 + 输出"
   两份 14GB 同时压在内存**。
3. **Normalizer 分块 fit**（`fit_chunked`）：逐 `CHUNK_ROWS=200k` 行 nan_to_num + float64 累加
   count/sum/sumsq 合成 mean/std，**不物化整张 nan_to_num 副本**；core+es 当训练区一并统计但
   **不真的 concatenate**，省一份全量副本，统计值与整块逐位等价。
4. **原地白化**（`SequenceDataset(own_features=True)` + `transform(inplace=True)`）：本折独占三份
   arrays，逐块在原缓冲上 nan_to_num + 仿射，**三个 dataset 不再各造一份 `[N,C]`**。
5. **三段分别现算**（`run_on_dl_fold`）：直接按 core / earlystop / scoring 三段**分别** `_build_arrays`，
   不"先建整训练帧再 subset"造 14GB+12GB 双份共存；三段日期互斥、并集 = 原训练区，统计口径
   不变、无泄漏。

**实测效果**：28 天折峰值 RSS **24GB → 12GB**，`val_corr` 完全不变（0.0113）；119 天全票折按
此外推峰值 ~**22GB**，34GB 机器安全（数组地板本身 ~16GB）。

### 12.3 崩溃可排查日志（正式跑必备）

长跑一旦崩，必须能立刻定位是**显存 OOM / 内存 OOM / 还是别的**。两层日志：

- **分阶段计时行**（`run_on_dl_fold` 打 stderr、`flush=True`，不被重定向缓冲）：每折一行
  `[fold k timing] load=.. norm=.. build_ds=.. fit_GPU=.. predict=.. metrics=.. | GPU忙占比≈..%`，
  用来区分"GPU 沉默"沉默在读盘/归一化/建集还是真在算。
- **后台资源监控**（正式跑期间每 ~15 分钟一采，落独立日志文件）：GPU 利用率 + 显存占用
  （`nvidia-smi`）+ 进程 RSS（`psutil`）。崩溃后回看这条曲线即可判定是否撞顶。
- **进程管理用 task 机制（非 ps kill）**：长跑挂后台任务，停止走 task stop，避免误杀/留孤儿。
