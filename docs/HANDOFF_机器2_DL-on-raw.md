# 第二台机器交接：跑 DL-on-raw（给第二台的 AI 冷启动看）

> 写于 2026-06-03（机器1）。你（机器2 的 AI）是冷启动、没有机器1 那段对话的记忆。本文档让你**开箱即用**地在第二台 GPU 上跑 DL-on-raw。先把「为什么/目标/前提/命令/判决」读完再动手。

---

## 0. 你的任务（一句话）

在第二台机器（有 GPU）上跑 **DL-on-raw（`XSECTION_RAW` 卡带）**，验证「直接吃原始盘口序列的 DL，能不能比传统模型多捞出 alpha」，看它在 **fold2** 上能否超过传统的 **0.0904**。机器1 同时在 CPU 上跑传统侧完整验证，两条腿并行。

---

## 1. 项目背景（精简）

- **任务**：MEOW 金融时序预测。预测 `fret12`（日内 12-interval 前向收益）。309 只票、日内 226 个 interval、Jun–Dec 144 天。
- **评分**：**pooled Pearson**（全样本拉平算一个相关系数，见 `meow/eval.py`）+ R² + MSE。
- **目标**：**12%+**（pooled Pearson）。老师明确同数据同题目有人做到 12%+，所以这是真实可达的、同一把尺子。
- **现状**：传统模型（ridge+lgbm 融合）~0.077；DL 三轮（TCN/GRU/截面）最高 ~0.062，都没破传统。

## 2. 为什么跑 DL-on-raw（关键发现，2026-06-03）

- **「数据锁死/无 LOB」是错的**：raw 其实是富数据——完整 4 档订单簿 + 逐笔成交流 + 挂撤单流 + 226 步日内序列（62 列）。详见 `docs/原始数据盘点与盘口建模诊断.md`。
- **前三轮的真缺口**：从没把「好架构」和「好输入」合在一起——
  - TCN-on-raw：吃 raw（好输入）但无截面（坏架构）→ 0.009；
  - 截面/GRU：带截面（好架构）但只喂 433 手工摘要（压缩输入）→ 0.062。
- **静态特征已近到头**：机器1 用 lgbm 加挂撤单/盘口/波动等归一化新特征，fold2 上只 +0.0008~+0.0019（小，且加更多反降）→ 静态摘要捞回的增量很小。
- **结论**：raw 缺失价值的**大头在时序动态**（挂撤单怎么随时间演化、队列动态），静态特征表达不了，**只有 DL 吃原始序列才挖得出** → 这就是你要验证的。

## 3. 已为你准备好的（都在 origin/feat 上，已 push）

- **`XSECTION_RAW` 卡带**（`models/dl_models.py` 的 `CrossSectionRawCartridge`）：= 截面模型架构（共享 GRU 时序腿 + 跨票 set-attention + 零初门控残差 + 训练段 OLS rescale）**原样复用**，输入绑定 `RAW_CHANNELS`。
- **`RawChannelAdapter`**（`models/dl_models.py`）：把 **59 个原始微结构通道**（含挂撤单、深档盘口、成交明细——经核查 raw 全部 59 个特征列都喂了、无遗漏）做归一化（价→相对 mid、量→log1p、midpx→日内对数收益）当通道。
- 注册自检 + GPU sanity（20 票/2 epoch）在机器1 已通过。

## 4. 环境前提（动手前必须满足）

1. **拉代码**：`git pull origin feat/dl-foundation`（XSECTION_RAW 卡带等都在这分支）。
2. **⚠️ 数据必须手动拷**：`data/*.h5`（Jun–Dec 144 天）是 **gitignore、不随 git 同步**！第二台 `git pull` 拿不到，得用 U 盘/网络把仓库根目录的 `data/` 整个拷到第二台同位置。**这是最大的坑，先确认 `data/` 里有 .h5 再跑。**
3. **环境**：装好 PyTorch + CUDA（GPU 驱动）；其余依赖见 `README.md` / `requirements`。
4. **import 约定**：`run_dl.py` 已自举 sys.path，直接 `python experiments/run_dl.py ...` 即可（src/config/models 三目录平铺）。

## 5. 跑的命令（DL-on-raw fold2 挑战）

```
python experiments/run_dl.py --stage sweep --model xsection_raw --adapter raw_channels --device cuda \
  --start 20230601 --end 20231130 --val-window 20 --step 20 --min-train-days 40 --max-folds 1 \
  --fold-select recent --grid-seq-len 32 --grid-hidden 32 --grid-layers 1 \
  --hparams dropout=0.2,weight_decay=0.001,max_epochs=15,patience=5,lambda_corr=0.3 \
  --seeds 42,43,44 --max-symbols 0 --dump-preds --run-id 20260603_xsection_raw_fold2 --out-dir results/dl
```

- `--max-folds 1 --fold-select recent`：取最近一折 = **fold2**（训练 Jun1–Nov1，打分 Nov3–Nov30），与传统 fold2 逐字节同边界。
- `--seeds 42,43,44`：3 seed（DL 有种子彩票，多 seed 求稳）。
- `--max-symbols 0`：全 309 票。`--dump-preds`：落逐票预测，供与传统精确对比。
- **先跑 5 分钟 sanity 确认链路**（可选）：把 `--max-symbols 20 --hparams max_epochs=2,...` 即可。
- 长跑建议后台 + 记日志：`... > logs/xsection_raw_fold2.log 2>&1 &`，并监控 GPU/内存别 OOM。

## 6. 怎么读结果 / 判决

- 结果落 `results/dl/20260603_xsection_raw_fold2/summary.json`，看 `val_corr.mean`（3 seed 均值）/ `val_corr.min`（最坏 seed）。
- **对手 = 传统 fold2 pooled Pearson = 0.0904**（机器1 的 P2 已算，传统融合代表 ridge+lgbm 在 fold2 打分窗）。
- **判决**：
  - **DL val_corr 明显 > 0.0904** → raw 时序动态有大料 → 立刻铺三折（`--max-folds 3`）+ 考虑更强架构（DeepLOB 式卷盘口形状）；
  - **DL ≈ 0.0904 或更低** → 查归一化（raw 量纲）/ 通道选择 / 架构，别盲目铺开。
- 精确同行对比：用 `--dump-preds` 落的预测，和机器1 的传统 fold2 预测（`results/dl/_p2_trad_folds/trad_preds_fold2_*.csv`，机器1 会同步）按 (date,symbol,interval) inner-join 算 Pearson。

## 7. 协调（两台机器并行，别冲突）

- **机器1（CPU）**：在跑传统侧完整验证（三折全票 + 新特征），改的是 `experiments/probe_*` 和特征实验脚本。
- **机器2（你，GPU）**：跑 DL，改的应是 DL/卡带侧。
- **别同时改同一文件**；`master` 谁都别动（交付分支）；都在 `feat/dl-foundation` 上协作，push 前先 `git pull --rebase` 避免分叉。
- 结果文件在 `results/`（gitignore，不随 git 同步）——跨机对比要手动同步预测 CSV，或各自报数。

## 8. 深读指针（要更多上下文时看）

- `docs/原始数据盘点与盘口建模诊断.md` —— raw 数据全貌 + 为什么转 DL（最重要）。
- `CLAUDE.md` —— 当前阶段进度看板 + 路线。
- `docs/specs/DL实验设计规格.md` —— DL 脊柱+卡带架构、评测协议。
- `models/dl_models.py` —— `RawChannelAdapter` + `CrossSectionRawCartridge`（你要跑的卡带）。

---

**最关键三件事**：① `git pull origin feat/dl-foundation`；② **确认 `data/*.h5` 已拷到第二台**；③ 跑第 5 节命令，拿 val_corr 和 0.0904 比。有疑问先读第 6/8 节。
