# 已验证结论 — 论文三块

本文档汇总经过实验验证、口径正确、可写进论文的三项 CTM 优化方法。
每一块都标注了**衡量口径**和**边界**(在哪不 work)，避免重蹈"口径错判"的覆辙。

> **数据来源(2026-07-28 更新)**: 下表数字来自 `paper_repro` **5-seed 复现**(`paper_repro/csv_data/repro_summary_0728.csv`, 118 run), baseline 用 repro 的 baseline 组(matched 5-seed), 而非旧的单 seed / st00 常量。分析脚本 `paper/analyze_repro_0724.py`。
>
> 衡量口径前提(强制): CTM 有两种打分 —— **final-tick**(只看最后一步, = `best_test_acc`)和 **most-certain-tick**(每样本挑最自信的一步, = `best_test_acc_mc`, 简称 mc)。后者天然高得多, 这是 CTM 机制本身的红利, 不是 idea 的功劳。判 idea 必须 mc 比 mc、final 比 final。

---

## 第一块:Cross-Tick JEPA — 让思考连贯(隐状态正则化)

### 一句话
训练一个轻量预测器,约束"相邻 tick 的隐状态可预测"(`synch[t] → synch[t+1]`),作为辅助损失。逼模型**连贯地思考**而非随机跳变。

### 机制
- 每个 tick 产生 synchronisation representation(神经元同步激活状态)。
- 轻量 bias-free MLP 预测下一 tick, cosine loss 约束。
- **两道防坍塌防线**: stop-gradient on target + cosine(只约束方向不约束幅度)。
- 作为**辅助损失**加到主任务 loss, **零推理开销**(predictor 只训练时用)。

### 实验证据(5-seed 复现, 0724)
| 任务 | 配置 | final-tick Δ | mc Δ | n |
|---|---|---|---|---|
| cifar10 | w=0.1 | **+4.5pp** (73.6% vs 69.0%) | +0.2pp(中性) | 5 |
| qamnist | w=0.5 | **+9.0pp** (47.9% vs 38.8%) | -0.0pp(中性) | 5 |
| mazes | w=0.1 | +0.0pp(中性, baseline 已 89% 饱和) | -0.1pp | 5 |

### 关键洞察
- **JEPA 抬的是 final-tick(最后一步), 不抬 mc 天花板**。这是它的设计使然: 让"更多 tick 都对", 但最优 tick 由容量决定。复现干净复现了这个机制(cifar10/qamnist final 显著正, mc 处处平)。
- **任务相关**: 视觉感知任务(cifar10/qamnist, 隐状态质量是瓶颈)受益; 序列任务不兼容(parity/sort 是 idea 坟场, JEPA 在 parity 上 0-iter killed)。
- 旧报告 cifar10 "+7.5pp" 是因旧 baseline(66.9%)偏低; matched 5-seed baseline 69.0% 下, 真实增益 +4.5pp, 方向稳健、4/5 seed 上升。

### 代码
- `baseline/utils/jepa.py` — predictor 定义 + `compute_jepa_loss`
- `model/model_ctm_llm.py:1255` — LLM 版内联实现
- `baseline/tasks/*/train.py` — 训练循环接线

---

## 第二块:Draft-Revise — 草稿-修订(对抗式自检)

### 一句话
CTM 正常是"思考完直接交答案"; draft-revise 改成**先出草稿, 再故意加噪声扰动, 学着修回来**。相当于"先打草稿再检查一遍"。

### 机制
- `draft_mode="revise"`: 两阶段 —— draft(快速粗答案) + revise(扰动后修订)。
- 关键超参: `draft_corrupt_prob`(噪声强度)、`draft_revise_weight`(修订 loss 权重)、`draft_block_size`。
- 修订 loss 鼓励模型从被扰动的草稿恢复出正确答案, 训练出"自检"能力。

### 实验证据(5-seed 复现, 0724)
| 任务 | 配置 | final-tick Δ | mc Δ | n |
|---|---|---|---|---|
| parity | w=0.1, cp=0.15 | **+2.5pp** (72.1% vs 69.7%) | +0.9pp(中性) | 5 |
| cifar10 | w=0.2, cp=0.3 | +1.8pp (70.8% vs 69.0%) | -0.2pp(中性) | 5 |
| mazes | w=0.1, cp=0.15 | +0.4pp(中性) | +0.3pp(中性) | 5 |

### 关键洞察
- **revise 抬的是 final-tick, 不抬 mc 天花板**(和 JEPA 同性质)。在 parity 上 final +2.5pp(3/5 seed 上升), 是它最稳的正信号。
- **⚠️ 重要更正**: 此前文档声称 "parity mc +10pp(0.882→0.984), 唯一抬动 mc 天花板的 win" —— **5-seed 复现证伪**。根因: 旧 baseline `0.882` 是**单颗坏种子**(s0); matched 5-seed baseline = `0.970`(双峰: s0=0.882 拉低, 其余 ~1.0)。对照公平 baseline, revise mc `0.979` 仅 +0.9pp(噪声内)。per-seed 显示 revise 抬的是**地板**(把坏种 s0 从 88→93)而非**天花板**(好种子两边都 ~100)。" +10pp" 是拿 revise 均值比 baseline 最差种子的口径错误。
- **副作用小**: cifar10/mazes 上 delta 在 ±0.3pp 内, 低风险。

### 代码
- `baseline/utils/dtt_ideas.py` — draft_mode 实现
- `baseline/tasks/*/train.py` — 接线(注意: parity/sort 上 revise_weight 曾漏接, 见 AGENTS.md 两步走教训)

---

## 第三块:Sparsity — 稀疏激活(效率优化)

### 一句话
每一步"思考"只激活 top-k 比例的神经元(`topk_neurons = r`), 用更少算力换可接受的掉点。**这是效率方法, 不是涨点方法**, 必须看 Pareto(精度 vs 算力), 不能只比精度。

### 机制
- `topk_neurons = r`: 每个 tick 只有 r 比例的 NLM 神经元被激活更新。
- **算力模型**: NLM 思考回路做 ~r 的功, 省 (1-r) 的 NLM 算力。**backbone(resnet)不稀疏化**, 端到端墙钟加速 < (1-r), 需 sparse kernel 才变现。

### 实验证据(Pareto, mc 口径, 5-seed 复现 0728)
| 任务 | r=0.10 | 0.25 | 0.50 | 0.75 | baseline mc |
|---|---|---|---|---|---|
| **mazes** | -0.5 | +0.2 | +0.6 | +0.2 | 90.0% |
| **cifar10** | -0.4 | +0.1 | +0.3 | -0.1 | 84.2% |
| **parity** | **-7.0** | +3.0 | -1.1 | +0.7 | 97.0% |

> cifar10 r=0.75: 已补跑至满 200k iter(0728), mc Δ -0.1pp(此前欠训致假象 -1.6pp 已修正)。parity r=0.25/0.75 各 4 seed(2 个 CUDA 失败); cifar10 r=0.5 为 4 seed(1 个欠训剔除)。sort r=0.5 = -0.3pp(旧"-12pp 大坑"不复现, 但 sort 仅 1 个 r + inert 嫌疑, 不作前沿点)。

### 关键洞察
- **视觉/空间任务稀疏近乎免费**: mazes 全 r(0.1–0.75)都在 baseline ±0.6pp 内; cifar10 全 r(0.1–0.75)在 ±0.4pp 内。即"省 25–90% NLM 算力几乎不掉点"。
- **算法任务有硬边界**: parity 在 r=0.1 崩(mc -7pp)。parity = 64 长二进制序列逐步 XOR 累加, 需要全量神经元参与每一步, 激进稀疏会丢失中间累积。r≥0.25 才稳。
- **任务差异是论文的诚实卖点**: 不是"稀疏万能", 而是"感知任务廉价的递归思考可被大幅压缩; 需要全神经元协作的算法任务不行"。
- **诚实前提**: 省的是 NLM(CTM 递归思考部分, 是 CTM 区别于普通 CNN 的核心开销), 不是 backbone。真正变现需 sparse kernel。

### 代码
- CTM forward 的 `topk_neurons` 路径(neuron_select_type)
- `paper/analyze_repro_0724.py` — Pareto 分析 + 出图(`runs/figures/ctm_paper/figE_sparsity_pareto_0728.png`)

---

## 三块的论文叙事

| 块 | 性质 | 衡量口径 | 证明什么 |
|---|---|---|---|
| **JEPA** | 涨精度(final) | final-tick | CTM 隐状态可被正则化得更连贯, 改善最后一步预测(cifar10 +4.5, qamnist +9.0) |
| **revise** | 涨精度(final) | final-tick | CTM 的"草稿-修订"思考能改善需要核对的任务(parity final +2.5), 但**不抬 mc 天花板** |
| **sparsity** | 省算力 | Pareto(精度 vs r) | CTM 神经元稀疏激活对感知任务廉价(mazes/cifar10 近乎免费), 算法任务有边界(parity r=0.1 崩) |

三个方法**互补不冲突**: JEPA/revise 改善表示/推理的最终预测质量(final-tick), sparsity 改善效率。可在不同任务/不同目标下分别启用。

### 共同的边界 / 失败模式(写论文要诚实交代)
- **parity 是 idea 坳场**: JEPA/halt/EMA/reflex 全 killed(0-iter, break 训练)。**仅 revise 和 sparsity 活下来**(revise 抬 final, sparsity r≥0.25 可用但 r=0.1 崩)。
- **sort 极度脆弱**: tick1-25 全崩(需 tick50), 所有 combo idea 退化到魔数, heads/sparsity/nst 参数对 sort inert(wiring 残缺)。
- **mc 机制本身是最大红利**: qamnist final 21%→mc 99.6%, 这归 CTM 架构, 不归任何 idea。
- **revise 的 mc 天花板效应已被复现证伪**: 任何"revise 抬 mc"的旧表述一律以本文档为准(改为抬 final)。
