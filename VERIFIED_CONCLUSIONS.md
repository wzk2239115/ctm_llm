# 已验证结论 — 论文三块

本文档汇总经过实验验证、口径正确、可写进论文的三项 CTM 优化方法。
每一块都标注了**衡量口径**和**边界**(在哪不 work),避免重蹈"口径错判"的覆辙。

> 衡量口径前提(强制):CTM 有两种打分 —— **final-tick**(只看最后一步)和 **most-certain-tick**(每样本挑最自信的一步,简称 mc)。后者天然高得多(qamnist final 36%→mc 99%),这是 CTM 机制本身的红利,不是 idea 的功劳。判 idea 必须 mc 比 mc、final 比 final,且 baseline 用 st00 **复现值**(`csv_data/ctm_paper_summary.csv`),不能用旧常量。

---

## 第一块:Cross-Tick JEPA — 让思考连贯(隐状态正则化)

### 一句话
训练一个轻量预测器,约束"相邻 tick 的隐状态可预测"(synch[t] 能预测 synch[t+1]),作为辅助损失。这逼模型**连贯地思考**而非随机跳变。

### 机制
- 每个 tick 产生 synchronisation representation(神经元同步激活状态)。
- 轻量 bias-free MLP: `synch[t] → predicted synch[t+1]`,cosine loss 约束。
- **两道防坍塌防线**:stop-gradient on target + cosine(只约束方向不约束幅度),避免表示退化成常数。
- 作为**辅助损失**加到主任务 loss,**零推理开销**(predictor 只训练时用)。

### 实验证据(final-tick 口径,公平 mc-vs-mc / final-vs-final)
| 任务 | 最佳配置 | final-tick Δ | mc Δ | n |
|---|---|---|---|---|
| cifar10 | w=0.1 | **+7.5pp** (74.4% vs 66.9%) | -0.5pp(中性) | 3 |
| qamnist | w=0.5 | **+17pp** (53.7% vs 36.6%) | 中性 | 2 |
| mazes | w=0.1 | -1.2pp(中性) | -1.2pp | 3 |
| parity | — | killed(0-iter, break 训练) | — | — |
| sort | — | 崩溃/退化 | — | — |

### 关键洞察
- **JEPA 提升的是 final-tick(最后一步预测),不抬 mc 天花板**。这符合设计:它让"更多 tick 都对",但最优 tick 由容量决定。
- **任务相关**:视觉感知任务(cifar10/qamnist)受益,因为它们的隐状态质量是瓶颈;序列任务(parity/sort)不兼容,parity 直接 break 训练。
- 旧报告 `paper/explain/fig5b_jepa_weight.md` 说 mazes +9.6pp 是 baseline 过时假象(已修正)。

### 代码
- `baseline/utils/jepa.py` — predictor 定义 + `compute_jepa_loss`
- `model/model_ctm_llm.py:1255` — LLM 版内联实现
- `baseline/tasks/*/train.py` — 训练循环接线

---

## 第二块:Draft-Revise — 草稿-修订(对抗式自检)

### 一句话
CTM 正常是"思考完直接交答案";draft-revise 改成**先出草稿,再故意加噪声扰动,学着修回来**。相当于"先打草稿再检查一遍"。

### 机制
- `draft_mode="revise"`:两阶段 —— draft(快速粗答案) + revise(扰动后修订)。
- 关键超参:`draft_corrupt_prob`(噪声强度)、`draft_revise_weight`(修订 loss 权重)、`draft_block_size`。
- 修订 loss 鼓励模型从被扰动的草稿恢复出正确答案,训练出"自检"能力。

### 实验证据(mc 口径,parity 是真 win)
| 任务 | mc Δ | n | 说明 |
|---|---|---|---|
| **parity** | **+10pp** (mc 0.882→0.984, 5 seed 中 4 个 99.9-100%) | 5 | **唯一抬动 mc 天花板的 win** |
| cifar10 | final +9.9pp(w0p2_cp0p3, 76.9% vs 66.9%); mc 中性 | 6 | 抬 final 不抬 mc |
| mazes | ±1pp(中性) | — | baseline 已 91%, 无空间 |
| sort | 负 + 退化(revise_weight 参数对 sort inert) | — | wiring 残缺 |

### 关键洞察
- **parity 是 draft-revise 的菜**:parity = 64 长二进制序列奇偶校验,需要逐步累加 XOR,容易中间算错、需要回头核对 —— 正是"草稿-修订"的对路场景。
- **revise 同时抬 final 和 mc**(parity 上),这点和 JEPA 不同(JEPA 只抬 final)。
- **副作用极小**:cifar10/mazes 上 delta 在 ±1pp(seed 噪声内),低风险。
- **parity 数据是 partial(130-160k/200k)**,但 mc 已近顶(99%),结论可信。

### 代码
- `baseline/utils/dtt_ideas.py` — draft_mode 实现
- `baseline/tasks/*/train.py` — 接线(注意:parity/sort 上 revise_weight 曾漏接,见 AGENTS.md 两步走教训)

---

## 第三块:Sparsity — 稀疏激活(效率优化)

### 一句话
每一步"思考"只激活 top-k 比例的神经元(`topk_neurons = r`),用更少算力换可接受的掉点。**这是效率方法,不是涨点方法**,必须看 Pareto(精度 vs 算力),不能只比精度。

### 机制
- `topk_neurons = r`:每个 tick 只有 r 比例的 NLM 神经元被激活更新。
- **算力模型**:NLM 思考回路做 ~r 的功,省 (1-r) 的 NLM 算力。**backbone(resnet)不稀疏化**,端到端墙钟加速 < (1-r),需 sparse kernel 才变现。

### 实验证据(Pareto 前沿, mazes 是唯一完整 sweep)
| r(算力比例) | 省算力 | mazes 精度 | 掉点 |
|---|---|---|---|
| **0.10** | **90%** | 90.3% | **-0.9pp** ← Pareto 甜点 |
| 0.25 | 75% | 90.7% | -0.5pp |
| 0.50 | 50% | 90.0% | -1.2pp |
| 0.75 | 25% | 89.3% | -1.9pp(且有个 0.866 离群) |
| 1.00(稠密) | 0% | 91.2% | baseline |

Pareto 前沿 = {r=0.1, r=0.25},都未被更便宜且更准的点支配。

### 关键洞察
- **mazes r=0.1 省 90% NLM 算力只掉 0.9pp** —— 教科书级效率 trade-off。低 r 区(r≤0.3)几乎免费。
- **任务差异大**:mazes 大 win;**sort 大坑**(r=0.5 掉 12pp, sort 需全量神经元)。
- **诚实前提**:省的是 NLM(CTM 的递归思考部分,是 CTM 区别于普通 CNN 的核心开销),不是 backbone。要真正变现需 sparse kernel 实现。
- cifar10/parity 缺完整 r sweep 数据,暂不判。

### 代码
- CTM forward 的 `topk_neurons` 路径(neuron_select_type)
- `paper/sparsity_efficiency.py` — Pareto 分析

---

## 三块的论文叙事

| 块 | 性质 | 衡量口径 | 证明什么 |
|---|---|---|---|
| **JEPA** | 涨精度(final) | final-tick | CTM 的隐状态可被正则化得更连贯,改善最后一步预测 |
| **revise** | 涨精度(final+mc) | mc | CTM 的"草稿-修订"思考能攻克需要核对的任务(parity) |
| **sparsity** | 省算力 | Pareto(精度 vs r) | CTM 的神经元稀疏激活是廉价的(低 r 几乎免费) |

三个方法**互补不冲突**:JEPA 改善表示质量,revise 改善推理过程,sparsity 改善效率。可在不同任务/不同目标下分别启用。

### 共同的边界 / 失败模式(写论文要诚实交代)
- **parity 是 idea 坟场**:JEPA/halt/EMA/reflex 全 killed(0-iter),只有 revise 活下来。parity_backbone 与辅助机制不兼容。
- **sort 极度脆弱**:tick1-25 全崩(需 tick50),所有 combo idea 退化到魔数,heads/sparsity/nst 参数对 sort inert(wiring 残缺)。
- **mc 机制本身是最大红利**:qamnist final 36%→mc 99%,这归 CTM 架构,不归任何 idea。
