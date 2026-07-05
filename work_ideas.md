# CTM 验证有效的优化方法 (Work Ideas)

汇总 CTM-LLM 实验中经过验证、口径正确的 work idea。
分为**已定稿三块**（可写论文）与**新候选一个**（待补 seed 入定稿）。

> 衡量口径前提（强制）：CTM 有两种打分 —— **final-tick**（只看最后一步）和 **most-certain-tick**（每样本挑最自信的一步，简称 mc）。后者天然更高（qamnist final 36% → mc 99%），这是 CTM 机制本身的红利，不是 idea 的功劳。判 idea 必须 mc 比 mc、final 比 final，且 baseline 用 st00 **复现值**（`csv_data/ctm_paper_summary.csv`），不能用旧常量。

---

## 一、Cross-Tick JEPA — 让思考连贯（隐状态正则化） ✅ 定稿

### 一句话
训练一个轻量预测器，约束"相邻 tick 的隐状态可预测"（`synch[t] → synch[t+1]`），作为辅助损失。逼模型**连贯地思考**而非随机跳变。

### 机制
- 轻量 bias-free MLP：`synch[t] → predicted synch[t+1]`，cosine loss 约束方向。
- **两道防坍塌防线**：stop-gradient on target + cosine（只约束方向不约束幅度）。
- 作为**辅助损失**加到主任务 loss，**零推理开销**（predictor 只训练时用）。

### 实验证据（final-tick 口径，公平 mc-vs-mc / final-vs-final）
| 任务 | 最佳配置 | final-tick Δ | mc Δ | n |
|---|---|---|---|---|
| cifar10 | w=0.1 | **+7.5pp** (74.4% vs 66.9%) | -0.5pp（中性） | 3 |
| qamnist | w=0.5 | **+17pp** (53.7% vs 36.6%) | 中性 | 2 |
| mazes | w=0.1 | -1.2pp（中性） | -1.2pp | 3 |
| parity | — | killed（0-iter，break 训练） | — | — |
| sort | — | 崩溃/退化 | — | — |

### 关键洞察
- **JEPA 提升的是 final-tick，不抬 mc 天花板**。符合设计：让"更多 tick 都对"，但最优 tick 由容量决定。
- **任务相关**：视觉感知任务受益（隐状态质量是瓶颈）；序列任务不兼容，parity 直接 break 训练。

### 代码
`baseline/utils/jepa.py`、`model/model_ctm_llm.py:1255`、`baseline/tasks/*/train.py`

---

## 二、Draft-Revise — 草稿-修订（对抗式自检） ✅ 定稿

### 一句话
CTM 正常是"思考完直接交答案"；draft-revise 改成**先出草稿，再故意加噪声扰动，学着修回来** —— 相当于"先打草稿再检查一遍"。

### 机制
- `draft_mode="revise"` 两阶段：draft（快速粗答案）+ revise（扰动后修订）。
- 关键超参：`draft_corrupt_prob`（噪声强度）、`draft_revise_weight`（修订 loss 权重）、`draft_block_size`。
- 修订 loss 鼓励模型从被扰动的草稿恢复出正确答案，训练出"自检"能力。

### 实验证据（mc 口径，parity 是真 win）
| 任务 | mc Δ | n | 说明 |
|---|---|---|---|
| **parity** | **+10pp** (mc 0.882→0.984, 5 seed 中 4 个 99.9-100%) | 5 | **唯一抬动 mc 天花板的 win** |
| cifar10 | final +9.9pp（w0p2_cp0p3, 76.9% vs 66.9%）；mc 中性 | 6 | 抬 final 不抬 mc |
| mazes | ±1pp（中性） | — | baseline 已 91%，无空间 |
| sort | 负 + 退化（revise_weight 参数对 sort inert） | — | wiring 残缺 |

### 关键洞察
- **parity 是 draft-revise 的菜**：64 长二进制序列奇偶校验，需逐步累加 XOR，容易中间算错、需要回头核对 —— 正是"草稿-修订"的对路场景。
- **revise 同时抬 final 和 mc**（parity 上），这点和 JEPA 不同（JEPA 只抬 final）。
- **副作用极小**：cifar10/mazes 上 delta 在 ±1pp（seed 噪声内），低风险。

### 代码
`baseline/utils/dtt_ideas.py`、`baseline/tasks/*/train.py`

---

## 三、Sparsity — 稀疏激活（效率优化） ✅ 定稿

### 一句话
每一步"思考"只激活 top-k 比例的神经元（`topk_neurons = r`），用更少算力换可接受的掉点。**这是效率方法，不是涨点方法**，必须看 Pareto（精度 vs 算力），不能只比精度。

### 机制
- `topk_neurons = r`：每个 tick 只有 r 比例的 NLM 神经元被激活更新。
- **算力模型**：NLM 思考回路做 ~r 的功，省 (1-r) 的 NLM 算力。**backbone（resnet）不稀疏化**，端到端墙钟加速 < (1-r)，需 sparse kernel 才变现。

### 实验证据（Pareto 前沿，mazes 是唯一完整 sweep）
| r（算力比例） | 省算力 | mazes 精度 | 掉点 |
|---|---|---|---|
| **0.10** | **90%** | 90.3% | **-0.9pp** ← Pareto 甜点 |
| 0.25 | 75% | 90.7% | -0.5pp |
| 0.50 | 50% | 90.0% | -1.2pp |
| 0.75 | 25% | 89.3% | -1.9pp（且有 0.866 离群） |
| 1.00（稠密） | 0% | 91.2% | baseline |

Pareto 前沿 = {r=0.1, r=0.25}，都未被更便宜且更准的点支配。

### 关键洞察
- **mazes r=0.1 省 90% NLM 算力只掉 0.9pp** —— 教科书级效率 trade-off，低 r 区（r≤0.3）几乎免费。
- **任务差异大**：mazes 大 win；**sort 大坑**（r=0.5 掉 12pp，sort 需全量神经元）。
- **诚实前提**：省的是 NLM（CTM 的递归思考部分，区别于普通 CNN 的核心开销），不是 backbone。真正变现需 sparse kernel 实现。

### 代码
CTM forward 的 `topk_neurons` 路径（neuron_select_type）、`paper/sparsity_efficiency.py`

---

## 四、Gate-JEPA — acc 门控自适应权重（免调参） 🔄 新候选

### 一句话
给 JEPA 辅助 loss 加一个 acc 门控 sigmoid：**主任务没学会时关掉 JEPA，学会再放开约束**。免去人工调 `cross_tick_jepa_weight`，且比 fixed 更好更稳。

### 机制
- `eff_w = base·sigmoid((acc_ema − τ) / T)`
  - `acc_ema`：跨 step EMA 的训练 acc buffer
  - `τ`：开启阈值（主任务学到阈值才放开 JEPA）
  - `T`：温度
- 主任务没学会（acc<τ）→ JEPA 关闭；学会 → 放开。
- 对比另外两个自适应方案（都失败了）：
  - **A balance** `eff_w = clip(ratio·L_main / (L_jepa+ε), lo, hi)`：parity 卡 chance（52.8%）、sort 大方差（38%±18）。
  - **C uncertainty**（Kendall 可学习 σ）：parity/sort 全崩（50%/2.6%），优化器把 σ 推向破坏训练的值。
- **3 个 adaptive 方案里只有 gate 成立**。

### 实验证据（mc-vs-mc 口径，st25，base_weight=0.1）
| task | mode | n | acc% | vs fixed | vs st00 | seeds | 状态 |
|---|---|---|---|---|---|---|---|
| parity | fixed | 3 | 94.06 | 0 | +5.85 | [82.2, 100, 100] | 未复现压制（st04 5k 步判 STALLED 是过早） |
| parity | **gate** | 3 | **97.71** | **+3.64** | **+9.50** | [93.1, 100, 100] | **WIN — 最佳，抬 mc 天花板** |
| parity | balance | 3 | 52.79 | -41.3 | -35.4 | [52.5, 52.8, 53.0] | BROKEN（chance） |
| parity | uncertainty | 3 | 50.63 | -43.4 | -37.6 | [50.5, 50.6, 50.8] | BROKEN |
| sort | fixed | 3 | 80.94 | 0 | -6.59 | [76.1, 81.6, 85.1] | 低于 baseline |
| sort | **gate** | 3 | **88.25** | **+7.31** | +0.72 | [87.0, 88.1, 89.6] | **WIN vs fixed**，追平 st00 |
| cifar10 | fixed/balance/gate | 3 | 84.1~84.5 | ≈0 | -0.7~-1.1 | — | 中性，全 ≈ st00（85.16） |
| mazes | fixed/balance/gate | 1~3 | 89.2~90.8 | ≈0 | -0.3~-2.0 | — | 中性，全 ≈ st00（91.17） |

### 关键洞察
- **gate 是少数能抬 mc 天花板的方法**（堪比 revise-on-parity 的 +10pp），parity 上 vs fixed +3.6pp、vs st00 +9.5pp（n=3）。
- **种子分布严格优于 fixed**：gate (93/100/100) vs fixed (82/100/100)，方向稳健。
- **语义自洽**：先学主任务再约束表示，既"免调参"又比 fixed 更好；balance/uncertainty 反而劣于 fixed。
- **诚实前提**：+9.5pp 依赖 2/3 种子触顶，需补 seed 3/4/5 把 n 抬到 5~6 才能坐实。

### 代码
`baseline/utils/jepa.py:AdaptiveJEPAController`（mode="gate" 路径）、`paper/analyze_st25.py`、数据 `csv_data/st25_0701_summary.csv`

---

## 四块的论文叙事

| 块 | 性质 | 衡量口径 | 证明什么 | 状态 |
|---|---|---|---|---|
| **JEPA** | 涨精度（final） | final-tick | CTM 的隐状态可被正则化得更连贯，改善最后一步预测 | ✅ 定稿 |
| **revise** | 涨精度（final+mc） | mc | CTM 的"草稿-修订"思考能攻克需要核对的任务（parity） | ✅ 定稿 |
| **sparsity** | 省算力 | Pareto（精度 vs r） | CTM 的神经元稀疏激活是廉价的（低 r 几乎免费） | ✅ 定稿 |
| **gate-JEPA** | 涨精度（mc）+ 免调参 | mc | CTM 的辅助约束应"主任务优先"，acc 门控优于固定权重 | 🔄 待补 seed |

四个方法**互补不冲突**：
- JEPA 改善**表示质量**（final）
- revise 改善**推理过程**（mc，parity 类核对任务）
- sparsity 改善**效率**（Pareto）
- gate-JEPA 改善**训练动力学**（mc，免调参）

可在不同任务/不同目标下分别启用。

---

## 共同的边界 / 失败模式（写论文要诚实交代）
- **parity 是 idea 坟场**：JEPA（fixed w=0.1 短看）/ halt / EMA / reflex 全 killed 或 STALLED，只有 revise 和 gate-JEPA 活下来。
- **sort 极度脆弱**：tick1-25 全崩（需 tick50），所有 combo idea 退化到魔数，heads/sparsity/nst/revise 参数对 sort inert（wiring 残缺）。
- **mc 机制本身是最大红利**：qamnist final 36% → mc 99%，这归 CTM 架构，不归任何 idea。

## 待办
- [ ] gate-JEPA 补 parity seed 3/4/5，把 +9.5pp 的 n 从 3 抬到 5~6，坐实 headline（对照 revise-on-parity n=5），然后入 `VERIFIED_CONCLUSIONS.md` 定稿。
- [ ] 查 st25 qamnist 为何全缺 / mazes_gate 为何只 1 seed，决定是否补跑。
- [ ] cifar10/parity 补 sparsity r sweep 补全 Pareto 图（figE）。
