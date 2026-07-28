# 面向连续思维机的三种优化方法:跨 tick 一致性、草稿-修订与稀疏激活

**(中文初稿 · 三支柱版 · 2026-07-28)**

> 数据基准: `paper_repro` 5-seed 复现(118 run), 主指标 = most-certain-tick accuracy (mc)。图: figMC(mc 天花板)、figS(per-tick 签名)、figE(sparsity Pareto)。

---

## 摘要

连续思维机(Continuous Thought Machine, CTM)以一种"神经元-局部-记忆"(NLM)的递归回路作为核心计算单元,在输入与输出之间进行多次"内部思考 tick",在每个 tick 都产出一份预测。这种"边想边答"的结构带来一个独有的红利:对同一个样本,可以挑选模型"最有把握"的那一步作答,即 **most-certain-tick(mc)**精度,它显著高于只看最后一步的 final-tick 精度。

本文在统一的 5-seed、mc 口径下,系统研究三种面向 CTM 的优化方法:(1)**跨 tick JEPA**——用相邻 tick 隐状态可预测性作辅助损失,让思考连贯;(2)**草稿-修订(draft-revise)**——先出草稿、加扰动后再学着修回,训练"自检"能力;(3)**稀疏激活(sparsity)**——每个 tick 只激活 top-k 比例神经元,以更少算力换可接受掉点。

核心发现:
- **JEPA 与 draft-revise 都不抬动 mc 天花板**,但会把 per-tick 精度曲线的后期部分显著抬高(cifar10 上末 tick 分别 +19.6 / +16.9 个百分点)。即"让更多 tick 都对",而非"让最优 tick 更好"。
- **稀疏激活对感知任务近乎免费**:mazes / cifar10 在 r∈[0.1, 0.75] 全程精度变化在 ±0.6pp 以内,可省 25–90% 的 NLM 算力;但**算法任务有硬边界**——parity 在 r=0.1 时 mc 暴跌 7pp(它需要全量神经元做 XOR 累加)。
- 我们诚实地给出**边界**:parity 是大多数辅助机制的"坟场"(JEPA/halt/EMA/reflex 全部 0-iter 崩溃,仅 draft-revise 与 sparsity 存活);sort 极度脆弱。

---

## 1 引言

CTM 的核心计算是一个递归的 NLM 回路:给定编码后的输入,回路迭代 `iterations` 步(称为 thought-tick),每步基于"同步表示"(synchronisation representation)产出一份预测与一个置信度。这与一次性给出答案的前馈网络本质不同——CTM 的"思考过程"本身是一个可被优化、可被正则化、可被压缩的对象。

**衡量口径的重要性。** CTM 在每个 tick 都有预测,于是有两种自然打分:
- **final-tick**:只看最后一个 tick 的精度(传统口径,最低);
- **most-certain-tick (mc)**:对每个样本挑其整个思考过程中"最自信"那一步,再平均(CTM 论文标准头条,显著更高)。

两者差距可以极大(qamnist 上 final ~39% 而 mc ~99.6%)。因此**判别任何 idea 的效果,必须用 matched 多 seed baseline、并在同一口径下比较**。本文一律以 mc 为主指标;此前我们曾误把汇总脚本里的 `best_test_acc`(实为"跨 tick 均值",源于一个 `_to_scalar` 的 fallback bug)当作 final-tick 使用,本文已更正(见 §6.1)。

**贡献。**
1. 在统一 5-seed + mc 口径下,验证 JEPA / draft-revise / sparsity 三种方法的效果与边界;
2. 用 per-tick 精度曲线(figS)首次清晰刻画"抬曲线不抬天花板"这一机制;
3. 给出 sparsity 的算力-精度 Pareto 前沿(figE)及任务依赖的硬边界;
4. 诚实记录失败模式与口径陷阱,为后续 CTM 研究提供可复现的基准。

---

## 2 背景:CTM 与内部思考 tick

CTM 的前向过程可概括为:

```
输入 x  →  backbone 编码  →  {for t in 1..T:  NLM 回路更新 → 同步表示 synch[t] → 预测头 → predictions[:, :, t], certainty[t]}
```

其中 `predictions` 形状为 `(B, 类别数, T)`,`certainty` 为 `(B, 2, T)`(归一化熵与 1-归一化熵)。most-certain-tick 即对每个样本取 `certainty[:,1].argmax(-1)` 所指的那一步。

关键超参包括:思考步数 `iterations`、NLM 神经元数 `d_model`、突触深度、backbone 类型等(各任务配置见 `paper/exp_runner.py` 的 `BASE_CONFIGS`)。

---

## 3 方法

### 3.1 跨 tick JEPA(隐状态一致性正则)

**动机。** CTM 的各 tick 之间没有显式约束,思考过程可能跳跃。我们希望"相邻 tick 的隐状态可预测",迫使思考连贯。

**机制。** 训练一个轻量、无偏置的 MLP 预测器 `synch[t] → predicted synch[t+1]`,以 cosine 相似度作损失,作为辅助损失加入主任务:

$$\mathcal{L} = \mathcal{L}_{task} + w \cdot \cos\_\text{loss}(\text{pred}(synch[t]),\ sg(synch[t+1]))$$

两道防坍塌防线:(1) target 端 stop-gradient;(2) cosine 只约束方向不约束幅度。预测器**仅训练时参与,零推理开销**。

### 3.2 草稿-修订 draft-revise(对抗式自检)

**动机。** 与其"想完直接交",不如"先打草稿,再故意扰动,学着修回"。

**机制。** `draft_mode="revise"` 分两阶段:先得到 draft(粗答案),再以概率 `draft_corrupt_prob` 对草稿加噪,用 `draft_revise_weight` 加权的修订损失鼓励模型从扰动中恢复正确答案。关键超参:`draft_corrupt_prob`(噪声强度)、`draft_revise_weight`、`draft_block_size`。

### 3.3 稀疏激活 sparsity(效率优化)

**动机。** NLM 的递归思考是 CTM 区别于普通 CNN 的核心开销。若每个 tick 只让 top-k 比例的神经元被激活更新,能否以可接受的掉点换大幅省算力?

**机制。** 设 `topk_neurons = r`,每个 tick 仅 r 比例的 NLM 神经元参与更新。算力模型:NLM 思考回路每 tick 做 ~r 的功,省 (1−r)。**backbone(resnet)不稀疏化**,故端到端墙钟加速 < (1−r),需 sparse kernel 才能真正变现。本文报告的是 NLM 算力比例 r,并明确标注该前提。

---

## 4 实验设置

- **任务**:cifar10(视觉分类,resnet18-1, T=50)、qamnist(视觉问答, T)、parity(64 位序列奇偶校验, T=75)、mazes(网格导航, resnet34-2, T=75)、sort(排列, T=50)。sort 因参数接线残缺仅作辅助。
- **seed**:每配置 5 seed。
- **主指标**:mc(`best_test_acc_mc`)。baseline 用 repro 的 baseline 组(matched 5-seed),而非历史单 seed 常量。
- **per-tick 数据**:来自 checkpoint 的 `test_accuracies`(每 tick 的精度数组,由 `scripts/extract_per_tick.py` 读取);cifar10/mazes 存数组,parity/qamnist 存标量。
- 数据: `paper_repro/csv_data/repro_summary_0728.csv`、`per_tick_0728.json`。

---

## 5 结果与分析

### 5.1 统一发现:mc 天花板不动(figMC)

| 任务 | baseline mc | JEPA mc | draft-revise mc |
|---|---|---|---|
| cifar10 | 84.23±0.68 | 84.44±0.14 | 84.06±0.90 |
| mazes   | 90.05±0.95 | 89.99±1.29 | 90.29±0.48 |
| parity  | 97.02±5.11 | (killed, 0-iter) | 97.93±3.02 |
| qamnist | 99.57±0.09 | 99.55±0.15 | (未跑) |

**结论**:JEPA 与 draft-revise 在所有跑通的任务上,mc 都落在 baseline 噪声内——**没有任何一种方法抬动了 mc 天花板**(figMC)。parity 上 JEPA 直接 0-iter 崩溃(见 §5.4 坟场)。这是一个统一的"负面"结果:靠辅助正则/自检,提升不了 CTM 的"最优一步"精度。

> 历史更正:此前我们曾报告 draft-revise 在 parity 上"mc +10pp(0.882→0.984),唯一抬动天花板的 win"。5-seed 复现证伪——旧 baseline 0.882 是单颗坏种子(s0),matched 5-seed baseline 为 0.970;对照公平 baseline,revise mc 仅 +0.9pp(噪声内)。该 "+10pp" 是拿 idea 均值去比 baseline 最差种子的口径错误。

### 5.2 但 per-tick 曲线被抬高了(figS)

既然 mc 不动,这两种方法到底改变了什么?per-tick 精度曲线(figS)给出了答案。

以 cifar10 为例(5 seed 均值,最后一次 eval):

| 模型 | 末 tick(final-tick) | 峰值 tick | mc★ |
|---|---|---|---|
| baseline | 45.7% | 75.3%(@tick 19) | 84.1% |
| JEPA     | **65.3%(+19.6)** | 79.4%(@tick 16) | 84.3% |
| draft-revise | **62.6%(+16.9)** | 73.8%(@tick 18) | 83.0% |

CTM 的 per-tick 精度曲线呈**非单调**:先随思考步上升、在中间某 tick 见顶,然后**末 tick 反而下跌**(baseline 从峰值 75% 跌到 45.7%)。mc(84%)坐在整条曲线之上——它是"每个样本各自挑最好的一步",故恒高于任何固定 tick。

JEPA / draft-revise 的作用是:**把曲线的右段(后期 tick)显著抬高**(末 tick +16~20pp),但峰值与 mc 基本不动。直觉解释:JEPA 的跨 tick 一致性正则逼相邻 tick 隐状态可预测 → 后期 tick 不再退化;draft-revise 的"修回"训练同理。两者都让"**更多 tick 都对**",而非"**最优 tick 更好**"——这正是它们抬不动 mc 天花板、却仍值得做的理由(更稳健的思考过程)。

mazes 上曲线近乎平坦(baseline 已 ~90% 饱和,无退化可修),故看不到此效应。

### 5.3 sparsity:感知任务廉价,算法任务有边界(figE)

以 mc 对 r(激活比例)作 Pareto(5 seed):

| 任务 | r=0.10 | 0.25 | 0.50 | 0.75 | baseline mc |
|---|---|---|---|---|---|
| mazes   | −0.5 | +0.2 | +0.6 | +0.2 | 90.0% |
| cifar10 | −0.4 | +0.1 | +0.3 | −0.1 | 84.2% |
| parity  | **−7.0** | +3.0 | −1.1 | +0.7 | 97.0% |

**感知/空间任务(mazes、cifar10)稀疏近乎免费**:r 从 1.0 降到 0.1,mc 变化都在 ±0.6pp 内,却可省 90% 的 NLM 算力。**算法任务 parity 有硬边界**:r=0.1 时 mc 暴跌 7pp——parity 需要对 64 位序列逐步 XOR 累加,激进稀疏会丢失中间累积,只有 r≥0.25 才稳定。

这是本文最可操作的结论:**对感知任务大胆压缩 NLM(省 50–90% 算力几乎不掉点);对需要全神经元协作的算法任务保持稠密**。

> 注:cifar10 r=0.75 曾因 overnight 中断只训到 36%,出现假象 −1.6pp;补满 200k iter 后实测 −0.1pp。sort 的旧"r=0.5 掉 12pp"不复现(−0.3pp),但 sort 仅 1 个 r 且参数 inert 嫌疑,不列为前沿点。

### 5.4 边界与失败模式

- **parity 是 idea 坟场**:JEPA / halt / EMA / reflex 全部 0-iter 崩溃(break 训练)。仅 draft-revise 与 sparsity 存活。parity_backbone 与这些辅助机制不兼容。
- **sort 极度脆弱**:tick1–25 全崩(需 tick50);所有组合 idea 退化到几个魔数,heads/sparsity/nst 参数对 sort inert(wiring 残缺)。
- **mc 红利归架构不归 idea**:qamnist 跨 tick 均值 ~39% 而 mc 99.6%,这个巨大跃升是 CTM "挑最自信步"机制本身带来的,不是任何 idea 的功劳。

---

## 6 讨论

### 6.1 口径陷阱:为什么必须 mc 比 mc

我们在过程中踩了两个口径坑,值得显式记录:

1. **mc-vs-final 混用**:CTM 的 mc 天然比 final-tick 高很多。若拿 idea 的 mc 去比 baseline 的 final-tick,会把架构红利算成 idea 的功劳。
2. **汇总脚本 bug**:`extract_ctm_paper_results.py` 的 `_to_scalar` 对 per-tick 数组 fallback 到 `np.mean()`,使汇总表里的 `best_test_acc` 实为"跨 tick 均值",曾被我们误当作 final-tick。本文已改用 mc 作主指标、per-tick 曲线(figS)作机制说明。

教训:**baseline 必须用 matched 多 seed,且指标定义要从数据原生结构里追到头**,不能盲信汇总中间量。

### 6.2 三种方法的互补定位

| 方法 | 性质 | 主指标 | 核心结论 |
|---|---|---|---|
| JEPA | 改 per-tick 曲线 | mc(平)+ figS | 不抬天花板, 抬后期 tick |
| draft-revise | 改 per-tick 曲线 | mc(平)+ figS | 同上; parity 的 mc +10pp 已证伪 |
| sparsity | 省算力 | Pareto(mc vs r) | 感知任务免费, 算法任务有边界 |

三者互补不冲突:JEPA / draft-revise 让思考过程更稳健(更多 tick 可用),sparsity 让思考更廉价。可在不同任务、不同目标下分别启用。

### 6.3 局限

- per-tick 曲线目前仅 cifar10 / mazes 有(parity/qamnist 的 checkpoint 存标量);算法任务的曲线形态待补。
- sparsity 的"省算力"是 NLM 层面,端到端墙钟加速需 sparse kernel 实现才能真正变现。
- 三种方法均未抬动 mc 天花板——这说明在当前容量下,mc 主要由模型容量决定,辅助正则/自检难以突破。

---

## 7 结论

本文在统一、严格的 5-seed + mc 口径下,刻画了面向 CTM 的三种优化方法。核心结论有二:(1)JEPA 与 draft-revise 虽不抬动 mc 天花板,却显著改善 per-tick 曲线后期(更稳健的思考);(2)sparsity 对感知任务近乎免费、对算法任务有明确边界,提供了可操作的效率指南。我们同时诚实记录了 parity/sort 的失败模式与两处口径陷阱,为后续 CTM 优化研究提供了可复现的基准。

---

## 图表索引

- **figMC** (`runs/figures/ctm_paper/figMC_ceiling_0728.png`):mc 天花板——JEPA/draft-revise 不抬动。
- **figS** (`figS_per_tick_signature_0728.png`):per-tick 精度签名——曲线非单调、mc 坐于其上、JEPA/revise 抬后期 tick。
- **figE** (`figE_sparsity_pareto_0728.png`):sparsity 的算力-精度 Pareto(mazes/cifar10/parity)。

## 代码与数据

- 分析脚本:`paper/analyze_repro_0724.py`、`paper/plot_per_tick_signature.py`、`paper/plot_mc_ceiling.py`、`paper/sparsity_efficiency.py`
- 数据:`paper_repro/csv_data/repro_summary_0728.csv`、`per_tick_0728.json`、`logs_curves_0728.json`
- 已验证结论(详):`VERIFIED_CONCLUSIONS.md`
