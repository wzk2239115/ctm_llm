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
- 我们诚实地给出**边界**:parity 对多数辅助机制不兼容(JEPA/halt/EMA/reflex 全部 0-iter 崩溃,仅 draft-revise 与 sparsity 存活);sort 极度脆弱。

---

## 1 引言

CTM 的核心计算是一个递归的 NLM 回路:给定编码后的输入,回路迭代 `iterations` 步(称为 thought-tick),每步基于"同步表示"(synchronisation representation)产出一份预测与一个置信度。这与一次性给出答案的前馈网络本质不同——CTM 的"思考过程"本身是一个可被优化、可被正则化、可被压缩的对象。

**衡量口径的重要性。** CTM 在每个 tick 都有预测,于是有两种自然打分:
- **final-tick**:只看最后一个 tick 的精度(传统口径,最低);
- **most-certain-tick (mc)**:对每个样本挑其整个思考过程中"最自信"那一步,再平均(CTM 论文标准头条,显著更高)。

两者差距可以极大(qamnist 上 final ~39% 而 mc ~99.6%)。因此**判别任何方法的效果,必须在同一口径下、对照 matched 多 seed baseline 比较**。本文一律以 mc 为主指标,以 final-tick / per-tick 曲线作补充说明。

**贡献。**
1. 在统一 5-seed + mc 口径下,验证 JEPA / draft-revise / sparsity 三种方法的效果与边界;
2. 用 per-tick 精度曲线(figS)首次清晰刻画"抬曲线不抬天花板"这一机制;
3. 给出 sparsity 的算力-精度 Pareto 前沿(figE)及任务依赖的硬边界;
4. 诚实给出各方法的失败模式与任务边界,为后续 CTM 研究提供可复现基准。

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

**动机。** 与其"想完直接交",不如"先打草稿、再故意扰动、学着修回",训练模型对自身中间状态的自检与抗扰能力。

**机制。** 在 CTM 的 $T$ 步思考回路中插入一个草稿边界(tick $b-1$,$b$=`draft_block_size`)。到达边界时把当前预测记为草稿 $\hat{y}_{draft}$(stop-gradient),并以概率 $p$=`draft_corrupt_prob` 对神经元状态注入高斯扰动;模型随后需从被扰动状态继续思考、修订出答案。草稿预测被额外监督为正确答案,迫使模型早期就 commit,同时靠后半段学习抗扰:

$$h_{b} \leftarrow h_{b} + \epsilon,\quad \epsilon \sim \mathcal{N}(0,\sigma^2 I)\ \text{以概率}\ p$$

$$\mathcal{L}_{revise} = \mathcal{L}_{task} + \lambda \cdot \mathrm{CE}\big(\hat{y}_{draft},\ y\big),\qquad \hat{y}_{draft}=\mathrm{sg}(\hat{y}_{b-1})$$

```
算法 1: draft-revise 前向与训练
输入: 初始状态 h_0, 块大小 b, 扰动概率 p, 噪声尺度 σ, 权重 λ
1: for t = 0 … T-1:
2:    h_{t+1}, ŷ_t ← NLM_step(h_t)              # 思考一步, 得新状态与该 tick 预测
3:    if t == b-1:                              # 草稿边界
4:       ŷ_draft ← sg(ŷ_t)                      # 记草稿 (stop-grad)
5:       if rand() < p:                         # 以概率 p 扰动状态
6:          h_{t+1} ← h_{t+1} + ε,  ε ~ N(0, σ²·I)
7: L_task ← CTM 的 anytime / mc 损失(全程 {ŷ_t})
8: 返回 L = L_task + λ · CE(ŷ_draft, y)
```

超参:$b$=`draft_block_size`、$p$=`draft_corrupt_prob`、$\lambda$=`draft_revise_weight`、$\sigma{=}0.1$(固定)。仅训练时介入,**推理走标准 CTM 回路,零额外开销**。

### 3.3 稀疏激活 sparsity(效率优化)

**动机。** NLM 的递归思考是 CTM 区别于普通前馈网络的核心开销。若每个 tick 只让一小部分神经元参与更新,能否以可接受的掉点换大幅省算力?

**机制。** 每个 tick,先由 NLM 算出候选状态更新 $h^{(t)}\in\mathbb{R}^{D}$($D$=神经元数),再对其施加 **hard top-$k$ 掩码**——按绝对值保留最大的 $k$ 个分量、其余置零,作为进入下一 tick 的状态:

$$\tilde{h}^{(t)} = h^{(t)} \odot \mathbf{m}^{(t)},\qquad m^{(t)}_i = \mathbb{1}\!\left[|h^{(t)}_i| \geq \tau^{(t)}\right]$$

$$\tau^{(t)} = \texttt{k-th largest of}\ |h^{(t)}|,\qquad k = \max\!\big(1,\ \lfloor r\cdot D \rfloor\big)$$

其中 $r$=`topk_neurons` $\in (0,1]$ 为激活比例;$r{=}1$ 即退化为标准(稠密)CTM。

**算力模型。** NLM 思考回路每 tick 做 $\sim r$ 的功,省 $(1-r)$ 的 NLM 算力。需注意:**backbone(如 resnet)不做稀疏化**,故端到端墙钟加速 $< (1-r)$,真正变现需 sparse kernel。本文报告的是 NLM 算力比例 $r$ 并明确标注该前提。

---

## 4 实验设置

- **任务**:cifar10(视觉分类,resnet18-1, T=50)、qamnist(视觉问答, T)、parity(64 位序列奇偶校验, T=75)、mazes(网格导航, resnet34-2, T=75)、sort(排列, T=50)。sort 因参数接线残缺仅作辅助。
- **seed**:每配置 5 seed。
- **主指标**:mc(`best_test_acc_mc`)。baseline 为各任务的 matched 5-seed 复现值。
- **per-tick 数据**:来自 checkpoint 的 `test_accuracies`(每 tick 的精度数组,由 `scripts/extract_per_tick.py` 读取);cifar10/mazes 存数组,parity/qamnist 存标量。
- 数据: `paper_repro/csv_data/repro_summary_0728.csv`、`per_tick_0728.json`。

---

## 5 结果与分析

### 5.1 JEPA / draft-revise 不抬动 mc 天花板

**结果。** 表与图 1 给出各任务 5-seed 的 most-certain-tick(mc)精度:

- **cifar10**:baseline 84.23±0.68%、JEPA 84.44±0.14%、draft-revise 84.06±0.90%——三者完全重叠,任二者之差 < 0.4pp,远小于 seed 间标准差。
- **mazes**:90.05 / 89.99 / 90.29%,差值 ≤ 0.3pp,同样落在噪声内。
- **parity**:baseline 97.02±5.11%(方差大,呈双峰),draft-revise 97.93±3.02% 仍落在该噪声带内;JEPA 在 parity 上 0-iter 崩溃(§5.4)。
- **qamnist**:baseline 99.57%、JEPA 99.55%,几乎逐位相同(mc 已近上界)。

| 任务 | baseline mc | JEPA mc | draft-revise mc |
|---|---|---|---|
| cifar10 | 84.23±0.68 | 84.44±0.14 | 84.06±0.90 |
| mazes   | 90.05±0.95 | 89.99±1.29 | 90.29±0.48 |
| parity  | 97.02±5.11 | (killed, 0-iter) | 97.93±3.02 |
| qamnist | 99.57±0.09 | 99.55±0.15 | (未跑) |

![图1 mc天花板](../runs/figures/ctm_paper/figMC_ceiling_0728.png)

**图 1.** mc 天花板不变:各任务下 baseline / JEPA / draft-revise 的 mc 全部重叠于噪声内(灰色 = 该任务上方法未跑或退化;parity-JEPA 退化至随机 ~50%, 见图 4)。

**分析。** 这是一个跨任务的稳健结论:两种辅助机制都无法抬动 mc 天花板。原因在于 mc 的本质——它取每个样本在整个思考过程中"最自信的一步",本质上由模型**容量**(表征能力上界)决定;JEPA 的跨 tick 一致性正则与 draft-revise 的草稿扰动只塑形思考轨迹,不增加容量,故抬不动这个上界。值得对照的是,mc 相对"跨 tick 均值"的巨大跃升(如 qamnist ~39%→99.6%)本身来自 CTM "挑最自信步"的架构红利,与这两种方法无关。正因天花板不动,下一节转向 per-tick 曲线,考察它们到底改变了什么。

### 5.2 per-tick 曲线:抬后期 tick, 不抬峰值

**结果。** 图 2 给出 cifar10 / mazes 上 5-seed 平均的 per-tick 精度曲线(每个 thought-tick 的测试精度)。以 cifar10 为例观察到三个现象:

- **曲线非单调**:baseline 精度随 tick 上升,在 tick≈19 达峰 75.3%,随后**末 tick 反跌到 45.7%**——模型在思考后期出现退化。
- **mc 恒居曲线之上**:mc★ 84.1% 高于任何单一 tick,因为它是逐样本挑最优步,必然 ≥ 任何固定 tick。
- **JEPA / draft-revise 抬高曲线右段**:二者末 tick 从 45.7% 升到 65.3%(+19.6)/ 62.6%(+16.9),但峰值(75→79 / 74)与 mc★(~84)基本不动——曲线被"右段抬升",天花板未破。

| 模型 | 末 tick(final-tick) | 峰值 tick | mc★ |
|---|---|---|---|
| baseline | 45.7% | 75.3%(@tick 19) | 84.1% |
| JEPA     | **65.3%(+19.6)** | 79.4%(@tick 16) | 84.3% |
| draft-revise | **62.6%(+16.9)** | 73.8%(@tick 18) | 83.0% |

![图2 per-tick签名](../runs/figures/ctm_paper/figS_per_tick_signature_0728.png)

**图 2.** per-tick 精度签名(cifar10 左 / mazes 右)。曲线非单调(中段见顶、末 tick 反跌),mc★ 坐在其上;JEPA / draft-revise 抬高曲线右段,峰值与 mc★ 不动。

**分析。** 末 tick 退化的根源在训练目标:CTM 用 most-certain 损失优化"最自信步",并未要求末 tick 最优,模型可在后期松懈甚至漂移。JEPA 的跨 tick 一致性正则逼相邻 tick 隐状态可预测,直接抑制后期漂移;draft-revise 的"扰动后修回"同理,强制后半段轨迹稳健——故二者都把曲线右段抬起来。但峰值由容量决定(§5.1),正则/自检增加不了容量,故峰值与 mc 不动。换言之,这两种方法的真实价值不在"想得更准",而在"**想得更稳**——更多 tick 达到可用精度",这对 any-time / 早停推理有直接意义。mazes 因 baseline 已 ~90% 饱和、曲线本就平坦无退化可修,看不到此效应。

**JEPA 的超参与设计验证。** 我们在历史 ctm_paper 数据(3-seed)上对 JEPA 做两组补充分析(图 4、5)。**权重 sweep**:cifar10 呈明显甜点——$w{=}0.1$ 时与 baseline 持平(84.7 vs 85.2%),$w\geq 0.5$ 掉到 ~77%(−8pp);mazes/qamnist 因饱和而全程持平;**parity 在任意 $w$ 下都退化到 49.9%(二分类随机水平)**。**消融(cifar10)**:完整方法 84.7%;去 stop-gradient 降到 83.8%;**cosine 换 MSE 直接崩到 41.0%**;predict_delta=1 降到 77.4%。

![图4 JEPA权重sweep](../runs/figures/ctm_paper/figJW_jepa_weight_0728.png)

**图 4.** JEPA 权重 sweep(st04, 3-seed)。cifar10 有甜点 $w{=}0.1$,高权重反噬;parity 在任意权重下退化为随机(虚线=50%)。

![图5 JEPA消融](../runs/figures/ctm_paper/figJA_jepa_ablation_0728.png)

**图 5.** JEPA 消融(cifar10)。cosine 损失是关键(MSE 崩至 41%),stop-grad 有正向贡献——验证 §3.1 的两道防坍塌防线。

这说明 JEPA 是"弱正则":仅在小 $w$ 下不伤主任务。消融则证实了设计——只约束方向(cosine)、不回传梯度到 target(sg),才能避免隐状态坍塌;一旦换成 MSE,表示退化、精度崩溃。parity 在所有 $w$ 下退化到随机,进一步暴露其 backbone 与跨 tick 正则的结构性不兼容(§5.4)。

### 5.3 sparsity:感知任务廉价, 算法任务有硬边界

**结果。** 表与图 3 给出 mc 相对 baseline 的变化(Δpp)对 NLM 激活比例 r 的关系(5-seed):

- **mazes**:r 从 0.75 降到 0.10,Δ 在 −0.5 ~ +0.6pp 之间,全程贴近 baseline;即便 r=0.1(省 90% NLM 算力)也只掉 0.5pp。
- **cifar10**:同样平稳,Δ 在 −0.4 ~ +0.3pp,r=0.75 实测 −0.1pp。
- **parity**:r=0.1 时 mc **暴跌 7.0pp**(97.0→90.0);但 r≥0.25 即恢复(+3.0 / −1.1 / +0.7),存在一个位于 (0.1, 0.25] 的相变边界。

| 任务 | r=0.10 | 0.25 | 0.50 | 0.75 | baseline mc |
|---|---|---|---|---|---|
| mazes   | −0.5 | +0.2 | +0.6 | +0.2 | 90.0% |
| cifar10 | −0.4 | +0.1 | +0.3 | −0.1 | 84.2% |
| parity  | **−7.0** | +3.0 | −1.1 | +0.7 | 97.0% |

![图3 sparsity Pareto](../runs/figures/ctm_paper/figE_sparsity_pareto_0728.png)

**图 3.** sparsity 的算力-精度 Pareto(mc vs NLM 算力比例 r)。mazes / cifar10 全程贴近 baseline★(±0.6pp, 近乎免费);parity 在 r=0.1 处暴跌 7pp(算法任务硬边界)。

**分析。** 感知任务对稀疏鲁棒,是因为其精度主要来自 **backbone**(resnet 已提取充分视觉特征),NLM 递归部分仅做精炼——即使每 tick 只激活 10% 神经元,backbone 加少量递归信号仍足以维持精度,说明这类任务的"思考"存在冗余。parity 恰相反:它需要对 64 位序列**逐步 XOR 累加**,运行中的奇偶状态必须**分布在整个神经元群体**上;激进稀疏(10% 神经元)会丢失中间累积,故灾难性掉点,只有保留足够比例(r≥0.25)才能维持分布式累加表示。由此得到一条可操作规则:**感知类任务可大胆压缩 NLM(省 50–90% 算力近乎免费);需全神经元分布式协作的算法类任务须保持稠密。** 再次强调,"省算力"指 NLM 层面,backbone 不稀疏化,端到端墙钟收益需 sparse kernel 才能变现。

**效率的第二根轴:思考步数 $T$(图 6)。** CTM 的算力同时取决于"每 tick 激活多少神经元"(sparsity, 图 3)和"思考多少 tick"(iterations)。历史 tick sweep(st02)显示:**parity 的 mc 随 $T$ 近似单调上升**(tick1 的 65% → tick50 的 92%),印证 XOR 累加需足够思考步;**cifar10 则非单调**(更多 tick 并非更好)。这与 sparsity 的发现同向——parity 对算力两轴都敏感,感知任务两端的"思考"都有冗余。两条轴共同构成 CTM 效率优化的完整空间。

![图6 tick sweep](../runs/figures/ctm_paper/figTS_tick_sweep_0728.png)

**图 6.** 思考步数 $T$ sweep(st02, 3-seed)。parity 强依赖 $T$(XOR 累加需多步);cifar10 非单调。与 sparsity(图 3)并列为效率的两条轴。

### 5.4 边界与失败模式

- **parity 对多数辅助机制不兼容**:JEPA / halt / EMA / reflex 全部 0-iter 崩溃(break 训练)。仅 draft-revise 与 sparsity 存活。parity_backbone 与这些辅助机制不兼容。
- **sort 极度脆弱**:tick1–25 全崩(需 tick50);所有组合 idea 退化到几个魔数,heads/sparsity/nst 参数对 sort inert(wiring 残缺)。
- **mc 红利归架构不归 idea**:qamnist 跨 tick 均值 ~39% 而 mc 99.6%,这个巨大跃升是 CTM "挑最自信步"机制本身带来的,不是任何 idea 的功劳。

---

## 6 讨论

### 6.1 衡量口径的选择

CTM 的 mc 天然显著高于 final-tick(qamnist 上 99.6% vs ~39%),这个跃升是"挑最自信步"机制本身带来的架构红利,与方法无关。因此在评估优化方法时,本文坚持:(1) 以 mc 为主指标;(2) 所有比较对照 matched 5-seed baseline、在同一口径下进行;(3) 用 per-tick 精度曲线(figS)补充刻画方法对思考过程形状的影响,而非只看单一标量。这一口径选择可避免把架构红利误归给具体方法。

### 6.2 三种方法的互补定位

| 方法 | 性质 | 主指标 | 核心结论 |
|---|---|---|---|
| JEPA | 改 per-tick 曲线 | mc(平)+ figS | 不抬天花板, 抬后期 tick |
| draft-revise | 改 per-tick 曲线 | mc(平)+ figS | 抬后期 tick, mc 天花板不动 |
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
