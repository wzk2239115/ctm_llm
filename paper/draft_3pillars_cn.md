# 面向连续思维机的三种优化方法:跨 tick 一致性、草稿-修订与稀疏激活

**(中文初稿 · 三支柱版 · 2026-08-04 — 审稿硬伤已修)**

---

## 摘要

连续思维机(Continuous Thought Machine, CTM)以一种"神经元-局部-记忆"(NLM)的递归回路作为核心计算单元,在输入与输出之间进行多次"内部思考 tick",在每个 tick 都产出一份预测。这种"边想边答"的结构带来一个独有的红利:对同一个样本,可以挑选模型"最有把握"的那一步作答,即 **most-certain-tick(mc)**精度,它显著高于只看最后一步的 final-tick 精度。

本文在统一的 5-seed、mc 口径下,系统研究三种面向 CTM 的优化方法:(1)**跨 tick JEPA**——用相邻 tick 隐状态可预测性作辅助损失,让思考连贯;(2)**草稿-修订(draft-revise)**——先出草稿、加扰动后再学着修回,训练"自检"能力;(3)**稀疏激活(sparsity)**——每个 tick 只激活 top-k 比例神经元,以更少算力换可接受掉点。

核心发现:
- **JEPA 与 draft-revise 塑形思考轨迹的不同部位, 都不构成容量突破**: JEPA 把 per-tick 曲线的后期部分显著抬高(cifar10 末 tick +19.6pp, "想得更稳"); draft-revise(去 detach 修复后 deep supervision 生效)走相反路线——在草稿边界 tick≈1 极早承诺(~84.8%, 接近 mc 天花板), 代价是后期退化, mc 仅微抬(+0.5pp)。
- **稀疏激活(sparsity, 本文实现为真稀疏 NLM 计算)对感知任务近乎免费**:mazes / cifar10 在 r∈[0.1, 0.75] 全程精度变化在 ±0.9pp 以内,可省 50–90% 的 NLM 算力;**而算法任务 parity 各 r 全部劣于 baseline**(r=0.1 崩 −16pp, r≥0.25 也 −1.5~−4pp)——硬边界, 无甜点(早期 post-hoc 掩码版本的 r=0.25 "+3pp" 在真稀疏下消失, 证实其为事后掩码假象)。
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

关键超参包括:思考步数 `iterations`、NLM 神经元数 `d_model`、突触深度、backbone 类型等。

---

## 3 方法

### 3.1 跨 tick JEPA(隐状态一致性正则)

**动机。** CTM 的各 tick 之间没有显式约束,思考过程可能跳跃。我们希望"相邻 tick 的隐状态可预测",迫使思考连贯。

**机制。** 训练一个轻量、无偏置的 MLP 预测器 `synch[t] → predicted synch[t+1]`,以 cosine 相似度作损失,作为辅助损失加入主任务:

$$\mathcal{L} = \mathcal{L}_{task} + w \cdot \cos\_\text{loss}(\text{pred}(synch[t]),\ sg(synch[t+1]))$$

两道防坍塌防线:(1) target 端 stop-gradient;(2) cosine 只约束方向不约束幅度。预测器**仅训练时参与,零推理开销**。

### 3.2 草稿-修订 draft-revise(早期承诺 + 噪声鲁棒)

**动机。** 与其"想完直接交",不如"先打草稿、再故意扰动、学着修回"——一方面用辅助监督迫使模型在思考早期就给出承诺(early commitment),另一方面靠扰动训练后半段轨迹从扰动中恢复的能力(noise-robust revision)。

**机制。** 在 CTM 的 $T$ 步思考回路中插入一个草稿边界(tick $b-1$,$b$=`draft_block_size`)。到达边界时取当前预测作草稿 $\hat{y}_{draft}$(**保留梯度**),并以概率 $p$=`draft_corrupt_prob` 对神经元状态注入高斯扰动;模型随后从被扰动状态继续思考、修订出答案。草稿预测被额外以交叉熵监督为正确答案,梯度回传至边界前的网络(deep supervision / early commitment);扰动则训练后半段轨迹的鲁棒性:

$$h_{b} \leftarrow h_{b} + \epsilon,\quad \epsilon \sim \mathcal{N}(0,\sigma^2 I)\ \text{以概率}\ p$$

$$\mathcal{L}_{revise} = \mathcal{L}_{task} + \lambda \cdot \mathrm{CE}\big(\hat{y}_{draft},\ y\big),\qquad \hat{y}_{draft}=\hat{y}_{b-1}\ \ \text{(live, 梯度回传)}$$

```
算法 1: draft-revise 前向与训练
输入: 初始状态 h_0, 块大小 b, 扰动概率 p, 噪声尺度 σ, 权重 λ
1: for t = 0 … T-1:
2:    h_{t+1}, ŷ_t ← NLM_step(h_t)              # 思考一步, 得新状态与该 tick 预测
3:    if t == b-1:                              # 草稿边界
4:       ŷ_draft ← ŷ_t                           # 记草稿(保留梯度, 作 deep supervision)
5:       if rand() < p:                         # 以概率 p 扰动状态
6:          h_{t+1} ← h_{t+1} + ε,  ε ~ N(0, σ²·I)
7: L_task ← CTM 的 anytime / mc 损失(全程 {ŷ_t})
8: 返回 L = L_task + λ · CE(ŷ_draft, y)
```

超参:$b$=`draft_block_size`、$p$=`draft_corrupt_prob`、$\lambda$=`draft_revise_weight`、$\sigma{=}0.1$(固定)。仅训练时介入,**推理走标准 CTM 回路,零额外开销**。

### 3.3 稀疏激活 sparsity(效率优化)

**动机。** NLM 的递归思考是 CTM 区别于普通前馈网络的核心开销。若每个 tick 只让一小部分神经元参与更新,能否以可接受的掉点换大幅省算力?

**机制。** CTM 的 NLM 由 $D$ 个**逐神经元独立**的线性层(SuperLinear)构成——这是真稀疏计算的关键:每个神经元 $i$ 的更新只依赖它自己的记忆轨迹, 与其他神经元无关, 故可**先选神经元、再只算选中的**。每个 tick, 用 synapse 输出 $s^{(t)}\!\in\!\mathbb{R}^{D}$ 的幅值选出 top-$k$ 个重要神经元(全 batch 共享同一 active 集 $\mathcal{I}^{(t)}$, $|\mathcal{I}|=k$, $k=\max(1,\lfloor rD\rfloor)$), 仅对它们 gather 记忆轨迹并跑 SuperLinear, 其余神经元本 tick 不计算、置零:

$$\mathcal{I}^{(t)} = \operatorname{Topk}\!\left(\tfrac{1}{B}\textstyle\sum_b |s^{(t)}_{b,i}|,\ k\right),\qquad \tilde{h}^{(t)}_i = \begin{cases}\mathrm{NLM}_i\!\big(\mathrm{trace}^{(t)}_i\big), & i\in\mathcal{I}^{(t)}\\ 0, & \text{otherwise}\end{cases}$$

其中 $r$=`topk_neurons` $\in (0,1]$ 为激活比例, $r{=}1$ 即退化为标准(稠密)CTM。**全 batch 共享 active 集**是为了让 gather 成为规则的列切片(而非逐样本稀疏索引), 从而 einsum 真正只算 $k$ 列、省 $\sim\!k/N$ 的 FLOPs, 无需自定义 sparse kernel。

**算力模型。** 本文实现的是**真稀疏 NLM 计算**(非事后掩码): 每个 tick 仅对 $k$ 个 active 神经元跑 SuperLinear, einsum 在 $k$ 列上, **理论上省 $\sim\!k/N$ 的 NLM FLOPs**(NLM 由逐神经元独立的 SuperLinear 构成, 故可只算选中神经元)。隔离微基准(仅 NLM 前向, $B{=}64, N{=}1024$)实测 $r{=}0.25$ 时 dense→sparse **1.71x** 加速。需注意 **backbone(resnet)不稀疏化**, 故端到端墙钟加速 < NLM 加速; 本文报告的 $r$ 既是 NLM 激活比例也是 NLM 算力比例。

---

## 4 实验设置

- **任务**:cifar10(视觉分类,resnet18-1, T=50)、qamnist(视觉问答, T)、parity(64 位序列奇偶校验, T=75)、mazes(网格导航, resnet34-2, T=75)、sort(排列, T=50)。sort 因参数接线残缺仅作辅助。
- **seed**:每配置 5 seed。
- **主指标**:mc(`best_test_acc_mc`)。baseline 为各任务的 matched 5-seed 复现值。
- **per-tick 数据**:来自训练时存档的每个 tick 测试精度数组(`test_accuracies`);cifar10/mazes 存数组,parity/qamnist 存标量。

---

## 5 结果与分析

### 5.1 mc 天花板: 谁能抬动它?

**结果。** 表与图 1 给出各任务 5-seed 下, 三种方法各自最佳配置的 mc:

- **cifar10 / mazes**:baseline、JEPA、draft-revise、sparsity(任意 r)四者完全重叠(差值 ≤ 0.9pp),均落在 seed 噪声内。
- **parity**:baseline 97.02±4.57%(方差大、呈双峰);JEPA 退化为随机(~50%);draft-revise ~97.9% 仍在噪声带内(⚠️ parity 的草稿 CE 因输出/标签 shape 不匹配被跳过, 故 parity revise 实为纯噪声注入, 详见 §3.2/局限);**真稀疏 sparsity 在 parity 上各 r 全部劣于 baseline**(r=0.1 崩到 80.65%, −16pp; r=0.25/0.5/0.75 也分别 −1.5/−4.1/−1.8pp), 没有甜点——这是算法任务的硬边界。
- **qamnist**:baseline 99.57%、JEPA 99.55%(mc 已近上界, 重叠)。

| 任务 | baseline | JEPA | draft-revise | sparsity(best r) |
|---|---|---|---|---|
| cifar10 | 84.23±0.61 | 84.44±0.12 | **84.68±0.19** | 84.76 (r=0.25, +0.5) |
| mazes   | 90.05±0.85 | 89.99±1.15 | 90.04±1.16 | 90.35 (r=0.5, +0.3) |
| parity  | 97.02±4.57 | ~50(退化) | ~97.9† | **全负**(r=0.25 最佳 95.47, −1.5; r=0.1 崩 −16) |
| qamnist | 99.57±0.08 | 99.55±0.13 | (饱和) | (未跑) |

† parity revise 的草稿 CE 被 shape guard 跳过, 实为纯噪声注入, 与 cifar10/mazes 的"deep supervision"非同一方法。

![图1 mc天花板](../runs/figures/ctm_paper/figMC_ceiling_fixes.png)

**图 1.** mc 天花板:JEPA 不抬动 mc;draft-revise 去 detach 修复后在 cifar10 上微抬 mc(84.68, 且种子方差显著收窄), 主要价值是极早承诺的 anytime 收益(§5.2);sparsity 在感知任务近免费、在算法任务 parity 上各 r 全部劣于 baseline(硬边界, 无甜点)。灰色 = 退化至随机或未跑。

**分析。** 这把"天花板"分成几种情况。**(1) JEPA 不抬动 mc; draft-revise(去 detach 修复后)在 cifar10 上微抬 mc(+0.45pp, 且种子方差显著收窄)**:二者的跨 tick 正则/草稿监督主要**塑形思考轨迹**(§5.2)、不增加容量——JEPA 把后期 tick 抬稳, draft-revise 把承诺推到极早(tick≈1)。draft-revise 的 mc 微抬可看作 deep supervision 稳定了训练, 但幅度小、不构成"容量突破"。需强调 mc 上界受容量、优化与校准等多因素共同影响, 本文不主张"mc 仅由容量决定"。**(2) sparsity 在 parity 上没有甜点, 各 r 全部劣于 baseline**:旧 post-hoc 掩码曾显示 r=0.25 反而 +3pp(像正则甜点), 但**改用真稀疏 NLM 计算后该现象消失**——r=0.25 实为 −1.5pp, r=0.1 崩 −16pp, r=0.5/0.75 也都负。这说明旧"+3pp"是 post-hoc(算完稠密结果再置零)的假象; 真正只算 k 个神经元的稀疏计算下, parity(XOR 累加需全量神经元)处处受损。硬边界更干净: 感知任务近免费, 算法任务无甜点。注意 mc 相对"跨 tick 均值"的巨大跃升(如 qamnist ~39%→99.6%)是 CTM"挑最自信步"的架构红利,与方法无关。

### 5.2 per-tick 曲线:JEPA 抬后期 / draft-revise 极早承诺

**结果。** 图 2 给出 cifar10 / mazes 上 5-seed 平均的 per-tick 精度曲线(每个 thought-tick 的测试精度)。以 cifar10 为例观察到三个现象:

- **曲线非单调**:baseline 精度随 tick 上升,在 tick≈19 达峰 75.3%,随后**末 tick 反跌到 45.7%**——模型在思考后期出现退化。
- **mc 通常居曲线之上**:mc★ 84.1% 经验上高于任何单一固定 tick 的精度。需注意这**不是数学必然**——mc 按**置信度(归一化熵)**逐样本挑 tick, 与看标签挑的 oracle-best 不同; 它高于峰值固定 tick, 依赖"高置信预测更可能正确"这一校准性。若模型对错误答案过度自信, mc 完全可能低于峰值固定 tick(故后续工作应补 ECE / 可靠性图)。
- **两种方法作用在曲线的不同部位(机制各异)**:**JEPA 抬高曲线右段**(末 tick 45.7%→65.3%, +19.6pp), 抑制后期漂移; 而 **draft-revise(去 detach 修复后 deep supervision 真正生效)在草稿边界 tick≈1 就冲到 ~84.8%**(baseline tick1 仅 ~46%), 实现**极早承诺**, 代价是后期退化更快(末 tick 反跌到 41.1%)。二者 mc★ 都 ~84–85, 未破天花板。

| 模型 | 末 tick(final-tick) | 峰值 tick | mc★ |
|---|---|---|---|
| baseline | 45.7% | 75.3%(@tick 19) | 84.1% |
| JEPA     | **65.3%(+19.6)** | 79.4%(@tick 16) | 84.3% |
| draft-revise | **41.1%(−4.6)** | **84.8%(@tick 1)** | 84.5% |

![图2 per-tick签名](../runs/figures/ctm_paper/figS_per_tick_signature_fixes.png)

**图 2.** per-tick 精度签名(cifar10 左 / mazes 右)。baseline 曲线非单调(中段见顶、末 tick 反跌), mc★ 坐在其上。两种方法机制清晰可分: **JEPA 抬高曲线右段**(后期更稳); **draft-revise 在 tick≈1 极早承诺**(deep supervision, ~84.8% 接近 mc 天花板), 后期则退化更快。二者 mc★ 都 ~84–85。

**分析。** 两种方法对思考轨迹的作用机制截然不同, 却都没真正抬动 mc 天花板。**JEPA**(跨 tick 一致性正则)逼相邻 tick 隐状态可预测, 直接抑制 baseline 的后期漂移, 故把曲线**右段**抬起来(末 tick +19.6pp)——"想得更稳"。**draft-revise**(草稿 CE 的 deep supervision, 去 detach 修复后真正生效)在草稿边界 tick≈1 就把预测逼到 ~84.8%(接近 mc 天花板), 实现**极早承诺**——这对 any-time 推理(几乎零思考就答对)有直接价值; 但早期强承诺加状态扰动使后期轨迹退化更快(末 tick 反低于 baseline)。二者 mc★ 都 ~84–85, 印证"正则/监督塑形轨迹而非容量上界"。mazes 因 baseline 已饱和、曲线平坦, 二者都看不出明显效应。

**JEPA 的超参与设计验证。** 我们在历史 ctm_paper 数据(3-seed)上对 JEPA 做两组补充分析(图 4、5)。**权重 sweep**:cifar10 呈明显甜点——$w{=}0.1$ 时与 baseline 持平(84.7 vs 85.2%),$w\geq 0.5$ 掉到 ~77%(−8pp);mazes/qamnist 因饱和而全程持平;**parity 在任意 $w$ 下都退化到 49.9%(二分类随机水平)**。**消融(cifar10)**:完整方法 84.7%;去 stop-gradient 降到 83.8%;**cosine 换 MSE 直接崩到 41.0%**;predict_delta=1 降到 77.4%。

![图4 JEPA权重sweep](../runs/figures/ctm_paper/figJW_jepa_weight_0728.png)

**图 4.** JEPA 权重 sweep(st04, 3-seed)。cifar10 有甜点 $w{=}0.1$,高权重反噬;parity 在任意权重下退化为随机(虚线=50%)。

![图5 JEPA消融](../runs/figures/ctm_paper/figJA_jepa_ablation_0728.png)

**图 5.** JEPA 消融(cifar10)。cosine 损失是关键(MSE 崩至 41%),stop-grad 有正向贡献——验证 §3.1 的两道防坍塌防线。

这说明 JEPA 是"弱正则":仅在小 $w$ 下不伤主任务。消融则证实了设计——只约束方向(cosine)、不回传梯度到 target(sg),才能避免隐状态坍塌;一旦换成 MSE,表示退化、精度崩溃。parity 在所有 $w$ 下退化到随机,进一步暴露其 backbone 与跨 tick 正则的结构性不兼容(§5.4)。

### 5.3 sparsity:感知任务廉价, 算法任务有硬边界

**结果。** 表与图 3 给出 mc 相对 baseline 的变化(Δpp)对 NLM 激活比例 r 的关系(5-seed):

- **mazes**:r 从 0.75 降到 0.10,Δ 在 −0.9 ~ +0.3pp 之间,全程贴近 baseline;即便 r=0.1(省 90% NLM 算力)也基本不掉点。
- **cifar10**:同样平稳甚至略正,Δ 在 +0.05 ~ +0.53pp。
- **parity**:**真稀疏下各 r 全部劣于 baseline**——r=0.1 暴跌 16.4pp, r=0.25/0.5/0.75 也分别 −1.6/−4.1/−1.8pp。**没有甜点**:旧 post-hoc 掩码曾显示 r=0.25 反而 +3pp, 但改用真稀疏 NLM 计算后该现象消失(说明旧"+3pp"是"算完稠密结果再置零"的假象, 见 §3.3/§5.1)。算法任务硬边界干净:处处受损, 无正则收益。

| 任务 | r=0.10 | 0.25 | 0.50 | 0.75 | baseline mc |
|---|---|---|---|---|---|
| mazes   | −0.0 | −0.9 | +0.3 | +0.1 | 90.0% |
| cifar10 | +0.1 | +0.5 | +0.3 | +0.1 | 84.2% |
| parity  | **−16.4** | −1.6 | −4.1 | −1.8 | 97.0% |

![图3 sparsity Pareto](../runs/figures/ctm_paper/figE_sparsity_pareto_fixes.png)

**图 3.** sparsity 的算力-精度 Pareto(mc vs NLM 算力比例 r, 真稀疏计算)。mazes / cifar10 全程贴近 baseline(±0.9pp, 近乎免费);parity 各 r 全部低于 baseline(硬边界, 无甜点——旧 post-hoc 的 r=0.25 "+3pp" 在真稀疏下消失)。

**分析。** 感知任务对稀疏鲁棒,是因为其精度主要来自 **backbone**(resnet 已提取充分视觉特征),NLM 递归部分仅做精炼——即使每 tick 只激活 10% 神经元,backbone 加少量递归信号仍足以维持精度,说明这类任务的"思考"存在冗余。parity 恰相反:它需要对 64 位序列**逐步 XOR 累加**,对 NLM 递归状态高度敏感;只算 k 个神经元的真稀疏会破坏全神经元协作,故各 r 都掉点(r=0.1 最甚)。掉点的确切机制(中间累积丢失 / top-k 梯度不连续 / LayerNorm 与稀疏状态不兼容 / 关键神经元被淘汰)本文不展开, 留作后续 probe 分析。**关于"甜点"的更正**:本研究早期用 post-hoc top-k 掩码(算完稠密 NLM 再置零)时, 曾观察到 parity r=0.25 反而 +3pp 的"正则甜点"; 但改用真稀疏 NLM 计算(§3.3)后该现象消失——这证实旧"+3pp"是事后掩码的假象(稠密计算已发生, 掩码只改了传给下一 tick 的状态), 真正省算力的稀疏计算下 parity 无甜点。可操作规则:**感知类任务大胆压缩 NLM(真省 50–90% NLM 算力近乎免费); 算法类任务(NLM 必需)稀疏处处有代价, 须权衡精度 vs 算力, 不存在"白赚"的 r。** 本文 sparsity 为真稀疏 NLM 计算(§3.3, 隔离微基准 1.71x NLM 前向加速, 理论省 ~k/N FLOPs); backbone 不稀疏化, 端到端加速 < NLM 加速。

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
| JEPA | 改 per-tick 曲线 | mc(平)+ figS | 不抬天花板, 抬后期 tick(更稳) |
| draft-revise | 改 per-tick 曲线 | mc(微抬)+ figS | tick≈1 极早承诺(deep superv.), mc 微抬+0.5pp |
| sparsity | 省算力 | Pareto(mc vs r) | 感知任务近免费; **parity 各 r 全负(硬边界, 真稀疏下无甜点)** |

三者优化目标不同(JEPA / draft-revise 塑形 per-tick 轨迹, sparsity 压算力并在 parity 上起正则), 原则上可叠加; 但**本文未做组合实验**(如 JEPA+sparsity、revise+sparsity、三者同开), 是否真互补留作后续验证——当前结论仅限各自单独启用。

### 6.3 局限

- per-tick 曲线目前仅 cifar10 / mazes 有(parity/qamnist 的 checkpoint 存标量);算法任务的曲线形态待补。
- sparsity 已是真稀疏 NLM 计算(§3.3, 隔离微基准 1.71x NLM 前向加速, 理论省 ~k/N FLOPs), 但 backbone 仍稠密, 故端到端加速 < NLM 加速; 真正端到端变现需把稀疏化下沉到 backbone 或借助 sparse kernel。
- parity 上 draft-revise 的草稿 CE 因输出/标签 shape 不匹配被跳过(§3.2), 故 parity revise 实为纯噪声注入, 与 cifar10/mazes 的"deep supervision + 噪声"不完全是同一方法, 跨任务比较时需注意。
- JEPA 未抬动 mc; draft-revise 在 cifar10 上微抬 mc(+0.5pp, 主要收益是 anytime 极早承诺); sparsity 在 parity 上各 r 全负(真稀疏下硬边界, 无甜点)。本文不主张 mc 仅由容量决定(正则/优化/校准亦可影响); 但历史 sweep 显示扩容量(增大 $d_{model}$、增加 tick 数)能系统性抬动 mc(cifar10 d_model2x +1.7pp、parity tick50→100%), 该方向不在本文三种"优化方法"目标内。

---

## 7 结论

本文在统一、严格的 5-seed + mc 口径下,刻画了面向 CTM 的三种优化方法。核心结论有三:(1)JEPA 与 draft-revise 塑形思考轨迹的不同部位、不构成容量突破——JEPA 抬后期 tick(更稳), draft-revise(去 detach 修复后)在 tick≈1 极早承诺(anytime), mc 至多微抬;(2)sparsity(本文实现为**真稀疏 NLM 计算**)对感知任务近乎免费(省 50–90% NLM 算力); 算法任务 parity 上各 r 全部劣于 baseline(硬边界——早期 post-hoc 版本的 r=0.25 "+3pp 甜点"在真稀疏下证伪);(3)我们给出 parity/sort 的失败模式与各方法的任务边界,为后续 CTM 优化研究提供可复现基准。

---

## 8 相关工作

- **连续思维机(CTM)。** 本文以 CTM 的"神经元-局部-记忆 + 同步表示 + 多 thought-tick"递归架构为研究对象, 沿用其 most-certain-tick 评测协议。
- **表征一致性 / 跨步可预测性。** §3.1 的跨 tick JEPA 辅助损失借鉴自 JEPA / BYOL / SimSiam 一族的"可预测性 + stop-gradient + 余弦"自监督范式, 区别在于我们把它施加在 CTM **相邻 thought-tick 的同步表示**上, 目标是稳态思考轨迹而非下游表征。
- **深度监督 / anytime 预测 / 早停。** §3.2 draft-revise 的"边界 tick 草稿 + 交叉熵监督"属于 deep supervision / early-commitment 一类; 其"思考早期就能给出可用预测"与 anytime / early-exit 文献目标一致。扰动后修订则与去噪 / state-noise 训练同源(注意本文是随机高斯扰动, 非对抗最坏情况扰动)。
- **动态稀疏 / 条件计算。** §3.3 的逐 tick top-k 神经元激活属动态稀疏范畴, 与 top-k RNN、MoE 路由、结构化 block 稀疏相关; 本文的特别之处在于利用 CTM NLM 的**逐神经元独立**结构实现真 gather/scatter 稀疏计算(非事后掩码), 无需自定义 sparse kernel 即可获得实测加速。

## 参考文献(占位, 待补全)

1. Stripes et al. *Continuous Thought Machines.* (CTM 原论文)
2. LeCun et al. *A Path Towards Autonomous Machine Intelligence (JEPA).* 2022.
3. Grill et al. *Bootstrap Your Own Latent (BYOL).* NeurIPS 2020.
4. Chen & He. *Exploring Simple Siamese Representation Learning (SimSiam).* CVPR 2021.
5. Lee et al. *Deeply-Supervised Nets.* AISTATS 2015.
6. Bengio et al. *Scheduled Sampling /anytime prediction.* 2015.
7. Fedus et al. *Switch Transformers / Mixture-of-Experts.* 2022.
8. Mocanu et al. *Scalable Training of Artificial Neural Networks with Sparse Networks.* 2018.

---

## 图表索引

- **图 1(figMC)**:mc 天花板——JEPA 不抬动; draft-revise 微抬(cifar10); sparsity 感知任务近免费、parity 硬边界。
- **图 2(figS)**:per-tick 签名——JEPA 抬后期 tick / draft-revise 极早承诺(tick≈1), mc★ 都~84-85。
- **图 3(figE)**:sparsity 的算力-精度 Pareto(mazes/cifar10/parity)。
- **图 4(figJW)**:JEPA 权重 sweep——cifar10 有甜点、parity 退化随机。
- **图 5(figJA)**:JEPA 消融——cosine 损失关键(MSE 崩至 41%)。
- **图 6(figTS)**:思考步数 sweep——效率的第二根轴。
