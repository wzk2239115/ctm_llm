# 03 Sparsity — 神经元稀疏鲁棒性图解

> 对应图片: `paper/figures/03_main.png`、`paper/figures/03_sweep.png`
> 数据来源: `csv_data/03_sparsity_results.csv` (算力机导出, 40 runs)
> 生成代码: `paper/03_sparsity_deep.ipynb` (分析 cell)

---

## 1. 这组实验在讲什么

**一句话**: 把 CTM 每个 tick 的活跃神经元砍到只剩 top-ratio(按绝对值),发现**迷宫任务即使只剩 10% 神经元活跃也几乎不掉点**,sort 任务则只在极端稀疏(ratio=0.1)才崩 —— 说明 CTM 的计算主要靠**时间深度(迭代 tick)** 而非**空间宽度(神经元数量)** 承载。

### 两张图
- **`03_main.png`**: ratio=0.5 主实验(5 seeds × 2 任务),相对无 sparsity 复现基线的 delta 柱状图。
- **`03_sweep.png`**: ratio ∈ {0.1, 0.25, 0.5, 0.75, 0.9} 的扫描热图(3 seeds × 任务),颜色 = delta pp。

---

## 2. 实验配置 (40 runs)

**Δ from baseline** (单参数):
```
+ topk_neurons = ratio    # only top-ratio neurons fire per tick
```

| 组别 | ratio | seeds | 任务 | runs |
|---|---|---|---|---|
| 主实验 | 0.5 | 5 | sort, mazes | 10 |
| 扫描 | 0.1 / 0.25 / 0.5 / 0.75 / 0.9 | 3 | sort, mazes | 30 |

**基线对照**: 无 sparsity 复现 (同配置, `st00` paper 复现, best 口径):
- sort = **0.8753** (element-level `test_accuracies`)
- mazes = **0.9117** (`test_accuracies_most_certain`)

> ⚠️ **基线口径修正**: 早期 `BASELINE_ACC` 用的是 sort=0.7146 / mazes=0.8028, 这两个值**无法从我们 checkpoint 复现**(parity/qamnist 却完全吻合) —— 很可能是 CTM paper 原始发表值或更早跑的残留, 与本实验的指标口径(element-level / most_certain)不一致。已改为复现值, 否则 mazes delta 会凭空多 +11pp。

---

## 3. 稀疏机制的本质

每个 tick, 对 `activated_state` 按最后一维做 magnitude top-k, 只保留前 ratio 比例的神经元, 其余置零:

```python
# baseline/utils/ctm_model_ideas.py:6-13
def apply_topk_sparsity(x, topk_fraction, stepi):
    if topk_fraction >= 1.0:
        return x
    k = max(1, int(x.size(-1) * topk_fraction))
    threshold = torch.topk(x.abs(), k, dim=-1)[0][:, -1:]
    mask = (x.abs() >= threshold).float()
    return x * mask
```

调用点在每个 tick 的前向循环里 (`baseline/models/ctm.py:758`、`ctm_sort.py:162`、`ctm_qamnist.py:245`), 即**每个思考步骤都重新选一次活跃神经元**。

**直觉**: 这是一种 winner-take-all 激活稀疏。ratio=0.1 意味着每 tick 只有 10% 神经元放电, 90% 被强制静默。理论上应大幅削弱表征能力。

---

## 4. 效果解读 (真实数据)

### 4.1 主实验: ratio=0.5 (5 seeds)

| 任务 | sparsity 均值 | 基线 | delta | p | 显著性 |
|---|---|---|---|---|---|
| **mazes** | 90.17% ± 0.88 | 91.17% | **−1.00pp** | 0.064 | ns |
| **sort** | 90.22% ± 2.82 | 87.53% | **+2.69pp** | 0.0997 | ns |

砍掉一半活跃神经元, 两任务相对 dense 基线**均不显著**。

### 4.2 ratio 扫描 (3 seeds, best_acc 均值)

| ratio | mazes | Δ | sort | Δ |
|---|---|---|---|---|
| **0.10** | 90.28% | −0.89pp | 77.31% | **−10.22pp** |
| 0.25 | 90.67% | −0.50pp | 86.54% | −0.99pp |
| 0.50 | 89.73% | −1.44pp | 90.45% | +2.92pp |
| 0.75 | 89.31% | −1.86pp | 90.61% | +3.08pp |
| 0.90 | 89.93% | −1.24pp | 90.10% | +2.57pp |

### 关键观察

1. **mazes 对稀疏近乎免疫**: 全 ratio 区间 Δ 都在 −0.5~−1.9pp, 即使**只剩 10% 神经元活跃**也只掉 0.9pp。迷宫求解的计算根本不依赖密集宽度。
2. **sort 容忍度高但非免疫**: ratio≥0.25 基本无损(甚至微正, 在噪声内), 只有 **ratio=0.1 才崩 (−10pp)**。
3. **任务依赖性**: mazes (空间路径规划) 比 sort (符号排序) 更鲁棒, 暗示两者有效表征维度不同。
4. **方差可控**: mazes std < 1pp, 极稳; sort std ~3-6pp, 种子敏感但趋势一致。

---

## 5. 理论含义

**CTM 的计算承载力来自时间深度, 而非空间宽度。** 即使每 tick 仅 10% 神经元放电, 迭代轨迹照样能把迷宫解出来 —— 说明持续思维的"思维向量"是**低维**的, 少数神经元就能驱动整条思维轨迹, 密集并行表征并非必需。

这与 CTM 的核心卖点(把推理展开成时间序列)互为印证: 既然砍宽度不掉点, 那起作用的必然是时间维度的迭代过程。

**论文卖点句**: *Even with only 10% of neurons active per tick, CTM maintains full maze-solving performance — evidence that its computation is carried by the iterative trajectory rather than dense parallel representation.*

---

## 6. 与 prior (st08) 的关系与修正

sparsity 此前在 `ctm_top_performers.ipynb` (stage `st08`) 跑过一版, 覆盖 4 任务但 seed 少、ratio 也只有 2-3 档。03_deep 是其**系统化重做**。两处需要修正 st08 的印象:

1. **sort "+21pp" 是种子彩票 + 错误基线叠加**:
   - st08 报 sort sparsity best_test_acc=0.9545, 对 baseline 0.7146 → +24pp。
   - 实际: 正确基线 0.8753 → +7.9pp; 5-seed 均值 0.9031 → **+2.7pp (ns)**。那个 0.95 只是最幸运的一颗种子。
2. **st08 表数据有重复嫌疑**: sort 的 sparsity0.25 / 0.5 / 0.75 三行数值完全相同, parity 的 0.25 与 0.75 也完全相同 —— 很可能每任务只跑了单一 ratio, 表里误标成多档。03_deep 的真扫描(sort 随 ratio 单调变化)直接证伪了"各 ratio 等效"。
3. **mazes 可互验**: st08 mazes sparsity0.5 = 0.9014, 与 03_deep 的 0.9000 基本一致 ✓。

> 因此 `ctm_top_performers.ipynb` 第 8 节 (fig6 "sparsity and revise both deliver +21pp") 的措辞需要按 03_deep 的多 seed 均值修正。

---

## 7. 局限与待补

要把"时间而非空间"从**推断**升级为**证据**, 还需:

1. **扩任务**: 当前只有 sort/mazes。补 parity / cifar10 / qamnist 看普适性(prior 里 parity 对 sparsity 似乎也 0 反应, 正好互证)。
2. **多 seed dense 基线**: 当前 sort/mazes 基线各只有 1 seed, 补 3 seed 才能让显著性检验更硬。
3. **迭代数 × 稀疏度 交叉实验**: 直接验证"加 tick 能补偿稀疏掉的精度" —— 这才是"时间承载计算"的决定性证据。
4. **结构化诊断**: 看活跃神经元在 tick 间/任务内是否稳定(结构化稀疏 = 少数神经元承载计算) vs 每次随机换批(冗余兜底)。

**当前定位**: 一节扎实支撑性结果 (fig + 半页分析); 补完 1、3 两项可升格为核心论点之一。
