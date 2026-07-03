# Flash Brain 研究判读手册 (整理用)

> 定位: CTM 不当 world-model transition (已证伪), 当 **memory policy**. 在 POMDP 上和
> RNN 系公平对标, 验证 CTM 持续记忆 + Flash Brain 多时间尺度的价值.
> 对照基线: mlp/lstm/gru/transformer (memory policy) + cem-jepa/cem-ctm (world-model planner).

---

## 实验 1: memory-policy ablation (精度层)

**脚本**: `paper/run_memory_policy_ablation.py`
**数据**: `csv_data/memory_ablation_results.csv` (注意 overnight 多批覆盖, 完整在 `logs/overnight_A_delay.log`)

### 判读标准
- **CTM/Flash vs RNN 系均值(lstm/gru/transformer)**: 这是核心对标. CTM-RNN > 0 = CTM 记忆优于标准循环.
- **记忆 vs mlp**: 记忆机制是否帮. POMDP 上应正(记忆有用), 全观测上应负(记忆是负担).

### 实际结论 (已确认)
- **pendulum-partial**: flash 86.7 > ctm 58.4 > transformer 41.7 > gru 20 ≈ mlp 20 > lstm 10. CTM/Flash 碾压 RNN 系.
- 全观测 pendulum: mlp 100 最好(记忆是负担), 但 flash 96.7(shallow 救场) ≫ ctm 85(纯 deep 吃亏).
- **CTM 在 POMDP 显著赢 RNN 系** (partial +50pp vs RNN 均值). 论文论点成立.
- robustness: CTM 赢跨 d_model(48/60/67) 和 memory_length(60/73/60), 不是超参侥幸.

### 诚实局限
- CTM 高方差 (10 seed std=21 on partial), vs transformer 的差不极显著, vs lstm/gru 才显著.
- 全观测/短延迟 mlp 最好, CTM 优势严格限于 POMDP.

---

## 实验 2: real-time benchmark (实时层 — Flash Brain 的差异化主场)

**脚本**: `paper/run_realtime_benchmark.py`
**数据**: `csv_data/realtime_benchmark.csv`

### 判读标准
- **latency**: fast policy(flash/ctm/lstm) 应 < 1ms; CEM 应 ms~s. 差 1-2 数量级 = Flash Brain 实时性卖点.
- **deadline-constrained success**: deadline 50→1ms 收紧, fast policy 不掉, CEM 在 5ms 崩(超时 zero action).

### 实际结论 (已确认)
- latency: flash 0.77ms, ctm 0.64, lstm 0.38 (fast); cem-jepa 6.5, cem-ctm 14.6 (slow). **flash 比 cem-ctm 快 ~19x**.
- throughput: flash 1305Hz vs cem-ctm 69Hz.
- deadline: fast 全程不掉(flash pendulum 全 deadline 82.8%); CEM 5ms 崩(pendulum cem 38→10).
- **Flash Brain 在 POMDP 双优**: partial 上 flash 70.6%+0.77ms, world-model 5%+14.6ms(双差).
- 把 "Flash vs world-model" 翻成: **world-model 在实时闭环控制根本不适用, Flash Brain 才是正确架构**.

### 诚实局限
- flash 不是绝对最快(0.77 > lstm 0.38, gate 有开销), 但 fast 整体 vs CEM 差 1-2 数量级, fast 内部差异不重要.
- CEM 在 partial 本身 success 极低(5-6.7%), deadline 效应在 partial 被噪声盖住, 在 pendulum(CEM 38-45%)才看清 5ms 崩.

---

## 实验 3: belief probe (机制层 — 最有新意)

**脚本**: `paper/probe_belief_encoding.py`
**数据**: `logs/overnight_B_probe_*.log` (每 backbone 一个)

### 判读标准
- **R²(θdot)**: linear probe feat → 被遮挡角速度. 高 = hidden 编码了 belief.
- 关键看 **R² vs success 的关系**: 如果"R² 高 → success 高", 编码=利用; 如果"R² 高但 success 低", 编码≠利用(RNN 记住了但 actor 用不上).

### 实际结论 (已确认, 反直觉)
- mlp: R²0.054 succ15 (没编码, 合理)
- ctm: R²**0.945** succ**75** (编码且利用)
- lstm: R²**0.960** succ23 (编码但不用!)
- gru: R²**0.991** succ15 (编码但不用!)
- transformer: R²0.558 succ56.7
- **核心发现**: LSTM/GRU 编码 θdot 的 R² 最高(0.96-0.99)但 success 最低. 推翻"CTM 赢因编码 belief"——RNN 也编码, 但 actor 用不上. CTM 的优势是**编码且有效利用**.
- 论文新意: "belief encoding ≠ belief usage", 标准 RNN 在 POMDP 失败不是"记不住"而是"记住了用不上".

### 诚实局限
- probe R² 单独不能解释 success(LSTM R² 更高但输). 真正机制是"编码-利用耦合度", probe 是间接证据. 更直接: ablate actor 对 hidden 的梯度(待做).

---

## 实验 4: gate adaptation (Flash Brain 自适应 — 负面, 修正叙事)

**脚本**: `paper/probe_gate_adaptation.py`
**数据**: `figures/fig4_gate_adaptation.png`

### 判读标准
- gate z (0=shallow, 1=deep) 应: 全观测低(shallow 主导), POMDP 高(deep 介入). 两 env 分化 = 自适应成立.

### 实际结论 (负面 — 自适应不成立)
- pendulum z=0.226, pendulum-partial z=0.241, Δ=+0.015(噪声级).
- gate 没按任务分化, 学到"两 env 都适度开 deep"(z≈0.23, 77% shallow + 23% deep).
- **修正叙事**: Flash Brain 的价值不是"动态自适应 multi-timescale", 而是 **shallow+deep 固定混合 > 任一单路径**(shallow 救全观测 + deep 救 POMDP 的互补性).
- fig4 不能用"自适应"论点, 改用"混合"论点(见实验 5).

### 为什么没自适应
- 23% 的 deep 贡献对两 env 都有用(全观测 deep 也微调, POMDP deep 推断), gate 没动力分化.
- 想要真自适应: gate 要看外部难度信号(prediction error/reward), 而非看 deep/shallow feat 本身. 待做(error-gate).

---

## 实验 5: flash 混合 ablation (修正 fig4 — 坐实"混合 > 单路径")

**脚本**: `paper/run_memory_policy_ablation.py --backbones mlp ctm flash flash-shallow flash-deep --envs pendulum pendulum-partial`
**状态**: 待跑

### 判读标准
- **flash vs flash-shallow(纯z=0)**: flash > shallow = deep 贡献有用.
- **flash vs flash-deep(纯z=1)**: flash > deep = shallow 贡献有用.
- 两边都 > : **"混合 > 单路径"坐实**, fig4 改用这个.

### 预期
- partial: flash(86.7) > flash-deep(≈ctm 58) > flash-shallow(≈mlp 20). 混合赢.
- pendulum: flash(96.7) > flash-deep(≈ctm 85, deep 吃亏) > flash-shallow(≈mlp 98? shallow 可能反超). 这里要看 flash 是否 ≥ shallow(不退化) + ≥ deep.

### 如果成立
- fig4 论点: Flash Brain = shallow+deep 互补混合, 全观测靠 shallow 不退化, POMDP 靠 deep 推断. 不是动态切换, 是固定互补.
- 这是 fig1(精度)+fig2(实时) 之外的第三层: **架构互补性**.

---

## 四张论文 figure 的论点 (整理用)

| fig | 论点 | 数据源 | 状态 |
|----|------|--------|------|
| fig1 accuracy | CTM/Flash 在 POMDP 精度赢 RNN 系 | memory_ablation | ✅ 成立 |
| fig2 realtime | Flash 比 world-model 快 1-2 数量级, 紧 deadline planner 崩 | realtime_benchmark | ✅ 成立 |
| fig3 mechanism | encoding≠usage (RNN 编码不用/CTM 编码且用) | belief_probe | ✅ 成立(最有新意) |
| fig4 gate | ~~自适应~~ → **混合 > 单路径** | gate_probe + 混合ablation | ⚠️ 修正中(实验5跑完定稿) |

## 整体叙事 (论文主线)

1. **CTM 不该当 world-model transition**(被 CEM 架空, 又慢又没好处) — 排除性结论
2. **CTM 该当 memory policy**: POMDP 上 CTM 赢 RNN 系(fig1) — 正面精度
3. **赢的机制是"有效利用 belief"**: RNN 编码但不用, CTM 编码且用(fig3) — 机制新意
4. **Flash Brain(shallow+deep 混合)比单 CTM 更全面**: 全观测不退化 + POMDP 增益(fig4 修正版) — 架构贡献
5. **Flash Brain 实时性碾压 world-model**: 紧 deadline 下 planner 崩, Flash 撑住(fig2) — 实时差异化

**核心新意**: fig3(encoding≠usage) 是对标准认知的修正; fig2 把 vs world-model 翻成"实时闭环 world-model 不适用". 这两点是主要贡献.

## 待办 (坐实剩余论点)
- [ ] 实验5 (混合 ablation) 跑完, 定稿 fig4
- [ ] (可选) error-gate: gate 看 prediction error 追求真自适应, 若成立是额外亮点
- [ ] (可选) 编码-利用耦合度 ablation: 量化 actor 对 hidden 的梯度, 比 probe 更直接证"利用"
