# 实验日志 (EXPERIMENT_LOG)

按日期**倒序**追加(最新在最上)。每次提交/分析一批实验,在这里留一段记录。
目的: 任何时候翻这一个文件,就能看懂"为什么跑、什么时候跑、当时的思路、结论是什么"。

记录规范见 `AGENTS.md`「实验记录规范」。

---

## 2026-07-05 — GCBC 离线范式 (对齐 stable-wm) + ExpertPolicy

- **思路(用户提供)**: 之前用 PPO-from-scratch + random-BC 收集训练 policy, 数据和 world-model 不一致(不公平) + random action 是噪声(BC 学随机行为=毒, pendulum-partial ctm 从 75→27.8)。用户指出: 应该参考 stable-wm——expert/oracle 数据收集一次, world-model 和 policy baseline **共享同一批数据**。学完 stable-wm + lewm 确认: 数据是 **expert 预收集**(非 random), 所有方法(world-model GCBC/GCIQL)读同一份。
- **核心改造**:
  - `worldmodel/policy.py:ExpertPolicy`: 11 env 启发式 oracle (PD/energy/scripted/push), 访问 env 内部 state, noise=0.1 增加 diversity. 对齐 stable-wm `ExpertPolicy(action_noise=)`.
  - `worldmodel/rl/gcbc.py:GCBCTrainer`: goal-conditioned BC, episode-level BPTT (max_seq_len=32 截断), expert buffer 共享. evaluate + evaluate_timed.
  - `paper/run_offline_compare.py`: 统一实验脚本 (collect→GCBC→eval).
  - CartPole goal_x 从 ±1.5 缩到 ±0.5 (大范围 expert 不可能).
  - Reacher partial 真正区分 (full=ee+joints 4dim, partial=ee-only 2dim); 之前 partial 被忽略.
- **发现**:
  - Swimmer env 动力学 bug: `thrust=sin(angles)*a`, `angles+=a*dt` → `∫thrust·dt=∫sin(θ)dθ` 是状态函数, 任何振荡回原点净 thrust=0. Swimmer 无法推进, 弃用.
  - Expert 质量: point/tworoom/reacher/mountaincar/acrobot=100%, pendulum=33%, cartpole=33%.
- **配置**: reacher + reacher-partial, 6 backbone (mlp/ctm/lstm/flash/flash-shallow/flash-deep), 3 seeds, collect=100eps, gcbc=1000steps. `csv_data/gcbc_reacher_v2.md`.
- **结果**:
  | env | mlp | ctm | lstm | flash | CTM-RNN |
  |---|---|---|---|---|---|
  | reacher (full) | 70.8 | 72.9 | 70.8 | 70.8 | +2.1 持平 |
  | reacher-partial | 60.4 | 72.9 | 66.7 | 62.5 | **+6.2 CTM 赢** |
  - partial obs 上 mlp 掉 10pp (70.8→60.4), CTM 不掉 (72.9→72.9) — 记忆补偿信息缺失
  - Flash 混合 > shallow (+6.2~+8.3 synergy 确认)
  - GCBC expert data work: expert 100% → GCBC 60-73%
- **结论**: GCBC 离线范式正确 (对齐 stable-wm), 替代了有害的 PPO+random-BC. CTM 记忆在 POMDP 上有真实优势 (partial +6.2 vs full +2.1). Flash 混合 synergy 确认但 flash 本身未赢 ctm (需改进 gate/结构).
- **下一步**: 算力机大规模对比 (全 env × 全 backbone × 多 seed) + world-model+CEM baseline (同一批 expert data) + PushT/Cube expert 修复 (角度控制) + Flash Brain 改进.

---

## 2026-06-30 — 自适应 JEPA 权重 (ABC 三方案) 解除 parity 压制

- **思路(用户提供)**: parity+JEPA w=0.1 全 KILLED 的根因是"约束太大、光顾着约束没学训练数据"——JEPA 辅助 loss 的梯度把 synch 表示推向"可预测下一 tick", 挤掉了 parity 判别信号。用户提: **能否给正则项加一个自适应乘数, 从而不用人工调 `cross_tick_jepa_weight`**。实现 ABC 三方案让实验说话。
- **根因诊断** (`paper/diagnose_parity_jepa_compute.py`): subprocess 调 parity.train, 算力机真实配置(d_model=1024 seq=64 batch=64 ticks=75)跑 5000 步:
  - baseline(jepa=0): 在学, final acc 0.587
  - fixed w=0.1: **acc 卡 0.498(随机), STALLED** — 不是 pool/crash, 是 w=0.1 本身压制主任务
  - fixed w=0.03: 0.559(在学) / w=0.01: 0.581(≈baseline) — 甜点 ≤0.03, 而 st04 sweep [0.1,0.5,1.0] 全超甜点
  - 旧 verdict 有 bug: `acc<0.52 AND loss>0.69` 判 STALLED, 但 JEPA 辅助 loss 会拉低总 loss(0.682<0.69)制造"在学"假象 → w=0.1 被误判 OK。已修为纯 acc 判据。
- **配置**: st25 `build_st25_adaptive_jepa`, 5 task(sort/parity/mazes/cifar10/qamnist) × 4 mode × 3 seed = 60 runs, 全部 base_weight=0.1(parity 压制点)。三方案实现于 `baseline/utils/jepa.py:AdaptiveJEPAController`:
  - **A balance**: `eff_w = clip(ratio·L_main / (L_jepa+ε), lo, hi)`, JEPA 贡献恒≈主任务的 ratio%(默认 0.3)。零额外反传, base_weight 退化为"语义比例"。
  - **B gate**: `eff_w = base·sigmoid((acc_ema−τ)/T)`, 主任务没学会(acc<τ)→JEPA 关闭, 学会→放开。acc_ema 跨 step EMA buffer。
  - **C uncertainty**: Kendall 2018, `loss=exp(−logσ)·L_jepa + logσ`, logσ 是可学习参数(挂 controller, 自动进 optimizer), 模型自己定相对权重。
  - fixed 模式完全向后兼容(走老 CrossTickJEPAPredictor 路径, 零开销)。
- **预期**: parity 上 fixed=STALLED(复现压制), 三个 adaptive 至少一个让 acc 脱离 0.5(解除压制); 其他 task(sort/cifar, w=0.1 本就 OK)上 adaptive 应保持或改善 final acc, 不退化。最看好 A balance(直接对症量级压制)和 B gate(主任务优先)。
- **smoke(本地 CPU 已过)**: 四模式各 5 步 parity.train, 全 exit=0, loss 量级精确反映 adaptive weight 生效: fixed≈0.79 / balance≈0.90(eff_w 0.215) / gate≈0.70(eff_w 0.012 门关) / uncertainty≈1.68(eff_w 1.0)。三方案真实接入。
- **结果**: 2026-07-01 起跑, ~705 收菜。实际回来 **41/60 runs**(qamnist 全缺, mazes_gate 只 1 seed), 全 OK 无 crash。口径 **mc-vs-mc**(cifar10/mazes/parity 用 `best_test_acc_mc`, sort 无 mc 用 `best_test_acc`); st25 各 task 配置已逐项核对 == st00 paper sweep, vs-st00 公平。分析脚本 `paper/analyze_st25.py`, 数据 `csv_data/st25_0701_summary.{csv,md}`。

  | task | mode | n | acc% | vs fixed | vs st00 | seeds | 状态 |
  |---|---|---|---|---|---|---|---|
  | parity | fixed | 3 | 94.06 | 0 | +5.85 | [82.2,100,100] | **未复现压制!**(st04 5k 步判 STALLED 是过早, 200k 收敛后 94%) |
  | parity | balance | 3 | 52.79 | -41.3 | -35.4 | [52.5,52.8,53.0] | **BROKEN**(chance, 正是当初要解的压制) |
  | parity | gate | 3 | **97.71** | **+3.64** | **+9.50** | [93.1,100,100] | **WIN — 最佳, 抬 mc 天花板** |
  | parity | uncertainty | 3 | 50.63 | -43.4 | -37.6 | [50.5,50.6,50.8] | BROKEN(chance) |
  | sort | fixed | 3 | 80.94 | 0 | -6.59 | [76.1,81.6,85.1] | 低于 baseline |
  | sort | gate | 3 | **88.25** | **+7.31** | +0.72 | [87.0,88.1,89.6] | **WIN vs fixed**, 追平 st00 |
  | sort | balance | 3 | 37.96 | -43.0 | -49.6 | [25.0,25.6,63.3] | 不稳(±18) |
  | sort | uncertainty | 3 | 2.60 | -78.3 | -84.9 | [0.8,3.4,3.6] | BROKEN(崩) |
  | cifar10 | fixed/balance/gate | 3 | 84.1~84.5 | ≈0 | -0.7~-1.1 | — | 中性, 全 ≈ st00(85.16) |
  | mazes | fixed/balance/gate | 1~3 | 89.2~90.8 | ≈0 | -0.3~-2.0 | — | 中性, 全 ≈ st00(91.17) |

- **结论**:
  1. **gate(B, acc 门控 sigmoid)是唯一成功的自适应方案, 且是这批的 headline**。parity 上 mc 97.71%, vs fixed +3.6pp、vs st00(无 JEPA)+9.5pp(n=3), 是少数真能抬 mc 天花板的方法(堪比 revise-on-parity 的 +10pp)。sort 上 vs fixed +7.3pp、追平 baseline。cifar10/mazes 上中性(≈ fixed ≈ st00), 不退化。语义自洽: gate 让主任务先学(acc<τ 时 JEPA 关), 学好再放开约束, 既"免调参"又比 fixed 更好。
  2. **balance(A)与 uncertainty(C)在硬任务上反而更糟**。balance 在 parity 卡 chance(52.8%)、sort 大方差(38%±18)——与预期"最看好 balance"完全相反; uncertainty(Kendall 可学习 σ)在 parity/sort 全崩(50%/2.6%), cifar10 final 也塌到 10%, 优化器把 σ 推向破坏训练的值。3 个 adaptive 方案里 2 个劣于 fixed, **只有 gate 成立**。
  3. **实验前提部分落空**: 当初依据是"st04 parity+JEPA w=0.1 全 KILLED", 但 st25 fixed 在 parity 上跑到 200k 收敛后 mc 94%(> st00 88.21%)。即 5k 步诊断(`diagnose_parity_jepa_compute.py`)看到的 0.498 STALLED 是**起步慢、非永久卡死**——纯 acc 判据把"慢启动"误判成"压制"。fixed 实际没压制 parity, 所以"adaptive 抢救 fixed"的故事不成立; 但 gate 仍独立地比 fixed 更好更稳, 是真 win。
  4. **口径提醒**: parity 各模式都有种子不稳(fixed/gate 各有 1 颗种子 ~82-93%、另两颗 ~100%), +9.5pp 依赖 2/3 种子触顶, 需更多 seed 坐实; 但 gate 的种子分布(93/100/100)严格优于 fixed(82/100/100), 方向稳健。
- **下一步**:
  1. (可选)补 parity gate 的 seed 3/4/5, 把 +9.5pp 的 n 从 3 抬到 5~6, 坐实 headline(对照 revise-on-parity n=5)。
  2. 查 qamnist 为何全缺 / mazes_gate 为何只 1 seed, 决定是否补跑。
  3. gate 可考虑入 `VERIFIED_CONCLUSIONS.md`(作为"抬 mc 天花板"的第 4 个方法, 仅次于 revise-on-parity); 先补 seed 再定稿。
  4. 修正 `diagnose_parity_jepa_compute.py` 的 STALLED 判据注释——它判的"压制"实为慢启动, 诊断步数需 ≥20k 才有参考价值。

---

## 2026-06-29 — JEPA 口径修正 + baseline 过时修复

- **思路(用户提供)**: cross_tick_jepa 是验证通过的优化方法, 之前的分析漏了它, 检查一下.
- **排查**: `paper/explain/fig5b_jepa_weight.md` 已结论"JEPA 有效 +9~30pp"(final-tick 口径). 但我之前的 `analyze_deep.py` 只用 mc-vs-mc 把 JEPA 判中性, 是口径偏差.
- **结果**:
  - **JEPA final-tick 真实有效**: cifar10 +7.5pp(n=3), qamnist +17pp(n=2). 这是真的, JEPA 让"相邻 tick 隐状态可预测", 提升最后一步预测.
  - **JEPA 不抬 mc 天花板**: cifar10/mazes/qamnist 的 mc delta 都 -0.5~-1pp(中性). JEPA 改善"平均 tick 质量"而非"最优 tick".
  - **fig5b 报告自己的 bug**: 它说 mazes +9.6pp(最佳场景), 但用了过时 baseline `BASELINE_PAPER["mazes"]=0.8028`(旧值). 真实 st00 mazes=0.9016, 实际 mazes JEPA = -1.2pp(中性, 不是最佳场景). cifar10 baseline(0.6443)也偏低, 真实 0.6690.
- **修复**: `scripts/plot_ctm_paper_results.py:BASELINE_PAPER` 改成 st00 复现值(cifar10 0.6690 / mazes 0.9016 / qamnist 0.3662 / sort 0.8753; parity 0.6797 不变). 修复后 fig5 mazes 的 baseline 线从 80% 升到 90%, JEPA 柱不再"超过".
- **修正后的有效方法清单**:
  | 方法 | 有效 | 真实证据 | 口径 |
  |---|---|---|---|
  | JEPA | ✅ | cifar10 +7.5pp, qamnist +17pp | final-tick |
  | revise | ✅ | parity mc +10pp | mc, n=5 |
  | sparsity | ✅ | mazes 省 90% 算力掉 1pp | Pareto 效率 |
- **下一步**: 教训——判 idea 有效性要先确认指标口径(final-tick vs mc), 且 baseline 必须用同口径的复现值, 不能用过时常量. fig5b 的 md 文本描述尚未同步更新(mazes"最佳场景"那段需改).

## 2026-06-29 — 0629 收菜 + idea 有效性深度分析

- **思路(用户提供)**: 算力机上 ctm_paper st00-st24 + deep 01-04 已经跑了一部分,开始收菜然后做深入分析;后来追问"哪个优化方法有效",并指出 sparsity 要结合"省算力 vs 掉点"看,不能只比精度。
- **配置**: `extract_ctm_paper_results.py --curves`(534 runs, 并行) + `export_paper_results.py` 收 deep 01_revise/02_jepa/03_sparsity/04_combos。
- **预期**: 找到几个 work 的 idea, 出 idea×task 有效性矩阵。
- **结果**:
  - **唯一真 win = revise on parity**: mc 0.882→0.99(+10pp, n=5), 副作用极小(cifar10/mazes ±1pp)。
  - **sparsity mazes 是效率 win**: r=0.1 省 90% NLM 算力只掉 0.9pp(Pareto 甜点)。
  - **踩坑 — mc-vs-final 膨胀**: deep CSV 的 best_acc 是 mc 指标, baseline 是 final-tick, 直接 delta 把 most-certain-tick 机制的 +18pp(cifar10) 算成 idea 功劳; mc-vs-mc 后 cifar10 的 revise/JEPA/combo "+17pp" 全变中性。
  - parity 上 JEPA/halt/EMA/reflex 全 KILLED(0-iter, break 训练); sort 上 heads/sparsity/nst/revise 全 DEGEN(no-op 魔数 0.9253 / 0.6458/0.7917/0.8021)。
- **产物**: `paper/{analyze_0629,idea_validity_0629,analyze_deep,sparsity_efficiency}.py` + figC/figD/figE; AGENTS.md 新增「实验结果分析准则」。
- **下一步**: cifar10/parity 补 sparsity r sweep 补全 figE; qamnist JEPA-pd1/halt0.6 补 seed(现 n=1); 待 st02b/st04b bugfix re-run 补数据。
