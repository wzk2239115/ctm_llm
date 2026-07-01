# 实验日志 (EXPERIMENT_LOG)

按日期**倒序**追加(最新在最上)。每次提交/分析一批实验,在这里留一段记录。
目的: 任何时候翻这一个文件,就能看懂"为什么跑、什么时候跑、当时的思路、结论是什么"。

记录规范见 `AGENTS.md`「实验记录规范」。

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
- **结果**: 待算力机收菜。
- **结论**: 待。
- **下一步**:
  1. 算力机快速验证: `python paper/diagnose_parity_jepa_compute.py --steps 5000 --timeout 2400`(看 ABC 哪个解除 parity 压制)
  2. 算力机长跑: `python scripts/experiment_plan_ctm_paper.py submit --stage st25 --no-wait`(60 runs, 多 task 多 seed)
  3. 收菜后用 mc-vs-mc 口径对比 fixed vs adaptive(注意 parity 的 final_iter==0 过滤), 出 fig 看 Pareto(精度 vs 是否免调参)

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
