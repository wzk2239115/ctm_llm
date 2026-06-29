# 实验日志 (EXPERIMENT_LOG)

按日期**倒序**追加(最新在最上)。每次提交/分析一批实验,在这里留一段记录。
目的: 任何时候翻这一个文件,就能看懂"为什么跑、什么时候跑、当时的思路、结论是什么"。

记录规范见 `AGENTS.md`「实验记录规范」。

---

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
