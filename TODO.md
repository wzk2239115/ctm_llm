# TODO

## CTM memory-policy — 收口分析任务

### 现状结论 (overnight 6 批已跑完)
- CTM 在 POMDP (pendulum-partial 70-78%, partial-delay3 61.7%, delay8 +18.9 vs RNN) 显著赢 RNN 系; 全观测/短延迟 mlp 够 (记忆是负担)
- robustness 通过: CTM 赢跨 d_model (48/60/67 随容量升) 和 memory_length (60/73/60), RNN 系不动
- **probe 反直觉 (最有价值)**: LSTM/GRU 的 R²(θdot) 最高 (0.96/0.99) 但 success 最低 (23/15); CTM R² 0.945 + success 75. 说明 **belief encoding ≠ belief usage** — RNN 编码了 belief 但 actor 用不上, CTM 既编码又利用

### 待办
- [ ] **编码-利用耦合度 ablation** (核心): 量化 actor 对 hidden 的梯度贡献/依赖, 坐实"CTM 利用 belief, RNN 利用失败". 比 linear probe 更直接证明利用 (probe 只证编码)
- [ ] **出三张论文 figure** (从 csv_data 现有结果): ① success×env 性能图 (CTM 在 POMDP 赢) ② R²-vs-success 散点 (RNN 编码不用, CTM 编码且用) ③ d_model scaling 曲线 (CTM 受益 RNN 不动)
- [ ] **CTM 高方差处理**: 10 seed pendulum-partial std=21 (有些 seed 差), 考虑加 seed 到 20 或调 lr/entropy 稳定; vs transformer 的差 ~1std 不极显著需收紧
- [ ] **叙事精修**: 主线改成 "CTM 优势不在编码 belief (RNN 也编码), 而在有效利用 belief 做决策" — encoding-vs-usage 解耦是核心新意

### 相关文件
- 实验: `paper/run_memory_policy_ablation.py` / `paper/probe_belief_encoding.py` / `scripts/overnight_memory.sh`
- 结果: `csv_data/memory_ablation_results.csv` / `logs/overnight_{A..F}_*.log`
- backbone: `worldmodel/rl/memory_policy.py` / `worldmodel/rl/ppo.py`
- env: `worldmodel/envs/__init__.py` (DelayObs + make_env -delayN)

---

## JEPA 在 sort 任务上效果不佳 — 排查计划

### 现象
- cifar10: w=0.1 → +10pp,有效
- mazes: 全权重 → +9pp,有效且稳定
- qamnist: 需确认
- **sort**: 高权重 (0.5/1.0) 反而低于 baseline (w=0.5 → −5pp, w=1.0 → −12pp)

### 待排查项

#### 1. 数据可信度
- [ ] 核实 sort/st04 的 jepa_w0.1/0.5/1.0 是否真的跑了 JEPA 路径 (之前发现 sort 任务多 stage 共享同一组 best_test_acc,怀疑 delta 配置没生效)
- [ ] 检查 `paper/exp_runner.py` 里 st04 在 sort 任务上是否正确传入了 `cross_tick_jepa_weight`
- [ ] 写 audit 脚本扫全表,标出"不同 stage/sweep 但 best_test_acc 完全相同"的行

#### 2. JEPA 机制与 sort 的兼容性
- [ ] sort 是算法任务 (序列排序),latent state 变化模式和视觉任务不同——JEPA "预测下一 tick 隐状态" 假设 latent 有可预测的时序结构,sort 的 state 可能不具备
- [ ] 检查 sort 训练曲线,看 JEPA loss 是否收敛 (辅助 loss 不收敛 = 正则化无效甚至有害)
- [ ] 对比 sort 上 JEPA vs draft-revise: draft-revise +21pp 但 JEPA 负增益,说明 sort 需要的是"多步精炼"而非"latent 可预测性"

#### 3. 权重敏感性
- [ ] sort 上 w=0.1 是否也是正增益 (数据: 0.1→83.4% = +12pp, 0.5→66.4% = −5pp)? 如果是,说明不是 JEPA 本身不兼容,而是 sort 对正则强度极敏感
- [ ] 扩展扫描 w=0.02/0.05,看是否存在更低的 sweet spot

#### 4. 实验复现
- [ ] 重跑 sort/st04 jepa_w0.1 × 3 seeds,确认结果可复现
- [ ] 如果重跑结果和 CSV 不一致,说明历史数据有问题,需全部重跑 st04

### 相关文件
- 数据: `paper/ctm_top_performers_table.csv` / `csv_data/ctm_paper_summary.csv`
- 实验定义: `paper/02_jepa_deep.ipynb`
- 实验运行器: `paper/exp_runner.py`
- JEPA 实现: 搜索 `cross_tick_jepa` (baseline 侧 + LLM 侧)
- 图表: `scripts/plot_ctm_paper_results.py:fig_jepa_weight` / `runs/figures/ctm_paper/fig5_jepa_weight.png`
