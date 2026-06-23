# TODO

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
