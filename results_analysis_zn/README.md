# CTM-LLM 实验结果分析

本文件夹记录了首轮完整的 71 个 CTM-LLM 扫描实验及其后续 49 个稀疏扫描实验。

数据来源：从 `runs/metrics` 导出的 `summary.csv`。
稀疏数据来源：从 `runs/metrics` 导出的 `sparsity_summary.csv`。

筛选规则：
- 仅保留匹配 `s00_` 到 `s05_` 的正式实验名称。
- 排除 `bt__` 批量调参、`qp__` 快速探测以及空行。
- 按 `experiment_name` 去重，保留最新有效指标行。

正式实验数量：

| 阶段 | 数量 | 主题 |
| --- | ---: | --- |
| s00 | 2 | 冒烟健全性检查 |
| s01 | 7 | Transformer 与 CTM 规模对比 |
| s02 | 21 | tick 深度、tick 损失、停机行为 |
| s03 | 14 | ELF 及多 token 预测变体 |
| s04 | 16 | cell 数量、cell 宽度与稀疏 top-k cell |
| s05 | 11 | CTM 消融实验与有前景的基础候选 |

稀疏实验数量：

| 阶段 | 数量 | 主题 |
| --- | ---: | --- |
| sp00 | 2 | 稀疏冒烟检查 |
| sp01 | 10 | cell 尺寸/数量密集与 top-k 对比 |
| sp02 | 12 | top-k 激活比例扫描 |
| sp03 | 12 | 稀疏下的 synapse/memory 简化 |
| sp04 | 9 | tick 数与稀疏 cell 交叉 |
| sp05 | 4 | 2000 步稀疏确认运行 |

## 核心发现

1. Transformer 在损失和成本方面仍是最强基线。
   最佳 Transformer 结果为 `s01_transformer_12l_h640`，损失 `4.6791`，吞吐量 `39484 tok/s`，峰值显存 `4.9 GB`。

2. 默认 CTM 在成本上尚不具备竞争力。
   `s01_ctm_12l_h640_tick4` 达到损失 `5.4135`，但吞吐量仅 `3868 tok/s`，显存 `31.6 GB`。

3. 更多 tick 并不会自动产生更好的思考。
   tick 扫描在 tick2 时最优：`s02_ctm_tick2` 损失 `5.3994`。tick8、tick12 和 tick16 更慢且更差。

4. ELF 尚未实现预期的多 token 优势。
   短 ELF 变体接近 tick4 基线，但长 horizon 加高 tick 数会同时恶化损失和吞吐量。

5. 当前的 cell 稀疏并非真正的成本节省稀疏。
   top-k 激活比例有记录，但显存并未相应下降。稀疏掩码尚未移除足够的底层张量运算。

6. 最强的 CTM 候选是消融变体，而非默认配置。
   `s05_synapse2_mh2` 是最佳 CTM 结果：损失 `5.3612`，吞吐量 `2629 tok/s`，峰值显存 `41.6 GB`。

7. 稀疏扫描发现了新的 CTM 短运行前沿。
   `sp05_confirm_d512_dense_sd2_mh2_tick2` 达到损失 `4.9729`，吞吐量 `4982 tok/s`，峰值显存 `24.1 GB`。

8. 当前 top-k 稀疏保持质量比预期更好，但仍然不节省显存。
   例如 `sp05_confirm_d512_topk256_sd2_mh2_tick2` 在激活比例 `0.5` 下达到损失 `5.0482`，但显存为 `24.5 GB`，略高于密集 d512 确认。

9. tick1 稀疏变体是最佳成本前沿，但需要更长确认。
   `sp04_d512_topk256_tick1` 达到损失 `5.2979`，吞吐量 `7455 tok/s`，峰值显存 `16.2 GB`。

## 推荐后续方向

以 `s05_synapse2_mh2` 作为下一轮 CTM 基础，然后围绕以下方向进行更小、更聚焦的扫描：

- tick1/tick2/tick3 配合改进的 tick 监督；
- 真正节省计算的动态停机；
- 短 horizon ELF 配合更强的多 token 损失；
- 真正的稀疏 cell 执行，避免非活跃 cell 的投影、trace 存储和重复全宽状态运算；
- 在等墙钟时间和等显存预算下的直接匹配 Transformer 对照；
- d512 和 d768 tick1/tick2 稀疏变体的 2000/4000 步确认。

## 文件

- `s00_smoke.md`
- `s01_baseline_scale.md`
- `s02_tick_dynamics.md`
- `s03_elf.md`
- `s04_cells_sparsity.md`
- `s05_ablations.md`
- `sp00_sparse_smoke.md`
- `sp01_cell_size_count.md`
- `sp02_topk_ratio.md`
- `sp03_synapse_memory.md`
- `sp04_tick_sparse.md`
- `sp05_best_sparse_confirm.md`
