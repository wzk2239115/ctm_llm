# CTM-LLM 实验结果分析

本文件夹记录了首轮完整的 71 个 CTM-LLM 扫描实验、后续 49 个稀疏扫描实验、32 个 MoE 稀疏扫描实验、82 个区域多轮 MoE 扫描实验、73 个实现验证扫描实验、225 个基线实验的隔夜稀疏 CTM 快速探测扫描，以及 draft-revise 实验扫描。

数据来源：从 `runs/metrics` 导出的 `summary.csv`。
稀疏数据来源：从 `runs/metrics` 导出的 `sparsity_summary.csv`。
MoE 稀疏数据来源：从 `runs/metrics` 导出的 `moe_sparsity_summary.csv`。
区域 MoE 数据来源：从 `runs/metrics` 导出的 `regional_moe_summary.csv`。
实现验证数据来源：从 `runs/metrics` 导出的 `impl_validation_73_summary.csv`。
隔夜稀疏 CTM 快速探测报告：`overnight_sparse_ctm_quick_probe_report.csv`。
隔夜稀疏 CTM 数据来源：从 `runs/metrics` 导出的 `overnight_sparse_ctm_summary.csv`（当前仅有表头，无最终数据行）。
Draft-revise 数据来源：从 `runs/metrics` 导出的 `csv_data/draft_revise_{dr00,dr01,dr03,dr06-dr12}_summary.csv`。

筛选规则：
- 仅保留匹配各扫描前缀的正式实验名称：`s00_` 到 `s05_`、`sp00_` 到 `sp05_`、`moe00_` 到 `moe07_`、`rg00_` 到 `rg11_`，以及 `iv00_` 到 `iv08_`。
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

MoE 稀疏实验数量：

| 阶段 | 数量 | 主题 |
| --- | ---: | --- |
| moe00 | 2 | 密集与路由冒烟检查 |
| moe01 | 5 | 路由器变体 |
| moe02 | 4 | 共享专家加路由专家 |
| moe03 | 3 | 细粒度专家尺寸/数量 |
| moe04 | 5 | 路由器正则化 |
| moe05 | 4 | 分发模式标签 |
| moe06 | 5 | 预热与专家 dropout |
| moe07 | 4 | 稀疏路由与 ELF/MTP 标签交叉 |

区域多轮 MoE 实验数量：

| 阶段 | 数量 | 主题 |
| --- | ---: | --- |
| rg00 | 3 | 密集、单轮路由和区域冒烟检查 |
| rg01 | 10 | 区域激活轮数和路由 top-k |
| rg02 | 8 | 共享/核心区域与纯路由区域 |
| rg03 | 8 | 负载均衡和轮间多样性正则化 |
| rg04 | 10 | CTM tick 数与区域轮数交叉 |
| rg05 | 4 | 2000 步首轮确认运行 |
| rg06 | 8 | 专家粒度和区域大小 |
| rg07 | 7 | d_model/基础尺寸扫描 |
| rg08 | 6 | 路由预热和专家 dropout |
| rg09 | 6 | tick 损失和停机压力 |
| rg10 | 5 | ELF/MTP 与区域路由交叉 |
| rg11 | 7 | 2000 步广泛确认运行 |

实现验证实验数量：

| 阶段 | 数量 | 主题 |
| --- | ---: | --- |
| iv00 | 5 | 密集、单轮 MoE、区域稀疏、停机和 MTP 的冒烟锚点 |
| iv01 | 8 | 后端对照组 |
| iv02 | 10 | 分组稀疏实现后的区域轮数和路由 top-k |
| iv03 | 7 | d_model/轮数/top-k 基础网格 |
| iv04 | 10 | 真正的 tick 早退 |
| iv05 | 10 | MTP/ELF 损失变体 |
| iv06 | 8 | 区域稀疏、停机和 MTP 组合 |
| iv07 | 9 | 更长的实现验证确认 |
| iv08 | 6 | 分发和容量模式检查 |

隔夜稀疏 CTM 快速探测数量：

| 阶段 | 基线实验数 | 可运行数 | 主题 |
| --- | ---: | ---: | --- |
| og00 | 6 | 6 | 密集/top-k/区域锚点 |
| og01 | 12 | 12 | 容量梯度代理 |
| og02 | 24 | 24 | 可变活跃计算 |
| og03 | 30 | 30 | 动态 tick 停机 |
| og04 | 16 | 16 | 路由器正则化和路由控制 |
| og05 | 12 | 12 | 分发/容量/丢弃模式 |
| og06 | 20 | 20 | ELF/MTP 损失 |
| og07 | 10 | 10 | 长确认候选 |
| og08 | 40 | 34 | 长 tick delta/cache 代理 |
| og09 | 34 | 26 | 快/慢、记忆时间尺度和招募代理 |
| og10 | 21 | 20 | 差异化 cell 和快/慢输出损失 |

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

10. MoE 风格路由验证了更低的激活比例，但尚未降低成本。
    `moe04_router_balance1e2_e16_s64_k2` 达到损失 `5.4439`，激活比例 `0.125`，但吞吐量和显存仍接近密集 CTM，因为当前路径是掩码而非真正的稀疏分发。

11. 最强的 MoE 想法是负载均衡、共享专家和稀疏预热。
    `moe02_shared2_routed4_e16_s64` 达到损失 `5.4506`，`moe06_warmup1000_drop0p05_e16_s64_k2` 达到损失 `5.4509`。

12. 区域多轮路由是目前最强的 CTM 方向。
    `rg11_confirm_d1024_p4_shared1_top1` 达到损失 `4.8437`，激活比例 `0.125`。`rg11_confirm_d512_p4_shared1_top1` 达到损失 `4.9395`，吞吐量 `4418 tok/s`，显存 `26.0 GB`，是最佳质量/成本区域候选。

13. 实现验证扫描发现了分组稀疏后端问题。
    `dense_mask` 和密集/top-k 对照在损失 `5.7` 到 `5.8` 范围内保持健康，但 `block_sparse`、`dropless` 和 `capacity_drop` 经常产生巨大损失或 `NaN`。在证明 block-sparse 对等性之前，应将早期区域结果视为掩码路由的建模证据。

14. 隔夜稀疏 CTM 快速探测是可行性筛选，而非损失结果。
    225 个基线实验中有 210 个有可运行的探测。快速/reflex、低活跃计算、MTP/ELF 和差异化 cell 探测适配良好；长 tick、长记忆、招募和快/慢输出系列需要更小的批量并单独分析。

15. 正式隔夜稀疏 CTM 摘要是部分幸存者导出，而非完整的 225 次运行矩阵。
    `csv_data/overnight_sparse_ctm_summary.csv` 包含 33 个已完成运行（26 个有限损失）；192 个计划运行在 2-GPU 通道上 OOM。详见 `overnight_sparse_ctm_summary.md` 了解仅通过实验的分析。

16. Draft-revise 扫描发现了一个有效机制：基于腐化的 draft-revise 训练达到损失 4.616（比区域锚点 5.401 提升 14.5%），但吞吐量成本为 3.4 倍。所有效率机制（残差计算、块跳过、递归 NLM、tick 控制器）均失败。详见 `draft_revise.md` 的完整分析。

## 推荐后续方向

以 `s05_synapse2_mh2` 作为下一轮 CTM 基础，然后围绕以下方向进行更小、更聚焦的扫描：

- tick1/tick2/tick3 配合改进的 tick 监督；
- 真正节省计算的动态停机；
- 短 horizon ELF 配合更强的多 token 损失；
- 真正的稀疏 cell 执行，避免非活跃 cell 的投影、trace 存储和重复全宽状态运算；
- MoE 风格的分组稀疏执行，配合负载均衡、共享专家和预热；
- 区域 p4/shared1/top1 作为主要 CTM 分支，尤其是 d512 和 d1024 确认；
- 在使用分组稀疏区域行做架构决策前，先验证 block-sparse 与 dense-mask 路由的对等性；
- 从快速探测优胜者中选择隔夜稀疏 CTM 子集，将长 tick 和快/慢输出系列隔离到低批量计划中；
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
- `moe_sparsity.md`
- `regional_moe.md`
- `impl_validation_73.md`
- `overnight_sparse_ctm_quick_probe.md`
- `overnight_sparse_ctm_summary.md`
- `draft_revise.md`
