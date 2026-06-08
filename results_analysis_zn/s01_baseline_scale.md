# s01 基线与规模对比

## 实验数量

7 个实验：

| 实验 | 类型 | 损失 | Tok/s | 峰值显存 MB |
| --- | --- | ---: | ---: | ---: |
| `s01_transformer_12l_h640` | transformer | 4.6791 | 39484 | 4936 |
| `s01_transformer_16l_h768` | transformer | 4.7011 | 35117 | 8013 |
| `s01_transformer_24l_h896` | transformer | 4.7344 | 20518 | 14342 |
| `s01_ctm_16l_tick1` | ctm | 5.3947 | 6937 | 18990 |
| `s01_ctm_12l_h640_tick4` | ctm | 5.4135 | 3868 | 31607 |
| `s01_ctm_16l_h768_tick4` | ctm | 5.5001 | 2409 | 50992 |
| `s01_ctm_24l_h896_tick4` | ctm | 5.5706 | 1165 | 62105 |

## 测试内容

本阶段在不同规模下对比标准 Transformer 基线与 CTM 变体：

- Transformer 在 12/16/24 层；
- CTM 在近似同等规模；
- 一个 CTM tick1 变体用于隔离重复 tick 的成本。

## 为什么需要此实验

CTM 的想法需要与常规 Transformer 进行扎实的对比。没有此阶段，后续的 tick、ELF 和 cell 实验将无法回答 CTM 是否真正带来了相对于标准基线的有效建模能力。

## 结果

Transformer 在损失和成本上均明显胜出：

- 最佳 Transformer 损失：`4.6791`；
- 本阶段最佳 CTM 损失：`5.3947`；
- CTM 在近似同等规模下显存远高于 Transformer；
- CTM 吞吐量远低于 Transformer，尤其是 tick4 时。

逐对比较：

- `s01_ctm_12l_h640_tick4` 比 `s01_transformer_12l_h640` 损失高约 `0.734`。
- `s01_ctm_16l_tick1` 比 `s01_transformer_16l_h768` 损失高约 `0.694`。
- `s01_ctm_24l_h896_tick4` 比 `s01_transformer_24l_h896` 损失高约 `0.836`。

## 结论

当前 CTM-LLM 框架尚未能与 Transformer 基线竞争。CTM 机制可能仍包含有价值的研究信号，但默认的扩展路径不是正确的方向。

## 关联比较

本阶段应结合以下内容阅读：

- `s02_tick_dynamics.md`：tick 数量是 CTM 成本增长的主要原因；
- `s05_ablations.md`：最佳 CTM 结果来自更小的 synapse/deep-memory 变体，而非默认放大；
- `s04_cells_sparsity.md`：更大的 cell 配置增加了成本但质量提升不足。

## 优势或劣势诊断

CTM 劣势：

- 重复 tick 计算倍增了有效深度；
- CTM 保留额外的循环 trace 和 tick 输出；
- 当前 CTM 尚未将额外计算转化为更低的损失；
- 更大的 CTM 配置在此训练预算下似乎更难优化。

CTM 潜在优势：

- `s01_ctm_16l_tick1` 比 tick4 CTM 便宜得多，且是较强的 CTM 基线之一。这表明应先在低 tick 数下优化 CTM，再增加思考深度。

## 后续

1. 将 Transformer 作为每个未来 CTM 结果的必比基线。
2. 使用 `s05_synapse2_mh2` 或 `s02_ctm_tick2` 作为下一个 CTM 基础，替代默认 tick4。
3. 增加匹配预算对比：
   - 等峰值显存；
   - 等墙钟时间；
   - 等处理 token 数；
   - 等参数量。
