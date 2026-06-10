# 实现验证 73

数据来源：`impl_validation_73_summary.csv`。

实验脚本：`scripts/experiment_plan_impl_validation.py`。

## 实验数量

73 个实验：

| 阶段 | 数量 | 有效损失 | 主题 |
| --- | ---: | ---: | --- |
| iv00 | 5 | 5 | 密集、单轮 MoE、区域稀疏、停机和 MTP 的冒烟锚点 |
| iv01 | 8 | 6 | 后端对照组：d512-d1536 的密集/top-k/单轮/区域 |
| iv02 | 10 | 8 | 分组稀疏实现后的区域激活轮数和路由 top-k |
| iv03 | 7 | 5 | d_model、轮数、共享路径和路由 top-k 基础网格 |
| iv04 | 10 | 10 | 真正的 tick 早退和计算惩罚 |
| iv05 | 10 | 10 | 区域稀疏后端上的 MTP/ELF 损失变体 |
| iv06 | 8 | 8 | 区域稀疏、停机和 MTP 的组合 |
| iv07 | 9 | 7 | 实现验证候选的更长确认运行 |
| iv08 | 6 | 4 | 分发/容量模式检查 |

总体而言，73 个实验中有 63 个产生了有限损失。10 个实验产生了 `NaN` 损失。

## 测试内容

本次实验是实现验证扫描，而非正常的架构扫描。目的是检查早期掩码路由实验中有前景的区域 CTM 想法在连接更新的实现路径后是否仍然有效：

- 通过 `moe_dispatch_mode=block_sparse` 的顺序分组稀疏区域后端；
- 通过 `tick_halt_mode=threshold/confidence` 的真正 tick 早退；
- 通过 `moe_mtp_mode` 的 MTP 多 horizon 损失；
- 分发/容量标志包括 `dense_mask`、`block_sparse`、`dropless` 和 `capacity_drop`；
- 对短验证阶段中看起来合理的候选进行更长确认运行。

计划生成器为 `scripts/experiment_plan_impl_validation.py`。摘要由其 `summarize` 子命令导出，从 `runs/metrics` 收集最终的 `iv00_` 到 `iv08_` 指标行。

## 结果

稳定结果主要是 dense-mask 或非区域对照：

| 实验 | 损失 | Tok/s | 峰值显存 GB | 质量成本分数 | 解读 |
| --- | ---: | ---: | ---: | ---: | --- |
| `iv01_backend_singlepass_d1024_e16_top2` | 5.7076 | 2098 | 15.3 | 0.0416 | 正常速度对照中最佳有限损失 |
| `iv08_dispatch_regional_densemask_label_d512_p4` | 5.7122 | 2092 | 10.2 | 0.0279 | dense-mask 区域标签路径健康 |
| `iv01_backend_topk_d512_k256` | 5.7832 | 3028 | 9.7 | 0.0186 | 最佳质量/成本分数 |
| `iv01_backend_dense_d1024` | 5.8119 | 2247 | 14.8 | 0.0382 | 密集 d1024 对照健康 |
| `iv01_backend_dense_d512` | 5.8146 | 2891 | 9.6 | 0.0192 | 密集 d512 对照健康 |

新的分组稀疏区域路径尚未健康。一些 block-sparse 区域行产生有限损失，但它们要么极慢要么数值不良：

- `iv03_base_d1024_p4_shared1_top2`：损失 `5.7189`，但仅 `99 tok/s`；
- `iv02_pass_p4_shared1_top2_d1024`：损失 `5.7604`，但仅 `87 tok/s`；
- `iv03_base_d512_p4_shared1_top1`：损失 `96.1680`；
- `iv03_base_d1024_p4_shared1_top1`：损失 `295.2789`；
- `iv03_base_d768_p4_shared1_top1` 和 `iv03_base_d1536_p3_shared1_top1`：`NaN`。

真正停机尚未节省计算。所有 `iv04` 停机变体报告 `effective_tick=1`，但吞吐量比无停机 tick2 对照低得多，损失不在有用范围内：

- 无停机 tick2 对照：损失 `54.8524`，`348 tok/s`；
- 最佳停机损失：`iv04_halt_threshold0p45_tick4_d512_p4`，损失 `42.0721`，仅 `64 tok/s`；
- 阈值 `0.00` 特别不稳定，损失 `226.5651`。

MTP/ELF 在此实现路径上也没有帮助：

- 最佳 `iv05` 行是 `iv05_mtp_elf_pow2_h4_d512_p4`，损失 `36.5999`；
- 普通 MTP 区域对照损失 `1756.1048`；
- 长 MTP horizon `1,2,4,8` 损失 `1154.3016`。

更长的 `iv07` 确认运行未能拯救分组稀疏路径。最佳是 `iv07_confirm_d512_p4_plain`，损失 `7.4439`，但仍远差于密集/top-k 对照和早期掩码区域结果。d768 确认行为 `NaN`，停机/MTP 确认严重退化。

`iv08` 中的分发检查最清楚地隔离了问题：

- `dense_mask`：损失 `5.7122`，`2092 tok/s`，健康；
- `block_sparse`：损失 `145.5542`，`345 tok/s`，不健康；
- `dropless`：损失 `785.1353`，不健康；
- `capacity_drop`：容量 `0.75` 和 `1.00` 时为 `NaN`，容量 `1.25` 时损失 `15332.9920`。

## 含义

本次扫描尚未验证新的分组稀疏后端作为训练路径。它验证了相反的情况：建模思想在 dense-mask/对照路径中仍然合理，但新实现路径存在正确性或训练动力学问题。

重要的区别是：

- 密集/top-k/dense-mask 对照在预期的损失范围 `5.7` 到 `5.8` 内训练；
- `block_sparse`、`dropless` 和 `capacity_drop` 频繁产生巨大损失、`NaN` 或严重的吞吐量崩溃；
- 停机和 MTP 被损坏的区域稀疏路径所混淆，因此其负面结果不应被解读为最终的架构结论。

因此，早期区域结果应被视为掩码路由的建模信号，而非真正稀疏执行的证明。本次实验很有价值，因为它找到了确切的下一个工程瓶颈：分组稀疏分发必须先在数值上与 dense-mask 路由等效，然后才能用于架构结论。

## 推荐后续实验

1. 在更多训练扫描之前添加 block-sparse 对等测试。
   在固定 mini-batch 上比较 `dense_mask` 和 `block_sparse` 的相同路由决策。检查 logits、逐 tick 损失、活跃区域索引、gather/scatter 顺序、残差合并和辅助损失。

2. 将分组稀疏后端简化为最小可复现用例。
   使用 d512，16 个专家，专家大小 32，shared1/top1，p1 或 p2，tick1/tick2，无停机，无 MTP，无 capacity drop。运行 20 步和 100 步冒烟检查，直到损失和 logits 匹配 dense-mask 路径。

3. 在证明 block-sparse 对等之前禁用 capacity/drop-token 实验。
   当基础分发路径已经失败时，`capacity_drop` 行太不稳定无法诊断。

4. 在区域稀疏后端稳定之前，将停机和 MTP 排除在主线之外。
   仅在 block-sparse 在短训练中匹配 dense-mask 后，才作为隔离对照重新引入。

5. 对等通过后，重跑紧凑验证矩阵：
   - `dense_mask` vs `block_sparse`；
   - p1、p2、p4；
   - shared1/top1 和 shared1/top2；
   - d512 和 d1024；
   - 100 步冒烟，然后 1000 步确认。

6. 在损失之外添加实现指标：
   活跃 FLOPs、gather/scatter 开销、活跃区域 trace 显存、非活跃区域跳过比率和每分发计时。当前活跃比例不足以证明真正的成本节省。

## 总结

`impl_validation_73` 发现的是后端实现问题，而非新的最佳架构。安全的下一步是修复并证明 `block_sparse` 与 `dense_mask` 的等价性，然后重跑区域 p4/shared1/top1 实验。在此之前，架构决策应继续使用稳定的密集/top-k/dense-mask 对照，而非分组稀疏区域行。
