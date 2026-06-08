# s03 ELF 与多 Token 预测

## 实验数量

14 个实验：

| 实验 | 损失 | Tok/s | 峰值显存 MB |
| --- | ---: | ---: | ---: |
| `s03_elf_next` | 5.5004 | 2375 | 50992 |
| `s03_elf_linear_h4_improve` | 5.5059 | 2409 | 50991 |
| `s03_elf_linear_h4` | 5.5105 | 2388 | 50991 |
| `s03_elf_pow2_h4` | 5.5125 | 2408 | 50990 |
| `s03_elf_linear_h8_improve_w0p3` | 5.5422 | 1102 | 63191 |
| `s03_elf_linear_h8` | 5.5706 | 1121 | 63191 |
| `s03_elf_linear_t12_h8` | 5.7823 | 574 | 47514 |
| `s03_elf_linear_t16_h8` | 5.7732 | 423 | 61800 |

## 测试内容

本阶段测试 ELF 风格的多 token 预测：

- ELF 阶段下的 next-token 基线；
- 线性 horizon 模式；
- pow2 horizon 模式；
- horizon 4 和 horizon 8；
- 改进权重；
- 与 tick8、tick12 和 tick16 的组合。

## 为什么需要此实验

一个关注点是 CTM-LLM 未利用 ELF 预测多个 token 的能力。本阶段测试多 token 预测是否能让 tick 更有用或提供更好的学习信号。

## 结果

短 horizon ELF 接近 tick4 CTM 基线但未产生大幅增益：

- `s03_elf_next`：损失 `5.5004`；
- `s03_elf_linear_h4_improve`：损失 `5.5059`；
- `s03_elf_linear_h4`：损失 `5.5105`；
- `s03_elf_pow2_h4`：损失 `5.5125`。

Horizon 8 和高 tick 设置退化：

- `s03_elf_linear_h8`：损失 `5.5706`；
- `s03_elf_linear_t12_h8`：损失 `5.7823`；
- `s03_elf_linear_t16_h8`：损失 `5.7732`。

最强的改进权重变体为 `h8_improve_w0p3`，损失 `5.5422`，但相对于更简单的 CTM 设置仍不够强。

## 结论

ELF 尚未解锁多 token 优势。短 horizon 变体可接受但未明显更好。长 horizon 配合高 tick 数放大了 tick 扫描中已观察到的成本和优化问题。

## 关联比较

结合以下内容使用：

- `s02_tick_dynamics.md`：在加入 ELF 之前高 tick 数已经较弱；
- `s05_ablations.md`：更简单的 CTM 消融击败了大多数 ELF 变体；
- `s01_baseline_scale.md` 中的 Transformer 基线：ELF 仍未能弥合基线差距。

## 优势或劣势诊断

ELF 结果较弱的潜在原因：

- 多 token 损失可能与有用的隐藏状态改善耦合过弱。
- Horizon 目标在训练早期可能过难。
- Tick 和 ELF 目标可能相互竞争而非协作。
- 当前的 lm_head 训练路径可能增加成本但未产生稳健的中间表示。

潜在优势：

- Horizon 4 是稳定的，可作为聚焦重新设计的合理候选。

## 后续

1. 先保持 ELF horizon 较短：horizon 2 或 4。
2. 在最佳 CTM 基础上训练 ELF 作为辅助目标，而非在高 tick 默认设置上。
3. 增加明确报告：
   - next-token 损失；
   - horizon-2 损失；
   - horizon-4 损失；
   - 未来 token 预测是否改善了下游 next-token 损失。
4. 使用 tick2 和 `s05_synapse2_mh2` 测试 ELF。
5. 在 tick 动态更健康之前，避免将长 horizon 与 tick12/tick16 组合。
