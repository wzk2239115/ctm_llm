# CTM-LLM Experiment Results Analysis

This folder records the first full 71-experiment CTM-LLM sweep, the follow-up
49-experiment sparsity sweep, and the 32-experiment MoE sparsity sweep.

Source summary: `summary.csv` exported from `runs/metrics`.
Sparsity source summary: `sparsity_summary.csv` exported from `runs/metrics`.
MoE sparsity source summary: `moe_sparsity_summary.csv` exported from `runs/metrics`.

Filtering rule:
- Keep only formal experiment names matching `s00_` to `s05_`.
- Exclude `bt__` batch-tune probes, `qp__` quick-probe probes, and empty rows.
- Deduplicate by `experiment_name` using the latest valid metric row.

Formal experiment count:

| Stage | Count | Topic |
| --- | ---: | --- |
| s00 | 2 | smoke sanity checks |
| s01 | 7 | Transformer vs CTM scale compass |
| s02 | 21 | tick depth, tick loss, halt behavior |
| s03 | 14 | ELF and multi-token prediction variants |
| s04 | 16 | cell count, cell width, and sparse top-k cells |
| s05 | 11 | CTM ablations and promising base candidates |

Sparsity experiment count:

| Stage | Count | Topic |
| --- | ---: | --- |
| sp00 | 2 | sparse smoke checks |
| sp01 | 10 | cell size/count dense vs top-k compass |
| sp02 | 12 | top-k active ratio sweep |
| sp03 | 12 | synapse/memory simplification under sparsity |
| sp04 | 9 | tick count crossed with sparse cells |
| sp05 | 4 | 2000-step sparse confirmation runs |

MoE sparsity experiment count:

| Stage | Count | Topic |
| --- | ---: | --- |
| moe00 | 2 | dense and routed smoke checks |
| moe01 | 5 | router variants |
| moe02 | 4 | shared experts plus routed experts |
| moe03 | 3 | fine-grained expert sizes/counts |
| moe04 | 5 | router regularization |
| moe05 | 4 | dispatch mode labels |
| moe06 | 5 | warmup and expert dropout |
| moe07 | 4 | sparse routing crossed with ELF/MTP labels |

## High-Level Findings

1. Transformer is still the strongest baseline on loss and cost.
   The best Transformer result is `s01_transformer_12l_h640` with loss `4.6791`, throughput `39484 tok/s`, and peak memory `4.9 GB`.

2. Default CTM is not yet cost competitive.
   `s01_ctm_12l_h640_tick4` reaches loss `5.4135`, but throughput is only `3868 tok/s` and memory is `31.6 GB`.

3. More ticks do not automatically create better thinking.
   Tick sweep is best at tick2: `s02_ctm_tick2` has loss `5.3994`. Tick8, tick12, and tick16 are slower and worse.

4. ELF is not yet delivering the intended multi-token advantage.
   Short ELF variants are near the tick4 baseline, but long horizon plus high tick count degrades both loss and throughput.

5. Current cell sparsity is not true cost-saving sparsity.
   Top-k active fractions are recorded, but memory does not drop accordingly. The sparse mask is not yet removing enough underlying tensor work.

6. The strongest CTM candidate is an ablation, not the default.
   `s05_synapse2_mh2` is the best CTM result: loss `5.3612`, throughput `2629 tok/s`, peak memory `41.6 GB`.

7. The sparsity sweep finds a new CTM short-run frontier.
   `sp05_confirm_d512_dense_sd2_mh2_tick2` reaches loss `4.9729`, throughput `4982 tok/s`, and peak memory `24.1 GB`.

8. Current top-k sparsity preserves quality better than expected, but still does not save memory.
   For example, `sp05_confirm_d512_topk256_sd2_mh2_tick2` reaches loss `5.0482` at active fraction `0.5`, but memory is `24.5 GB`, slightly above the dense d512 confirm.

9. Tick1 sparse variants are the best cost frontier, but need longer confirmation.
   `sp04_d512_topk256_tick1` reaches loss `5.2979`, throughput `7455 tok/s`, and peak memory `16.2 GB`.

10. MoE-style routing validates lower active fraction, but not lower cost yet.
   `moe04_router_balance1e2_e16_s64_k2` reaches loss `5.4439` with active fraction `0.125`, but throughput and memory remain close to dense CTM because the current path is masked rather than true sparse dispatch.

11. The strongest MoE ideas are load balance, shared experts, and sparse warmup.
   `moe02_shared2_routed4_e16_s64` reaches loss `5.4506`, and `moe06_warmup1000_drop0p05_e16_s64_k2` reaches loss `5.4509`.

## Recommended Next Direction

Use `s05_synapse2_mh2` as the next CTM base. Then run smaller, more targeted sweeps around:

- tick1/tick2/tick3 with improved tick supervision;
- dynamic halt that actually saves compute;
- ELF short horizon with a stronger multi-token loss;
- true sparse cell execution that avoids inactive cell projections, trace storage, and repeated full-width state work;
- MoE-style grouped sparse execution with load balance, shared experts, and warmup;
- direct matched Transformer controls at equal wall-clock budget and equal memory budget.
- 2000/4000-step confirmation of d512 and d768 tick1/tick2 sparse variants.

## Files

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
