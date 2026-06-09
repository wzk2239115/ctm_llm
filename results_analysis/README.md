# CTM-LLM Experiment Results Analysis

This folder records the first full 71-experiment CTM-LLM sweep, the follow-up
49-experiment sparsity sweep, the 32-experiment MoE sparsity sweep, the
82-experiment regional multi-pass MoE sweep, the 73-experiment
implementation-validation sweep, and the 225-base-experiment overnight sparse
CTM quick-probe sweep.

Source summary: `summary.csv` exported from `runs/metrics`.
Sparsity source summary: `sparsity_summary.csv` exported from `runs/metrics`.
MoE sparsity source summary: `moe_sparsity_summary.csv` exported from `runs/metrics`.
Regional MoE source summary: `regional_moe_summary.csv` exported from `runs/metrics`.
Implementation-validation source summary: `impl_validation_73_summary.csv` exported from `runs/metrics`.
Overnight sparse CTM quick-probe report: `overnight_sparse_ctm_quick_probe_report.csv`.
Overnight sparse CTM source summary: `overnight_sparse_ctm_summary.csv` exported from `runs/metrics` (currently header-only, no final rows).

Filtering rule:
- Keep only formal experiment names matching each sweep prefix: `s00_` to `s05_`, `sp00_` to `sp05_`, `moe00_` to `moe07_`, `rg00_` to `rg11_`, and `iv00_` to `iv08_`.
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

Regional multi-pass MoE experiment count:

| Stage | Count | Topic |
| --- | ---: | --- |
| rg00 | 3 | dense, single-pass routed, and regional smoke checks |
| rg01 | 10 | regional activation pass count and routed top-k |
| rg02 | 8 | shared/core regions versus routed-only regions |
| rg03 | 8 | load-balance and inter-pass diversity regularization |
| rg04 | 10 | CTM tick count crossed with regional pass count |
| rg05 | 4 | 2000-step first confirmation runs |
| rg06 | 8 | expert granularity and region size |
| rg07 | 7 | d_model/base-size sweep |
| rg08 | 6 | routing warmup and expert dropout |
| rg09 | 6 | tick loss and halt pressure |
| rg10 | 5 | ELF/MTP crossed with regional routing |
| rg11 | 7 | 2000-step broad confirmation runs |

Implementation-validation experiment count:

| Stage | Count | Topic |
| --- | ---: | --- |
| iv00 | 5 | smoke anchors for dense, single-pass MoE, regional sparse, halt, and MTP |
| iv01 | 8 | backend controls |
| iv02 | 10 | regional pass count and routed top-k after grouped sparse implementation |
| iv03 | 7 | d_model/pass/top-k base grid |
| iv04 | 10 | real tick early-exit |
| iv05 | 10 | MTP/ELF loss variants |
| iv06 | 8 | regional sparse, halt, and MTP composition |
| iv07 | 9 | longer implementation-validation confirmations |
| iv08 | 6 | dispatch and capacity mode checks |

Overnight sparse CTM quick-probe count:

| Stage | Base Experiments | Runnable | Topic |
| --- | ---: | ---: | --- |
| og00 | 6 | 6 | dense/top-k/regional anchors |
| og01 | 12 | 12 | capacity-gradient proxy |
| og02 | 24 | 24 | variable active compute |
| og03 | 30 | 30 | dynamic tick halt |
| og04 | 16 | 16 | router regularization and routing controls |
| og05 | 12 | 12 | dispatch/capacity/drop modes |
| og06 | 20 | 20 | ELF/MTP losses |
| og07 | 10 | 10 | long confirmation candidates |
| og08 | 40 | 34 | long-tick delta/cache proxies |
| og09 | 34 | 26 | fast/slow, memory-timescale, and recruitment proxies |
| og10 | 21 | 20 | differentiated cells and fast/slow output losses |

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

12. Regional multi-pass routing is the strongest CTM direction so far.
   `rg11_confirm_d1024_p4_shared1_top1` reaches loss `4.8437` with active fraction `0.125`. `rg11_confirm_d512_p4_shared1_top1` reaches loss `4.9395` at `4418 tok/s` and `26.0 GB`, making it the best quality/cost regional candidate.

13. The implementation-validation sweep found a grouped sparse backend problem.
   `dense_mask` and dense/top-k controls remain healthy around loss `5.7` to `5.8`, but `block_sparse`, `dropless`, and `capacity_drop` often produce huge loss or `NaN`. Treat earlier regional results as masked-routing modeling evidence until block-sparse parity is proven.

14. The overnight sparse CTM quick probe is a feasibility screen, not a loss result.
   210 of 225 base experiments had a runnable probe. Fast/reflex, low-active compute, MTP/ELF, and differentiated-cell probes fit well; long-tick, long-memory, recruitment, and fast/slow output families need smaller batches and separate profiling.

15. The formal overnight sparse CTM summary is a partial survivor export, not the full 225-run matrix.
   `csv_data/overnight_sparse_ctm_summary.csv` contains 33 completed runs (26 finite loss); 192 planned runs OOM'd on 2-GPU lanes. See `overnight_sparse_ctm_summary.md` for analysis of the passing experiments only.

## Recommended Next Direction

Use `s05_synapse2_mh2` as the next CTM base. Then run smaller, more targeted sweeps around:

- tick1/tick2/tick3 with improved tick supervision;
- dynamic halt that actually saves compute;
- ELF short horizon with a stronger multi-token loss;
- true sparse cell execution that avoids inactive cell projections, trace storage, and repeated full-width state work;
- MoE-style grouped sparse execution with load balance, shared experts, and warmup;
- regional p4/shared1/top1 as the main CTM branch, especially d512 and d1024 confirmations;
- block-sparse parity against dense-mask routing before using grouped sparse regional rows for architecture decisions;
- overnight sparse CTM subsets chosen from the quick-probe winners, with long-tick and fast/slow-output families isolated into low-batch plans;
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
- `regional_moe.md`
- `impl_validation_73.md`
- `overnight_sparse_ctm_quick_probe.md`
- `overnight_sparse_ctm_summary.md`
