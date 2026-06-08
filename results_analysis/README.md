# CTM-LLM Experiment Results Analysis

This folder records the first full 71-experiment CTM-LLM sweep.

Source summary: `summary.csv` exported from `runs/metrics`.

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

## Recommended Next Direction

Use `s05_synapse2_mh2` as the next CTM base. Then run smaller, more targeted sweeps around:

- tick1/tick2/tick3 with improved tick supervision;
- dynamic halt that actually saves compute;
- ELF short horizon with a stronger multi-token loss;
- true sparse cell execution that avoids inactive cell projections, trace storage, and repeated full-width state work;
- direct matched Transformer controls at equal wall-clock budget and equal memory budget.

## Files

- `s00_smoke.md`
- `s01_baseline_scale.md`
- `s02_tick_dynamics.md`
- `s03_elf.md`
- `s04_cells_sparsity.md`
- `s05_ablations.md`

