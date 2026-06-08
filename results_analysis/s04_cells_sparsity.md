# s04 Cells And Sparsity

## Experiment Count

16 experiments:

| Experiment | Loss | Tok/s | Peak Mem MB | Active Fraction |
| --- | ---: | ---: | ---: | ---: |
| `s04_cells_d1024_mh2_m8_sd2_dense` | 5.4360 | 1504 | 68728 | 1.0 |
| `s04_cells_d1024_mh2_m8_sd2_topk512` | 5.4420 | 1445 | 70264 | 0.5 |
| `s04_cells_d1024_mh2_m8_sd2_topk256` | 5.4471 | 1451 | 70263 | 0.25 |
| `s04_cells_d768_mh3_m10_sd2_dense` | 5.4681 | 1792 | 61075 | 1.0 |
| `s04_cells_d768_mh3_m10_sd2_topk384` | 5.4841 | 1721 | 62228 | 0.5 |
| `s04_cells_d2048_mh1_m6_sd1_topk512` | 5.5308 | 784 | 72061 | 0.25 |
| `s04_cells_d2048_mh1_m6_sd1_topk256` | 5.5519 | 784 | 72068 | 0.125 |

## What Was Tested

This stage tested whether CTM can benefit from more cells and sparse active cell selection:

- different `d_model`/cell widths;
- different memory-hidden configurations;
- dense cell execution;
- top-k sparse cells;
- top-k rescale on/off;
- active cell fractions from `1.0` down to `0.125`.

## Why This Experiment Was Needed

The biological-brain intuition suggests many cells with sparse activation. If this works, increasing cell count while reducing active computation should preserve or improve quality while lowering cost.

## Result

The strongest cells result is dense:

- `s04_cells_d1024_mh2_m8_sd2_dense`: loss `5.4360`, throughput `1504 tok/s`, peak memory `68.7 GB`.

Top-k sparsity is close in loss but does not save memory:

- topk512 active fraction `0.5`: loss `5.4420`, memory `70.3 GB`;
- topk256 active fraction `0.25`: loss `5.4471`, memory `70.3 GB`.

This is the key result: active fraction decreases, but memory does not decrease. In some cases, sparse variants use more memory than dense variants.

Large cell settings are expensive:

- d2048 top-k variants are around `70-72 GB`;
- throughput falls to around `784 tok/s`;
- loss is not better than the d1024 dense/topk family.

## Conclusion

The current top-k cell sparsity is algorithmically visible but not computationally efficient. It changes active cell fraction, but it does not yet remove enough tensor work or trace storage to reduce cost.

## Linked Comparisons

Use this with:

- `s02_tick_dynamics.md`: tick traces plus cell expansion compound memory cost;
- `s05_ablations.md`: smaller synapse/deep-memory choices beat large cell sweeps;
- `s01_baseline_scale.md`: large cell CTM is still far more expensive than Transformer.

## Advantage Or Disadvantage Diagnosis

Potential advantage:

- Top-k sparsity does not catastrophically hurt loss. The model can tolerate fewer active cells.

Main disadvantage:

- The implementation likely still materializes full dense states, projections, traces, or attention paths.
- Inactive cells are masked too late to save memory.
- The routing/gating path adds overhead.

## Follow-Up

1. Make sparsity real at the tensor level:
   - avoid computing inactive cell projections;
   - avoid storing inactive state traces;
   - avoid full-width intermediate tick outputs;
   - group cells into blocks that can be skipped efficiently.
2. Compare dense vs sparse at fixed active compute, not only fixed total cell count.
3. Add metrics:
   - allocated memory per component;
   - active projection FLOPs;
   - active trace memory;
   - routing overhead.
4. Prioritize d1024 top-k variants over d2048 variants for the next round.

