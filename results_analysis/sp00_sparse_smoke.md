# sp00 Sparse Smoke

## Experiment Count

2 experiments:

| Experiment | Loss | Tok/s | Peak Mem GB | Active Fraction |
| --- | ---: | ---: | ---: | ---: |
| `sp00_sparse_d512_dense` | 11.6885 | 2613 | 39.9 | 1.0 |
| `sp00_sparse_d512_topk256` | 11.2335 | 2515 | 40.6 | 0.5 |

## What Was Tested

This smoke stage tested whether the dedicated sparsity plan can run both dense and top-k CTM cells with the same d512 base.

## Why This Experiment Was Needed

The sparsity script is separate from the main experiment plan, so it needed a cheap sanity check before launching the full matrix. The key question was whether `cell_sparsity_mode=topk`, `cell_topk`, active-cell metrics, and pool scheduling all work together.

## Result

Both experiments ran and reported metrics. The top-k smoke reached active fraction `0.5`, confirming that the sparse routing path is active. Loss is not meaningful because these runs stop at 100 steps, but the memory signal is already informative: top-k used `40.6 GB`, slightly more than dense at `39.9 GB`.

## Conclusion

The sparse control path works, but sparse activation is not yet sparse execution. Even in the smallest smoke pair, top-k does not reduce memory.

## Linked Comparisons

Use this with:

- `sp01_cell_size_count.md`: checks whether the same dense-vs-topk pattern persists after 1000 steps;
- `sp02_topk_ratio.md`: checks whether more aggressive active fractions change memory;
- `s04_cells_sparsity.md`: matches the previous main-plan observation that top-k lowers active fraction without lowering cost.

## Advantage Or Disadvantage Diagnosis

Advantage:

- The top-k path is stable enough for large-scale sweeps.
- Active-cell metrics are being recorded correctly.

Disadvantage:

- The inactive cells are likely masked after dense intermediate tensors have already been materialized.
- Routing overhead can offset any small compute savings.

## Follow-Up

Keep sp00 only as a script sanity stage. Do not use it for model-quality decisions. Future smoke runs should add one true-sparse kernel/projection variant once implemented.
