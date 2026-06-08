# sp02 Top-K Ratio

## Experiment Count

12 experiments:

| Experiment | Loss | Tok/s | Peak Mem GB | Active Fraction | Quality Cost Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sp02_d512_dense` | 5.5104 | 2752 | 39.9 | 1.0 | 0.0798 |
| `sp02_d512_topk256` | 5.4927 | 2613 | 40.6 | 0.5 | 0.0854 |
| `sp02_d512_topk128` | 5.5029 | 2662 | 40.6 | 0.25 | 0.0840 |
| `sp02_d512_topk64` | 5.5120 | 2654 | 40.6 | 0.125 | 0.0844 |
| `sp02_d768_dense` | 5.5087 | 1939 | 53.4 | 1.0 | 0.1519 |
| `sp02_d768_topk384` | 5.5467 | 1866 | 54.6 | 0.5 | 0.1622 |
| `sp02_d768_topk192` | 5.5342 | 1860 | 54.6 | 0.25 | 0.1624 |
| `sp02_d768_topk96` | 5.5049 | 1886 | 54.6 | 0.125 | 0.1593 |
| `sp02_d1024_dense` | 5.4638 | 1501 | 67.1 | 1.0 | 0.2444 |
| `sp02_d1024_topk512` | 5.4452 | 1448 | 68.6 | 0.5 | 0.2581 |
| `sp02_d1024_topk256` | 5.4483 | 1448 | 68.6 | 0.25 | 0.2582 |
| `sp02_d1024_topk128` | 5.4398 | 1460 | 68.6 | 0.125 | 0.2557 |

## What Was Tested

This stage fixed three cell sizes, d512, d768, and d1024, then compared dense execution against top-k active fractions of `0.5`, `0.25`, and `0.125`.

## Why This Experiment Was Needed

Before implementing true sparse execution, the model needs to show that quality survives aggressive sparsity. If loss collapses at low active fractions, there is little reason to build sparse kernels. If loss survives, the implementation work becomes justified.

## Result

Loss is surprisingly tolerant to aggressive top-k:

- d512 ranges only from `5.4927` to `5.5120` across top-k ratios;
- d768 is best at the most aggressive ratio, `topk96`, with loss `5.5049`;
- d1024 is also best at the most aggressive ratio, `topk128`, with loss `5.4398`.

Memory does not follow active fraction. d1024 dense uses `67.1 GB`, while every top-k d1024 variant uses about `68.6 GB`. d512 and d768 show the same pattern at smaller scale.

## Conclusion

Top-k ratio can be pushed to `0.125` without obvious quality collapse. This is a strong signal that CTM cells contain redundancy and that sparse execution could work. The current implementation, however, proves only logical sparsity, not cost sparsity.

## Linked Comparisons

Use this with:

- `sp01_cell_size_count.md`: confirms that top-k can improve loss at small/mid sizes but not memory;
- `sp03_synapse_memory.md`: checks whether sparse cells prefer simpler synapse/memory settings;
- `sp04_tick_sparse.md`: checks whether fewer ticks dominate top-k ratio effects;
- `s04_cells_sparsity.md`: repeats the main-plan conclusion at a cleaner ratio grid.

## Advantage Or Disadvantage Diagnosis

Advantage:

- d512 and d1024 tolerate active fraction `0.125`.
- Aggressive top-k may provide useful competition/regularization among cells.

Disadvantage:

- Memory is almost constant across active fractions.
- Throughput usually decreases under top-k because routing overhead is added while dense work remains.

## Follow-Up

Implement a true sparse path and rerun this exact ratio grid. The target validation is simple: memory and throughput should scale with active fraction while loss remains near the current top-k results.
