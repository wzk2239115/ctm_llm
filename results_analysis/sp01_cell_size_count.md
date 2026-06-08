# sp01 Cell Size And Count

## Experiment Count

10 experiments:

| Experiment | Loss | Tok/s | Peak Mem GB | Active Fraction | Quality Cost Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sp01_d256_dense` | 5.5414 | 4334 | 26.4 | 1.0 | 0.0337 |
| `sp01_d256_topk128` | 5.4441 | 4139 | 26.8 | 0.5 | 0.0352 |
| `sp01_d384_dense` | 5.4757 | 3401 | 33.2 | 1.0 | 0.0534 |
| `sp01_d384_topk192` | 5.4284 | 3269 | 33.7 | 0.5 | 0.0560 |
| `sp01_d512_dense` | 5.5235 | 2728 | 39.9 | 1.0 | 0.0807 |
| `sp01_d512_topk256` | 5.4810 | 2622 | 40.6 | 0.5 | 0.0849 |
| `sp01_d768_dense` | 5.5226 | 1932 | 53.4 | 1.0 | 0.1527 |
| `sp01_d768_topk384` | 5.5290 | 1855 | 54.6 | 0.5 | 0.1629 |
| `sp01_d1024_dense` | 5.4241 | 1501 | 67.1 | 1.0 | 0.2426 |
| `sp01_d1024_topk512` | 5.4370 | 1444 | 68.6 | 0.5 | 0.2584 |

## What Was Tested

This stage swept cell width/count from d256 to d1024, with a dense version and a 50-percent top-k version at each size.

## Why This Experiment Was Needed

The central sparsity hypothesis is that more smaller cells may have a better quality/cost frontier than fewer large dense cells. This stage is the first compass for that hypothesis: it asks where the cell-size sweet spot lives before introducing tick or synapse/memory interactions.

## Result

The best loss is `sp01_d1024_dense` at `5.4241`, but it costs `67.1 GB` and only reaches `1501 tok/s`. The best practical candidates are smaller:

- `sp01_d256_topk128`: loss `5.4441`, `4139 tok/s`, `26.8 GB`;
- `sp01_d384_topk192`: loss `5.4284`, `3269 tok/s`, `33.7 GB`;
- `sp01_d512_topk256`: loss `5.4810`, `2622 tok/s`, `40.6 GB`.

Top-k improves loss for d256, d384, and d512, but not for d768 or d1024. Across every matched pair, top-k uses slightly more memory and slightly lower throughput than dense.

## Conclusion

The experiment supports the smaller-cell direction. d256 and d384 top-k variants approach or pass the previous CTM best-loss threshold while using far less memory than d1024. However, top-k is not reducing cost yet; it acts more like a regularizer or routing prior than a compute-saving mechanism.

## Linked Comparisons

Use this with:

- `sp02_topk_ratio.md`: checks whether the d256/d384/d512 advantage survives more aggressive top-k ratios;
- `sp04_tick_sparse.md`: shows that d512/d768 become much stronger when tick count is reduced;
- `sp05_best_sparse_confirm.md`: confirms d512 as the strongest current region;
- `s05_ablations.md`: compares against `s05_synapse2_mh2`, the previous CTM best.

## Advantage Or Disadvantage Diagnosis

Advantage:

- Smaller cells have much better throughput and memory behavior.
- Top-k can improve loss at small to mid cell sizes, suggesting sparse selection may be acting as useful competition between cells.

Disadvantage:

- Larger cells remain expensive and do not clearly justify their cost.
- Top-k memory is higher than dense because dense tensors are still allocated and routing adds overhead.

## Follow-Up

Use d512 and d768 as the main sparse research region. Keep d1024 only as a quality ceiling. Run longer confirm sweeps for d512/d768 at tick1 and tick2, and implement true sparse execution before spending more compute on top-k ratio grids.
