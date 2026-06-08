# sp03 Synapse Memory

## Experiment Count

12 experiments:

| Experiment | Loss | Tok/s | Peak Mem GB | Active Fraction | Quality Cost Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sp03_d512_sd1_mh1_topk256` | 5.4559 | 2907 | 34.7 | 0.5 | 0.0652 |
| `sp03_d512_sd1_mh2_topk256` | 5.4785 | 2713 | 38.1 | 0.5 | 0.0769 |
| `sp03_d512_sd2_mh1_topk256` | 5.4577 | 2796 | 37.3 | 0.5 | 0.0727 |
| `sp03_d512_sd2_mh2_topk256` | 5.4852 | 2616 | 40.6 | 0.5 | 0.0852 |
| `sp03_d768_sd1_mh1_topk384` | 5.5029 | 2061 | 45.9 | 0.5 | 0.1225 |
| `sp03_d768_sd1_mh2_topk384` | 5.4429 | 1924 | 50.9 | 0.5 | 0.1441 |
| `sp03_d768_sd2_mh1_topk384` | 5.5091 | 1984 | 49.5 | 0.5 | 0.1375 |
| `sp03_d768_sd2_mh2_topk384` | 5.5230 | 1865 | 54.6 | 0.5 | 0.1617 |
| `sp03_d1024_sd1_mh1_topk512` | 5.4675 | 1589 | 57.2 | 0.5 | 0.1967 |
| `sp03_d1024_sd1_mh2_topk512` | 5.4577 | 1478 | 63.9 | 0.5 | 0.2360 |
| `sp03_d1024_sd2_mh1_topk512` | 5.4434 | 1548 | 61.9 | 0.5 | 0.2176 |
| `sp03_d1024_sd2_mh2_topk512` | 5.4523 | 1444 | 68.6 | 0.5 | 0.2591 |

## What Was Tested

This stage crossed `synapse_depth` 1/2 with `memory_hidden_dims` 1/2 under 50-percent top-k sparsity for d512, d768, and d1024.

## Why This Experiment Was Needed

The previous best CTM candidate was `s05_synapse2_mh2`. This stage asks whether that same heavier synapse/memory setting remains optimal once cells are top-k sparse, or whether sparse cells prefer a lighter controller.

## Result

The optimal synapse/memory setting changes by cell size:

- d512 is best with `sd1_mh1`: loss `5.4559`, `34.7 GB`, `2907 tok/s`;
- d768 is best with `sd1_mh2`: loss `5.4429`, but uses `50.9 GB`;
- d1024 is best with `sd2_mh1`: loss `5.4434`, `61.9 GB`.

The heavy `sd2_mh2` setting is not the best in this sparse stage. It is consistently slower and more memory-heavy, and its loss advantage does not appear.

## Conclusion

Sparse cells do not automatically benefit from the heavier `synapse2_mh2` recipe. Once cells are gated, a lighter synapse/memory module can be better. For d512 especially, `sd1_mh1` is the cleanest cost-quality point.

## Linked Comparisons

Use this with:

- `s05_ablations.md`: compares against the original `s05_synapse2_mh2` result;
- `sp05_best_sparse_confirm.md`: shows that `sd2_mh2` still performs well at tick2/d512 in longer confirm, but may not be the cheapest best path;
- `sp04_tick_sparse.md`: suggests tick count has a larger effect than synapse depth in this region.

## Advantage Or Disadvantage Diagnosis

Advantage:

- Sparse CTM can use simpler synapse/memory settings without obvious loss collapse.
- d512 `sd1_mh1` improves cost efficiency relative to d512 `sd2_mh2`.

Disadvantage:

- The best loss among this stage is still only around `5.44`, so this cross alone does not explain the stronger sp05 confirm result.
- Memory still scales with d_model and does not fall with active fraction.

## Follow-Up

Add d512 `sd1_mh1` and `sd2_mh1` to the next 2000/4000-step confirm list. The current sp05 confirm only checks `sd2_mh2`, so it may be missing an even cheaper d512 winner.
