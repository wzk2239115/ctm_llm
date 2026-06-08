# sp04 Tick Sparse

## Experiment Count

9 experiments:

| Experiment | Loss | Tok/s | Peak Mem GB | Active Fraction | Quality Cost Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sp04_d512_topk256_tick1` | 5.2979 | 7455 | 16.2 | 0.5 | 0.0115 |
| `sp04_d512_topk256_tick2` | 5.4166 | 4623 | 24.5 | 0.5 | 0.0287 |
| `sp04_d512_topk256_tick4` | 5.4902 | 2596 | 40.6 | 0.5 | 0.0859 |
| `sp04_d768_topk384_tick1` | 5.3400 | 5381 | 20.3 | 0.5 | 0.0201 |
| `sp04_d768_topk384_tick2` | 5.4458 | 3318 | 32.0 | 0.5 | 0.0524 |
| `sp04_d768_topk384_tick4` | 5.5361 | 1859 | 54.6 | 0.5 | 0.1625 |
| `sp04_d1024_topk512_tick1` | 5.4326 | 4277 | 24.4 | 0.5 | 0.0310 |
| `sp04_d1024_topk512_tick2` | 5.4227 | 2577 | 39.5 | 0.5 | 0.0831 |
| `sp04_d1024_topk512_tick4` | 5.4504 | 1449 | 68.6 | 0.5 | 0.2580 |

## What Was Tested

This stage crossed sparse cell bases with tick counts 1, 2, and 4:

- d512/topk256;
- d768/topk384;
- d1024/topk512.

## Why This Experiment Was Needed

Tick is supposed to be the CTM advantage. The main sweep showed tick2 was better than tick4/8/12/16, but the sparse setting could change that optimum. This stage asks whether sparse cells make deeper ticking useful, or whether cheaper shallow ticking dominates.

## Result

Tick1 is the cost frontier and often the loss frontier:

- `sp04_d512_topk256_tick1`: loss `5.2979`, `7455 tok/s`, `16.2 GB`;
- `sp04_d768_topk384_tick1`: loss `5.3400`, `5381 tok/s`, `20.3 GB`;
- `sp04_d1024_topk512_tick2`: loss `5.4227`, slightly better than d1024 tick1, but much slower and heavier.

Tick4 is clearly not competitive in this sparse matrix. It uses the most memory and has worse loss for all three d_model values.

## Conclusion

For short 1000-step training, sparse CTM currently benefits more from fewer ticks than from more ticks. This does not yet prove tick1 is the final architecture, because tick2 may need longer training to realize its advantage. But it strongly proves that tick4 should not be the default for this sparse family.

## Linked Comparisons

Use this with:

- `s02_tick_dynamics.md`: both sweeps say more ticks are not automatically better;
- `sp05_best_sparse_confirm.md`: tick2 confirm runs reach very strong loss after 2000 steps;
- `sp01_cell_size_count.md`: d512/d768 become much better when tick count drops from 4 to 1/2.

## Advantage Or Disadvantage Diagnosis

Advantage:

- Tick1 gives very high throughput and low memory while beating the previous CTM best loss.
- Sparse cells plus shallow ticking may be a strong practical CTM regime.

Disadvantage:

- The intended "deep thinking" effect has not emerged from static higher tick counts.
- Tick4 compounds trace/tick-output memory and slows training without improving loss.

## Follow-Up

Run 2000/4000-step confirm for d512 and d768 at tick1 vs tick2. If tick1 stays ahead, make shallow ticking the default and reserve higher ticks for dynamic halt or hard-token routing. If tick2 overtakes, tune the training objective so useful extra ticks appear naturally instead of being forced.
