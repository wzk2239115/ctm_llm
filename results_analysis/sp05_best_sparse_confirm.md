# sp05 Best Sparse Confirm

## Experiment Count

4 experiments:

| Experiment | Loss | Tok/s | Peak Mem GB | Active Fraction | Quality Cost Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sp05_confirm_d512_dense_sd2_mh2_tick2` | 4.9729 | 4982 | 24.1 | 1.0 | 0.0241 |
| `sp05_confirm_d512_topk256_sd2_mh2_tick2` | 5.0482 | 4789 | 24.5 | 0.5 | 0.0258 |
| `sp05_confirm_d768_topk384_sd2_mh2_tick2` | 5.0999 | 3429 | 32.0 | 0.5 | 0.0475 |
| `sp05_confirm_d1024_topk512_sd2_mh2_tick2` | 5.0306 | 2711 | 39.5 | 0.5 | 0.0733 |

## What Was Tested

This stage reran the strongest sparse candidates for 2000 steps using tick2 and `sd2_mh2`, plus a d512 dense control.

## Why This Experiment Was Needed

The earlier sparse stages were mostly 1000-step compass runs. The confirm stage checks whether promising candidates survive a longer training window and whether sparse quality remains close to dense quality.

## Result

The best result is dense d512:

- `sp05_confirm_d512_dense_sd2_mh2_tick2`: loss `4.9729`, `4982 tok/s`, `24.1 GB`.

Sparse d512 is close:

- `sp05_confirm_d512_topk256_sd2_mh2_tick2`: loss `5.0482`, `4789 tok/s`, `24.5 GB`.

d1024 top-k has excellent loss, but weaker cost:

- `sp05_confirm_d1024_topk512_sd2_mh2_tick2`: loss `5.0306`, `2711 tok/s`, `39.5 GB`.

d768 top-k is a middle option:

- loss `5.0999`, `3429 tok/s`, `32.0 GB`.

All four confirm runs beat the previous CTM best `s05_synapse2_mh2` loss `5.3612`.

## Conclusion

The sparse sweep found a much better CTM regime than the original full-plan best. The strongest current architecture is not bigger cells, more ticks, or longer memory; it is d512 with tick2 and the synapse/memory recipe, with dense and top-k both viable.

The key sparse-specific conclusion is subtle: top-k is quality-viable but not cost-saving yet. d512 top-k loses only `0.0753` loss versus d512 dense, but it uses slightly more memory. That is exactly the evidence needed to justify true sparse execution work.

## Linked Comparisons

Use this with:

- `sp04_tick_sparse.md`: tick1 is cheaper and very strong at 1000 steps, but tick2 confirm has better final loss;
- `sp03_synapse_memory.md`: `sd2_mh2` confirm is strong, but lighter d512 settings still need longer confirmation;
- `s05_ablations.md`: the new d512 confirm beats the previous best CTM by a large margin;
- `s01_baseline_scale.md`: Transformer still needs matched comparison at this improved CTM regime.

## Advantage Or Disadvantage Diagnosis

Advantage:

- d512 dense/tick2 is a new CTM quality frontier.
- d512 top-k/tick2 nearly matches dense while activating only half the cells.
- d1024 top-k is a quality ceiling if memory budget allows it.

Disadvantage:

- top-k does not reduce memory, so its current benefit is not infrastructure cost reduction.
- d1024 remains slower and less cost efficient than d512.
- tick2 confirm does not answer whether tick1 would remain stronger after 2000/4000 steps.

## Follow-Up

1. Run d512 tick1/tick2 and d768 tick1/tick2 confirms at 2000 and 4000 steps.
2. Add d512 `sd1_mh1` and `sd2_mh1` confirms from `sp03`.
3. Implement true sparse execution:
   - skip inactive cell projections;
   - avoid inactive state trace storage;
   - avoid auxiliary lm_head over inactive tick/cell outputs;
   - record real sparse FLOPs and memory by component.
4. Add matched Transformer controls at the same memory and wall-clock budget as the d512 confirm family.
