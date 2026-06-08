# s05 Ablations And Next Base

## Experiment Count

11 experiments:

| Experiment | Loss | Tok/s | Peak Mem MB |
| --- | ---: | ---: | ---: |
| `s05_synapse2_mh2` | 5.3612 | 2629 | 41607 |
| `s05_deep_nlms0` | 5.4360 | 2795 | 37152 |
| `s05_dinput384_heads8` | 5.4436 | 2366 | 51670 |
| `s05_dinput128_heads4` | 5.4487 | 2414 | 50481 |
| `s05_no_selfcond` | 5.4551 | 2449 | 50154 |
| `s05_memlen14` | 5.4553 | 2280 | 52526 |
| `s05_tiny_nlm` | 5.4631 | 2703 | 40604 |
| `s05_shallow_synapse` | 5.5179 | 2619 | 45942 |
| `s05_short_memory` | 5.5233 | 2504 | 49449 |
| `s05_no_cross_state` | 5.5350 | 2390 | 51001 |
| `s05_no_selfcond_no_cross` | 5.5505 | 2414 | 50152 |

## What Was Tested

This stage tested targeted CTM ablations:

- synapse depth;
- memory hidden dims;
- deep NLM on/off;
- memory length;
- input dimension and head count;
- self conditioning;
- cross-layer state.

## Why This Experiment Was Needed

The earlier stages showed that default CTM is expensive and not Transformer-competitive. Ablations identify which CTM details are actually helping, which are unnecessary, and which simpler configuration should become the next base.

## Result

The best CTM result in the full sweep is:

- `s05_synapse2_mh2`: loss `5.3612`, throughput `2629 tok/s`, peak memory `41.6 GB`.

This beats the default CTM scale/tick settings and is more practical than large cell variants.

Other useful results:

- `s05_deep_nlms0`: loss `5.4360`, throughput `2795 tok/s`, memory `37.2 GB`.
- `s05_tiny_nlm`: loss `5.4631`, throughput `2703 tok/s`, memory `40.6 GB`.
- `s05_no_selfcond`: loss `5.4551`, so self conditioning is not clearly essential in this setting.
- `s05_no_cross_state`: loss `5.5350`, worse than `no_selfcond`, suggesting cross-layer state is more important than self conditioning.
- `s05_short_memory`: loss `5.5233`, worse than baseline memory length, so overly short memory hurts.

## Conclusion

The best CTM direction is not bigger cells or more ticks. It is a cleaner, cheaper CTM core around `s05_synapse2_mh2`.

## Linked Comparisons

Use this with:

- `s01_baseline_scale.md`: this is the best CTM candidate to compare against Transformer next;
- `s02_tick_dynamics.md`: pair this base with tick1/tick2/tick3 sweeps;
- `s03_elf.md`: add short-horizon ELF to this base;
- `s04_cells_sparsity.md`: combine this base with true sparse execution, not just top-k masking.

## Advantage Or Disadvantage Diagnosis

Why `s05_synapse2_mh2` may be better:

- It reduces unnecessary synapse complexity.
- It may improve optimization by simplifying the cell dynamics.
- It preserves enough memory structure without over-deep internal nonlinear modules.

Why other ablations lag:

- Removing cross-layer state weakens information flow.
- Short memory reduces temporal context.
- Shallow synapse depth may underfit compared with the more balanced synapse2 setting.

## Follow-Up

Use `s05_synapse2_mh2` as the next CTM base and run a focused second sweep:

1. tick count: 1, 2, 3, 4;
2. ELF horizon: none, 2, 4;
3. dynamic halt with actual compute skipping;
4. true sparse cells with block-level skipping;
5. matched Transformer controls at equal memory and equal wall-clock.

Success criteria for the next round:

- CTM loss below `5.30` in the same training budget;
- peak memory below `45 GB`;
- throughput above `3000 tok/s`;
- evidence that extra ticks improve selected hard examples rather than all tokens uniformly.

