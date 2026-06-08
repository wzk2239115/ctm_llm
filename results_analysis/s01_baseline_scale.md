# s01 Baseline And Scale Compass

## Experiment Count

7 experiments:

| Experiment | Type | Loss | Tok/s | Peak Mem MB |
| --- | --- | ---: | ---: | ---: |
| `s01_transformer_12l_h640` | transformer | 4.6791 | 39484 | 4936 |
| `s01_transformer_16l_h768` | transformer | 4.7011 | 35117 | 8013 |
| `s01_transformer_24l_h896` | transformer | 4.7344 | 20518 | 14342 |
| `s01_ctm_16l_tick1` | ctm | 5.3947 | 6937 | 18990 |
| `s01_ctm_12l_h640_tick4` | ctm | 5.4135 | 3868 | 31607 |
| `s01_ctm_16l_h768_tick4` | ctm | 5.5001 | 2409 | 50992 |
| `s01_ctm_24l_h896_tick4` | ctm | 5.5706 | 1165 | 62105 |

## What Was Tested

This stage compared standard Transformer baselines against CTM variants across scale:

- Transformer at 12/16/24 layers;
- CTM at similar rough scales;
- one CTM tick1 variant to isolate the cost of repeated ticks.

## Why This Experiment Was Needed

The CTM idea needs a grounded comparison against a conventional Transformer. Without this stage, later tick, ELF, and cell experiments would not answer whether CTM is actually buying useful modeling power relative to a standard baseline.

## Result

Transformer wins clearly on both loss and cost:

- best Transformer loss: `4.6791`;
- best CTM loss in this stage: `5.3947`;
- CTM memory is much higher at matched rough scales;
- CTM throughput is much lower, especially at tick4.

Pairwise observations:

- `s01_ctm_12l_h640_tick4` is about `0.734` loss worse than `s01_transformer_12l_h640`.
- `s01_ctm_16l_tick1` is about `0.694` loss worse than `s01_transformer_16l_h768`.
- `s01_ctm_24l_h896_tick4` is about `0.836` loss worse than `s01_transformer_24l_h896`.

## Conclusion

The current CTM-LLM framework is not yet competitive with Transformer baselines. The CTM mechanism may still contain useful research signals, but the default scaling path is not the right one.

## Linked Comparisons

This stage should be read together with:

- `s02_tick_dynamics.md`: tick count is a major reason CTM cost grows;
- `s05_ablations.md`: the best CTM result comes from a smaller synapse/deep-memory variant, not the default scale-up;
- `s04_cells_sparsity.md`: larger cell configurations increase cost without enough quality improvement.

## Advantage Or Disadvantage Diagnosis

CTM disadvantages:

- repeated tick computation multiplies effective depth;
- CTM keeps additional recurrent traces and tick outputs;
- current CTM does not yet convert extra compute into lower loss;
- larger CTM configurations appear harder to optimize in this training budget.

Potential CTM advantage:

- `s01_ctm_16l_tick1` is much cheaper than tick4 CTM and is one of the stronger CTM baselines. This suggests that CTM should first be optimized at low tick counts before increasing thinking depth.

## Follow-Up

1. Treat Transformer as the mandatory comparison baseline for every future CTM result.
2. Use `s05_synapse2_mh2` or `s02_ctm_tick2` as the next CTM base instead of default tick4.
3. Add matched-budget comparisons:
   - equal peak memory;
   - equal wall-clock;
   - equal tokens processed;
   - equal parameter count.

