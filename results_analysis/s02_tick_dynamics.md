# s02 Tick Dynamics

## Experiment Count

21 experiments:

| Experiment | Loss | Tok/s | Peak Mem MB | Tick Count | Effective Tick |
| --- | ---: | ---: | ---: | ---: | ---: |
| `s02_ctm_tick2` | 5.3994 | 4214 | 29813 | 2 | -1.0 |
| `s02_ctm_tick1` | 5.4104 | 6921 | 18990 | 1 | -1.0 |
| `s02_ctm_tick3` | 5.4598 | 3085 | 40397 | 3 | -1.0 |
| `s02_ctm_tick4` | 5.4903 | 2403 | 50989 | 4 | -1.0 |
| `s02_ctm_tick8_loss_min_conf` | 5.5529 | 1124 | 63195 | 8 | -1.0 |
| `s02_ctm_tick8` | 5.5597 | 1159 | 63195 | 8 | -1.0 |
| `s02_ctm_tick6` | 5.5884 | 1704 | 60800 | 6 | -1.0 |
| `s02_ctm_tick12` | 5.7765 | 634 | 47515 | 12 | -1.0 |
| `s02_ctm_tick16` | 5.7766 | 442 | 61801 | 16 | -1.0 |

Halt and tick-loss variants are also included in this stage.

## What Was Tested

This stage tested whether increasing CTM tick count creates useful deeper computation:

- fixed tick counts from 1 to 16;
- different tick loss modes at tick8;
- halt modes at tick8 and tick16;
- halt thresholds and confidence temperatures.

## Why This Experiment Was Needed

Tick is the central CTM mechanism. If tick depth works, the model should show either:

- better loss as tick count increases;
- adaptive effective tick behavior;
- similar quality with lower compute through halt;
- clearer improvement in later ticks.

## Result

The best fixed tick result is tick2:

| Tick | Loss | Tok/s | Peak Mem MB |
| ---: | ---: | ---: | ---: |
| 1 | 5.4104 | 6921 | 18990 |
| 2 | 5.3994 | 4214 | 29813 |
| 3 | 5.4598 | 3085 | 40397 |
| 4 | 5.4903 | 2403 | 50989 |
| 6 | 5.5884 | 1704 | 60800 |
| 8 | 5.5597 | 1159 | 63195 |
| 12 | 5.7765 | 634 | 47515 |
| 16 | 5.7766 | 442 | 61801 |

Tick8 loss variants did not change the conclusion:

- `min_conf`: loss `5.5529`;
- base tick8: loss `5.5597`;
- mean: loss `5.5728`;
- last: loss `5.5789`.

Halt variants show effective tick values, but they do not improve loss enough. Threshold halt at tick16 often reports effective tick `16.0`, meaning it does not actually save compute. Confidence halt around tick16 reports effective tick around `8.48`, but loss remains around `5.78` to `5.81`.

## Conclusion

More ticks do not currently produce better reasoning. The useful zone is tick1 to tick2. Beyond that, the model pays a large compute and memory cost and quality degrades.

## Linked Comparisons

Read this with:

- `s01_baseline_scale.md`: CTM cost disadvantage is strongly tied to ticks;
- `s03_elf.md`: long horizon ELF plus high tick count worsens the same failure mode;
- `s05_ablations.md`: better CTM results come from architectural simplification, not more ticks.

## Advantage Or Disadvantage Diagnosis

Possible reasons for the disadvantage:

- Tick supervision does not force later ticks to add useful information.
- Repeated ticks may over-process shallow token representations.
- Optimization gets harder as tick count grows.
- Memory grows due to tick traces and per-tick outputs.
- Halt criteria are not yet aligned with actual useful computation.

Possible advantage:

- Tick2 slightly improves over tick1, so a small amount of recurrent computation can help.

## Follow-Up

1. Make tick2 the default research setting for the next round.
2. Add tick curriculum:
   - train tick1/tick2 first;
   - gradually expose tick4+;
   - regularize late ticks to improve over early ticks.
3. Redesign halt:
   - halt should reduce actual compute, not just produce an effective tick metric;
   - compare fixed tick2 against dynamic tick with the same average compute.
4. Add per-token tick analysis:
   - easy tokens should halt early;
   - hard tokens should use more ticks;
   - report token-level effective tick histograms.

