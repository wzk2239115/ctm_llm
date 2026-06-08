# s03 ELF And Multi-Token Prediction

## Experiment Count

14 experiments:

| Experiment | Loss | Tok/s | Peak Mem MB |
| --- | ---: | ---: | ---: |
| `s03_elf_next` | 5.5004 | 2375 | 50992 |
| `s03_elf_linear_h4_improve` | 5.5059 | 2409 | 50991 |
| `s03_elf_linear_h4` | 5.5105 | 2388 | 50991 |
| `s03_elf_pow2_h4` | 5.5125 | 2408 | 50990 |
| `s03_elf_linear_h8_improve_w0p3` | 5.5422 | 1102 | 63191 |
| `s03_elf_linear_h8` | 5.5706 | 1121 | 63191 |
| `s03_elf_linear_t12_h8` | 5.7823 | 574 | 47514 |
| `s03_elf_linear_t16_h8` | 5.7732 | 423 | 61800 |

## What Was Tested

This stage tested ELF-style multi-token prediction:

- next-token baseline under the ELF stage;
- linear horizon modes;
- pow2 horizon mode;
- horizon 4 and horizon 8;
- improvement weights;
- combinations with tick8, tick12, and tick16.

## Why This Experiment Was Needed

One of the concerns was that CTM-LLM was not using ELF's ability to predict multiple tokens. This stage tested whether multi-token prediction can make ticks more useful or provide a better learning signal.

## Result

Short horizon ELF is close to the tick4 CTM baseline but does not create a large gain:

- `s03_elf_next`: loss `5.5004`;
- `s03_elf_linear_h4_improve`: loss `5.5059`;
- `s03_elf_linear_h4`: loss `5.5105`;
- `s03_elf_pow2_h4`: loss `5.5125`.

Horizon 8 and high tick settings degrade:

- `s03_elf_linear_h8`: loss `5.5706`;
- `s03_elf_linear_t12_h8`: loss `5.7823`;
- `s03_elf_linear_t16_h8`: loss `5.7732`.

The strongest improvement-weight variant is `h8_improve_w0p3`, loss `5.5422`, but it is still not strong enough relative to simpler CTM settings.

## Conclusion

ELF is not yet unlocking multi-token advantage. Short horizon variants are acceptable but not clearly better. Long horizon combined with high tick count amplifies the cost and optimization problems seen in the tick sweep.

## Linked Comparisons

Use this stage with:

- `s02_tick_dynamics.md`: high tick count is already weak before ELF is added;
- `s05_ablations.md`: simpler CTM ablations beat most ELF variants;
- Transformer baselines in `s01_baseline_scale.md`: ELF still does not close the baseline gap.

## Advantage Or Disadvantage Diagnosis

Potential reasons for weak ELF results:

- Multi-token loss may be too weakly coupled to useful hidden-state improvements.
- Horizon targets may be too hard early in training.
- Tick and ELF objectives may compete instead of cooperate.
- The current lm_head training path may add cost without producing robust intermediate representations.

Potential advantage:

- Horizon 4 is stable. That makes it a reasonable candidate for focused redesign.

## Follow-Up

1. Keep ELF horizon short first: horizon 2 or 4.
2. Train ELF as an auxiliary objective on the best CTM base, not on a high-tick default.
3. Add explicit reporting:
   - next-token loss;
   - horizon-2 loss;
   - horizon-4 loss;
   - whether future-token predictions improve downstream next-token loss.
4. Test ELF with tick2 and `s05_synapse2_mh2`.
5. Avoid combining long horizon with tick12/tick16 until tick dynamics are healthier.

