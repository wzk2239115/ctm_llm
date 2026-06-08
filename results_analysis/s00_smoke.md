# s00 Smoke Experiments

## Experiment Count

2 experiments:

| Experiment | Type | Loss | Tok/s | Peak Mem MB |
| --- | --- | ---: | ---: | ---: |
| `s00_transformer_smoke` | transformer | 14.4588 | 76617 | 1033 |
| `s00_ctm_smoke` | ctm | 11.1223 | 24864 | 4773 |

## What Was Tested

This stage tested whether both model paths can train end-to-end and write usable metrics:

- standard Transformer smoke path;
- CTM smoke path with a small 4-layer, 3-tick configuration.

## Why This Experiment Was Needed

Before running the full CTM plan, the infrastructure needed a minimal verification that:

- model creation works;
- distributed training launches correctly;
- metric CSV logging works;
- CTM-specific outputs such as per-tick losses and certainties are produced.

## Result

Both smoke experiments completed. CTM smoke loss is lower than Transformer smoke in this tiny setup, but this is not a meaningful model-quality comparison because the smoke configurations are intentionally small and not matched as serious baselines.

The important result is infrastructure reliability: both paths produced metrics, allowing the later 71-experiment plan to run.

## Conclusion

The smoke stage succeeded. It should be treated as a system check, not as evidence that CTM beats Transformer.

## Linked Comparisons

Use this stage only as a sanity reference for:

- `s01_baseline_scale.md`, where serious Transformer-vs-CTM comparisons begin;
- batch profiling and pool scheduling stability.

## Advantage Or Disadvantage Diagnosis

CTM smoke uses more memory than Transformer smoke because CTM stores and computes additional recurrent state and tick outputs. This overhead appears throughout the full sweep.

## Follow-Up

Keep smoke experiments in the plan, but reduce their role to CI-like checks:

- verify metrics writing;
- verify CTM forward/backward;
- verify Transformer baseline path;
- verify pool launch before expensive sweeps.

