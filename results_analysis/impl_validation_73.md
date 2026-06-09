# Implementation Validation 73

Source summary: `impl_validation_73_summary.csv`.

Experiment script: `scripts/experiment_plan_impl_validation.py`.

## Experiment Count

73 experiments:

| Stage | Count | Valid Loss | Topic |
| --- | ---: | ---: | --- |
| iv00 | 5 | 5 | smoke anchors for dense, single-pass MoE, regional sparse, halt, and MTP |
| iv01 | 8 | 6 | backend controls: dense/top-k/single-pass/regional at d512-d1536 |
| iv02 | 10 | 8 | regional activation pass count and routed top-k after grouped sparse implementation |
| iv03 | 7 | 5 | d_model, pass count, shared path, and routed top-k base grid |
| iv04 | 10 | 10 | real tick early-exit and compute penalty |
| iv05 | 10 | 10 | MTP/ELF loss variants on regional sparse backend |
| iv06 | 8 | 8 | composition of regional sparse, halt, and MTP |
| iv07 | 9 | 7 | longer confirmation runs for implementation-validation candidates |
| iv08 | 6 | 4 | dispatch/capacity mode checks |

Overall, 63 of 73 experiments produced a finite loss. 10 experiments produced `NaN` loss.

## What Was Tested

This experiment was an implementation-validation sweep, not a normal architecture sweep. The purpose was to check whether the promising regional CTM ideas from earlier masked-routing experiments still work after wiring the newer implementation paths:

- sequential grouped sparse regional backend through `moe_dispatch_mode=block_sparse`;
- real tick early-exit through `tick_halt_mode=threshold/confidence`;
- MTP multi-horizon loss through `moe_mtp_mode`;
- dispatch/capacity flags including `dense_mask`, `block_sparse`, `dropless`, and `capacity_drop`;
- longer confirmation runs for the candidates that looked plausible in the short validation stages.

The plan generator is `scripts/experiment_plan_impl_validation.py`. The summary was exported by its `summarize` subcommand, which collects final `iv00_` to `iv08_` metric rows from `runs/metrics`.

## Result

The stable results are mostly the dense-mask or non-regional controls:

| Experiment | Loss | Tok/s | Peak Mem GB | Quality Cost Score | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `iv01_backend_singlepass_d1024_e16_top2` | 5.7076 | 2098 | 15.3 | 0.0416 | best finite loss among normal-speed controls |
| `iv08_dispatch_regional_densemask_label_d512_p4` | 5.7122 | 2092 | 10.2 | 0.0279 | dense-mask regional label path is healthy |
| `iv01_backend_topk_d512_k256` | 5.7832 | 3028 | 9.7 | 0.0186 | best quality/cost score |
| `iv01_backend_dense_d1024` | 5.8119 | 2247 | 14.8 | 0.0382 | dense d1024 control is healthy |
| `iv01_backend_dense_d512` | 5.8146 | 2891 | 9.6 | 0.0192 | dense d512 control is healthy |

The new grouped sparse regional path is not yet healthy. Some block-sparse regional rows produce finite losses, but they are either extremely slow or numerically bad:

- `iv03_base_d1024_p4_shared1_top2`: loss `5.7189`, but only `99 tok/s`;
- `iv02_pass_p4_shared1_top2_d1024`: loss `5.7604`, but only `87 tok/s`;
- `iv03_base_d512_p4_shared1_top1`: loss `96.1680`;
- `iv03_base_d1024_p4_shared1_top1`: loss `295.2789`;
- `iv03_base_d768_p4_shared1_top1` and `iv03_base_d1536_p3_shared1_top1`: `NaN`.

Real halt does not save compute yet. All `iv04` halt variants report `effective_tick=1`, but throughput is much lower than the no-halt tick2 control, and losses are not in the useful range:

- no-halt tick2 control: loss `54.8524`, `348 tok/s`;
- best halt loss: `iv04_halt_threshold0p45_tick4_d512_p4`, loss `42.0721`, only `64 tok/s`;
- threshold `0.00` is especially unstable at loss `226.5651`.

MTP/ELF also does not help on this implementation path:

- best `iv05` row is `iv05_mtp_elf_pow2_h4_d512_p4`, loss `36.5999`;
- plain no-MTP regional control is loss `1756.1048`;
- long MTP horizon `1,2,4,8` is loss `1154.3016`.

The longer `iv07` confirmation runs do not rescue the grouped sparse path. The best is `iv07_confirm_d512_p4_plain` with loss `7.4439`, but it is still much worse than the dense/top-k controls and far worse than the earlier masked regional results. The d768 confirm rows are `NaN`, and halt/MTP confirmations degrade badly.

The dispatch check in `iv08` isolates the problem most clearly:

- `dense_mask`: loss `5.7122`, `2092 tok/s`, healthy;
- `block_sparse`: loss `145.5542`, `345 tok/s`, unhealthy;
- `dropless`: loss `785.1353`, unhealthy;
- `capacity_drop`: `NaN` at capacity `0.75` and `1.00`, and loss `15332.9920` at capacity `1.25`.

## What This Means

This sweep does not validate the new grouped sparse backend as a training path yet. It validates the opposite: the modeling idea remains plausible in the dense-mask/control paths, but the new implementation path has a correctness or training-dynamics problem.

The important distinction is:

- dense/top-k/dense-mask controls train in the expected loss range around `5.7` to `5.8`;
- `block_sparse`, `dropless`, and `capacity_drop` frequently produce huge loss, `NaN`, or severe throughput collapse;
- halt and MTP are confounded by the broken regional sparse path, so their negative result should not be interpreted as a final architecture conclusion.

The earlier regional result should therefore be treated as a masked-routing modeling signal, not as proof of real sparse execution. This experiment is useful because it found the exact next engineering bottleneck: grouped sparse dispatch must be made numerically equivalent to dense-mask routing before it can be used for architecture conclusions.

## Recommended Next Experiments

1. Add a block-sparse parity test before more training sweeps.
   Compare `dense_mask` and `block_sparse` for the same routing decisions on a fixed mini-batch. Check logits, per-tick losses, active region indices, gather/scatter order, residual merge, and aux loss.

2. Reduce the grouped sparse backend to a minimal reproducible case.
   Use d512, 16 experts, expert size 32, shared1/top1, p1 or p2, tick1/tick2, no halt, no MTP, no capacity drop. Run 20-step and 100-step smoke checks until loss and logits match the dense-mask path.

3. Disable capacity/drop-token experiments until block-sparse parity is proven.
   The `capacity_drop` rows are too unstable to diagnose while the base dispatch path is already failing.

4. Keep halt and MTP out of the mainline until the regional sparse backend is stable.
   Reintroduce them only as isolated controls after block-sparse matches dense-mask on short training.

5. After parity passes, rerun a compact validation matrix:
   - `dense_mask` vs `block_sparse`;
   - p1, p2, p4;
   - shared1/top1 and shared1/top2;
   - d512 and d1024;
   - 100-step smoke, then 1000-step confirmation.

6. Add implementation metrics alongside loss:
   active FLOPs, gather/scatter overhead, trace memory for active regions, inactive-region skip ratio, and per-dispatch timing. The current active fraction is not enough to prove real cost savings.

## Bottom Line

`impl_validation_73` found a backend implementation problem, not a new best architecture. The safe next direction is to fix and prove `block_sparse` equivalence against `dense_mask`, then rerun the regional p4/shared1/top1 experiments. Until then, architecture decisions should continue to use the stable dense/top-k/dense-mask controls rather than the grouped sparse regional rows.
