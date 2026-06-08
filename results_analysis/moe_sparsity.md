# MoE Sparsity Sweep

## Experiment Count

32 experiments:

| Stage | Count | Topic |
| --- | ---: | --- |
| moe00 | 2 | dense and routed smoke checks |
| moe01 | 5 | router variants |
| moe02 | 4 | shared experts plus routed experts |
| moe03 | 3 | fine-grained expert sizes/counts |
| moe04 | 5 | router regularization |
| moe05 | 4 | dispatch mode labels |
| moe06 | 5 | warmup and expert dropout |
| moe07 | 4 | sparse routing crossed with ELF/MTP labels |

Source summary: `moe_sparsity_summary.csv` exported from `runs/metrics`.

## Top Results

| Experiment | Loss | Tok/s | Peak Mem GB | Active Fraction | MoE Aux Loss | Key Setting |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `moe04_router_balance1e2_e16_s64_k2` | 5.4439 | 2595 | 40.3 | 0.125 | 0.000006 | top-k, 16 experts, k=2, balance=1e-2 |
| `moe02_shared2_routed4_e16_s64` | 5.4506 | 2627 | 39.5 | 0.375 | 0.000000 | 2 shared + 4 routed experts |
| `moe06_warmup1000_drop0p05_e16_s64_k2` | 5.4509 | 2636 | 39.5 | 0.125 | 0.000000 | top-k warmup 1000, dropout 0.05 |
| `moe03_fine_e16_s64_k2` | 5.4536 | 2634 | 39.5 | 0.125 | 0.000000 | 16 fine-grained experts, k=2 |
| `moe06_warmup1000_drop0p0_e16_s64_k2` | 5.4594 | 2629 | 39.5 | 0.125 | 0.000000 | top-k warmup 1000 |
| `moe06_warmup2000_drop0p05_e16_s64_k2` | 5.4607 | 2618 | 39.5 | 0.125 | 0.000000 | top-k warmup 2000, dropout 0.05 |
| `moe02_shared2_routed2_e16_s64` | 5.4690 | 2597 | 39.5 | 0.250 | 0.000000 | 2 shared + 2 routed experts |
| `moe04_router_balance1e3_e16_s64_k2` | 5.4690 | 2605 | 40.3 | 0.125 | 0.000001 | top-k, 16 experts, k=2, balance=1e-3 |

## What Was Tested

This sweep tested MoE ideas on top of CTM cell sparsity:

- learned top-k routing with different k values;
- expert-choice and hash routing;
- shared experts plus routed experts;
- more fine-grained expert partitions;
- load-balance, entropy, z-loss, and aux-free-bias router regularization;
- warmup and expert dropout;
- dispatch-mode interfaces for dense-mask, dropless, capacity-drop, and block-sparse paths;
- sparse routing crossed with ELF/MTP labels.

## Why This Experiment Was Needed

The earlier top-k cell sparsity sweep showed that CTM can tolerate fewer active cells, but the mask did not create real cost savings. This MoE sweep asks a sharper question: can cell groups be routed like experts, so that the architecture keeps quality with much lower active fraction and creates a cleaner path toward true sparse execution?

## Result

The best MoE run is:

- `moe04_router_balance1e2_e16_s64_k2`: loss `5.4439`, `2595 tok/s`, `40.3 GB`, active fraction `0.125`.

This is close to the previous dense CTM best from the first full sweep:

- `s05_synapse2_mh2`: loss `5.3612`, `2629 tok/s`, `40.6 GB`, active fraction `1.0`.

The important signal is that MoE can activate only about one eighth of the cell groups while staying within roughly `0.08` loss of that dense CTM candidate. That is a real modeling result.

However, this is not yet a real infrastructure win. Throughput and memory are almost unchanged versus the dense CTM candidate because the current implementation still uses masking/group selection rather than a true sparse dispatch kernel. It records sparse active fraction, but it does not yet skip enough tensor work, activation storage, or trace storage.

## What Worked

### Load Balance

`moe04_router_balance1e2_e16_s64_k2` is the best overall result. A small load-balance auxiliary loss appears useful for CTM-style routing. It likely prevents early expert collapse without forcing the router too hard.

The weaker `1e-3` setting is still good, but not best:

- `balance=1e-2`: loss `5.4439`;
- `balance=1e-3`: loss `5.4690`.

### Shared Experts

Shared experts are strong:

- `moe02_shared2_routed4_e16_s64`: loss `5.4506`, active fraction `0.375`;
- `moe02_shared2_routed2_e16_s64`: loss `5.4690`, active fraction `0.250`;
- `moe02_shared1_routed2_e16_s64`: loss `5.4934`, active fraction `0.188`.

This suggests a useful CTM pattern: keep a small always-on pathway for stable common computation, then route the rest sparsely.

### Warmup And Expert Dropout

Top-k warmup is consistently strong:

- `warmup=1000, dropout=0.05`: loss `5.4509`;
- `warmup=1000, dropout=0.0`: loss `5.4594`;
- `warmup=2000, dropout=0.05`: loss `5.4607`;
- `warmup=500, dropout=0.0`: loss `5.4756`.

The best warmup setting is almost tied with the best load-balance result. This supports the idea that hard sparse routing should emerge gradually during training instead of being imposed too abruptly.

### Aggressive Top-1 Routing

`moe01_router_top1_e16_s64_k1` reaches loss `5.5162` with active fraction `0.062`. It is not the best quality result, but it is important because it is the strongest extreme-sparsity candidate. It deserves longer confirmation after adding load-balance.

## What Did Not Work Yet

### Hash And Expert Choice

Hash routing and expert-choice routing are not first-tier in this run:

- `moe01_router_hash_e16_s64_k2`: loss `5.6260`;
- `moe01_router_expert_choice_e16_s64_k2`: loss `5.5697`.

Expert-choice has the highest throughput among the MoE runs, but the quality gap makes it less attractive as the next primary direction.

### Router Z-Loss

`moe04_router_zloss1e3_e16_s64_k2` is clearly weak at loss `5.8143`. The tested z-loss strength appears too harmful for this CTM routing setup.

### Aux-Free Bias

`moe04_router_auxfree_bias_e16_s64_k2` reaches loss `5.5712`, which is worse than simple load-balance and warmup. This idea should not lead the next round.

### Dispatch Mode Labels

The dispatch-mode experiments are useful API checks but not final performance evidence:

- `dense_mask`: best loss `5.4439`;
- `block_sparse`: loss `5.5382`;
- `capacity_drop`: loss `5.5669`;
- `dropless`: loss `5.5685`.

These modes do not yet implement a true high-performance sparse kernel path, so memory and throughput should not be interpreted as the real ceiling for sparse CTM.

## Comparison To Previous CTM Results

| Candidate | Loss | Tok/s | Peak Mem GB | Active Fraction | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `s05_synapse2_mh2` | 5.3612 | 2629 | 40.6 | 1.0 | best dense CTM from first sweep |
| `moe04_router_balance1e2_e16_s64_k2` | 5.4439 | 2595 | 40.3 | 0.125 | best MoE sparse candidate |
| `moe02_shared2_routed4_e16_s64` | 5.4506 | 2627 | 39.5 | 0.375 | best shared-expert candidate |
| `moe06_warmup1000_drop0p05_e16_s64_k2` | 5.4509 | 2636 | 39.5 | 0.125 | best warmup candidate |
| `sp05_confirm_d512_dense_sd2_mh2_tick2` | 4.9729 | 4982 | 24.1 | 1.0 | stronger later sparse-confirm base |
| `s01_transformer_12l_h640` | 4.6791 | 39484 | 4.9 | 1.0 | Transformer still dominates cost/quality |

The MoE sweep should be read as a routing/sparsity validation, not as a new global best. The later d512 confirm run remains a better CTM base by loss and cost. The value of MoE is that it shows CTM can preserve reasonable quality with much lower active fraction.

## Conclusion

MoE-style CTM routing is promising. The best result activates only `12.5%` of cell groups while staying close to the earlier dense CTM candidate. Load-balance, shared experts, and warmup are the strongest ideas.

The missing piece is true sparse execution. Until inactive expert groups are skipped at the tensor/kernel level, the active fraction is a modeling metric rather than a cost metric.

## Recommended Next Experiments

1. Confirm the first-tier MoE candidates for 4000 or 8000 steps:
   - `balance1e-2 + e16 + k2`;
   - `warmup1000 + dropout0.05 + e16 + k2`;
   - `shared1 + routed2 + balance1e-2`;
   - `shared2 + routed4 + balance1e-2`;
   - `top1 + balance1e-2`.
2. Move the MoE router onto the stronger d512/tick2/sd2_mh2 sparse-confirm base.
3. Add true grouped sparse execution:
   - gather active expert groups before CTM cell computation;
   - compute only active cell projections;
   - store trace only for active groups;
   - scatter active groups back into the dense state only at the residual boundary.
4. Add cost instrumentation:
   - routed active FLOPs;
   - dispatch/gather/scatter overhead;
   - trace memory by active group;
   - lm_head/tick auxiliary cost.
5. Keep Transformer controls in the loop at fixed memory, fixed wall-clock, and fixed token budget.

