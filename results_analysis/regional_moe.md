# Regional Multi-Pass MoE Sweep

## Experiment Count

82 experiments:

| Stage | Count | Topic |
| --- | ---: | --- |
| rg00 | 3 | dense, single-pass routed, and regional smoke checks |
| rg01 | 10 | regional activation pass count and routed top-k |
| rg02 | 8 | shared/core regions versus routed-only regions |
| rg03 | 8 | load-balance and inter-pass diversity regularization |
| rg04 | 10 | CTM tick count crossed with regional pass count |
| rg05 | 4 | 2000-step first confirmation runs |
| rg06 | 8 | expert granularity and region size |
| rg07 | 7 | d_model/base-size sweep |
| rg08 | 6 | routing warmup and expert dropout |
| rg09 | 6 | tick loss and halt pressure |
| rg10 | 5 | ELF/MTP crossed with regional routing |
| rg11 | 7 | 2000-step broad confirmation runs |

Source summary: `regional_moe_summary.csv` exported from `runs/metrics`.

## What Was Tested

This sweep tested the biological-brain-inspired idea that CTM should not only choose a sparse region once per tick. Instead, one tick can contain multiple sparse regional activation passes. Each pass activates a small core/shared pathway plus a routed region, then the pass outputs are fused.

The main tested axes were:

- one-shot routed MoE versus regional multi-pass routing;
- activation pass count from 1 to 6;
- shared/core region count and routed top-k count;
- inter-pass diversity regularization;
- CTM tick count versus regional pass count;
- expert granularity, region width, and d_model;
- warmup/dropout, tick loss, halt pressure, and ELF/MTP interactions;
- 2000-step confirmation of the strongest candidates.

## Top Results

| Experiment | Loss | Tok/s | Peak Mem GB | Active Fraction | Steps | Key Setting |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `rg11_confirm_d1024_p4_shared1_top1` | 4.8437 | 2598 | 42.5 | 0.125 | 2000 | d1024, tick2, 4 regional passes, shared1/top1 |
| `rg11_confirm_d512_p4_shared1_top1` | 4.9395 | 4418 | 26.0 | 0.125 | 2000 | d512, tick2, 4 regional passes, shared1/top1 |
| `rg11_confirm_d1536_p3_shared1_top1` | 4.9524 | 1819 | 58.4 | 0.083 | 2000 | d1536, tick2, 3 regional passes |
| `rg11_confirm_d512_p3_shared1_top1` | 4.9716 | 4569 | 25.6 | 0.125 | 2000 | d512, tick2, 3 regional passes |
| `rg05_confirm_regional_p2_shared1_top1_e16_s64` | 5.0583 | 2689 | 41.0 | 0.125 | 2000 | d1024, 2 regional passes |
| `rg05_confirm_singlepass_top2_balance_e16_s64` | 5.1404 | 2756 | 40.3 | 0.125 | 2000 | single-pass routed baseline |

## Conclusions From Specific Results

### Regional Multi-Pass Is The New CTM Frontier

The best result is `rg11_confirm_d1024_p4_shared1_top1`:

- loss `4.8437`;
- throughput `2598 tok/s`;
- peak memory `42.5 GB`;
- active fraction `0.125`.

This beats the previous CTM frontier from the sparsity confirm sweep:

- `sp05_confirm_d512_dense_sd2_mh2_tick2`: loss `4.9729`, `4982 tok/s`, `24.1 GB`.

It also beats the earlier dense CTM best:

- `s05_synapse2_mh2`: loss `5.3612`, `2629 tok/s`, `40.6 GB`.

Inference: regional multi-pass routing is not just another sparse mask. It is a stronger CTM inductive bias. It gives CTM a more useful internal computation pattern than simply increasing cell count, increasing tick count, or doing single-pass MoE routing.

### Four Regional Passes Are The Strongest Signal

The rg01 pass sweep at 1000 steps shows the first hint:

| Experiment | Passes | Top-k | Loss |
| --- | ---: | ---: | ---: |
| `rg01_p1_shared1_top1_e16_s64` | 1 | 1 | 5.3567 |
| `rg01_p2_shared1_top1_e16_s64` | 2 | 1 | 5.4580 |
| `rg01_p3_shared1_top1_e16_s64` | 3 | 1 | 5.4769 |
| `rg01_p4_shared1_top1_e16_s64` | 4 | 1 | 5.3591 |
| `rg01_p5_shared1_top1_e16_s64` | 5 | 1 | 5.3635 |
| `rg01_p6_shared1_top1_e16_s64` | 6 | 1 | 5.4225 |
| `rg01_p4_shared1_top2_e16_s64` | 4 | 2 | 5.3440 |

The 2000-step confirmation makes the signal much stronger:

- `rg11_confirm_d1024_p4_shared1_top1`: loss `4.8437`;
- `rg11_confirm_d1024_p3_shared1_top1`: loss `5.1668`;
- `rg11_confirm_d1024_p2_shared1_top1`: loss `5.0992`;
- `rg05_confirm_singlepass_top2_balance_e16_s64`: loss `5.1404`.

Inference: the advantage is not merely "more active experts". The best setting is p4/shared1/top1, not a broad top-k or a larger shared path. Four sparse regional passes appear to create a useful internal sequence of local computations inside one CTM tick.

### d512 p4 Is The Best Quality/Cost Candidate

`rg11_confirm_d512_p4_shared1_top1` reaches:

- loss `4.9395`;
- throughput `4418 tok/s`;
- peak memory `26.0 GB`;
- active fraction `0.125`.

This is only `0.0958` loss worse than the global regional best `rg11_confirm_d1024_p4_shared1_top1`, but it is much faster and uses much less memory.

Inference: d1024 p4 is the quality leader, but d512 p4 is the practical training candidate. If the goal is an efficient CTM direction rather than a one-off best loss, d512 p4 should become the main base.

### d1536 Improves Quality Less Than Its Cost Increase

`rg11_confirm_d1536_p3_shared1_top1` reaches loss `4.9524`, but only `1819 tok/s` and `58.4 GB`.

It is worse than d1024 p4 on quality and much worse than d512 p4 on cost.

Inference: bigger regional CTM is not automatically better. The useful axis is pass structure, not simply larger d_model.

### Shared1 Is The Right Core Path For Now

The best runs all use shared1/top1:

- `rg11_confirm_d1024_p4_shared1_top1`;
- `rg11_confirm_d512_p4_shared1_top1`;
- `rg11_confirm_d1536_p3_shared1_top1`;
- `rg11_confirm_d512_p3_shared1_top1`.

The strongest shared/top-k result outside rg11 is `rg01_p4_shared1_top2_e16_s64`, loss `5.3440`, but its active fraction is `0.1875`, higher than shared1/top1.

Inference: a small always-on core plus one routed region per pass is the cleanest current recipe. Larger shared paths and top2 routing should be treated as quality-ceiling variants, not the default.

### Tick2 Remains The Sweet Spot

The rg04 tick/pass sweep does not show a tick3 advantage:

- `rg04_tick2_p4_shared1_top1_e16_s64`: loss `5.3484`;
- tick3 variants are weaker and more expensive;
- `rg04_tick1_p1_shared1_top1_e16_s64`: loss `5.3632`, `4277 tok/s`, `24.8 GB`.

Inference: regional pass depth is a better lever than adding more CTM ticks. Current next experiments should keep tick2 as the default and only test tick1/tick3 as controls.

### Warmup And Dropout Hurt This Setup

The rg08 schedule sweep is weak:

- `rg08_sched_warm500_drop0p0_p3_shared1_top1`: loss `5.4577`;
- `rg08_sched_warm1000_drop0p0_p3_shared1_top1`: loss `5.5004`;
- `rg08_sched_warm2000_drop0p05_p3_shared1_top1`: loss `5.6055`.

Inference: early broad activation likely weakens the regional sparse bias. For now, do not use top-k warmup or expert dropout as defaults in regional CTM.

### ELF/MTP Still Does Not Help

The rg10 ELF/MTP sweep does not beat plain regional routing:

- `rg10_mtp_mtp_1_2_4_p3_shared1_top1`: loss `5.4623`;
- `rg10_mtp_elf_linear_h2_p3_shared1_top1`: loss `5.4723`;
- `rg10_mtp_elf_linear_h4_p3_shared1_top1`: loss `5.4826`;
- `rg10_mtp_mtp_tickwise_p3_shared1_top1`: loss `5.4977`.

Inference: multi-token prediction is still not the next lever. Regional routing should be stabilized and confirmed first.

### True Cost Savings Are Not Proven Yet

The best d1024 p4 run has active fraction `0.125`, but memory is still `42.5 GB`. This is because the current implementation performs regional masking/fusion, not true sparse dispatch that skips inactive region compute and trace storage.

Inference: the modeling idea is validated, but hardware cost savings require a second engineering step.

## Recommended Next Direction

Use regional p4 as the main branch.

Immediate next experiments:

1. Run 4000/8000-step confirmations:
   - `d512_p4_shared1_top1`;
   - `d1024_p4_shared1_top1`;
   - `d768_p4_shared1_top1`;
   - `d512_p3_shared1_top1`.
2. Run a focused p4 grid:
   - d_model: `512`, `768`, `1024`;
   - shared experts: `0`, `1`, `2`;
   - routed top-k: `1`, `2`;
   - diversity: `0`, `1e-4`, `1e-3`, `3e-3`.
3. Keep tick2 as the default.
4. Do not use warmup/dropout or ELF/MTP in the next mainline unless used as controls.
5. Start true grouped sparse execution once p4 remains strong after longer confirmation:
   - gather active region state before CTM cell computation;
   - compute synapse and trace updates only for active regions;
   - store trace only for active regions;
   - scatter updated regions back at the residual boundary;
   - record active FLOPs, trace memory, and dispatch overhead.

## Working Interpretation

The current best mental model is:

- CTM tick count is global thought depth;
- regional activation pass count is local brain-region sweep depth;
- p4/shared1/top1 is the first setting where local sweep depth becomes useful enough to outperform prior CTM variants.

This suggests that future CTM progress should focus less on simply increasing ticks and more on making each tick internally structured, sparse, and regionally compositional.

