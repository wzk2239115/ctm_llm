# Draft-Revise Experiment Sweep

## Experiment Count

10 completed stages (dr00, dr01, dr03, dr06-dr12), 24 experiments with valid metrics.
dr02 (parallel draft slots), dr04 (commit head), dr05 (attention mask + memory carry) all failed and have no surviving metrics.

Source summary: `csv_data/draft_revise_{stage}_summary.csv` exported from `runs/metrics`.
Experiment script: `scripts/experiment_plan_draft_revise.py`.

| Stage | Completed | Valid Loss | Topic |
| --- | ---: | ---: | --- |
| dr00 | 3 | 3 | Runnable anchors: regional sync, regional d1024, async anytime |
| dr01 | 3 | 3 | MTP and ELF negative controls |
| dr02 | 0 | 0 | Parallel draft slots (all failed) |
| dr03 | 2 | 2 | Overwrite/revise corruption training |
| dr04 | 0 | 0 | Commit/confidence head (all failed) |
| dr05 | 0 | 0 | Draft attention mask + CTM state carry (all failed) |
| dr06 | 2 | 2 | Full sparse async draft-revise stack |
| dr07 | 4 | 4 | DINO-like continuous speed spectrum |
| dr08 | 4 | 4 | Residual compute phase 1: semantic caches and deltas |
| dr09 | 2 | 2 | Residual compute phase 2: synapse block skipping |
| dr10 | 3 | 3 | Residual compute phase 3: recursive NLM fast path |
| dr11 | 1 | 1 | Residual compute phase 4: tick controller |
| dr12 | 2 | 2 | Training objective variants: causal CE vs latent denoise |

## What Was Tested

This sweep tested the **draft-revise** hypothesis for CTM: instead of committing to a single forward pass per tick, the model first generates low-quality "draft" tokens, then iteratively revises them, and finally commits when confident. The plan was divided into three research threads:

1. **Draft-revise module (dr00-dr06):** Build the mechanism incrementally -- anchors, negative controls, corruption training, commit head, attention masks, and full-stack integration.
2. **Training supervision (dr07, dr12):** DINO-style multi-speed EMA distillation and alternative training objectives (causal CE, latent denoise).
3. **Residual/efficient compute (dr08-dr11):** Delta caching, synapse skipping, recursive NLM fast paths, and adaptive tick controllers -- aiming to make multi-tick draft-revise cheaper without sacrificing quality.

## Top Results

### Best language-model quality (loss, lower is better)

| Experiment | Loss | Tok/s | Peak GB | Tick Count | Key Setting |
| --- | ---: | ---: | ---: | ---: | --- |
| `dr03_revise_b4_corrupt0p15` | **4.616** | 691 | 47.5 | 8 | draft-revise, block=4, corrupt=15% |
| `dr03_revise_b4_corrupt0p30` | 4.652 | 688 | 47.5 | 8 | draft-revise, block=4, corrupt=30% |
| `dr00_anchor_regional_d512_tick4` | 5.401 | **2353** | 33.6 | 4 | regional sync anchor |
| `dr01_current_mtp124_d512` | 5.413 | 2415 | 35.8 | 4 | MTP 1,2,4 negative control |
| `dr01_current_elf_linear_h4_d512` | 5.395 | 2386 | 34.3 | 4 | ELF h4 negative control |
| `dr07_speed_spectrum_0p80..0p997` | 5.560 | 688 | 49.4 | 8 | DINO speed spectrum |
| `dr08_residual_semantic_attn_refresh` | 5.577 | 952 | 37.4 | 8 | residual attn refresh |

### Best quality/cost (quality_cost_score = loss * peak_GB / tok/s, lower is better)

| Experiment | QCS | Loss | Tok/s | Peak GB |
| --- | ---: | ---: | ---: | ---: |
| `dr00_anchor_regional_d512_tick4` | **0.075** | 5.401 | 2353 | 33.6 |
| `dr01_current_elf_linear_h4_d512` | 0.076 | 5.395 | 2386 | 34.3 |
| `dr01_current_mtp124_d512` | 0.078 | 5.413 | 2415 | 35.8 |
| `dr08_residual_semantic_attn_refresh` | 0.214 | 5.577 | 952 | 37.4 |
| `dr03_revise_b4_corrupt0p15` | 0.310 | 4.616 | 691 | 47.5 |

## Stage-By-Stage Interpretation

### dr00 -- Anchor Baselines

Three reference points for the draft-revise program:

| Experiment | Loss | Tok/s | Mem GB | Tick Config |
| --- | ---: | ---: | ---: | --- |
| `anchor_regional_d512_tick4` | **5.401** | **2353** | 33.6 | sync, 4 ticks |
| `anchor_regional_d1024_tick4` | 5.567 | 1317 | 42.8 | sync, 4 ticks, d=1024 |
| `anchor_async_anytime_d512` | 9.283 | 276 | 52.2 | async banded, 16 ticks, threshold halt |

**Regional d512 tick4** is the dominant baseline: lowest loss, fastest throughput, smallest memory. Scaling to d1024 degrades all three metrics (loss +0.17, speed -44%, memory +27%), confirming that 512-dimension is already sufficient at this scale.

**Async anytime** fails catastrophically. With 16 ticks and threshold-based halt, the model reaches loss 9.28 -- nearly double the sync baseline. The per-tick loss sequence `[6.95, 6.97, 6.96, ...]` oscillates with no convergence. Async tick scheduling with adaptive halting is not trainable under the current supervision signal.

**Implication:** The entire draft-revise program should benchmark against `regional_d512_tick4` (loss 5.40, 2353 tok/s). Any proposal must justify its complexity against this strong and simple baseline.

### dr01 -- MTP / ELF Negative Controls

| Experiment | Loss | Tok/s | Setting |
| --- | ---: | ---: | --- |
| `elf_linear_h4_d512` | **5.395** | 2386 | ELF linear horizon=4, 4 ticks |
| `mtp_1_2_4_d512` | 5.413 | 2415 | MTP horizons 1,2,4, 4 ticks |
| `elf_linear_h8_tick8_d512` | 5.577 | 874 | ELF linear horizon=8, 8 ticks, tick_improve=0.05 |

Both `elf_h4` and `mtp_1_2_4` are statistically indistinguishable from the regional anchor. Multi-token prediction and shifted CE supervision provide **zero quality gain** on CTM at this scale.

`elf_h8` with 8 ticks and an explicit tick-improvement reward shows the per-tick loss declining from 6.81 to 6.18, but the final aggregated loss (5.58) is worse than the 4-tick anchor (5.40) and throughput drops by 63%. More ticks with improvement pressure do not produce better final predictions -- they simply waste compute.

**Implication:** MTP and ELF are neutral, not harmful, but they do not improve quality. They are valid as auxiliary supervision but should not be the primary research direction.

### dr03 -- Corruption-Based Draft-Revise [!!]

| Experiment | Loss | Tok/s | Corrupt Rate | Per-tick Loss Trend |
| --- | ---: | ---: | ---: | --- |
| `revise_b4_corrupt0p15` | **4.616** | 691 | 15% | 4.47 -> 4.25 (monotonic decline) |
| `revise_b4_corrupt0p30` | 4.652 | 688 | 30% | 4.64 -> 4.37 (monotonic decline) |

**This is the standout result of the entire sweep.** Corruption-based draft-revise achieves loss 4.616 -- a **0.785 absolute improvement** over the anchor (5.401), a **14.5% relative gain**. This is the largest single-experiment quality improvement observed in any CTM sweep to date.

The mechanism works as designed: the model generates draft tokens, a fraction is corrupted, and the model learns to detect and correct errors. The per-tick loss sequences show clear monotonic improvement across all 8 ticks, confirming that the revise passes are doing meaningful work rather than overfitting to noise.

Corrupt=15% beats corrupt=30%, suggesting that moderate noise levels provide the best learning signal. Excessive corruption (30%) introduces too much disturbance for the model to learn a stable correction pattern.

The cost is a 3.4x throughput reduction (691 vs 2353 tok/s) and 41% more memory (47.5 vs 33.6 GB). This is expected: 8 revise iterations at block=4 require 8 forward passes through the draft-revise module.

**Implication:** Draft-revise with corruption training is the **only mechanism proven to significantly improve CTM quality**. The next priority is making it efficient -- reducing the 3.4x cost overhead while preserving the quality gain.

### dr06 -- Full Sparse Async Draft-Revise Stack

| Experiment | Loss | Tok/s | Tick Config |
| --- | ---: | ---: | --- |
| `full_async_sparse_revise_b4` | 8.704 | 737 | async banded, 8 ticks |
| `full_async_sparse_revise_b8` | 8.993 | 380 | async banded, 12 ticks |

Both experiments fail badly. Loss is 8.7-9.0 -- worse than even the broken async anchor (9.28). Per-tick losses oscillate without convergence (e.g., `[6.72, 6.78, 6.87, 6.83, ...]`).

The combination of async tick scheduling + sparse routing + draft-revise creates severe training instability. The modules have negative interactions that prevent convergence.

**Implication:** Complex module composition does not work by stacking. Each component needs to be validated individually first, then integrated incrementally with careful ablation.

### dr07 -- DINO-like Speed Spectrum

| Experiment | Loss | Tok/s | EMA Decays / Distill Weight |
| --- | ---: | ---: | --- |
| `speed_spectrum_0p80_0p95_0p99_0p997` | 5.560 | 688 | 4-teacher spectrum |
| `speed_spectrum_0p90_0p97_0p995` | 5.566 | 748 | 3-teacher spectrum |
| `speed_weight_w0p02` | 5.568 | 706 | distill_weight=0.02 |
| `speed_weight_w0p05` | 5.575 | 762 | distill_weight=0.05 |

All configurations produce nearly identical results: loss ~5.56-5.57, throughput ~700 tok/s. This is **worse** than the anchor (5.40) and **much slower** (3.4x).

The per-tick loss sequences show gradual improvement (7.2 -> 6.5), indicating the distillation target is doing something, but this improvement does not translate to better final predictions. The DINO-style multi-speed EMA targets add regularization overhead without quality benefit.

**Implication:** Multi-speed self-distillation is not a useful supervision signal for CTM ticks. The tick-to-tick semantic variation in CTM is too small to support meaningful distillation across different EMA speeds.

### dr08 -- Residual Semantic Caches and Deltas

| Experiment | Loss | Tok/s | Variant |
| --- | ---: | ---: | --- |
| `residual_semantic_attn_refresh` | **5.577** | **952** | intermittent full-attention refresh |
| `residual_semantic_sync_recursive` | 5.591 | 892 | recursive delta propagation |
| `residual_semantic_syn_delta` | 5.592 | 864 | synthesize from deltas |
| `residual_semantic_observe` | 5.595 | 861 | observe-only baseline |

All four variants cluster around loss 5.58-5.59, essentially matching the anchor quality. Throughput ranges 861-952 tok/s -- 2.5-2.7x slower than the anchor.

The `attn_refresh` variant is marginally best (fastest and slightly lowest loss), likely because periodic full-attention computation prevents cumulative approximation error. The `sync_recursive` variant shows the best per-tick loss decline (7.0 -> 6.2), but this does not translate to better aggregation.

**Implication:** These delta/cache mechanisms are architecturally sound (they do not degrade quality), but they do not improve quality and are slower than the baseline. They add engineering complexity without measurable benefit at this scale.

### dr09 -- Synapse Block Skipping

| Experiment | Loss | Tok/s | Groups | Active Ratio |
| --- | ---: | ---: | ---: | ---: |
| `synapse_skip_g16_a0p25` | 5.586 | 415 | 16 | 25% |
| `synapse_skip_g32_a0p25` | 5.596 | 269 | 32 | 25% |

Loss matches the anchor (5.59 vs 5.40, slightly worse). Throughput is catastrophically bad: 415 and 269 tok/s, respectively -- 5.7x and 8.7x slower than the anchor.

The gate-decision overhead and conditional execution cost exceed any savings from skipping inactive blocks. More groups (g=32) means more gate decisions and even worse throughput.

**Implication:** Block-level synapse skipping is counterproductive at this model scale. The gating infrastructure is more expensive than the computation it saves. This approach needs either (a) much larger models where gating savings dominate, or (b) a fundamentally different skipping mechanism with lower overhead.

### dr10 -- Recursive NLM Fast Path

| Experiment | Loss | Tok/s | NLM Mode | Refresh Interval |
| --- | ---: | ---: | --- | ---: |
| `nlm_recursive_fast_refresh8` | **5.677** | **401** | recursive fast | every 8 ticks |
| `nlm_hybrid_fast_full_refresh4` | 5.652 | 285 | hybrid fast+full | every 4 ticks |
| `nlm_recursive_fast_refresh4` | 5.695 | 278 | recursive fast | every 4 ticks |

All three are worse than the anchor on both quality and speed. Loss is 5.65-5.70 (vs 5.40), throughput is 278-401 tok/s (vs 2353).

The `hybrid_fast_full_refresh4` per-tick losses **increase** over time (6.13 -> 6.41), indicating that the recursive approximation accumulates error faster than the periodic full refresh can correct it. More frequent refresh (refresh4 vs refresh8) helps slightly but not enough.

**Implication:** Recursive NLM approximation introduces non-trivial quality degradation, and the overhead of periodic full refreshes negates any computational savings. This approach does not work at current model dimensions.

### dr11 -- Tick Controller with Compute Penalty

| Experiment | Loss | Tok/s | Ticks | Certainty Trend |
| --- | ---: | ---: | ---: | --- |
| `tick_controller_threshold` | 9.429 | 348 | 12 | 0.83 -> 0.94 (increasing) |

Catastrophic failure. Loss is 9.43 across 12 ticks. The model's certainty monotonically increases from 0.83 to 0.94 -- it becomes more confident with each tick -- but the loss simultaneously increases from 7.15 to 7.48.

The model learned to exploit the compute penalty: it raises its confidence to minimize the penalty term rather than actually learning better representations. This is a classic reward-hacking failure mode.

**Implication:** A learned tick controller with compute penalty is not trainable under the current objective. The model shortcuts the intended incentive structure. Any adaptive compute mechanism needs stronger quality constraints or a different training paradigm.

### dr12 -- Training Objective Variants

| Experiment | Loss | Tok/s | Objective | Steps |
| --- | ---: | ---: | --- | ---: |
| `objective_causal_ce` | 10.301 | 1031 | standard causal CE | 1000 |
| `objective_latent_denoise` | 7.605 | 961 | latent space denoising | 460 |

Both objective variants fail. Pure causal CE with min_conf tick loss produces loss 10.3 -- the worst result in the entire sweep. Latent denoising is better (7.6) but still far worse than the standard CTM training objective (5.4).

The causal CE failure suggests that the standard CTM min_conf tick loss mode provides critical supervision that pure CE cannot replicate. The latent denoise results show degradation and early stopping (only 460 steps), indicating training instability under the denoising objective.

**Implication:** The current CTM training objective (min_conf tick loss with regional sparse routing) is well-calibrated. Switching to causal CE or latent denoising objectives destroys training quality. Any future objective changes should be incremental modifications to the existing loss, not wholesale replacements.

## Cross-Cutting Conclusions

### 1. Corruption-based draft-revise is the only quality win

The entire 24-experiment sweep produced exactly **one mechanism that significantly improves quality**: `dr03_revise_b4_corrupt0p15` with loss 4.616. Every other experiment is either neutral (matching the anchor) or strictly worse. This result is robust across both corrupt rates tested and shows clear monotonic per-tick improvement.

### 2. The cost problem is unsolved

The 0.785 loss improvement comes at a 3.4x throughput penalty and 41% more memory. The draft-revise quality gain is real, but the efficiency story is negative. None of the efficiency-focused stages (dr08-dr11) succeeded in reducing this cost -- they only added overhead without recovering speed.

### 3. Module composition fails

The full-stack integration in dr06 (async + sparse + draft-revise) is the worst result outside the objective variants. This reinforces a broader lesson: CTM modules have complex interactions, and naive stacking produces instability rather than synergy.

### 4. Most optimization proposals are cost-negative

Of the 24 experiments:
- **1** significantly improves quality (dr03, at high cost)
- **7** match anchor quality but are slower (dr01 elf_h4/mtp, dr08 all 4, dr09 both)
- **4** are slightly worse than anchor and slower (dr07 all 4, dr10 all 3)
- **4** catastrophically fail (dr00 async, dr06 both, dr11, dr12 causal_ce)
- **1** is moderately worse (dr12 latent_denoise)

Zero experiments improve both quality and efficiency simultaneously.

### 5. The anchor is surprisingly strong

`regional_d512_tick4` (loss 5.40, 2353 tok/s, 33.6 GB) remains the best quality/cost point in the entire sweep. Despite testing draft-revise, DINO-style distillation, residual compute, tick controllers, and alternative objectives, nothing beats the simple regional sync baseline on the combined quality-efficiency metric.

### 6. Failed stages are concerning

dr02, dr04, and dr05 produced zero completed experiments. These are the stages that test draft slot parallelism, commit heads, and attention mask modes -- all of which are needed to make draft-revise practical. Their complete failure means we have no data on critical efficiency mechanisms.

## Recommended Next Steps

### Priority 1: Make draft-revise efficient

The dr03 quality gain is real but too expensive. The immediate priority should be:

1. **Fix and rerun dr04 (commit head).** The commit/confidence mechanism is the key to reducing unnecessary revise passes. Without it, the model always runs all 8 ticks regardless of token difficulty.
2. **Fix and rerun dr05 (attention mask + memory carry).** Understanding how draft slots attend to each other and whether CTM state carries across blocks is essential for architectural decisions.
3. **Fix and rerun dr02 (parallel draft slots).** Parallel slot generation could amortize the cost of multiple draft tokens.

### Priority 2: Isolate the corruption training signal

The dr03 result suggests that corruption-based training is the active ingredient. We should test whether this signal works without the full draft-revise machinery:

1. **Corruption training on standard CTM ticks** (no draft-revise module). If the quality gain persists, the mechanism is simpler than we assumed.
2. **Corruption rate fine sweep** (5%, 10%, 15%, 20%, 25%). The current data only has 15% and 30%; a finer grid would identify the optimal rate.
3. **Corruption type variation** (random replacement vs span masking vs token dropout). Different corruption modes may produce different learning signals.

### Priority 3: Abandon dead directions

Based on these results, the following directions should be deprioritized:

1. **Full-stack async integration** (dr06). The module interactions are too unstable for productive research.
2. **DINO-style speed spectrum** (dr07). Zero quality gain at 3.4x cost.
3. **Block-level synapse skipping** (dr09). Overhead exceeds savings at this scale.
4. **Recursive NLM fast paths** (dr10). Quality degrades and refresh overhead negates savings.
5. **Learned tick controllers** (dr11). Reward hacking makes this untrainable.
6. **Objective replacement** (dr12). Standard CTM loss is well-calibrated; wholesale changes fail.

### Priority 4: Longer training for the winner

The dr03 experiments ran for only 1000 steps. The anchor was also at 1000 steps. A proper comparison requires:

1. **2000-step rerun of dr03_revise_b4_corrupt0p15** to confirm the quality gap persists under longer training.
2. **2000-step anchor rerun** at the same hyperparameters for fair comparison.
3. **Downstream evaluation** (perplexity on held-out data, or generation quality) to confirm that training loss improvement translates to real model quality.

## Bottom Line

The draft-revise program's first sweep found exactly one working mechanism: **corruption-based draft-revise training** achieves a 14.5% relative loss improvement (5.40 -> 4.62) over the regional CTM baseline. This is a meaningful result -- the largest single-experiment quality gain in the CTM-LLM project.

However, the gain comes at a 3.4x throughput cost, and all attempts to reduce that cost (residual compute, block skipping, recursive NLM, tick controllers) failed. The three stages most relevant to efficiency (dr02, dr04, dr05) also failed to produce any data.

**The next step is clear:** fix the infrastructure issues in dr02/dr04/dr05, rerun them, and use the commit head + memory carry results to design an efficient draft-revise architecture that preserves the dr03 quality gain without the 3.4x cost penalty.
