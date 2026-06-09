# Overnight Sparse CTM Summary (Partial Run)

Source summary: `csv_data/overnight_sparse_ctm_summary.csv` (33 rows exported from `runs/metrics`).

Experiment script: `scripts/experiment_plan_overnight_sparse_ctm.py`.

Plan size: 225 formal experiments (`og00`–`og10`). This analysis covers **only the 33 runs that produced a final metrics CSV**. It does **not** represent the full overnight matrix.

## Coverage And Caveats

| Metric | Value |
| --- | ---: |
| Planned formal runs | 225 |
| Runs with final `og*.csv` in summary | 33 |
| Runs with finite `loss` | 26 |
| Runs with `loss=NaN` but completed steps | 7 |
| Runs that OOM'd with no csv | 192 |
| Failure cause (192 rows) | 189 OOM, 3 RuntimeError |

Source failure export: `runs/metrics/overnight_sparse_fail_summary.csv` on the compute machine.

### Success vs failure by stage

| Stage | Planned | Success csv | Failed | Success rate | What failed |
| --- | ---: | ---: | ---: | ---: | --- |
| og00 | 6 | 6 | 0 | 100% | — |
| og01 | 12 | 12 | 0 | 100% | — |
| og02 | 24 | 2 | 22 | 8% | d1024/d1536 active-compute sweeps |
| og03 | 30 | 4 | 26 | 13% | mostly d1024 halt grids; d512 partial |
| og04 | 16 | 0 | 16 | 0% | entire router-reg sweep |
| og05 | 12 | 3 | 9 | 25% | d1024 dispatch + cap variants |
| og06 | 20 | 0 | 20 | 0% | entire ELF/MTP grid |
| og07 | 10 | 0 | 10 | 0% | all long confirms (6k–9k steps) |
| og08 | 40 | 1 | 39 | 3% | long-tick delta/halt/MTP proxies |
| og09 | 34 | 5 | 29 | 15% | memory tiers, anytime, recruitment |
| og10 | 21 | 0 | 21 | 0% | fast/slow compile family |
| **Total** | **225** | **33** | **192** | **15%** | |

**Pattern:** only **og00 + og01 anchors** fully survived. Every later stage is mostly OOM on **2-GPU lanes** with default batch sizes tuned for larger memory. This is an **infrastructure / batch mismatch**, not evidence that og04–og10 ideas are bad.

**How to read this file:** the pool finished many jobs, but ~85% OOM'd on 2-GPU lanes before writing usable metrics. The 33 survivors are heavily **selection-biased** toward smaller batches, smaller active width, dense/top-k paths, and configs that fit in ~20–65 GB peak memory. Missing stages (`og04`, `og06`, `og07`, `og10`, most of `og02`/`og05`/`og08`) cannot be interpreted as negative results—they simply did not produce metrics.

Additional interpretive filters:

- **`global_step` matters.** Several rows stopped far below their target (`og03` halt runs at 520–1600 steps; `og08` at 320; `og09_recruit` at 140). High loss on those rows often reflects **early exit / under-training**, not a stable architecture ranking.
- **`loss=NaN` at full steps** (`og01` d1536/d2048, `og01` e32_s16 top1, `og02` sh0, `og05` cap075) indicates **numerical / training instability**, not a usable quality measurement.
- **`tick_halt_mode=threshold`** rows with `tick_count=1` and `effective_tick=1.0` mean halt selected the first tick almost always; compare them to no-halt anchors, not to each other in isolation.
- Compare against prior **`regional_moe` best ~4.84** (`rg11_confirm_d1024_p4_shared1_top1`) only as a **reference**, not an apples-to-apples benchmark: these overnight survivors often used different step budgets, 2-GPU lanes, and batch sizes.

## Experiment Count In This Summary

| Stage | In summary | Valid loss | Topic |
| --- | ---: | ---: | --- |
| og00 | 6 | 6 | Anchors: dense, post-activation top-k, regional MoE, halt+MTP composite |
| og01 | 12 | 7 | Capacity gradient: expert count × expert size at fixed d_model |
| og02 | 2 | 1 | Variable active compute: shared/top-k/pass sweeps |
| og03 | 4 | 4 | Dynamic tick halting threshold × tick budget |
| og04 | 0 | 0 | Router regularization (no surviving metrics) |
| og05 | 3 | 2 | Dispatch mode: block / dropless / capacity drop |
| og06 | 0 | 0 | ELF / MTP multi-horizon sweep (no surviving metrics) |
| og07 | 0 | 0 | Long confirmation runs (no surviving metrics) |
| og08 | 1 | 1 | Keyframe-delta proxy: many ticks, low active width |
| og09 | 5 | 5 | Fast–slow proxy: reflex path, memory scale, anytime, recruitment |
| og10 | 0 | 0 | Fast/slow output compile (no surviving metrics) |

## What These Experiments Were Testing

The overnight plan groups runnable proxies for sparse-compute research ideas already in the training stack:

- **og00 — Anchors.** Fair reference points before sweeping harder sparse knobs: dense CTM, post-activation top-k sparsity, regional multi-pass MoE (`p4/shared1/top1`), and a composite anchor with tick halt + MTP.
- **og01 — Capacity gradient.** At fixed `d_model`, vary expert granularity (`e8×128`, `e16×64`, `e32×16`, etc.), activation passes, shared experts, and routed top-k. Proxy for “heterogeneous / nested capacity” without new cell types.
- **og02 — Variable active compute.** At fixed d512/d1024/d1536, sweep how many experts/regions are active per tick (`shared`, `topk`, `activation_passes`).
- **og03 — Dynamic ticks.** Compare no-halt vs threshold halt (0.15–0.60) at tick budgets 4/6/8; tests whether CTM can stop thinking early.
- **og05 — Dispatch capacity.** Regional routing with `block_sparse`, `dropless`, and `capacity_drop` at factors 0.75–1.25; proxy for token-budgeted sparse execution.
- **og08 — Delta-time proxy.** Many ticks (8–16) with restricted active cells per tick; simulates “sticky / keyframe” sparse recurrence before true delta-cache code exists.
- **og09 — Fast–slow / developmental proxy.** Reflex-like short-memory sparse paths, regional memory-length tiers, anytime early-tick supervision, and high-compute “recruitment” with MTP + improve loss.

## Top Results (Finite Loss Only)

### Best language-model quality (`loss`, lower is better)

| Experiment | Loss | Tok/s | Peak GB | Active frac | Steps | Key setting |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `og00_anchor_topk_d1024_k256` | **3.575** | 2646 | 49.6 | 0.250 | 6000 | post-activation top-k, k=256 |
| `og01_capacity_d1024_e8_s128_p4_sh1_top1` | **3.846** | 676 | 50.9 | 0.250 | 3500 | 8 experts × 128-d, top1 |
| `og01_capacity_d1024_e8_s128_p4_sh2_top1` | **3.957** | 658 | 63.4 | 0.375 | 3500 | same, shared2 |
| `og09_fastpath_reflex_d512_k128_tick1_mem4` | **4.016** | **8684** | 19.7 | 0.250 | 3500 | reflex: tick1, mem4, topk128 |
| `og00_anchor_topk_d1024_k128` | 4.171 | 2066 | 49.6 | 0.125 | 6000 | post-activation top-k, k=128 |
| `og00_anchor_dense_d512_tick2` | 4.695 | **5449** | 30.3 | 1.000 | 2500 | dense d512 anchor |
| `og00_anchor_dense_d1024_tick2` | 4.751 | 3033 | 48.6 | 1.000 | 2500 | dense d1024 anchor |
| `og09_anytime_d1024_early_tick1` | 5.227 | 1635 | 30.6 | 0.063 | 3500 | regional, 1 tick, early output |

### Best quality/cost (`quality_cost_score = loss × peak_GB / tok/s`, lower is better)

| Experiment | QCS | Loss | Tok/s | Peak GB |
| --- | ---: | ---: | ---: | ---: |
| `og09_fastpath_reflex_d512_k128_tick1_mem4` | **0.0089** | 4.016 | 8684 | 19.7 |
| `og00_anchor_dense_d512_tick2` | 0.0261 | 4.695 | 5449 | 30.3 |
| `og09_fastpath_reflex_d512_k256_tick2_mem4` | 0.0369 | 4.726 | 3815 | 30.5 |
| `og00_anchor_topk_d1024_k256` | 0.0671 | 3.575 | 2646 | 49.6 |
| `og00_anchor_dense_d1024_tick2` | 0.0762 | 4.751 | 3033 | 48.6 |

### Worst finite-loss survivors (likely broken or under-trained)

| Experiment | Loss | Steps | Likely reason |
| --- | ---: | ---: | --- |
| `og02_active_d512_sh1_top1_p3` | 548.28 | 3000 | unstable regional active-compute point |
| `og05_dispatch_d512_cap125` | 333.94 | 1220 | capacity_drop + early stop / bad dispatch |
| `og01_capacity_d512_e32_s16_p4_sh1_top2` | 324.08 | 2560 | extreme granularity + top2 instability |
| `og03_halt_d512_thr0p3_tick6` | 117.88 | 740 | halt exited early, under-trained |
| `og01_capacity_d1024_e16_s64_p4_sh1_top1` | 38.25 | 3500 | top1 regional worse than top2 sibling |

## Stage-By-Stage Interpretation

### og00 — Anchors

Within the surviving subset, **post-activation top-k at d1024 is the quality leader**:

- `og00_anchor_topk_d1024_k256`: loss **3.575**, active fraction **0.25**, trained **6000 steps**.
- Beats dense d1024 anchor (**4.751**) and dense d512 (**4.695**) on loss.
- Beats the regional anchor `og00_anchor_regional_d1024_p4_shared1_top1` (**7.837**, only **561 tok/s**) by a wide margin in this partial pool.

Tick behavior on dense anchors is healthy: tick-2 losses decrease slightly (`4.924 → 4.747` on d1024; `4.486 → 4.477` on d512), so extra ticks are not hurting, but the gain is small at tick=2.

The composite anchor `og00_anchor_regional_d512_p4_halt0p30_mtp124` is **not ready**:

- loss **16.0**, only **2260 steps**, `tick_count=1`, `effective_tick=1.0`.
- Halt + MTP together collapsed to “always stop at tick 1” with poor LM loss. Treat as a **supervision/halt interaction bug or mis-tuned combo**, not evidence against MTP or regional MoE individually.

**Inference:** among configs that survived OOM, **classical post-activation sparsity + longer training** beat regional MoE anchors in this run. That contradicts the earlier `regional_moe` sweep—but this overnight slice is not a fair replay of `rg11` (different batch, GPU count, step budget, and selection bias).

### og01 — Capacity Gradient

Clear pattern among finite-loss rows:

1. **Fewer, wider experts help at d1024:** `e8_s128` (loss ~3.85–3.96) beats `e16_s64 top1` (38.25) and matches the global best tier.
2. **`top2` can rescue bad `top1` regional points:** same `e16_s64 p4 sh1` grid: top1 **38.25** vs top2 **4.636**.
3. **Extreme fragmentation hurts:** `e32_s16 top2` → loss **324**; `e512_e16_s32 top1` → **22.7**.
4. **Large d_model points mostly NaN:** d1536/d2048 rows completed steps but emitted NaN loss → unstable, not merely slow.

**Inference:** the capacity-gradient proxy says **coarser expert blocks + moderate active width** are easier to train under memory pressure; **very fine expert grids need top2 or wider active cells**, not strict top1.

### og02 — Variable Active Compute

Only two csv survivors; one NaN, one catastrophic:

- `og02_active_d512_sh1_top1_p3`: loss **548**, full 3000 steps.

**Inference:** no positive signal from og02 in this export. The stage needs re-run on full nodes with reduced batch—not interpretable yet.

### og03 — Dynamic Tick Halting

All four survivors use `tick_halt_mode=threshold` and show **`tick_count=1`, `effective_tick=1.0`**:

| Experiment | Loss | Steps | Threshold | Configured ticks |
| --- | ---: | ---: | ---: | ---: |
| `og03_halt_d512_thr0p15_tick4` | 23.38 | 1600 | 0.15 | 4 |
| `og03_halt_d512_thr0p3_tick4` | — | — | 0.30 | 4 |
| `og03_halt_d512_thr0p6_tick4` | 42.99 | 1040 | 0.60 | 4 |
| `og03_halt_d512_thr0p3_tick6` | 117.88 | 740 | 0.30 | 6 |
| `og03_halt_d512_thr0p3_tick8` | 31.61 | 520 | 0.30 | 8 |

Losses are high because training **stopped early** (low `global_step`) while halt metrics show the model never used later ticks.

**Inference:** in this partial run, threshold halt acted as an **early-stop regularizer**, not a useful “think longer when needed” mechanism. Re-test only after fixing batch/memory and comparing against matched-step no-halt controls.

### og05 — Dispatch Mode

| Experiment | Loss | Steps | Dispatch |
| --- | ---: | ---: | --- |
| `og05_dispatch_d512_dropless` | **6.942** | 3000 | dropless |
| `og05_dispatch_d512_cap125` | 333.94 | 1220 | capacity_drop ×1.25 |
| `og05_dispatch_d512_cap075` | NaN | 2620 | capacity_drop ×0.75 |

**Inference:** consistent with `impl_validation_73`: **`dropless` can work; aggressive `capacity_drop` is risky**. Only one positive dropless point here—not enough to adopt fleet-wide, but enough to prioritize dropless over capacity_drop in the next full-node rerun.

### og08 — Delta-Time Proxy

Single survivor: `og08_delta_time_d512_tick8_p4_sh1_top1`

- loss **33.48**, **320 steps** only, 8 configured ticks.
- Per-tick losses rise monotonically (`24 → 74` across ticks); long internal time without stable sparse delta machinery looks **harmful under budget**.

**Inference:** the proxy does not yet show “many cheap ticks beat few full ticks.” Needs full-step rerun before architectural conclusions.

### og09 — Fast–Slow / Developmental Proxy

Strongest **efficiency** story in the export:

- **`og09_fastpath_reflex_d512_k128_tick1_mem4`:** loss **4.016**, **8684 tok/s**, **19.7 GB**, QCS **0.0089**.
- **`og09_fastpath_reflex_d512_k256_tick2_mem4`:** loss **4.726**, 3815 tok/s—slightly worse quality, still excellent throughput.
- **`og09_anytime_d1024_early_tick1`:** loss **5.227**, reasonable quality with only **1 tick** and **6.25% active cells**.

Weaker points:

- `og09_memory_d512_reflex_tick2_mem4`: loss **5.43**, only **1280 steps** (regional reflex path under-trained).
- `og09_recruit_counterfactual_d512`: loss **9.47**, **140 steps**; per-tick losses improve slowly (`7.49 → 6.87`) but run aborted early—**promising curve, no final verdict**.

**Inference:** the **reflex / early-tick / short-memory** proxy is the most actionable overnight finding: near-anchor quality at much lower memory and much higher throughput. Recruitment and memory-tier regional paths need longer runs.

## Cross-Cutting Conclusions (Survivor Subset Only)

1. **Best absolute LM loss in this file:** `og00_anchor_topk_d1024_k256` (**3.575**). It suggests post-activation top-k + 6000-step budget is a serious quality candidate— but only among OOM-filtered configs.
2. **Best practical training candidate:** `og09_fastpath_reflex_d512_k128_tick1_mem4` — loss **4.016** with **~8.7k tok/s** and **~20 GB** peak. Best QCS by a wide margin.
3. **Dense d512 anchor remains strong on throughput:** loss **4.695** at **5449 tok/s**; still the simplest high-throughput baseline in this slice.
4. **Regional MoE anchor underperformed here** (`og00` regional d1024 loss **7.837**). Do **not** overturn `regional_moe` conclusions from this biased subset; rerun `og00`/`og07` anchors on **full 8-GPU nodes** with matched steps.
5. **top-k width matters:** k256 beats k128 on d1024 top-k anchors (3.575 vs 4.171) with 2× active fraction—quality gain costs memory and some throughput.
6. **top2 routing fixes some bad top1 regional points** (`og01` e16_s64: 38 → 4.6). Router width is an under-tested axis in the surviving data.
7. **Halt + MTP composite and threshold-halt sweeps are not production-ready** from this export; losses and step counts indicate supervision/coverage problems, not finished comparisons.
8. **Seven NaN rows at full steps** flag numerical instability for large/granular configs—exclude from ranking, investigate loss spikes / router collapse offline.

## How To Read The Metrics Columns

- **`loss`:** training CE at logged step (lower is better). Final row in each `runs/metrics/{name}.csv` is what summarize exports.
- **`tokens_per_sec`:** throughput during the logging window; useful for cost, not quality alone.
- **`peak_memory_mb`:** max CUDA memory on the logging rank; with 2-GPU DDP this reflects **per-process** usage, not full-node footprint.
- **`active_cell_fraction`:** fraction of d_model cells active after sparsity/MoE masking; lower means sparser compute.
- **`best_tick` / `conf_tick`:** argmin tick loss / argmin entropy tick on the batch; `-1` when halt logic does not apply.
- **`effective_tick`:** halt-weighted expected tick when `tick_halt_mode != none`; `1.0` means “always behave like tick 1.”
- **`quality_cost_score`:** `loss × peak_GB / tok/s`; lower merges quality, memory, and speed—useful for picking **training** candidates, not deployment quality alone.
- **`losses_per_tick`:** JSON list; check monotonicity to see whether later ticks refine predictions.

## What This File Does **Not** Support

- Ranking **`og04` router regularization**, **`og06` ELF/MTP grid**, **`og07` long confirms**, or **`og10` fast/slow compile**—no metrics survived.
- Claiming **regional MoE is worse than top-k sparsity** globally—the regional winners simply OOM'd or were not re-run at fair batch/node budget.
- Treating **`og03` halt losses** as fair quality comparisons without matching `global_step` to no-halt baselines.
- Using **`og02` / large `og01` NaN rows** as evidence about d1536/d2048 capacity—those runs are instability signals, not finished experiments.

## Recommended Next Steps

### Rerun priority (given 189/192 OOM)

| Priority | Stage | Why |
| --- | --- | --- |
| **P0 — do not rerun blindly** | og00, og01 | Already complete; use as anchors only |
| **P1 — full 8-GPU node** | og07 | Long confirms; 0% success, highest scientific value per run |
| **P1 — full node + lower batch** | og06 | ELF/MTP grid; 0% success, directly tests parallel supervision |
| **P2 — full node or tail scheduling** | og04, og05, og10 | 0–25% success; moderate run count (16–21 each) |
| **P2 — lane batch re-probe first** | og08, og09 | 39 and 29 failures; heaviest stages but some survivors show signal |
| **P3 — after og01 lessons** | og02, og03 | Apply `top2` / smaller batch from og01; og03 needs matched-step halt controls |

Operational checklist for the 192 failures:

1. Run **`probe-and-run` or `quick-probe` on 2-GPU lanes** for og02+ only, or switch formal reruns to **`--gpus_per_lane 8` / `--tail_full_nodes`** so each job gets a full node.
2. Halve batch for **regional + long tick + MTP** families before retrying og06/og08/og09/og10.
3. Run **og07 separately** on full 4-node pool with the batch profile from og00 survivors—not from qp probes on 2-GPU lanes.

### Analysis actions (from the 33 survivors)

1. Treat **`og00_anchor_topk_d1024_k256`**, **`og01_capacity_d1024_e8_s128_*`**, and **`og09_fastpath_reflex_d512_k128_*`** as the three pillars (quality, capacity structure, efficiency).
2. **Re-test regional anchors** (`og00_regional_*`, planned `og07_confirm_*`) at **8 GPU / fair batch** before merging with `regional_moe` conclusions.
3. **Decouple halt and MTP** (`og00` composite failure) and re-run `og03` with fixed step budget for fair halt comparisons.
4. **Extend `og09_recruit_counterfactual`** to full steps—it improved across ticks but stopped at 140 steps.

## Bottom Line

This CSV is a **biased survivor sample** (33/225), not the completed overnight study. Within that sample, **post-activation top-k at d1024** achieves the best loss, and the **reflex fast-path (`og09_fastpath`)** achieves the best quality/cost. Regional MoE, halt, MTP composites, and large-scale capacity grids **cannot be judged from the missing 192 runs**; they require a full-node rerun before changing the project’s base architecture.
