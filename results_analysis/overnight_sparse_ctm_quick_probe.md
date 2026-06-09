# Overnight Sparse CTM Quick Probe

Source report: `overnight_sparse_ctm_quick_probe_report.csv`.

Experiment script: `scripts/experiment_plan_overnight_sparse_ctm.py`.

## Experiment Count

This file is a quick-probe report, not a final loss summary. It records short batch/memory probes for overnight sparse CTM experiments.

- 386 probe attempts.
- 225 unique base experiments.
- 210 base experiments had at least one `ok` probe.
- 15 base experiments were unresolved after retries.
- Memory limit was `69632 MB`, approximately 68 GB, corresponding to 90% of an 80 GB GPU.

Status by probe attempt:

| Status | Count | Meaning |
| --- | ---: | --- |
| `ok` | 235 | ran under the memory limit |
| `oom` | 130 | CUDA out of memory |
| `over_memory` | 21 | completed but exceeded the target memory limit |

Best runnable batch size by base experiment:

| Batch Size | Base Experiments |
| ---: | ---: |
| 16 | 145 |
| 14 | 3 |
| 12 | 14 |
| 10 | 14 |
| 8 | 5 |
| 6 | 25 |
| 4 | 4 |

## What Was Tested

This overnight plan is a large sparse-compute CTM feasibility matrix. It uses the implemented knobs from `experiment_plan_impl_validation.py`, then extends them into overnight-scale candidates.

The script groups experiments by stage:

| Stage | Base Experiments | Runnable | Topic |
| --- | ---: | ---: | --- |
| og00 | 6 | 6 | dense/top-k/regional anchors |
| og01 | 12 | 12 | capacity-gradient proxy: expert granularity and active width |
| og02 | 24 | 24 | variable active compute: shared/top-k/pass count |
| og03 | 30 | 30 | dynamic tick halt and tick-budget feasibility |
| og04 | 16 | 16 | router regularization and single-pass routing controls |
| og05 | 12 | 12 | dispatch/capacity/drop mode probes |
| og06 | 20 | 20 | ELF/MTP multi-horizon supervision probes |
| og07 | 10 | 10 | long confirmation candidates |
| og08 | 40 | 34 | long-tick delta/cache/sticky-routing proxies |
| og09 | 34 | 26 | fast/slow, memory-timescale, anytime, and recruitment proxies |
| og10 | 21 | 20 | differentiated cells and real fast/slow output losses |

The key point is that this report answers "can this configuration fit and at what rough throughput?", not "does this configuration learn well?". There is no final loss column in this CSV.

## Result

Most early-stage candidates are runnable at `batch_size=16`. Stages `og00` through `og07` all found at least one successful probe for every base experiment. The failures begin when the plan pushes long internal time, long memory windows, and fast/slow auxiliary heads.

Highest-throughput runnable probes:

| Experiment | Batch | Tok/s | Peak Mem GB | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `og09_fastpath_reflex_d256_tick1_mem4` | 16 | 13968 | 14.5 | very cheap reflex-path proxy |
| `og09_anytime_d512_early_tick1` | 16 | 9255 | 14.1 | early-output path is cheap |
| `og09_fastpath_reflex_d512_k128_tick1_mem4` | 16 | 9019 | 19.2 | sparse d512 tick1 is practical |
| `og09_anytime_d1024_early_tick1` | 16 | 5994 | 18.3 | d1024 early tick still fits well |
| `og09_fastpath_reflex_d512_k256_tick2_mem4` | 16 | 5263 | 29.8 | fast tick2 path remains practical |
| `og00_anchor_dense_d512_tick2` | 16 | 5164 | 30.3 | dense d512/tick2 anchor is healthy |
| `og02_active_d512_sh0_top1_p2` | 16 | 4080 | 20.5 | low-active regional compute is cheap |

The best capacity/active-compute probes show that small active regional paths are feasible:

- `og02_active_d512_sh0_top1_p2`: `4080 tok/s`, `20.5 GB`;
- `og02_active_d1024_sh0_top1_p2`: `2881 tok/s`, `28.5 GB`;
- `og01_capacity_d512_e32_s16_p4_sh1_top1`: `1732 tok/s`, `23.1 GB`;
- `og02_active_d512_sh1_top1_p3`: `2202 tok/s`, `24.3 GB`.

Tick halt is expensive in this quick probe. `og03` is fully runnable after retries, but average throughput is only about `183 tok/s`. The d512 tick4 halt variants fit at `batch_size=16` around `36.5 GB`, but tick6/tick8 and d1024 runs require smaller batches and are much slower.

The long-tick proxy stage is the main memory bottleneck:

- `og08` has 40 base experiments, 34 runnable, 6 unresolved;
- many `tick12`/`tick16` d512 and d1024 variants OOM at `batch_size=16`;
- several only become runnable at `batch_size=10`, `8`, `6`, or `4`;
- average selected throughput is only about `326 tok/s`.

The fast/slow and differentiated-cell stage splits into two very different groups:

- `og10_diffcell_*` is promising for feasibility: d512 p3/p4 runs at `batch_size=16` with `20.8` to `23.3 GB`; d1024/d1536 variants also fit at `batch_size=16`.
- `og10_fastslow_*` is expensive: many need `batch_size=12` or `6`, throughput is often below `100 tok/s`, and `og10_fastslow_d1024_anytime_slow_compile` remains unresolved.

## What This Means

This quick probe says the overnight plan should not be launched blindly as one uniform matrix.

Feasible first-tier directions:

- fast/reflex paths from `og09`;
- d512/d1024 low-active compute from `og02`;
- differentiated-cell experiments from `og10_diffcell_*`;
- d512/d1024 MTP/ELF probes from `og06`;
- dispatch/capacity probes from `og05`, at least as feasibility checks.

Risky or expensive directions:

- long tick count with p3/p4 regional passes in `og08`;
- very slow memory-timescale and recruitment proxies in `og09`;
- fast/slow anytime output heads in `og10_fastslow_*`;
- d1024 tick8 and long-horizon halt/MTP combinations.

The key technical implication is that memory still scales sharply with tick count, memory length, pass count, and auxiliary output heads. Sparse active width alone is not enough to make long internal time cheap. If the goal is overnight throughput, low-active p1/p2 and short tick settings are the practical branch; if the goal is testing long thought, batch size must be reduced and the run budget should be much smaller.

## Recommended Next Experiments

1. Build a batch profile from this report and do not run the full plan at a fixed `batch_size=16`.
   Use the highest successful batch per base experiment, and cap the risky families manually.

2. Launch a first overnight subset from the stable high-throughput groups:
   - `og09_fastpath_reflex_*`;
   - `og09_anytime_*_early_tick1`;
   - `og02_active_d512_sh0_top1_p2`;
   - `og02_active_d1024_sh0_top1_p2`;
   - `og10_diffcell_d512_p3_sh1_top1`;
   - `og10_diffcell_d1024_p3_sh1_top1`;
   - selected `og06_mtp_*` d512/d1024 controls.

3. Split long-tick experiments into a separate low-batch plan.
   For `og08`, use `batch_size=4` to `10`, shorter `max_steps`, and only p1/p2 first. Do not mix them with normal overnight candidates.

4. Treat fast/slow output heads as implementation-cost experiments before architecture experiments.
   Run `og10_fastslow_*` only after profiling which auxiliary heads and tick outputs dominate memory.

5. Add a quick-probe summary artifact in future runs.
   The useful output is per base experiment: best runnable batch, peak memory, throughput, and failure reason. This avoids interpreting repeated attempts as independent experiments.

6. Before final training, combine this feasibility result with the previous `impl_validation_73` conclusion.
   Configurations using true grouped sparse dispatch still need backend parity checks; quick-probe success only proves they fit in memory, not that the loss path is correct.

## Bottom Line

`overnight_sparse_ctm_quick_probe_report.csv` is a feasibility screen. It shows that most standard sparse CTM overnight candidates can run, especially fast/reflex, low-active regional, MTP/ELF, and differentiated-cell probes. The long-tick, long-memory, recruitment, and fast/slow output families are memory- and throughput-heavy and should be isolated into smaller low-batch experiments.
