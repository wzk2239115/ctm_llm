# Why Each Draft-Revise Idea Failed: Root Cause Taxonomy

Every failed experiment from the draft-revise sweep is classified into one of four
categories:

1. **Implementation bug** -- the code does not correctly implement the intended
   mechanism.
2. **Mechanically correct but CTM-ignorant** -- the code works as written, but the
   design ignores how CTM's tick/state/trace architecture actually operates.
3. **Right idea, wrong angle** -- the concept is sound for CTM but needs to be
   adapted to the framework before it can work.
4. **Fundamentally incompatible with CTM** -- the idea's assumptions contradict CTM
  's inductive biases at a structural level.

Source code references use the file paths in `model/`.

---

## dr00 Async Anytime (loss 9.28) -- Category 4: Incompatible

**What was tested:** `async_tick_mode=banded` with 16 ticks, threshold halt,
fast/slow/habit output weights.

**Root cause:** The CTM tick loop is a *recurrent state machine* -- each tick
reads and updates `activated` and `state_trace` in-place. The async banded mode
(`model_ctm_async.py`) tries to run different layers on different clocks, so layer
N at tick T may use stale state from layer N-1 at tick T-k. This breaks the
fundamental assumption that `cross_layer_state` passes a temporally consistent
activated/trace pair between layers.

Specifically, in the sync path (`model_ctm_llm.py:740-774`), the loop is:
```
for layer, past_kv in zip(self.layers, past_kv):
    result = layer(h, prev_activated=prev_activated, prev_trace=prev_trace, ...)
    prev_activated = extras['final_activated']
    prev_trace = extras['final_trace']
```
Every layer sees the same tick count and the activated/trace state is fresh from
the *immediately preceding* layer. The async variant breaks this: slow-band layers
run fewer ticks, producing stale `prev_activated`/`prev_trace` that the
fast-band layer at the same logical "time step" never actually computed. The
per-tick loss oscillates (`[6.95, 6.97, 6.96, ...]`) because the recurrent
state never converges -- it is constantly perturbed by stale cross-layer inputs.

**Why it cannot be fixed easily:** Any asynchronous clock design for CTM must
solve the cross-layer temporal consistency problem. Either (a) add a
synchronization barrier between layers that defeats the purpose of async, or (b)
design a completely new state-passing protocol that can tolerate temporal drift.
Neither is a small fix.

---

## dr01 MTP / ELF (loss 5.39-5.58, zero gain) -- Category 2: CTM-ignorant

**What was tested:** MTP horizons 1,2,4 and ELF linear horizon 4, both on 4-tick
regional CTM.

**Root cause:** MTP and ELF are designed for *autoregressive transformers* where
each forward pass produces exactly one next-token distribution. The idea is that
predicting k tokens ahead provides a richer gradient signal. But CTM already runs
multiple ticks -- the model *already* produces multiple distributions per input
position, one per tick, with the `min_conf` loss selecting the best one.

Looking at the loss computation in `model_ctm_llm.py:1068-1112`:
```python
for t in range(num_ticks):
    logits_t = self.lm_head(tick_outs[..., t])
    next_loss, next_mask = self._per_sample_lm_loss(logits_t, labels, horizon=1)
    tick_components.append(next_loss)
    # MTP horizons add shifted-horizon losses here
    # ELF horizons add longer-horizon losses here
```

The tick loop already gives the model multiple "attempts" at next-token
prediction. Adding MTP on top means predicting tokens at horizon h from *each* tick,
which is redundant -- the model already gets gradient signal from multiple ticks.
The CTM's internal recurrence subsumes the benefit that MTP provides to static
transformers.

**Why ELF h8 is worse:** At 8 ticks with `tick_improve_weight=0.05`, the model
receives conflicting supervision: `min_conf` selects the lowest-loss tick, while
`tick_improve` rewards monotonically decreasing loss. These two objectives pull
in opposite directions. The model settles on a compromise that is worse than either
alone.

**Verdict:** MTP/ELF are not harmful, but they are redundant given CTM's multi-tick
structure. The multi-tick recurrence is CTM's native version of MTP.

---

## dr06 Full Sparse Async Draft-Revise Stack (loss 8.7-9.0) -- Category 4:
Incompatible (composition)

**What was tested:** async banded + sparse regional + draft-revise combined.

**Root cause:** This is dr00's async failure multiplied by two additional sources
of instability. The code paths:

1. **Async cross-layer drift** (same as dr00).
2. **Draft slot interaction with sparse routing:** The `DraftSlotHead`
(`draft_modules.py`) sits *after* `output_proj` at line 644:
   ```python
   if all_draft_slot_logits is not None:
       _, slot_logits = self.draft_slot_head(tick_out, lm_head=draft_lm_head)
   ```
   The slot head receives the *regional masked* output, which has 12.5% active
   cells. The slot attention (`_slot_attention`) computes self-attention over
   block_size=4 slots using this heavily masked representation. When the mask
   changes between async bands (because different layers route to different
   experts at different ticks), the slot head receives inconsistent inputs.
3. **Draft corruption on async state:** The corruption mechanism
   (`_corrupt_draft_labels` at `model_ctm_llm.py:889-899`) corrupts labels
   randomly, but the revise loss is computed against `slot_logits` that were
   generated from asynchronously-updated state. The model cannot learn a stable
   correction pattern because the relationship between slot logits and labels is
   non-stationary.

**Verdict:** Naive module stacking on CTM is destructive. Each module assumes a
temporally consistent state, and async breaks that assumption.

---

## dr07 DINO Speed Spectrum (loss ~5.56, zero gain) -- Category 3: Right idea,
wrong angle

**What was tested:** Multi-EMA teacher distillation at different decay rates,
supervising fast/mid/slow tick representations.

**Root cause:** The DINO mechanism (`speed_spectrum.py`) maintains a full copy of
the backbone as teacher (`self.teacher_model = copy.deepcopy(backbone)` at line
82). During distillation, it runs the teacher forward to get teacher hidden states
(line 96-99):
```python
teacher_hidden = self.teacher_model(
    input_ids, track=False, num_iters=num_iters,
    return_all_ticks=False,
).hidden
```
The student's tick 0 hidden is distilled against the teacher's final hidden. This
has two problems:

1. **Tick-to-tick variation is too small:** In CTM, consecutive ticks differ only
   by one synapse+trace update. The hidden states at tick 0 and tick 7 are
   semantically very similar -- they are recurrent refinements of the same input.
   Unlike DINO in vision where augmentations create genuinely different views,
   CTM ticks are *not* meaningfully different "views" of the input. The distillation
   loss is near-trivially satisfiable and adds no useful gradient.

2. **3.4x cost overhead from teacher forward pass:** The teacher model is a full
   copy of the backbone. Running it every step doubles memory and roughly doubles
   compute. Combined with 8 student ticks, this explains the 3.4x throughput
   reduction.

**How to fix the angle:** Instead of distilling between ticks, distill between
*different models* or *different augmentation views*. For example, a small fast
CTM (tick 2) could be distilled from a large slow CTM (tick 8), which would be
more analogous to the original DINO's student/teacher size difference. Or use
input-level augmentations (token dropout, span masking) to create genuinely
different views, rather than relying on tick-index as the source of variation.

---

## dr08 Residual Semantic Caches (loss 5.58, zero gain) -- Category 2:
CTM-ignorant

**What was tested:** Delta tracking, KV caching, recursive sync, observe-only
baseline.

**Root cause:** The residual compute module (`residual_compute.py`) was designed
for *transformer* residual connections where each layer's output can be cached and
reused. In CTM, the core computation per tick is the synapse-activate-trace
pipeline:

```
pre_syn = [attn, activated] -> synapses -> state -> trace -> NLM -> activated
```
The `activated` tensor at tick T+1 is a *function of* the `activated` tensor at
tick T through the synapse. There is no "residual" in the transformer sense -- the
state is *recurrently updated*, not residually added. Caching the delta between
consecutive ticks (`tick_deltas = tick_outs[..., 1:] - tick_outs[..., :-1]` in
`compute_residual_metrics`) measures *how much the output changed*, but this
delta is not reusable -- you cannot skip a synapse forward pass by adding a cached
delta to stale state, because the synapse is nonlinear.

The `attn_refresh` variant works best (5.577) because it is closest to just
running the full forward pass -- the "refresh" means "do the real computation
periodically." But the intervening ticks still accumulate approximation error from
delta-based updates, so the quality never exceeds the baseline.

**Verdict:** The residual/delta concept is mechanically correct but assumes a
residual architecture. CTM is recurrent, not residual. Delta caching does not save
computation because the core operation (synapse forward) cannot be approximated by
linear deltas.

---

## dr09 Synapse Block Skipping (loss 5.59, 5.7x slower) -- Category 1:
Implementation bug (overhead > savings)

**What was tested:** Group the sequence into blocks, compute novelty scores, skip
blocks with low novelty.

**Root cause:** The implementation in `run_grouped_block_delta_synapse`
(`residual_compute.py:114-169`) divides the sequence T dimension into groups:
```python
num_groups = max(1, int(config.residual_num_groups))  # 16 or 32
group_size = max(1, int(math.ceil(T / float(num_groups))))
```
For T=512 and groups=32, each group is 16 tokens. For each group, it:
1. Computes novelty score: `(pre_syn - cached_pre_syn).abs().mean()` -- one
   mean reduction.
2. Ranks all 32 groups by novelty: sorted() call.
3. Selects active groups based on threshold and active_ratio.
4. For skipped groups, copies cached state; for active groups, runs full synapse.
5. Merges outputs into a pre-allocated tensor.

Steps 1-3 and 5 are pure overhead that does NOT exist in the baseline. The baseline
simply runs `self.synapses(pre_syn)` once over the full sequence. The block skip
path does all the bookkeeping *plus* a subset of synapse forwards. For groups=32
with active_ratio=0.25, you skip ~75% of synapse compute but add 32 novelty
computations, 32 cache lookups, a sort, and a scatter merge.

**Why this is Category 1 not Category 4:** The *idea* of skipping computation for
unchanged sequence positions is valid for CTM (and is essentially what the
regional MoE routing already does along the *expert* dimension). But the
implementation applies it along the *sequence* dimension with per-group novelty, which
has too much overhead at this model scale (d=512, T=512). At much larger models
(d=4096, T=4096), the novelty computation overhead would be amortized, but at
d=512 it dominates.

**Verdict:** The block skip implementation is correct but the overhead structure
is wrong for this model scale. It needs to be integrated into the expert routing
mechanism (skip entire experts, not sequence chunks) to be efficient.

---

## dr10 Recursive NLM Fast Path (loss 5.65-5.70) -- Category 2: CTM-ignorant

**What was tested:** Skip full trace-window NLM attention, carry activation forward,
periodic full refresh.

**Root cause:** The recursive NLM (`nlm_recursive.py`) tries to avoid recomputing
the full trace_processor forward pass by caching the output and adding a delta:
```python
carried = cache['activated'].to(...)
slot_delta = state_trace[..., -1] - cache['trace'][..., -1].to(...)
activated = carried + slot_delta
```
This is a first-order Taylor approximation of the NLM function. It works when:
1. The NLM is nearly linear (so first-order approximation is accurate).
2. The delta between consecutive ticks is small.

In CTM, the trace_processor is a two-layer MLP with GLU activations:
```python
SuperLinear(memory_length, 2 * memory_hidden_dims, d_model) -> GLU ->
SuperLinear(memory_hidden_dims, 2, d_model) -> GLU -> Squeeze
```
This is highly nonlinear. The GLU gating means the function is not amenable to
linear approximation. The per-tick delta in the trace is also not small -- each
tick shifts the trace window by one position and writes a completely new state
vector. The "delta" between `trace[..., -1]` at tick T and tick T+1 is the
difference between the new synapse output and whatever was in the oldest trace
slot -- which can be arbitrarily large.

The `hybrid_fast_full_refresh4` results confirm this: per-tick loss *increases*
(6.13 -> 6.41) because the approximation error accumulates faster than the
periodic full refresh can correct it.

**Verdict:** The recursive approximation is mathematically sound for nearly-linear
functions applied to slowly-changing inputs. CTM's trace NLM is neither nearly
linear nor slowly-changing. The idea would need a fundamentally different
approximation strategy (e.g., low-rank adaptation of the NLM weights rather than
linear delta propagation).

---

## dr11 Tick Controller (loss 9.43, reward hacking) -- Category 3: Right idea,
wrong angle

**What was tested:** Learned/threshold tick controller that decides full/residual/
stop per tick, with compute penalty.

**Root cause:** The controller (`tick_controller.py`) uses a confidence-based
stopping rule:
```python
if float(prev_confidence) >= stop_threshold:
    return 'stop'
```
And the compute penalty (`residual_compute_weight`) penalizes executed ticks:
```python
executed_ratio = 1.0 - 0.5 * skip_ratio - 0.5 * nlm_fast_ratio
penalty = compute_weight * executed_ratio
```
The model minimizes total loss = language_loss + compute_penalty. The easiest way
to reduce compute_penalty is to output high-confidence (low-entropy) logits early,
regardless of actual prediction quality. The certainty trend (0.83 -> 0.94) confirms
this: the model learns to be confidently wrong.

The core issue is that confidence (entropy of softmax output) is *trivially
gameable* in a language model. The model can push its logits to extreme values,
producing high-confidence but inaccurate predictions. The compute penalty creates
an incentive to do this as early as possible.

**How to fix the angle:** The adaptive compute idea is sound for CTM -- some tokens
genuinely need more thought than others. But the stopping criterion must be based
on *quality metrics*, not confidence. Options:
1. Use a separate verification head that checks prediction accuracy against a
   reference (e.g., the next-token label itself, with a proper information
   bottleneck).
2. Use an unsupervised "surprisal" signal: if the model's top-1 prediction
   probability is high *and* the entropy of the prediction distribution is low,
   the token is likely "easy." But this needs to be computed on held-out data, not
   training data, to avoid the gaming problem.
3. Use a budget-based approach: allocate a fixed compute budget per sequence,
   and let the model learn to *allocate* it (not reduce it).

---

## dr12 Objective Variants (causal_ce loss 10.3, latent_denoise loss 7.6) --
Category 2: CTM-ignorant

**What was tested:** Standard causal CE loss, and latent space denoising (flow
matching style).

**Root cause (causal_ce):** The `causal_ce` mode (`objective_elf.py`) replaces the
CTM training loop with standard causal cross-entropy:
```python
logits = lm_head(hidden)
ce = base_ce_loss_fn(logits, labels)
```
This completely ignores the tick structure. In the standard CTM path, the loss is
computed per-tick and aggregated with `min_conf`:
```python
for t in range(num_ticks):
    logits_t = self.lm_head(tick_outs[..., t])
    next_loss = self._per_sample_lm_loss(logits_t, labels, horizon=1)
```
The `min_conf` selection is critical: it picks the tick with lowest entropy,
effectively giving the model multiple attempts and selecting the best. Standard
causal CE uses only the final hidden state, throwing away this multi-attempt
advantage. Loss 10.3 (vs anchor 5.4) shows how much quality the min_conf
mechanism provides.

**Root cause (latent_denoise):** The denoising objective applies flow-matching-style
noise to the hidden states:
```python
noisy = t * latent + (1.0 - t) * noise
pred = denoise_head(noisy)
target = latent - noise
denoise_loss = F.mse_loss(pred, target)
```
This is designed for continuous latent spaces (e.g., VAE latent variables or image
patches). CTM hidden states are *discrete-token-conditioned* representations --
they encode next-token prediction information, not a smooth generative manifold.
The MSE between `pred` and `latent - noise` does not correspond to any meaningful
signal for next-token prediction. The model has no incentive to maintain the
discrete structure that makes the hidden states useful for language modeling.

**Verdict:** Both objectives are mechanically correct implementations of their
respective paradigms, but they ignore the two things that make CTM work:
(1) multi-tick min_conf selection, and (2) discrete-token-conditioned hidden
states.

---

## dr02/dr04/dr05 (all failed, no completed experiments) -- Category 1:
Implementation bugs

**What was tested:**
- dr02: Parallel draft slots with slot-aware heads.
- dr04: Commit/confidence head for safe prefix emission.
- dr05: Draft slot attention mask and CTM state carry.

**Root cause:** These stages produced zero completed experiments, meaning all runs
crashed or NaN'd before reaching `max_steps`. The most likely causes based on code
review:

1. **dr02 parallel mode:** The `DraftSlotHead` in parallel mode generates
   `block_size` slots in a single forward pass. The `slot_in` linear projects from
   `hidden_size` to `hidden_size * block_size`. With block_size=8 and
   hidden_size=512, this is a 512 -> 4096 projection. The slot attention then
   computes self-attention over 8 slots of dimension 512. Combined with 8 ticks and
   12 layers, this is 12 * 8 = 96 slot-attention passes per forward step. The
   memory footprint likely exceeded GPU capacity for the batch sizes used.

2. **dr04 commit head:** The `_draft_commit_loss` function
   (`model_ctm_llm.py:901-926`) computes binary cross-entropy between confidence
   and correctness:
   ```python
   F.binary_cross_entropy(avg_conf, avg_match, reduction='none')
   ```
   The BCE gradient can be numerically unstable when `avg_conf` approaches 0 or 1,
   which happens frequently during training. Combined with corruption training
   (corrupt_prob=0.30), the model may receive conflicting signals: labels are
   corrupted (making `avg_match` low) but the model is also trained to predict
   the *corrupted* labels (making `avg_conf` high). This conflict can cause
   gradient explosion.

3. **dr05 memory carry:** The `draft_memory_carry=1` flag is supposed to carry
   CTM state across draft blocks, but there is no code in `DraftSlotHead` or the
   CTM tick loop that actually implements this carry. The `draft_memory_carry`
   config field exists but is never read by any forward pass code. This means
   `carry=1` and `carry=0` produce identical behavior, but the experiment plan
   sets `memory_length=8` for carry=1, which changes the trace processor's input
   size and may cause shape mismatches.

**Verdict:** These are implementation bugs (OOM, numerical instability, dead code)
that need to be fixed before any architectural conclusions can be drawn.

---

## Summary Table

| Stage | Category | Key Takeaway |
| --- | --- | --- |
| dr00 async | 4: Incompatible | Async clocks break cross-layer state consistency in recurrent CTM |
| dr01 MTP/ELF | 2: CTM-ignorant | Multi-tick min_conf already subsumes MTP/ELF |
| dr02 parallel | 1: Bug | Likely OOM from large slot projections; needs profiling |
| dr03 corruption | (Works) | The only quality win; needs efficiency work |
| dr04 commit | 1: Bug | BCE numerical instability with conflicting corruption signals |
| dr05 memory carry | 1: Bug | `draft_memory_carry` is dead code; `memory_length` change may crash |
| dr06 full stack | 4: Incompatible | Compounds dr00's async failure with sparse routing instability |
| dr07 DINO | 3: Wrong angle | Tick-to-tick variation too small for meaningful distillation |
| dr08 residual | 2: CTM-ignorant | Delta caching assumes residual arch; CTM is recurrent |
| dr09 block skip | 1: Bug (overhead) | Novelty bookkeeping cost exceeds compute savings at d=512 |
| dr10 recursive NLM | 2: CTM-ignorant | Linear approximation fails on nonlinear GLU NLM |
| dr11 tick ctrl | 3: Wrong angle | Confidence-based stopping is trivially gameable |
| dr12 objectives | 2: CTM-ignorant | Ignores min_conf multi-tick advantage and discrete hidden states |

## Distribution

| Category | Count | Stages |
| --- | ---: | --- |
| 1: Implementation bug | 4 | dr02, dr04, dr05, dr09 |
| 2: CTM-ignorant | 5 | dr01, dr08, dr10, dr12 (causal+denoise) |
| 3: Right idea, wrong angle | 2 | dr07, dr11 |
| 4: Incompatible | 2 | dr00, dr06 |

## Meta-Lesson

The most common failure mode (category 2: CTM-ignorant, 5 of 13 failures) is
applying transformer-era optimizations to CTM without accounting for CTM's
recurrent state machine architecture. Specifically:

- **Residual/delta caching** assumes additive skip connections (transformer). CTM
  uses recurrent state updates through nonlinear synapses.
- **MTP/ELF** assumes one forward pass = one prediction (transformer). CTM already
  produces multiple predictions per input through ticks.
- **Standard CE** assumes a single hidden state for loss computation (transformer).
  CTM uses multi-tick min_conf selection.
- **Recursive linear approximation** assumes smooth, slowly-changing functions.
  CTM's trace NLM has GLU nonlinearities and large per-tick state shifts.

The second most common failure (category 1: bug, 4 of 13) reflects the
engineering cost of a rapidly-growing codebase. The draft-revise, residual compute,
and tick controller modules were all added in parallel without sufficient unit
testing, leading to OOM, numerical instability, and dead code paths.

Category 4 (incompatible, 2 of 13) identifies async clocks and full-stack
composition as structurally incompatible with CTM's current architecture. These
would require fundamental redesign, not incremental fixes.
