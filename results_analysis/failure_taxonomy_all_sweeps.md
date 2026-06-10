# Why Each Idea Failed: Root Cause Taxonomy Across All CTM-LLM Sweeps

Every failed experiment idea is classified into one of four categories:

1. **Implementation bug** -- the code does not correctly implement the intended
   mechanism.
2. **Mechanically correct but CTM-ignorant** -- the code works as written, but the
   design ignores how CTM's tick/state/trace architecture actually operates.
3. **Right idea, wrong angle** -- the concept is sound for CTM but needs to be
   adapted to the framework before it can work.
4. **Fundamentally incompatible with CTM** -- the idea's assumptions contradict CTM
   's inductive biases at a structural level.

---

## Part 1: More Ticks = More Reasoning (s02 tick dynamics)

### Category 3: Right idea, wrong angle

**What was tested:** Tick counts from 1 to 16, different loss modes, halt modes.

**Result:** Tick2 is slightly best (5.40), tick4+ degrades. Tick12/16 loss ~5.78.

**Root cause:** The idea that "more recurrent iterations = deeper computation = better
reasoning" is intuitively correct for recurrent architectures. The problem is the
*supervision signal*.

In CTM, each tick produces a hidden state and a next-token loss. The `min_conf`
aggregation selects the tick with lowest entropy. But all ticks receive gradient
signal from the same next-token label. There is no mechanism that forces tick T+1
to *add* information that tick T did not have. The model learns to repeat the same
computation across ticks because the loss function does not distinguish "I already
know this" from "I need to think more."

The synapse-activate-trace pipeline (`model_ctm_llm.py:564-644`) is:
```
sync_a -> q_proj -> attention -> synapses -> state_trace -> NLM -> activated -> sync_o -> output
```
Each tick runs this pipeline once. The state_trace shifts by one position and
writes the new state. With the same input embedding, same KV cache, and same
attention pattern, the synapse receives nearly identical input at each tick. The
trace window provides temporal context, but if nothing external changes between
ticks, the recurrent update converges within 2-3 iterations and subsequent ticks
are redundant.

**How to fix the angle:** The multi-tick idea needs a supervision signal that
explicitly rewards *new* information per tick. Options:
- Per-tick loss with an explicit improvement margin: tick T+1 must have lower loss
  than tick T, not just low absolute loss.
- Different targets per tick: tick 1 predicts token n, tick 2 predicts token n+1
  or a higher-level semantic target.
- External input injection: change the attention context between ticks (e.g.,
  inject different position encodings or masked views) so each tick sees something
  new.

### Related failure: tick halt (s02, dr11, og03)

**Category 3 + 1 (bug):** Halt mechanisms consistently produce `effective_tick=1`
or stop immediately, meaning the model never uses the extra compute. The halt
decision is based on softmax confidence, which the model can game by pushing logits
to extremes (see dr11 analysis below). Additionally, `tick_halt_mode=threshold` in
the training loop (`model_ctm_llm.py:659-664`) only checks halt at inference
(`enable_tick_halt=False` during training), so the model never *learns* when to halt
during training. This is an implementation bug: the halt signal is not available as
training supervision.

---

## Part 2: MTP / ELF Multi-Token Prediction (s03, dr01, rg10)

### Category 2: CTM-ignorant (redundant)

**What was tested:** Multi-horizon prediction (h=2,4,8), pow2 scheduling,
improvement weights, crossed with ticks.

**Result:** ELF h4 is neutral (5.51 vs anchor 5.49). ELF h8 + many ticks degrades.
Zero gain across all sweeps (s03, dr01, rg10).

**Root cause:** MTP and ELF were designed for *autoregressive transformers* where
one forward pass = one prediction. Adding multi-horizon prediction gives the model
richer gradient signal because it must predict future tokens from the same hidden
state.

CTM already runs multiple ticks. Each tick is effectively a "free" MTP attempt:
the model computes attention, synapse, trace, and NLM at each tick, producing a new
hidden state and a new next-token distribution. The `min_conf` aggregation then
picks the best one. This is structurally equivalent to running MTP with multiple
attempts, except CTM re-computes the internal representation each time instead of
re-using a single hidden state.

Adding ELF on top means each tick now also predicts tokens at horizon h. But the
CTM's internal recurrence already ensures that later ticks have seen more recurrent
processing. The shifted CE loss from ELF is redundant with the tick-loss
improvement signal.

**Evidence:** The `tick_improve_weight` experiments in s03 (`s03_elf_linear_h4_improve`
loss 5.506 vs `s03_elf_next` loss 5.500) show that explicit tick-improvement
pressure does not help. The model cannot improve across ticks because (a) the
supervision is the same label at each tick, and (b) the recurrent state saturates
within 2-3 iterations.

---

## Part 3: Scaling Up CTM (s01, s04)

### Category 2: CTM-ignorant (wrong scaling axis)

**What was tested:** 12/16/24-layer CTM, d1024/d2048 cells, tick4, tick8.

**Result:** Transformer 12l (4.68) beats all CTM configs. Larger CTM is more
expensive without quality gain.

**Root cause:** CTM's recurrent state machine is not architecturally equivalent to a
deeper transformer. Each CTM tick runs one attention + one synapse + one NLM per
layer. With 12 layers and 4 ticks, the model runs 48 attention operations, 48
synapse forward passes, and 48 NLM passes. A 12-layer transformer with 4x the
hidden size runs 12 attention operations and 12 FFN passes.

The CTM tick multiplicity does not create "more depth" in the transformer sense.
Transformer depth means sequential composition of different attention patterns at
different layers, each with its own parameters. CTM ticks reuse the *same*
parameters at each layer, so tick N and tick N+1 at the same layer are parameter-
identical. The model can refine its representation through recurrence, but it
cannot learn new attention patterns beyond what the single set of layer parameters
supports.

**Implication:** CTM should not be scaled by adding more ticks or more layers in
parallel. The scaling axis should be (a) per-tick computation quality (better
synapse/trace architecture), (b) inter-layer diversity (different parameters per
tick, i.e., "deep CTM"), or (c) sparsity that reduces cost without reducing quality.

---

## Part 4: Top-K Cell Sparsity Without True Sparse Execution (s04, sp00-sp05)

### Category 1: Implementation bug (masking != sparsity)

**What was tested:** top-k cell selection with active fractions from 0.0625 to 1.0,
across d256-d2048.

**Result:** Top-k is quality-tolerant (loss stays flat down to 12.5% active
fraction) but memory does not decrease. In some cases, sparse uses *more* memory
than dense.

**Root cause:** The implementation in `model_ctm_llm.py:387-402`:
```python
scores = activated.detach().abs()
idx = torch.topk(scores, self.cell_topk, dim=-1).indices
mask = torch.zeros_like(activated).scatter_(-1, idx, 1.0)
mask = mask * (self.d_model / self.cell_topk)
return activated * mask
```
This applies the sparsity mask *after* the full dense computation. The synapse,
trace, NLM, and attention all operate on full dense tensors. The mask only
zeroes out elements in the final `activated` output. This means:
1. All dense intermediates are materialized (synapse input/output, trace storage).
2. The masking adds overhead (top-k computation, scatter, multiply).
3. The `rescale` factor (`d_model / cell_topk`) amplifies active elements,
   potentially increasing gradient magnitude.

For MoE routing (`moe_regional.py:72-173`), the pattern is similar: routing scores
are computed on full dense activations, then a mask is applied. The mask creates
the *illusion* of sparsity without the *infrastructure benefit*.

**Verdict:** The sparsity idea is correct for CTM. The implementation is a
prototype that proves routing viability but does not implement true sparse execution.
The missing step is: compute only active expert groups (gather/sparse forward),
store only active traces, and scatter results back. This requires kernel-level
changes, not just masking.

---

## Part 5: Block-Sparse Dispatch Parity (iv73)

### Category 1: Implementation bug (numerical divergence)

**What was tested:** `dense_mask` vs `block_sparse` vs `dropless` vs `capacity_drop`
dispatch modes.

**Result:** `dense_mask` is healthy (loss 5.71). `block_sparse` is broken (loss
145.6). `dropless` is broken (loss 785.1). `capacity_drop` produces NaN.

**Root cause:** The `block_sparse` path in `_run_group_sparse_regional_tick`
(`moe_regional.py:291-445`) dispatches computation per-expert: it gathers tokens
assigned to each expert, runs the expert's synapse and trace processor on those
tokens only, then scatters results back. The `dense_mask` path computes the full
dense forward pass and applies a mask.

These two paths should produce numerically identical results when the routing
decisions are the same, but they don't. The likely causes:
1. **Gather/scatter order:** Block-sparse processes experts sequentially; dense_mask
   processes all experts in one matrix multiply. The order of operations can differ
   in floating-point arithmetic.
2. **Trace management:** Block-sparse maintains per-expert traces that are updated
   independently. Dense-mask maintains a single shared trace that is masked. When
   trace state diverges between the two paths, the NLM output diverges, and the
   error compounds across ticks.
3. **Gradient flow:** The block-sparse path has different gradient paths through
   the gather/scatter operations, which can cause training instability.

**Verdict:** The grouped sparse backend needs a parity test (same routing, same
mini-batch, compare logits/loss tick-by-tick) before any architecture conclusions.

---

## Part 6: Async Banded Clocks (dr00, dr06)

### Category 4: Fundamentally incompatible

(Already analyzed in draft_revise_failure_analysis.md. Summary: async breaks
cross-layer temporal consistency in CTM's recurrent state machine.)

---

## Part 7: Draft-Revise and Corruption Training (dr00-dr06)

### dr02/dr04/dr05: Category 1: Implementation bugs

- **dr02 parallel:** Likely OOM from large slot projections (512 -> 4096) in
  `DraftSlotHead.slot_in`, multiplied by 8 ticks and 12 layers.
- **dr04 commit:** BCE numerical instability when corruption creates conflicting
  confidence/match signals.
- **dr05 memory carry:** `draft_memory_carry` config field is dead code -- never
  read by any forward pass.

### dr03 corruption: Works (the only quality win)

**Category 3 that found the right angle:** Corruption-based training works because
it provides *novel per-tick supervision*. Unlike standard tick training where every
tick sees the same label, corruption means some tokens are wrong, and the model
must detect and correct them. This creates a genuine reason for later ticks to
differ from earlier ticks -- the revision signal.

---

## Part 8: Residual Compute / Delta Caching (dr08, dr09, dr10)

### Category 2: CTM-ignorant

(Already analyzed in draft_revise_failure_analysis.md. Summary: CTM's synapse is
recurrent, not residual. Delta caching assumes additive skip connections.)

---

## Part 9: DINO Speed Spectrum (dr07)

### Category 3: Right idea, wrong angle

(Already analyzed. Summary: tick-to-tick variation is too small for distillation.
Should distill between different models or different augmentation views.)

---

## Part 10: Tick Controller with Compute Penalty (dr11)

### Category 3: Right idea, wrong angle

(Already analyzed. Summary: confidence is trivially gameable. Need quality-based
stopping, not entropy-based.)

---

## Part 11: Training Objective Replacement (dr12)

### Category 2: CTM-ignorant

(Already analyzed. Summary: standard causal CE discards min_conf multi-tick
advantage. Latent denoising applies to continuous manifolds, not discrete-token
hidden states.)

---

## Part 12: MoE Routing Variants (moe01)

### Hash routing (moe01, loss 5.63): Category 2: CTM-ignorant

**Root cause:** Hash routing (`moe_regional.py:97-100`) maps tokens to experts
using `(pos + offset + layer_id) % routed_count`. This is a *position-based*
routing that does not consider token content at all. In CTM, the activated state
carries recurrent information that is different for each position and each tick.
Hash routing ignores this and assigns the same expert pattern regardless of what
the cell is computing. This is appropriate for static sparsity in transformers
(where position is the main routing signal) but wrong for CTM (where the activated
state, not position, determines which computation is needed).

### Expert-choice routing (moe01, loss 5.57): Category 2: CTM-ignorant

**Root cause:** Expert-choice uses the *mean* routing score across the batch to
select experts (`moe_regional.py:103-104`). In CTM, different positions in the
same sequence have very different recurrent states (early positions have shallow
traces, late positions have deeper traces). A batch-mean score is meaningless for
CTM routing -- it averages away the position-dependent signal.

### Router z-loss (moe04, loss 5.81): Category 1: Implementation (too strong)

**Root cause:** The z-loss weight (1e-3) was too aggressive for this model scale.
Z-loss is designed to prevent routing logits from growing large, which is useful
for very deep models with thousands of experts. At 16 experts and d=512, the
routing space is small, and aggressive z-loss over-regularizes the router, forcing
all experts to have near-uniform probability and destroying the routing signal.

---

## Part 13: Warmup and Dropout for Sparse CTM (rg08)

### Category 2: CTM-ignorant

**What was tested:** Top-k warmup (500/1000/2000 steps) and expert dropout (0.05).

**Result:** Both hurt. Warmup 500: loss 5.46. Warmup 2000 + dropout: loss 5.61.

**Root cause:** The MoE warmup and dropout were designed for standard MoE transformers
where routing starts random and needs time to specialize. In CTM with regional
routing, the routing is *deterministic at each tick* -- it selects experts based on
activated state magnitude. The routing pattern changes every tick because the
activated state evolves. Warmup forces early uniform routing, which means the CTM
runs with all experts active for the first 500-2000 steps. During this period,
the CTM cell dynamics learn to rely on full activation, and when sparse routing
kicks in later, the model must adapt, causing a quality dip.

Expert dropout randomly drops routed experts during training. In CTM, dropping
an expert means one region of the recurrent state is zeroed out. Unlike
transformers where the residual connection can compensate for a dropped expert, CTM
has no skip connection -- the dropped region's contribution to the trace and
synchronization is lost, creating a hole in the recurrent state.

---

## Part 14: Regional MoE Composition with Halt/MTP (og00 composite)

### Category 1: Implementation bug (composite supervision conflict)

**What was tested:** Regional p4 + halt threshold 0.30 + MTP 1,2,4.

**Result:** Loss 16.0, effective_tick=1.0 (always halts at first tick).

**Root cause:** The composite experiment stacks halt + MTP + regional routing
without checking for supervision conflicts. Halt is checked *only at inference*
(`model_ctm_llm.py:659-664`, `enable_tick_halt=False` during training). So during
training, the model runs all ticks with both MTP loss and regional routing. At
inference, halt kicks in and the model immediately stops at tick 1. But the model
was never trained to produce good tick-1 outputs -- it was trained with 4 ticks,
where tick-1 is the weakest. The inference behavior (always halt at tick 1 with
poor predictions) is a direct consequence of training/inference mismatch.

---

## Part 15: Overnight Sparse CTM -- Long-Tick Delta Proxies (og08)

### Category 2: CTM-ignorant

**What was tested:** 8-16 ticks with restricted active cells per tick, simulating
"keyframe" sparse recurrence.

**Result:** Single survivor (d512, tick8, p4) at loss 33.5, only 320 steps.
Per-tick losses increase monotonically.

**Root cause:** The delta-time proxy assumes that running many cheap ticks with
sparse activation can replace fewer full ticks. But in CTM, the *quality* of each
tick depends on the full activated state, not just the active fraction. When only
12.5% of cells are active, the synchronization computation (`_compute_synch`)
operates on a mostly-zero vector. The synapse receives a mixture of real and zero
inputs, and the trace accumulates degraded state. Over many ticks, this degradation
compounds, producing monotonically increasing loss.

---

## Part 16: Fast/Slow Output Paths (og09 fastslow, og10)

### Category 1: Implementation bug (too expensive + undertrained)

**What was tested:** Differentiated fast/slow output heads, anytime early-tick
supervision, reflex paths.

**Result:** Most og10 experiments OOM'd. og09 fastslow variants had mixed results.
The reflex path (`og09_fastpath_reflex`) worked well (loss 4.02, 8684 tok/s).

**Root cause (for failures):** The fast/slow output architecture requires
additional output heads, additional loss terms (`fast_output_weight`,
`slow_output_weight`, `habit_output_weight`), and additional tick supervision
(`fast_output_ticks`). These all increase memory and computation. The
`fast_output_mode=anytime` variant produces output at specific tick indices, which
requires maintaining tick outputs across all ticks and computing additional loss
terms.

The reflex path works because it is *simplifying* the architecture (fewer ticks,
shorter memory) rather than adding complexity. This reinforces a broader pattern:
CTM improvements come from simplification, not from adding more modules.

---

## Master Summary Table

| Idea | Source | Category | Root Cause Summary |
| --- | --- | --- | --- |
| More ticks (3-16) | s02 | 3: Wrong angle | Same supervision per tick; state saturates in 2-3 iterations |
| Tick halt (threshold/confidence) | s02, og03, dr11 | 3+1 | Confidence is gameable; halt not trained during training (bug) |
| ELF/MTP multi-token prediction | s03, dr01, rg10 | 2: CTM-ignorant | Multi-tick recurrence already subsumes MTP |
| Scale up CTM (layers/ticks) | s01, s04 | 2: CTM-ignorant | Ticks reuse same params, not transformer-equivalent depth |
| Top-k cell sparsity | s04, sp00-sp05 | 1: Bug | Mask applied after dense compute; no real cost savings |
| Block-sparse dispatch | iv73, og05 | 1: Bug | Numerical divergence vs dense_mask; trace management error |
| Async banded clocks | dr00, dr06 | 4: Incompatible | Breaks cross-layer temporal state consistency |
| Parallel draft slots | dr02 | 1: Bug | OOM from large slot projections |
| Commit/confidence head | dr04 | 1: Bug | BCE instability with corruption conflict |
| Memory carry | dr05 | 1: Bug | Dead code; `draft_memory_carry` never read |
| Corruption draft-revise | dr03 | Works | Novel per-tick supervision provides genuine revision signal |
| DINO speed spectrum | dr07 | 3: Wrong angle | Tick-to-tick variation too small for distillation |
| Residual delta caching | dr08 | 2: CTM-ignorant | CTM is recurrent, not residual; synapse is nonlinear |
| Block-level skip | dr09 | 1: Bug | Novelty bookkeeping overhead exceeds savings at d=512 |
| Recursive NLM fast path | dr10 | 2: CTM-ignorant | GLU nonlinearity prevents linear delta approximation |
| Tick controller + penalty | dr11 | 3: Wrong angle | Confidence-based stopping is trivially gameable |
| Causal CE objective | dr12 | 2: CTM-ignorant | Discards min_conf multi-tick advantage |
| Latent denoise objective | dr12 | 2: CTM-ignorant | MSE on discrete-token-conditioned hidden states is meaningless |
| Hash routing | moe01 | 2: CTM-ignorant | Position-based; ignores CTM's content-dependent activated state |
| Expert-choice routing | moe01 | 2: CTM-ignorant | Batch-mean scores ignore position-dependent recurrent state |
| Router z-loss | moe04 | 1: Bug | Weight too aggressive for 16-expert scale |
| Warmup/dropout for sparse CTM | rg08 | 2: CTM-ignorant | Forces uniform routing early; dropout creates holes in recurrent state |
| Regional + halt + MTP composite | og00 | 1: Bug | Training/inference mismatch; halt not active during training |
| Long-tick delta proxies | og08 | 2: CTM-ignorant | Sparse activation degrades state; compounds over many ticks |
| Fast/slow output paths | og09/og10 | 1: Bug | Too expensive (OOM); reflex simplification works |

## Distribution by Category

| Category | Count | Percentage | Theme |
| --- | ---: | ---: | --- |
| 1: Implementation bug | 10 | 38% | Masking instead of sparse execution, numerical divergence, dead code, OOM |
| 2: CTM-ignorant | 10 | 38% | Applying transformer optimizations to recurrent architecture |
| 3: Right idea, wrong angle | 5 | 19% | Sound concept needs supervision/architecture adaptation for CTM |
| 4: Fundamentally incompatible | 1 | 4% | Async clocks cannot work with CTM's recurrent state machine |

## Meta-Lessons

### Lesson 1: The #1 failure pattern is "implementing masking, not sparsity"

The sparsity work across s04, sp00-sp05, moe, iv73, and dr09 all share the same
core issue: the code applies masks *after* dense computation rather than skipping
computation for inactive elements. This is not just an optimization issue -- it
means the model never trains under true sparse conditions, so the routing
decisions are learned in a regime that does not match inference.

**Fix:** Implement gather/scatter dispatch for expert groups. Only materialize
active expert tensors. Store only active traces. This is the single highest-
leverage engineering task.

### Lesson 2: CTM is recurrent, not residual

The most common conceptual error (Category 2) is treating CTM like a transformer.
Transformer optimizations (MTP, residual caching, delta propagation, dropout,
hash routing) assume:
- One forward pass = one prediction (MTP assumption fails because CTM has ticks).
- Skip connections allow dropping sub-computations (fails because CTM has no residual
  path; the synapse is the only path from attention to output).
- Position determines routing (fails because CTM's routing depends on the
  activated state, which changes per position, per tick, per layer).
- Gradient flow through skip connections compensates for dropped computations
  (fails because dropping an expert region in CTM creates a hole in the recurrent
  state that propagates through the trace).

### Lesson 3: Simplification beats complexity across all sweeps

Every positive result comes from *removing* complexity:
- `s05_synapse2_mh2`: simpler synapse beats default.
- `sp04_tick1`: fewer ticks beats more ticks.
- `sp03_sd1_mh1`: lighter synapse/memory beats heavier settings under sparsity.
- `rg11_p4_shared1_top1`: small core + one routed region beats complex multi-region.
- `og09_fastpath_reflex`: short memory + tick1 beats multi-tick architecture.
- `dr03_corrupt0p15`: corruption-based training (a single additional loss term)
  beats all complex draft-revise stacks.

Every negative result comes from *adding* complexity: more ticks, more modules,
more loss terms, more routing modes, more output heads.

### Lesson 4: Per-tick supervision is the key unsolved problem

The CTM framework's biggest weakness is not the architecture -- it is the
supervision. The current `min_conf` loss selects the best tick but does not
force ticks to be different. This means:
- More ticks add cost without adding information (s02).
- ELF/MTP are redundant with multi-tick (s03).
- Halt cannot be trained because there is no per-tick improvement signal (dr11).
- Delta caching cannot save compute because there is no basis for "this tick
  doesn't need re-computation" (dr08-dr10).

The one exception is corruption training (dr03), which works precisely because
it *does* provide novel per-tick supervision -- the model must correct corrupted
tokens, giving later ticks a genuine reason to differ from earlier ticks.

**Future direction:** Design per-tick supervision that explicitly rewards
complementary computation across ticks, not redundant computation.
