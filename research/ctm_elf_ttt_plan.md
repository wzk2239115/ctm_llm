# CTM+ELF+TTT Research Plan

Date: 2026-06-06

This document defines the next research stage after the CTM+ELF training
validation release. The previous version proves that the current CTM-LLM can
train stably on two nodes and fit the SFT corpus. The next question is whether
CTM's internal clock can become an actual test-time learner instead of only a
multi-step recurrent computation.

## Research Thesis

CTM gives the model internal thought time. ELF-style causal training makes that
thought path useful for next-token modeling. TTT should make the thought process
adaptive: while the model is thinking on the current sample, a small local state
or module should learn from a self-supervised objective before the final answer
is emitted.

Short version:

```text
CTM       = think for multiple ticks
ELF       = supervise the language path
TTT       = learn during the thinking ticks
CTM+ELF+TTT = thought time + language supervision + test-time adaptation
```

## Papers To Anchor The Design

- Test-Time Training with Self-Supervision for Generalization under Distribution
  Shifts, Yu Sun et al., ICML 2020.
  https://proceedings.mlr.press/v119/sun20b.html
- Learning to (Learn at Test Time): RNNs with Expressive Hidden States, Yu Sun
  et al., 2024.
  https://arxiv.org/abs/2407.04620
- TTT++: When Does Self-Supervised Test-Time Training Help?, NeurIPS 2021.
  https://proceedings.neurips.cc/paper/2021/hash/873be07017487acdc3b80cc4c05fdf3a-Abstract.html
- Tent: Fully Test-Time Adaptation by Entropy Minimization, ICLR 2021.
  https://openreview.net/forum?id=uXl3bZLkr3c

The closest architectural match is TTT Layers, because CTM already has an
internal recurrent state (`activated`, `state_trace`, synchronization state,
and per-tick outputs). TENT is useful only as a baseline. TTT++ is important
because it explains why ungated test-time updates can make the model worse.

## Current CTM Hooks

The current implementation already exposes useful insertion points:

- `CTMBlock.forward` has an internal tick loop.
- `state_trace` is the per-neuron temporal memory.
- `activated` is the current neuron activation state.
- `sync_a` is the action synchronization representation used to query attention.
- `sync_o` is the output synchronization representation used to produce
  per-tick hidden outputs.
- `forward_train` supervises final LM loss and per-tick LM loss.
- `best_tick` and `conf_tick` are already logged during training.

That means the first TTT work should not start by rewriting the whole model.
The safest path is to add probes and small optional modules around the existing
tick loop.

## Design Principle

Do not update the whole CTM at test time.

The first useful TTT version should update only a tiny fast state or adapter. The
base model must remain stable, otherwise failures will be impossible to debug
and long online sessions may drift.

Use this separation:

```text
theta_base  = trained CTM+ELF weights, frozen during inference
phi_fast    = tiny TTT adapter or learner, resettable per request
state_fast  = per-request CTM/TTT state
controller  = decides whether to keep adapting, stop, or rollback
```

## Stage 0: CTM Clock Probe

Goal: prove the internal clock carries useful measurable signal before adding
new train-time complexity.

Add a probe/eval path that logs, per generated token:

- chosen token
- token entropy
- best tick by per-tick loss when labels are available
- confidence tick by normalized entropy
- per-tick top token / entropy
- optional per-layer tick summaries

Success criteria:

- Different prompts produce different tick patterns.
- Harder prompts use later ticks more often than easy prompts.
- Tick entropy changes across iterations instead of staying flat.
- Reducing `num_iters` measurably changes eval loss or generation behavior.

Failure signal:

- Tick metrics are almost constant across prompts and layers.
- Later ticks do not improve loss/entropy.
- `num_iters=1` behaves nearly the same as `num_iters=4/8/16`.

## Stage 1: TTT Probe Loss Without Test-Time Update

Goal: add self-supervised objectives that can be measured inside CTM without
changing inference behavior.

Candidate probe losses:

1. Next-latent prediction:

```text
predict h[t+1] from sync_o[t]
```

2. Masked-latent reconstruction:

```text
mask part of h or sync_o, reconstruct masked dimensions
```

3. Synchronization consistency:

```text
same prompt with token dropout / span mask should have nearby sync_o
```

4. Tick improvement objective:

```text
later tick should reduce next-token loss or entropy compared with earlier tick
```

Initial recommendation: start with next-latent prediction and tick improvement.
They are language-native and do not require data augmentation policy design.

Success criteria:

- The self-supervised probe loss correlates with downstream LM loss.
- Good generations have lower probe loss or cleaner tick trajectories than bad
  generations.
- Probe loss can distinguish easy and hard prompts.

## Stage 2: Safe CTM TTT Layer

Goal: introduce test-time learning while keeping the trained CTM weights frozen.

Add a tiny optional module inside the CTM tick loop:

```text
state_trace -> trace_processor -> activated
activated -> ttt_layer_phi -> delta_activated
activated + delta_activated -> sync_o -> output
```

At inference time:

1. Initialize or load `phi_fast`.
2. Run CTM for a prompt prefix.
3. Compute self-supervised loss on the prefix.
4. Take `K` gradient steps on `phi_fast` only.
5. Generate the next token with the adapted TTT Layer.
6. Reset or keep `phi_fast` depending on session policy.

Recommended defaults:

```text
ttt_hidden_mult = 2
ttt_steps       = 1 to 3
ttt_lr          = 1e-5 to 1e-3
ttt_max_tokens  = prompt prefix only at first
rollback        = enabled
```

The TTT Layer is a residual bottleneck:

```text
activated = activated + gate * up(silu(down(norm(activated))))
```

Keep it disabled by default so old checkpoints and the successful validation
path remain compatible.

## Stage 3: In-Tick TTT Objectives

Goal: move beyond prefix LM loss and make the internal tick itself expose local
self-supervised learning signals.

After Stage 2 works, add internal objectives around `CTMBlock.forward`:

```text
sync_o[t] -> predict sync_o[t+1]
activated[t] -> predict masked activated[t]
tick t logits -> improve tick t+1 logits
```

These objectives can train or adapt the existing TTT Layer more directly than
prefix LM loss.

Safer state update:

```text
activated = activated + gate * delta_activated
```

Riskier variant:

```text
temporarily update a copy of learner parameters inside the tick loop
```

Prefer state update over parameter update first. It is easier to make
deterministic, easier to reset per request, and less likely to corrupt the base
model.

## Gating And Rollback

TTT must have a brake.

Useful gates:

- Do not adapt if entropy is already low.
- Do not adapt if self-supervised loss increases after a step.
- Do not adapt if adapted logits drift too far from base logits.
- Do not adapt on very short prompts.
- Roll back if next-token entropy drops but sampled output quality worsens on a
  simple consistency check.

Suggested metrics:

```text
probe_loss_before
probe_loss_after
entropy_before
entropy_after
kl(base_logits || adapted_logits)
tick_selected_before
tick_selected_after
```

Initial rollback rule:

```text
accept update only if:
  probe_loss_after <= probe_loss_before
  and kl(base_logits, adapted_logits) <= kl_budget
```

## Experiments

### E0: Clock Ablation

Run the 1024/16L checkpoint with:

```text
num_iters = 1, 2, 4, 8, 16
```

Measure:

- eval loss/ppl
- generation samples
- per-token entropy
- tick histogram

Question:

Does more thought time help before any TTT update exists?

### E1: Probe Correlation

Train no new model. Compute probe losses and correlate them with LM loss on
random held-out samples.

Question:

Is the proposed self-supervised objective meaningful?

### E2: Adapter TTT On Prefix

Freeze the model. Add the adapter and adapt only on the prompt prefix before
generation.

Question:

Can a small test-time update reduce prefix loss without damaging generated
tokens?

### E3: Online Adapter TTT

Keep a session-local adapter across turns.

Question:

Does adaptation improve session coherence, or does it drift?

### E4: In-Tick State TTT

Move the fast learner into the CTM tick loop.

Question:

Does internal learning change `best_tick` and `conf_tick` in a useful way?

## Risks

- Self-supervised objective may be misaligned with language quality.
- Entropy can reward confident nonsense.
- Updating too many parameters can cause drift and catastrophic forgetting.
- Prefix adaptation can overfit the prompt and hurt continuation.
- Current SFT data quality may hide architectural gains.
- Extra TTT steps may make inference too slow unless gated aggressively.

## Recommended Implementation Order

1. Add `ctm_clock_probe.py` for tick/entropy/token visualization.
2. Add `eval_loss.py --num_iters` sweeps and a small report script.
3. Add optional config fields for TTT, all disabled by default.
4. Add `TTTAdapter` after the final hidden state.
5. Add an inference-only `generate_ttt` path that updates only adapter copies.
6. Add TensorBoard/JSON logging for TTT accept/reject, KL, entropy, and probe
   loss.
7. Only then experiment with in-tick state adaptation.

## First Minimal Version

The first implementation should be deliberately small:

```text
feature name: ttt_layer
base weights: frozen
updated at test time: CTM TTT Layer parameters only
self-supervised loss: prefix next-token LM loss
accept rule: loss decreases and KL within budget
scope: eval/generation only, training path unchanged
```

This first version will not prove the full CTM+TTT theory, but it will answer
the most urgent engineering question:

```text
Can CTM+ELF benefit from a safe per-request test-time learner without breaking
the stable training path?
```

## Implemented Prototype: CTM TTT Layers + `ttt_eval.py`

The first runnable prototype is:

```bash
ttt_eval.py
```

The model now has an optional CTM TTT Layer inside each internal tick. This is
closer to "Learning to (Learn at Test Time): RNNs with Expressive Hidden
States" than an external adapter, because the learner directly modifies CTM's
activated state before synchronization:

```text
state_trace -> trace_processor -> activated
activated -> TTTMLP_phi -> delta_activated
activated + delta_activated -> sync_o -> output
```

`phi` is a real parameterized MLP. It is zero-initialized through its output
projection, so enabling it on an old checkpoint starts near the base CTM
behavior. During test-time training, `ttt_eval.py` updates the real
`.ttt_layer.*` parameters with `optimizer.step()`.

Default behavior:

1. Load the trained CTM+ELF checkpoint.
2. Enable optional CTM TTT Layers with `--ttt_layer 1`.
3. Freeze all parameters except `--ttt_target`.
4. Build a chat-formatted prompt prefix.
5. Use prefix next-token LM loss as the self-supervised TTT objective.
6. Run `--ttt_steps` AdamW updates on the selected real parameters.
7. Accept the update only if prefix loss decreases and last-token KL stays
   under `--ttt_kl_budget`.
8. Roll back the selected parameters if the update is rejected.
9. Generate with the accepted adapted parameters, or with restored base
   parameters after rollback.

Example for the successful 1024/16L validation checkpoint:

```bash
python ttt_eval.py \
  --weight out/ctm_2node_1024_16l_bs16_44ep_1024_resume.pth \
  --hidden_size 1024 \
  --num_hidden_layers 16 \
  --d_model 512 \
  --d_input 256 \
  --heads 8 \
  --n_synch_out 512 \
  --n_synch_action 512 \
  --iterations 4 \
  --memory_length 5 \
  --synapse_depth 2 \
  --ttt_layer 1 \
  --ttt_target ttt_layers \
  --ttt_steps 1 \
  --ttt_lr 1e-5 \
  --ttt_kl_budget 0.25 \
  --temperature 0.3 \
  --top_p 0.8 \
  --top_k 40 \
  --repetition_penalty 1.08 \
  --prompt "Explain what a neural network is in simple words." \
  --max_new_tokens 160
```

More aggressive targets:

```bash
--ttt_target last_ttt_layer
--ttt_target last_mlp
--ttt_target last_block
--ttt_target all_norms
--ttt_target lm_head
```

Use `--ttt_target all` only as a failure-mode experiment. It updates the whole
model at test time and can easily destroy the base behavior.

If an accepted adapted model should be saved for inspection:

```bash
--save_ttt_weight out/ttt_probe_accepted.pth
```

This prototype is a baseline for proving the engineering claim:

```text
TTT is actually changing CTM+ELF TTT Layer parameters during inference.
```

It is now aligned with the stronger research direction:

```text
CTM internal ticks contain a real learner whose parameters can change at test time.
```

The remaining scientific claim still needs experiments:

```text
Do those test-time updates improve useful task-specific computation rather than
just lowering prefix loss?
```

That requires per-tick CTM clock probes, held-out loss sweeps, generation
comparison, and ablations over `--ttt_layer`, `--ttt_steps`, `--ttt_target`,
and `--num_iters`.

## Implemented Prototype: Code-Stream TTT Layer Training

The second runnable prototype is:

```bash
train_ttt_stream.py
```

This script uses a streaming text/code corpus to continue-train only CTM TTT
Layer parameters. The base CTM+ELF checkpoint stays frozen. This is not pure
single-sample inference-time TTT; it is TTT-layer domain adaptation / continued
pretraining. It is useful for testing whether the TTT Layer can absorb a new
domain such as code while preserving the base model.

Default OpenCoder setup:

```text
dataset: OpenCoder-LLM/opc-annealing-corpus
subset : algorithmic_corpus
field  : text
mode   : streaming=True
target : .ttt_layer.* parameters only
```

Run through the no-SSH pool:

```bash
./scripts/ctmctl pool submit infra/clusters/h100_2nodes_ttt_code.env \
  --max_steps 1000 \
  --save_weight ctm_ttt_opc_code_1024_smoke \
  --run_name ctm-ttt-opc-code-1024-smoke
```

Longer run:

```bash
./scripts/ctmctl pool submit infra/clusters/h100_2nodes_ttt_code.env \
  --max_steps 20000 \
  --save_interval 2000 \
  --learning_rate 1e-5 \
  --save_weight ctm_ttt_opc_code_1024_20k \
  --run_name ctm-ttt-opc-code-1024-20k
```

Single node or manual torchrun:

```bash
torchrun --nproc_per_node=8 train_ttt_stream.py \
  --from_weight out/ctm_2node_1024_16l_bs16_44ep_1024_resume.pth \
  --dataset_name OpenCoder-LLM/opc-annealing-corpus \
  --dataset_config algorithmic_corpus \
  --hidden_size 1024 \
  --num_hidden_layers 16 \
  --d_model 512 \
  --d_input 256 \
  --heads 8 \
  --n_synch_out 512 \
  --n_synch_action 512 \
  --iterations 4 \
  --memory_length 5 \
  --synapse_depth 2 \
  --max_steps 1000
```

Important interpretation:

```text
If code loss decreases, the TTT Layer can learn the code distribution.
If code generation improves without hurting SFT eval too much, the TTT Layer is
acting as a useful fast domain learner.
If SFT eval collapses or KL/generation drift is large, update only
last_ttt_layer or reduce LR/steps.
```

Raw code-domain adaptation should first be evaluated as prefix completion, not
chat instruction following. Use:

```bash
python eval_completion.py \
  --weight out/ctm_ttt_opc_code_1024_bs16_seq1024_1h_1024.pth \
  --hidden_size 1024 \
  --num_hidden_layers 16 \
  --d_model 512 \
  --d_input 256 \
  --heads 8 \
  --n_synch_out 512 \
  --n_synch_action 512 \
  --iterations 4 \
  --memory_length 5 \
  --synapse_depth 2 \
  --ttt_layer 1 \
  --prompt $'def is_palindrome(s):\n    ' \
  --max_new_tokens 160
```

Instruction prompts such as "Write a Python function..." require either
instruction-code SFT data or synthetic instruction construction from code
snippets. A raw code corpus teaches code continuation, not necessarily
user-request-to-code mapping.
