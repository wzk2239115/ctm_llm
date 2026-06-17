#!/usr/bin/env python3
"""
HRM-Inspired CTM Experiment Plan — Lessons from Hierarchical Reasoning Models.

Applies techniques learned from HRM / HRM-Text to the CTM baseline architecture.
Each stage tests a specific hypothesis derived from HRM's design choices.

Key HRM ideas under test:
  - Gradient control: truncated BPTT, BP warmup, state detach (HRM's nocarry)
  - Optimization: Adam-atan2, stablemax CE
  - Attention: gated attention (HRM-Text's sigmoid gate)
  - Structure: hierarchical recurrence (HRM's H×L nested loops)
  - Training: EMA weights, adaptive computation time (ACT)

Usage:
    python scripts/experiment_plan_hrm_inspired.py plan [--stage all]
    python scripts/experiment_plan_hrm_inspired.py submit --stage hr01 --no-wait
    python scripts/experiment_plan_hrm_inspired.py csv [--stage all]
"""

import json, os, re, shlex, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POOL_CONFIG = "infra/envs/h100_baseline.env"
MASTER_ADDR = "11.131.210.78"
PORT = 8765
BASELINE_NODES = ("11.131.209.154", "11.131.210.3", "11.131.210.78", "11.131.211.9")
GPUS_PER_NODE = 8
_slot_idx = [0]

STAGES_ORDERED = [
    "hr00", "hr01", "hr02", "hr03",
    "hr04", "hr05",
    "hr06", "hr07",
    "hr08",
    "hr09", "hr10",
    "hr11", "hr12", "hr13", "hr14",
    "hr15",
]
ALL_STAGES = STAGES_ORDERED + ["all"]


def _next_slot():
    i = _slot_idx[0]
    _slot_idx[0] += 1
    node = BASELINE_NODES[i // GPUS_PER_NODE % len(BASELINE_NODES)]
    gpu = i % GPUS_PER_NODE
    return f"{node}:{gpu}"


def _p(train_module, extra_args=None):
    parts = [train_module]
    if extra_args:
        for k, v in extra_args.items():
            if v is None:
                continue
            if isinstance(v, bool):
                parts.append(f"--{k}" if v else f"--no-{k}")
            elif isinstance(v, list):
                parts.append(f"--{k}")
                parts.extend(str(x) for x in v)
            else:
                parts.append(f"--{k}")
                parts.append(str(v))
    return " ".join(parts)


def exp(name, question, command, tags=None, impl_status="needs_impl", hypothesis=None):
    return {
        "name": name,
        "question": question,
        "command": command,
        "tags": tags or [],
        "node_addr": _next_slot(),
        "impl_status": impl_status,
        "hypothesis": hypothesis or question,
    }


# ═══════════════════════════════════════════════════════════════
# Per-task base configs — same as paper plan for fair comparison
# ═══════════════════════════════════════════════════════════════

SORT_BASE = dict(
    seed=412, iterations=50, memory_length=25,
    d_model=512, d_input=128, n_synch_out=32, n_synch_action=32,
    synapse_depth=4, heads=4, memory_hidden_dims=4, dropout=0.0,
    deep_memory=True, do_normalisation=False,
    positional_embedding_type="none",
    neuron_select_type="random-pairing",
    n_random_pairing_self=0, N_to_sort=30,
    batch_size=32, batch_size_test=32,
    lr=1e-3, training_iterations=100001,
    warmup_steps=5000, use_scheduler=True, scheduler_type="cosine",
    weight_decay=0.0, gradient_clipping=-1,
    track_every=1000, save_every=10000,
    reload=False, device=[0],
)

PARITY_BASE = dict(
    seed=0, iterations=75, memory_length=25,
    parity_sequence_length=64,
    d_model=1024, d_input=512,
    n_synch_out=32, n_synch_action=32,
    synapse_depth=1, heads=8, memory_hidden_dims=16, dropout=0.0,
    deep_memory=True, do_normalisation=False,
    positional_embedding_type="custom-rotational-1d",
    backbone_type="parity_backbone",
    weight_decay=0.0, gradient_clipping=0.9,
    use_scheduler=True, scheduler_type="cosine",
    batch_size=64, batch_size_test=256,
    lr=1e-4, training_iterations=200001,
    warmup_steps=500, track_every=1000,
    save_every=10000, reload=False,
    device=[0], use_amp=False,
    neuron_select_type="random", n_test_batches=20,
)

MAZES_BASE = dict(
    model="ctm", neuron_select_type="first-last",
    dataset="mazes-small",
    synapse_depth=8, heads=16, iterations=75, memory_length=25,
    d_model=2048, d_input=512,
    backbone_type="resnet34-2",
    n_synch_out=64, n_synch_action=32, memory_hidden_dims=32,
    deep_memory=True, do_normalisation=False,
    weight_decay=0.0, dropout=0.1,
    positional_embedding_type="none",
    maze_route_length=100, cirriculum_lookahead=5, expand_range=False,
    batch_size=64, batch_size_test=128,
    lr=1e-4, training_iterations=100001,
    warmup_steps=10000,
    use_scheduler=True, scheduler_type="cosine",
    gradient_clipping=-1,
    track_every=1000, save_every=10000,
    reload=False, device=[0],
    data_root="baseline/data/mazes", n_test_batches=20,
)

CIFAR10_BASE = dict(
    model="ctm", dataset="cifar10",
    d_model=256, d_input=64, synapse_depth=5, heads=16,
    n_synch_out=256, n_synch_action=512,
    n_random_pairing_self=0, neuron_select_type="random-pairing",
    iterations=50, memory_length=15,
    deep_memory=True, memory_hidden_dims=64,
    dropout=0.0, dropout_nlm=0, do_normalisation=False,
    positional_embedding_type="none", backbone_type="resnet18-1",
    training_iterations=200001, warmup_steps=1000,
    use_scheduler=True, scheduler_type="cosine",
    weight_decay=1e-4,
    save_every=2000, track_every=2000, n_test_batches=50,
    batch_size=512, batch_size_test=512,
    lr=1e-4, device=[0], seed=1, data_root="baseline/data/",
)

QAMNIST_BASE = dict(
    seed=0, model_type="ctm", memory_length=30,
    q_num_images=3, q_num_images_delta=2, q_num_repeats_per_input=10,
    q_num_operations=3, q_num_operations_delta=2, q_num_answer_steps=10,
    d_model=1024, d_input=64, n_synch_out=32, n_synch_action=32,
    synapse_depth=1, heads=4, memory_hidden_dims=16, dropout=0.0,
    deep_memory=True, do_normalisation=False, weight_decay=0.0,
    use_scheduler=True, scheduler_type="cosine",
    batch_size=64, batch_size_test=256,
    lr=1e-4, training_iterations=200001,
    warmup_steps=500, track_every=1000, save_every=10000,
    reload=False, device=[0], use_amp=False,
    neuron_select_type="random",
    data_root="baseline/data/", n_test_batches=20,
)

TASKS = {
    "sort":    ("baseline.tasks.sort.train",               SORT_BASE,    50),
    "parity":  ("baseline.tasks.parity.train",             PARITY_BASE,  75),
    "mazes":   ("baseline.tasks.mazes.train",              MAZES_BASE,   75),
    "cifar10": ("baseline.tasks.image_classification.train", CIFAR10_BASE, 50),
    "qamnist": ("baseline.tasks.qamnist.train",            QAMNIST_BASE,  None),
}

# Tasks to use for each phase (subset for faster iteration)
FAST_TASKS = ["sort", "parity"]           # quick validation
MID_TASKS = ["sort", "parity", "cifar10"] # broader validation
ALL_TASK_LIST = list(TASKS.keys())


def with_seed(cfg, seed):
    c = dict(cfg)
    c["seed"] = seed
    return c


# ═══════════════════════════════════════════════════════════════
# STAGE BUILDERS
# ═══════════════════════════════════════════════════════════════

def build_hr00_baselines(plan):
    """Control group: plain CTM paper configs (for comparison with HRM variants).

    These are the SAME configs as st00, included here as a self-contained
    control group so results from this plan can be analyzed independently.
    """
    for task_name in MID_TASKS:
        module, base, _ = TASKS[task_name]
        cfg = dict(base)
        cfg["log_dir"] = f"logs/hrm_inspired/hr00/{task_name}_baseline"
        plan.append(exp(
            f"hr00_{task_name}_baseline",
            f"{task_name}: CTM baseline (control for HRM experiments)",
            _p(module, cfg),
            tags=[task_name, "baseline"],
            impl_status="ready",
            hypothesis="CTM paper config serves as control.",
        ))
    return plan


# ─── Phase A: Gradient Control ───────────────────────────────
# Hypothesis: CTM's full BPTT through all ticks is unnecessary.
# HRM/HRM-Text only backprop through the last few ticks (nocarry + bp_warmup).
# If true: huge memory/speed gains, possibly better optimization.


def build_hr01_truncated_bptt(plan):
    """Truncated BPTT: only backprop through the last N ticks.

    Implementation: wrap ticks < (iterations - bp_steps) in torch.no_grad().
    bp_steps=1 means only the last tick gets gradients (like HRM's 1-step grad).
    bp_steps=0 means full BPTT (equivalent to baseline).
    """
    for task_name in MID_TASKS:
        module, base, _ = TASKS[task_name]
        for bp_steps in [1, 2, 5, 10]:
            cfg = dict(with_seed(base, 0))
            cfg["bp_steps"] = bp_steps
            cfg["log_dir"] = f"logs/hrm_inspired/hr01/{task_name}_bp{bp_steps}"
            plan.append(exp(
                f"hr01_{task_name}_bp{bp_steps}",
                f"{task_name}: truncated BPTT, bp_steps={bp_steps}",
                _p(module, cfg),
                tags=[task_name, "gradient-control", "truncated-bptt"],
                impl_status="ready",
                hypothesis=f"Only {bp_steps} ticks of gradient suffices; "
                           f"baseline uses {base.get('iterations', '?')} ticks of BPTT.",
            ))
    return plan


def build_hr02_bp_warmup(plan):
    """BP warmup: ramp bp_steps from 1 to full during training.

    HRM-Text uses bp_warmup_ratio=0.2 (20% of training ramps from bp_min to bp_max).
    This combines truncated BPTT with a schedule — start easy, gradually add depth.

    Needs: --bp_warmup_ratio, --bp_min_steps, --bp_max_steps args.
    """
    for task_name in MID_TASKS:
        module, base, _ = TASKS[task_name]
        for ratio in [0.1, 0.2, 0.5]:
            cfg = dict(with_seed(base, 0))
            cfg["bp_warmup_ratio"] = ratio
            cfg["bp_min_steps"] = 2
            cfg["bp_max_steps"] = min(10, base.get("iterations", 50))
            cfg["log_dir"] = f"logs/hrm_inspired/hr02/{task_name}_warmup{str(ratio).replace('.','p')}"
            plan.append(exp(
                f"hr02_{task_name}_warmup{str(ratio).replace('.','p')}",
                f"{task_name}: BP warmup ratio={ratio} (min=2, max={cfg['bp_max_steps']})",
                _p(module, cfg),
                tags=[task_name, "gradient-control", "bp-warmup"],
                impl_status="ready",
                hypothesis=f"Gradual gradient ramp ({ratio*100:.0f}% of training) "
                           f"improves optimization vs full BPTT from step 0.",
            ))
    return plan


def build_hr03_state_detach(plan):
    """State detach: cut gradient every K ticks (chunked BPTT).

    Different from truncated BPTT: instead of cutting at the end, we cut periodically.
    This allows gradient flow within chunks but not across them.
    """
    for task_name in MID_TASKS:
        module, base, _ = TASKS[task_name]
        for detach_every in [1, 2, 5, 10]:
            cfg = dict(with_seed(base, 0))
            cfg["detach_every"] = detach_every
            cfg["log_dir"] = f"logs/hrm_inspired/hr03/{task_name}_detach{detach_every}"
            plan.append(exp(
                f"hr03_{task_name}_detach{detach_every}",
                f"{task_name}: detach state every {detach_every} ticks",
                _p(module, cfg),
                tags=[task_name, "gradient-control", "state-detach"],
                impl_status="ready",
                hypothesis=f"Detaching every {detach_every} ticks trades gradient "
                           f"depth for memory; optimal chunk size unknown.",
            ))
    return plan


# ─── Phase B: Optimization ───────────────────────────────────


def build_hr04_adam_atan2(plan):
    """Adam-atan2 optimizer: use atan2(exp_avg, denom) instead of exp_avg/denom.

    HRM-Text uses this at 1B scale and finds it beneficial.
    Adam-atan2 bounds the update to [-lr, +lr] per step, improving stability.
    """
    for task_name in MID_TASKS:
        module, base, _ = TASKS[task_name]
        # adam_atan2 with default betas
        cfg = dict(with_seed(base, 0))
        cfg["optimizer_type"] = "adam_atan2"
        cfg["log_dir"] = f"logs/hrm_inspired/hr04/{task_name}_atan2"
        plan.append(exp(
            f"hr04_{task_name}_atan2",
            f"{task_name}: Adam-atan2 optimizer",
            _p(module, cfg),
            tags=[task_name, "optimizer", "adam-atan2"],
            impl_status="ready",
            hypothesis="atan2-based update improves stability over standard Adam.",
        ))
        # adam_atan2 with higher betas (as in HRM-Text: 0.9, 0.95)
        cfg2 = dict(with_seed(base, 0))
        cfg2["optimizer_type"] = "adam_atan2"
        cfg2["beta1"] = 0.9
        cfg2["beta2"] = 0.95
        cfg2["log_dir"] = f"logs/hrm_inspired/hr04/{task_name}_atan2_b95"
        plan.append(exp(
            f"hr04_{task_name}_atan2_b95",
            f"{task_name}: Adam-atan2, betas=(0.9, 0.95)",
            _p(module, cfg2),
            tags=[task_name, "optimizer", "adam-atan2"],
            impl_status="ready",
            hypothesis="HRM-Text betas (0.9, 0.95) + atan2 works well.",
        ))
    return plan


def build_hr05_stablemax_ce(plan):
    """Stablemax cross-entropy: alternative to softmax CE.

    HRM uses stablemax (s(x) = 1/(1-x) for x<0, x+1 for x>=0) which avoids
    overflow/underflow issues of softmax. Particularly relevant for CTM's
    multi-tick predictions where logits can be extreme.

    Needs: stablemax_cross_entropy function + --loss_type arg.
    """
    for task_name in MID_TASKS:
        module, base, _ = TASKS[task_name]
        cfg = dict(with_seed(base, 0))
        cfg["loss_type"] = "stablemax_ce"
        cfg["log_dir"] = f"logs/hrm_inspired/hr05/{task_name}_stablemax"
        plan.append(exp(
                f"hr05_{task_name}_stablemax",
                f"{task_name}: stablemax cross-entropy loss",
                _p(module, cfg),
                tags=[task_name, "loss", "stablemax-ce"],
                impl_status="ready",
                hypothesis="Stablemax CE is more numerically stable for CTM's "
                       "multi-tick predictions, potentially improving late-tick accuracy.",
        ))
    return plan


# ─── Phase C: Attention Improvements ─────────────────────────


def build_hr06_gated_attention(plan):
    """Gated attention: multiply attention output by sigmoid(gate).

    HRM-Text splits QKV projection into gate+query+key+value and applies
    sigmoid(gate) * attn_output. This gives the model a learnable per-head
    gate on attention, improving expressiveness.
    """
    for task_name in ["parity", "cifar10"]:
        module, base, _ = TASKS[task_name]
        cfg = dict(with_seed(base, 0))
        cfg["gated_attention"] = True
        cfg["log_dir"] = f"logs/hrm_inspired/hr06/{task_name}_gated_attn"
        plan.append(exp(
            f"hr06_{task_name}_gated_attn",
            f"{task_name}: gated attention (sigmoid gate on attn output)",
            _p(module, cfg),
            tags=[task_name, "attention", "gated"],
            impl_status="ready",
            hypothesis="Sigmoid gating on attention output gives CTM finer "
                       "control over information flow from data to latent state.",
        ))
    return plan


def build_hr07_input_injection(plan):
    """Input injection strategies: how attention output combines with latent state.

    CTM currently concatenates: pre_synapse = concat(attn_out, activated_state)
    HRM uses additive injection: z_L = L_level(z_L, z_H + input)
    Other options: GRU gating, fixed gating.

    Needs: --input_injection arg + forward loop variant.
    """
    injection_modes = [
        ("concat", "concat (baseline)"),
        ("additive", "additive (HRM-style)"),
        ("gru_gate", "GRU gate"),
    ]
    for task_name in ["parity"]:
        module, base, _ = TASKS[task_name]
        for mode, desc in injection_modes:
            if mode == "concat":
                continue  # baseline, already in hr00
            cfg = dict(with_seed(base, 0))
            cfg["input_injection"] = mode
            cfg["log_dir"] = f"logs/hrm_inspired/hr07/{task_name}_inj_{mode}"
            plan.append(exp(
                f"hr07_{task_name}_inj_{mode}",
                f"{task_name}: input injection = {desc}",
                _p(module, cfg),
                tags=[task_name, "attention", "injection", mode],
                impl_status="ready",
                hypothesis=f"{desc} injection may be more parameter-efficient "
                           f"than concat (saves d_input params in synapse input).",
            ))
    return plan


# ─── Phase D: Hierarchical Recurrence ────────────────────────
# The core HRM contribution: two interdependent recurrent modules (H × L).
# CTM has a flat loop of T iterations. This phase tests whether nesting helps.


def build_hr08_hierarchical(plan):
    """Hierarchical recurrence: nested H_cycles × L_cycles loop.

    HRM structure:
        for h in range(H_cycles):
            for l in range(L_cycles):
                z_L = L_step(z_L, z_H + attn_out)   # fast, detailed
            z_H = H_step(z_H, z_L)                   # slow, abstract

    CTM adaptation: split the flat tick loop into outer (H) and inner (L) loops.
    The H level uses a separate (deeper?) synapse model. After each L batch,
    H reads the L state and updates its own.

    iterations = H_cycles * L_cycles (kept constant for fair comparison).

    Needs: --h_cycles, --l_cycles args + major forward loop restructure.
    """
    configs = [
        # (h_cycles, l_cycles, description)
        (1, 50, "flat (control: H=1)"),
        (2, 25, "H=2 L=25"),
        (5, 10, "H=5 L=10"),
        (2, 2,  "H=2 L=2 (compact, like HRM)"),
    ]
    for task_name in FAST_TASKS:
        module, base, default_iters = TASKS[task_name]
        for h, l, desc in configs:
            total = h * l
            if total != default_iters:
                continue  # keep total iterations constant
            cfg = dict(with_seed(base, 0))
            cfg["h_cycles"] = h
            cfg["l_cycles"] = l
            cfg["log_dir"] = f"logs/hrm_inspired/hr08/{task_name}_h{h}_l{l}"
            impl = "ready"
            plan.append(exp(
                f"hr08_{task_name}_h{h}_l{l}",
                f"{task_name}: hierarchical H={h} L={l} ({desc})",
                _p(module, cfg),
                tags=[task_name, "hierarchical", f"h{h}", f"l{l}"],
                impl_status=impl,
                hypothesis=f"Hierarchical ({h}×{l}) vs flat ({total} ticks): "
                           f"does two-timescale processing help CTM?",
            ))
    # Also test with different iterations
    for task_name in FAST_TASKS:
        module, base, _ = TASKS[task_name]
        for h, l, desc in [(2, 2, "compact"), (2, 5, "medium"), (3, 3, "balanced")]:
            cfg = dict(with_seed(base, 0))
            cfg["h_cycles"] = h
            cfg["l_cycles"] = l
            cfg["iterations"] = h * l
            cfg["memory_length"] = max(2, (h * l) // 2)
            cfg["log_dir"] = f"logs/hrm_inspired/hr08/{task_name}_h{h}_l{l}_iter{h*l}"
            plan.append(exp(
                f"hr08_{task_name}_h{h}_l{l}_iter{h*l}",
                f"{task_name}: hierarchical H={h} L={l}, total={h*l} ticks",
                _p(module, cfg),
                tags=[task_name, "hierarchical", f"h{h}", f"l{l}", "short"],
                impl_status="ready",
                hypothesis=f"Compact hierarchical ({h}×{l}={h*l} ticks) may "
                           f"match flat ({base['iterations']} ticks) performance.",
            ))
    return plan


# ─── Phase E: Training Techniques ────────────────────────────


def build_hr09_ema_weights(plan):
    """EMA of model weights: maintain exponentially averaged weights for eval.

    HRM-Text uses ema=0.9999. EMA often improves final model quality and
    reduces noise in evaluation. Different from the existing EMA speed
    spectrum (which is for distillation).

    Needs: EMA tracking in optimizer + --ema_decay arg.
    """
    for task_name in MID_TASKS:
        module, base, _ = TASKS[task_name]
        for decay in [0.999, 0.9999]:
            cfg = dict(with_seed(base, 0))
            cfg["ema_decay"] = decay
            cfg["log_dir"] = f"logs/hrm_inspired/hr09/{task_name}_ema{str(decay).replace('.','p')}"
            plan.append(exp(
                f"hr09_{task_name}_ema{str(decay).replace('.','p')}",
                f"{task_name}: EMA weights decay={decay}",
                _p(module, cfg),
                tags=[task_name, "training", "ema"],
                impl_status="ready",
                hypothesis=f"EMA (decay={decay}) smooths weights, improving "
                           f"evaluation stability.",
            ))
    return plan


def build_hr10_act_halting(plan):
    """Adaptive Computation Time with Q-learning halting (HRM's ACT).

    HRM learns when to halt via a Q-head outputting (halt, continue) logits.
    Uses bootstrapped Q-learning (PQN-style) with no replay buffer.
    CTM already has certainty-based halting (threshold mode); this tests
    whether LEARNED halting via Q-learning is better.

    Needs: Q-head on synchronisation_out + Q-learning loss in training loop.
    """
    for task_name in FAST_TASKS:
        module, base, _ = TASKS[task_name]
        for halt_max in [16, 50]:
            for exploration in [0.05, 0.1]:
                cfg = dict(with_seed(base, 0))
                cfg["act_halt"] = True
                cfg["halt_max_steps"] = halt_max
                cfg["halt_exploration_prob"] = exploration
                cfg["log_dir"] = f"logs/hrm_inspired/hr10/{task_name}_act_h{halt_max}_e{str(exploration).replace('.','p')}"
                plan.append(exp(
                    f"hr10_{task_name}_act_h{halt_max}_e{str(exploration).replace('.','p')}",
                    f"{task_name}: ACT Q-learning halt, max={halt_max} explore={exploration}",
                    _p(module, cfg),
                    tags=[task_name, "act", "q-learning"],
                    impl_status="ready",
                    hypothesis=f"Q-learned halting (max={halt_max}, eps={exploration}) "
                               f"vs fixed iterations: does learned adaptive depth help?",
                ))
    return plan


# ─── Phase F: Cross-Idea Combinations ────────────────────────


def build_hr11_bptt_atan2(plan):
    """Truncated BPTT + Adam-atan2: the two simplest HRM wins combined."""
    for task_name in FAST_TASKS:
        module, base, _ = TASKS[task_name]
        cfg = dict(with_seed(base, 0))
        cfg["bp_steps"] = 5
        cfg["optimizer_type"] = "adam_atan2"
        cfg["log_dir"] = f"logs/hrm_inspired/hr11/{task_name}_bp5_atan2"
        plan.append(exp(
            f"hr11_{task_name}_bp5_atan2",
            f"{task_name}: truncated BPTT(5) + Adam-atan2",
            _p(module, cfg),
            tags=[task_name, "combo", "truncated-bptt", "adam-atan2"],
            impl_status="ready",
            hypothesis="Combining gradient truncation with atan2 optimizer "
                       "gives both speed and stability improvements.",
        ))
    return plan


def build_hr12_bptt_hierarchical(plan):
    """Truncated BPTT + Hierarchical: HRM uses both nocarry + H×L structure."""
    for task_name in FAST_TASKS:
        module, base, _ = TASKS[task_name]
        cfg = dict(with_seed(base, 0))
        cfg["h_cycles"] = 2
        cfg["l_cycles"] = 5
        cfg["iterations"] = 10
        cfg["bp_steps"] = 3
        cfg["memory_length"] = 5
        cfg["log_dir"] = f"logs/hrm_inspired/hr12/{task_name}_h2_l5_bp3"
        plan.append(exp(
            f"hr12_{task_name}_h2_l5_bp3",
            f"{task_name}: hierarchical(2×5) + BPTT(3)",
            _p(module, cfg),
            tags=[task_name, "combo", "hierarchical", "truncated-bptt"],
            impl_status="ready",
            hypothesis="HRM's core: hierarchical structure + truncated gradient. "
                       "Only backprop through last H cycle.",
        ))
    return plan


def build_hr13_hierarchical_gated(plan):
    """Hierarchical + Gated attention: structural + attention improvements."""
    for task_name in FAST_TASKS:
        module, base, _ = TASKS[task_name]
        cfg = dict(with_seed(base, 0))
        cfg["h_cycles"] = 2
        cfg["l_cycles"] = 5
        cfg["iterations"] = 10
        cfg["memory_length"] = 5
        cfg["gated_attention"] = True
        cfg["log_dir"] = f"logs/hrm_inspired/hr13/{task_name}_h2_l5_gated"
        plan.append(exp(
            f"hr13_{task_name}_h2_l5_gated",
            f"{task_name}: hierarchical(2×5) + gated attention",
            _p(module, cfg),
            tags=[task_name, "combo", "hierarchical", "gated"],
            impl_status="ready",
            hypothesis="Gated attention within hierarchical structure gives "
                       "finer control at each L step.",
        ))
    return plan


def build_hr14_bptt_atan2_stablemax(plan):
    """Triple combo: BPTT + atan2 + stablemax — all non-architectural improvements."""
    for task_name in FAST_TASKS:
        module, base, _ = TASKS[task_name]
        cfg = dict(with_seed(base, 0))
        cfg["bp_steps"] = 5
        cfg["optimizer_type"] = "adam_atan2"
        cfg["loss_type"] = "stablemax_ce"
        cfg["log_dir"] = f"logs/hrm_inspired/hr14/{task_name}_bp5_atan2_smax"
        plan.append(exp(
            f"hr14_{task_name}_bp5_atan2_smax",
            f"{task_name}: BPTT(5) + Adam-atan2 + stablemax CE",
            _p(module, cfg),
            tags=[task_name, "combo", "truncated-bptt", "adam-atan2", "stablemax-ce"],
            impl_status="ready",
            hypothesis="All three training-level improvements together.",
        ))
    return plan


def build_hr15_full_hrm_style(plan):
    """Full HRM-style CTM: hierarchical + BP warmup + atan2 + gated attention.

    The culmination: combine all winning ideas into one model.
    This is the closest analog to HRM's design philosophy applied to CTM.
    """
    for task_name in FAST_TASKS:
        module, base, _ = TASKS[task_name]
        cfg = dict(with_seed(base, 0))
        cfg["h_cycles"] = 2
        cfg["l_cycles"] = 5
        cfg["iterations"] = 10
        cfg["memory_length"] = 5
        cfg["bp_warmup_ratio"] = 0.2
        cfg["bp_min_steps"] = 2
        cfg["bp_max_steps"] = 5
        cfg["optimizer_type"] = "adam_atan2"
        cfg["gated_attention"] = True
        cfg["log_dir"] = f"logs/hrm_inspired/hr15/{task_name}_full_hrm"
        plan.append(exp(
            f"hr15_{task_name}_full_hrm",
            f"{task_name}: full HRM-style (H=2,L=5 + warmup + atan2 + gated)",
            _p(module, cfg),
            tags=[task_name, "full-combo", "hierarchical", "bp-warmup", "adam-atan2", "gated"],
            impl_status="ready",
            hypothesis="Full HRM philosophy applied to CTM: hierarchical structure "
                       "with gradient control, stable optimization, and gated attention.",
        ))
    return plan


# ─── Registry ───

STAGE_BUILDERS = {
    "hr00": build_hr00_baselines,
    "hr01": build_hr01_truncated_bptt,
    "hr02": build_hr02_bp_warmup,
    "hr03": build_hr03_state_detach,
    "hr04": build_hr04_adam_atan2,
    "hr05": build_hr05_stablemax_ce,
    "hr06": build_hr06_gated_attention,
    "hr07": build_hr07_input_injection,
    "hr08": build_hr08_hierarchical,
    "hr09": build_hr09_ema_weights,
    "hr10": build_hr10_act_halting,
    "hr11": build_hr11_bptt_atan2,
    "hr12": build_hr12_bptt_hierarchical,
    "hr13": build_hr13_hierarchical_gated,
    "hr14": build_hr14_bptt_atan2_stablemax,
    "hr15": build_hr15_full_hrm_style,
}

STAGE_DESCRIPTIONS = {
    "hr00": "Baselines (control group for comparison)",
    "hr01": "Truncated BPTT: only backprop last N ticks [Phase A: Gradient Control]",
    "hr02": "BP warmup: ramp gradient depth during training [Phase A]",
    "hr03": "State detach: chunked BPTT, cut gradient every K ticks [Phase A]",
    "hr04": "Adam-atan2 optimizer (from HRM-Text) [Phase B: Optimization]",
    "hr05": "Stablemax cross-entropy loss (from HRM) [Phase B]",
    "hr06": "Gated attention: sigmoid gate on attn output [Phase C: Attention]",
    "hr07": "Input injection: additive/GRU vs concat [Phase C]",
    "hr08": "Hierarchical recurrence: nested H×L loops (HRM's core) [Phase D: Structure]",
    "hr09": "EMA model weights (from HRM-Text) [Phase E: Training]",
    "hr10": "ACT Q-learning halting (from HRM) [Phase E]",
    "hr11": "Combo: truncated BPTT + Adam-atan2 [Phase F: Combos]",
    "hr12": "Combo: hierarchical + truncated BPTT [Phase F]",
    "hr13": "Combo: hierarchical + gated attention [Phase F]",
    "hr14": "Combo: BPTT + atan2 + stablemax (training-only) [Phase F]",
    "hr15": "Full HRM-style CTM: all winning ideas combined [Phase F]",
}

# Implementation priority order (what to implement first)
IMPL_PRIORITY = {
    "hr01": 1,   # truncated BPTT — simplest, highest impact
    "hr04": 2,   # Adam-atan2 — trivial to implement
    "hr03": 3,   # state detach — simple
    "hr05": 4,   # stablemax CE — simple
    "hr02": 5,   # BP warmup — builds on truncated BPTT
    "hr06": 6,   # gated attention — small model change
    "hr07": 7,   # input injection — small model change
    "hr08": 8,   # hierarchical — major model change
    "hr09": 9,   # EMA — training loop change
    "hr10": 10,  # ACT halting — major training change
}


def build_plan(stage="all"):
    plan = []
    if stage == "all":
        for s in STAGES_ORDERED:
            if s in STAGE_BUILDERS:
                STAGE_BUILDERS[s](plan)
    elif stage in STAGE_BUILDERS:
        STAGE_BUILDERS[s](plan)
    else:
        print(f"Unknown stage: {stage}. Available: {STAGES_ORDERED}")
        return []
    return plan


def count_by_status(plan):
    ready = sum(1 for e in plan if e.get("impl_status") == "ready")
    needs_impl = sum(1 for e in plan if e.get("impl_status") != "ready")
    return ready, needs_impl


def print_plan(plan):
    ready, needs_impl = count_by_status(plan)
    print(f"\n{'='*80}")
    print(f"HRM-INSPIRED CTM EXPERIMENT PLAN — {len(plan)} experiments total")
    print(f"  Ready: {ready}  |  Needs implementation: {needs_impl}")
    print(f"{'='*80}\n")

    stages = {}
    for e in plan:
        prefix = e["name"].split("_")[0]
        stages.setdefault(prefix, []).append(e)

    for stage_name in STAGES_ORDERED:
        exps = stages.get(stage_name, [])
        if not exps:
            continue
        desc = STAGE_DESCRIPTIONS.get(stage_name, "")
        r = sum(1 for e in exps if e.get("impl_status") == "ready")
        ni = len(exps) - r
        priority = IMPL_PRIORITY.get(stage_name, "?")
        status = f"[{r} ready, {ni} needs_impl, priority={priority}]"
        print(f"\n  ── {stage_name}: {desc} {status} ──")
        for e in exps:
            impl_mark = " \u26a0" if e.get("impl_status") != "ready" else " \u2713"
            print(f"     {e['name']}{impl_mark}")
            print(f"       Q: {e['question']}")
            if "hypothesis" in e:
                print(f"       H: {e['hypothesis']}")

    print(f"\n{'='*80}")
    print(f"TOTAL: {len(plan)} ({ready} ready, {needs_impl} needs implementation)")
    print(f"\nImplementation priority order:")
    for sname in sorted(IMPL_PRIORITY, key=lambda x: IMPL_PRIORITY[x]):
        if sname in stages:
            print(f"  {IMPL_PRIORITY[sname]}. {sname}: {STAGE_DESCRIPTIONS[sname]}")
    print(f"{'='*80}\n")


# ─── Pool submission (same infrastructure as paper plan) ───

def submit_to_pool(exp_entry, config, master_addr=None, port=None):
    node_addrs = [exp_entry.get("node_addr")]
    payload = {
        "config": config,
        "extra_args": exp_entry["command"],
        "node_addrs": node_addrs,
        "env": {
            "CTM_EXPERIMENT_NAME": exp_entry["name"],
            "CTM_METRICS_DIR": "runs/metrics",
            "CTM_LOG_DIR": "runs/logs/hrm_inspired",
        },
    }
    base = f"http://{master_addr}:{port}"
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base}/submit",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        resp = opener.open(req, timeout=10)
        result = json.loads(resp.read())
        return result.get("task") or result
    except Exception as e:
        print(f"[submit] error: {e}")
        return None


def wait_until_idle(master_addr, port, task_id, poll_interval=30.0):
    base = f"http://{master_addr}:{port}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    final = {"completed", "failed", "cancelled"}
    while True:
        try:
            resp = opener.open(f"{base}/status", timeout=10)
            status = json.loads(resp.read())
            tasks = status.get("tasks", [])
            for t in tasks:
                if t["task_id"] == task_id and t["status"] in final:
                    print(f"  [pool] task {task_id} -> {t['status']}")
                    return t["status"]
        except Exception:
            pass
        time.sleep(poll_interval)


def cmd_plan(args):
    plan = build_plan(args.stage)
    print_plan(plan)


def cmd_submit(args):
    plan = build_plan(args.stage)
    if args.ready_only:
        to_submit = [e for e in plan if e.get("impl_status") == "ready"]
    else:
        to_submit = plan

    if not to_submit:
        print("No experiments to submit.")
        return

    print(f"Submitting {len(to_submit)} experiments to pool at {MASTER_ADDR}:{PORT}")
    for e in to_submit:
        status = "ready" if e.get("impl_status") == "ready" else "NEEDS_IMPL"
        print(f"  [{status}] {e['name']}")

    if args.dry_run:
        return

    for e in to_submit:
        print(f"Submitting {e['name']}...")
        result = submit_to_pool(e, POOL_CONFIG, MASTER_ADDR, PORT)
        if result is None:
            print(f"  FAILED to submit {e['name']}")
            continue
        task_id = result if isinstance(result, str) else result.get("task_id", "")
        print(f"  task_id={task_id}")
        if args.wait:
            final_status = wait_until_idle(MASTER_ADDR, PORT, task_id)
            print(f"  -> {final_status}")
            if final_status == "failed" and args.stop_on_fail:
                print("Stopping due to failure.")
                break
            time.sleep(5)


def cmd_csv(args):
    import csv
    plan = build_plan(args.stage)
    path = args.output or f"runs/experiment_plans/hrm_inspired_plan.csv"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "question", "hypothesis", "command", "tags", "impl_status"])
        w.writeheader()
        for e in plan:
            w.writerow({
                "name": e["name"],
                "question": e["question"],
                "hypothesis": e.get("hypothesis", ""),
                "command": e["command"],
                "tags": ";".join(e["tags"]),
                "impl_status": e.get("impl_status", "needs_impl"),
            })
    print(f"Wrote {len(plan)} experiments to {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="HRM-Inspired CTM Experiments")
    sub = parser.add_subparsers(dest="command")

    p_plan = sub.add_parser("plan", help="Show experiment plan")
    p_plan.add_argument("--stage", default="all")

    p_submit = sub.add_parser("submit", help="Submit experiments to pool")
    p_submit.add_argument("--stage", default="all")
    p_submit.add_argument("--dry-run", action="store_true")
    p_submit.add_argument("--wait", action="store_true", default=True)
    p_submit.add_argument("--no-wait", action="store_false", dest="wait")
    p_submit.add_argument("--stop-on-fail", action="store_true", default=True)
    p_submit.add_argument("--ready-only", action="store_true", default=True,
                          help="Only submit experiments marked as ready")
    p_submit.add_argument("--include-needs-impl", action="store_false", dest="ready_only",
                          help="Also submit experiments that need implementation")
    p_submit.add_argument("--master-addr", default=MASTER_ADDR)
    p_submit.add_argument("--port", type=int, default=PORT)

    p_csv = sub.add_parser("csv", help="Export plan to CSV")
    p_csv.add_argument("--stage", default="all")
    p_csv.add_argument("--output", default=None)

    args = parser.parse_args()
    if args.command == "plan":
        cmd_plan(args)
    elif args.command == "submit":
        cmd_submit(args)
    elif args.command == "csv":
        cmd_csv(args)
    else:
        parser.print_help()
