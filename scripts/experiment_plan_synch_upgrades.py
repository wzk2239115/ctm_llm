#!/usr/bin/env python3
"""
CTM Synch-Upgrade Experiment Plan — Replacing Fixed Subspace Selection.

Motivation
----------
CTM's `compute_synchronisation` (baseline/models/ctm.py:214) selects a FIXED set
of neuron pairs at __init__ via `register_buffer`, then never moves them. Only
the per-pair decay rates (`decay_params_{action,out}`) are learnable. This is
brittle: the random init may pick a poor subspace and the network can never
escape it. This script enumerates 7 families of upgrades from the literature:

  1. Soft / approximate-discrete selection  (su01-su04)
  2. Dynamic & adaptive mapping              (su05-su07)
  3. Orthogonal & disentanglement constraints(su08-su09)
  4. Brute-force search (RL)                 (su10)
  5. Cross-family combos                     (su11-su14)

ALL upgrade stages are impl_status="needs_impl" — they require new CLI flags
and corresponding code in baseline/models/ctm.py:compute_synchronisation.
Each stage docstring lists exactly what to add. su00 (baseline reproduction
of the three fixed modes) is the only "ready" stage.

Usage:
    python scripts/experiment_plan_synch_upgrades.py plan [--stage all]
    python scripts/experiment_plan_synch_upgrades.py plan --stage su01
    python scripts/experiment_plan_synch_upgrades.py submit --stage su01 --no-wait
    python scripts/experiment_plan_synch_upgrades.py csv
"""

import json, os, re, shlex, subprocess, sys, time, urllib.request
import _pool_config
MASTER_ADDR = _pool_config.MASTER_ADDR
PORT = _pool_config.PORT
BASELINE_NODES = _pool_config.BASELINE_NODES
POOL_CONFIG = _pool_config.POOL_CONFIG
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GPUS_PER_NODE = 8
_slot_idx = [0]

STAGES_ORDERED = [
    "su00",                                                              # baseline
    "su01", "su02", "su03", "su04",                                      # soft / discrete
    "su05", "su06", "su07",                                              # dynamic
    "su08", "su09",                                                      # orthogonal
    "su10",                                                              # RL search
    "su11", "su12", "su13", "su14",                                      # combos
]
ALL_STAGES = STAGES_ORDERED + ["all"]

NEEDS_IMPL = "needs_impl"
READY = "ready"


def _next_slot():
    i = _slot_idx[0]
    _slot_idx[0] += 1
    if not BASELINE_NODES:
        return None
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


def exp(name, question, command, tags=None, impl_status=NEEDS_IMPL):
    return {
        "name": name,
        "question": question,
        "command": command,
        "tags": tags or [],
        "node_addr": _next_slot(),
        "impl_status": impl_status,
    }


# ─── Per-task base configs (mirror ctm_paper; kept self-contained) ───

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
    "sort":    ("baseline.tasks.sort.train",                SORT_BASE),
    "parity":  ("baseline.tasks.parity.train",              PARITY_BASE),
    "mazes":   ("baseline.tasks.mazes.train",               MAZES_BASE),
    "cifar10": ("baseline.tasks.image_classification.train", CIFAR10_BASE),
    "qamnist": ("baseline.tasks.qamnist.train",             QAMNIST_BASE),
}

SEEDS = [0, 1, 2]

# Tasks where compute-heavy upgrades (RL search, dynamic routing on long
# sequences) are tractable enough to actually finish.
LIGHT_TASKS = ("sort", "parity", "cifar10")


def with_seed(cfg, seed):
    c = dict(cfg)
    c["seed"] = seed
    return c


def tagged(plan, tag, exp_list):
    for e in exp_list:
        e["tags"].append(tag)
    plan.extend(exp_list)


# ═══════════════════════════════════════════════════════════════
# STAGE BUILDERS
# ═══════════════════════════════════════════════════════════════

def build_su00_baseline_fixed_modes(plan):
    """Reproduce the 3 fixed neuron_select_type modes as comparison floor.

    READY: uses only existing CLI flags (--neuron_select_type). These runs
    are the single-variable baseline for every other stage in this plan.
    If st01 nst ablations in ctm_paper.py already cover this for the same
    configs, these runs may be skipped — but we keep them here so the
    synch-upgrade track has a self-contained reference set.
    """
    for task_name, (module, base) in TASKS.items():
        for nst in ("first-last", "random", "random-pairing"):
            cfg = dict(with_seed(base, 0))
            cfg["neuron_select_type"] = nst
            if nst == "random-pairing":
                cfg["n_random_pairing_self"] = 0
            # first-last requires d_model >= n_synch_out + n_synch_action
            if nst == "first-last":
                d_model = cfg.get("d_model", 512)
                n_out = cfg.get("n_synch_out", 32)
                n_act = cfg.get("n_synch_action", 32)
                if d_model < n_out + n_act:
                    continue
            # random / first-last produce O(n_synch^2) sync size — skip if huge
            if nst in ("random", "first-last"):
                n_out = cfg.get("n_synch_out", 32)
                n_act = cfg.get("n_synch_action", 32)
                if max(n_out, n_act) > 200:
                    continue
            cfg["log_dir"] = f"logs/synch_upgrades/su00/{task_name}_nst_{nst}"
            plan.append(exp(
                f"su00_{task_name}_nst_{nst}",
                f"{task_name}: fixed neuron_select={nst} (baseline)",
                _p(module, cfg),
                tags=[task_name, "baseline-fixed", "nst"],
                impl_status=READY,
            ))
    return plan


# ─── Category 1: Soft / Approximate-Discrete Selection ─────────────

def build_su01_soft_gating(plan):
    """Softmax-based learnable gating (foundational upgrade).

    REPLACES: activated_state[:, idx_left/right] with
        w = softmax(gate_left / temp); s_left = (s @ w.T)
    where gate_left ∈ R^{K × D} is nn.Parameter. Fully differentiable, drop-in.

    IMPLEMENTED (2026-06-17):
        - baseline/models/ctm.py:
          * __init__ now accepts synch_gate_mode='fixed', synch_gate_temp=1.0
          * set_synchronisation_parameters() branches: 'fixed' registers
            neuron_indices buffers (legacy); 'soft' registers gate_{left,right}
            Parameters initialized to SOFT_INIT_SCALE=5.0 one-hot at the
            sampled indices (so model starts as softened random-pairing).
          * compute_synchronisation() branches: 'soft' path computes
            w = softmax(gate/temp, dim=-1); s_left = activated_state @ w.T
          * calculate_synch_representation_size() returns n_synch for non-fixed
            modes (random-pairing semantics).
          * verify_args() only enforces neuron_select_type when mode='fixed'.
        - baseline/models/ctm_sort.py, ctm_qamnist.py: pass new args through super().
        - baseline/tasks/{sort,parity,qamnist,mazes,image_classification}/train.py:
          register --synch_gate_mode {fixed,soft} and --synch_gate_temp CLI flags;
          pass them through to model __init__.
        - baseline/tasks/parity/utils.py, baseline/tasks/qamnist/utils.py:
          prepare_model() passes the new args via getattr (backward compatible).

    TESTED:
        - Structural: soft model has 4 gate params (out_gate_{left,right},
          action_gate_{left,right}), shape [K, D], requires_grad=True.
        - compute_synchronisation('soft') produces correct shapes [B, K].
        - Gradient flow confirmed to gate_left, gate_right, decay_params_out
          (decay only flows on tick >= 2, by design — first tick has decay_alpha=None).
        - Temperature sweep verified: temp=1.0 → top-1 weight 0.70, temp=0.1 → 1.00
          (degrades gracefully to hard indexing at low temp).
        - state_dict has orthogonal keys vs fixed mode (no silent compatibility).
        - CLI argparse: 'fixed'/'soft' accepted; 'gumbel' correctly rejected.

    CLI ARGS (all 5 tasks):
        --synch_gate_mode {fixed,soft}   (default fixed)
        --synch_gate_temp float          (default 1.0; only consulted when mode=soft)
    """
    for task_name, (module, base) in TASKS.items():
        for temp in (1.0, 0.5, 0.1):
            cfg = dict(with_seed(base, 0))
            cfg["synch_gate_mode"] = "soft"
            cfg["synch_gate_temp"] = temp
            cfg["log_dir"] = f"logs/synch_upgrades/su01/{task_name}_soft_t{temp}"
            plan.append(exp(
                f"su01_{task_name}_soft_t{str(temp).replace('.','p')}",
                f"{task_name}: soft gate temp={temp}",
                _p(module, cfg),
                tags=[task_name, "soft-gating", f"temp{temp}"],
                impl_status=READY,
            ))
    return plan


def build_su02_gumbel_softmax(plan):
    """Gumbel-Softmax reparameterization with temperature annealing.

    TRAINING: w = gumbel_softmax(gate / tau, tau=tau, hard=False)
    INFERENCE: w = one_hot(argmax(gate))  (truly discrete, matches CTM)
    Anneal tau: temp_init → temp_min over training.

    MODEL CODE NEEDED:
        - synch_gate_mode == "gumbel"
        - Add self.gate_tau, self.gate_temp_init, self.gate_temp_min, schedule
        - Update tau each step based on global_step and schedule
        - In forward: if training: gumbel_softmax(...) else one_hot(argmax)

    CLI ARGS NEEDED:
        --synch_gate_temp_init float (default 1.0)
        --synch_gate_temp_min  float (default 0.1)
        --synch_gate_temp_schedule {linear,cosine,exp}
    """
    schedules = [("linear", 0.1), ("cosine", 0.1), ("exp", 0.01), ("linear", 0.01)]
    for task_name, (module, base) in TASKS.items():
        for sched, t_min in schedules:
            cfg = dict(with_seed(base, 0))
            cfg["synch_gate_mode"] = "gumbel"
            cfg["synch_gate_temp_init"] = 1.0
            cfg["synch_gate_temp_min"] = t_min
            cfg["synch_gate_temp_schedule"] = sched
            cfg["log_dir"] = f"logs/synch_upgrades/su02/{task_name}_gumbel_{sched}_tmin{t_min}"
            plan.append(exp(
                f"su02_{task_name}_gumbel_{sched}_tmin{str(t_min).replace('.','p')}",
                f"{task_name}: Gumbel schedule={sched} t_min={t_min}",
                _p(module, cfg),
                tags=[task_name, "gumbel", f" sched-{sched}", f"tmin{t_min}"],
            ))
    return plan


def build_su03_sparsemax_entmax(plan):
    """Sparsemax / entmax — strictly-zero sparse selection.

    sparsemax maps logits to a projection onto the probability simplex,
    which can produce exact zeros (vs softmax's long tail).
    entmax(α=1.5) interpolates between sparsemax (α=1) and softmax (α=2).

    MODEL CODE NEEDED:
        - synch_gate_mode == "sparsemax" or "entmax"
        - Sparsemax impl (20 lines, Euclidean projection onto simplex)
        - Entmax impl (iterative bisection, ~30 lines)
        - Replace `softmax(gate/temp)` with the chosen transformation

    CLI ARGS NEEDED:
        --synch_gate_entmax_alpha float (default 1.5; ignored unless mode=entmax)

    REFERENCE: Martins & Astudillo 2016 (sparsemax), Peters et al. 2019 (entmax).
    """
    variants = [("sparsemax", None), ("entmax", 1.5), ("entmax", 1.75)]
    for task_name, (module, base) in TASKS.items():
        for mode, alpha in variants:
            cfg = dict(with_seed(base, 0))
            cfg["synch_gate_mode"] = mode
            if alpha is not None:
                cfg["synch_gate_entmax_alpha"] = alpha
            tag = mode if alpha is None else f"{mode}_a{str(alpha).replace('.','p')}"
            cfg["log_dir"] = f"logs/synch_upgrades/su03/{task_name}_{tag}"
            plan.append(exp(
                f"su03_{task_name}_{tag}",
                f"{task_name}: {mode}" + (f" α={alpha}" if alpha else ""),
                _p(module, cfg),
                tags=[task_name, "sparse-select", mode] + ([f"alpha{alpha}"] if alpha else []),
            ))
    return plan


def build_su04_topk_gating(plan):
    """Top-K sparse gating (Sparse-MoE style).

    score = softmax(gate(state)) ; keep top-K entries, zero the rest.
    Computational win: only K rows of pairwise_product actually computed.
    Representation win: input-dependent dynamic subspace, exactly K-active.

    MODEL CODE NEEDED:
        - synch_gate_mode == "topk"
        - Add self.gate_proj = nn.Linear(D, D) producing per-neuron scores
        - In forward:
            scores = softmax(self.gate_proj(s), dim=-1)        # [B, D]
            topk_val, topk_idx = scores.topk(self.synch_topk_k, dim=-1)
            # gather pairwise products only for selected neurons
        - Loss: load-balance auxiliary loss (similar to GShard) optional

    CLI ARGS NEEDED:
        --synch_topk_k int (number of active neurons per tick)
    """
    for task_name, (module, base) in TASKS.items():
        for k in (2, 4, 8, 16):
            cfg = dict(with_seed(base, 0))
            cfg["synch_gate_mode"] = "topk"
            cfg["synch_topk_k"] = k
            cfg["log_dir"] = f"logs/synch_upgrades/su04/{task_name}_topk_k{k}"
            plan.append(exp(
                f"su04_{task_name}_topk_k{k}",
                f"{task_name}: top-K sparse K={k}",
                _p(module, cfg),
                tags=[task_name, "topk-moe", f"k{k}"],
            ))
    return plan


# ─── Category 2: Dynamic / Adaptive Mapping ────────────────────────

def build_su05_dynamic_routing(plan):
    """Capsule-style routing-by-agreement.

    Instead of fixed pairs, low-layer "votes" route themselves to higher
    subspaces by iterative agreement:
        b_ij ← 0
        for r in range(routing_iters):
            c_ij = softmax(b_ij)
            s_j  = Σ_i c_ij * vote_ij
            v_j  = squash(s_j)
            b_ij += Σ_i vote_ij · v_j
    Strong inductive bias for part-whole decomposition.

    MODEL CODE NEEDED:
        - synch_gate_mode == "dynroute"
        - Add routing iteration loop INSIDE compute_synchronisation
        - vote matrix is the bilinear outer-product (already computed)
        - Squash function: v / (||v|| + eps) * ||v||^2 / (1 + ||v||^2)

    CLI ARGS NEEDED:
        --synch_routing_iters int (default 3)

    REFERENCE: Sabour et al. 2017 (Dynamic Routing Between Capsules).
    Skip qamnist — too expensive for the answer-step loop.
    """
    for task_name, (module, base) in TASKS.items():
        if task_name == "qamnist":
            continue
        for iters in (1, 3, 5):
            cfg = dict(with_seed(base, 0))
            cfg["synch_gate_mode"] = "dynroute"
            cfg["synch_routing_iters"] = iters
            cfg["log_dir"] = f"logs/synch_upgrades/su05/{task_name}_dynroute_r{iters}"
            plan.append(exp(
                f"su05_{task_name}_dynroute_r{iters}",
                f"{task_name}: dynamic routing iters={iters}",
                _p(module, cfg),
                tags=[task_name, "dynroute", f"r{iters}"],
            ))
    return plan


def build_su06_attention_pairing(plan):
    """Self-attention-based adaptive neuron pairing.

    Each "pair slot" k is a learnable query that attends to all neurons:
        Q = W_q(s)              # [B, K, dk]
        K = W_k(s)              # [B, D, dk]
        attn = softmax(Q K^T / √dk)    # [B, K, D] — soft, input-dependent
        s_left_k  = Σ_d attn[k,d] · s[d]
        s_right_k = s_left_k (self-pair) or second attention head
    More expressive than fixed / soft gating; parameter cost O(D·K).

    MODEL CODE NEEDED:
        - synch_gate_mode == "attn"
        - self.q_proj = nn.Linear(D, K * dk), self.k_proj = nn.Linear(D, dk)
        - reshape, bmm, softmax, gather in compute_synchronisation

    CLI ARGS NEEDED:
        --synch_attn_heads int (default 4)
        --synch_attn_dk int (default 32)
    """
    for task_name, (module, base) in TASKS.items():
        for heads in (4, 8, 16):
            cfg = dict(with_seed(base, 0))
            cfg["synch_gate_mode"] = "attn"
            cfg["synch_attn_heads"] = heads
            cfg["log_dir"] = f"logs/synch_upgrades/su06/{task_name}_attn_h{heads}"
            plan.append(exp(
                f"su06_{task_name}_attn_h{heads}",
                f"{task_name}: attention pairing heads={heads}",
                _p(module, cfg),
                tags=[task_name, "attn-pairing", f"h{heads}"],
            ))
    return plan


def build_su07_hypernet(plan):
    """Hypernetwork generates subspace weights from a context vector.

    Instead of fixed `gate_left/right` parameters, a small network
    `hyper: context → gate_left, gate_right` produces the selection
    weights on-the-fly. Context can be a learnable embedding (static
    hypernet) or a learned summary of the input (dynamic hypernet).

    MODEL CODE NEEDED:
        - synch_gate_mode == "hypernet"
        - self.hyper_context = nn.Parameter(randn(ctx_dim))  # learnable context
        - self.hyper = nn.Linear(ctx_dim, 2 * K * D)
        - In forward: gate = self.hyper(self.hyper_context).chunk(2)
        - Then same as soft gating: w = softmax(gate); s @ w.T

    CLI ARGS NEEDED:
        --synch_hypernet_dim int (default 64, the ctx_dim)

    REFERENCE: Ha et al. 2016 (HyperNetworks).
    """
    for task_name, (module, base) in TASKS.items():
        for hdim in (32, 128):
            cfg = dict(with_seed(base, 0))
            cfg["synch_gate_mode"] = "hypernet"
            cfg["synch_hypernet_dim"] = hdim
            cfg["log_dir"] = f"logs/synch_upgrades/su07/{task_name}_hypernet_d{hdim}"
            plan.append(exp(
                f"su07_{task_name}_hypernet_d{hdim}",
                f"{task_name}: hypernet ctx_dim={hdim}",
                _p(module, cfg),
                tags=[task_name, "hypernet", f"d{hdim}"],
            ))
    return plan


# ─── Category 3: Orthogonal / Disentanglement Constraints ──────────

def build_su08_orthogonal_reg(plan):
    """Orthogonality penalty on the gate columns.

    Adds L = λ * ||G^T G - I||_F^2 to the loss, where G is the [D, K]
    selection matrix (gate weights). Forces every pair-slot to occupy
    a distinct direction in neuron space — prevents collapse where all
    K slots look at the same few neurons.

    Can be combined with any non-RL gating mode; here we apply on top
    of the default fixed-random baseline to isolate the regularizer's
    effect on the existing architecture.

    MODEL CODE NEEDED:
        - synch_orthogonal_weight: float (default 0.0)
        - In training step: add λ * frobenius(gate_left.T @ gate_left - I_K)
          to total loss (and same for gate_right)
        - For fixed-index modes, build a soft one-hot G from indices first

    CLI ARGS NEEDED:
        --synch_orthogonal_weight float (default 0.0)
        --synch_orthogonal_mode {penalty,svd} (default penalty)
    """
    for task_name, (module, base) in TASKS.items():
        for w in (0.01, 0.1, 1.0):
            cfg = dict(with_seed(base, 0))
            cfg["synch_orthogonal_weight"] = w
            cfg["synch_orthogonal_mode"] = "penalty"
            cfg["log_dir"] = f"logs/synch_upgrades/su08/{task_name}_ortho_w{w}"
            plan.append(exp(
                f"su08_{task_name}_ortho_w{str(w).replace('.','p')}",
                f"{task_name}: orthogonal penalty λ={w}",
                _p(module, cfg),
                tags=[task_name, "orthogonal", f"w{w}"],
            ))
    return plan


def build_su09_svd_orthogonal(plan):
    """SVD-based hard orthogonalization (vs soft penalty in su08).

    Every N steps, decompose the gate matrix G = U Σ V^T and replace
    G ← U V^T (Σ removed → columns become exactly orthonormal).
    More aggressive than penalty; can hurt early-training plasticity
    if applied too frequently.

    MODEL CODE NEEDED:
        - synch_orthogonal_mode == "svd"
        - synch_svd_freq: apply SVD every N steps (1 = always, 10 = decimate)
        - In training step (under no_grad): svd, reassign gate parameter data

    CLI ARGS NEEDED:
        --synch_svd_freq int (default 1)
    """
    for task_name, (module, base) in TASKS.items():
        for freq in (1, 10):
            cfg = dict(with_seed(base, 0))
            cfg["synch_orthogonal_mode"] = "svd"
            cfg["synch_orthogonal_weight"] = 1.0  # mode flag, magnitude irrelevant for SVD
            cfg["synch_svd_freq"] = freq
            cfg["log_dir"] = f"logs/synch_upgrades/su09/{task_name}_svd_f{freq}"
            plan.append(exp(
                f"su09_{task_name}_svd_f{freq}",
                f"{task_name}: SVD orthogonalize every {freq} steps",
                _p(module, cfg),
                tags=[task_name, "svd", f"f{freq}"],
            ))
    return plan


# ─── Category 4: Brute-Force Search (RL) ───────────────────────────

def build_su10_rl_search(plan):
    """REINFORCE-learned discrete subspace selection.

    Treat "which K neurons to pair" as a sequence of K discrete choices.
    Policy: small MLP over a running embedding of the current state.
    Reward: validation accuracy improvement over a fixed-window baseline.

    VERY expensive (multiple rollouts per step) — proof-of-concept on
    parity only. Skipped from `--stage all` by default; run explicitly.

    MODEL CODE NEEDED:
        - synch_gate_mode == "rl_reinforce"
        - Policy network, replay buffer, baseline network
        - Reward shaping: per-episode validation accuracy delta
        - Save: discrete indices learned → can be loaded back as fixed
          indices for the regular training loop

    CLI ARGS NEEDED:
        --synch_rl_mode {reinforce}
        --synch_rl_rollouts int (default 4)
        --synch_rl_baseline_decay float (default 0.9)
    """
    # Only parity (lightest task with discrete accuracy signal)
    module, base = TASKS["parity"]
    for rollouts in (2, 4):
        cfg = dict(with_seed(base, 0))
        cfg["synch_gate_mode"] = "rl_reinforce"
        cfg["synch_rl_rollouts"] = rollouts
        cfg["training_iterations"] = 50000  # reduced — RL is slow
        cfg["log_dir"] = f"logs/synch_upgrades/su10/parity_rl_r{rollouts}"
        plan.append(exp(
            f"su10_parity_rl_r{rollouts}",
            f"parity: RL subspace search rollouts={rollouts} (POC)",
            _p(module, cfg),
            tags=["parity", "rl-search", f"r{rollouts}"],
        ))
    return plan


# ═══════════════════════════════════════════════════════════════
# CROSS-FAMILY COMBOS
# ═══════════════════════════════════════════════════════════════

def build_su11_soft_orthogonal(plan):
    """Soft gating + orthogonal penalty (category 1 + 3).

    Best-of-both-worlds combo: differentiable selection (su01) with
    disentanglement constraint (su08). Picks su01's strongest temp
    (0.5) and su08's mid penalty (0.1) as defaults.
    """
    for task_name, (module, base) in TASKS.items():
        cfg = dict(with_seed(base, 0))
        cfg["synch_gate_mode"] = "soft"
        cfg["synch_gate_temp"] = 0.5
        cfg["synch_orthogonal_weight"] = 0.1
        cfg["synch_orthogonal_mode"] = "penalty"
        cfg["log_dir"] = f"logs/synch_upgrades/su11/{task_name}_soft_ortho"
        plan.append(exp(
            f"su11_{task_name}_soft_ortho",
            f"{task_name}: soft(t=0.5) + ortho(λ=0.1)",
            _p(module, cfg),
            tags=[task_name, "soft-gating", "orthogonal", "combo"],
        ))
    return plan


def build_su12_topk_orthogonal(plan):
    """Top-K sparse gating + orthogonal penalty.

    Tests whether explicit disentanglement helps MoE-style routing.
    Mid K (8) + mid penalty (0.1) as defaults.
    """
    for task_name, (module, base) in TASKS.items():
        cfg = dict(with_seed(base, 0))
        cfg["synch_gate_mode"] = "topk"
        cfg["synch_topk_k"] = 8
        cfg["synch_orthogonal_weight"] = 0.1
        cfg["synch_orthogonal_mode"] = "penalty"
        cfg["log_dir"] = f"logs/synch_upgrades/su12/{task_name}_topk_ortho"
        plan.append(exp(
            f"su12_{task_name}_topk_ortho",
            f"{task_name}: top-K(k=8) + ortho(λ=0.1)",
            _p(module, cfg),
            tags=[task_name, "topk-moe", "orthogonal", "combo"],
        ))
    return plan


def build_su13_gumbel_topk(plan):
    """Gumbel-Softmax + Top-K (discrete + sparse combo).

    Anneals Gumbel temperature to near-zero AND enforces top-K mask
    on top of Gumbel samples. Yields exactly-K-active discrete routing
    with full differentiability during training.

    Defaults: linear anneal 1.0 → 0.1, K = 8.
    """
    for task_name, (module, base) in TASKS.items():
        cfg = dict(with_seed(base, 0))
        cfg["synch_gate_mode"] = "gumbel_topk"
        cfg["synch_gate_temp_init"] = 1.0
        cfg["synch_gate_temp_min"] = 0.1
        cfg["synch_gate_temp_schedule"] = "linear"
        cfg["synch_topk_k"] = 8
        cfg["log_dir"] = f"logs/synch_upgrades/su13/{task_name}_gumbel_topk"
        plan.append(exp(
            f"su13_{task_name}_gumbel_topk",
            f"{task_name}: Gumbel(1.0→0.1) + top-K(k=8)",
            _p(module, cfg),
            tags=[task_name, "gumbel", "topk-moe", "combo"],
        ))
    return plan


def build_su14_attn_orthogonal(plan):
    """Attention pairing + orthogonal penalty.

    Tests whether disentangling attention heads (via orthogonal penalty
    on per-head query vectors) improves the input-dependent pairing.
    Defaults: 8 heads + 0.1 penalty.
    """
    for task_name, (module, base) in TASKS.items():
        cfg = dict(with_seed(base, 0))
        cfg["synch_gate_mode"] = "attn"
        cfg["synch_attn_heads"] = 8
        cfg["synch_orthogonal_weight"] = 0.1
        cfg["synch_orthogonal_mode"] = "penalty"
        cfg["log_dir"] = f"logs/synch_upgrades/su14/{task_name}_attn_ortho"
        plan.append(exp(
            f"su14_{task_name}_attn_ortho",
            f"{task_name}: attn(h=8) + ortho(λ=0.1)",
            _p(module, cfg),
            tags=[task_name, "attn-pairing", "orthogonal", "combo"],
        ))
    return plan


# ─── Registry ───

STAGE_BUILDERS = {
    "su00": build_su00_baseline_fixed_modes,
    "su01": build_su01_soft_gating,
    "su02": build_su02_gumbel_softmax,
    "su03": build_su03_sparsemax_entmax,
    "su04": build_su04_topk_gating,
    "su05": build_su05_dynamic_routing,
    "su06": build_su06_attention_pairing,
    "su07": build_su07_hypernet,
    "su08": build_su08_orthogonal_reg,
    "su09": build_su09_svd_orthogonal,
    "su10": build_su10_rl_search,
    "su11": build_su11_soft_orthogonal,
    "su12": build_su12_topk_orthogonal,
    "su13": build_su13_gumbel_topk,
    "su14": build_su14_attn_orthogonal,
}

STAGE_DESCRIPTIONS = {
    "su00": "Baseline: 3 fixed neuron_select_type modes (READY, uses existing flags)",
    "su01": "Softmax soft gating — 3 temperatures (IMPLEMENTED, READY)",
    "su02": "Gumbel-Softmax with temperature annealing — 4 schedules",
    "su03": "Sparsemax / entmax — 3 sparse transformations",
    "su04": "Top-K sparse gating (MoE-style) — 4 K values",
    "su05": "Dynamic routing (capsule-style) — 3 routing_iters",
    "su06": "Attention-based adaptive pairing — 3 head counts",
    "su07": "Hypernetwork-generated subspace weights — 2 ctx dims",
    "su08": "Orthogonality penalty on gate columns — 3 weights",
    "su09": "SVD-based hard orthogonalization — 2 frequencies",
    "su10": "REINFORCE-learned discrete selection — parity POC only",
    "su11": "Combo: soft gating + orthogonal penalty",
    "su12": "Combo: top-K sparse + orthogonal penalty",
    "su13": "Combo: Gumbel-Softmax + top-K (discrete + sparse)",
    "su14": "Combo: attention pairing + orthogonal penalty",
}


def _expand_seeds(plan, seeds=SEEDS):
    """Expand experiments to 3 seeds (skip su00 baselines, which stay 1-seed)."""
    new_plan = []
    for e in plan:
        if e["name"].startswith("su00_"):
            new_plan.append(e)
            continue
        if e.get("impl_status") != READY:
            new_plan.append(e)
            continue
        existing_seeds = [int(t[4:]) for t in e.get("tags", []) if t.startswith("seed")]
        existing_seed = existing_seeds[0] if existing_seeds else 0
        new_plan.append(e)
        for seed in seeds:
            if seed == existing_seed:
                continue
            name_base = re.sub(r'_s\d+$', '', e["name"])
            log_dir_match = re.search(r'--log_dir\s+(\S+)', e["command"])
            new_cmd = re.sub(r'--seed\s+\d+', f'--seed {seed}', e["command"])
            if log_dir_match:
                base_dir = re.sub(r'_s\d+$', '', log_dir_match.group(1))
                new_cmd = re.sub(r'--log_dir\s+\S+', f'--log_dir {base_dir}_s{seed}', new_cmd)
            new_tags = [t for t in e["tags"] if not t.startswith("seed")] + [f"seed{seed}"]
            new_plan.append({
                "name": f"{name_base}_s{seed}",
                "question": e["question"],
                "command": new_cmd,
                "tags": new_tags,
                "node_addr": _next_slot(),
                "impl_status": e.get("impl_status", READY),
            })
    return new_plan


def build_plan(stage="all", seed_expand=True):
    plan = []
    if stage == "all":
        for s in STAGES_ORDERED:
            if s == "su10":
                continue  # POC RL is opt-in only — too expensive for `all`
            if s in STAGE_BUILDERS:
                STAGE_BUILDERS[s](plan)
    elif stage in STAGE_BUILDERS:
        STAGE_BUILDERS[stage](plan)
    else:
        print(f"Unknown stage: {stage}. Available: {STAGES_ORDERED}")
        return []
    if seed_expand and stage == "all":
        plan = _expand_seeds(plan)
    return plan


def count_by_status(plan):
    ready = sum(1 for e in plan if e.get("impl_status") == READY)
    needs_impl = sum(1 for e in plan if e.get("impl_status") != READY)
    return ready, needs_impl


def print_plan(plan):
    ready, needs_impl = count_by_status(plan)
    print(f"\n{'='*82}")
    print(f"CTM SYNCH-UPGRADE EXPERIMENT PLAN — {len(plan)} experiments total")
    print(f"  Ready: {ready}  |  Needs implementation: {needs_impl}")
    print(f"{'='*82}\n")

    stages = {}
    for e in plan:
        prefix = e["name"].split("_")[0]
        stages.setdefault(prefix, []).append(e)

    for stage_name in STAGES_ORDERED:
        exps = stages.get(stage_name, [])
        if not exps:
            continue
        desc = STAGE_DESCRIPTIONS.get(stage_name, "")
        r = sum(1 for e in exps if e.get("impl_status") == READY)
        ni = len(exps) - r
        status = f"[{r} ready, {ni} needs_impl]"
        print(f"\n  ── {stage_name}: {desc} {status} ──")
        for e in exps:
            impl_mark = " [needs_impl]" if e.get("impl_status") != READY else ""
            print(f"     {e['name']}{impl_mark}")
            print(f"       {e['question']}")

    print(f"\n{'='*82}")
    print(f"TOTAL: {len(plan)} ({ready} ready, {needs_impl} needs implementation)")
    print(f"{'='*82}\n")


# ─── Pool submission ───

def submit_to_pool(exp, config, master_addr=None, port=None):
    node_addrs = exp.get("node_addrs") or [exp.get("node_addr")]
    payload = {
        "config": config,
        "extra_args": exp["command"],
        "node_addrs": node_addrs,
        "env": {
            "CTM_EXPERIMENT_NAME": exp["name"],
            "CTM_METRICS_DIR": "runs/metrics",
            "CTM_LOG_DIR": "runs/logs/synch_upgrades",
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
    if not args.include_unimplemented:
        submittable = [e for e in plan if e.get("impl_status") == READY]
        skipped = len(plan) - len(submittable)
        if skipped:
            print(f"NOTE: skipping {skipped} needs_impl experiments (pass --include-unimplemented to submit anyway)")
    else:
        submittable = plan

    if not submittable:
        print("No submittable experiments in this stage.")
        if not args.include_unimplemented:
            print("All experiments are needs_impl. Implement the model code first,")
            print("or pass --include-unimplemented to submit anyway (will fail at runtime).")
        return

    print(f"Submitting {len(submittable)} experiments to pool at {MASTER_ADDR}:{PORT}")
    for e in submittable:
        print(f"  {e['name']}")

    if args.dry_run:
        return

    for e in submittable:
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
    path = args.output or f"runs/experiment_plans/synch_upgrades_plan.csv"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "question", "command", "tags", "impl_status"])
        w.writeheader()
        for e in plan:
            w.writerow({
                "name": e["name"],
                "question": e["question"],
                "command": e["command"],
                "tags": ";".join(e["tags"]),
                "impl_status": e.get("impl_status", NEEDS_IMPL),
            })
    print(f"Wrote {len(plan)} experiments to {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CTM Synch-Upgrade Plan: 7 idea families + combos")
    sub = parser.add_subparsers(dest="command")

    p_plan = sub.add_parser("plan")
    p_plan.add_argument("--stage", default="all")

    p_submit = sub.add_parser("submit")
    p_submit.add_argument("--stage", default="all")
    p_submit.add_argument("--dry-run", action="store_true")
    p_submit.add_argument("--wait", action="store_true", default=True)
    p_submit.add_argument("--no-wait", action="store_false", dest="wait")
    p_submit.add_argument("--stop-on-fail", action="store_true", default=True)
    p_submit.add_argument("--include-unimplemented", action="store_true",
                          help="Submit needs_impl experiments (they will fail at runtime)")
    p_submit.add_argument("--master-addr", default=MASTER_ADDR)
    p_submit.add_argument("--port", type=int, default=PORT)

    p_csv = sub.add_parser("csv")
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
