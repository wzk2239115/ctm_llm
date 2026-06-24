#!/usr/bin/env python3
"""
Decoupled Thought Training (DTT) — One-Step Gradient for CTM.

Root cause analysis (from hr01 results):
  - bp_steps=1 on CIFAR-10:  73.79% (baseline 60.28%)  ← WORKS
  - bp_steps=1 on Sort:       0.31% (baseline 91.42%)   ← FAILS
  - bp_steps=1 on Parity:    61.06% (baseline 67.95%)   ← DEGRADED

Why? CIFAR-10's loss is per-tick independent (select most-certain tick).
Sort's loss is CTC (needs full-sequence gradient across ALL ticks).
One-step gradient kills the CTC alignment signal.

DTT solution: reformulate task losses so each tick independently predicts
the full answer, enabling one-step gradient to work universally.

Three components:
  1. Per-tick decoupled loss  — replace CTC with per-tick independent CE
  2. Progressive weighting    — later ticks weighted higher (refinement)
  3. State momentum correction — EMA of detached state to prevent drift

Usage:
    python scripts/experiment_plan_dtt.py plan [--stage all]
    python scripts/experiment_plan_dtt.py submit --stage dtt01 --no-wait
    python scripts/experiment_plan_dtt.py csv [--stage all]

Code changes required (see baseline/utils/dtt_ideas.py):
  - New args: --sort_loss_mode, --dtt_progressive_mode, --dtt_momentum_weight
  - New loss: per_tick_sort_loss() in baseline/utils/losses.py
  - Sort model: out_dims changes when sort_loss_mode=per_tick_ce
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
    "dtt00",            # baselines (control)
    "dtt01",            # per-tick CE loss (full BPTT, validate loss reformulation)
    "dtt02",            # per-tick CE + bp_steps sweep (validate one-step gradient)
    "dtt03",            # per-tick CE + progressive weighting
    "dtt04",            # per-tick CE + bp_steps + progressive (combine 1+2+3)
    "dtt05",            # per-tick CE + bp_steps + state momentum
    "dtt06",            # full DTT (all components)
    "dtt07",            # cross-task: bp_steps on cifar10/parity (already per-tick)
    "dtt08",            # efficiency benchmarks (short runs for timing)
]
ALL_STAGES = STAGES_ORDERED + ["all"]


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
# Per-task base configs — same as hrm_inspired for fair comparison
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

TASKS = {
    "sort":    ("baseline.tasks.sort.train",               SORT_BASE,    50),
    "parity":  ("baseline.tasks.parity.train",             PARITY_BASE,  75),
    "cifar10": ("baseline.tasks.image_classification.train", CIFAR10_BASE, 50),
}

FAST_TASKS = ["sort"]
MID_TASKS = ["sort", "parity", "cifar10"]


def with_seed(cfg, seed):
    c = dict(cfg)
    c["seed"] = seed
    return c


# ═══════════════════════════════════════════════════════════════
# STAGE BUILDERS
# ═══════════════════════════════════════════════════════════════

def build_dtt00_baselines(plan):
    """Control group: plain CTM paper configs (same as hr00).

    These establish the accuracy targets that DTT variants should match.
    """
    for task_name in MID_TASKS:
        module, base, _ = TASKS[task_name]
        cfg = dict(base)
        cfg["log_dir"] = f"logs/dtt/dtt00/{task_name}_baseline"
        plan.append(exp(
            f"dtt00_{task_name}_baseline",
            f"{task_name}: CTM baseline (control for DTT experiments)",
            _p(module, cfg),
            tags=[task_name, "baseline"],
            impl_status="ready",
            hypothesis="CTM paper config serves as accuracy target.",
        ))
    return plan


# ─── Stage dtt01: Per-tick CE Loss (validate loss reformulation) ───
# The most critical experiment: can per-tick CE replace CTC for sort?
# If this works with FULL BPTT, we know the loss reformulation is viable.
# Then dtt02 tests it with one-step gradient.

def build_dtt01_per_tick_ce(plan):
    """Replace CTC with per-tick independent CE for sort.

    Each tick predicts the FULL sorted sequence (all N positions),
    not one token per tick (CTC style). This decouples the loss from
    requiring gradient across all ticks.

    Needs:
      - --sort_loss_mode per_tick_ce  (new arg)
      - Model out_dims = N*(N+1) when per_tick_ce mode
      - per_tick_sort_loss() in losses.py

    This stage runs with FULL BPTT (bp_steps=0) to isolate the loss
    effect from gradient truncation.
    """
    module, base, _ = TASKS["sort"]
    for N in [10, 20, 30]:
        cfg = dict(with_seed(base, 0))
        cfg["N_to_sort"] = N
        cfg["sort_loss_mode"] = "per_tick_ce"
        cfg["log_dir"] = f"logs/dtt/dtt01/sort_per_tick_ce_N{N}"
        plan.append(exp(
            f"dtt01_sort_per_tick_ce_N{N}",
            f"sort(N={N}): per-tick CE loss (full BPTT) — can per-tick CE learn sort?",
            _p(module, cfg),
            tags=["sort", "per-tick-ce", f"N{N}"],
            impl_status="needs_impl",
            hypothesis=f"Per-tick CE can learn sort(N={N}) without CTC, "
                       f"because each tick independently predicts the full ordering. "
                       f"Expected accuracy: comparable to CTC baseline.",
        ))

    # Also test on parity (already per-tick, but with stablemax for stability)
    p_module, p_base, _ = TASKS["parity"]
    cfg = dict(with_seed(p_base, 0))
    cfg["loss_type"] = "stablemax_ce"
    cfg["log_dir"] = f"logs/dtt/dtt01/parity_stablemax_full_bp"
    plan.append(exp(
        f"dtt01_parity_stablemax_full_bp",
        "parity: stablemax CE (full BPTT) — stablemax is better suited for one-step grad",
        _p(p_module, cfg),
        tags=["parity", "stablemax", "full-bptt"],
        impl_status="ready",
        hypothesis="Stablemax CE avoids extreme logit saturation that "
                   "one-step gradient would exacerbate.",
    ))
    return plan


# ─── Stage dtt02: Per-tick CE + bp_steps Sweep ───
# THE KEY EXPERIMENT: does one-step gradient work with per-tick CE?
# If dtt01 shows per-tick CE can learn sort, this tests whether bp_steps=1
# now works (vs the 0% failure in hr01).

def build_dtt02_per_tick_ce_bp_steps(plan):
    """Per-tick CE + truncated BPTT: the core DTT hypothesis.

    hr01 showed bp_steps=1 gives 0% on sort with CTC.
    This stage tests whether per-tick CE FIXES that failure.

    bp_steps sweep: [1, 2, 5, 10]
    bp_steps=1 is the HRM-style one-step gradient.
    """
    module, base, _ = TASKS["sort"]
    for N in [10, 30]:
        for bp in [1, 2, 5, 10]:
            cfg = dict(with_seed(base, 0))
            cfg["N_to_sort"] = N
            cfg["sort_loss_mode"] = "per_tick_ce"
            cfg["bp_steps"] = bp
            cfg["log_dir"] = f"logs/dtt/dtt02/sort_per_tick_ce_N{N}_bp{bp}"
            plan.append(exp(
                f"dtt02_sort_per_tick_ce_N{N}_bp{bp}",
                f"sort(N={N}): per-tick CE + bp_steps={bp} — one-step gradient with decoupled loss",
                _p(module, cfg),
                tags=["sort", "per-tick-ce", f"N{N}", "bp-steps", f"bp{bp}"],
                impl_status="needs_impl",
                hypothesis=f"With per-tick CE, bp_steps={bp} should learn sort "
                           f"(vs 0% with CTC in hr01). Expected: bp_steps={bp} "
                           f"achieves >50% accuracy.",
            ))
    return plan


# ─── Stage dtt03: Progressive Weighting ───
# Weight later ticks higher: early ticks are "drafts", later are "refined".

def build_dtt03_progressive(plan):
    """Per-tick CE + progressive loss weighting.

    Later ticks get higher loss weight. This creates a curriculum:
    - Early ticks learn rough predictions (low weight, low penalty)
    - Later ticks learn precise predictions (high weight, high penalty)
    - With bp_steps=1, only the last (highest-weighted) tick needs gradient.
    """
    module, base, _ = TASKS["sort"]
    for mode in ["linear", "certainty", "exp"]:
        for bp in [1, 5]:
            cfg = dict(with_seed(base, 0))
            cfg["sort_loss_mode"] = "per_tick_ce"
            cfg["dtt_progressive_mode"] = mode
            cfg["bp_steps"] = bp
            cfg["log_dir"] = f"logs/dtt/dtt03/sort_prog_{mode}_bp{bp}"
            plan.append(exp(
                f"dtt03_sort_prog_{mode}_bp{bp}",
                f"sort: per-tick CE + progressive({mode}) + bp_steps={bp}",
                _p(module, cfg),
                tags=["sort", "per-tick-ce", "progressive", mode, f"bp{bp}"],
                impl_status="needs_impl",
                hypothesis=f"Progressive weighting ({mode}) with bp_steps={bp} "
                           f"lets early ticks be rough drafts and later ticks "
                           f"be precise — reducing the gradient signal needed.",
            ))
    return plan


# ─── Stage dtt04: Per-tick CE + bp_steps + Progressive (combine) ───

def build_dtt04_combined(plan):
    """Per-tick CE + bp_steps + progressive weighting combined.

    The practical sweet spot: moderate truncation + progressive weighting.
    """
    module, base, _ = TASKS["sort"]
    configs = [
        # (N, bp, progressive_mode, label)
        (30, 1, "certainty", "bp1_certainty"),
        (30, 2, "linear", "bp2_linear"),
        (30, 5, "certainty", "bp5_certainty"),
        (30, 10, "linear", "bp10_linear"),
        (10, 1, "linear", "N10_bp1_linear"),
        (10, 1, "certainty", "N10_bp1_certainty"),
    ]
    for N, bp, mode, label in configs:
        cfg = dict(with_seed(base, 0))
        cfg["N_to_sort"] = N
        cfg["sort_loss_mode"] = "per_tick_ce"
        cfg["bp_steps"] = bp
        cfg["dtt_progressive_mode"] = mode
        cfg["log_dir"] = f"logs/dtt/dtt04/sort_{label}"
        plan.append(exp(
            f"dtt04_sort_{label}",
            f"sort(N={N}): per-tick CE + bp{bp} + progressive({mode})",
            _p(module, cfg),
            tags=["sort", "per-tick-ce", "progressive", f"bp{bp}", mode],
            impl_status="needs_impl",
            hypothesis=f"Combined per-tick CE + bp{bp} + {mode} weighting "
                       f"should match or exceed CTC baseline accuracy.",
        ))
    return plan


# ─── Stage dtt05: State Momentum Correction ───
# Keep EMA of detached synchronisation accumulators to prevent drift.

def build_dtt05_momentum(plan):
    """Per-tick CE + bp_steps + state momentum correction.

    When state is detached between gradient steps, the accumulators
    (decay_alpha, decay_beta) drift from their optimal values over training.
    State momentum maintains an EMA of the "correct" accumulator values
    and applies a correction term.

    Needs:
      - --dtt_momentum_weight FLOAT (0.0 = disabled)
      - --dtt_momentum_decay FLOAT (EMA decay, default 0.99)
      - EMA buffers in CTM forward (alpha_ema, beta_ema)
    """
    module, base, _ = TASKS["sort"]
    for mw in [0.1, 0.3, 0.5]:
        for bp in [1, 5]:
            cfg = dict(with_seed(base, 0))
            cfg["sort_loss_mode"] = "per_tick_ce"
            cfg["bp_steps"] = bp
            cfg["dtt_momentum_weight"] = mw
            cfg["dtt_momentum_decay"] = 0.99
            cfg["dtt_progressive_mode"] = "certainty"
            cfg["log_dir"] = f"logs/dtt/dtt05/sort_momentum{mw}_bp{bp}"
            plan.append(exp(
                f"dtt05_sort_momentum{mw}_bp{bp}",
                f"sort: per-tick CE + bp{bp} + momentum(w={mw})",
                _p(module, cfg),
                tags=["sort", "per-tick-ce", "momentum", f"bp{bp}", f"mw{mw}"],
                impl_status="needs_impl",
                hypothesis=f"State momentum (w={mw}) prevents detached accumulator "
                           f"drift, improving stability over long training.",
            ))
    return plan


# ─── Stage dtt06: Full DTT (all components) ───

def build_dtt06_full(plan):
    """Full DTT: per-tick CE + best bp_steps + progressive + momentum + atan2.

    The culmination: combine all winning components.
    Configs are chosen based on expected best from dtt02-dtt05.
    """
    module, base, _ = TASKS["sort"]

    # Config A: aggressive one-step (bp=1)
    cfg_a = dict(with_seed(base, 0))
    cfg_a["sort_loss_mode"] = "per_tick_ce"
    cfg_a["bp_steps"] = 1
    cfg_a["dtt_progressive_mode"] = "certainty"
    cfg_a["dtt_momentum_weight"] = 0.3
    cfg_a["dtt_momentum_decay"] = 0.99
    cfg_a["optimizer_type"] = "adam_atan2"
    cfg_a["beta2"] = 0.95
    cfg_a["log_dir"] = f"logs/dtt/dtt06/sort_full_bp1"
    plan.append(exp(
        f"dtt06_sort_full_bp1",
        "sort: full DTT (bp1 + progressive + momentum + atan2)",
        _p(module, cfg_a),
        tags=["sort", "full-dtt", "bp1"],
        impl_status="needs_impl",
        hypothesis="Full DTT with one-step gradient should match baseline accuracy "
                   "at a fraction of the compute cost.",
    ))

    # Config B: moderate truncation (bp=5)
    cfg_b = dict(cfg_a)
    cfg_b["bp_steps"] = 5
    cfg_b["log_dir"] = f"logs/dtt/dtt06/sort_full_bp5"
    plan.append(exp(
        f"dtt06_sort_full_bp5",
        "sort: full DTT (bp5 + progressive + momentum + atan2)",
        _p(module, cfg_b),
        tags=["sort", "full-dtt", "bp5"],
        impl_status="needs_impl",
        hypothesis="Full DTT with bp5 as a safer compromise.",
    ))

    # Config C: smaller N for fast iteration
    cfg_c = dict(cfg_a)
    cfg_c["N_to_sort"] = 10
    cfg_c["training_iterations"] = 50001
    cfg_c["log_dir"] = f"logs/dtt/dtt06/sort_N10_full_bp1"
    plan.append(exp(
        f"dtt06_sort_N10_full_bp1",
        "sort(N=10): full DTT bp1 — fast validation",
        _p(module, cfg_c),
        tags=["sort", "full-dtt", "bp1", "N10", "fast"],
        impl_status="needs_impl",
        hypothesis="Quick validation of full DTT on smaller sort.",
    ))
    return plan


# ─── Stage dtt07: Cross-task Validation ───
# CIFAR-10 and Parity already have per-tick independent losses.
# This stage validates that bp_steps works on them too (expected from hr01).

def build_dtt07_cross_task(plan):
    """Apply bp_steps to CIFAR-10 and Parity (already per-tick losses).

    These tasks already have per-tick independent losses (select most-certain
    tick). hr01 showed bp_steps works partially. This stage adds:
      - Progressive weighting
      - State momentum
      - Adam-atan2
    to see if they improve further.
    """
    # CIFAR-10: bp_steps already works, add progressive + momentum
    c_module, c_base, _ = TASKS["cifar10"]
    for bp in [1, 5]:
        cfg = dict(with_seed(c_base, 0))
        cfg["bp_steps"] = bp
        cfg["dtt_progressive_mode"] = "certainty"
        cfg["dtt_momentum_weight"] = 0.1
        cfg["optimizer_type"] = "adam_atan2"
        cfg["beta2"] = 0.95
        cfg["log_dir"] = f"logs/dtt/dtt07/cifar10_bp{bp}_dtt"
        plan.append(exp(
            f"dtt07_cifar10_bp{bp}_dtt",
            f"cifar10: bp{bp} + progressive + momentum + atan2",
            _p(c_module, cfg),
            tags=["cifar10", "cross-task", f"bp{bp}", "dtt"],
            impl_status="needs_impl",
            hypothesis="DTT components improve CIFAR-10 beyond hr01's bp-only results.",
        ))

    # Parity: bp_steps degraded in hr01, test if DTT fixes it
    p_module, p_base, _ = TASKS["parity"]
    for bp in [1, 5]:
        cfg = dict(with_seed(p_base, 0))
        cfg["bp_steps"] = bp
        cfg["loss_type"] = "stablemax_ce"
        cfg["dtt_progressive_mode"] = "certainty"
        cfg["dtt_momentum_weight"] = 0.3
        cfg["optimizer_type"] = "adam_atan2"
        cfg["beta2"] = 0.95
        cfg["log_dir"] = f"logs/dtt/dtt07/parity_bp{bp}_dtt"
        plan.append(exp(
            f"dtt07_parity_bp{bp}_dtt",
            f"parity: bp{bp} + stablemax + progressive + momentum + atan2",
            _p(p_module, cfg),
            tags=["parity", "cross-task", f"bp{bp}", "dtt"],
            impl_status="needs_impl",
            hypothesis="DTT components fix parity's degradation from hr01 "
                       "(bp1: 61% → target >70%).",
        ))
    return plan


# ─── Stage dtt08: Efficiency Benchmarks ───
# Short runs to measure wall-clock time and memory.

def build_dtt08_efficiency(plan):
    """Efficiency benchmarks: measure time and memory of DTT vs full BPTT.

    Short runs (1000 iterations) to measure:
      - Wall-clock time per iteration
      - Peak GPU memory
      - Tokens/sec throughput
    """
    module, base, _ = TASKS["sort"]
    for bp in [0, 1, 5, 10]:  # bp=0 is full BPTT baseline
        cfg = dict(with_seed(base, 0))
        cfg["sort_loss_mode"] = "per_tick_ce"
        cfg["bp_steps"] = bp
        cfg["training_iterations"] = 1000
        cfg["track_every"] = 100
        cfg["save_every"] = 10000
        label = "full_bptt" if bp == 0 else f"bp{bp}"
        cfg["log_dir"] = f"logs/dtt/dtt08/sort_efficiency_{label}"
        plan.append(exp(
            f"dtt08_sort_efficiency_{label}",
            f"sort: efficiency benchmark bp_steps={bp} (1000 iters)",
            _p(module, cfg),
            tags=["sort", "efficiency", f"bp{bp}"],
            impl_status="needs_impl",
            hypothesis=f"bp_steps={bp} reduces memory from O(T) to O(bp) "
                       f"and speeds up training by ~{50/max(bp,1):.0f}x.",
        ))
    return plan


# ─── Registry ───

STAGE_BUILDERS = {
    "dtt00": build_dtt00_baselines,
    "dtt01": build_dtt01_per_tick_ce,
    "dtt02": build_dtt02_per_tick_ce_bp_steps,
    "dtt03": build_dtt03_progressive,
    "dtt04": build_dtt04_combined,
    "dtt05": build_dtt05_momentum,
    "dtt06": build_dtt06_full,
    "dtt07": build_dtt07_cross_task,
    "dtt08": build_dtt08_efficiency,
}

STAGE_DESCRIPTIONS = {
    "dtt00": "Baselines (control group for comparison)",
    "dtt01": "Per-tick CE loss: replace CTC with per-tick independent CE [LOSS REFORMULATION]",
    "dtt02": "Per-tick CE + bp_steps sweep: validate one-step gradient [CORE EXPERIMENT]",
    "dtt03": "Per-tick CE + progressive weighting: curriculum for tick losses",
    "dtt04": "Combined: per-tick CE + bp_steps + progressive weighting",
    "dtt05": "Per-tick CE + bp_steps + state momentum: prevent detached state drift",
    "dtt06": "Full DTT: all components combined (per-tick CE + bp + progressive + momentum + atan2)",
    "dtt07": "Cross-task: apply DTT to CIFAR-10 and Parity",
    "dtt08": "Efficiency benchmarks: time/memory of DTT vs full BPTT",
}

IMPL_PRIORITY = {
    "dtt01": 1,   # per-tick CE — the fundamental loss change, highest priority
    "dtt02": 2,   # per-tick CE + bp_steps — the core hypothesis test
    "dtt08": 3,   # efficiency — quick to run, important for paper
    "dtt03": 4,   # progressive weighting — loss enhancement
    "dtt04": 5,   # combined — builds on dtt02+dtt03
    "dtt05": 6,   # state momentum — stability enhancement
    "dtt07": 7,   # cross-task — generalization
    "dtt06": 8,   # full DTT — final validation
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
    print(f"DECOUPLED THOUGHT TRAINING (DTT) — {len(plan)} experiments total")
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
            count = len(stages[sname])
            print(f"  {IMPL_PRIORITY[sname]}. {sname}: {STAGE_DESCRIPTIONS[sname]} ({count} exps)")
    print(f"\nRequired code changes:")
    print(f"  1. baseline/utils/dtt_ideas.py  — new args + loss functions")
    print(f"  2. baseline/utils/losses.py     — per_tick_sort_loss()")
    print(f"  3. baseline/tasks/sort/train.py — wire new args + loss")
    print(f"  4. baseline/models/ctm_sort.py  — out_dims for per_tick_ce mode")
    print(f"{'='*80}\n")


# ─── Pool submission ───

def submit_to_pool(exp_entry, config, master_addr=None, port=None):
    payload = {
        "config": config,
        "extra_args": exp_entry["command"],
        "node_addrs": [],
        "env": {
            "CTM_EXPERIMENT_NAME": exp_entry["name"],
            "CTM_METRICS_DIR": "runs/metrics",
            "CTM_LOG_DIR": "runs/logs/dtt",
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

    master = args.master_addr
    port = args.port
    print(f"Submitting {len(to_submit)} experiments to pool at {master}:{port}")
    for e in to_submit:
        status = "ready" if e.get("impl_status") == "ready" else "NEEDS_IMPL"
        print(f"  [{status}] {e['name']}")

    if args.dry_run:
        return

    for e in to_submit:
        print(f"Submitting {e['name']}...")
        result = submit_to_pool(e, POOL_CONFIG, master, port)
        if result is None:
            print(f"  FAILED to submit {e['name']}")
            continue
        task_id = result if isinstance(result, str) else result.get("task_id", "")
        print(f"  task_id={task_id}")
        if args.wait:
            final_status = wait_until_idle(master, port, task_id)
            print(f"  -> {final_status}")
            if final_status == "failed" and args.stop_on_fail:
                print("Stopping due to failure.")
                break
            time.sleep(5)


def cmd_csv(args):
    import csv
    plan = build_plan(args.stage)
    path = args.output or f"runs/experiment_plans/dtt_plan.csv"
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
    parser = argparse.ArgumentParser(description="Decoupled Thought Training (DTT) Experiments")
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
