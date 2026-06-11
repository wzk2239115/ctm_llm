#!/usr/bin/env python3
"""
CTM Paper Experiment Plan — All Ideas × All Tasks × Grid Search.

Baseline: original CTM (continuous-thought-machines) on 5 tasks.
Apply ALL ideas from the CTM-LLM project, measure performance deltas.

Usage:
    # Show plan summary
    python scripts/experiment_plan_ctm_paper.py plan [--stage all]
    
    # Submit one stage to pool
    python scripts/experiment_plan_ctm_paper.py submit --stage st00
    
    # Generate CSV
    python scripts/experiment_plan_ctm_paper.py csv [--stage all]
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
    "st00", "st01", "st02", "st03", "st04", "st05", "st06",
    "st07", "st08", "st09", "st10", "st11", "st12",
    "st13", "st14", "st15", "st16", "st17", "st18",
    "st19", "st20", "st21", "st22", "st23", "st24",
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


def exp(name, question, command, tags=None, impl_status="ready"):
    return {
        "name": name,
        "question": question,
        "command": command,
        "tags": tags or [],
        "node_addr": _next_slot(),
        "impl_status": impl_status,
    }


# ─── Per-task base configs (paper configs, reduced training for feasibility) ───

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

# Task → (module, base_config, default_iterations)
TASKS = {
    "sort": ("baseline.tasks.sort.train", SORT_BASE, 50),
    "parity": ("baseline.tasks.parity.train", PARITY_BASE, 75),
    "mazes": ("baseline.tasks.mazes.train", MAZES_BASE, 75),
    "cifar10": ("baseline.tasks.image_classification.train", CIFAR10_BASE, 50),
    "qamnist": ("baseline.tasks.qamnist.train", QAMNIST_BASE, None),  # varies per phase
}

SEEDS = [0, 1, 2]


def with_seed(cfg, seed):
    c = dict(cfg)
    c["seed"] = seed
    return c


def tagged(plan, tag, exp_list):
    """Tag a list of experiments and add to plan."""
    for e in exp_list:
        e["tags"].append(tag)
    plan.extend(exp_list)


# ═══════════════════════════════════════════════════════════════
# STAGE BUILDERS
# ═══════════════════════════════════════════════════════════════

def build_st00_paper_reproduction(plan):
    """Paper config baselines + LSTM/FF baselines."""
    for task_name, (module, base, _) in TASKS.items():
        for seed in SEEDS[:1]:  # 1 seed for reproduction
            plan.append(exp(
                f"st00_{task_name}_paper_s{seed}",
                f"{task_name}: CTM paper config (seed={seed})",
                _p(module, {**with_seed(base, seed), "log_dir": f"logs/ctm_paper/st00/{task_name}_paper_s{seed}"}),
                tags=[task_name, "paper", f"seed{seed}"],
            ))
        # LSTM baselines (where available)
        if task_name in ("parity",):
            lstm_cfg = dict(with_seed(base, 0))
            lstm_cfg["model_type"] = "lstm"
            plan.append(exp(
                f"st00_{task_name}_lstm_paper",
                f"{task_name}: LSTM baseline",
                _p(module, {**lstm_cfg, "log_dir": f"logs/ctm_paper/st00/{task_name}_lstm"}),
                tags=[task_name, "lstm"],
            ))
        if task_name in ("cifar10",):
            ff_cfg = dict(with_seed(base, 0))
            ff_cfg["model"] = "ff"
            cmp = dict(ff_cfg)
            for key in ["d_input", "heads", "iterations", "positional_embedding_type",
                         "synapse_depth", "n_synch_out", "n_synch_action", "neuron_select_type",
                         "n_random_pairing_self", "memory_length", "deep_memory", "memory_hidden_dims",
                         "dropout_nlm", "do_normalisation"]:
                cmp.pop(key, None)
            plan.append(exp(
                f"st00_{task_name}_ff_baseline",
                f"{task_name}: FF baseline",
                _p(module, {**cmp, "log_dir": f"logs/ctm_paper/st00/{task_name}_ff"}),
                tags=[task_name, "ff"],
            ))
    return plan


def build_st01_architecture_sweep(plan):
    """d_model, synapse_depth, memory_hidden_dims, heads, neuron_select_type sweeps."""
    sweeps = {
        "d_model": [("d4", 0.25), ("d2", 0.5), ("2x", 2.0)],
        "synapse_depth": [("sd1", 1), ("sd2", 2), ("sd8", 8)],
        "memory_hidden_dims": [("mh1", 1), ("mh2", 2), ("mh32", 32)],
        "heads": [("h2", 2), ("h8", 8), ("h16", 16)],
    }
    for task_name, (module, base, _) in TASKS.items():
        if task_name in ("qamnist",):
            continue  # skip for now
        for param, variants in sweeps.items():
            for tag_suffix, value in variants:
                if param == "d_model":
                    new_cfg = dict(with_seed(base, 0))
                    new_cfg[param] = max(64, int(base[param] * value))
                    new_cfg["d_input"] = max(32, new_cfg[param] // 4)
                else:
                    new_cfg = dict(with_seed(base, 0))
                    new_cfg[param] = value
                plan.append(exp(
                    f"st01_{task_name}_{param}{tag_suffix}",
                    f"{task_name}: {param}={value}",
                    _p(module, {**new_cfg, "log_dir": f"logs/ctm_paper/st01/{task_name}_{param}{tag_suffix}"}),
                    tags=[task_name, "arch-sweep", param],
                ))
        # neuron_select_type ablation
        for nst in ["random", "first-last", "random-pairing"]:
            cfg = dict(with_seed(base, 0))
            cfg["neuron_select_type"] = nst
            plan.append(exp(
                f"st01_{task_name}_nst_{nst}",
                f"{task_name}: neuron_select={nst}",
                _p(module, {**cfg, "log_dir": f"logs/ctm_paper/st01/{task_name}_nst_{nst}"}),
                tags=[task_name, "arch-sweep", "nst"],
            ))
    return plan


def build_st02_tick_sweep(plan):
    """Tick count sweep: 1, 2, 5, 10, 25, 50."""
    tick_values = [1, 2, 5, 10, 25, 50]
    for task_name, (module, base, _) in TASKS.items():
        if task_name == "qamnist":
            continue
        for t in tick_values:
            cfg = dict(with_seed(base, 0))
            cfg["iterations"] = t
            cfg["memory_length"] = max(2, t // 2)
            plan.append(exp(
                f"st02_{task_name}_tick{t}",
                f"{task_name}: {t} ticks",
                _p(module, {**cfg, "log_dir": f"logs/ctm_paper/st02/{task_name}_tick{t}"}),
                tags=[task_name, "tick-sweep"],
            ))
    # QAMNIST tick sweep (repeats_per_input)
    for t in [1, 5, 10, 20]:
        cfg = dict(with_seed(QAMNIST_BASE, 0))
        cfg["q_num_repeats_per_input"] = t
        cfg["q_num_answer_steps"] = t
        plan.append(exp(
            f"st02_qamnist_repeat{t}",
            f"QAMNIST: {t} repeats/tick",
            _p("baseline.tasks.qamnist.train", {**cfg, "log_dir": f"logs/ctm_paper/st02/qamnist_repeat{t}"}),
            tags=["qamnist", "tick-sweep"],
        ))
    return plan


def build_st03_simplified_core(plan):
    """Simplified core: synapse_depth=2, memory_hidden_dims=2 at tick2."""
    for task_name, (module, base, default_iters) in TASKS.items():
        if task_name == "qamnist":
            continue
        for sd, mh in [(2, 2), (1, 2)]:
            cfg = dict(with_seed(base, 0))
            cfg["synapse_depth"] = sd
            cfg["memory_hidden_dims"] = mh
            cfg["iterations"] = 2 if task_name != "mazes" else 4
            cfg["memory_length"] = 5
            plan.append(exp(
                f"st03_{task_name}_sd{sd}_mh{mh}_tick{cfg['iterations']}",
                f"{task_name}: simplified sd={sd} mh={mh} tick={cfg['iterations']}",
                _p(module, {**cfg, "log_dir": f"logs/ctm_paper/st03/{task_name}_sd{sd}_mh{mh}"}),
                tags=[task_name, "simplified-core"],
            ))
    return plan


def build_st04_jepa_sweep(plan):
    """Cross-tick JEPA: weight, loss, stop_grad sweeps on paper config."""
    jepa_defaults = dict(
        cross_tick_jepa_hidden_dim=128,
        cross_tick_jepa_predictor_depth=2,
        cross_tick_jepa_dropout=0.0,
    )
    for task_name, (module, base, _) in TASKS.items():
        # weight sweep
        for w in [0.1, 0.5, 1.0]:
            cfg = dict(with_seed(base, 0))
            cfg["log_dir"] = f"logs/ctm_paper/st04/{task_name}_jepa_w{w}"
            cfg.update(jepa_defaults)
            cfg["cross_tick_jepa_weight"] = w
            plan.append(exp(
                f"st04_{task_name}_jepa_w{str(w).replace('.','p')}",
                f"{task_name}: JEPA weight={w}",
                _p(module, cfg),
                tags=[task_name, "jepa", "weight"],
            ))
        # loss type
        for loss_type in ["mse"]:
            cfg = dict(with_seed(base, 0))
            cfg["log_dir"] = f"logs/ctm_paper/st04/{task_name}_jepa_{loss_type}"
            cfg.update(jepa_defaults)
            cfg["cross_tick_jepa_weight"] = 1.0
            cfg["cross_tick_jepa_loss"] = loss_type
            plan.append(exp(
                f"st04_{task_name}_jepa_{loss_type}",
                f"{task_name}: JEPA loss={loss_type}",
                _p(module, cfg),
                tags=[task_name, "jepa", "loss"],
            ))
        # no stop-grad
        cfg = dict(with_seed(base, 0))
        cfg["log_dir"] = f"logs/ctm_paper/st04/{task_name}_jepa_nostopgrad"
        cfg.update(jepa_defaults)
        cfg["cross_tick_jepa_weight"] = 1.0
        cfg["cross_tick_jepa_target_stop_grad"] = False
        plan.append(exp(
            f"st04_{task_name}_jepa_nostopgrad",
            f"{task_name}: JEPA no stop-grad",
            _p(module, cfg),
            tags=[task_name, "jepa", "stop-grad"],
        ))
        # predictor depth
        for pd in [1, 4]:
            cfg = dict(with_seed(base, 0))
            cfg["log_dir"] = f"logs/ctm_paper/st04/{task_name}_jepa_pd{pd}"
            cfg.update(jepa_defaults)
            cfg["cross_tick_jepa_weight"] = 1.0
            cfg["cross_tick_jepa_predictor_depth"] = pd
            plan.append(exp(
                f"st04_{task_name}_jepa_pd{pd}",
                f"{task_name}: JEPA predictor_depth={pd}",
                _p(module, cfg),
                tags=[task_name, "jepa", "predictor-depth"],
            ))
    return plan


def build_st05_jepa_plus(plan):
    """JEPA combined with other modifications: tick2, simplified_core."""
    for task_name, (module, base, _) in TASKS.items():
        if task_name in ("qamnist", "mazes"):
            continue
        jepa = dict(
            cross_tick_jepa_weight=1.0,
            cross_tick_jepa_loss="cosine",
            cross_tick_jepa_hidden_dim=128,
            cross_tick_jepa_predictor_depth=2,
            cross_tick_jepa_dropout=0.0,
            cross_tick_jepa_target_stop_grad=True,
        )
        # JEPA + tick2
        cfg = dict(with_seed(base, 0))
        cfg["iterations"] = 2
        cfg["memory_length"] = 5
        cfg.update(jepa)
        plan.append(exp(
            f"st05_{task_name}_jepa_tick2",
            f"{task_name}: JEPA + tick2",
            _p(module, {**cfg, "log_dir": f"logs/ctm_paper/st05/{task_name}_jepa_tick2"}),
            tags=[task_name, "jepa", "tick2"],
        ))
        # JEPA + simplified core
        cfg = dict(with_seed(base, 0))
        cfg["synapse_depth"] = 2
        cfg["memory_hidden_dims"] = 2
        cfg.update(jepa)
        plan.append(exp(
            f"st05_{task_name}_jepa_sd2_mh2",
            f"{task_name}: JEPA + simplified core",
            _p(module, {**cfg, "log_dir": f"logs/ctm_paper/st05/{task_name}_jepa_sd2_mh2"}),
            tags=[task_name, "jepa", "simplified"],
        ))
        # JEPA + tick2 + simplified
        cfg = dict(with_seed(base, 0))
        cfg["iterations"] = 2
        cfg["memory_length"] = 5
        cfg["synapse_depth"] = 2
        cfg["memory_hidden_dims"] = 2
        cfg.update(jepa)
        plan.append(exp(
            f"st05_{task_name}_jepa_tick2_sd2_mh2",
            f"{task_name}: JEPA + tick2 + simplified",
            _p(module, {**cfg, "log_dir": f"logs/ctm_paper/st05/{task_name}_jepa_tick2_sd2_mh2"}),
            tags=[task_name, "jepa", "tick2", "simplified"],
        ))
    return plan


# ─── Ideas that need implementation ───

def build_st06_tick_halt(plan):
    """Early-exit/stop when confidence exceeds threshold. impl_status=needs_impl."""
    for task_name, (module, base, _) in TASKS.items():
        for thresh in [0.3, 0.6, 0.9]:
            for compute_w in [0.0, 1e-3]:
                cfg = dict(with_seed(base, 0))
                cfg["tick_halt_mode"] = "threshold"
                cfg["tick_halt_threshold"] = thresh
                cfg["tick_compute_weight"] = compute_w
                plan.append(exp(
                    f"st06_{task_name}_halt{str(thresh).replace('.','p')}_cw{str(compute_w).replace('.','p')}",
                    f"{task_name}: halt@{thresh} compute_w={compute_w}",
                    _p(module, {**cfg, "log_dir": f"logs/ctm_paper/st06/{task_name}_halt{thresh}"}),
                    tags=[task_name, "tick-halt"],
                    impl_status="ready",
                ))
    return plan


def build_st07_multi_tick_loss(plan):
    """Multi-tick/anytime supervision: supervise all intermediate ticks."""
    for task_name, (module, base, _) in TASKS.items():
        for mode in ["mean", "min_conf"]:
            cfg = dict(with_seed(base, 0))
            cfg["tick_loss_mode"] = mode
            plan.append(exp(
                f"st07_{task_name}_tickloss_{mode}",
                f"{task_name}: tick_loss={mode}",
                _p(module, {**cfg, "log_dir": f"logs/ctm_paper/st07/{task_name}_tickloss_{mode}"}),
                tags=[task_name, "multi-tick-loss"],
                impl_status="ready",
            ))
    return plan


def build_st08_cell_sparsity(plan):
    """Top-k neuron sparsity: only activate top-k neurons per tick."""
    for task_name, (module, base, _) in TASKS.items():
        d = base.get("d_model", 512)
        for frac in [0.25, 0.50, 0.75]:
            topk = max(8, int(d * frac))
            cfg = dict(with_seed(base, 0))
            cfg["topk_neurons"] = topk
            plan.append(exp(
                f"st08_{task_name}_sparsity{str(frac).replace('.','p')}",
                f"{task_name}: top-k={topk} ({frac})",
                _p(module, {**cfg, "log_dir": f"logs/ctm_paper/st08/{task_name}_sparsity{frac}"}),
                tags=[task_name, "sparsity"],
                impl_status="ready",
            ))
    return plan


def build_st09_reflex_head(plan):
    """Reflex output head: lightweight head producing output after very few ticks."""
    for task_name, (module, base, _) in TASKS.items():
        for reflex_weight in [0.1, 0.2]:
            cfg = dict(with_seed(base, 0))
            cfg["reflex_head"] = True
            cfg["reflex_weight"] = reflex_weight
            cfg["reflex_ticks"] = 1
            cfg["reflex_distill"] = True
            plan.append(exp(
                f"st09_{task_name}_reflex_w{str(reflex_weight).replace('.','p')}",
                f"{task_name}: reflex head w={reflex_weight}",
                _p(module, {**cfg, "log_dir": f"logs/ctm_paper/st09/{task_name}_reflex"}),
                tags=[task_name, "reflex"],
                impl_status="ready",
            ))
    return plan


def build_st10_draft_revise(plan):
    """Draft-and-revise: draft pass + revision pass with noise injection."""
    for task_name, (module, base, _) in TASKS.items():
        if task_name in ("qamnist",):
            continue
        for revise_w in [0.1, 0.2]:
            for corrupt_p in [0.15, 0.30]:
                cfg = dict(with_seed(base, 0))
                cfg["draft_mode"] = "revise"
                cfg["draft_block_size"] = 2
                cfg["draft_revise_weight"] = revise_w
                cfg["draft_corrupt_prob"] = corrupt_p
                plan.append(exp(
                    f"st10_{task_name}_revise_w{str(revise_w).replace('.','p')}_cp{str(corrupt_p).replace('.','p')}",
                    f"{task_name}: revise w={revise_w} corrupt={corrupt_p}",
                    _p(module, {**cfg, "log_dir": f"logs/ctm_paper/st10/{task_name}_revise"}),
                    tags=[task_name, "draft-revise"],
                    impl_status="ready",
                ))
    return plan


def build_st11_async_ticks(plan):
    """Async tick clocks: different neuron bands tick at different frequencies."""
    for task_name, (module, base, _) in TASKS.items():
        if task_name in ("qamnist",):
            continue
        for periods in ["1,2,4", "1,2,4,8"]:
            cfg = dict(with_seed(base, 0))
            cfg["async_tick_mode"] = "banded"
            cfg["async_tick_periods"] = periods
            cfg["async_fast_output_weight"] = 0.20
            plan.append(exp(
                f"st11_{task_name}_async_{periods.replace(',','_')}",
                f"{task_name}: async ticks periods={periods}",
                _p(module, {**cfg, "log_dir": f"logs/ctm_paper/st11/{task_name}_async"}),
                tags=[task_name, "async-ticks"],
                impl_status="ready",
            ))
    return plan


def build_st12_ema_speed_spectrum(plan):
    """DINO-like speed spectrum: multiple EMA teachers with different speeds."""
    for task_name, (module, base, _) in TASKS.items():
        if task_name in ("qamnist",):
            continue
        for decays in ["0.90,0.99", "0.90,0.97,0.995"]:
            for weight in [0.02, 0.05]:
                cfg = dict(with_seed(base, 0))
                cfg["ema_speed_mode"] = "ema_spectrum"
                cfg["ema_speed_decays"] = decays
                cfg["ema_distill_weight"] = weight
                plan.append(exp(
                    f"st12_{task_name}_ema_{decays.replace(',','_').replace('.','p')}_w{str(weight).replace('.','p')}",
                    f"{task_name}: EMA spectrum decays={decays} w={weight}",
                    _p(module, {**cfg, "log_dir": f"logs/ctm_paper/st12/{task_name}_ema"}),
                    tags=[task_name, "ema-spectrum"],
                    impl_status="ready",
                ))
    return plan


# ═══════════════════════════════════════════════════════════════
# CROSS-IDEA COMBO STAGES
# ═══════════════════════════════════════════════════════════════

def _jepa_defaults():
    return dict(
        cross_tick_jepa_weight=1.0,
        cross_tick_jepa_loss="cosine",
        cross_tick_jepa_hidden_dim=128,
        cross_tick_jepa_predictor_depth=2,
        cross_tick_jepa_dropout=0.0,
        cross_tick_jepa_target_stop_grad=True,
    )


def build_st13_jepa_halt(plan):
    """JEPA + tick halt: representation + early exit."""
    jepa = _jepa_defaults()
    for task_name, (module, base, _) in TASKS.items():
        for thresh in [0.6, 0.9]:
            for cw in [0.0, 1e-3]:
                cfg = dict(with_seed(base, 0))
                cfg["tick_halt_mode"] = "threshold"
                cfg["tick_halt_threshold"] = thresh
                cfg["tick_compute_weight"] = cw
                cfg.update(jepa)
                plan.append(exp(
                    f"st13_{task_name}_jepa_halt{str(thresh).replace('.','p')}_cw{str(cw).replace('.','p')}",
                    f"{task_name}: JEPA+halt@{thresh} cw={cw}",
                    _p(module, {**cfg, "log_dir": f"logs/ctm_paper/st13/{task_name}_jepa_halt"}),
                    tags=[task_name, "jepa", "tick-halt"],
                ))
    return plan


def build_st14_jepa_sparsity(plan):
    """JEPA + top-k sparsity."""
    jepa = _jepa_defaults()
    for task_name, (module, base, _) in TASKS.items():
        d = base.get("d_model", 512)
        for frac in [0.5, 0.75]:
            topk = max(8, int(d * frac))
            cfg = dict(with_seed(base, 0))
            cfg["topk_neurons"] = topk
            cfg.update(jepa)
            plan.append(exp(
                f"st14_{task_name}_jepa_sparsity{str(frac).replace('.','p')}",
                f"{task_name}: JEPA+sparsity topk={topk}",
                _p(module, {**cfg, "log_dir": f"logs/ctm_paper/st14/{task_name}_jepa_sparsity"}),
                tags=[task_name, "jepa", "sparsity"],
            ))
    return plan


def build_st15_halt_sparsity(plan):
    """Tick halt + top-k sparsity."""
    for task_name, (module, base, _) in TASKS.items():
        d = base.get("d_model", 512)
        for frac in [0.5, 0.75]:
            topk = max(8, int(d * frac))
            cfg = dict(with_seed(base, 0))
            cfg["tick_halt_mode"] = "threshold"
            cfg["tick_halt_threshold"] = 0.6
            cfg["topk_neurons"] = topk
            plan.append(exp(
                f"st15_{task_name}_halt_sparsity{str(frac).replace('.','p')}",
                f"{task_name}: halt@0.6+sparsity topk={topk}",
                _p(module, {**cfg, "log_dir": f"logs/ctm_paper/st15/{task_name}_halt_sparsity"}),
                tags=[task_name, "tick-halt", "sparsity"],
            ))
    return plan


def build_st16_jepa_async(plan):
    """JEPA + async banded ticks."""
    jepa = _jepa_defaults()
    for task_name, (module, base, _) in TASKS.items():
        if task_name in ("qamnist",):
            continue
        for periods in ["1,2,4", "1,2,4,8"]:
            cfg = dict(with_seed(base, 0))
            cfg["async_tick_mode"] = "banded"
            cfg["async_tick_periods"] = periods
            cfg.update(jepa)
            plan.append(exp(
                f"st16_{task_name}_jepa_async_{periods.replace(',','_')}",
                f"{task_name}: JEPA+async periods={periods}",
                _p(module, {**cfg, "log_dir": f"logs/ctm_paper/st16/{task_name}_jepa_async"}),
                tags=[task_name, "jepa", "async-ticks"],
            ))
    return plan


def build_st17_halt_multitick(plan):
    """Halt + multi-tick loss."""
    for task_name, (module, base, _) in TASKS.items():
        cfg = dict(with_seed(base, 0))
        cfg["tick_halt_mode"] = "threshold"
        cfg["tick_halt_threshold"] = 0.6
        cfg["tick_loss_mode"] = "mean"
        plan.append(exp(
            f"st17_{task_name}_halt_multitick",
            f"{task_name}: halt@0.6+multi-tick",
            _p(module, {**cfg, "log_dir": f"logs/ctm_paper/st17/{task_name}_halt_multitick"}),
            tags=[task_name, "tick-halt", "multi-tick-loss"],
        ))
    return plan


def build_st18_sparsity_async(plan):
    """Top-k sparsity + async banded ticks."""
    for task_name, (module, base, _) in TASKS.items():
        if task_name in ("qamnist",):
            continue
        d = base.get("d_model", 512)
        for frac in [0.5, 0.75]:
            topk = max(8, int(d * frac))
            for periods in ["1,2,4", "1,2,4,8"]:
                cfg = dict(with_seed(base, 0))
                cfg["topk_neurons"] = topk
                cfg["async_tick_mode"] = "banded"
                cfg["async_tick_periods"] = periods
                plan.append(exp(
                    f"st18_{task_name}_sparsity{str(frac).replace('.','p')}_async_{periods.replace(',','_')}",
                    f"{task_name}: sparsity topk={topk}+async periods={periods}",
                    _p(module, {**cfg, "log_dir": f"logs/ctm_paper/st18/{task_name}_sparsity_async"}),
                    tags=[task_name, "sparsity", "async-ticks"],
                ))
    return plan


def build_st19_reflex_halt(plan):
    """Reflex head + halt."""
    for task_name, (module, base, _) in TASKS.items():
        cfg = dict(with_seed(base, 0))
        cfg["reflex_head"] = True
        cfg["reflex_weight"] = 0.2
        cfg["reflex_ticks"] = 1
        cfg["reflex_distill"] = True
        cfg["tick_halt_mode"] = "threshold"
        cfg["tick_halt_threshold"] = 0.6
        plan.append(exp(
            f"st19_{task_name}_reflex_halt",
            f"{task_name}: reflex w=0.2+halt@0.6",
            _p(module, {**cfg, "log_dir": f"logs/ctm_paper/st19/{task_name}_reflex_halt"}),
            tags=[task_name, "reflex", "tick-halt"],
        ))
    return plan


def build_st20_jepa_multitick(plan):
    """JEPA + multi-tick loss."""
    jepa = _jepa_defaults()
    for task_name, (module, base, _) in TASKS.items():
        cfg = dict(with_seed(base, 0))
        cfg["tick_loss_mode"] = "mean"
        cfg.update(jepa)
        plan.append(exp(
            f"st20_{task_name}_jepa_multitick",
            f"{task_name}: JEPA+multi-tick",
            _p(module, {**cfg, "log_dir": f"logs/ctm_paper/st20/{task_name}_jepa_multitick"}),
            tags=[task_name, "jepa", "multi-tick-loss"],
        ))
    return plan


def build_st21_jepa_reflex(plan):
    """JEPA + reflex head."""
    jepa = _jepa_defaults()
    for task_name, (module, base, _) in TASKS.items():
        cfg = dict(with_seed(base, 0))
        cfg["reflex_head"] = True
        cfg["reflex_weight"] = 0.2
        cfg["reflex_ticks"] = 1
        cfg["reflex_distill"] = True
        cfg.update(jepa)
        plan.append(exp(
            f"st21_{task_name}_jepa_reflex",
            f"{task_name}: JEPA+reflex w=0.2",
            _p(module, {**cfg, "log_dir": f"logs/ctm_paper/st21/{task_name}_jepa_reflex"}),
            tags=[task_name, "jepa", "reflex"],
        ))
    return plan


def build_st22_sparsity_multitick(plan):
    """Sparsity + multi-tick loss."""
    for task_name, (module, base, _) in TASKS.items():
        d = base.get("d_model", 512)
        for frac in [0.5, 0.75]:
            topk = max(8, int(d * frac))
            cfg = dict(with_seed(base, 0))
            cfg["topk_neurons"] = topk
            cfg["tick_loss_mode"] = "mean"
            plan.append(exp(
                f"st22_{task_name}_sparsity{str(frac).replace('.','p')}_multitick",
                f"{task_name}: sparsity topk={topk}+multi-tick",
                _p(module, {**cfg, "log_dir": f"logs/ctm_paper/st22/{task_name}_sparsity_multitick"}),
                tags=[task_name, "sparsity", "multi-tick-loss"],
            ))
    return plan


def build_st23_async_multitick(plan):
    """Async ticks + multi-tick loss."""
    for task_name, (module, base, _) in TASKS.items():
        if task_name in ("qamnist",):
            continue
        for periods in ["1,2,4", "1,2,4,8"]:
            cfg = dict(with_seed(base, 0))
            cfg["async_tick_mode"] = "banded"
            cfg["async_tick_periods"] = periods
            cfg["tick_loss_mode"] = "mean"
            plan.append(exp(
                f"st23_{task_name}_async_{periods.replace(',','_')}_multitick",
                f"{task_name}: async periods={periods}+multi-tick",
                _p(module, {**cfg, "log_dir": f"logs/ctm_paper/st23/{task_name}_async_multitick"}),
                tags=[task_name, "async-ticks", "multi-tick-loss"],
            ))
    return plan


def build_st24_jepa_halt_sparsity(plan):
    """JEPA + halt + sparsity triple combo."""
    jepa = _jepa_defaults()
    for task_name, (module, base, _) in TASKS.items():
        if task_name in ("mazes", "qamnist"):
            continue
        d = base.get("d_model", 512)
        for frac in [0.5]:
            topk = max(8, int(d * frac))
            cfg = dict(with_seed(base, 0))
            cfg["tick_halt_mode"] = "threshold"
            cfg["tick_halt_threshold"] = 0.6
            cfg["topk_neurons"] = topk
            cfg.update(jepa)
            plan.append(exp(
                f"st24_{task_name}_jepa_halt_sparsity",
                f"{task_name}: JEPA+halt@0.6+sparsity",
                _p(module, {**cfg, "log_dir": f"logs/ctm_paper/st24/{task_name}_jepa_halt_sparsity"}),
                tags=[task_name, "jepa", "tick-halt", "sparsity"],
            ))
    return plan


# ─── Registry ───

STAGE_BUILDERS = {
    "st00": build_st00_paper_reproduction,
    "st01": build_st01_architecture_sweep,
    "st02": build_st02_tick_sweep,
    "st03": build_st03_simplified_core,
    "st04": build_st04_jepa_sweep,
    "st05": build_st05_jepa_plus,
    "st06": build_st06_tick_halt,
    "st07": build_st07_multi_tick_loss,
    "st08": build_st08_cell_sparsity,
    "st09": build_st09_reflex_head,
    "st10": build_st10_draft_revise,
    "st11": build_st11_async_ticks,
    "st12": build_st12_ema_speed_spectrum,
    "st13": build_st13_jepa_halt,
    "st14": build_st14_jepa_sparsity,
    "st15": build_st15_halt_sparsity,
    "st16": build_st16_jepa_async,
    "st17": build_st17_halt_multitick,
    "st18": build_st18_sparsity_async,
    "st19": build_st19_reflex_halt,
    "st20": build_st20_jepa_multitick,
    "st21": build_st21_jepa_reflex,
    "st22": build_st22_sparsity_multitick,
    "st23": build_st23_async_multitick,
    "st24": build_st24_jepa_halt_sparsity,
}

STAGE_DESCRIPTIONS = {
    "st00": "Paper reproduction + LSTM/FF baselines (ready)",
    "st01": "Architecture sweep: d_model, sd, mh, heads, NST (ready)",
    "st02": "Tick count sweep (ready)",
    "st03": "Simplified core sd2_mh2 at tick2 (ready)",
    "st04": "Cross-tick JEPA weight/loss/stopgrad sweeps (ready)",
    "st05": "JEPA + tick2/simplified combos (ready)",
    "st06": "Tick early-exit/halt with compute penalty (ready)",
    "st07": "Multi-tick loss / anytime supervision (ready)",
    "st08": "Cell sparsity top-k activation (ready)",
    "st09": "Reflex output head (ready)",
    "st10": "Draft-revise loop (ready)",
    "st11": "Async banded tick clocks (ready)",
    "st12": "EMA speed spectrum distillation (ready)",
    "st13": "JEPA + tick halt (ready)",
    "st14": "JEPA + top-k sparsity (ready)",
    "st15": "Halt + sparsity (ready)",
    "st16": "JEPA + async ticks (ready)",
    "st17": "Halt + multi-tick loss (ready)",
    "st18": "Sparsity + async ticks (ready)",
    "st19": "Reflex head + halt (ready)",
    "st20": "JEPA + multi-tick loss (ready)",
    "st21": "JEPA + reflex head (ready)",
    "st22": "Sparsity + multi-tick loss (ready)",
    "st23": "Async ticks + multi-tick loss (ready)",
    "st24": "JEPA + halt + sparsity triple combo (ready)",
}


def _expand_seeds(plan, seeds=[0, 1, 2]):
    """Expand experiments to 3 seeds (skip st00 paper reproductions)."""
    new_plan = []
    for e in plan:
        if e["name"].startswith("st00_"):
            new_plan.append(e)
            continue
        if e.get("impl_status") != "ready" and not e["name"].startswith("st10_"):
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
                "impl_status": e.get("impl_status", "ready"),
            })
    return new_plan


def build_plan(stage="all", seed_expand=True):
    plan = []
    if stage == "all":
        for s in STAGES_ORDERED:
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
    ready = sum(1 for e in plan if e.get("impl_status") == "ready")
    needs_impl = sum(1 for e in plan if e.get("impl_status") != "ready")
    return ready, needs_impl


def print_plan(plan):
    ready, needs_impl = count_by_status(plan)
    print(f"\n{'='*80}")
    print(f"CTM PAPER EXPERIMENT PLAN — {len(plan)} experiments total")
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
        status = f"[{r} ready, {ni} needs_impl]"
        print(f"\n  ── {stage_name}: {desc} {status} ──")
        for e in exps:
            impl_mark = " ⚠" if e.get("impl_status") != "ready" else ""
            print(f"     {e['name']}{impl_mark}")
            print(f"       {e['question']}")

    print(f"\n{'='*80}")
    print(f"TOTAL: {len(plan)} ({ready} ready, {needs_impl} needs implementation)")
    print(f"{'='*80}\n")


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
            "CTM_LOG_DIR": "runs/logs/ctm_paper",
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


def wait_until_idle(master_addr, port, task_id, experiment_name=None, poll_interval=30.0):
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
    ready_only = [e for e in plan if e.get("impl_status") == "ready"]
    if args.include_unimplemented:
        ready_only = plan

    if not ready_only:
        print("No ready experiments to submit. Implement needed ideas first.")
        return

    print(f"Submitting {len(ready_only)} experiments to pool at {MASTER_ADDR}:{PORT}")
    for e in ready_only:
        print(f"  {e['name']}")

    if args.dry_run:
        return

    for e in ready_only:
        print(f"Submitting {e['name']}...")
        result = submit_to_pool(e, POOL_CONFIG, MASTER_ADDR, PORT)
        if result is None:
            print(f"  FAILED to submit {e['name']}")
            continue
        task_id = result if isinstance(result, str) else result.get("task_id", "")
        print(f"  task_id={task_id}")
        if args.wait:
            final_status = wait_until_idle(MASTER_ADDR, PORT, task_id, e["name"])
            print(f"  -> {final_status}")
            if final_status == "failed" and args.stop_on_fail:
                print("Stopping due to failure.")
                break
            time.sleep(5)


def cmd_csv(args):
    import csv
    plan = build_plan(args.stage)
    path = args.output or f"runs/experiment_plans/ctm_paper_plan.csv"
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
                "impl_status": e.get("impl_status", "ready"),
            })
    print(f"Wrote {len(plan)} experiments to {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CTM Paper: All Ideas × All Tasks")
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
                          help="Submit experiments for ideas not yet implemented")
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
