#!/usr/bin/env python3
"""
Multi-Scale Hierarchy (MSH) — N-Level Jacobian Decoupling for CTM.

Mathematical motivation:
  Traditional RNN: T-step Jacobian product ∏J_t is unstable.
  HRM (2-level): splits into ∏J_H · (∏J_L), where ρ(J_L)<1 → safe.
  
  Generalization to N levels:
    levels = [l_0, l_1, ..., l_{N-1}],  T = ∏ l_i
    Each level i has spectral radius budget: ρ(J_i) ≤ l_i^{-1/l_i}
    Shorter l_i → larger ρ budget → more expressive per step
    Gradient path = l_{N-1} (outermost level updates)
    
    With levels[-1]=1: one-step gradient at top level,
    while maintaining T=∏l_i steps of rich multi-scale dynamics.

Example level configurations for T=50:
  [10, 5]      → 2-level, gradient path=5  (HRM default)
  [5, 5, 2]    → 3-level, gradient path=2
  [10, 5, 1]   → 3-level, gradient path=1  (one-step gradient!)
  [5, 5, 2, 1] → 4-level, gradient path=1  (one-step, richer dynamics)

Usage:
    python scripts/experiment_plan_msh.py plan [--stage all]
    python scripts/experiment_plan_msh.py submit --stage msh02 --no-wait

Code changes required:
  - baseline/utils/dtt_ideas.py: add_msh_args(), MSH forward logic
  - baseline/tasks/sort/train.py: wire MSH args + build level synapses
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
    "msh00",
    "msh01",
    "msh02",
    "msh03",
    "msh04",
    "msh05",
    "msh06",
    "msh07",
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
# Base configs
# ═══════════════════════════════════════════════════════════════

SORT_BASE = dict(
    seed=0, iterations=50, memory_length=25,
    d_model=512, d_input=128, n_synch_out=32, n_synch_action=32,
    synapse_depth=4, heads=4, memory_hidden_dims=4, dropout=0.0,
    deep_memory=True, do_normalisation=False,
    positional_embedding_type="none",
    neuron_select_type="random-pairing",
    n_random_pairing_self=0, N_to_sort=10,
    batch_size=32, batch_size_test=32,
    lr=1e-3, training_iterations=100001,
    warmup_steps=5000, use_scheduler=True, scheduler_type="cosine",
    weight_decay=0.0, gradient_clipping=-1,
    track_every=1000, save_every=10000,
    reload=False, device=[0],
    sort_loss_mode="per_tick_ce",
)

SORT_BASE_N30 = dict(SORT_BASE, N_to_sort=30)

TASKS = {
    "sort_N10": ("baseline.tasks.sort.train", SORT_BASE, 50),
    "sort_N30": ("baseline.tasks.sort.train", SORT_BASE_N30, 50),
}


def with_seed(cfg, seed):
    c = dict(cfg)
    c["seed"] = seed
    return c


# ═══════════════════════════════════════════════════════════════
# STAGE BUILDERS
# ═══════════════════════════════════════════════════════════════

def build_msh00_baselines(plan):
    """Baselines: flat CTM (no hierarchy) with per-tick CE loss."""
    for task_name in ["sort_N10", "sort_N30"]:
        module, base, _ = TASKS[task_name]
        cfg = dict(base)
        cfg["log_dir"] = f"logs/msh/msh00/{task_name}_flat_per_tick_ce"
        plan.append(exp(
            f"msh00_{task_name}_flat",
            f"{task_name}: flat CTM, per-tick CE, full BPTT (control)",
            _p(module, cfg),
            tags=[task_name, "baseline", "flat", "per-tick-ce"],
            impl_status="ready",
            hypothesis="Flat CTM with per-tick CE and full BPTT — accuracy target.",
        ))
    return plan


# ─── msh01: 2-Level Hierarchy Sweep ───

def build_msh01_2level(plan):
    """2-level hierarchy: validate existing h_cycles/l_cycles with per-tick CE.

    Tests different L/H splits for T=50:
      [10, 5]:  L=10, H=5  (HRM default, gradient path=5)
      [25, 2]:  L=25, H=2  (short gradient path)
      [50, 1]:  L=50, H=1  (one-step gradient, equivalent to bp_steps=1)
    """
    module, base, _ = TASKS["sort_N10"]
    configs = [
        ("10,5",  "L=10 H=5, grad_path=5"),
        ("25,2",  "L=25 H=2, grad_path=2"),
        ("50,1",  "L=50 H=1, grad_path=1 (one-step)"),
    ]
    for levels_str, desc in configs:
        cfg = dict(with_seed(base, 0))
        cfg["msh_levels"] = levels_str
        cfg["log_dir"] = f"logs/msh/msh01/sort_N10_2level_{levels_str.replace(',','x')}"
        plan.append(exp(
            f"msh01_sort_N10_2level_{levels_str.replace(',','x')}",
            f"sort(N=10): 2-level [{levels_str}] — {desc}",
            _p(module, cfg),
            tags=["sort", "N10", "2level", f"levels_{levels_str}"],
            impl_status="needs_impl",
            hypothesis=f"2-level [{levels_str}] with per-tick CE. "
                       f"Gradient path={'1 (one-step)' if levels_str.endswith(',1') else levels_str.split(',')[1]}.",
        ))
    return plan


# ─── msh02: 3-Level Hierarchy (CORE) ───

def build_msh02_3level(plan):
    """3-level hierarchy: the core contribution.

    Tests whether adding a middle level improves over 2-level.
    Key comparison: [10,5,1] (3-level one-step) vs [50,1] (2-level one-step).

    Level configs for T=50:
      [2, 5, 5]:  fast=2, mid=5, slow=5, grad_path=5
      [5, 5, 2]:  fast=5, mid=5, slow=2, grad_path=2
      [10, 5, 1]: fast=10, mid=5, slow=1, grad_path=1 (one-step!)
      [2, 25, 1]: fast=2, mid=25, slow=1, grad_path=1 (asymmetric)
    """
    module_n10, base_n10, _ = TASKS["sort_N10"]
    module_n30, base_n30, _ = TASKS["sort_N30"]

    configs_n10 = [
        ("2,5,5",   "fast=2 mid=5 slow=5, grad_path=5"),
        ("5,5,2",   "fast=5 mid=5 slow=2, grad_path=2"),
        ("10,5,1",  "fast=10 mid=5 slow=1, ONE-STEP"),
        ("2,25,1",  "fast=2 mid=25 slow=1, asymmetric one-step"),
    ]
    for levels_str, desc in configs_n10:
        cfg = dict(with_seed(base_n10, 0))
        cfg["msh_levels"] = levels_str
        cfg["log_dir"] = f"logs/msh/msh02/sort_N10_3level_{levels_str.replace(',','x')}"
        plan.append(exp(
            f"msh02_sort_N10_3level_{levels_str.replace(',','x')}",
            f"sort(N=10): 3-level [{levels_str}] — {desc}",
            _p(module_n10, cfg),
            tags=["sort", "N10", "3level", f"levels_{levels_str}"],
            impl_status="needs_impl",
            hypothesis=f"3-level [{levels_str}]: adding middle level gives each level "
                       f"a larger spectral radius budget vs 2-level.",
        ))

    # N=30 for the one-step config only
    cfg_n30 = dict(with_seed(base_n30, 0))
    cfg_n30["msh_levels"] = "10,5,1"
    cfg_n30["log_dir"] = f"logs/msh/msh02/sort_N30_3level_10x5x1"
    plan.append(exp(
        f"msh02_sort_N30_3level_10x5x1",
        "sort(N=30): 3-level [10,5,1] — one-step gradient on larger sort",
        _p(module_n30, cfg_n30),
        tags=["sort", "N30", "3level", "one-step"],
        impl_status="needs_impl",
        hypothesis="3-level one-step [10,5,1] on N=30: does multi-scale help "
                   "where 2-level one-step [50,1] failed?",
    ))
    return plan


# ─── msh03: 4-Level Hierarchy ───

def build_msh03_4level(plan):
    """4-level hierarchy: extreme decomposition.

    With levels[-1]=1, achieves one-step gradient with 4 scales of dynamics.
    Key question: does the extra level add value or just complexity?

    Configs for T=50:
      [5, 5, 2, 1]: 4-level one-step, balanced
      [2, 5, 5, 1]: 4-level one-step, fast inner
      [10, 5, 1, 1]: degenerate (same as 3-level [10,5,1])
    """
    module, base, _ = TASKS["sort_N10"]
    configs = [
        ("5,5,2,1",  "balanced 4-level one-step"),
        ("2,5,5,1",  "fast-inner 4-level one-step"),
    ]
    for levels_str, desc in configs:
        cfg = dict(with_seed(base, 0))
        cfg["msh_levels"] = levels_str
        cfg["log_dir"] = f"logs/msh/msh03/sort_N10_4level_{levels_str.replace(',','x')}"
        plan.append(exp(
            f"msh03_sort_N10_4level_{levels_str.replace(',','x')}",
            f"sort(N=10): 4-level [{levels_str}] — {desc}",
            _p(module, cfg),
            tags=["sort", "N10", "4level", f"levels_{levels_str}"],
            impl_status="needs_impl",
            hypothesis=f"4-level [{levels_str}]: extra temporal scale vs 3-level. "
                       f"Diminishing returns expected but may help on complex tasks.",
        ))
    return plan


# ─── msh04: Spectral Normalization ───

def build_msh04_spectral(plan):
    """Spectral normalization on level synapses: enforce contractive dynamics.

    ρ(J_i) ≤ l_i^{-1/l_i} ensures each level's Jacobian product
    stays bounded. Spectral norm with scale < 1 enforces this.

    Tests: [10,5,1] with and without spectral norm.
    """
    module, base, _ = TASKS["sort_N10"]
    for sn_scale in [0.0, 0.8, 0.9, 0.95]:
        cfg = dict(with_seed(base, 0))
        cfg["msh_levels"] = "10,5,1"
        if sn_scale > 0:
            cfg["msh_sn_scale"] = sn_scale
        label = f"sn{str(sn_scale).replace('.','p')}" if sn_scale > 0 else "no_sn"
        cfg["log_dir"] = f"logs/msh/msh04/sort_N10_3level_10x5x1_{label}"
        plan.append(exp(
            f"msh04_sort_N10_3level_10x5x1_{label}",
            f"sort(N=10): 3-level [10,5,1] + spectral_norm={sn_scale}",
            _p(module, cfg),
            tags=["sort", "N10", "3level", "spectral", f"sn{sn_scale}"],
            impl_status="needs_impl",
            hypothesis=f"Spectral norm scale={sn_scale}: {'enforces ρ(J)<1 for '
                       f'contractive lower levels' if sn_scale > 0 else 'no constraint, baseline dynamics'}.",
        ))
    return plan


# ─── msh05: MSH + bp_steps ───

def build_msh05_bp_steps(plan):
    """MSH + gradient truncation: combine hierarchy with bp_steps.

    With N-level hierarchy, bp_steps controls how many ticks at the
    TOP level get gradient. bp_steps=1 = one-step gradient.

    Tests: [5,5,2] with bp_steps = 1, 2
    (gradient path = 2, so bp_steps ≤ 2 covers full macro path)
    """
    module, base, _ = TASKS["sort_N10"]
    for levels_str, bp_list in [
        ("5,5,2",  [1, 2]),
        ("10,5,1", [1]),       # already one-step at top level
        ("2,5,5",  [1, 5]),
    ]:
        for bp in bp_list:
            cfg = dict(with_seed(base, 0))
            cfg["msh_levels"] = levels_str
            cfg["bp_steps"] = bp
            cfg["log_dir"] = f"logs/msh/msh05/sort_N10_{levels_str.replace(',','x')}_bp{bp}"
            plan.append(exp(
                f"msh05_sort_N10_{levels_str.replace(',','x')}_bp{bp}",
                f"sort(N=10): [{levels_str}] + bp_steps={bp}",
                _p(module, cfg),
                tags=["sort", "N10", "msh", f"levels_{levels_str}", f"bp{bp}"],
                impl_status="needs_impl",
                hypothesis=f"[{levels_str}] + bp={bp}: hierarchy structures the "
                           f"dynamics, bp truncates the gradient path.",
            ))
    return plan


# ─── msh06: Level Ablation (fix grad_path=1, vary structure) ───

def build_msh06_ablation(plan):
    """Level ablation: fix gradient_path=1, vary the inner structure.

    All configs have levels[-1]=1 (one-step gradient).
    The question: how should the remaining T-1 steps be distributed?

    [50, 1]:     2-level, all in L     (brute: one big inner loop)
    [10, 5, 1]:  3-level, balanced     (multi-scale)
    [2, 25, 1]:  3-level, asymmetric   (fast inner, big middle)
    [5, 5, 2, 1]: 4-level, balanced    (finest multi-scale)
    [2, 5, 5, 1]: 4-level, fast-inner  (different multi-scale)
    """
    module, base, _ = TASKS["sort_N10"]
    configs = [
        ("50,1",    "2-level one-step (baseline for comparison)"),
        ("10,5,1",  "3-level balanced"),
        ("2,25,1",  "3-level asymmetric"),
        ("5,5,2,1", "4-level balanced"),
        ("2,5,5,1", "4-level fast-inner"),
    ]
    for levels_str, desc in configs:
        n_levels = len(levels_str.split(','))
        cfg = dict(with_seed(base, 0))
        cfg["msh_levels"] = levels_str
        cfg["log_dir"] = f"logs/msh/msh06/sort_N10_onestep_{levels_str.replace(',','x')}"
        plan.append(exp(
            f"msh06_sort_N10_onestep_{levels_str.replace(',','x')}",
            f"sort(N=10): one-step [{levels_str}] ({n_levels}L) — {desc}",
            _p(module, cfg),
            tags=["sort", "N10", "one-step", "ablation", f"{n_levels}level"],
            impl_status="needs_impl",
            hypothesis=f"[{levels_str}] ({n_levels} levels): all have grad_path=1. "
                       f"More levels = each level has larger ρ budget = more expressive. "
                       f"Predict: 3L > 2L, 4L ≈ 3L (diminishing returns).",
        ))
    return plan


# ─── msh07: Full System ───

def build_msh07_full(plan):
    """Full MSH system: best hierarchy + spectral norm + atan2 + progressive.

    Combines all winning components from msh01-msh06.
    """
    module_n10, base_n10, _ = TASKS["sort_N10"]
    module_n30, base_n30, _ = TASKS["sort_N30"]

    for task_name, module, base in [("N10", module_n10, base_n10), ("N30", module_n30, base_n30)]:
        for levels_str in ["10,5,1", "5,5,2,1"]:
            cfg = dict(with_seed(base, 0))
            cfg["msh_levels"] = levels_str
            cfg["msh_sn_scale"] = 0.9
            cfg["optimizer_type"] = "adam_atan2"
            cfg["beta2"] = 0.95
            cfg["dtt_progressive_mode"] = "certainty"
            label = levels_str.replace(',', 'x')
            cfg["log_dir"] = f"logs/msh/msh07/sort_{task_name}_full_{label}"
            plan.append(exp(
                f"msh07_sort_{task_name}_full_{label}",
                f"sort({task_name}): full MSH [{levels_str}] + SN + atan2 + progressive",
                _p(module, cfg),
                tags=["sort", task_name, "full-msh", f"levels_{levels_str}"],
                impl_status="needs_impl",
                hypothesis=f"Full system [{levels_str}]: all components combined. "
                           f"Target: match or exceed flat baseline accuracy at "
                           f"fraction of compute cost (one-step gradient).",
            ))
    return plan


# ─── Registry ───

STAGE_BUILDERS = {
    "msh00": build_msh00_baselines,
    "msh01": build_msh01_2level,
    "msh02": build_msh02_3level,
    "msh03": build_msh03_4level,
    "msh04": build_msh04_spectral,
    "msh05": build_msh05_bp_steps,
    "msh06": build_msh06_ablation,
    "msh07": build_msh07_full,
}

STAGE_DESCRIPTIONS = {
    "msh00": "Baselines: flat CTM with per-tick CE (control group)",
    "msh01": "2-level hierarchy sweep: different L/H splits for T=50",
    "msh02": "3-level hierarchy: does a middle level help? [CORE]",
    "msh03": "4-level hierarchy: extreme temporal decomposition",
    "msh04": "Spectral normalization: enforce ρ(J)<1 contractive dynamics",
    "msh05": "MSH + bp_steps: combine hierarchy with gradient truncation",
    "msh06": "Level ablation: fix grad_path=1, vary inner structure [KEY ABLATION]",
    "msh07": "Full system: best hierarchy + spectral + atan2 + progressive",
}

IMPL_PRIORITY = {
    "msh02": 1,
    "msh06": 2,
    "msh01": 3,
    "msh04": 4,
    "msh07": 5,
    "msh05": 6,
    "msh03": 7,
    "msh00": 8,
}


def build_plan(stage="all"):
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
    return plan


def count_by_status(plan):
    ready = sum(1 for e in plan if e.get("impl_status") == "ready")
    needs_impl = sum(1 for e in plan if e.get("impl_status") != "ready")
    return ready, needs_impl


def print_plan(plan):
    ready, needs_impl = count_by_status(plan)
    print(f"\n{'='*80}")
    print(f"MULTI-SCALE HIERARCHY (MSH) — {len(plan)} experiments total")
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
    print(f"  1. baseline/utils/dtt_ideas.py — add_msh_args() + MSH forward helper")
    print(f"  2. baseline/tasks/sort/train.py — wire MSH args + build level synapses")
    print(f"  3. baseline/models/ctm_sort.py — N-level forward loop")
    print(f"{'='*80}\n")


def submit_to_pool(exp_entry, config, master_addr=None, port=None):
    payload = {
        "config": config,
        "extra_args": exp_entry["command"],
        "node_addrs": [],
        "env": {
            "CTM_EXPERIMENT_NAME": exp_entry["name"],
            "CTM_METRICS_DIR": "runs/metrics",
            "CTM_LOG_DIR": "runs/logs/msh",
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
            for t in status.get("tasks", []):
                if t["task_id"] == task_id and t["status"] in final:
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
    path = args.output or f"runs/experiment_plans/msh_plan.csv"
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


def cmd_run_local(args):
    """Print ready-to-paste shell commands for local multi-GPU execution."""
    plan = build_plan(args.stage)
    to_run = plan if args.include_needs_impl else [e for e in plan if e.get("impl_status") == "ready"]
    if not to_run:
        print("No experiments to run.")
        return
    print(f"# {len(to_run)} experiments, distributing across {args.gpus} GPUs\n")
    for i, e in enumerate(to_run):
        gpu = i % args.gpus
        cmd = e["command"]
        cmd = cmd.replace("--device 0", f"--device 0")
        log = e.get("name", f"exp{i}")
        print(f"# GPU {gpu}: {e['name']}")
        print(f"CUDA_VISIBLE_DEVICES={gpu} python -m {cmd} 2>&1 | tee logs/msh/{log}.log &\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Scale Hierarchy (MSH) Experiments")
    sub = parser.add_subparsers(dest="command")

    p_plan = sub.add_parser("plan", help="Show experiment plan")
    p_plan.add_argument("--stage", default="all")

    p_submit = sub.add_parser("submit", help="Submit experiments to pool")
    p_submit.add_argument("--stage", default="all")
    p_submit.add_argument("--dry-run", action="store_true")
    p_submit.add_argument("--wait", action="store_true", default=True)
    p_submit.add_argument("--no-wait", action="store_false", dest="wait")
    p_submit.add_argument("--stop-on-fail", action="store_true", default=True)
    p_submit.add_argument("--ready-only", action="store_true", default=True)
    p_submit.add_argument("--include-needs-impl", action="store_false", dest="ready_only")
    p_submit.add_argument("--master-addr", default=MASTER_ADDR)
    p_submit.add_argument("--port", type=int, default=PORT)

    p_csv = sub.add_parser("csv", help="Export plan to CSV")
    p_csv.add_argument("--stage", default="all")
    p_csv.add_argument("--output", default=None)

    p_run = sub.add_parser("run", help="Print local multi-GPU shell commands")
    p_run.add_argument("--stage", default="all")
    p_run.add_argument("--gpus", type=int, default=8)
    p_run.add_argument("--include-needs-impl", action="store_true", default=False)

    args = parser.parse_args()
    if args.command == "plan":
        cmd_plan(args)
    elif args.command == "submit":
        cmd_submit(args)
    elif args.command == "csv":
        cmd_csv(args)
    elif args.command == "run":
        cmd_run_local(args)
    else:
        parser.print_help()
