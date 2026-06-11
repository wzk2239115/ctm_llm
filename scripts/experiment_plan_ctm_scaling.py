#!/usr/bin/env python3
"""
CTM Scaling Experiment Plan — 1x / 2x / 4x / 8x d_model for all 5 tasks.

Each scale runs plain CTM AND CTM+JEPA+halt+sparsity (best combo).
Shows whether our ideas scale up with model size.

Usage:
    python scripts/experiment_plan_ctm_scaling.py plan [--stage all]
    python scripts/experiment_plan_ctm_scaling.py submit --stage all --no-wait
    python scripts/experiment_plan_ctm_scaling.py csv [--stage all]
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

STAGES_ORDERED = ["st_plain"]
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


# ─── Per-task base configs (pulled from experiment_plan_ctm_paper.py) ───

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
    "sort": ("baseline.tasks.sort.train", SORT_BASE, 50),
    "parity": ("baseline.tasks.parity.train", PARITY_BASE, 75),
    "mazes": ("baseline.tasks.mazes.train", MAZES_BASE, 75),
    "cifar10": ("baseline.tasks.image_classification.train", CIFAR10_BASE, 50),
    "qamnist": ("baseline.tasks.qamnist.train", QAMNIST_BASE, None),
}

SEEDS = [0, 1, 2]


# ─── Scaling helpers ───

SCALE_FACTORS = {
    "sort":   [1, 2, 4, 8],
    "parity": [1, 2, 4],
    "mazes":  [1, 2],
    "cifar10":[1, 2, 4, 8],
    "qamnist":[1, 2, 4],
}


def scale_cfg(base, factor):
    c = dict(base)
    c["d_model"] = int(base["d_model"] * factor)
    for k in ("d_input", "heads", "memory_hidden_dims", "n_synch_out", "n_synch_action"):
        if k in base:
            c[k] = max(1, int(base[k] * factor))
    return c


# ═══════════════════════════════════════════════════════════════
# STAGE BUILDERS
# ═══════════════════════════════════════════════════════════════

def build_st_plain(plan):
    """All tasks × all scales × plain CTM + CTM+JEPA+halt+sparsity."""
    jepa = dict(
        cross_tick_jepa_weight=1.0,
        cross_tick_jepa_loss="cosine",
        cross_tick_jepa_hidden_dim=128,
        cross_tick_jepa_predictor_depth=2,
        cross_tick_jepa_dropout=0.0,
        cross_tick_jepa_target_stop_grad=True,
    )
    for task_name, (module, base, _) in TASKS.items():
        for f in SCALE_FACTORS.get(task_name, [1]):
            # Plain CTM
            cfg = scale_cfg(base, f)
            plan.append(exp(
                f"st_plain_{task_name}_{f}x",
                f"{task_name}: {f}× d_model={cfg['d_model']} plain",
                _p(module, {**cfg, "log_dir": f"logs/ctm_scaling/{task_name}_{f}x_plain"}),
                tags=[task_name, f"scale-{f}x", "plain"],
            ))
            # CTM + ideas
            cfg_i = scale_cfg(base, f)
            cfg_i["tick_halt_mode"] = "threshold"
            cfg_i["tick_halt_threshold"] = 0.6
            cfg_i["topk_neurons"] = max(8, int(cfg_i["d_model"] * 0.5))
            cfg_i.update(jepa)
            plan.append(exp(
                f"st_ideas_{task_name}_{f}x",
                f"{task_name}: {f}× d_model={cfg_i['d_model']} + ideas",
                _p(module, {**cfg_i, "log_dir": f"logs/ctm_scaling/{task_name}_{f}x_ideas"}),
                tags=[task_name, f"scale-{f}x", "jepa", "tick-halt", "sparsity"],
            ))
    return plan


# ─── Registry ───

STAGE_BUILDERS = {
    "st_plain": build_st_plain,
}

STAGE_DESCRIPTIONS = {
    "st_plain": "All tasks × scales (1x,2x,4x,8x) × plain & ideas",
}


def _expand_seeds(plan, seeds=None):
    if seeds is None:
        seeds = SEEDS
    new_plan = []
    for e in plan:
        tags = e.get("tags", [])
        existing_seeds = [int(t[4:]) for t in tags if t.startswith("seed")]
        existing_seed = existing_seeds[0] if existing_seeds else 0
        base_entry = dict(e)
        base_entry["tags"] = [t for t in tags if not t.startswith("seed")] + [f"seed{existing_seed}"]
        if "--seed" not in base_entry["command"]:
            base_entry["command"] += f" --seed {existing_seed}"
        new_plan.append(base_entry)
        for seed in seeds:
            if seed == existing_seed:
                continue
            log_dir_match = re.search(r'--log_dir\s+(\S+)', e["command"])
            new_cmd = re.sub(r'--seed\s+\d+', f'--seed {seed}', e["command"])
            if log_dir_match:
                base_dir = log_dir_match.group(1).rstrip("/")
                new_cmd = re.sub(r'--log_dir\s+\S+', f'--log_dir {base_dir}_s{seed}', new_cmd)
            new_tags = [t for t in tags if not t.startswith("seed")] + [f"seed{seed}"]
            new_plan.append({
                "name": f"{e['name']}_s{seed}",
                "question": e["question"],
                "command": new_cmd,
                "tags": new_tags,
                "node_addr": _next_slot(),
                "impl_status": "ready",
            })
    return new_plan


def build_plan(stage="all", seed_expand=True):
    plan = []
    if stage == "all" or stage == "st_plain":
        STAGE_BUILDERS["st_plain"](plan)
    elif stage in STAGE_BUILDERS:
        STAGE_BUILDERS[stage](plan)
    else:
        print(f"Unknown stage: {stage}. Available: {list(STAGE_BUILDERS)}")
        return []
    if seed_expand:
        plan = _expand_seeds(plan)
    return plan


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def count_by_status(plan):
    ready = sum(1 for e in plan if e.get("impl_status") == "ready")
    needs_impl = sum(1 for e in plan if e.get("impl_status") != "ready")
    return ready, needs_impl


def print_plan(plan):
    ready, needs_impl = count_by_status(plan)
    print(f"\n{'='*80}")
    print(f"  CTM Scaling Experiment Plan")
    print(f"{'='*80}")
    stages = {}
    for e in plan:
        stage = e["name"][:e["name"].index("_", 3)] if "_" in e["name"][3:] else "unknown"
        stages.setdefault(stage, []).append(e)
    for stage_name in sorted(stages.keys()):
        exps = stages[stage_name]
        status = "ready" if all(e.get("impl_status") == "ready" for e in exps) else "needs-impl"
        desc = STAGE_DESCRIPTIONS.get(stage_name, "")
        print(f"\n  ── {stage_name}: {desc} [{status}] ──")
        for e in exps:
            print(f"     {e['name']}")
            print(f"       {e['question']}")
    print(f"\n{'='*80}")
    print(f"TOTAL: {len(plan)} ({ready} ready, {needs_impl} needs implementation)")
    print(f"{'='*80}\n")


def submit_to_pool(exp, config, master_addr=None, port=None):
    node_addrs = exp.get("node_addrs") or [exp.get("node_addr")]
    payload = {
        "config": config,
        "extra_args": exp["command"],
        "node_addrs": node_addrs,
        "env": {
            "CTM_EXPERIMENT_NAME": exp["name"],
            "CTM_METRICS_DIR": "runs/metrics",
            "CTM_LOG_DIR": "runs/logs/ctm_scaling",
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
    config = str(ROOT / POOL_CONFIG)
    plan = build_plan(args.stage)
    if args.dry_run:
        print(f"[dry-run] would submit {len(plan)} experiments to {args.master_addr}:{args.port}")
        for e in plan:
            print(f"  {e['name']}")
        return
    print(f"Submitting {len(plan)} experiments to {args.master_addr}:{args.port}...")
    for i, e in enumerate(plan):
        if e.get("impl_status") != "ready" and not args.include_unimplemented:
            print(f"  [{i+1}/{len(plan)}] skipping {e['name']} (impl_status={e['impl_status']})")
            continue
        result = submit_to_pool(e, config, master_addr=args.master_addr, port=args.port)
        if result:
            task_id = result.get("task_id") or result.get("id", "?")
            print(f"  [{i+1}/{len(plan)}] {e['name']} -> task {task_id}")
            if args.wait:
                wait_until_idle(args.master_addr, args.port, task_id)
        else:
            print(f"  [{i+1}/{len(plan)}] {e['name']} -> FAILED")
            if args.stop_on_fail:
                print("  Stopping due to --stop-on-fail")
                break
    print("Done.")


def cmd_csv(args):
    import csv
    plan = build_plan(args.stage)
    path = args.output or f"runs/experiment_plans/ctm_scaling_plan.csv"
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
    parser = argparse.ArgumentParser(description="CTM Scaling: model size sweep")
    sub = parser.add_subparsers(dest="command")

    p_plan = sub.add_parser("plan")
    p_plan.add_argument("--stage", default="all")

    p_submit = sub.add_parser("submit")
    p_submit.add_argument("--stage", default="all")
    p_submit.add_argument("--dry-run", action="store_true")
    p_submit.add_argument("--wait", action="store_true", default=True)
    p_submit.add_argument("--no-wait", action="store_false", dest="wait")
    p_submit.add_argument("--stop-on-fail", action="store_true", default=True)
    p_submit.add_argument("--include-unimplemented", action="store_true")
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
