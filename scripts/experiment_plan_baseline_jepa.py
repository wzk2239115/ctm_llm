#!/usr/bin/env python3
"""Baseline CTM Cross-Tick JEPA Experiments.

Tests whether cross-tick JEPA latent prediction improves multi-tick
representation quality on all 5 baseline tasks (sort, parity, mazes,
image_classification, qamnist).

Usage:
    # Dry-run: print plan
    python scripts/experiment_plan_baseline_jepa.py plan

    # Submit all to pool (sequential, waits for each to finish)
    python scripts/experiment_plan_baseline_jepa.py submit [--stage jx00]

    # Run a single experiment locally
    python scripts/experiment_plan_baseline_jepa.py run --name jx00_sort_baseline
"""

import json, os, shlex, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POOL_CONFIG = "infra/envs/h100_baseline.env"
MASTER_ADDR = "11.131.210.78"
PORT = 8765
BASELINE_NODES = ("11.131.209.154", "11.131.210.3", "11.131.210.78", "11.131.211.9")
GPUS_PER_NODE = 8
_slot_idx = [0]


def _next_slot():
    i = _slot_idx[0]
    _slot_idx[0] += 1
    node = BASELINE_NODES[i // GPUS_PER_NODE % len(BASELINE_NODES)]
    gpu = i % GPUS_PER_NODE
    return f"{node}:{gpu}"


def _p(train_module, extra_args=None):
    """Build command string: module --flag1 val1 --flag2 val2 ..."""
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


def exp(name, question, command, tags=None):
    return {
        "name": name,
        "question": question,
        "command": command,
        "tags": tags or [],
        "node_addr": _next_slot(),
    }


def build_plan(stage="all"):
    plan = []

    # ── Shared base hyperparameters per task ──
    sort_base = dict(
        seed=412, iterations=50, memory_length=25,
        d_model=512, d_input=128,
        n_synch_out=32, n_synch_action=32,
        synapse_depth=4, heads=4,
        memory_hidden_dims=4, dropout=0.0,
        deep_memory=True, do_normalisation=False,
        positional_embedding_type="none",
        neuron_select_type="random-pairing",
        n_random_pairing_self=0, N_to_sort=30,
        batch_size=32, batch_size_test=32,
        lr=1e-3, training_iterations=100001,
        warmup_steps=5000,
        use_scheduler=True, scheduler_type="cosine",
        weight_decay=0.0, gradient_clipping=-1,
        track_every=1000, save_every=10000,
        reload=False, device=[0],
    )

    parity_base = dict(
        seed=0, iterations=75, memory_length=25,
        parity_sequence_length=64,
        d_model=1024, d_input=512,
        n_synch_out=32, n_synch_action=32,
        synapse_depth=1, heads=8,
        memory_hidden_dims=16, dropout=0.0,
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

    mazes_base = dict(
        model="ctm", neuron_select_type="first-last",
        dataset="mazes-small",
        synapse_depth=8, heads=16,
        iterations=75, memory_length=25,
        d_model=2048, d_input=512,
        backbone_type="resnet34-2",
        n_synch_out=64, n_synch_action=32,
        memory_hidden_dims=32,
        deep_memory=True, do_normalisation=False,
        weight_decay=0.0, dropout=0.1,
        positional_embedding_type="none",
        maze_route_length=100,
        cirriculum_lookahead=5,
        expand_range=False,
        batch_size=64, batch_size_test=128,
        lr=1e-4, training_iterations=100001,
        warmup_steps=10000,
        use_scheduler=True, scheduler_type="cosine",
        gradient_clipping=-1,
        track_every=1000, save_every=10000,
        reload=False, device=[0],
        data_root="baseline/data/mazes",
        n_test_batches=20,
    )

    cifar10_base = dict(
        model="ctm", dataset="cifar10",
        d_model=256, d_input=64,
        synapse_depth=5, heads=16,
        n_synch_out=256, n_synch_action=512,
        n_random_pairing_self=0,
        neuron_select_type="random-pairing",
        iterations=50, memory_length=15,
        deep_memory=True, memory_hidden_dims=64,
        dropout=0.0, dropout_nlm=0,
        do_normalisation=False,
        positional_embedding_type="none",
        backbone_type="resnet18-1",
        training_iterations=600001,
        warmup_steps=1000,
        use_scheduler=True, scheduler_type="cosine",
        weight_decay=1e-4,
        save_every=1000, track_every=2000,
        n_test_batches=50,
        batch_size=512, batch_size_test=512,
        lr=1e-4, device=[0], seed=1,
        data_root="baseline/data/",
    )

    qamnist_base = dict(
        seed=0, model_type="ctm",
        memory_length=30,
        q_num_images=3, q_num_images_delta=2,
        q_num_repeats_per_input=10,
        q_num_operations=3, q_num_operations_delta=2,
        q_num_answer_steps=10,
        d_model=1024, d_input=64,
        n_synch_out=32, n_synch_action=32,
        synapse_depth=1, heads=4,
        memory_hidden_dims=16, dropout=0.0,
        deep_memory=True, do_normalisation=False,
        weight_decay=0.0,
        use_scheduler=True, scheduler_type="cosine",
        batch_size=64, batch_size_test=256,
        lr=1e-4, training_iterations=300001,
        warmup_steps=500, track_every=1000,
        save_every=10000, reload=False,
        device=[0], use_amp=False,
        neuron_select_type="random",
        data_root="baseline/data/",
        n_test_batches=20,
    )

    # ── JEPA hyperparameters (used when jepa_weight > 0) ──
    jepa_default = dict(
        cross_tick_jepa_weight=1.0,
        cross_tick_jepa_loss="cosine",
        cross_tick_jepa_hidden_dim=128,
        cross_tick_jepa_predictor_depth=2,
        cross_tick_jepa_dropout=0.0,
        cross_tick_jepa_target_stop_grad=True,
    )

    if stage in ("jx00", "all"):
        # ── SORT baselines ──
        plan.append(exp(
            "jx00_sort_baseline",
            "Sort: CTM baseline (no JEPA)",
            _p("baseline.tasks.sort.train", {**sort_base, "log_dir": "logs/baseline_jepa/jx00_sort_baseline"}),
            tags=["sort", "baseline"],
        ))
        plan.append(exp(
            "jx00_sort_jepa",
            "Sort: CTM + cross-tick JEPA",
            _p("baseline.tasks.sort.train", {**sort_base, "log_dir": "logs/baseline_jepa/jx00_sort_jepa", **jepa_default}),
            tags=["sort", "jepa"],
        ))

        # ── PARITY baselines ──
        plan.append(exp(
            "jx00_parity_baseline",
            "Parity: CTM baseline (no JEPA)",
            _p("baseline.tasks.parity.train", {**parity_base, "log_dir": "logs/baseline_jepa/jx00_parity_baseline"}),
            tags=["parity", "baseline"],
        ))
        plan.append(exp(
            "jx00_parity_jepa",
            "Parity: CTM + cross-tick JEPA",
            _p("baseline.tasks.parity.train", {**parity_base, "log_dir": "logs/baseline_jepa/jx00_parity_jepa", **jepa_default}),
            tags=["parity", "jepa"],
        ))

        # ── MAZES baselines ──
        plan.append(exp(
            "jx00_mazes_baseline",
            "Mazes: CTM baseline (no JEPA)",
            _p("baseline.tasks.mazes.train", {**mazes_base, "log_dir": "logs/baseline_jepa/jx00_mazes_baseline"}),
            tags=["mazes", "baseline"],
        ))
        plan.append(exp(
            "jx00_mazes_jepa",
            "Mazes: CTM + cross-tick JEPA",
            _p("baseline.tasks.mazes.train", {**mazes_base, "log_dir": "logs/baseline_jepa/jx00_mazes_jepa", **jepa_default}),
            tags=["mazes", "jepa"],
        ))

        # ── IMAGE CLASSIFICATION / CIFAR-10 baselines ──
        plan.append(exp(
            "jx00_cifar10_baseline",
            "CIFAR-10: CTM baseline (no JEPA)",
            _p("baseline.tasks.image_classification.train", {**cifar10_base, "log_dir": "logs/baseline_jepa/jx00_cifar10_baseline"}),
            tags=["cifar10", "baseline"],
        ))
        plan.append(exp(
            "jx00_cifar10_jepa",
            "CIFAR-10: CTM + cross-tick JEPA",
            _p("baseline.tasks.image_classification.train", {**cifar10_base, "log_dir": "logs/baseline_jepa/jx00_cifar10_jepa", **jepa_default}),
            tags=["cifar10", "jepa"],
        ))

        # ── QAMNIST baselines ──
        plan.append(exp(
            "jx00_qamnist_baseline",
            "QAMNIST: CTM baseline (no JEPA)",
            _p("baseline.tasks.qamnist.train", {**qamnist_base, "log_dir": "logs/baseline_jepa/jx00_qamnist_baseline"}),
            tags=["qamnist", "baseline"],
        ))
        plan.append(exp(
            "jx00_qamnist_jepa",
            "QAMNIST: CTM + cross-tick JEPA",
            _p("baseline.tasks.qamnist.train", {**qamnist_base, "log_dir": "logs/baseline_jepa/jx00_qamnist_jepa", **jepa_default}),
            tags=["qamnist", "jepa"],
        ))

    if stage in ("jx01", "all"):
        # ── JEPA ablation: loss function variants (cosine vs mse) ──
        plan.append(exp(
            "jx01_sort_jepa_mse",
            "Sort: JEPA with MSE loss",
            _p("baseline.tasks.sort.train", {**sort_base, "log_dir": "logs/baseline_jepa/jx01_sort_jepa_mse", **jepa_default, "cross_tick_jepa_loss": "mse"}),
            tags=["sort", "jepa", "mse"],
        ))
        plan.append(exp(
            "jx01_parity_jepa_mse",
            "Parity: JEPA with MSE loss",
            _p("baseline.tasks.parity.train", {**parity_base, "log_dir": "logs/baseline_jepa/jx01_parity_jepa_mse", **jepa_default, "cross_tick_jepa_loss": "mse"}),
            tags=["parity", "jepa", "mse"],
        ))
        plan.append(exp(
            "jx01_cifar10_jepa_mse",
            "CIFAR-10: JEPA with MSE loss",
            _p("baseline.tasks.image_classification.train", {**cifar10_base, "log_dir": "logs/baseline_jepa/jx01_cifar10_jepa_mse", **jepa_default, "cross_tick_jepa_loss": "mse"}),
            tags=["cifar10", "jepa", "mse"],
        ))
        # JEPA weight sweep: 0.1, 0.5
        for w in ["0p1", "0p5"]:
            wv = float(w.replace("p", "."))
            plan.append(exp(
                f"jx01_sort_jepa_w{w}",
                f"Sort: JEPA weight={wv}",
                _p("baseline.tasks.sort.train", {**sort_base, "log_dir": f"logs/baseline_jepa/jx01_sort_jepa_w{w}", **jepa_default, "cross_tick_jepa_weight": wv}),
                tags=["sort", "jepa", "weight-sweep"],
            ))
            plan.append(exp(
                f"jx01_parity_jepa_w{w}",
                f"Parity: JEPA weight={wv}",
                _p("baseline.tasks.parity.train", {**parity_base, "log_dir": f"logs/baseline_jepa/jx01_parity_jepa_w{w}", **jepa_default, "cross_tick_jepa_weight": wv}),
                tags=["parity", "jepa", "weight-sweep"],
            ))

    return plan


def submit_to_pool(exp, config, master_addr=None, port=None):
    node_addrs = exp.get("node_addrs") or [exp.get("node_addr")]
    payload = {
        "config": config,
        "extra_args": exp["command"],
        "node_addrs": node_addrs,
        "env": {
            "CTM_EXPERIMENT_NAME": exp["name"],
            "CTM_METRICS_DIR": "runs/metrics",
            "CTM_LOG_DIR": "runs/logs/baseline_jepa",
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


def print_failure_details(experiment_name):
    fail_paths = [
        ROOT / "runs" / "metrics" / f"{experiment_name}.fail.json",
    ]
    for fail_path in fail_paths:
        if not fail_path.is_file():
            continue
        try:
            with open(fail_path, encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            print(f"    could not read failure report {fail_path}: {exc}")
            return
        err = payload.get("error", "")
        log_path = payload.get("log_path")
        print(f"    failure report: {fail_path}")
        if log_path:
            print(f"    log: {log_path}")
        for line in err.splitlines()[-40:]:
            print(f"    {line}")
        return
    log_paths = [
        ROOT / "runs" / "logs" / "baseline_jepa" / f"{experiment_name}.log",
        ROOT / "runs" / "logs" / "pool_last_run.log",
    ]
    for log_path in log_paths:
        if log_path.is_file():
            print(f"    log: {log_path}")
            with open(log_path, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()[-80:]
            for line in lines:
                print(f"    {line}", end="")
            return


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
                    rc = t.get("return_code")
                    if rc is not None:
                        print(f"  [pool] return_code={rc}")
                    acks = status.get("acks", {}).get(task_id, {})
                    for addr, ack in sorted(acks.items()):
                        print(f"  [pool] ack {addr}: status={ack.get('status')} msg={ack.get('message', '')}")
                    if t["status"] == "failed" and experiment_name:
                        print_failure_details(experiment_name)
                    return t["status"]
        except Exception:
            pass
        time.sleep(poll_interval)


def cmd_run(args):
    exps = build_plan(args.stage)
    target = [e for e in exps if e["name"] == args.name]
    if not target:
        print(f"Experiment '{args.name}' not found.")
        return
    e = target[0]
    cmd = f"python -m {e['command']}"
    print(f"Running: {cmd}")
    print(f"Question: {e['question']}")
    subprocess.run(shlex.split(cmd))


def cmd_submit(args):
    exps = build_plan(args.stage)
    print(f"Submitting {len(exps)} experiments to pool at {MASTER_ADDR}:{PORT}")
    print(f"Config: {POOL_CONFIG}")
    for e in exps:
        print(f"  {e['name']}: {e['question']}")
    print()
    if args.dry_run:
        return
    for e in exps:
        print(f"Submitting {e['name']}...")
        result = submit_to_pool(e, POOL_CONFIG, MASTER_ADDR, PORT)
        if result is None:
            print(f"  FAILED to submit {e['name']}")
            if args.stop_on_fail:
                break
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


def cmd_plan(args):
    exps = build_plan(args.stage)
    print(f"\n{'=' * 80}")
    print(f"BASELINE JEPA PLAN — {len(exps)} experiments")
    print(f"{'=' * 80}\n")
    for e in exps:
        print(f"  {e['name']}")
        print(f"    {e['question']}")
        print(f"    tags: {', '.join(e['tags'])}")
        print(f"    node: {e['node_addr']}")
        print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Baseline CTM JEPA experiments")
    sub = parser.add_subparsers(dest="command")

    p_plan = sub.add_parser("plan")
    p_plan.add_argument("--stage", default="all", help="Stage to build")

    p_run = sub.add_parser("run")
    p_run.add_argument("--name", required=True, help="Experiment name")
    p_run.add_argument("--stage", default="all")

    p_submit = sub.add_parser("submit")
    p_submit.add_argument("--stage", default="all")
    p_submit.add_argument("--dry-run", action="store_true")
    p_submit.add_argument("--wait", action="store_true", default=True)
    p_submit.add_argument("--no-wait", action="store_false", dest="wait")
    p_submit.add_argument("--stop-on-fail", action="store_true", default=True)

    p_submit.add_argument("--master-addr", default=MASTER_ADDR)
    p_submit.add_argument("--port", type=int, default=PORT)

    args = parser.parse_args()
    if args.command == "plan":
        cmd_plan(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "submit":
        cmd_submit(args)
    else:
        parser.print_help()
