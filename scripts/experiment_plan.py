#!/usr/bin/env python3
import argparse
import csv
import glob
import math
import os
import shlex
import subprocess
import sys
import time
import urllib.request
import json


URL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

BASE_ARGS = {
    "epochs": 1,
    "batch_size": 4,
    "accumulation_steps": 1,
    "num_hidden_layers": 16,
    "hidden_size": 768,
    "d_model": 512,
    "d_input": 256,
    "heads": 8,
    "n_synch_out": 512,
    "n_synch_action": 512,
    "iterations": 4,
    "memory_length": 10,
    "memory_hidden_dims": 4,
    "deep_nlms": 1,
    "synapse_depth": 3,
    "tick_loss_mode": "min_conf",
    "elf_horizon_mode": "none",
    "elf_max_horizon": 4,
    "tick_improve_weight": 0.0,
    "tick_improve_margin": 0.0,
    "tick_halt_mode": "none",
    "tick_halt_threshold": 0.65,
    "tick_halt_temperature": 0.25,
    "tick_compute_weight": 0.0,
    "cell_sparsity_mode": "none",
    "cell_topk": 512,
    "cell_sparsity_rescale": 1,
    "self_cond": 1,
    "cross_layer_state": 1,
    "max_seq_len": 512,
    "log_interval": 20,
    "save_interval": 1000,
    "max_steps": 1000,
    "dtype": "bfloat16",
}

PLAN_SIZES = ("core", "full", "wide")


def merge_args(**overrides):
    data = dict(BASE_ARGS)
    data.update(overrides)
    return data


def experiment(name, question, args):
    args = dict(args)
    args["experiment_name"] = name
    args["swanlab_name"] = name
    args["save_weight"] = name
    return {"name": name, "question": question, "args": args}


def clone_experiment(exp, name, args_overrides, question=None):
    args = dict(exp["args"])
    args.update(args_overrides)
    args["experiment_name"] = name
    args["swanlab_name"] = name
    args["save_weight"] = name
    return {
        "name": name,
        "base_name": exp["name"],
        "question": question or exp["question"],
        "args": args,
    }


def include_plan_size(plan_size, min_size):
    return PLAN_SIZES.index(plan_size) >= PLAN_SIZES.index(min_size)


def ctm_name(prefix, **items):
    parts = [prefix]
    for key, value in items.items():
        parts.append(f"{key}{str(value).replace('.', 'p')}")
    return "_".join(parts)


def validate_plan(plan):
    for exp in plan:
        args = exp["args"]
        name = exp["name"]
        heads = int(args.get("heads", 1))
        hidden = int(args.get("hidden_size", 1))
        d_input = int(args.get("d_input", 1))
        d_model = int(args.get("d_model", 1))
        n_synch_out = int(args.get("n_synch_out", 1))
        n_synch_action = int(args.get("n_synch_action", 1))
        if args.get("model_type") == "transformer" and hidden % heads != 0:
            raise ValueError(
                f"{name}: hidden_size({hidden}) must be divisible by heads({heads})")
        if args.get("model_type") == "ctm":
            if d_input % heads != 0:
                raise ValueError(
                    f"{name}: d_input({d_input}) must be divisible by heads({heads})")
            if d_model < max(n_synch_out, n_synch_action):
                raise ValueError(
                    f"{name}: d_model({d_model}) must be >= n_synch_out({n_synch_out}) "
                    f"and n_synch_action({n_synch_action})")
    return plan


def build_plan(stage, plan_size="full"):
    plan = []
    if stage in ("smoke", "all"):
        smoke = dict(BASE_ARGS, max_steps=100, num_hidden_layers=4,
                     hidden_size=512, d_model=256, d_input=128, heads=4,
                     n_synch_out=256, n_synch_action=256,
                     iterations=3, max_seq_len=256, synapse_depth=2)
        plan.append(experiment(
            "s00_transformer_smoke",
            "Verify standard Transformer baseline path and metrics logging.",
            merge_args(**smoke, model_type="transformer")))
        plan.append(experiment(
            "s00_ctm_smoke",
            "Verify CTM path and per-tick metrics logging.",
            merge_args(**smoke, model_type="ctm")))

    if stage in ("compass", "all"):
        scales = [
            ("12l_h640", 12, 640, 384, 192, 8),
            ("16l_h768", 16, 768, 512, 256, 8),
        ]
        if include_plan_size(plan_size, "full"):
            scales.append(("24l_h896", 24, 896, 640, 320, 8))
        if include_plan_size(plan_size, "wide"):
            scales.append(("32l_h1024", 32, 1024, 768, 384, 8))
        for tag, layers, hidden, d_model, d_input, heads in scales:
            common = {
                "num_hidden_layers": layers,
                "hidden_size": hidden,
                "d_model": d_model,
                "d_input": d_input,
                "heads": heads,
                "n_synch_out": d_model,
                "n_synch_action": d_model,
            }
            plan.append(experiment(
                f"s01_transformer_{tag}",
                "Transformer scale baseline for CTM cost/loss comparison.",
                merge_args(model_type="transformer", iterations=1, **common)))
            plan.append(experiment(
                f"s01_ctm_{tag}_tick4",
                "CTM at matched outer layer/hidden scale with 4 ticks.",
                merge_args(model_type="ctm", iterations=4, **common)))
        plan.append(experiment(
            "s01_ctm_16l_tick1",
            "CTM with one tick to isolate cell-block overhead from iterative thinking.",
            merge_args(model_type="ctm", iterations=1)))

    if stage in ("ticks", "all"):
        tick_values = [1, 2, 4, 8, 16]
        if include_plan_size(plan_size, "full"):
            tick_values = [1, 2, 3, 4, 6, 8, 12, 16]
        if include_plan_size(plan_size, "wide"):
            tick_values += [24, 32]
        for ticks in tick_values:
            plan.append(experiment(
                f"s02_ctm_tick{ticks}",
                "Measure loss/cost curve as maximum CTM ticks increase.",
                merge_args(model_type="ctm", iterations=ticks)))
        if include_plan_size(plan_size, "full"):
            for mode in ["mean", "last", "min_conf"]:
                plan.append(experiment(
                    f"s02_ctm_tick8_loss_{mode}",
                    "Compare per-tick loss aggregation to see whether later ticks specialize.",
                    merge_args(model_type="ctm", iterations=8, tick_loss_mode=mode)))
        for key, overrides in [
            ("halt_conf_t8_c0", {"iterations": 8, "tick_halt_mode": "confidence",
                                 "tick_compute_weight": 0.0}),
            ("halt_conf_t8_c01", {"iterations": 8, "tick_halt_mode": "confidence",
                                  "tick_compute_weight": 0.01}),
            ("halt_conf_t16_c01", {"iterations": 16, "tick_halt_mode": "confidence",
                                   "tick_compute_weight": 0.01}),
            ("halt_thresh_t16", {"iterations": 16, "tick_halt_mode": "threshold",
                                 "tick_halt_threshold": 0.65}),
        ]:
            plan.append(experiment(
                f"s02_ctm_{key}",
                "Train CTM with adaptive tick halting signals and optional compute penalty.",
                merge_args(model_type="ctm", **overrides)))
        if include_plan_size(plan_size, "full"):
            for temp in [0.15, 0.35, 0.60]:
                plan.append(experiment(
                    ctm_name("s02_ctm_halt_conf_t16", temp=temp),
                    "Sweep confidence-halting temperature for natural early/late tick separation.",
                    merge_args(
                        model_type="ctm",
                        iterations=16,
                        tick_halt_mode="confidence",
                        tick_halt_temperature=temp,
                        tick_compute_weight=0.01,
                    )))
            for threshold in [0.55, 0.70, 0.85]:
                plan.append(experiment(
                    ctm_name("s02_ctm_halt_thresh_t16", th=threshold),
                    "Sweep hard confidence threshold used to estimate effective thinking depth.",
                    merge_args(
                        model_type="ctm",
                        iterations=16,
                        tick_halt_mode="threshold",
                        tick_halt_threshold=threshold,
                    )))

    if stage in ("elf", "all"):
        for key, overrides in [
            ("next", {"elf_horizon_mode": "none"}),
            ("linear_h4", {"elf_horizon_mode": "linear", "elf_max_horizon": 4}),
            ("pow2_h4", {"elf_horizon_mode": "pow2", "elf_max_horizon": 4}),
            ("linear_h8", {"elf_horizon_mode": "linear", "elf_max_horizon": 8,
                           "iterations": 8}),
            ("linear_h4_improve", {"elf_horizon_mode": "linear",
                                   "elf_max_horizon": 4,
                                   "tick_improve_weight": 0.1}),
        ]:
            plan.append(experiment(
                f"s03_elf_{key}",
                "Give different ticks different prediction horizons and measure whether tick roles separate.",
                merge_args(model_type="ctm", **overrides)))
        if include_plan_size(plan_size, "full"):
            for ticks in [8, 12, 16]:
                for horizon in [4, 8]:
                    plan.append(experiment(
                        f"s03_elf_linear_t{ticks}_h{horizon}",
                        "Cross ELF horizon with more internal ticks to test multi-token prediction utility.",
                        merge_args(
                            model_type="ctm",
                            iterations=ticks,
                            elf_horizon_mode="linear",
                            elf_max_horizon=horizon,
                        )))
            for weight in [0.03, 0.1, 0.3]:
                plan.append(experiment(
                    ctm_name("s03_elf_linear_h8_improve", w=weight),
                    "Sweep improvement regularization so later ticks earn their extra compute.",
                    merge_args(
                        model_type="ctm",
                        iterations=8,
                        elf_horizon_mode="linear",
                        elf_max_horizon=8,
                        tick_improve_weight=weight,
                    )))
        if include_plan_size(plan_size, "wide"):
            for mode in ["pow2", "linear"]:
                plan.append(experiment(
                    f"s03_elf_{mode}_t16_h16",
                    "Wide ELF run with long horizon and 16 ticks.",
                    merge_args(
                        model_type="ctm",
                        iterations=16,
                        elf_horizon_mode=mode,
                        elf_max_horizon=16,
                    )))

    if stage in ("cells", "all"):
        cell_grid = [
            (256, 4, 10, 3, "none", 256),
            (512, 4, 10, 3, "none", 512),
            (1024, 2, 8, 2, "none", 1024),
            (1024, 2, 8, 2, "topk", 512),
            (1536, 1, 6, 1, "none", 1536),
            (1536, 1, 6, 1, "topk", 512),
            (2048, 1, 6, 1, "topk", 512),
        ]
        if include_plan_size(plan_size, "full"):
            cell_grid += [
                (768, 3, 10, 2, "none", 768),
                (768, 3, 10, 2, "topk", 384),
                (1024, 1, 6, 1, "topk", 256),
                (1024, 2, 8, 2, "topk", 256),
                (1536, 1, 6, 1, "topk", 384),
                (2048, 1, 6, 1, "topk", 256),
                (2048, 1, 4, 1, "topk", 512),
            ]
        if include_plan_size(plan_size, "wide"):
            cell_grid += [
                (3072, 1, 4, 1, "topk", 256),
                (3072, 1, 4, 1, "topk", 512),
                (4096, 1, 4, 1, "topk", 512),
            ]
        for d_model, mem_h, mem_len, depth, sparsity, topk in cell_grid:
            sparse_tag = f"{sparsity}{topk}" if sparsity != "none" else "dense"
            plan.append(experiment(
                f"s04_cells_d{d_model}_mh{mem_h}_m{mem_len}_sd{depth}_{sparse_tag}",
                "Trade larger cell count against smaller per-cell memory/synapse models.",
                merge_args(
                    model_type="ctm",
                    d_model=d_model,
                    n_synch_out=d_model,
                    n_synch_action=d_model,
                    memory_hidden_dims=mem_h,
                    memory_length=mem_len,
                    synapse_depth=depth,
                    cell_sparsity_mode=sparsity,
                    cell_topk=topk,
                    iterations=4,
                )))
        if include_plan_size(plan_size, "full"):
            for rescale in [0, 1]:
                plan.append(experiment(
                    f"s04_cells_topk512_rescale{rescale}",
                    "Check whether sparse activation rescaling stabilizes top-k cells.",
                    merge_args(
                        model_type="ctm",
                        d_model=1536,
                        n_synch_out=1536,
                        n_synch_action=1536,
                        memory_hidden_dims=1,
                        memory_length=6,
                        synapse_depth=1,
                        cell_sparsity_mode="topk",
                        cell_topk=512,
                        cell_sparsity_rescale=rescale,
                    )))

    if stage in ("ablations", "all"):
        for key, overrides in [
            ("no_selfcond", {"self_cond": 0}),
            ("no_cross_state", {"cross_layer_state": 0}),
            ("shallow_synapse", {"synapse_depth": 1}),
            ("short_memory", {"memory_length": 6}),
            ("tiny_nlm", {"memory_hidden_dims": 1, "deep_nlms": 1}),
        ]:
            plan.append(experiment(
                f"s05_{key}",
                "Ablate CTM details to find elegant low-cost simplifications.",
                merge_args(model_type="ctm", iterations=4, **overrides)))
        if include_plan_size(plan_size, "full"):
            for key, overrides in [
                ("no_selfcond_no_cross", {"self_cond": 0, "cross_layer_state": 0}),
                ("dinput128_heads4", {"d_input": 128, "heads": 4}),
                ("dinput384_heads8", {"d_input": 384, "heads": 8}),
                ("memlen14", {"memory_length": 14}),
                ("synapse2_mh2", {"synapse_depth": 2, "memory_hidden_dims": 2}),
                ("deep_nlms0", {"deep_nlms": 0}),
            ]:
                plan.append(experiment(
                    f"s05_{key}",
                    "Search for low-cost CTM simplifications and sensitivity points.",
                    merge_args(model_type="ctm", iterations=4, **overrides)))
        if include_plan_size(plan_size, "wide"):
            for key, overrides in [
                ("short_memory_sparse", {
                    "memory_length": 4,
                    "memory_hidden_dims": 1,
                    "synapse_depth": 1,
                    "d_model": 1536,
                    "n_synch_out": 1536,
                    "n_synch_action": 1536,
                    "cell_sparsity_mode": "topk",
                    "cell_topk": 384,
                }),
                ("tick8_simple_cell", {
                    "iterations": 8,
                    "memory_hidden_dims": 1,
                    "synapse_depth": 1,
                }),
            ]:
                plan.append(experiment(
                    f"s05_{key}",
                    "Wide ablation for sparse/simple CTM variants.",
                    merge_args(model_type="ctm", **overrides)))

    return validate_plan(plan)


def build_batch_tune_plan(stage, batch_sizes, max_steps, log_interval, plan_size="full"):
    plan = []
    for exp in build_plan(stage, plan_size):
        for batch_size in batch_sizes:
            name = f"bt__{exp['name']}__bs{batch_size}"
            plan.append(clone_experiment(
                exp,
                name,
                {
                    "batch_size": batch_size,
                    "max_steps": max_steps,
                    "log_interval": log_interval,
                    "save_interval": max_steps + 1000000,
                    "no_tensorboard": None,
                },
                question=f"Batch-size probe for {exp['name']} at batch_size={batch_size}.",
            ))
    return plan


def load_batch_profile(path):
    if not path:
        return {}
    if not os.path.exists(path):
        print(f"warning: batch profile not found, using default batches: {path}", file=sys.stderr)
        return {}
    profile = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row.get("experiment_name") or row.get("base_experiment") or row.get("name")
            batch = row.get("batch_size") or row.get("recommended_batch_size")
            if name and batch:
                profile[name] = int(float(batch))
    return profile


def apply_batch_profile(plan, profile):
    if not profile:
        return plan
    tuned = []
    for exp in plan:
        if exp["name"] in profile:
            exp = clone_experiment(exp, exp["name"], {"batch_size": profile[exp["name"]]})
        tuned.append(exp)
    return tuned


def arg_items(args):
    items = []
    for key, value in args.items():
        if value is None:
            items.append(f"--{key}")
        else:
            items.extend([f"--{key}", str(value)])
    return items


def submit_command(exp, config, master_addr=None, port=None, wait=None):
    cmd = ["./scripts/ctmctl", "pool", "submit", config]
    if master_addr:
        cmd.extend(["--master_addr", master_addr])
    if port:
        cmd.extend(["--port", str(port)])
    if wait is not None:
        cmd.extend(["--wait", str(wait)])
    if exp.get("node_addrs"):
        cmd.extend(["--nodes", ",".join(exp["node_addrs"])])
    cmd.extend(arg_items(exp["args"]))
    return " ".join(shlex.quote(x) for x in cmd)


def write_manifest(plan, path, config, master_addr, port, wait):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        fields = ["index", "name", "base_name", "batch_size", "nodes", "question", "command"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, exp in enumerate(plan):
            writer.writerow({
                "index": i,
                "name": exp["name"],
                "base_name": exp.get("base_name", exp["name"]),
                "batch_size": exp["args"].get("batch_size", ""),
                "nodes": ",".join(exp.get("node_addrs") or []),
                "question": exp["question"],
                "command": submit_command(exp, config, master_addr, port, wait),
            })


def print_commands(args):
    plan = apply_batch_profile(
        build_plan(args.stage, args.plan_size),
        load_batch_profile(args.batch_profile),
    )
    if args.output:
        write_manifest(plan, args.output, args.config, args.master_addr, args.port, args.wait)
        print(f"wrote {len(plan)} experiments: {args.output}")
    for i, exp in enumerate(plan):
        print(f"\n# {i:02d} {exp['name']}")
        print(f"# {exp['question']}")
        print(submit_command(exp, args.config, args.master_addr, args.port, args.wait))


def print_batch_commands(args):
    plan = build_batch_tune_plan(
        args.stage, args.batch_sizes, args.tune_steps, args.tune_log_interval,
        args.plan_size)
    plan = assign_node_groups(plan, parse_node_groups(args.node_groups))
    if args.output:
        write_manifest(plan, args.output, args.config, args.master_addr, args.port, args.wait)
        print(f"wrote {len(plan)} batch probes: {args.output}")
    for i, exp in enumerate(plan):
        print(f"\n# {i:02d} {exp['name']}")
        print(f"# {exp['question']}")
        print(submit_command(exp, args.config, args.master_addr, args.port, args.wait))


def pool_status(master_addr, port):
    with URL_OPENER.open(f"http://{master_addr}:{port}/status", timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_json(url, payload, timeout=10):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with URL_OPENER.open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8") or "{}")


def parse_node_groups(items):
    groups = []
    for item in items or []:
        parts = [part.strip() for part in item.split(",") if part.strip()]
        if parts:
            groups.append(parts)
    return groups


def assign_node_groups(plan, node_groups):
    if not node_groups:
        return plan
    assigned = []
    for i, exp in enumerate(plan):
        exp = dict(exp)
        exp["node_addrs"] = node_groups[i % len(node_groups)]
        assigned.append(exp)
    return assigned


def wait_until_idle(master_addr, port, task_id, poll_interval):
    while True:
        status = pool_status(master_addr, port)
        nodes = status.get("nodes", {})
        running = [
            addr for addr, node in nodes.items()
            if str(node.get("status", "")).startswith(f"running:{task_id}")
        ]
        if not running:
            return
        time.sleep(poll_interval)


def node_group_idle(master_addr, port, node_group):
    status = pool_status(master_addr, port)
    return node_group_idle_from_status(status, node_group)


def node_group_idle_from_status(status, node_group):
    nodes = status.get("nodes", {})
    for addr in node_group:
        node = nodes.get(addr, {})
        if str(node.get("status", "")).startswith("running:"):
            return False
    return True


def submit_exp(args, exp, wait=30.0):
    extra_args = " ".join(shlex.quote(item) for item in arg_items(exp["args"]))
    payload = {
        "config": args.config,
        "extra_args": extra_args,
        "node_addrs": exp.get("node_addrs") or [],
    }
    resp = post_json(f"http://{args.master_addr}:{args.port}/submit", payload)
    task = resp["task"]
    print(f"submitted task {task['task_id']}: {exp['name']} nodes={payload['node_addrs'] or 'all'}")
    if wait > 0:
        expected = set(payload["node_addrs"])
        if expected:
            deadline = time.time() + wait
            seen = set()
            while time.time() < deadline:
                status = pool_status(args.master_addr, args.port)
                acks = status.get("acks", {}).get(task["task_id"], {})
                for addr, ack in sorted(acks.items()):
                    if addr not in seen:
                        seen.add(addr)
                        print(f"ack {addr}: {ack.get('status')} {ack.get('message', '')}")
                if expected.issubset(seen):
                    break
                time.sleep(1)
    return task


def run_plan(args):
    if getattr(args, "batch_tune", False):
        plan = build_batch_tune_plan(
            args.stage, args.batch_sizes, args.tune_steps, args.tune_log_interval,
            args.plan_size)
    else:
        plan = apply_batch_profile(
            build_plan(args.stage, args.plan_size),
            load_batch_profile(args.batch_profile),
        )
    for i, exp in enumerate(plan):
        cmd = submit_command(exp, args.config, args.master_addr, args.port, args.wait)
        print(f"\n[{i + 1}/{len(plan)}] {exp['name']}")
        print(cmd)
        proc = subprocess.run(shlex.split(cmd), check=False)
        if proc.returncode != 0:
            raise SystemExit(proc.returncode)
        status = pool_status(args.master_addr, args.port)
        task = status.get("task") or {}
        task_id = task.get("task_id")
        if task_id:
            time.sleep(args.startup_grace)
            wait_until_idle(args.master_addr, args.port, task_id, args.poll_interval)


def run_parallel(args):
    if getattr(args, "batch_tune", False):
        plan = build_batch_tune_plan(
            args.stage, args.batch_sizes, args.tune_steps, args.tune_log_interval,
            args.plan_size)
    else:
        plan = apply_batch_profile(
            build_plan(args.stage, args.plan_size),
            load_batch_profile(args.batch_profile),
        )

    node_groups = parse_node_groups(args.node_groups)
    if not node_groups:
        raise SystemExit("--node_groups is required for run-parallel")

    queue = list(plan)
    running = {}
    completed = 0
    while queue or running:
        status = pool_status(args.master_addr, args.port)
        for idx, group in enumerate(node_groups):
            if idx in running or not queue:
                continue
            if not node_group_idle_from_status(status, group):
                continue
            exp = dict(queue.pop(0))
            exp["node_addrs"] = group
            print(f"\n[{completed + len(running) + 1}/{len(plan)}] lane={idx} {exp['name']}")
            task = submit_exp(args, exp, wait=0)
            running[idx] = {
                "task_id": task["task_id"],
                "nodes": group,
                "name": exp["name"],
                "submitted_at": time.time(),
            }

        done = []
        status = pool_status(args.master_addr, args.port)
        for idx, item in running.items():
            if time.time() - item["submitted_at"] < args.startup_grace:
                continue
            if node_group_idle_from_status(status, item["nodes"]):
                done.append(idx)
        for idx in done:
            item = running.pop(idx)
            completed += 1
            print(f"[done {completed}/{len(plan)}] lane={idx} {item['name']} task={item['task_id']}")

        if queue or running:
            time.sleep(args.poll_interval)


def latest_rows(metrics_dir):
    rows = []
    for path in glob.glob(os.path.join(metrics_dir, "*.csv")):
        with open(path, newline="", encoding="utf-8") as f:
            data = list(csv.DictReader(f))
        if not data:
            continue
        row = data[-1]
        row["metrics_file"] = path
        rows.append(row)
    return rows


def parse_float(row, key, default=math.nan):
    try:
        value = row.get(key, "")
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def batch_probe_base_name(experiment_name):
    if not experiment_name.startswith("bt__"):
        return None
    body = experiment_name[len("bt__"):]
    if "__bs" not in body:
        return None
    return body.rsplit("__bs", 1)[0]


def recommend_batches(args):
    rows = latest_rows(args.metrics_dir)
    probes = []
    for row in rows:
        base = batch_probe_base_name(row.get("experiment_name", ""))
        if not base:
            continue
        peak = parse_float(row, "peak_memory_mb")
        batch = int(parse_float(row, "batch_size", 0))
        tokens_per_sec = parse_float(row, "tokens_per_sec")
        if batch <= 0 or math.isnan(peak):
            continue
        probes.append({
            "base_experiment": base,
            "experiment_name": row["experiment_name"],
            "batch_size": batch,
            "peak_memory_mb": peak,
            "tokens_per_sec": tokens_per_sec,
            "loss": row.get("loss", ""),
            "metrics_file": row.get("metrics_file", ""),
        })

    limit_mb = args.target_memory_gb * 1024 * args.memory_util
    grouped = {}
    for probe in probes:
        grouped.setdefault(probe["base_experiment"], []).append(probe)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fields = [
        "experiment_name", "recommended_batch_size", "peak_memory_mb",
        "tokens_per_sec", "memory_limit_mb", "num_successful_probes",
        "selected_probe", "metrics_file",
    ]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for base, items in sorted(grouped.items()):
            under = [x for x in items if x["peak_memory_mb"] <= limit_mb]
            if under:
                # Prefer highest stable batch, then highest throughput.
                selected = sorted(
                    under,
                    key=lambda x: (x["batch_size"], x["tokens_per_sec"]),
                    reverse=True,
                )[0]
            else:
                # Everything that succeeded exceeded the target utilization.
                selected = sorted(items, key=lambda x: (x["peak_memory_mb"], x["batch_size"]))[0]
            writer.writerow({
                "experiment_name": base,
                "recommended_batch_size": selected["batch_size"],
                "peak_memory_mb": round(selected["peak_memory_mb"], 3),
                "tokens_per_sec": round(selected["tokens_per_sec"], 3)
                if not math.isnan(selected["tokens_per_sec"]) else "",
                "memory_limit_mb": round(limit_mb, 3),
                "num_successful_probes": len(items),
                "selected_probe": selected["experiment_name"],
                "metrics_file": selected["metrics_file"],
            })
    print(f"wrote batch recommendations: {args.output}")


def latest_row_by_experiment(metrics_dir):
    by_name = {}
    for row in latest_rows(metrics_dir):
        name = row.get("experiment_name", "")
        if name:
            by_name[name] = row
    return by_name


def failure_reports_by_experiment(metrics_dir):
    def priority(report):
        status_score = 0 if report.get("status") == "oom" else 1
        rank = report.get("rank")
        try:
            rank_score = int(rank)
        except (TypeError, ValueError):
            rank_score = 0
        return (status_score, rank_score, report.get("failure_file", ""))

    grouped = {}
    for path in sorted(glob.glob(os.path.join(metrics_dir, "*.fail.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        name = data.get("experiment_name")
        if name:
            data["failure_file"] = path
            grouped.setdefault(name, []).append(data)
    return {
        name: sorted(items, key=priority)[0]
        for name, items in grouped.items()
    }


def batch_report(args):
    planned = build_batch_tune_plan(
        args.stage,
        args.batch_sizes,
        args.tune_steps,
        args.tune_log_interval,
        args.plan_size,
    )
    metrics = latest_row_by_experiment(args.metrics_dir)
    failures = failure_reports_by_experiment(args.metrics_dir)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fields = [
        "base_experiment", "probe_experiment", "batch_size", "status",
        "peak_memory_mb", "tokens_per_sec", "steps_per_sec", "loss",
        "global_step", "world_size", "metrics_file", "failure_file",
        "failure_rank", "error_type", "error", "question",
    ]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for exp in planned:
            row = metrics.get(exp["name"])
            failure = failures.get(exp["name"])
            status = failure.get("status", "failed") if failure else ("ok" if row else "missing_metrics")
            writer.writerow({
                "base_experiment": exp.get("base_name", exp["name"]),
                "probe_experiment": exp["name"],
                "batch_size": exp["args"].get("batch_size", ""),
                "status": status,
                "peak_memory_mb": (
                    failure.get("peak_memory_mb", "")
                    if failure else row.get("peak_memory_mb", "") if row else ""
                ),
                "tokens_per_sec": row.get("tokens_per_sec", "") if row else "",
                "steps_per_sec": row.get("steps_per_sec", "") if row else "",
                "loss": row.get("loss", "") if row else "",
                "global_step": row.get("global_step", "") if row else "",
                "world_size": (
                    failure.get("world_size", "")
                    if failure else row.get("world_size", "") if row else ""
                ),
                "metrics_file": row.get("metrics_file", "") if row else "",
                "failure_file": failure.get("failure_file", "") if failure else "",
                "failure_rank": failure.get("rank", "") if failure else "",
                "error_type": failure.get("error_type", "") if failure else "",
                "error": failure.get("error", "")[:500] if failure else "",
                "question": exp["question"],
            })
    print(f"wrote batch probe report: {args.output}")


def export_final_plan(args):
    profile = load_batch_profile(args.batch_profile)
    plan = apply_batch_profile(build_plan(args.stage, args.plan_size), profile)
    write_manifest(plan, args.output, args.config, args.master_addr, args.port, args.wait)
    print(f"wrote final execution plan: {args.output}")


def summarize(args):
    rows = latest_rows(args.metrics_dir)
    rows.sort(key=lambda r: r.get("experiment_name", ""))
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fields = [
        "experiment_name", "model_type", "loss", "tokens_per_sec",
        "peak_memory_mb", "best_tick", "conf_tick", "tick_count",
        "effective_tick", "active_cell_fraction",
        "losses_per_tick", "certainties_per_tick",
        "hidden_size", "num_hidden_layers", "d_model", "d_input",
        "iterations", "memory_length", "memory_hidden_dims", "deep_nlms",
        "synapse_depth", "tick_loss_mode", "elf_horizon_mode",
        "elf_max_horizon", "tick_improve_weight", "tick_improve_margin",
        "tick_halt_mode", "tick_halt_threshold", "tick_halt_temperature",
        "tick_compute_weight", "cell_sparsity_mode", "cell_topk",
        "cell_sparsity_rescale",
        "self_cond", "cross_layer_state", "global_step",
        "metrics_file",
    ]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote summary: {args.output}")


def parse_args():
    parser = argparse.ArgumentParser(description="CTM experiment plan helper")
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--stage", default="compass",
                        choices=["smoke", "compass", "ticks", "elf", "cells", "ablations", "all"])
    common.add_argument("--config", default="infra/clusters/h100_2nodes.env")
    common.add_argument("--master_addr", default=None)
    common.add_argument("--port", type=int, default=None)
    common.add_argument("--wait", type=float, default=30.0)
    common.add_argument("--plan_size", default="full", choices=PLAN_SIZES,
                        help="core=current compact matrix, full=default sufficient sweep, wide=extra exploratory runs.")

    p = sub.add_parser("commands", parents=[common])
    p.add_argument("--output", default="runs/experiment_plans/ctm_plan.csv")
    p.add_argument("--batch_profile", default=None)
    p.set_defaults(func=print_commands)

    p = sub.add_parser("final-plan", parents=[common])
    p.add_argument("--output", default="runs/experiment_plans/final_plan.csv")
    p.add_argument("--batch_profile", default="runs/metrics/batch_profile.csv")
    p.set_defaults(func=export_final_plan)

    p = sub.add_parser("batch-commands", parents=[common])
    p.add_argument("--batch_sizes", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12])
    p.add_argument("--tune_steps", type=int, default=80)
    p.add_argument("--tune_log_interval", type=int, default=20)
    p.add_argument("--node_groups", nargs="*", default=None,
                   help="Optional node groups assigned round-robin, e.g. ip1 ip2 or ip1,ip2.")
    p.add_argument("--output", default="runs/experiment_plans/batch_tune_plan.csv")
    p.set_defaults(func=print_batch_commands)

    p = sub.add_parser("run", parents=[common])
    p.add_argument("--startup_grace", type=float, default=20.0)
    p.add_argument("--poll_interval", type=float, default=30.0)
    p.add_argument("--batch_profile", default=None)
    p.add_argument("--batch_tune", action="store_true")
    p.add_argument("--batch_sizes", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12])
    p.add_argument("--tune_steps", type=int, default=80)
    p.add_argument("--tune_log_interval", type=int, default=20)
    p.set_defaults(func=run_plan)

    p = sub.add_parser("run-parallel", parents=[common])
    p.add_argument("--startup_grace", type=float, default=20.0)
    p.add_argument("--poll_interval", type=float, default=30.0)
    p.add_argument("--batch_profile", default=None)
    p.add_argument("--batch_tune", action="store_true")
    p.add_argument("--batch_sizes", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12])
    p.add_argument("--tune_steps", type=int, default=80)
    p.add_argument("--tune_log_interval", type=int, default=20)
    p.add_argument("--node_groups", nargs="+", required=True,
                   help="Node groups for parallel lanes, e.g. ip1 ip2 ip3 ip4 or ip1,ip2 ip3,ip4.")
    p.set_defaults(func=run_parallel)

    p = sub.add_parser("summarize")
    p.add_argument("--metrics_dir", default="runs/metrics")
    p.add_argument("--output", default="runs/metrics/summary.csv")
    p.set_defaults(func=summarize)

    p = sub.add_parser("recommend-batches")
    p.add_argument("--metrics_dir", default="runs/metrics")
    p.add_argument("--output", default="runs/metrics/batch_profile.csv")
    p.add_argument("--target_memory_gb", type=float, default=80.0)
    p.add_argument("--memory_util", type=float, default=0.90)
    p.set_defaults(func=recommend_batches)

    p = sub.add_parser("batch-report")
    p.add_argument("--stage", default="all",
                   choices=["smoke", "compass", "ticks", "elf", "cells", "ablations", "all"])
    p.add_argument("--plan_size", default="full", choices=PLAN_SIZES)
    p.add_argument("--metrics_dir", default="runs/metrics")
    p.add_argument("--output", default="runs/metrics/batch_probe_report.csv")
    p.add_argument("--batch_sizes", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12])
    p.add_argument("--tune_steps", type=int, default=80)
    p.add_argument("--tune_log_interval", type=int, default=20)
    p.set_defaults(func=batch_report)

    args = parser.parse_args()
    if args.cmd in ("run", "run-parallel"):
        if args.master_addr is None:
            args.master_addr = "11.131.210.78"
        if args.port is None:
            args.port = 8765
    return args


if __name__ == "__main__":
    args = parse_args()
    args.func(args)
