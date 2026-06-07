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
    "use_swanlab": None,
}


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


def build_plan(stage):
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
        plan.append(experiment(
            "s01_transformer_16l",
            "Baseline: standard Transformer with same hidden size/layer count.",
            merge_args(model_type="transformer", iterations=1)))
        plan.append(experiment(
            "s01_ctm_16l_tick4",
            "Baseline: current CTM with 4 ticks.",
            merge_args(model_type="ctm", iterations=4)))

    if stage in ("ticks", "all"):
        for ticks in [1, 2, 4, 8, 16]:
            plan.append(experiment(
                f"s02_ctm_tick{ticks}",
                "Measure loss/cost curve as maximum CTM ticks increase.",
                merge_args(model_type="ctm", iterations=ticks)))
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

    if stage in ("cells", "all"):
        for d_model, mem_h, mem_len, depth, sparsity, topk in [
            (256, 4, 10, 3, "none", 256),
            (512, 4, 10, 3, "none", 512),
            (1024, 2, 8, 2, "none", 1024),
            (1024, 2, 8, 2, "topk", 512),
            (1536, 1, 6, 1, "none", 1536),
            (1536, 1, 6, 1, "topk", 512),
            (2048, 1, 6, 1, "topk", 512),
        ]:
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

    return plan


def build_batch_tune_plan(stage, batch_sizes, max_steps, log_interval):
    plan = []
    for exp in build_plan(stage):
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
    cmd.extend(arg_items(exp["args"]))
    return " ".join(shlex.quote(x) for x in cmd)


def write_manifest(plan, path, config, master_addr, port, wait):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        fields = ["index", "name", "base_name", "batch_size", "question", "command"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, exp in enumerate(plan):
            writer.writerow({
                "index": i,
                "name": exp["name"],
                "base_name": exp.get("base_name", exp["name"]),
                "batch_size": exp["args"].get("batch_size", ""),
                "question": exp["question"],
                "command": submit_command(exp, config, master_addr, port, wait),
            })


def print_commands(args):
    plan = apply_batch_profile(build_plan(args.stage), load_batch_profile(args.batch_profile))
    if args.output:
        write_manifest(plan, args.output, args.config, args.master_addr, args.port, args.wait)
        print(f"wrote {len(plan)} experiments: {args.output}")
    for i, exp in enumerate(plan):
        print(f"\n# {i:02d} {exp['name']}")
        print(f"# {exp['question']}")
        print(submit_command(exp, args.config, args.master_addr, args.port, args.wait))


def print_batch_commands(args):
    plan = build_batch_tune_plan(
        args.stage, args.batch_sizes, args.tune_steps, args.tune_log_interval)
    if args.output:
        write_manifest(plan, args.output, args.config, args.master_addr, args.port, args.wait)
        print(f"wrote {len(plan)} batch probes: {args.output}")
    for i, exp in enumerate(plan):
        print(f"\n# {i:02d} {exp['name']}")
        print(f"# {exp['question']}")
        print(submit_command(exp, args.config, args.master_addr, args.port, args.wait))


def pool_status(master_addr, port):
    with urllib.request.urlopen(f"http://{master_addr}:{port}/status", timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


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


def run_plan(args):
    if getattr(args, "batch_tune", False):
        plan = build_batch_tune_plan(
            args.stage, args.batch_sizes, args.tune_steps, args.tune_log_interval)
    else:
        plan = apply_batch_profile(build_plan(args.stage), load_batch_profile(args.batch_profile))
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

    p = sub.add_parser("commands", parents=[common])
    p.add_argument("--output", default="runs/experiment_plans/ctm_plan.csv")
    p.add_argument("--batch_profile", default=None)
    p.set_defaults(func=print_commands)

    p = sub.add_parser("batch-commands", parents=[common])
    p.add_argument("--batch_sizes", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12])
    p.add_argument("--tune_steps", type=int, default=80)
    p.add_argument("--tune_log_interval", type=int, default=20)
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

    args = parser.parse_args()
    if args.cmd == "run":
        if args.master_addr is None:
            args.master_addr = "11.131.210.78"
        if args.port is None:
            args.port = 8765
    return args


if __name__ == "__main__":
    args = parse_args()
    args.func(args)
