#!/usr/bin/env python3
"""
Baseline CTM Benchmark — Paper Reproduction + Validated CTM-LLM Idea Transfer

Runs the original CTM on all 6 paper tasks (parity, sort, QAMNIST, mazes,
image_classification, rl) with paper configs, then applies validated CTM-LLM
ideas (regional multi-pass, sparsity, simplified core, reflex/fast-path) to
measure their impact on the original tasks.

Usage (on compute machine, 8×H100):
    # Dry-run: print plan only
    python scripts/experiment_plan_baseline_bench.py plan

    # Write CSV
    python scripts/experiment_plan_baseline_bench.py csv

    # Run a single experiment by name (locally)
    python scripts/experiment_plan_baseline_bench.py run --name bl00_parity_ctm_paper

    # Submit all to cluster pool (sequentially)
    python scripts/experiment_plan_baseline_bench.py submit --stage bl00

    # Submit all stages to cluster pool
    python scripts/experiment_plan_baseline_bench.py submit --stage all

    # Run a full stage sequentially on this machine
    python scripts/experiment_plan_baseline_bench.py run-stage --stage bl00

Output:
    scripts/runs/baseline_bench/{name}.log
    scripts/runs/metrics/baseline_bench_summary.csv
"""

import csv
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
RUNS_DIR = SCRIPTS / "runs" / "baseline_bench"
METRICS_DIR = SCRIPTS / "runs" / "metrics"
DEFAULT_CONFIG = "infra/envs/smoke_baseline.env"
DEFAULT_MASTER_ADDR = "11.131.210.78"
DEFAULT_PORT = 8765

STAGES = (
    "bl00", "bl01", "bl02", "bl03", "bl04", "bl05", "bl06",
    "all",
)
PREFIXES = tuple(f"{s}_" for s in STAGES if s != "all")
METRICS_PREFIX = "baseline_bench"


# ─── helpers ───────────────────────────────────────────────────────────

def _p(train_module, extra_args=None):
    """Build a command string: <module> --flag1 val1 --flag2 val2 ...
    
    For direct execution: python -m <result>
    For pool submission: <result> (run_via_pool.sh adds python -m)
    """
    parts = [train_module]
    if extra_args:
        for k, v in extra_args.items():
            if v is None:
                continue
            if isinstance(v, bool):
                if v:
                    parts.append(f"--{k}")
                else:
                    parts.append(f"--no-{k}")
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
    }


# ─── Stage builders ────────────────────────────────────────────────────

def build_plan(stage, plan_size="full"):
    plan = []

    if stage in ("bl00", "all"):
        plan.extend(_build_bl00_paper_reproduction())
    if stage in ("bl01", "all"):
        plan.extend(_build_bl01_simplified_core())
    if stage in ("bl02", "all"):
        plan.extend(_build_bl02_tick_sweep())
    if stage in ("bl03", "all"):
        plan.extend(_build_bl03_sparse_cell())
    if stage in ("bl04", "all"):
        plan.extend(_build_bl04_baselines_lstm_ff())
    if stage in ("bl05", "all"):
        plan.extend(_build_bl05_rl_paper())
    if stage in ("bl06", "all"):
        plan.extend(_build_bl06_validated_ideas_transfer())

    return plan


# ─── BL00: Paper Reproduction (paper configs, reduced steps for feasibility) ─

def _build_bl00_paper_reproduction():
    """Reproduce paper configs at reduced training length to establish baseline numbers."""
    return [
        # ── Parity: CTM paper config (75 ticks, d1024) ──
        exp(
            "bl00_parity_ctm_paper",
            "Parity: CTM paper config (d1024, 75 ticks, 200K iters)",
            _p("baseline.tasks.parity.train", {
                "log_dir": "logs/baseline_bench/bl00_parity_ctm_paper",
                "seed": 0,
                "iterations": 75,
                "memory_length": 25,
                "parity_sequence_length": 64,
                "d_model": 1024,
                "d_input": 512,
                "n_synch_out": 32,
                "n_synch_action": 32,
                "synapse_depth": 1,
                "heads": 8,
                "memory_hidden_dims": 16,
                "dropout": 0.0,
                "deep_memory": True,
                "do_normalisation": False,
                "positional_embedding_type": "custom-rotational-1d",
                "backbone_type": "parity_backbone",
                "weight_decay": 0.0,
                "gradient_clipping": 0.9,
                "use_scheduler": True,
                "scheduler_type": "cosine",
                "batch_size": 64,
                "batch_size_test": 256,
                "lr": 1e-4,
                "training_iterations": 200001,
                "warmup_steps": 500,
                "track_every": 1000,
                "save_every": 10000,
                "reload": False,
                "device": [0],
                "use_amp": False,
                "neuron_select_type": "random",
                "n_test_batches": 20,
            }),
            tags=["parity", "ctm", "paper"],
        ),
        # ── Parity: shorter tick sweeps (for comparison) ──
        exp(
            "bl00_parity_ctm_tick25",
            "Parity: CTM 25 ticks (paper mid-range)",
            _p("baseline.tasks.parity.train", {
                "log_dir": "logs/baseline_bench/bl00_parity_ctm_tick25",
                "seed": 0, "iterations": 25, "memory_length": 10,
                "parity_sequence_length": 64,
                "d_model": 1024, "d_input": 512,
                "n_synch_out": 32, "n_synch_action": 32,
                "synapse_depth": 1, "heads": 8,
                "memory_hidden_dims": 16, "dropout": 0.0,
                "deep_memory": True, "do_normalisation": False,
                "positional_embedding_type": "custom-rotational-1d",
                "backbone_type": "parity_backbone",
                "weight_decay": 0.0, "gradient_clipping": 0.9,
                "use_scheduler": True, "scheduler_type": "cosine",
                "batch_size": 64, "batch_size_test": 256,
                "lr": 1e-4, "training_iterations": 100001,
                "warmup_steps": 500, "track_every": 1000,
                "save_every": 10000, "reload": False,
                "device": [0], "use_amp": False,
                "neuron_select_type": "random", "n_test_batches": 20,
            }),
            tags=["parity", "ctm", "tick-ablation"],
        ),
        exp(
            "bl00_parity_ctm_tick10",
            "Parity: CTM 10 ticks",
            _p("baseline.tasks.parity.train", {
                "log_dir": "logs/baseline_bench/bl00_parity_ctm_tick10",
                "seed": 0, "iterations": 10, "memory_length": 5,
                "parity_sequence_length": 64,
                "d_model": 1024, "d_input": 512,
                "n_synch_out": 32, "n_synch_action": 32,
                "synapse_depth": 1, "heads": 8,
                "memory_hidden_dims": 16, "dropout": 0.0,
                "deep_memory": True, "do_normalisation": False,
                "positional_embedding_type": "custom-rotational-1d",
                "backbone_type": "parity_backbone",
                "weight_decay": 0.0, "gradient_clipping": 0.9,
                "use_scheduler": True, "scheduler_type": "cosine",
                "batch_size": 64, "batch_size_test": 256,
                "lr": 1e-4, "training_iterations": 100001,
                "warmup_steps": 500, "track_every": 1000,
                "save_every": 10000, "reload": False,
                "device": [0], "use_amp": False,
                "neuron_select_type": "random", "n_test_batches": 20,
            }),
            tags=["parity", "ctm", "tick-ablation"],
        ),

        # ── Sort: CTM default config (30 numbers, 50 ticks) ──
        exp(
            "bl00_sort_ctm_paper",
            "Sort: CTM paper config (d512, 50 ticks, 100K iters, N=30)",
            _p("baseline.tasks.sort.train", {
                "log_dir": "logs/baseline_bench/bl00_sort_ctm_paper",
                "seed": 412, "iterations": 50, "memory_length": 25,
                "d_model": 512, "d_input": 128,
                "n_synch_out": 32, "n_synch_action": 32,
                "synapse_depth": 4, "heads": 4,
                "memory_hidden_dims": 4, "dropout": 0.0,
                "deep_memory": True, "do_normalisation": False,
                "positional_embedding_type": "none",
                "neuron_select_type": "random-pairing",
                "n_random_pairing_self": 0,
                "N_to_sort": 30,
                "batch_size": 32, "batch_size_test": 32,
                "lr": 1e-3, "training_iterations": 100001,
                "warmup_steps": 5000,
                "use_scheduler": True, "scheduler_type": "cosine",
                "weight_decay": 0.0, "gradient_clipping": -1,
                "track_every": 1000, "save_every": 10000,
                "reload": False, "device": [0],
            }),
            tags=["sort", "ctm", "paper"],
        ),

        # ── QAMNIST: CTM paper config (10 ticks/input, 300K iters) ──
        exp(
            "bl00_qamnist_ctm_paper",
            "QAMNIST: CTM paper config (d1024, 10 ticks/input, 300K iters)",
            _p("baseline.tasks.qamnist.train", {
                "log_dir": "logs/baseline_bench/bl00_qamnist_ctm_paper",
                "seed": 0, "model_type": "ctm",
                "memory_length": 30,
                "q_num_images": 3, "q_num_images_delta": 2,
                "q_num_repeats_per_input": 10,
                "q_num_operations": 3, "q_num_operations_delta": 2,
                "q_num_answer_steps": 10,
                "d_model": 1024, "d_input": 64,
                "n_synch_out": 32, "n_synch_action": 32,
                "synapse_depth": 1, "heads": 4,
                "memory_hidden_dims": 16, "dropout": 0.0,
                "deep_memory": True, "do_normalisation": False,
                "weight_decay": 0.0,
                "use_scheduler": True, "scheduler_type": "cosine",
                "batch_size": 64, "batch_size_test": 256,
                "lr": 1e-4, "training_iterations": 300001,
                "warmup_steps": 500, "track_every": 1000,
                "save_every": 10000, "reload": False,
                "device": [0], "use_amp": False,
                "neuron_select_type": "random",
                "data_root": "baseline/data/",
                "n_test_batches": 20,
            }),
            tags=["qamnist", "ctm", "paper"],
        ),

        # ── Mazes: CTM paper config — using small-mazes (15x15) ──
        # Note: data is at baseline/data/mazes/ (originally small-mazes, 15x15)
        # dataset arg format: mazes-{size}, extracts size as subfolder
        # We only have 'small' equivalent, so use mazes-small with data_root=baseline/data
        exp(
            "bl00_mazes_ctm_small",
            "Mazes: CTM on small mazes (d2048, 75 ticks, ~1M iters)",
            _p("baseline.tasks.mazes.train", {
                "log_dir": "logs/baseline_bench/bl00_mazes_ctm_small",
                "model": "ctm",
                "neuron_select_type": "first-last",
                "dataset": "mazes-small",
                "synapse_depth": 8, "heads": 16,
                "iterations": 75, "memory_length": 25,
                "d_model": 2048, "d_input": 512,
                "backbone_type": "resnet34-2",
                "n_synch_out": 64, "n_synch_action": 32,
                "memory_hidden_dims": 32,
                "deep_memory": True, "do_normalisation": False,
                "weight_decay": 0.0, "dropout": 0.1,
                "positional_embedding_type": "none",
                "maze_route_length": 100,
                "cirriculum_lookahead": 5,
                "expand_range": False,
                "batch_size": 64, "batch_size_test": 128,
                "lr": 1e-4, "training_iterations": 100001,
                "warmup_steps": 10000,
                "use_scheduler": True, "scheduler_type": "cosine",
                "gradient_clipping": -1,
                "track_every": 1000, "save_every": 10000,
                "reload": False, "device": [0],
                "data_root": "baseline/data/mazes",
                "n_test_batches": 20,
            }),
            tags=["mazes", "ctm", "paper"],
        ),

        # ── CIFAR-10: CTM paper config ──
        exp(
            "bl00_cifar10_ctm_paper",
            "CIFAR-10: CTM paper config (d256, 50 ticks, ResNet18-1, 600K iters)",
            _p("baseline.tasks.image_classification.train", {
                "log_dir": "logs/baseline_bench/bl00_cifar10_ctm_paper",
                "model": "ctm", "dataset": "cifar10",
                "d_model": 256, "d_input": 64,
                "synapse_depth": 5, "heads": 16,
                "n_synch_out": 256, "n_synch_action": 512,
                "n_random_pairing_self": 0,
                "neuron_select_type": "random-pairing",
                "iterations": 50, "memory_length": 15,
                "deep_memory": True, "memory_hidden_dims": 64,
                "dropout": 0.0, "dropout_nlm": 0,
                "do_normalisation": False,
                "positional_embedding_type": "none",
                "backbone_type": "resnet18-1",
                "training_iterations": 600001,
                "warmup_steps": 1000,
                "use_scheduler": True, "scheduler_type": "cosine",
                "weight_decay": 1e-4,
                "save_every": 1000, "track_every": 2000,
                "n_test_batches": 50,
                "batch_size": 512, "batch_size_test": 512,
                "lr": 1e-4, "device": [0], "seed": 1,
                "data_root": "baseline/data/",
            }),
            tags=["cifar10", "ctm", "paper"],
        ),
    ]


# ─── BL01: Simplified Core (validated idea #4 from CTM-LLM) ─────────────

def _build_bl01_simplified_core():
    """Test simplified synapse/memory configs (sd2_mh2) on parity and QAMNIST."""
    base_parity = {
        "parity_sequence_length": 64,
        "d_model": 1024, "d_input": 512,
        "n_synch_out": 32, "n_synch_action": 32,
        "heads": 8, "dropout": 0.0,
        "deep_memory": True, "do_normalisation": False,
        "positional_embedding_type": "custom-rotational-1d",
        "backbone_type": "parity_backbone",
        "weight_decay": 0.0, "gradient_clipping": 0.9,
        "use_scheduler": True, "scheduler_type": "cosine",
        "batch_size": 64, "batch_size_test": 256,
        "lr": 1e-4, "training_iterations": 100001,
        "warmup_steps": 500, "track_every": 1000,
        "save_every": 10000, "reload": False,
        "device": [0], "use_amp": False,
        "neuron_select_type": "random", "n_test_batches": 20,
    }
    return [
        exp(
            "bl01_parity_sd2_mh2",
            "Parity: simplified core (synapse_depth=2, mem_hidden=2, 75 ticks)",
            _p("baseline.tasks.parity.train", {
                **base_parity,
                "log_dir": "logs/baseline_bench/bl01_parity_sd2_mh2",
                "seed": 0, "iterations": 75, "memory_length": 25,
                "synapse_depth": 2, "memory_hidden_dims": 2,
            }),
            tags=["parity", "simplified-core"],
        ),
        exp(
            "bl01_parity_sd2_mh2_tick2",
            "Parity: simplified core (sd2, mh2) with tick2",
            _p("baseline.tasks.parity.train", {
                **base_parity,
                "log_dir": "logs/baseline_bench/bl01_parity_sd2_mh2_tick2",
                "seed": 0, "iterations": 2, "memory_length": 2,
                "synapse_depth": 2, "memory_hidden_dims": 2,
            }),
            tags=["parity", "simplified-core", "tick-ablation"],
        ),
        exp(
            "bl01_qamnist_sd2_mh2",
            "QAMNIST: simplified core (sd2, mh2, 50 ticks)",
            _p("baseline.tasks.qamnist.train", {
                "log_dir": "logs/baseline_bench/bl01_qamnist_sd2_mh2",
                "seed": 0, "model_type": "ctm",
                "memory_length": 30,
                "q_num_images": 3, "q_num_images_delta": 2,
                "q_num_repeats_per_input": 10,
                "q_num_operations": 3, "q_num_operations_delta": 2,
                "q_num_answer_steps": 10,
                "d_model": 1024, "d_input": 64,
                "n_synch_out": 32, "n_synch_action": 32,
                "synapse_depth": 2, "heads": 4,
                "memory_hidden_dims": 2, "dropout": 0.0,
                "deep_memory": True, "do_normalisation": False,
                "weight_decay": 0.0,
                "use_scheduler": True, "scheduler_type": "cosine",
                "batch_size": 64, "batch_size_test": 256,
                "lr": 1e-4, "training_iterations": 100001,
                "warmup_steps": 500, "track_every": 1000,
                "save_every": 10000, "reload": False,
                "device": [0], "use_amp": False,
                "neuron_select_type": "random",
                "data_root": "baseline/data/",
                "n_test_batches": 20,
            }),
            tags=["qamnist", "simplified-core"],
        ),
        exp(
            "bl01_sort_sd2_mh2",
            "Sort: simplified core (sd2, mh2, 50 ticks)",
            _p("baseline.tasks.sort.train", {
                "log_dir": "logs/baseline_bench/bl01_sort_sd2_mh2",
                "seed": 412, "iterations": 50, "memory_length": 25,
                "d_model": 512, "d_input": 128,
                "n_synch_out": 32, "n_synch_action": 32,
                "synapse_depth": 2, "heads": 4,
                "memory_hidden_dims": 2, "dropout": 0.0,
                "deep_memory": True, "do_normalisation": False,
                "positional_embedding_type": "none",
                "neuron_select_type": "random-pairing",
                "n_random_pairing_self": 0,
                "N_to_sort": 30,
                "batch_size": 32, "batch_size_test": 32,
                "lr": 1e-3, "training_iterations": 100001,
                "warmup_steps": 5000,
                "use_scheduler": True, "scheduler_type": "cosine",
                "weight_decay": 0.0, "gradient_clipping": -1,
                "track_every": 1000, "save_every": 10000,
                "reload": False, "device": [0],
            }),
            tags=["sort", "simplified-core"],
        ),
    ]


# ─── BL02: Tick Sweep (validated idea #9: tick2 is best) ───────────────

def _build_bl02_tick_sweep():
    """Tick sweep on sort and QAMNIST to confirm tick2 sweet spot."""
    return [
        exp(
            "bl02_sort_tick1",
            "Sort: 1 tick (vs default 50)",
            _p("baseline.tasks.sort.train", {
                "log_dir": "logs/baseline_bench/bl02_sort_tick1",
                "seed": 412, "iterations": 1, "memory_length": 2,
                "d_model": 512, "d_input": 128,
                "n_synch_out": 32, "n_synch_action": 32,
                "synapse_depth": 4, "heads": 4,
                "memory_hidden_dims": 4, "dropout": 0.0,
                "deep_memory": True, "do_normalisation": False,
                "positional_embedding_type": "none",
                "neuron_select_type": "random-pairing",
                "n_random_pairing_self": 0, "N_to_sort": 30,
                "batch_size": 32, "batch_size_test": 32,
                "lr": 1e-3, "training_iterations": 100001,
                "warmup_steps": 5000,
                "use_scheduler": True, "scheduler_type": "cosine",
                "weight_decay": 0.0, "gradient_clipping": -1,
                "track_every": 1000, "save_every": 10000,
                "reload": False, "device": [0],
            }),
            tags=["sort", "tick-ablation"],
        ),
        exp(
            "bl02_sort_tick2",
            "Sort: 2 ticks",
            _p("baseline.tasks.sort.train", {
                "log_dir": "logs/baseline_bench/bl02_sort_tick2",
                "seed": 412, "iterations": 2, "memory_length": 5,
                "d_model": 512, "d_input": 128,
                "n_synch_out": 32, "n_synch_action": 32,
                "synapse_depth": 4, "heads": 4,
                "memory_hidden_dims": 4, "dropout": 0.0,
                "deep_memory": True, "do_normalisation": False,
                "positional_embedding_type": "none",
                "neuron_select_type": "random-pairing",
                "n_random_pairing_self": 0, "N_to_sort": 30,
                "batch_size": 32, "batch_size_test": 32,
                "lr": 1e-3, "training_iterations": 100001,
                "warmup_steps": 5000,
                "use_scheduler": True, "scheduler_type": "cosine",
                "weight_decay": 0.0, "gradient_clipping": -1,
                "track_every": 1000, "save_every": 10000,
                "reload": False, "device": [0],
            }),
            tags=["sort", "tick-ablation"],
        ),
        exp(
            "bl02_qamnist_tick1",
            "QAMNIST: 1 tick/input (vs default 10)",
            _p("baseline.tasks.qamnist.train", {
                "log_dir": "logs/baseline_bench/bl02_qamnist_tick1",
                "seed": 0, "model_type": "ctm",
                "memory_length": 3,
                "q_num_images": 3, "q_num_images_delta": 2,
                "q_num_repeats_per_input": 1,
                "q_num_operations": 3, "q_num_operations_delta": 2,
                "q_num_answer_steps": 3,
                "d_model": 1024, "d_input": 64,
                "n_synch_out": 32, "n_synch_action": 32,
                "synapse_depth": 1, "heads": 4,
                "memory_hidden_dims": 16, "dropout": 0.0,
                "deep_memory": True, "do_normalisation": False,
                "weight_decay": 0.0,
                "use_scheduler": True, "scheduler_type": "cosine",
                "batch_size": 64, "batch_size_test": 256,
                "lr": 1e-4, "training_iterations": 100001,
                "warmup_steps": 500, "track_every": 1000,
                "save_every": 10000, "reload": False,
                "device": [0], "use_amp": False,
                "neuron_select_type": "random",
                "data_root": "baseline/data/",
                "n_test_batches": 20,
            }),
            tags=["qamnist", "tick-ablation"],
        ),
        exp(
            "bl02_qamnist_tick5",
            "QAMNIST: 5 ticks/input",
            _p("baseline.tasks.qamnist.train", {
                "log_dir": "logs/baseline_bench/bl02_qamnist_tick5",
                "seed": 0, "model_type": "ctm",
                "memory_length": 15,
                "q_num_images": 3, "q_num_images_delta": 2,
                "q_num_repeats_per_input": 5,
                "q_num_operations": 3, "q_num_operations_delta": 2,
                "q_num_answer_steps": 5,
                "d_model": 1024, "d_input": 64,
                "n_synch_out": 32, "n_synch_action": 32,
                "synapse_depth": 1, "heads": 4,
                "memory_hidden_dims": 16, "dropout": 0.0,
                "deep_memory": True, "do_normalisation": False,
                "weight_decay": 0.0,
                "use_scheduler": True, "scheduler_type": "cosine",
                "batch_size": 64, "batch_size_test": 256,
                "lr": 1e-4, "training_iterations": 100001,
                "warmup_steps": 500, "track_every": 1000,
                "save_every": 10000, "reload": False,
                "device": [0], "use_amp": False,
                "neuron_select_type": "random",
                "data_root": "baseline/data/",
                "n_test_batches": 20,
            }),
            tags=["qamnist", "tick-ablation"],
        ),
    ]


# ─── BL03: Small-dimension Sparse (validated idea #2: d512/tick2 sparse) ──

def _build_bl03_sparse_cell():
    """Test smaller d_model configs for efficiency frontier."""
    return [
        exp(
            "bl03_parity_d512_tick2",
            "Parity: d512, 2 ticks (CTM-LLM sweet spot)",
            _p("baseline.tasks.parity.train", {
                "log_dir": "logs/baseline_bench/bl03_parity_d512_tick2",
                "seed": 0, "iterations": 2, "memory_length": 2,
                "parity_sequence_length": 64,
                "d_model": 512, "d_input": 256,
                "n_synch_out": 32, "n_synch_action": 32,
                "synapse_depth": 2, "heads": 4,
                "memory_hidden_dims": 2, "dropout": 0.0,
                "deep_memory": True, "do_normalisation": False,
                "positional_embedding_type": "custom-rotational-1d",
                "backbone_type": "parity_backbone",
                "weight_decay": 0.0, "gradient_clipping": 0.9,
                "use_scheduler": True, "scheduler_type": "cosine",
                "batch_size": 64, "batch_size_test": 256,
                "lr": 1e-4, "training_iterations": 100001,
                "warmup_steps": 500, "track_every": 1000,
                "save_every": 10000, "reload": False,
                "device": [0], "use_amp": False,
                "neuron_select_type": "random", "n_test_batches": 20,
            }),
            tags=["parity", "small-dim"],
        ),
        exp(
            "bl03_sort_d256_tick2",
            "Sort: d256, 2 ticks (small-dim)",
            _p("baseline.tasks.sort.train", {
                "log_dir": "logs/baseline_bench/bl03_sort_d256_tick2",
                "seed": 412, "iterations": 2, "memory_length": 5,
                "d_model": 256, "d_input": 64,
                "n_synch_out": 32, "n_synch_action": 32,
                "synapse_depth": 2, "heads": 4,
                "memory_hidden_dims": 2, "dropout": 0.0,
                "deep_memory": True, "do_normalisation": False,
                "positional_embedding_type": "none",
                "neuron_select_type": "random-pairing",
                "n_random_pairing_self": 0, "N_to_sort": 30,
                "batch_size": 32, "batch_size_test": 32,
                "lr": 1e-3, "training_iterations": 100001,
                "warmup_steps": 5000,
                "use_scheduler": True, "scheduler_type": "cosine",
                "weight_decay": 0.0, "gradient_clipping": -1,
                "track_every": 1000, "save_every": 10000,
                "reload": False, "device": [0],
            }),
            tags=["sort", "small-dim"],
        ),
        exp(
            "bl03_cifar10_d128_tick2",
            "CIFAR-10: d128, 2 ticks (ultra-small)",
            _p("baseline.tasks.image_classification.train", {
                "log_dir": "logs/baseline_bench/bl03_cifar10_d128_tick2",
                "model": "ctm", "dataset": "cifar10",
                "d_model": 128, "d_input": 32,
                "synapse_depth": 2, "heads": 4,
                "n_synch_out": 64, "n_synch_action": 64,
                "n_random_pairing_self": 0,
                "neuron_select_type": "random-pairing",
                "iterations": 2, "memory_length": 5,
                "deep_memory": True, "memory_hidden_dims": 4,
                "dropout": 0.0, "dropout_nlm": 0,
                "do_normalisation": False,
                "positional_embedding_type": "none",
                "backbone_type": "resnet18-1",
                "training_iterations": 200001,
                "warmup_steps": 1000,
                "use_scheduler": True, "scheduler_type": "cosine",
                "weight_decay": 1e-4,
                "save_every": 2000, "track_every": 2000,
                "n_test_batches": 50,
                "batch_size": 512, "batch_size_test": 512,
                "lr": 1e-4, "device": [0], "seed": 1,
                "data_root": "baseline/data/",
            }),
            tags=["cifar10", "small-dim"],
        ),
    ]


# ─── BL04: LSTM and FF Baselines ───────────────────────────────────────

def _build_bl04_baselines_lstm_ff():
    """Run LSTM and FF baselines for comparison on parity, QAMNIST, CIFAR-10."""
    return [
        exp(
            "bl04_parity_lstm_paper",
            "Parity: LSTM baseline (paper config, 75 ticks)",
            _p("baseline.tasks.parity.train", {
                "log_dir": "logs/baseline_bench/bl04_parity_lstm_paper",
                "seed": 0, "model_type": "lstm",
                "iterations": 75, "memory_length": 25,
                "parity_sequence_length": 64,
                "d_model": 1024, "d_input": 512,
                "n_synch_out": 32, "n_synch_action": 32,
                "synapse_depth": 1, "heads": 8,
                "memory_hidden_dims": 16, "dropout": 0.0,
                "deep_memory": True, "do_normalisation": False,
                "positional_embedding_type": "custom-rotational-1d",
                "backbone_type": "parity_backbone",
                "weight_decay": 0.0, "gradient_clipping": 0.9,
                "use_scheduler": True, "scheduler_type": "cosine",
                "batch_size": 64, "batch_size_test": 256,
                "lr": 1e-4, "training_iterations": 200001,
                "warmup_steps": 500, "track_every": 1000,
                "save_every": 10000, "reload": False,
                "device": [0], "use_amp": False,
                "neuron_select_type": "random", "n_test_batches": 20,
            }),
            tags=["parity", "lstm", "baseline"],
        ),
        exp(
            "bl04_qamnist_lstm_paper",
            "QAMNIST: LSTM baseline (10 ticks/input, 300K iters)",
            _p("baseline.tasks.qamnist.train", {
                "log_dir": "logs/baseline_bench/bl04_qamnist_lstm_paper",
                "seed": 0, "model_type": "lstm",
                "memory_length": 30,
                "q_num_images": 3, "q_num_images_delta": 2,
                "q_num_repeats_per_input": 10,
                "q_num_operations": 3, "q_num_operations_delta": 2,
                "q_num_answer_steps": 10,
                "d_model": 1024, "d_input": 64,
                "n_synch_out": 32, "n_synch_action": 32,
                "synapse_depth": 1, "heads": 4,
                "memory_hidden_dims": 16, "dropout": 0.0,
                "deep_memory": True, "do_normalisation": False,
                "weight_decay": 0.0,
                "use_scheduler": True, "scheduler_type": "cosine",
                "batch_size": 64, "batch_size_test": 256,
                "lr": 1e-4, "training_iterations": 300001,
                "warmup_steps": 500, "track_every": 1000,
                "save_every": 10000, "reload": False,
                "device": [0], "use_amp": False,
                "neuron_select_type": "random",
                "data_root": "baseline/data/",
                "n_test_batches": 20,
            }),
            tags=["qamnist", "lstm", "baseline"],
        ),
        exp(
            "bl04_cifar10_lstm_paper",
            "CIFAR-10: LSTM baseline (paper config, 600K iters)",
            _p("baseline.tasks.image_classification.train", {
                "log_dir": "logs/baseline_bench/bl04_cifar10_lstm_paper",
                "model": "lstm", "dataset": "cifar10",
                "num_layers": 2,
                "d_model": 256, "d_input": 64,
                "heads": 16, "iterations": 50,
                "dropout": 0.0,
                "positional_embedding_type": "none",
                "backbone_type": "resnet18-1",
                "training_iterations": 600001,
                "warmup_steps": 2000,
                "use_scheduler": True, "scheduler_type": "cosine",
                "weight_decay": 1e-4,
                "save_every": 1000, "track_every": 2000,
                "n_test_batches": 50,
                "batch_size": 512, "batch_size_test": 512,
                "lr": 1e-4, "device": [0], "seed": 1,
                "data_root": "baseline/data/",
            }),
            tags=["cifar10", "lstm", "baseline"],
        ),
        exp(
            "bl04_cifar10_ff_paper",
            "CIFAR-10: FF baseline (paper config, 600K iters)",
            _p("baseline.tasks.image_classification.train", {
                "log_dir": "logs/baseline_bench/bl04_cifar10_ff_paper",
                "model": "ff", "dataset": "cifar10",
                "d_model": 256,
                "memory_hidden_dims": 64,
                "dropout": 0.0, "dropout_nlm": 0,
                "backbone_type": "resnet18-1",
                "training_iterations": 600001,
                "warmup_steps": 1000,
                "use_scheduler": True, "scheduler_type": "cosine",
                "weight_decay": 1e-4,
                "save_every": 1000, "track_every": 2000,
                "n_test_batches": 50,
                "batch_size": 512, "batch_size_test": 512,
                "lr": 1e-4, "device": [0], "seed": 1,
                "data_root": "baseline/data/",
            }),
            tags=["cifar10", "ff", "baseline"],
        ),
    ]


# ─── BL05: RL Paper Configs ────────────────────────────────────────────

def _build_bl05_rl_paper():
    """RL tasks: Acrobot and CartPole with CTM and LSTM (3 seeds each)."""
    exps = []
    for run_seed in [1, 2, 3]:
        exps.append(exp(
            f"bl05_acrobot_ctm2_seed{run_seed}",
            f"Acrobot: CTM 2 ticks, seed={run_seed} (paper config)",
            _p("baseline.tasks.rl.train", {
                "model_type": "ctm", "env_id": "Acrobot-v1",
                "log_dir": f"logs/baseline_bench/bl05_acrobot_ctm2_seed{run_seed}",
                "tb_log_dir": "logs/baseline_bench/bl05_rl_runs",
                "seed": run_seed, "iterations": 2,
                "run_name": f"bl05_acrobot_ctm2_s{run_seed}",
                "d_model": 256, "d_input": 64,
                "memory_hidden_dims": 4, "n_synch_out": 16,
                "discount_gamma": 0.99, "gae_lambda": 0.95,
                "ent_coef": 0.1, "vf_coef": 0.25,
                "memory_length": 10,
                "max_environment_steps": 500,
                "total_timesteps": 2000000,
                "num_steps": 100, "anneal_lr": True,
                "num_envs": 12, "update_epochs": 1,
                "mask_velocity": True,
                "continuous_state_trace": True,
                "dropout": 0.0, "lr": 5e-4,
                "track_every": 1000, "save_every": 100,
                "reload": False, "device": [0],
                "neuron_select_type": "first-last",
            }),
            tags=["rl", "acrobot", "ctm"],
        ))
        exps.append(exp(
            f"bl05_acrobot_lstm2_seed{run_seed}",
            f"Acrobot: LSTM 2 ticks, seed={run_seed} (paper config)",
            _p("baseline.tasks.rl.train", {
                "model_type": "lstm", "env_id": "Acrobot-v1",
                "log_dir": f"logs/baseline_bench/bl05_acrobot_lstm2_seed{run_seed}",
                "tb_log_dir": "logs/baseline_bench/bl05_rl_runs",
                "seed": run_seed, "iterations": 2,
                "run_name": f"bl05_acrobot_lstm2_s{run_seed}",
                "d_model": 256, "d_input": 64,
                "memory_hidden_dims": 4, "n_synch_out": 16,
                "discount_gamma": 0.99, "gae_lambda": 0.95,
                "ent_coef": 0.1, "vf_coef": 0.25,
                "memory_length": 10,
                "max_environment_steps": 500,
                "total_timesteps": 2000000,
                "num_steps": 100, "anneal_lr": True,
                "num_envs": 12, "update_epochs": 1,
                "mask_velocity": True,
                "continuous_state_trace": True,
                "dropout": 0.0, "lr": 5e-4,
                "track_every": 1000, "save_every": 100,
                "reload": False, "device": [0],
                "neuron_select_type": "first-last",
            }),
            tags=["rl", "acrobot", "lstm"],
        ))
    for run_seed in [1, 2, 3]:
        exps.append(exp(
            f"bl05_cartpole_ctm2_seed{run_seed}",
            f"CartPole: CTM 2 ticks, seed={run_seed} (paper config)",
            _p("baseline.tasks.rl.train", {
                "model_type": "ctm", "env_id": "CartPole-v1",
                "log_dir": f"logs/baseline_bench/bl05_cartpole_ctm2_seed{run_seed}",
                "tb_log_dir": "logs/baseline_bench/bl05_rl_runs",
                "seed": run_seed, "iterations": 2,
                "run_name": f"bl05_cartpole_ctm2_s{run_seed}",
                "d_model": 128, "d_input": 128,
                "memory_hidden_dims": 4, "n_synch_out": 16,
                "memory_length": 20,
                "max_environment_steps": 200,
                "total_timesteps": 10000000,
                "num_steps": 50, "num_envs": 256,
                "update_epochs": 4, "lr": 1e-3,
                "anneal_lr": True,
                "mask_velocity": True,
                "continuous_state_trace": True,
                "discount_gamma": 0.99, "gae_lambda": 0.95,
                "ent_coef": 0.1, "vf_coef": 0.25,
                "dropout": 0.0,
                "track_every": 1000, "save_every": 100,
                "reload": False, "device": [0],
                "neuron_select_type": "first-last",
            }),
            tags=["rl", "cartpole", "ctm"],
        ))
    return exps


# ─── BL06: Validated CTM-LLM Ideas → Original CTM Tasks ────────────────

def _build_bl06_validated_ideas_transfer():
    """
    Map the top validated CTM-LLM ideas to baseline CTM tasks:
    1. tick2 as default (idea #9)
    2. simplified core sd2_mh2 (idea #4)  
    3. small d_model (idea #2)
    4. neuron_select_type ablation
    """
    return [
        # ── neuron_select_type ablation on parity ──
        exp(
            "bl06_parity_firstlast",
            "Parity: first-last neuron select (vs random)",
            _p("baseline.tasks.parity.train", {
                "log_dir": "logs/baseline_bench/bl06_parity_firstlast",
                "seed": 0, "iterations": 75, "memory_length": 25,
                "parity_sequence_length": 64,
                "d_model": 1024, "d_input": 512,
                "n_synch_out": 32, "n_synch_action": 32,
                "synapse_depth": 1, "heads": 8,
                "memory_hidden_dims": 16, "dropout": 0.0,
                "deep_memory": True, "do_normalisation": False,
                "positional_embedding_type": "custom-rotational-1d",
                "backbone_type": "parity_backbone",
                "weight_decay": 0.0, "gradient_clipping": 0.9,
                "use_scheduler": True, "scheduler_type": "cosine",
                "batch_size": 64, "batch_size_test": 256,
                "lr": 1e-4, "training_iterations": 100001,
                "warmup_steps": 500, "track_every": 1000,
                "save_every": 10000, "reload": False,
                "device": [0], "use_amp": False,
                "neuron_select_type": "first-last",
                "n_test_batches": 20,
            }),
            tags=["parity", "neuron-select"],
        ),
        exp(
            "bl06_parity_random_pairing",
            "Parity: random-pairing neuron select (vs random)",
            _p("baseline.tasks.parity.train", {
                "log_dir": "logs/baseline_bench/bl06_parity_random_pairing",
                "seed": 0, "iterations": 75, "memory_length": 25,
                "parity_sequence_length": 64,
                "d_model": 1024, "d_input": 512,
                "n_synch_out": 32, "n_synch_action": 32,
                "synapse_depth": 1, "heads": 8,
                "memory_hidden_dims": 16, "dropout": 0.0,
                "deep_memory": True, "do_normalisation": False,
                "positional_embedding_type": "custom-rotational-1d",
                "backbone_type": "parity_backbone",
                "weight_decay": 0.0, "gradient_clipping": 0.9,
                "use_scheduler": True, "scheduler_type": "cosine",
                "batch_size": 64, "batch_size_test": 256,
                "lr": 1e-4, "training_iterations": 100001,
                "warmup_steps": 500, "track_every": 1000,
                "save_every": 10000, "reload": False,
                "device": [0], "use_amp": False,
                "neuron_select_type": "random-pairing",
                "n_test_batches": 20,
            }),
            tags=["parity", "neuron-select"],
        ),

        # ── CIFAR-10: tick ablation ──
        exp(
            "bl06_cifar10_tick2",
            "CIFAR-10: CTM with 2 ticks (vs default 50)",
            _p("baseline.tasks.image_classification.train", {
                "log_dir": "logs/baseline_bench/bl06_cifar10_tick2",
                "model": "ctm", "dataset": "cifar10",
                "d_model": 256, "d_input": 64,
                "synapse_depth": 5, "heads": 16,
                "n_synch_out": 256, "n_synch_action": 512,
                "n_random_pairing_self": 0,
                "neuron_select_type": "random-pairing",
                "iterations": 2, "memory_length": 5,
                "deep_memory": True, "memory_hidden_dims": 64,
                "dropout": 0.0, "dropout_nlm": 0,
                "do_normalisation": False,
                "positional_embedding_type": "none",
                "backbone_type": "resnet18-1",
                "training_iterations": 200001,
                "warmup_steps": 1000,
                "use_scheduler": True, "scheduler_type": "cosine",
                "weight_decay": 1e-4,
                "save_every": 2000, "track_every": 2000,
                "n_test_batches": 50,
                "batch_size": 512, "batch_size_test": 512,
                "lr": 1e-4, "device": [0], "seed": 1,
                "data_root": "baseline/data/",
            }),
            tags=["cifar10", "tick-ablation"],
        ),
        exp(
            "bl06_cifar10_tick10",
            "CIFAR-10: CTM with 10 ticks",
            _p("baseline.tasks.image_classification.train", {
                "log_dir": "logs/baseline_bench/bl06_cifar10_tick10",
                "model": "ctm", "dataset": "cifar10",
                "d_model": 256, "d_input": 64,
                "synapse_depth": 5, "heads": 16,
                "n_synch_out": 256, "n_synch_action": 512,
                "n_random_pairing_self": 0,
                "neuron_select_type": "random-pairing",
                "iterations": 10, "memory_length": 10,
                "deep_memory": True, "memory_hidden_dims": 64,
                "dropout": 0.0, "dropout_nlm": 0,
                "do_normalisation": False,
                "positional_embedding_type": "none",
                "backbone_type": "resnet18-1",
                "training_iterations": 200001,
                "warmup_steps": 1000,
                "use_scheduler": True, "scheduler_type": "cosine",
                "weight_decay": 1e-4,
                "save_every": 2000, "track_every": 2000,
                "n_test_batches": 50,
                "batch_size": 512, "batch_size_test": 512,
                "lr": 1e-4, "device": [0], "seed": 1,
                "data_root": "baseline/data/",
            }),
            tags=["cifar10", "tick-ablation"],
        ),
    ]


# ─── Plan output / CSV ─────────────────────────────────────────────────

def write_plan_csv(plan, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "question", "command", "tags"])
        writer.writeheader()
        for e in plan:
            writer.writerow({
                "name": e["name"],
                "question": e["question"],
                "command": e["command"],
                "tags": ";".join(e["tags"]),
            })


def print_plan_summary(plan):
    stages = {}
    for e in plan:
        prefix = e["name"].split("_")[0]
        stages.setdefault(prefix, []).append(e)

    print(f"\n{'='*80}")
    print(f"BASELINE BENCHMARK PLAN — {len(plan)} experiments total")
    print(f"{'='*80}\n")

    for stage_name, exps in sorted(stages.items()):
        print(f"  {stage_name}: {len(exps)} experiments")
    print()

    stage_descriptions = {
        "bl00": "Paper reproduction (parity, sort, QAMNIST, mazes, CIFAR-10)",
        "bl01": "Simplified core (synapse_depth=2, memory_hidden=2)",
        "bl02": "Tick sweep (sort & QAMNIST at 1/2/5/10 ticks)",
        "bl03": "Small-dimension sparse (d512/d256/d128 tick2)",
        "bl04": "LSTM and FF baselines (parity, QAMNIST, CIFAR-10)",
        "bl05": "RL paper configs (Acrobot + CartPole, 3 seeds each)",
        "bl06": "Validated CTM-LLM ideas transfer (neuron_select, tick ablation)",
    }
    for stage_name, exps in sorted(stages.items()):
        desc = stage_descriptions.get(stage_name, "")
        print(f"\n  ── {stage_name}: {desc} ──")
        for e in exps:
            print(f"     {e['name']}")
            print(f"       {e['question']}")

    print(f"\n{'='*80}")
    print("Paper reference results (from arXiv 2505.05522):")
    print("  Parity (75 ticks):    100% accuracy (some seeds)")
    print("  QAMNIST (10 ticks):   >96% accuracy (CTM) vs <=21% (LSTM)")
    print("  Sort (30 numbers):    qualitative success (CTC-based)")
    print("  Mazes (39x39):        95% per-step, 66% full solve (15x15 ablation)")
    print("  ImageNet-1K:          Top-1 72.47%, Top-5 89.89%")
    print("  CIFAR-10:             CTM > LSTM > FF (graphical)")
    print("  RL (Acrobot/CartPole): CTM ≈ LSTM baselines")
    print(f"{'='*80}\n")


# ─── CLI ────────────────────────────────────────────────────────────────

def submit_to_pool(exp, config, master_addr=None, port=None):
    """Submit a baseline experiment to the cluster pool.
    
    The pool worker runs: bash scripts/train_cluster.sh --config <config> <extra_args>
    With smoke_baseline.env, TRAIN_ENTRY=scripts/run_via_pool.sh, which executes
    `python -m $TRAIN_ARGS`. So extra_args is the task module + flags.
    """
    node_addrs = exp.get("node_addrs") or []
    payload = {
        "config": config,
        "extra_args": exp["command"],
        "node_addrs": node_addrs,
        "env": {
            "CTM_EXPERIMENT_NAME": exp["name"],
            "CTM_METRICS_DIR": "runs/metrics",
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
    """Poll pool until the task is completed or failed."""
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
                    if t["status"] == "failed":
                        log_path = "runs/logs/pool_last_run.log"
                        if os.path.isfile(log_path):
                            print(f"  [pool] last run log ({log_path}):")
                            with open(log_path) as f:
                                for line in f:
                                    print(f"    {line}", end="")
                    return t["status"]
        except Exception:
            pass
        time.sleep(poll_interval)


def run_submit(args):
    """Submit experiments to the cluster pool, one at a time (sequential)."""
    plan = build_plan(args.stage)
    if not plan:
        print("No experiments to submit.")
        return
    print(f"Submitting {len(plan)} experiments to pool at {args.master_addr}:{args.port}")
    for i, e in enumerate(plan):
        print(f"\n[{i+1}/{len(plan)}] {e['name']}")
        print(f"  {e['question']}")
        task = submit_to_pool(e, args.config, args.master_addr, args.port)
        if task and "task_id" in task:
            tid = task["task_id"]
            print(f"  submitted as {tid}, waiting...")
            wait_until_idle(args.master_addr, args.port, tid, args.poll_interval)
        else:
            print(f"  WARNING: submit failed, skipping")
            continue
    print(f"\nAll {len(plan)} experiments submitted and completed.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Baseline CTM Benchmark")
    parser.add_argument("action",
                        choices=["plan", "csv", "run", "run-stage", "count", "submit", "commands"],
                        help="Action: plan=print summary, csv=write CSV, run=execute one, "
                             "run-stage=execute stage sequentially, submit=submit to pool, "
                             "commands=print all commands, count=show totals")
    parser.add_argument("--name", type=str, help="Experiment name (for 'run')")
    parser.add_argument("--stage", type=str, default="all",
                        help="Stage to build (default: all)")
    parser.add_argument("--plan-size", type=str, default="full",
                        choices=["core", "full", "wide"])
    parser.add_argument("--output", type=str,
                        default=str(RUNS_DIR / "baseline_bench_plan.csv"))
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG,
                        help="Cluster env file for pool submit")
    parser.add_argument("--master_addr", type=str, default=DEFAULT_MASTER_ADDR)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--poll_interval", type=float, default=30.0)
    args = parser.parse_args()

    plan = build_plan(args.stage, args.plan_size)

    if args.action == "plan":
        print_plan_summary(plan)

    elif args.action == "csv":
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        write_plan_csv(plan, args.output)
        print(f"Wrote {len(plan)} experiments to {args.output}")

    elif args.action == "count":
        print(f"{len(plan)} experiments")
        stages = {}
        for e in plan:
            prefix = e["name"].split("_")[0]
            stages.setdefault(prefix, []).append(e)
        for s, exps in sorted(stages.items()):
            print(f"  {s}: {len(exps)}")

    elif args.action == "commands":
        for e in plan:
            print(f"\n# {e['name']}: {e['question']}")
            print(e["command"])

    elif args.action == "run":
        if not args.name:
            print("Error: --name required for 'run'")
            sys.exit(1)
        target = None
        for e in plan:
            if e["name"] == args.name:
                target = e
                break
        if not target:
            print(f"Error: experiment '{args.name}' not found")
            sys.exit(1)
        print(f"Running: {target['name']}")
        print(f"Command: python -m {target['command']}")
        sys.exit(os.system(f"python -m {target['command']}"))

    elif args.action == "run-stage":
        stage = args.stage
        if not stage or stage == "all":
            print("Error: specify --stage for run-stage (not 'all')")
            sys.exit(1)
        stage_exps = [e for e in plan if e["name"].startswith(stage + "_")]
        if not stage_exps:
            print(f"Error: no experiments found for stage '{stage}'")
            sys.exit(1)
        print(f"Running {len(stage_exps)} experiments for stage {stage}")
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        for e in stage_exps:
            log_path = RUNS_DIR / f"{e['name']}.log"
            print(f"  [{e['name']}] → {log_path}")
            with open(log_path, "w") as log_f:
                proc = subprocess.run(
                    f"python -m {e['command']}", shell=True, cwd=str(ROOT),
                    stdout=log_f, stderr=subprocess.STDOUT,
                )
            status = "OK" if proc.returncode == 0 else f"FAIL({proc.returncode})"
            print(f"  [{e['name']}] {status}")

    elif args.action == "submit":
        run_submit(args)


if __name__ == "__main__":
    main()
