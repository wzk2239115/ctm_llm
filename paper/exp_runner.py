"""Shared experiment runner for paper notebooks.

Provides:
  - BASE_CONFIGS: per-task CTM paper configs (self-contained, no side-effect imports)
  - Experiment dataclass + builder helpers for revise / jepa / sparsity / combos
  - run_all(): parallel execution on N local GPUs via subprocess
  - status(): quick progress snapshot
  - collect(): load best_test_acc from saved checkpoints
  - plot_delta_bars(): baseline vs experiment comparison

Usage in notebooks:
    from exp_runner import Experiment, make_revise, run_all, status, collect
    exps = make_revise(tasks=["sort","mazes"], seeds=[0,1,2])
    run_all(exps, gpus=8, log_root="logs/deep/01_revise")
"""

from __future__ import annotations

import subprocess
import time
import re
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ═══════════════════════════════════════════════════
# Per-task base configs (CTM paper defaults)
# ═══════════════════════════════════════════════════

SORT_BASE = dict(
    iterations=50, memory_length=25,
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
    device=[0],
)

PARITY_BASE = dict(
    iterations=75, memory_length=25,
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
    save_every=10000, device=[0], use_amp=False,
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
    device=[0], data_root="baseline/data/mazes", n_test_batches=20,
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
    model_type="ctm", memory_length=30,
    q_num_images=3, q_num_images_delta=2, q_num_repeats_per_input=10,
    q_num_operations=3, q_num_operations_delta=2, q_num_answer_steps=10,
    d_model=1024, d_input=64, n_synch_out=32, n_synch_action=32,
    synapse_depth=1, heads=4, memory_hidden_dims=16, dropout=0.0,
    deep_memory=True, do_normalisation=False, weight_decay=0.0,
    use_scheduler=True, scheduler_type="cosine",
    batch_size=64, batch_size_test=256,
    lr=1e-4, training_iterations=200001,
    warmup_steps=500, track_every=1000, save_every=10000,
    device=[0], use_amp=False,
    neuron_select_type="random",
    data_root="baseline/data/", n_test_batches=20,
)

BASE_CONFIGS = {
    "sort":    ("baseline.tasks.sort.train",               SORT_BASE),
    "parity":  ("baseline.tasks.parity.train",             PARITY_BASE),
    "mazes":   ("baseline.tasks.mazes.train",              MAZES_BASE),
    "cifar10": ("baseline.tasks.image_classification.train", CIFAR10_BASE),
    "qamnist": ("baseline.tasks.qamnist.train",            QAMNIST_BASE),
}

BASELINE_ACC = {
    "cifar10": 0.6443, "mazes": 0.8028, "parity": 0.6797,
    "qamnist": 0.3662, "sort": 0.7146,
}

TASK_COLORS = {
    "cifar10": "#1f77b4", "mazes": "#ff7f0e",
    "parity": "#2ca02c", "qamnist": "#d62728", "sort": "#9467bd",
}


# ═══════════════════════════════════════════════════
# Experiment dataclass + builders
# ═══════════════════════════════════════════════════

@dataclass
class Experiment:
    name: str
    task: str
    module: str
    config: dict[str, Any]
    tags: list[str] = field(default_factory=list)

    def __repr__(self):
        return f"Exp({self.name})"


def _exp(task: str, name_suffix: str, extra: dict, tags: list[str] | None = None) -> Experiment:
    module, base = BASE_CONFIGS[task]
    cfg = dict(base)
    cfg.update(extra)
    name = f"{task}_{name_suffix}"
    return Experiment(name=name, task=task, module=module, config=cfg, tags=tags or [])


def make_revise(tasks, seeds, w=0.1, cp=0.15):
    exps = []
    for task in tasks:
        for s in seeds:
            extra = dict(
                seed=s, draft_mode="revise", draft_block_size=2,
                draft_revise_weight=w, draft_corrupt_prob=cp,
            )
            exps.append(_exp(task, f"revise_w{w}_cp{cp}_s{s}", extra, ["revise"]))
    return exps


def make_jepa(tasks, seeds, weights=(0.1,), loss="cosine", stop_grad=True):
    exps = []
    jepa_defaults = dict(
        cross_tick_jepa_hidden_dim=128,
        cross_tick_jepa_predictor_depth=2,
        cross_tick_jepa_dropout=0.0,
    )
    for task in tasks:
        for w in weights:
            for s in seeds:
                extra = dict(jepa_defaults)
                extra["seed"] = s
                extra["cross_tick_jepa_weight"] = w
                extra["cross_tick_jepa_loss"] = loss
                extra["cross_tick_jepa_target_stop_grad"] = stop_grad
                wstr = str(w).replace(".", "p")
                exps.append(_exp(task, f"jepa_w{wstr}_s{s}", extra, ["jepa"]))
    return exps


def make_sparsity(tasks, seeds, ratios=(0.5,)):
    exps = []
    for task in tasks:
        for r in ratios:
            for s in seeds:
                extra = dict(seed=s, topk_neurons=r)
                rstr = str(r).replace(".", "p")
                exps.append(_exp(task, f"sparsity{rstr}_s{s}", extra, ["sparsity"]))
    return exps


def make_combo(tasks, seeds, use_revise=False, use_jepa=False, use_sparsity=False,
               jepa_w=0.1, sparsity_r=0.5):
    exps = []
    parts = []
    extra = {}
    if use_revise:
        extra.update(draft_mode="revise", draft_block_size=2,
                     draft_revise_weight=0.1, draft_corrupt_prob=0.15)
        parts.append("revise")
    if use_jepa:
        extra.update(
            cross_tick_jepa_weight=jepa_w,
            cross_tick_jepa_hidden_dim=128,
            cross_tick_jepa_predictor_depth=2,
            cross_tick_jepa_dropout=0.0,
        )
        parts.append(f"jepa{str(jepa_w).replace('.','p')}")
    if use_sparsity:
        extra["topk_neurons"] = sparsity_r
        parts.append(f"spar{str(sparsity_r).replace('.','p')}")
    combo_name = "+".join(parts) if parts else "baseline"
    for task in tasks:
        for s in seeds:
            e = dict(extra, seed=s)
            exps.append(_exp(task, f"{combo_name}_s{s}", e, ["combo"] + parts))
    return exps


def make_baselines(tasks, seeds):
    exps = []
    for task in tasks:
        for s in seeds:
            exps.append(_exp(task, f"baseline_s{s}", {"seed": s}, ["baseline"]))
    return exps


# ═══════════════════════════════════════════════════
# Command builder + parallel runner
# ═══════════════════════════════════════════════════

def build_cmd(exp: Experiment, gpu: int, log_dir: str) -> str:
    parts = [f"CUDA_VISIBLE_DEVICES={gpu}", "python", "-m", exp.module]
    for k, v in exp.config.items():
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
    parts.append(f"--log_dir")
    parts.append(log_dir)
    return " ".join(parts)


def run_all(experiments, gpus=8, log_root="logs/deep", dry_run=False):
    """Run experiments in parallel on N GPUs. Blocks until all done."""
    log_root = Path(log_root)
    log_root.mkdir(parents=True, exist_ok=True)

    if dry_run:
        for i, exp in enumerate(experiments):
            gpu = i % gpus
            edir = log_root / exp.name
            cmd = build_cmd(exp, gpu, str(edir))
            print(f"[GPU {gpu}] {exp.name}\n  {cmd[:120]}...")
        print(f"\n({len(experiments)} experiments, dry-run only)")
        return

    running = {}
    pending = list(experiments)
    done, failed = [], []
    t0 = time.time()

    while pending or running:
        while pending and len(running) < gpus:
            free = [g for g in range(gpus) if g not in running]
            if not free:
                break
            gpu = free[0]
            exp = pending.pop(0)
            edir = log_root / exp.name
            edir.mkdir(parents=True, exist_ok=True)
            logfile = open(edir / "train.log", "w")
            cmd = build_cmd(exp, gpu, str(edir))
            proc = subprocess.Popen(cmd, shell=True, stdout=logfile, stderr=subprocess.STDOUT)
            running[gpu] = (exp, proc, logfile)
            print(f"[GPU {gpu}] START {exp.name}  (elapsed {time.time()-t0:.0f}s)")

        for gpu in list(running):
            exp, proc, logfile = running[gpu]
            if proc.poll() is not None:
                logfile.close()
                rc = proc.returncode
                if rc == 0:
                    done.append(exp)
                    print(f"[GPU {gpu}] DONE  {exp.name}  (elapsed {time.time()-t0:.0f}s)")
                else:
                    failed.append(exp)
                    print(f"[GPU {gpu}] FAIL  {exp.name}  (rc={rc}, elapsed {time.time()-t0:.0f}s)")
                del running[gpu]

        time.sleep(3)

    print(f"\n{'='*60}")
    print(f"Finished: {len(done)} ok, {len(failed)} failed, {time.time()-t0:.0f}s total")
    if failed:
        print(f"Failed: {[e.name for e in failed]}")
    return done, failed


# ═══════════════════════════════════════════════════
# Status + results collection
# ═══════════════════════════════════════════════════

def status(log_root="logs/deep"):
    """Quick progress snapshot of all experiments under log_root."""
    import glob
    log_root = Path(log_root)
    if not log_root.exists():
        print(f"{log_root} does not exist yet")
        return
    rows = []
    for edir in sorted(log_root.iterdir()):
        if not edir.is_dir():
            continue
        log_file = edir / "train.log"
        ckpts = sorted(edir.glob("checkpoint*.pt"))
        if not log_file.exists() and not ckpts:
            rows.append((edir.name, "pending", 0, "-"))
            continue
        if not ckpts:
            rows.append((edir.name, "running" if log_file.exists() else "pending", 0, "-"))
            continue
        last_iter = 0
        m = re.search(r"_(\d+)\.pt$", ckpts[-1].name)
        if m:
            last_iter = int(m.group(1))
        has_final = (edir / "checkpoint.pt").exists()
        state = "done" if has_final else "running"
        rows.append((edir.name, state, last_iter, ckpts[-1].name))
    print(f"{'experiment':40s} {'state':8s} {'last_iter':>10s}  ckpt")
    print("-" * 80)
    for name, state, it, ck in rows:
        print(f"{name:40s} {state:8s} {it:>10d}  {ck}")
    print(f"\n{len(rows)} experiments total")


def _find_latest_ckpt(d):
    ckpts = sorted(d.glob("checkpoint*.pt"))
    if not ckpts:
        return None
    p = d / "checkpoint.pt"
    if p.exists():
        return p
    num = [c for c in ckpts if re.search(r"_(\d+)\.pt$", c)]
    if num:
        return max(num, key=lambda c: int(re.search(r"_(\d+)\.pt$", c).group(1)))
    return ckpts[-1]


def collect(log_root="logs/deep"):
    """Load best_test_acc from each experiment's latest checkpoint."""
    import torch
    import pandas as pd
    log_root = Path(log_root)
    rows = []
    for edir in sorted(log_root.iterdir()):
        if not edir.is_dir():
            continue
        ck = _find_latest_ckpt(edir)
        if ck is None:
            continue
        try:
            d = torch.load(ck, map_location="cpu", weights_only=False)
        except Exception as e:
            rows.append(dict(name=edir.name, ckpt=ck.name, error=str(e)[:80]))
            continue
        acc = None
        for k in ["test_accuracies_full_list", "test_accuracies", "val_accuracies", "accuracy"]:
            v = d.get(k) if isinstance(d, dict) else None
            if isinstance(v, (list, tuple)) and v:
                nums = [x for x in v if isinstance(x, (int, float))]
                if nums:
                    acc = nums[-1]
                    break
        task = edir.name.split("_")[0]
        if task == "cifar10":
            task = "cifar10"
        rows.append(dict(
            name=edir.name, task=task, ckpt=ck.name,
            best_acc=acc, baseline=BASELINE_ACC.get(task),
            delta=(acc - BASELINE_ACC[task]) if acc and task in BASELINE_ACC else None,
        ))
    return pd.DataFrame(rows)


def plot_delta_bars(df, title="Experiments vs baseline", savepath=None):
    """Bar chart: delta vs baseline per experiment, colored by task."""
    import matplotlib.pyplot as plt
    import numpy as np
    if df.empty or "delta" not in df:
        print("no data to plot")
        return
    d = df.dropna(subset=["delta"]).sort_values("delta", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4, len(d) * 0.35)))
    colors = [TASK_COLORS.get(t, "#888") for t in d["task"]]
    bars = ax.barh(range(len(d)), d["delta"] * 100, color=colors,
                   edgecolor="black", linewidth=0.4, alpha=0.85)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d["name"], fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    for i, (_, r) in enumerate(d.iterrows()):
        v = r["delta"] * 100
        ax.text(v + 0.3 if v >= 0 else v - 0.3, i,
                f"{v:+.1f}pp", va="center", fontsize=8,
                ha="left" if v >= 0 else "right")
    ax.set_xlabel("delta vs baseline (pp)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.2)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()
