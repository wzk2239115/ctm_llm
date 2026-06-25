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

ROOT = Path(__file__).resolve().parent.parent


def _resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _total_gpu_gb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL)
        return int(out.splitlines()[0].strip()) / 1024
    except Exception:
        return 80.0


def _est_gb(exp):
    d_model = exp.config.get("d_model", 512) or 512
    gb = max(2.0, d_model * 0.01)
    jw = exp.config.get("cross_tick_jepa_weight", 0)
    if jw and float(jw) > 0:
        gb *= 1.3
    if exp.config.get("draft_mode") == "revise":
        gb *= 1.15
    return gb

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
    "cifar10": 0.6443, "mazes": 0.9117, "parity": 0.6797,
    "qamnist": 0.3662, "sort": 0.8753,
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


def _check_oom(log_path):
    try:
        content = log_path.read_text(errors="replace")
        return any(kw in content for kw in [
            "OutOfMemoryError", "out of memory", "CUBLAS_STATUS_ALLOC_FAILED",
        ])
    except Exception:
        return False


def _gpu_free_map():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL)
        return {int(l.split(",")[0].strip()): int(l.split(",")[1].strip()) / 1024
                for l in out.strip().splitlines() if "," in l}
    except Exception:
        return {}


def run_all(experiments, gpus=8, log_root="logs/deep", dry_run=False, mem_util=0.80,
            max_retries=2, poll_interval=3):
    """Run experiments in parallel on N GPUs, multiple per GPU based on VRAM.

    Automatically packs experiments onto GPUs by estimated memory usage.
    e.g. 80GB H100 + d_model=512 (~6GB/task) => ~10 parallel per GPU.
    """
    log_root = _resolve(log_root)
    log_root.mkdir(parents=True, exist_ok=True)
    total_gb = _total_gpu_gb()
    cap_gb = total_gb * mem_util

    if dry_run:
        max_gb = max(_est_gb(e) for e in experiments)
        min_gb = min(_est_gb(e) for e in experiments)
        per_gpu_max = max(1, int(cap_gb / max_gb))
        total_parallel = gpus * per_gpu_max
        print(f"GPUs: {gpus} x {total_gb:.0f}GB (cap {cap_gb:.0f}GB/GPU)")
        print(f"Task est: {min_gb:.1f}~{max_gb:.1f} GB  =>  ~{per_gpu_max} per GPU = {total_parallel} parallel")
        print(f"Total experiments: {len(experiments)}  =>  ~{len(experiments)/total_parallel:.1f} rounds")
        print()
        for i, exp in enumerate(experiments[:5]):
            est = _est_gb(exp)
            print(f"  [{exp.name}] est={est:.1f}GB")
        if len(experiments) > 5:
            print(f"  ... ({len(experiments)-5} more)")
        print(f"\n({len(experiments)} experiments, dry-run only)")
        return

    running = {}
    gpu_n = {g: 0 for g in range(gpus)}
    gpu_max_slots = {g: 999 for g in range(gpus)}
    retry_count = {}
    pending = list(experiments)
    done, failed = [], []
    t0 = time.time()
    rr = 0

    print(f"GPUs: {gpus} x {total_gb:.0f}GB (cap {cap_gb:.0f}GB/GPU, "
          f"round-robin + live nvidia-smi, OOM retry x{max_retries})")

    while pending or running:
        free_map = _gpu_free_map()
        while pending:
            exp = pending[0]
            est = _est_gb(exp)
            placed = False
            for offset in range(gpus):
                gpu = (rr + offset) % gpus
                if gpu_n[gpu] >= gpu_max_slots[gpu]:
                    continue
                free_gb = free_map.get(gpu, cap_gb)
                if free_gb < est * 1.3:
                    continue
                pending.pop(0)
                edir = log_root / exp.name
                edir.mkdir(parents=True, exist_ok=True)
                logfile = open(edir / "train.log", "w")
                cmd = build_cmd(exp, gpu, str(edir))
                proc = subprocess.Popen(cmd, shell=True, stdout=logfile,
                                         stderr=subprocess.STDOUT, cwd=str(ROOT))
                running[proc.pid] = (exp, proc, logfile, gpu, est, edir)
                gpu_n[gpu] += 1
                free_map[gpu] = max(0, free_gb - est)
                rr = (gpu + 1) % gpus
                print(f"[GPU {gpu}] START {exp.name}  ({len(running)} running, "
                      f"free={free_gb:.0f}GB est={est:.1f}GB n={gpu_n[gpu]}, "
                      f"elapsed {time.time()-t0:.0f}s)")
                placed = True
                break
            if not placed:
                break

        finished = False
        for pid in list(running):
            exp, proc, logfile, gpu, est, edir = running[pid]
            if proc.poll() is not None:
                logfile.close()
                rc = proc.returncode
                gpu_n[gpu] -= 1
                log_path = edir / "train.log"
                if rc == 0:
                    done.append(exp)
                    print(f"[GPU {gpu}] DONE  {exp.name}  (elapsed {time.time()-t0:.0f}s)")
                else:
                    is_oom = _check_oom(log_path)
                    retries = retry_count.get(exp.name, 0)
                    if is_oom and retries < max_retries:
                        retry_count[exp.name] = retries + 1
                        if gpu_n[gpu] + 1 > 1:
                            gpu_max_slots[gpu] = max(1, gpu_n[gpu])
                        pending.insert(0, exp)
                        print(f"[GPU {gpu}] OOM  {exp.name}  "
                              f"(retry {retries+1}/{max_retries}, "
                              f"GPU{gpu} max_slots -> {gpu_max_slots[gpu]})")
                    else:
                        failed.append(exp)
                        err_tail = ""
                        try:
                            lines = log_path.read_text(errors="replace").strip().split("\n")
                            err_tail = "\n".join(lines[-8:])
                        except Exception:
                            pass
                        kind = "OOM-final" if is_oom else "FAIL"
                        print(f"[GPU {gpu}] {kind}  {exp.name}  (rc={rc})")
                        if err_tail:
                            print(f"  {'─'*50}")
                            for line in err_tail.split("\n"):
                                print(f"  {line}")
                            print(f"  {'─'*50}")
                        print(f"  log: {log_path}")
                del running[pid]
                finished = True

        if not finished:
            time.sleep(poll_interval)

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
    log_root = _resolve(log_root)
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
    num = [c for c in ckpts if re.search(r"_(\d+)\.pt$", c.name)]
    if num:
        return max(num, key=lambda c: int(re.search(r"_(\d+)\.pt$", c.name).group(1)))
    return ckpts[-1]


def collect(log_root="logs/deep"):
    """Load best_test_acc from each experiment's latest checkpoint."""
    import torch
    import pandas as pd
    log_root = _resolve(log_root)
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
        for k in ["test_accuracies_full_list", "test_accuracies", "val_accuracies",
                  "accuracy", "test_accuracies_most_certain"]:
            v = d.get(k) if isinstance(d, dict) else None
            if isinstance(v, (list, tuple)) and v:
                nums = [x for x in v if isinstance(x, (int, float))]
                if nums:
                    acc = max(nums)
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


def collect_csv(csv_path):
    """Load results exported from the compute machine (no checkpoints needed).

    Reads a CSV with columns: name,task,best_acc,... and returns a DataFrame
    in the same schema as collect() (name,task,ckpt,best_acc,baseline,delta).
    """
    import pandas as pd
    p = _resolve(csv_path)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if df.empty or "best_acc" not in df:
        return pd.DataFrame()
    out = pd.DataFrame({
        "name": df["name"],
        "task": df["task"],
        "ckpt": df.get("metric", ""),
        "best_acc": pd.to_numeric(df["best_acc"], errors="coerce"),
    })
    out["baseline"] = out["task"].map(BASELINE_ACC)
    out["delta"] = out.apply(
        lambda r: (r["best_acc"] - r["baseline"])
        if pd.notna(r["best_acc"]) and pd.notna(r["baseline"]) else None, axis=1)
    return out


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
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()


# ═══════════════════════════════════════════════════
# Prior-art loading + advanced multi-figure visualization
# ═══════════════════════════════════════════════════

def load_prior(csv_path=None):
    """Load existing ctm_paper_summary.csv for prior-art comparison."""
    import pandas as pd
    if csv_path is None:
        for p in [Path("csv_data"), Path("../csv_data"), Path.cwd().parent / "csv_data"]:
            if (p / "ctm_paper_summary.csv").exists():
                csv_path = p / "ctm_paper_summary.csv"
                break
    if csv_path is None or not Path(csv_path).exists():
        return None
    df = pd.read_csv(csv_path)
    for c in ["final_iter", "best_test_acc", "final_test_acc", "best_test_acc_mc", "n_points"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    RANDOM = {"cifar10": 0.10, "mazes": 0.10, "parity": 0.55, "qamnist": 0.10, "sort": 0.05}
    def _ok(r):
        if pd.isna(r.get("final_iter")) or r.get("final_iter", 0) < 100:
            return False
        return r.get("best_test_acc", 0) >= RANDOM.get(r.get("task"), 0.10)
    df = df[df.apply(_ok, axis=1)].copy()
    return df


def load_prior_curves(curves_path=None):
    """Load existing ctm_paper_curves.json."""
    import json
    if curves_path is None:
        for p in [Path("csv_data"), Path("../csv_data"), Path.cwd().parent / "csv_data"]:
            if (p / "ctm_paper_curves.json").exists():
                curves_path = p / "ctm_paper_curves.json"
                break
    if curves_path is None or not Path(curves_path).exists():
        return None
    with open(curves_path) as f:
        return json.load(f)


def plot_prior_bar(df_prior, tasks, stage, sweep, title="Prior results", savepath=None):
    """Baseline vs idea bar chart from prior ctm_paper data (with seed errorbars)."""
    import matplotlib.pyplot as plt
    import numpy as np
    if df_prior is None:
        print("no prior data")
        return
    bl_vals, idea_vals, idea_errs = [], [], []
    for t in tasks:
        bl_vals.append(BASELINE_ACC.get(t, 0) * 100)
        sub = df_prior[(df_prior.task == t) & (df_prior.stage == stage) & (df_prior.sweep == sweep)]
        if not sub.empty:
            idea_vals.append(sub["best_test_acc"].mean() * 100)
            idea_errs.append(sub["best_test_acc"].std(ddof=1) * 100 if len(sub) > 1 else 0)
        else:
            idea_vals.append(0)
            idea_errs.append(0)
    x = np.arange(len(tasks))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(tasks) * 2), 5.5))
    ax.bar(x - w/2, bl_vals, w, label="paper baseline", color="#bbb", edgecolor="black", lw=0.5)
    ax.bar(x + w/2, idea_vals, w, yerr=idea_errs, capsize=4,
           label=stage, color="#2ca02c", edgecolor="black", lw=0.5, alpha=0.85)
    for i, (b, v) in enumerate(zip(bl_vals, idea_vals)):
        ax.text(i - w/2, b + 1, f"{b:.1f}", ha="center", fontsize=9)
        if v > 0:
            d = v - b
            ax.text(i + w/2, v + 1, f"{v:.1f}\\n({d:+.1f})", ha="center", fontsize=9,
                    fontweight="bold", color="#2ca02c" if d >= 0 else "#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=11)
    ax.set_ylabel("best test acc (%)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()


def plot_prior_curves(curves, task, specs, title=None, savepath=None):
    """Convergence curves. specs = [(stage, sweep, label, color), ...]"""
    import matplotlib.pyplot as plt
    import numpy as np
    if curves is None:
        print("no curves data")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for stage, sweep, label, color in specs:
        keys = [k for k in curves if k.startswith(f"{stage}/{task}_{sweep}")]
        if not keys:
            continue
        all_iters, all_accs = [], []
        for k in keys:
            c = curves[k]
            all_iters.append(c["iters"])
            all_accs.append([a * 100 for a in c["test_acc"]])
        min_len = min(len(x) for x in all_iters)
        iters = all_iters[0][:min_len]
        accs = np.array([a[:min_len] for a in all_accs]).mean(axis=0)
        ax.plot(iters, accs, linewidth=2, label=label, color=color)
    ax.set_xlabel("training iteration", fontsize=11)
    ax.set_ylabel("test acc (%)", fontsize=11)
    ax.set_title(title or f"{task} convergence", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()


def plot_box_seeds(df, group_col="task", value_col="best_acc", title="Seed variance", savepath=None):
    """Box plot showing seed-to-seed variance per group."""
    import matplotlib.pyplot as plt
    import numpy as np
    if df.empty or value_col not in df:
        print("no data")
        return
    dv = df.dropna(subset=[value_col]).copy()
    dv[value_col] = dv[value_col] * 100
    groups = sorted(dv[group_col].unique())
    data = [dv[dv[group_col] == g][value_col].values for g in groups]
    fig, ax = plt.subplots(figsize=(max(8, len(groups) * 1.8), 5.5))
    bp = ax.boxplot(data, labels=groups, patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="red", markersize=6))
    for patch, g in zip(bp["boxes"], groups):
        patch.set_facecolor(TASK_COLORS.get(g, "#888"))
        patch.set_alpha(0.6)
    ax.set_ylabel("best test acc (%)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()


def summary_stats(df, groupby=("task",), value_col="best_acc"):
    """Mean +/- std summary table with delta vs baseline."""
    import pandas as pd
    if df.empty or value_col not in df:
        return None
    dv = df.dropna(subset=[value_col]).copy()
    dv[value_col] = dv[value_col] * 100
    g = dv.groupby(list(groupby))[value_col]
    result = g.agg(["count", "mean", "std", "min", "max"]).round(2)
    deltas = []
    for idx in result.index:
        task = idx[0] if isinstance(idx, tuple) else idx
        bl = BASELINE_ACC.get(task)
        deltas.append(f"{result.loc[idx, 'mean'] - bl*100:+.2f}pp" if bl else "-")
    result["delta_vs_bl"] = deltas
    result.columns = ["seeds", "mean_acc", "std_acc", "min_acc", "max_acc", "delta_vs_bl"]
    return result


def plot_sweep_curve(df, x_col, task_col="task", value_col="best_acc", title="Sweep", savepath=None):
    """Sweep curve with errorbar per task."""
    import matplotlib.pyplot as plt
    import numpy as np
    if df.empty or value_col not in df:
        print("no data")
        return
    dv = df.dropna(subset=[value_col, x_col]).copy()
    dv[value_col] = dv[value_col] * 100
    tasks = sorted(dv[task_col].unique())
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for task in tasks:
        sub = dv[dv[task_col] == task]
        xs = sorted(sub[x_col].unique())
        means = [sub[sub[x_col] == x][value_col].mean() for x in xs]
        stds = [sub[sub[x_col] == x][value_col].std(ddof=1) if len(sub[sub[x_col] == x]) > 1 else 0 for x in xs]
        ax.errorbar(xs, means, yerr=stds, fmt="-o", color=TASK_COLORS.get(task, "#888"),
                    linewidth=2, markersize=8, capsize=5, label=task)
        bl = BASELINE_ACC.get(task)
        if bl:
            ax.axhline(bl * 100, color=TASK_COLORS.get(task, "#888"), ls=":", alpha=0.3, lw=1)
    ax.set_xlabel(x_col, fontsize=11)
    ax.set_ylabel("best test acc (%)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()


def plot_combo_synergy(df, singles_stages, combo_stage, task, title=None, savepath=None):
    """Compare combo vs best single vs baseline for a task."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    bl = BASELINE_ACC.get(task, 0) * 100
    labels, vals, colors = ["baseline"], [bl], ["#bbb"]
    best_single = 0
    for stage in singles_stages:
        sub = df[(df.task == task) & (df.stage == stage)]
        if not sub.empty:
            v = sub["best_acc"].mean() * 100
            labels.append(stage)
            vals.append(v)
            colors.append("#2ca02c")
            best_single = max(best_single, v)
    sub_c = df[(df.task == task) & (df.stage == combo_stage)]
    if not sub_c.empty:
        labels.append(combo_stage)
        vals.append(sub_c["best_acc"].mean() * 100)
        colors.append("#d62728")
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, vals, color=colors, edgecolor="black", lw=0.5, alpha=0.85)
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v:.1f}", ha="center", fontsize=10, fontweight="bold")
    ax.axhline(best_single, color="#2ca02c", ls="--", alpha=0.5, label=f"best single ({best_single:.1f}%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, fontsize=9)
    ax.set_ylabel("best test acc (%)", fontsize=11)
    ax.set_title(title or f"{task}: combo vs singles", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()


# ═════════════════════════════════════════════════════════════════
# Paper-grade: heatmaps, ablations, significance tests, sweep builders
# ═════════════════════════════════════════════════════════════════

def plot_sweep_heatmap(df, x_col, y_col="task", value_col="best_acc",
                        title="Hyperparameter sweep", savepath=None):
    import matplotlib.pyplot as plt
    import numpy as np
    if df.empty or value_col not in df:
        print("no data"); return
    dv = df.dropna(subset=[value_col, x_col]).copy()
    x_vals = sorted(dv[x_col].unique())
    y_vals = sorted(dv[y_col].unique())
    mat = np.full((len(y_vals), len(x_vals)), np.nan)
    for i, yv in enumerate(y_vals):
        bl = BASELINE_ACC.get(yv)
        for j, xv in enumerate(x_vals):
            sub = dv[(dv[y_col] == yv) & (dv[x_col] == xv)]
            if not sub.empty:
                v = sub[value_col].mean()
                mat[i, j] = (v - bl) * 100 if bl else v * 100
    fig, ax = plt.subplots(figsize=(max(7, len(x_vals) * 1.2), max(3, len(y_vals) * 0.8)))
    finite = mat[~np.isnan(mat)]
    vmax = max(abs(finite.min()), abs(finite.max()), 5) if len(finite) else 5
    im = ax.imshow(mat, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(x_vals))); ax.set_xticklabels([str(v) for v in x_vals], fontsize=10)
    ax.set_yticks(range(len(y_vals))); ax.set_yticklabels(y_vals, fontsize=11)
    for i in range(len(y_vals)):
        for j in range(len(x_vals)):
            if not np.isnan(mat[i, j]):
                c = "white" if abs(mat[i, j]) > vmax * 0.6 else "black"
                ax.text(j, i, f"{mat[i, j]:+.1f}", ha="center", va="center", fontsize=9, fontweight="bold", color=c)
            else:
                ax.text(j, i, "-", ha="center", fontsize=10, color="gray")
    ax.set_xlabel(x_col, fontsize=11)
    plt.colorbar(im, ax=ax, shrink=0.8, label="delta vs baseline (pp)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()


def significance_test(df_treat, df_control=None, value_col="best_acc", group_col="task"):
    import pandas as pd
    from scipy import stats as sp_stats
    rows = []
    use_const = df_control is None
    tasks = sorted(df_treat[group_col].unique()) if group_col in df_treat else []
    for task in tasks:
        treat = df_treat[df_treat[group_col] == task][value_col].dropna()
        if use_const:
            ctrl_mean = BASELINE_ACC.get(task)
            if ctrl_mean is None or len(treat) < 2:
                continue
            t_stat, p_val = sp_stats.ttest_1samp(treat, ctrl_mean)
        else:
            ctrl = df_control[df_control[group_col] == task][value_col].dropna()
            if len(treat) < 2 or len(ctrl) < 2:
                continue
            ctrl_mean = ctrl.mean()
            t_stat, p_val = sp_stats.ttest_ind(treat, ctrl)
        delta = treat.mean() - ctrl_mean
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        rows.append(dict(task=task, mean=f"{treat.mean()*100:.2f}%", std=f"{treat.std(ddof=1)*100:.2f}",
                         delta=f"{delta*100:+.2f}pp", p=f"{p_val:.4f}", sig=sig, n=len(treat)))
    return pd.DataFrame(rows)


def plot_ablation_bars(df, ablation_col="variant", value_col="best_acc",
                        task_col="task", title="Ablation study", savepath=None):
    import matplotlib.pyplot as plt
    import numpy as np
    if df.empty or value_col not in df:
        print("no data"); return
    variants = sorted(df[ablation_col].unique())
    tasks = sorted(df[task_col].unique())
    x = np.arange(len(tasks)); n = len(variants); w = 0.75 / max(n, 1)
    fig, ax = plt.subplots(figsize=(max(9, len(tasks) * 2), 5.5))
    cmap = plt.cm.Set2(np.linspace(0, 1, max(n, 1)))
    for i, var in enumerate(variants):
        means, stds = [], []
        for task in tasks:
            sub = df[(df[ablation_col] == var) & (df[task_col] == task)]
            means.append(sub[value_col].mean() * 100 if not sub.empty else 0)
            stds.append(sub[value_col].std(ddof=1) * 100 if len(sub) > 1 else 0)
        ax.bar(x + i*w - 0.375 + w/2, means, w, yerr=stds, capsize=3,
               label=var, color=cmap[i], edgecolor="black", lw=0.4, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(tasks, fontsize=11)
    ax.set_ylabel("best test acc (%)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()


def make_revise_sweep(tasks, seeds):
    """2D sweep: revise_weight x corrupt_prob."""
    exps = []
    for task in tasks:
        for w in [0.05, 0.1, 0.2, 0.3]:
            for cp in [0.05, 0.15, 0.30]:
                for s in seeds:
                    e = dict(seed=s, draft_mode="revise", draft_block_size=2,
                             draft_revise_weight=w, draft_corrupt_prob=cp)
                    ws, cs = str(w).replace(".", "p"), str(cp).replace(".", "p")
                    exps.append(_exp(task, f"swp_w{ws}_cp{cs}_s{s}", e, ["sweep"]))
    return exps


def make_jepa_sweep(tasks, seeds):
    exps = []
    for task in tasks:
        for w in [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]:
            for s in seeds:
                e = dict(seed=s, cross_tick_jepa_weight=w,
                         cross_tick_jepa_hidden_dim=128, cross_tick_jepa_predictor_depth=2,
                         cross_tick_jepa_dropout=0.0)
                exps.append(_exp(task, f"swp_w{str(w).replace('.','p')}_s{s}", e, ["sweep"]))
    return exps


def make_sparsity_sweep(tasks, seeds):
    exps = []
    for task in tasks:
        for r in [0.1, 0.25, 0.5, 0.75, 0.9]:
            for s in seeds:
                exps.append(_exp(task, f"swp_r{str(r).replace('.','p')}_s{s}",
                                 dict(seed=s, topk_neurons=r), ["sweep"]))
    return exps


def make_revise_ablation(tasks, seeds):
    configs = [
        ("full",           dict(draft_mode="revise", draft_block_size=2, draft_revise_weight=0.1, draft_corrupt_prob=0.15)),
        ("no_noise",       dict(draft_mode="revise", draft_block_size=2, draft_revise_weight=0.1, draft_corrupt_prob=0.0)),
        ("no_revise_loss", dict(draft_mode="revise", draft_block_size=2, draft_revise_weight=0.0, draft_corrupt_prob=0.15)),
        ("block1",         dict(draft_mode="revise", draft_block_size=1, draft_revise_weight=0.1, draft_corrupt_prob=0.15)),
        ("block3",         dict(draft_mode="revise", draft_block_size=3, draft_revise_weight=0.1, draft_corrupt_prob=0.15)),
    ]
    exps = []
    for task in tasks:
        for variant, extra in configs:
            for s in seeds:
                exps.append(_exp(task, f"abl_{variant}_s{s}", dict(extra, seed=s), ["ablation", variant]))
    return exps


def make_jepa_ablation(tasks, seeds):
    jb = dict(cross_tick_jepa_hidden_dim=128, cross_tick_jepa_predictor_depth=2,
              cross_tick_jepa_dropout=0.0, cross_tick_jepa_weight=0.1)
    configs = [
        ("full",        dict(jb)),
        ("loss_mse",    dict(jb, cross_tick_jepa_loss="mse")),
        ("no_stopgrad", dict(jb, cross_tick_jepa_target_stop_grad=False)),
        ("depth1",      dict(jb, cross_tick_jepa_predictor_depth=1)),
        ("depth4",      dict(jb, cross_tick_jepa_predictor_depth=4)),
        ("hid64",       dict(jb, cross_tick_jepa_hidden_dim=64)),
        ("hid256",      dict(jb, cross_tick_jepa_hidden_dim=256)),
    ]
    exps = []
    for task in tasks:
        for variant, extra in configs:
            for s in seeds:
                exps.append(_exp(task, f"abl_{variant}_s{s}", dict(extra, seed=s), ["ablation", variant]))
    return exps
