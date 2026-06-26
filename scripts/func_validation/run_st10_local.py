#!/usr/bin/env python3
"""Run st10 (draft-revise) experiments locally on N GPUs, round-robin."""

import subprocess, sys, time, argparse, os
from pathlib import Path

# Manually replicate build_st10_draft_revise from experiment_plan_ctm_paper.py
# to avoid importing the whole heavy plan file (which pulls in torch etc.).
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TASKS = {
    "sort":   ("baseline.tasks.sort.train", dict(
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
        reload=False,
    )),
    "parity": ("baseline.tasks.parity.train", dict(
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
        use_amp=False,
        neuron_select_type="random", n_test_batches=20,
    )),
    "mazes":  ("baseline.tasks.mazes.train", dict(
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
        reload=False,
        data_root="baseline/data/mazes", n_test_batches=20,
    )),
    "cifar10": ("baseline.tasks.image_classification.train", dict(
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
        lr=1e-4, seed=1, data_root="baseline/data/",
    )),
}

def build_st10_configs():
    configs = []
    for task_name in ("sort", "parity", "mazes", "cifar10"):
        module, base = TASKS[task_name]
        for revise_w in [0.1, 0.2]:
            for corrupt_p in [0.15, 0.30]:
                cfg = dict(base)
                cfg["seed"] = 0  # all st10 use seed 0
                cfg["draft_mode"] = "revise"
                cfg["draft_block_size"] = 2
                cfg["draft_revise_weight"] = revise_w
                cfg["draft_corrupt_prob"] = corrupt_p
                name = f"{task_name}_revise_w{str(revise_w).replace('.','p')}_cp{str(corrupt_p).replace('.','p')}"
                log_dir = f"logs/ctm_paper/st10/{name}"
                cfg["log_dir"] = log_dir
                configs.append((task_name, module, name, cfg))
    return configs


def main():
    ap = argparse.ArgumentParser(description="Run st10 draft-revise locally")
    ap.add_argument("--devices", type=str, default="0,1,2,3",
                    help="comma-separated GPU IDs")
    ap.add_argument("--pack", type=int, default=2,
                    help="concurrent jobs per GPU (watch memory!)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--training-iterations", type=int, default=None,
                    help="override training_iterations (e.g. 500 for quick test)")
    args = ap.parse_args()

    devices = [int(x) for x in args.devices.split(",") if x.strip()]
    configs = build_st10_configs()
    pack = args.pack
    print(f"Total experiments: {len(configs)}, GPUs: {devices}, pack={pack}")
    print()

    if args.training_iterations is not None:
        for _, _, _, cfg in configs:
            cfg["training_iterations"] = args.training_iterations

    def _launch(cfg, device):
        module = cfg.pop("module")
        exp_name = cfg.pop("name")
        _ = cfg.pop("task_name")
        cfg["device"] = device
        cmd = [sys.executable, "-m", module]
        for k, v in cfg.items():
            if v is None:
                continue
            if isinstance(v, bool):
                cmd.append(f"--{k}" if v else f"--no-{k}")
            elif isinstance(v, list):
                cmd.append(f"--{k}")
                cmd.extend(str(x) for x in v)
            else:
                cmd.append(f"--{k}")
                cmd.append(str(v))
        log_dir = str(ROOT / cfg.get("log_dir", f"logs/st10_{exp_name}"))
        os.makedirs(log_dir, exist_ok=True)
        log_file = f"{log_dir}/train.log"
        if args.dry_run:
            print(f"[DRY-RUN] GPU {device}: {' '.join(cmd)}")
            return None
        print(f"[{device}] Launching {exp_name}...  log: {log_file}")
        f = open(log_file, "w")
        return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=True,
                                cwd=str(ROOT))

    dev_count = {d: 0 for d in devices}
    # Assign each config a device round-robin, track by name
    name_to_info = {c[2]: c for c in configs}
    queue = list(name_to_info.keys())
    # device assignments (round-robin on original order)
    dev_of = {c[2]: devices[i % len(devices)] for i, c in enumerate(configs)}

    next_idx = 0
    running = {}  # name → (Popen, device)
    while next_idx < len(queue) or running:
        launched = False
        while next_idx < len(queue):
            name = queue[next_idx]
            dev = dev_of[name]
            if dev_count[dev] >= pack:
                break
            task_name, module, _, cfg = name_to_info[name]
            cfg["task_name"] = task_name
            cfg["module"] = module
            cfg["name"] = name
            p = _launch(dict(cfg), dev)
            if p:
                running[name] = (p, dev)
                dev_count[dev] += 1
            next_idx += 1
            launched = True
        if not running:
            break
        if not launched:
            time.sleep(10)
        done_names = []
        for name, (p, dev) in list(running.items()):
            ret = p.poll()
            if ret is not None:
                done_names.append((name, dev, p))
        for name, dev, p in done_names:
            running.pop(name)
            dev_count[dev] -= 1
            if dev_count[dev] < 0:
                dev_count[dev] = 0
            print(f"  done: {name} (rc={p.returncode})")
            sys.stdout.flush()

    print("\nAll experiments completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
