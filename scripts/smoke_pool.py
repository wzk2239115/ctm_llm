#!/usr/bin/env python3
"""
Pool-based smoke test: starts local server + workers, submits all smoke
experiments via pool, verifies completion.
Usage: python scripts/smoke_pool.py [--iterations 2]
"""
import argparse
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
POOL_SCRIPT = os.path.join(ROOT, "scripts", "cluster_pool.py")
TRAIN_ENTRY = "scripts/run_via_pool.py"

SMOKE_TASKS = [
    # (name, module, config_overrides)
    ("sort_paper", "baseline.tasks.sort.train", dict(
        seed=412, memory_length=5,
        d_model=128, d_input=32, n_synch_out=16, n_synch_action=16,
        synapse_depth=2, heads=2, memory_hidden_dims=2, dropout=0.0,
        deep_memory=True, do_normalisation=False,
        positional_embedding_type="none",
        neuron_select_type="random-pairing",
        n_random_pairing_self=0, N_to_sort=10,
        batch_size=2, batch_size_test=2,
        lr=1e-3,
        warmup_steps=1, use_scheduler=False,
        weight_decay=0.0, gradient_clipping=-1,
        track_every=100, save_every=1000,
        reload=False,
        log_dir=f"/tmp/smoke_pool/sort",
    )),
    ("parity_paper", "baseline.tasks.parity.train", dict(
        seed=0, memory_length=5,
        parity_sequence_length=9,
        d_model=64, d_input=32,
        n_synch_out=8, n_synch_action=8,
        synapse_depth=1, heads=2, memory_hidden_dims=4, dropout=0.0,
        deep_memory=True, do_normalisation=False,
        positional_embedding_type="custom-rotational-1d",
        backbone_type="parity_backbone",
        weight_decay=0.0, gradient_clipping=0.9,
        use_scheduler=False,
        batch_size=4, batch_size_test=4,
        lr=1e-4,
        warmup_steps=1, track_every=100,
        save_every=1000, reload=False,
        use_amp=False,
        neuron_select_type="random", n_test_batches=1,
        log_dir=f"/tmp/smoke_pool/parity",
    )),
    ("mazes_paper", "baseline.tasks.mazes.train", dict(
        model="ctm", neuron_select_type="first-last",
        dataset="mazes-small",
        synapse_depth=2, heads=2, memory_length=5,
        d_model=64, d_input=16,
        backbone_type="none",
        n_synch_out=8, n_synch_action=4, memory_hidden_dims=2,
        deep_memory=True, do_normalisation=False,
        weight_decay=0.0, dropout=0.0,
        positional_embedding_type="none",
        maze_route_length=10, cirriculum_lookahead=2, expand_range=False,
        batch_size=2, batch_size_test=2,
        lr=1e-4,
        warmup_steps=1, use_scheduler=False,
        gradient_clipping=-1,
        track_every=100, save_every=1000,
        reload=False, n_test_batches=1,
        data_root="baseline/data/mazes",
        log_dir=f"/tmp/smoke_pool/mazes",
    )),
    ("cifar10_paper", "baseline.tasks.image_classification.train", dict(
        model="ctm", dataset="cifar10",
        d_model=64, d_input=16, synapse_depth=2, heads=2,
        n_synch_out=16, n_synch_action=16,
        n_random_pairing_self=0, neuron_select_type="random-pairing",
        memory_length=5, deep_memory=True, memory_hidden_dims=4,
        dropout=0.0, dropout_nlm=0, do_normalisation=False,
        positional_embedding_type="none", backbone_type="resnet18-1",
        warmup_steps=1, use_scheduler=False,
        weight_decay=0.0,
        save_every=100, track_every=100, n_test_batches=1,
        batch_size=16, batch_size_test=16,
        lr=1e-4, seed=1,
        log_dir=f"/tmp/smoke_pool/cifar10",
    )),
    ("qamnist_paper", "baseline.tasks.qamnist.train", dict(
        seed=0, model_type="ctm", memory_length=5,
        q_num_images=2, q_num_images_delta=1, q_num_repeats_per_input=2,
        q_num_operations=2, q_num_operations_delta=1, q_num_answer_steps=2,
        d_model=64, d_input=16, n_synch_out=8, n_synch_action=8,
        synapse_depth=1, heads=2, memory_hidden_dims=4, dropout=0.0,
        deep_memory=True, do_normalisation=False, weight_decay=0.0,
        use_scheduler=False,
        batch_size=4, batch_size_test=4,
        lr=1e-4,
        warmup_steps=1, track_every=100, save_every=1000,
        reload=False, use_amp=False,
        neuron_select_type="random",
        data_root="baseline/data/",
        n_test_batches=1,
        log_dir=f"/tmp/smoke_pool/qamnist",
    )),
]


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def gpu_count():
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        return sum(1 for line in out.splitlines() if line.strip())
    except Exception:
        return 0


def make_task_config(name, module, overrides, tmpdir, iterations):
    """Build a cluster config env file for one smoke task."""
    log_dir = f"/tmp/smoke_pool/{name}"
    device = "[0]"  # CUDA_VISIBLE_DEVICES isolates per GPU
    args_list = []
    for k, v in overrides.items():
        if v is None:
            continue
        if isinstance(v, bool):
            args_list.append(f"--{k}" if v else f"--no-{k}")
        elif isinstance(v, (list, tuple)):
            args_list.append(f"--{k}")
            args_list.extend(str(x) for x in v)
        else:
            args_list.append(f"--{k}")
            args_list.append(str(v))
    args_list += ["--iterations", str(iterations), "--training_iterations", str(iterations)]

    config_path = os.path.join(tmpdir, f"{name}.env")
    with open(config_path, "w") as f:
        f.write(f"TRAIN_DIR={ROOT}\n")
        f.write(f"TRAIN_ENTRY={TRAIN_ENTRY}\n")
        f.write(f"NNODES=1\n")
        f.write(f"NPROC_PER_NODE=1\n")
        f.write(f"MASTER_ADDR=127.0.0.1\n")
        f.write(f"MASTER_PORT=29501\n")
        f.write(f"TRAIN_ENV=infra/envs/smoke_baseline.env\n")
        f.write(f"NUMEXPR_MAX_THREADS=1\n")
    return config_path, [module] + args_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--port", type=int, default=0, help="Pool server port (0=auto)")
    parser.add_argument("--workers", type=int, default=None, help="Worker count (default=GPU count)")
    args = parser.parse_args()

    ngpus = gpu_count()
    nworkers = args.workers or max(ngpus, 1)
    port = args.port or find_free_port()
    base_url = f"http://127.0.0.1:{port}"

    tmpdir = "/tmp/smoke_pool"
    os.makedirs(tmpdir, exist_ok=True)

    print(f"  GPUs: {ngpus}  Workers: {nworkers}  Port: {port}")
    print(f"  Tasks: {len(SMOKE_TASKS)}  Iterations: {args.iterations}")

    # 1. Start pool server
    print("\n  Starting pool server...")
    server_proc = subprocess.Popen(
        [sys.executable, POOL_SCRIPT, "server", "--host", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(1)  # wait for server to start

    # 2. Start workers
    print(f"  Starting {nworkers} workers...")
    workers = []
    for gpu in range(nworkers):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu % ngpus) if ngpus > 0 else ""
        proc = subprocess.Popen(
            [sys.executable, POOL_SCRIPT, "worker",
             "--config", "infra/envs/h100_baseline.env",
             "--master_addr", "127.0.0.1",
             "--port", str(port),
             "--node_addr", "127.0.0.1",
             "--gpu-slots", "1",
             "--interval", "2",
             "--no_auto_pull"],
            env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        workers.append(proc)

    time.sleep(2)  # wait for workers to register

    # 3. Submit all tasks
    print(f"\n  Submitting {len(SMOKE_TASKS)} tasks...")
    task_ids = {}
    for name, module, overrides in SMOKE_TASKS:
        cfg_path, extra_args = make_task_config(name, module, dict(overrides), tmpdir, args.iterations)
        payload = {
            "config": cfg_path,
            "extra_args": " ".join(shlex.quote(a) for a in extra_args),
            "node_addrs": ["127.0.0.1"],
            "env": {"CTM_EXPERIMENT_NAME": name},
        }
        req = urllib.request.Request(
            f"{base_url}/submit",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read())
            tid = result["task"]["task_id"]
            task_ids[tid] = name
            print(f"    {name:20s} → {tid}")
        except Exception as e:
            print(f"    {name:20s} → FAILED submit: {e}")

    # 4. Wait for completion
    print(f"\n  Waiting for {len(task_ids)} tasks to complete...")
    deadline = time.time() + 600  # 10 min max
    completed_statuses = {}
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/status", timeout=5) as resp:
                status = json.loads(resp.read())
        except Exception:
            time.sleep(3)
            continue

        for t in status.get("tasks", []):
            tid = t["task_id"]
            if tid in task_ids and tid not in completed_statuses:
                st = t.get("status")
                if st in ("completed", "failed", "cancelled"):
                    rc = t.get("return_code")
                    completed_statuses[tid] = (st, rc)
                    name = task_ids[tid]
                    tag = "✅" if st == "completed" and rc == 0 else "❌"
                    print(f"    {tag} {name:20s} {st} (rc={rc})")

        done = len(completed_statuses)
        total = len(task_ids)
        remaining = [task_ids[tid] for tid in task_ids if tid not in completed_statuses]
        bar = "[" + "#" * done + "-" * (total - done) + "]"
        print(f"\r  {bar} {done}/{total}  remaining: {remaining}", end="", flush=True)
        if done == total:
            print()
            break
        time.sleep(3)

    # 5. Stop workers + server
    print("\n  Cleaning up...")
    for w in workers:
        try:
            w.terminate()
            w.wait(timeout=5)
        except Exception:
            try:
                w.kill()
            except Exception:
                pass
    try:
        server_proc.terminate()
        server_proc.wait(timeout=5)
    except Exception:
        try:
            server_proc.kill()
        except Exception:
            pass

    # 6. Summary
    passed = sum(1 for (st, rc) in completed_statuses.values() if st == "completed" and rc == 0)
    failed = sum(1 for (st, rc) in completed_statuses.values() if st != "completed" or rc != 0)
    total = len(task_ids)
    print(f"\n{'='*60}")
    print(f"  SMOKE POOL RESULTS: {passed}/{total} passed ({failed} failed)")
    for tid, (st, rc) in sorted(completed_statuses.items(), key=lambda x: task_ids.get(x[0], x[0])):
        name = task_ids.get(tid, tid)
        tag = "✅" if st == "completed" and rc == 0 else "❌"
        print(f"    {tag} {name:20s} {st} (rc={rc})")
    print(f"{'='*60}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n  Interrupted")
        sys.exit(130)
