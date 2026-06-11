#!/usr/bin/env python3
"""
Parallel smoke test across all GPUs.
Usage: python scripts/smoke_baseline.py [--iterations 2]
"""
import argparse
import concurrent.futures
import subprocess
import sys
import time

ROOT = "/home/jovyan/h800fast/wangzekai/ctm_llm"

TASKS = {
    "sort": ("baseline.tasks.sort.train", dict(
        seed=412, iterations=2, memory_length=5,
        d_model=128, d_input=32, n_synch_out=16, n_synch_action=16,
        synapse_depth=2, heads=2, memory_hidden_dims=2, dropout=0.0,
        deep_memory=True, do_normalisation=False,
        positional_embedding_type="none",
        neuron_select_type="random-pairing",
        n_random_pairing_self=0, N_to_sort=10,
        batch_size=2, batch_size_test=2,
        lr=1e-3, training_iterations=2,
        warmup_steps=1, use_scheduler=False,
        weight_decay=0.0, gradient_clipping=-1,
        track_every=100, save_every=1000,
        reload=False, log_dir="/tmp/smoke/sort",
    )),
    "parity": ("baseline.tasks.parity.train", dict(
        seed=0, iterations=2, memory_length=5,
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
        lr=1e-4, training_iterations=2,
        warmup_steps=1, track_every=100,
        save_every=1000, reload=False,
        use_amp=False,
        neuron_select_type="random", n_test_batches=1,
        log_dir="/tmp/smoke/parity",
    )),
    "mazes": ("baseline.tasks.mazes.train", dict(
        model="ctm", neuron_select_type="first-last",
        dataset="mazes-small",
        synapse_depth=2, heads=2, iterations=2, memory_length=5,
        d_model=64, d_input=16,
        backbone_type="none",
        n_synch_out=8, n_synch_action=4, memory_hidden_dims=2,
        deep_memory=True, do_normalisation=False,
        weight_decay=0.0, dropout=0.0,
        positional_embedding_type="none",
        maze_route_length=10, cirriculum_lookahead=2, expand_range=False,
        batch_size=2, batch_size_test=2,
        lr=1e-4, training_iterations=2,
        warmup_steps=1,
        use_scheduler=False,
        gradient_clipping=-1,
        track_every=100, save_every=1000,
        reload=False,
        n_test_batches=1,
        data_root="baseline/data/mazes",
        log_dir="/tmp/smoke/mazes",
    )),
    "cifar10": ("baseline.tasks.image_classification.train", dict(
        model="ctm", dataset="cifar10",
        d_model=64, d_input=16, synapse_depth=2, heads=2,
        n_synch_out=16, n_synch_action=16,
        n_random_pairing_self=0, neuron_select_type="random-pairing",
        iterations=2, memory_length=5,
        deep_memory=True, memory_hidden_dims=4,
        dropout=0.0, dropout_nlm=0, do_normalisation=False,
        positional_embedding_type="none", backbone_type="resnet18-1",
        training_iterations=2, warmup_steps=1,
        use_scheduler=False,
        weight_decay=0.0,
        save_every=100, track_every=100, n_test_batches=1,
        batch_size=16, batch_size_test=16,
        lr=1e-4, seed=1,
        log_dir="/tmp/smoke/cifar10",
    )),
    "qamnist": ("baseline.tasks.qamnist.train", dict(
        seed=0, model_type="ctm", memory_length=5,
        q_num_images=2, q_num_images_delta=1, q_num_repeats_per_input=2,
        q_num_operations=2, q_num_operations_delta=1, q_num_answer_steps=2,
        d_model=64, d_input=16, n_synch_out=8, n_synch_action=8,
        synapse_depth=1, heads=2, memory_hidden_dims=4, dropout=0.0,
        deep_memory=True, do_normalisation=False, weight_decay=0.0,
        use_scheduler=False,
        batch_size=4, batch_size_test=4,
        lr=1e-4, training_iterations=2,
        warmup_steps=1, track_every=100, save_every=1000,
        reload=False, use_amp=False,
        neuron_select_type="random",
        data_root="baseline/data/",
        n_test_batches=1,
        log_dir="/tmp/smoke/qamnist",
    )),
}


def _cmd(module, cfg):
    parts = [sys.executable, "-m", module]
    for k, v in cfg.items():
        if v is None:
            continue
        if isinstance(v, bool):
            parts.append(f"--{k}" if v else f"--no-{k}")
        elif isinstance(v, (list, tuple)):
            parts.append(f"--{k}")
            parts.extend(str(x) for x in v)
        else:
            parts.append(f"--{k}")
            parts.append(str(v))
    return parts


def run_test(name, cmd, device):
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    elapsed = time.time() - start
    ok = result.returncode == 0
    tag = f"  {'✅' if ok else '❌'} {name} (GPU {device}, {elapsed:.1f}s)"
    if not ok:
        tag += "\n" + "\n".join("     " + l for l in (result.stderr or "").split("\n")[-8:] if l.strip())
    print(tag, flush=True)
    return name, ok


def build_specs(iterations):
    specs = []
    for task_name, (module, base_cfg) in TASKS.items():
        cfg = dict(base_cfg)
        cfg["iterations"] = iterations
        cfg["training_iterations"] = iterations
        specs.append((f"{task_name}_paper", module, cfg))

    sort_mod, sort_base = TASKS["sort"]
    base = dict(sort_base)
    base["iterations"] = iterations
    base["training_iterations"] = iterations

    specs.append(("sort_jepa", sort_mod, dict(base, cross_tick_jepa_weight=1.0, cross_tick_jepa_hidden_dim=32, cross_tick_jepa_predictor_depth=1, cross_tick_jepa_dropout=0.0)))
    specs.append(("sort_halt", sort_mod, dict(base, tick_halt_mode="threshold", tick_halt_threshold=0.6)))
    specs.append(("sort_sparsity", sort_mod, dict(base, topk_neurons=0.5)))
    specs.append(("sort_reflex", sort_mod, dict(base, reflex_head=True, reflex_weight=0.2, reflex_ticks=1)))
    specs.append(("sort_multitick", sort_mod, dict(base, tick_loss_mode="mean")))
    specs.append(("sort_draft", sort_mod, dict(base, draft_mode="revise", draft_block_size=1, draft_revise_weight=0.1, draft_corrupt_prob=0.15)))
    specs.append(("sort_async", sort_mod, dict(base, async_tick_mode="banded", async_tick_periods="1,2")))
    specs.append(("sort_diffmem", sort_mod, dict(base, diff_memory=True, diff_memory_lengths="2,4")))
    specs.append(("sort_ema", sort_mod, dict(base, ema_speed_mode="ema_spectrum", ema_speed_decays="0.9,0.99", ema_distill_weight=0.02, ema_warmup_steps=1)))
    specs.append(("sort_jepa_halt", sort_mod, dict(base, cross_tick_jepa_weight=1.0, cross_tick_jepa_hidden_dim=32, cross_tick_jepa_predictor_depth=1, cross_tick_jepa_dropout=0.0, tick_halt_mode="threshold", tick_halt_threshold=0.6)))

    p_mod, p_base = TASKS["parity"]
    p_cfg = dict(p_base)
    p_cfg["iterations"] = iterations
    p_cfg["training_iterations"] = iterations
    specs.append(("parity_jepa", p_mod, dict(p_cfg, cross_tick_jepa_weight=1.0, cross_tick_jepa_hidden_dim=32, cross_tick_jepa_predictor_depth=1, cross_tick_jepa_dropout=0.0)))

    m_mod, m_base = TASKS["mazes"]
    m_cfg = dict(m_base)
    m_cfg["iterations"] = iterations
    m_cfg["training_iterations"] = iterations
    specs.append(("mazes_draft", m_mod, dict(m_cfg, draft_mode="revise", draft_block_size=1, draft_revise_weight=0.1)))

    c_mod, c_base = TASKS["cifar10"]
    c_cfg = dict(c_base)
    c_cfg["iterations"] = iterations
    c_cfg["training_iterations"] = iterations
    specs.append(("cifar10_sparsity", c_mod, dict(c_cfg, topk_neurons=0.75)))

    q_mod, q_base = TASKS["qamnist"]
    q_cfg = dict(q_base)
    q_cfg["iterations"] = iterations
    q_cfg["training_iterations"] = iterations
    specs.append(("qamnist_halt", q_mod, dict(q_cfg, tick_halt_mode="threshold", tick_halt_threshold=0.6)))

    return specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--gpus", type=int, default=None, help="Number of GPUs (default: auto-detect)")
    args = parser.parse_args()

    import torch
    num_gpus = args.gpus or torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs found, defaulting to CPU")
        num_gpus = 1

    specs = build_specs(args.iterations)
    print(f"  Launching {len(specs)} tests across {num_gpus} GPUs...\n")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as pool:
        futures = {}
        for i, (name, module, cfg) in enumerate(specs):
            cfg = dict(cfg, device=[i % num_gpus])
            cmd = _cmd(module, cfg)
            futures[pool.submit(run_test, name, cmd, i % num_gpus)] = name

        for future in concurrent.futures.as_completed(futures):
            name, ok = future.result()
            results.append((name, ok))

    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"  SMOKE TEST RESULTS: {passed}/{total} passed")
    for name, ok in sorted(results, key=lambda x: x[0]):
        print(f"    {'✅' if ok else '❌'} {name}")
    print(f"{'='*60}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
