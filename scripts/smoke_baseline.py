#!/usr/bin/env python3
"""
Smoke test for baseline CTM experiments.
Verifies all 5 tasks + key ideas run for 1-2 iterations without crashing.
Usage: python scripts/smoke_baseline.py [--iterations 2]
"""
import argparse
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
        reload=False, device=[0], log_dir="/tmp/smoke/sort",
    )),
    "parity": ("baseline.tasks.parity.train", dict(
        seed=0, iterations=2, memory_length=5,
        parity_sequence_length=8,
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
        device=[0], use_amp=False,
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
        reload=False, device=[0],
        n_test_batches=1,
        data_root="/home/jovyan/h800fast/wangzekai/minimind-o/dataset/",
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
        lr=1e-4, device=[0], seed=1,
        data_root="/home/jovyan/h800fast/wangzekai/minimind-o/dataset/",
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
        reload=False, device=[0], use_amp=False,
        neuron_select_type="random",
        data_root="/home/jovyan/h800fast/wangzekai/minimind-o/dataset/",
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
        else:
            parts.append(f"--{k}")
            parts.append(str(v))
    return parts


def run_test(name, cmd):
    print(f"\n{'='*60}")
    print(f"  SMOKE: {name}")
    print(f"{'='*60}")
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    elapsed = time.time() - start
    if result.returncode == 0:
        print(f"  ✅ PASS ({elapsed:.1f}s)")
        return True
    else:
        print(f"  ❌ FAIL (rc={result.returncode}, {elapsed:.1f}s)")
        for line in (result.stderr or "").split("\n")[-20:]:
            if line.strip():
                print(f"     {line}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=2)
    args = parser.parse_args()

    results = []

    # ── 1. Paper config smoke (1 iter per task) ──
    print("\n═══ TASK PAPER CONFIGS ═══")
    for task_name, (module, base_cfg) in TASKS.items():
        cfg = dict(base_cfg)
        cfg["iterations"] = args.iterations
        cfg["training_iterations"] = args.iterations
        results.append((f"{task_name}_paper", run_test(f"{task_name} paper", _cmd(module, cfg))))

    # ── 2. Idea configs (on sort, fastest) ──
    print("\n═══ IDEA SMOKES (sort) ═══")
    sort_mod, sort_base = TASKS["sort"]
    base = dict(sort_base)

    # JEPA
    cfg = dict(base)
    cfg.update(cross_tick_jepa_weight=1.0, cross_tick_jepa_hidden_dim=32,
               cross_tick_jepa_predictor_depth=1, cross_tick_jepa_dropout=0.0)
    results.append(("sort_jepa", run_test("sort + JEPA", _cmd(sort_mod, cfg))))

    # Tick halt
    cfg = dict(base)
    cfg.update(tick_halt_mode="threshold", tick_halt_threshold=0.6)
    results.append(("sort_halt", run_test("sort + halt", _cmd(sort_mod, cfg))))

    # Cell sparsity
    cfg = dict(base)
    cfg.update(topk_neurons=0.5)
    results.append(("sort_sparsity", run_test("sort + sparsity", _cmd(sort_mod, cfg))))

    # Reflex head
    cfg = dict(base)
    cfg.update(reflex_head=True, reflex_weight=0.2, reflex_ticks=1)
    results.append(("sort_reflex", run_test("sort + reflex", _cmd(sort_mod, cfg))))

    # Multi-tick loss
    cfg = dict(base)
    cfg.update(tick_loss_mode="mean")
    results.append(("sort_multitick", run_test("sort + multi-tick", _cmd(sort_mod, cfg))))

    # Draft-revise
    cfg = dict(base)
    cfg.update(draft_mode="revise", draft_block_size=1, draft_revise_weight=0.1, draft_corrupt_prob=0.15)
    results.append(("sort_draft", run_test("sort + draft-revise", _cmd(sort_mod, cfg))))

    # Async ticks
    cfg = dict(base)
    cfg.update(async_tick_mode="banded", async_tick_periods="1,2")
    results.append(("sort_async", run_test("sort + async ticks", _cmd(sort_mod, cfg))))

    # Differentiated memory
    cfg = dict(base)
    cfg.update(diff_memory=True, diff_memory_lengths="2,4")
    results.append(("sort_diffmem", run_test("sort + diff memory", _cmd(sort_mod, cfg))))

    # EMA spectrum
    cfg = dict(base)
    cfg.update(ema_speed_mode="ema_spectrum", ema_speed_decays="0.9,0.99", ema_distill_weight=0.02, ema_warmup_steps=1)
    results.append(("sort_ema", run_test("sort + EMA spectrum", _cmd(sort_mod, cfg))))

    # ── 3. Combo: JEPA + halt ──
    cfg = dict(base)
    cfg.update(cross_tick_jepa_weight=1.0, cross_tick_jepa_hidden_dim=32,
               cross_tick_jepa_predictor_depth=1, cross_tick_jepa_dropout=0.0,
               tick_halt_mode="threshold", tick_halt_threshold=0.6)
    results.append(("sort_jepa_halt", run_test("sort + JEPA+halt", _cmd(sort_mod, cfg))))

    # ── 4. Idea smoke on other tasks (one each) ──
    print("\n═══ CROSS-TASK IDEA SMOKES ═══")
    # Parity + JEPA
    p_mod, p_base = TASKS["parity"]
    cfg = dict(p_base)
    cfg.update(cross_tick_jepa_weight=1.0, cross_tick_jepa_hidden_dim=32,
               cross_tick_jepa_predictor_depth=1, cross_tick_jepa_dropout=0.0)
    results.append(("parity_jepa", run_test("parity + JEPA", _cmd(p_mod, cfg))))

    # Mazes + draft-revise
    m_mod, m_base = TASKS["mazes"]
    cfg = dict(m_base)
    cfg.update(draft_mode="revise", draft_block_size=1, draft_revise_weight=0.1)
    results.append(("mazes_draft", run_test("mazes + draft-revise", _cmd(m_mod, cfg))))

    # CIFAR10 + sparsity
    c_mod, c_base = TASKS["cifar10"]
    cfg = dict(c_base)
    cfg.update(topk_neurons=0.75)
    results.append(("cifar10_sparsity", run_test("cifar10 + sparsity", _cmd(c_mod, cfg))))

    # QAMNIST + halt
    q_mod, q_base = TASKS["qamnist"]
    cfg = dict(q_base)
    cfg.update(tick_halt_mode="threshold", tick_halt_threshold=0.6)
    results.append(("qamnist_halt", run_test("qamnist + halt", _cmd(q_mod, cfg))))

    # ── Summary ──
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"  SMOKE TEST RESULTS: {passed}/{total} passed")
    for name, ok in results:
        print(f"    {'✅' if ok else '❌'} {name}")
    print(f"{'='*60}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
