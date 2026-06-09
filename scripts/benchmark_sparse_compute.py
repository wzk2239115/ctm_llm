#!/usr/bin/env python3
"""Wall-clock benchmark for CTM sparse-compute strategies.

Measures forward+backward step time, peak GPU memory, executed ticks, and
theoretical active-cell fraction.  Use this to verify whether a sparsity
knob actually reduces GPU cost, not just FLOPs on paper.

Example:
  python scripts/benchmark_sparse_compute.py --variant all --device cuda
  python scripts/benchmark_sparse_compute.py --variant regional_block_sparse --steps 50
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.config import CTMLLMConfig
from model.model_ctm_llm import CTMForCausalLM


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass
class Variant:
    name: str
    category: str
    description: str
    saves_compute: str
    overrides: dict[str, Any] = field(default_factory=dict)


BASE_CFG = dict(
    vocab_size=6400,
    hidden_size=768,
    num_hidden_layers=4,
    d_model=512,
    d_input=256,
    heads=8,
    n_synch_out=512,
    n_synch_action=512,
    iterations=4,
    memory_length=8,
    memory_hidden_dims=2,
    deep_nlms=True,
    synapse_depth=2,
    self_cond=True,
    cross_layer_state=False,
    dropout=0.0,
    cell_sparsity_mode="none",
    cell_topk=512,
    cell_sparsity_rescale=True,
    moe_routing_mode="none",
    moe_num_experts=1,
    moe_topk_experts=1,
    moe_shared_experts=0,
    moe_expert_size=0,
    moe_activation_passes=1,
    moe_dispatch_mode="dense_mask",
    tick_halt_mode="none",
    tick_halt_threshold=0.65,
    tick_compute_weight=0.0,
)


def active_cell_fraction(cfg: dict[str, Any]) -> float:
    d_model = int(cfg.get("d_model", 512))
    if cfg.get("cell_sparsity_mode", "none") == "none":
        frac = 1.0
    else:
        frac = min(max(int(cfg.get("cell_topk", d_model)), 1), d_model) / max(d_model, 1)
    routing = cfg.get("moe_routing_mode", "none")
    if routing in ("regional_topk", "regional_shared_topk"):
        num_experts = max(int(cfg.get("moe_num_experts", 1)), 1)
        active_experts = int(cfg.get("moe_topk_experts", 1)) + int(cfg.get("moe_shared_experts", 0))
        frac = min(max(active_experts, 1), num_experts) / num_experts
    return frac


def build_variants() -> list[Variant]:
    d512 = 512
    k64 = 64
    return [
        Variant(
            "dense_full",
            "baseline",
            "Dense CTM, all cells active every tick.",
            "no",
        ),
        Variant(
            "numerical_topk_k64",
            "numerical_mask",
            "Post-activation top-k mask; synapse still runs on all 512 cells.",
            "no",
            dict(cell_sparsity_mode="topk", cell_topk=k64),
        ),
        Variant(
            "numerical_topk_k128",
            "numerical_mask",
            "Post-activation top-k mask at 25% active fraction.",
            "no",
            dict(cell_sparsity_mode="topk", cell_topk=128),
        ),
        Variant(
            "tick_fixed_2",
            "tick_sparse",
            "Fixed 2 internal ticks — true depth reduction when batch is uniform.",
            "yes",
            dict(iterations=2),
        ),
        Variant(
            "tick_fixed_8",
            "tick_sparse",
            "Fixed 8 internal ticks — upper-bound compute for tick dimension.",
            "no",
            dict(iterations=8),
        ),
        Variant(
            "moe_dense_mask_top2",
            "moe_mask",
            "Single-pass MoE with dense_mask dispatch; all experts still execute.",
            "no",
            dict(
                cell_sparsity_mode="topk",
                cell_topk=128,
                moe_routing_mode="topk",
                moe_num_experts=16,
                moe_topk_experts=2,
                moe_expert_size=32,
                moe_load_balance_weight=1e-2,
            ),
        ),
        Variant(
            "regional_dense_mask",
            "regional_mask",
            "Regional routing with dense_mask — mask after full expert compute.",
            "no",
            dict(
                cell_sparsity_mode="topk",
                cell_topk=64,
                moe_routing_mode="regional_shared_topk",
                moe_num_experts=16,
                moe_topk_experts=1,
                moe_shared_experts=1,
                moe_expert_size=32,
                moe_activation_passes=4,
                moe_dispatch_mode="dense_mask",
                moe_load_balance_weight=1e-2,
                moe_region_diversity_weight=1e-3,
            ),
        ),
        Variant(
            "regional_block_sparse",
            "block_sparse",
            "Grouped sparse regional backend — only routed blocks execute.",
            "yes",
            dict(
                cell_sparsity_mode="topk",
                cell_topk=64,
                moe_routing_mode="regional_shared_topk",
                moe_num_experts=16,
                moe_topk_experts=1,
                moe_shared_experts=1,
                moe_expert_size=32,
                moe_activation_passes=4,
                moe_dispatch_mode="block_sparse",
                moe_load_balance_weight=1e-2,
                moe_region_diversity_weight=1e-3,
            ),
        ),
        Variant(
            "regional_block_sparse_p1",
            "block_sparse",
            "Block-sparse with a single activation pass (minimal routed width).",
            "yes",
            dict(
                cell_sparsity_mode="topk",
                cell_topk=64,
                moe_routing_mode="regional_shared_topk",
                moe_num_experts=16,
                moe_topk_experts=1,
                moe_shared_experts=1,
                moe_expert_size=32,
                moe_activation_passes=1,
                moe_dispatch_mode="block_sparse",
                moe_load_balance_weight=1e-2,
            ),
        ),
        Variant(
            "regional_block_sparse_d1024",
            "block_sparse",
            "Block-sparse regional at d1024 (16×64 experts).",
            "yes",
            dict(
                d_model=1024,
                n_synch_out=1024,
                n_synch_action=1024,
                cell_sparsity_mode="topk",
                cell_topk=128,
                moe_routing_mode="regional_shared_topk",
                moe_num_experts=16,
                moe_topk_experts=1,
                moe_shared_experts=1,
                moe_expert_size=64,
                moe_activation_passes=4,
                moe_dispatch_mode="block_sparse",
                moe_load_balance_weight=1e-2,
                moe_region_diversity_weight=1e-3,
            ),
        ),
        Variant(
            "tick_halt_threshold",
            "tick_sparse",
            "Adaptive early-exit via confidence threshold (inference-style halt).",
            "maybe",
            dict(
                cell_sparsity_mode="topk",
                cell_topk=64,
                moe_routing_mode="regional_shared_topk",
                moe_num_experts=16,
                moe_topk_experts=1,
                moe_shared_experts=1,
                moe_expert_size=32,
                moe_activation_passes=4,
                moe_dispatch_mode="block_sparse",
                moe_load_balance_weight=1e-2,
                tick_halt_mode="threshold",
                tick_halt_threshold=0.30,
                iterations=8,
            ),
        ),
        Variant(
            "dispatch_dropless",
            "dispatch",
            "Regional dropless dispatch — no token dropping, grouped execution.",
            "maybe",
            dict(
                cell_sparsity_mode="topk",
                cell_topk=64,
                moe_routing_mode="regional_shared_topk",
                moe_num_experts=16,
                moe_topk_experts=1,
                moe_shared_experts=1,
                moe_expert_size=32,
                moe_activation_passes=4,
                moe_dispatch_mode="dropless",
                moe_load_balance_weight=1e-2,
            ),
        ),
        Variant(
            "dispatch_capacity_drop",
            "dispatch",
            "Regional capacity_drop dispatch with capacity_factor=1.0.",
            "maybe",
            dict(
                cell_sparsity_mode="topk",
                cell_topk=64,
                moe_routing_mode="regional_shared_topk",
                moe_num_experts=16,
                moe_topk_experts=1,
                moe_shared_experts=1,
                moe_expert_size=32,
                moe_activation_passes=4,
                moe_dispatch_mode="capacity_drop",
                moe_capacity_factor=1.0,
                moe_drop_tokens=False,
                moe_load_balance_weight=1e-2,
            ),
        ),
        Variant(
            "block_granularity_32x16",
            "block_sparse",
            "32 groups × 16 cells — finer block routing granularity.",
            "yes",
            dict(
                cell_sparsity_mode="topk",
                cell_topk=64,
                moe_routing_mode="regional_shared_topk",
                moe_num_experts=32,
                moe_topk_experts=1,
                moe_shared_experts=1,
                moe_expert_size=16,
                moe_activation_passes=4,
                moe_dispatch_mode="block_sparse",
                moe_load_balance_weight=1e-2,
            ),
        ),
    ]


def make_config(variant: Variant) -> CTMLLMConfig:
    cfg = dict(BASE_CFG)
    cfg.update(variant.overrides)
    return CTMLLMConfig(**cfg)


def mean_executed_ticks(model: CTMForCausalLM) -> float:
    ticks = [getattr(layer, "last_executed_ticks", 0) for layer in model.model.layers]
    return sum(ticks) / max(len(ticks), 1)


@torch.no_grad()
def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_variant(
    variant: Variant,
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    seq_len: int,
    steps: int,
    warmup: int,
) -> dict[str, Any]:
    config = make_config(variant)
    model = CTMForCausalLM(config).to(device=device, dtype=dtype)
    model.train()

    input_ids = torch.randint(1, config.vocab_size, (batch_size, seq_len), device=device)
    labels = input_ids.clone()

    def one_step():
        model.zero_grad(set_to_none=True)
        loss, _, _ = model.forward_train(input_ids, labels)
        loss.backward()

    # Warmup
    for _ in range(warmup):
        one_step()
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    times = []
    for _ in range(steps):
        _sync(device)
        t0 = time.perf_counter()
        one_step()
        _sync(device)
        times.append(time.perf_counter() - t0)

    peak_mb = 0.0
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    cfg = dict(BASE_CFG)
    cfg.update(variant.overrides)
    step_ms = 1000.0 * (sum(times) / len(times))
    tokens = batch_size * seq_len
    return {
        "variant": variant.name,
        "category": variant.category,
        "saves_compute": variant.saves_compute,
        "description": variant.description,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "steps": steps,
        "step_ms": round(step_ms, 3),
        "tokens_per_sec": round(tokens / (step_ms / 1000.0), 1),
        "peak_memory_mb": round(peak_mb, 1),
        "executed_ticks": round(mean_executed_ticks(model), 3),
        "active_cell_fraction": round(active_cell_fraction(cfg), 4),
        "iterations_config": int(cfg.get("iterations", 4)),
        "dispatch_mode": cfg.get("moe_dispatch_mode", "dense_mask"),
        "tick_halt_mode": cfg.get("tick_halt_mode", "none"),
    }


def print_table(rows: list[dict[str, Any]], baseline_tps: float | None):
    headers = [
        "variant", "category", "saves", "tok/s", "step_ms",
        "peak_MB", "exec_ticks", "active_frac", "speedup",
    ]
    print("\n" + " | ".join(headers))
    print("-" * (len(headers) * 14))
    for row in rows:
        speedup = ""
        if baseline_tps and baseline_tps > 0:
            speedup = f"{row['tokens_per_sec'] / baseline_tps:.2f}x"
        print(
            f"{row['variant']:<28} | {row['category']:<14} | {row['saves_compute']:<5} | "
            f"{row['tokens_per_sec']:>7.0f} | {row['step_ms']:>7.1f} | "
            f"{row['peak_memory_mb']:>7.0f} | {row['executed_ticks']:>5.2f} | "
            f"{row['active_cell_fraction']:>5.3f} | {speedup:>6}"
        )


def write_csv(path: str, rows: list[dict[str, Any]], baseline_tps: float | None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    for row in rows:
        if baseline_tps and baseline_tps > 0:
            row["speedup_vs_dense"] = round(row["tokens_per_sec"] / baseline_tps, 4)
        else:
            row["speedup_vs_dense"] = ""
    fields = [
        "variant", "category", "saves_compute", "description",
        "batch_size", "seq_len", "steps", "step_ms", "tokens_per_sec",
        "peak_memory_mb", "executed_ticks", "active_cell_fraction",
        "iterations_config", "dispatch_mode", "tick_halt_mode", "speedup_vs_dense",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="CTM sparse-compute wall-clock benchmark")
    parser.add_argument(
        "--variant", default="all",
        help="Variant name or 'all' (default). Use --list to see names.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPE_MAP))
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", default="runs/metrics/sparse_compute_benchmark.csv")
    parser.add_argument("--list", action="store_true", help="List variant names and exit.")
    return parser.parse_args()


def main():
    args = parse_args()
    variants = {v.name: v for v in build_variants()}

    if args.list:
        for v in build_variants():
            print(f"{v.name:30s} [{v.category:14s}] saves={v.saves_compute:5s}  {v.description}")
        return

    if args.variant == "all":
        selected = build_variants()
    elif args.variant in variants:
        selected = [variants[args.variant]]
    else:
        raise SystemExit(f"Unknown variant {args.variant!r}. Use --list.")

    device = torch.device(args.device)
    dtype = DTYPE_MAP[args.dtype]
    if device.type != "cuda":
        print("warning: running on CPU; GPU wall-clock numbers will not be representative.")

    rows = []
    for variant in selected:
        print(f"benchmarking {variant.name} ...", flush=True)
        try:
            row = benchmark_variant(
                variant,
                device=device,
                dtype=dtype,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                steps=args.steps,
                warmup=args.warmup,
            )
            rows.append(row)
        except Exception as exc:
            print(f"  FAILED {variant.name}: {exc}", flush=True)
            rows.append({
                "variant": variant.name,
                "category": variant.category,
                "saves_compute": variant.saves_compute,
                "description": variant.description,
                "tokens_per_sec": math.nan,
                "error": str(exc),
            })

    baseline_tps = None
    for row in rows:
        if row.get("variant") == "dense_full" and not math.isnan(row.get("tokens_per_sec", math.nan)):
            baseline_tps = row["tokens_per_sec"]
            break

    print_table([r for r in rows if "step_ms" in r], baseline_tps)
    write_csv(args.output, [r for r in rows if "step_ms" in r], baseline_tps)


if __name__ == "__main__":
    main()
