#!/usr/bin/env python3
import csv
import math
import os

import experiment_plan_impl_validation as base


MOE_STAGES = (
    "moe00",
    "moe01",
    "moe02",
    "moe03",
    "moe04",
    "moe05",
    "moe06",
    "moe07",
    "all",
)
MOE_PREFIXES = tuple(f"{stage}_" for stage in MOE_STAGES if stage != "all")

METRICS_PREFIX = "moe_sparsity"


def metrics_path(name):
    return f"runs/metrics/{METRICS_PREFIX}_{name}"


def build_plan(stage, plan_size="full"):
    plan = []

    if stage in ("moe00", "all"):
        base.add_sparse_experiment(
            plan,
            "moe00_dense_d512_tick2",
            "Dense CTM anchor for MoE-style sparsity comparisons.",
            512,
            topk=512,
            iterations=2,
            synapse_depth=2,
            memory_hidden_dims=2,
            max_steps=100,
        )
        base.add_moe_experiment(
            plan,
            "moe00_router_top2_e8_s64",
            "Smoke-check MoE-style routed cells using current dense-mask backend.",
            num_experts=8,
            expert_size=64,
            topk_experts=2,
            iterations=2,
            synapse_depth=2,
            memory_hidden_dims=2,
            max_steps=100,
        )

    if stage in ("moe01", "all"):
        configs = [
            ("top1", 1),
            ("top2", 2),
            ("top4", 4),
        ]
        if base.include_plan_size(plan_size, "full"):
            configs.extend([("expert_choice", 2), ("hash", 2)])
        for routing, topk_experts in configs:
            base.add_moe_experiment(
                plan,
                f"moe01_router_{routing}_e16_s64_k{topk_experts}",
                "Compare Switch/GShard/Mixtral-style router choices at fixed expert granularity.",
                num_experts=16,
                expert_size=64,
                topk_experts=topk_experts,
                routing=routing,
                iterations=2,
                synapse_depth=2,
                memory_hidden_dims=2,
            )

    if stage in ("moe02", "all"):
        shared_grid = [(0, 2), (1, 2), (2, 2)] if plan_size == "core" else [
            (0, 2),
            (1, 2),
            (2, 2),
            (2, 4),
        ]
        for shared_experts, topk_experts in shared_grid:
            base.add_moe_experiment(
                plan,
                f"moe02_shared{shared_experts}_routed{topk_experts}_e16_s64",
                "Test DeepSeekMoE-style shared experts plus routed experts.",
                num_experts=16,
                expert_size=64,
                topk_experts=topk_experts,
                shared_experts=shared_experts,
                routing="shared_topk" if shared_experts else "topk",
                iterations=2,
                synapse_depth=2,
                memory_hidden_dims=2,
            )

    if stage in ("moe03", "all"):
        grains = [(8, 128, 2), (16, 64, 2), (32, 32, 4)]
        if base.include_plan_size(plan_size, "wide"):
            grains.extend([(64, 16, 8), (32, 64, 4)])
        for num_experts, expert_size, topk_experts in grains:
            base.add_moe_experiment(
                plan,
                f"moe03_fine_e{num_experts}_s{expert_size}_k{topk_experts}",
                "Test fine-grained expert specialization while keeping active cells similar.",
                num_experts=num_experts,
                expert_size=expert_size,
                topk_experts=topk_experts,
                routing="topk",
                iterations=2,
                synapse_depth=2,
                memory_hidden_dims=2,
            )

    if stage in ("moe04", "all"):
        regularizers = [
            ("balance1e3", 1e-3, 0.0, 0.0, 0),
            ("balance1e2", 1e-2, 0.0, 0.0, 0),
            ("entropy1e3", 0.0, 1e-3, 0.0, 0),
            ("zloss1e3", 0.0, 0.0, 1e-3, 0),
        ]
        if base.include_plan_size(plan_size, "full"):
            regularizers.append(("auxfree_bias", 0.0, 0.0, 0.0, 1))
        for tag, balance, entropy, z_loss, aux_free in regularizers:
            base.add_moe_experiment(
                plan,
                f"moe04_router_{tag}_e16_s64_k2",
                "Compare load-balance, entropy, z-loss, and aux-loss-free routing stabilizers.",
                num_experts=16,
                expert_size=64,
                topk_experts=2,
                routing="topk",
                moe_load_balance_weight=balance,
                moe_router_entropy_weight=entropy,
                moe_router_z_loss_weight=z_loss,
                moe_aux_loss_free_bias=aux_free,
                iterations=2,
                synapse_depth=2,
                memory_hidden_dims=2,
            )

    if stage in ("moe05", "all"):
        dispatches = [
            ("dense_mask", 1.0, 0),
            ("dropless", 1.0, 0),
            ("capacity_drop", 1.25, 1),
        ]
        if base.include_plan_size(plan_size, "full"):
            dispatches.append(("block_sparse", 1.0, 0))
        for dispatch, capacity, drop_tokens in dispatches:
            base.add_moe_experiment(
                plan,
                f"moe05_dispatch_{dispatch}_cap{str(capacity).replace('.', 'p')}",
                "Compare MegaBlocks-style dropless/block-sparse dispatch labels against dense-mask baseline.",
                num_experts=16,
                expert_size=64,
                topk_experts=2,
                routing="topk",
                dispatch=dispatch,
                moe_capacity_factor=capacity,
                moe_drop_tokens=drop_tokens,
                iterations=2,
                synapse_depth=2,
                memory_hidden_dims=2,
            )

    if stage in ("moe06", "all"):
        warmups = [(0, 0.0), (500, 0.0), (1000, 0.0)]
        if base.include_plan_size(plan_size, "full"):
            warmups.extend([(1000, 0.05), (2000, 0.05)])
        for warmup, dropout in warmups:
            base.add_moe_experiment(
                plan,
                f"moe06_warmup{warmup}_drop{str(dropout).replace('.', 'p')}_e16_s64_k2",
                "Test ST-MoE-style router warmup and expert dropout for stability.",
                num_experts=16,
                expert_size=64,
                topk_experts=2,
                routing="topk_warmup" if warmup else "topk",
                moe_topk_warmup_steps=warmup,
                moe_expert_dropout=dropout,
                iterations=2,
                synapse_depth=2,
                memory_hidden_dims=2,
            )

    if stage in ("moe07", "all"):
        mtp = [
            ("none", "none", ""),
            ("elf_linear_h4", "linear", "1,2,3,4"),
            ("mtp_1_2_4", "none", "1,2,4"),
        ]
        if base.include_plan_size(plan_size, "full"):
            mtp.append(("mtp_tickwise", "linear", "1,2,4,8"))
        for tag, elf_mode, horizons in mtp:
            base.add_moe_experiment(
                plan,
                f"moe07_{tag}_shared1_routed2_e16_s64",
                "Combine sparse routed cells with ELF/MTP-style multi-token prediction labels.",
                num_experts=16,
                expert_size=64,
                topk_experts=2,
                shared_experts=1,
                routing="shared_topk",
                moe_mtp_mode=tag,
                moe_mtp_horizons=horizons,
                elf_horizon_mode=elf_mode,
                elf_max_horizon=4 if "8" not in horizons else 8,
                tick_improve_weight=0.03 if tag != "none" else 0.0,
                iterations=2,
                synapse_depth=2,
                memory_hidden_dims=2,
            )

    return base.validate_plan(plan)


def summarize(args):
    rows = [
        row for row in base.latest_rows(args.metrics_dir)
        if base.is_final_metrics_row(row)
    ]
    for row in rows:
        loss = base.parse_float(row, "loss")
        peak_memory_mb = base.parse_float(row, "peak_memory_mb")
        tokens_per_sec = base.parse_float(row, "tokens_per_sec")
        peak_memory_gb = peak_memory_mb / 1024 if not math.isnan(peak_memory_mb) else math.nan
        row["loss_per_gb"] = (
            loss / peak_memory_gb
            if not math.isnan(loss) and not math.isnan(peak_memory_gb) and peak_memory_gb > 0
            else ""
        )
        row["tokens_per_gb"] = (
            tokens_per_sec / peak_memory_gb
            if not math.isnan(tokens_per_sec) and not math.isnan(peak_memory_gb) and peak_memory_gb > 0
            else ""
        )
        row["quality_cost_score"] = (
            loss * peak_memory_gb / tokens_per_sec
            if (
                not math.isnan(loss)
                and not math.isnan(peak_memory_gb)
                and not math.isnan(tokens_per_sec)
                and tokens_per_sec > 0
            )
            else ""
        )
    rows.sort(key=lambda r: r.get("experiment_name", ""))
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fields = [
        "experiment_name", "model_type", "loss", "tokens_per_sec",
        "peak_memory_mb", "best_tick", "conf_tick", "tick_count",
        "effective_tick", "active_cell_fraction",
        "moe_aux_loss", "loss_per_gb", "tokens_per_gb", "quality_cost_score",
        "losses_per_tick", "certainties_per_tick",
        "hidden_size", "num_hidden_layers", "d_model", "d_input",
        "iterations", "memory_length", "memory_hidden_dims", "deep_nlms",
        "synapse_depth", "tick_loss_mode", "elf_horizon_mode",
        "elf_max_horizon", "tick_improve_weight", "tick_improve_margin",
        "tick_halt_mode", "tick_halt_threshold", "tick_halt_temperature",
        "tick_compute_weight", "cell_sparsity_mode", "cell_topk",
        "cell_sparsity_rescale",
        "moe_routing_mode", "moe_num_experts", "moe_topk_experts",
        "moe_shared_experts", "moe_expert_size", "moe_load_balance_weight",
        "moe_router_entropy_weight", "moe_router_z_loss_weight",
        "moe_capacity_factor", "moe_drop_tokens", "moe_dispatch_mode",
        "moe_topk_warmup_steps", "moe_aux_loss_free_bias",
        "moe_expert_dropout", "moe_mtp_mode", "moe_mtp_horizons",
        "self_cond", "cross_layer_state", "global_step",
        "metrics_file",
    ]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote summary: {args.output}")


base.configure_plan_defaults(
    metrics_prefix=METRICS_PREFIX,
    cluster_config="infra/clusters/h100_2nodes.env",
    dispatch_block_sparse=False,
    build_plan=build_plan,
    summarize_fn=summarize,
    stages=MOE_STAGES,
    prefixes=MOE_PREFIXES,
)


def normalize_default_outputs(args):
    remap = {
        "runs/experiment_plans/impl_validation_plan.csv":
            "runs/experiment_plans/moe_sparsity_plan.csv",
        "runs/experiment_plans/impl_validation_batch_tune_plan.csv":
            "runs/experiment_plans/moe_sparsity_batch_tune_plan.csv",
        "runs/metrics/impl_validation_summary.csv":
            "runs/metrics/moe_sparsity_summary.csv",
        "runs/metrics/impl_validation_batch_profile.csv":
            "runs/metrics/moe_sparsity_batch_profile.csv",
        "runs/metrics/impl_validation_batch_profile_quick.csv":
            "runs/metrics/moe_sparsity_batch_profile_quick.csv",
        "runs/metrics/impl_validation_quick_probe_report.csv":
            "runs/metrics/moe_sparsity_quick_probe_report.csv",
        "runs/metrics/impl_validation_batch_probe_report.csv":
            "runs/metrics/moe_sparsity_batch_probe_report.csv",
    }
    for attr in ("output", "report_output", "batch_profile", "quick_output"):
        val = getattr(args, attr, None)
        if val and val in remap:
            setattr(args, attr, remap[val])


if __name__ == "__main__":
    args = base.parse_args()
    normalize_default_outputs(args)
    args.func(args)
