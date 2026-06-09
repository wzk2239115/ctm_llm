#!/usr/bin/env python3
import argparse

import experiment_plan_impl_validation as base


REGIONAL_STAGES = (
    "rg00",
    "rg01",
    "rg02",
    "rg03",
    "rg04",
    "rg05",
    "rg06",
    "rg07",
    "rg08",
    "rg09",
    "rg10",
    "rg11",
    "all",
)
REGIONAL_PREFIXES = tuple(f"{stage}_" for stage in REGIONAL_STAGES if stage != "all")

METRICS_PREFIX = "regional_moe"


def metrics_path(name):
    return f"runs/metrics/{METRICS_PREFIX}_{name}"


def build_plan(stage, plan_size="full"):
    plan = []

    if stage in ("rg00", "all"):
        base.add_sparse_experiment(
            plan,
            "rg00_dense_d512_tick2",
            "Dense CTM anchor for regional multi-pass routing comparisons.",
            512,
            topk=512,
            iterations=2,
            synapse_depth=2,
            memory_hidden_dims=2,
            max_steps=100,
        )
        base.add_moe_experiment(
            plan,
            "rg00_singlepass_top2_e16_s64",
            "Single-pass MoE routed-cell anchor before regional multi-pass routing.",
            num_experts=16,
            expert_size=64,
            topk_experts=2,
            routing="topk",
            moe_load_balance_weight=1e-2,
            iterations=2,
            synapse_depth=2,
            memory_hidden_dims=2,
            max_steps=100,
        )
        base.add_regional_experiment(
            plan,
            "rg00_regional_p2_shared1_top1_e16_s64",
            "Smoke-check two-pass regional activation with one shared expert and one routed region per pass.",
            activation_passes=2,
            shared_experts=1,
            topk_experts=1,
            iterations=2,
            synapse_depth=2,
            memory_hidden_dims=2,
            max_steps=100,
        )

    if stage in ("rg01", "all"):
        pass_grid = [(1, 1), (2, 1), (3, 1), (4, 1)]
        if base.include_plan_size(plan_size, "full"):
            pass_grid.extend([(5, 1), (6, 1), (2, 2), (3, 2), (4, 2), (5, 2)])
        if base.include_plan_size(plan_size, "wide"):
            pass_grid.extend([(6, 2), (3, 3), (4, 3)])
        for passes, topk_experts in pass_grid:
            base.add_regional_experiment(
                plan,
                f"rg01_p{passes}_shared1_top{topk_experts}_e16_s64",
                "Compare regional activation pass count at fixed expert granularity.",
                activation_passes=passes,
                shared_experts=1,
                topk_experts=topk_experts,
                iterations=2,
                synapse_depth=2,
                memory_hidden_dims=2,
            )

    if stage in ("rg02", "all"):
        shared_grid = [(0, 1), (1, 1), (2, 1), (1, 2)]
        if base.include_plan_size(plan_size, "full"):
            shared_grid.extend([(2, 2), (3, 1), (3, 2), (4, 1)])
        if base.include_plan_size(plan_size, "wide"):
            shared_grid.extend([(0, 2), (4, 2), (6, 1)])
        for shared_experts, topk_experts in shared_grid:
            base.add_regional_experiment(
                plan,
                f"rg02_p3_shared{shared_experts}_top{topk_experts}_e16_s64",
                "Test brain-like core/shared regions versus purely routed regional passes.",
                activation_passes=3,
                topk_experts=topk_experts,
                shared_experts=shared_experts,
                iterations=2,
                synapse_depth=2,
                memory_hidden_dims=2,
            )

    if stage in ("rg03", "all"):
        regularizers = [
            ("balance1e2_div0", 1e-2, 0.0),
            ("balance1e3_div0", 1e-3, 0.0),
            ("balance1e2_div1e3", 1e-2, 1e-3),
            ("balance1e2_div1e2", 1e-2, 1e-2),
        ]
        if base.include_plan_size(plan_size, "full"):
            regularizers.extend([
                ("balance0_div1e3", 0.0, 1e-3),
                ("balance1e2_div3e3", 1e-2, 3e-3),
                ("balance3e2_div1e3", 3e-2, 1e-3),
                ("balance3e3_div1e3", 3e-3, 1e-3),
            ])
        if base.include_plan_size(plan_size, "wide"):
            regularizers.extend([
                ("balance1e1_div1e3", 1e-1, 1e-3),
                ("balance1e2_div3e2", 1e-2, 3e-2),
            ])
        for tag, balance, diversity in regularizers:
            base.add_regional_experiment(
                plan,
                f"rg03_p3_shared1_top1_{tag}_e16_s64",
                "Compare load-balance and inter-pass diversity for regional routing.",
                activation_passes=3,
                shared_experts=1,
                topk_experts=1,
                balance=balance,
                diversity=diversity,
                iterations=2,
                synapse_depth=2,
                memory_hidden_dims=2,
            )

    if stage in ("rg04", "all"):
        ticks = [(1, 1), (1, 2), (2, 2), (3, 2), (4, 2)]
        if base.include_plan_size(plan_size, "full"):
            ticks.extend([(5, 2), (6, 2), (2, 3), (3, 3), (4, 3)])
        if base.include_plan_size(plan_size, "wide"):
            ticks.extend([(5, 3), (6, 3), (3, 4), (4, 4)])
        for passes, iterations in ticks:
            base.add_regional_experiment(
                plan,
                f"rg04_tick{iterations}_p{passes}_shared1_top1_e16_s64",
                "Cross CTM tick count with regional activation passes to separate thought depth from regional sweep depth.",
                activation_passes=passes,
                shared_experts=1,
                topk_experts=1,
                iterations=iterations,
                synapse_depth=2,
                memory_hidden_dims=2,
            )

    if stage in ("rg05", "all"):
        confirms = [
            ("singlepass_top2_balance", "topk", 1, 0, 2, 0.0),
            ("regional_p2_shared1_top1", "regional_shared_topk", 2, 1, 1, 0.0),
            ("regional_p3_shared1_top1", "regional_shared_topk", 3, 1, 1, 0.0),
            ("regional_p3_shared1_top1_div1e3", "regional_shared_topk", 3, 1, 1, 1e-3),
        ]
        for tag, routing, passes, shared, topk_experts, diversity in confirms:
            if routing.startswith("regional"):
                base.add_regional_experiment(
                    plan,
                    f"rg05_confirm_{tag}_e16_s64",
                    "Longer confirmation of the strongest regional routing candidates.",
                    activation_passes=passes,
                    shared_experts=shared,
                    topk_experts=topk_experts,
                    routing=routing,
                    diversity=diversity,
                    balance=1e-2,
                    iterations=2,
                    synapse_depth=2,
                    memory_hidden_dims=2,
                    max_steps=2000,
                )
            else:
                base.add_moe_experiment(
                    plan,
                    f"rg05_confirm_{tag}_e16_s64",
                    "Longer single-pass routed baseline for regional routing confirmation.",
                    num_experts=16,
                    expert_size=64,
                    topk_experts=topk_experts,
                    shared_experts=shared,
                    routing=routing,
                    moe_load_balance_weight=1e-2,
                    iterations=2,
                    synapse_depth=2,
                    memory_hidden_dims=2,
                    max_steps=2000,
                )

    if stage in ("rg06", "all"):
        grains = [
            (8, 64, 1, 1, 2),
            (8, 128, 1, 1, 3),
            (16, 32, 1, 1, 3),
            (16, 64, 1, 1, 3),
            (16, 64, 2, 1, 3),
            (32, 32, 1, 1, 3),
            (32, 32, 2, 1, 3),
            (32, 64, 1, 1, 3),
        ]
        if base.include_plan_size(plan_size, "wide"):
            grains.extend([
                (64, 16, 1, 1, 4),
                (64, 16, 2, 1, 4),
                (64, 32, 1, 1, 4),
            ])
        for num_experts, expert_size, topk_experts, shared_experts, passes in grains:
            base.add_regional_experiment(
                plan,
                f"rg06_grain_e{num_experts}_s{expert_size}_p{passes}_shared{shared_experts}_top{topk_experts}",
                "Sweep regional expert granularity and pass depth at comparable active regions.",
                num_experts=num_experts,
                expert_size=expert_size,
                activation_passes=passes,
                shared_experts=shared_experts,
                topk_experts=topk_experts,
                balance=1e-2,
                diversity=1e-3 if passes >= 3 else 0.0,
                iterations=2,
                synapse_depth=2,
                memory_hidden_dims=2,
            )

    if stage in ("rg07", "all"):
        bs = [
            (512, 16, 32, 3, 1, 1),
            (768, 24, 32, 3, 1, 1),
            (1024, 16, 64, 2, 1, 1),
            (1024, 16, 64, 3, 1, 1),
            (1024, 16, 64, 3, 1, 2),
            (1536, 24, 64, 3, 1, 1),
            (1536, 24, 64, 3, 2, 1),
        ]
        if base.include_plan_size(plan_size, "wide"):
            bs.extend([
                (2048, 32, 64, 3, 1, 1),
                (2048, 32, 64, 4, 1, 1),
            ])
        for d_model, num_experts, expert_size, passes, shared_experts, topk_experts in bs:
            base.add_regional_experiment(
                plan,
                f"rg07_base_d{d_model}_e{num_experts}_s{expert_size}_p{passes}_shared{shared_experts}_top{topk_experts}",
                "Move regional routing across d_model budgets to find the best quality/cost base.",
                num_experts=num_experts,
                expert_size=expert_size,
                activation_passes=passes,
                shared_experts=shared_experts,
                topk_experts=topk_experts,
                balance=1e-2,
                diversity=1e-3 if passes >= 3 else 0.0,
                iterations=2,
                synapse_depth=2,
                memory_hidden_dims=2,
            )

    if stage in ("rg08", "all"):
        schedules = [
            (0, 0.00),
            (500, 0.00),
            (1000, 0.00),
            (1000, 0.03),
            (1000, 0.05),
            (2000, 0.05),
        ]
        if base.include_plan_size(plan_size, "wide"):
            schedules.extend([(2000, 0.10), (4000, 0.05)])
        for warmup, dropout in schedules:
            base.add_regional_experiment(
                plan,
                f"rg08_sched_warm{warmup}_drop{str(dropout).replace('.', 'p')}_p3_shared1_top1",
                "Test whether warmup/dropout stabilizes regional routing emergence.",
                activation_passes=3,
                shared_experts=1,
                topk_experts=1,
                balance=1e-2,
                diversity=1e-3,
                moe_topk_warmup_steps=warmup,
                moe_expert_dropout=dropout,
                iterations=2,
                synapse_depth=2,
                memory_hidden_dims=2,
            )

    if stage in ("rg09", "all"):
        loss_modes = [
            ("minconf", "min_conf", "none", 0.0),
            ("mean", "mean", "none", 0.0),
            ("last", "last", "none", 0.0),
            ("confhalt_cw1e3", "min_conf", "confidence", 1e-3),
            ("confhalt_cw3e3", "min_conf", "confidence", 3e-3),
            ("threshold_cw1e3", "min_conf", "threshold", 1e-3),
        ]
        if base.include_plan_size(plan_size, "wide"):
            loss_modes.extend([
                ("mean_confhalt_cw1e3", "mean", "confidence", 1e-3),
                ("last_confhalt_cw1e3", "last", "confidence", 1e-3),
            ])
        for tag, tick_loss, halt, compute_weight in loss_modes:
            base.add_regional_experiment(
                plan,
                f"rg09_tickloss_{tag}_p3_shared1_top1",
                "Test whether regional passes pair better with different tick supervision and halt pressure.",
                activation_passes=3,
                shared_experts=1,
                topk_experts=1,
                balance=1e-2,
                diversity=1e-3,
                tick_loss_mode=tick_loss,
                tick_halt_mode=halt,
                tick_compute_weight=compute_weight,
                iterations=2,
                synapse_depth=2,
                memory_hidden_dims=2,
            )

    if stage in ("rg10", "all"):
        mtp = [
            ("none", "none", "", 0.0),
            ("elf_linear_h2", "linear", "1,2", 0.01),
            ("elf_linear_h4", "linear", "1,2,3,4", 0.03),
            ("mtp_1_2_4", "none", "1,2,4", 0.03),
            ("mtp_tickwise", "linear", "1,2,4,8", 0.03),
        ]
        if base.include_plan_size(plan_size, "wide"):
            mtp.extend([
                ("elf_pow2_h4", "pow2", "1,2,4", 0.03),
                ("mtp_dense_h4", "none", "1,2,3,4", 0.05),
            ])
        for tag, elf_mode, horizons, improve_weight in mtp:
            base.add_regional_experiment(
                plan,
                f"rg10_mtp_{tag}_p3_shared1_top1",
                "Combine regional routing with ELF/MTP labels to see whether local passes help multi-token prediction.",
                activation_passes=3,
                shared_experts=1,
                topk_experts=1,
                balance=1e-2,
                diversity=1e-3,
                moe_mtp_mode=tag,
                moe_mtp_horizons=horizons,
                elf_horizon_mode=elf_mode,
                elf_max_horizon=8 if "8" in horizons else 4,
                tick_improve_weight=improve_weight,
                iterations=2,
                synapse_depth=2,
                memory_hidden_dims=2,
            )

    if stage in ("rg11", "all"):
        confirms = [
            ("d512_p3_shared1_top1", 16, 32, 3, 1, 1, 1e-3),
            ("d512_p4_shared1_top1", 16, 32, 4, 1, 1, 1e-3),
            ("d1024_p2_shared1_top1", 16, 64, 2, 1, 1, 0.0),
            ("d1024_p3_shared1_top1", 16, 64, 3, 1, 1, 1e-3),
            ("d1024_p3_shared1_top2", 16, 64, 3, 1, 2, 1e-3),
            ("d1024_p4_shared1_top1", 16, 64, 4, 1, 1, 1e-3),
            ("d1536_p3_shared1_top1", 24, 64, 3, 1, 1, 1e-3),
        ]
        for tag, num_experts, expert_size, passes, shared_experts, topk_experts, diversity in confirms:
            base.add_regional_experiment(
                plan,
                f"rg11_confirm_{tag}",
                "Longer confirmation candidates for harvesting stable regional-routing curves.",
                num_experts=num_experts,
                expert_size=expert_size,
                activation_passes=passes,
                shared_experts=shared_experts,
                topk_experts=topk_experts,
                balance=1e-2,
                diversity=diversity,
                iterations=2,
                synapse_depth=2,
                memory_hidden_dims=2,
                max_steps=2000,
            )

    return base.validate_plan(plan)


base.configure_plan_defaults(
    metrics_prefix=METRICS_PREFIX,
    cluster_config="infra/clusters/h100_2nodes.env",
    dispatch_block_sparse=False,
    build_plan=build_plan,
    stages=REGIONAL_STAGES,
    prefixes=REGIONAL_PREFIXES,
)


def normalize_default_outputs(args):
    remap = {
        "runs/experiment_plans/impl_validation_plan.csv":
            "runs/experiment_plans/regional_moe_plan.csv",
        "runs/experiment_plans/impl_validation_batch_tune_plan.csv":
            "runs/experiment_plans/regional_moe_batch_tune_plan.csv",
        "runs/metrics/impl_validation_summary.csv":
            "runs/metrics/regional_moe_summary.csv",
        "runs/metrics/impl_validation_batch_profile.csv":
            "runs/metrics/regional_moe_batch_profile.csv",
        "runs/metrics/impl_validation_batch_profile_quick.csv":
            "runs/metrics/regional_moe_batch_profile_quick.csv",
        "runs/metrics/impl_validation_quick_probe_report.csv":
            "runs/metrics/regional_moe_quick_probe_report.csv",
        "runs/metrics/impl_validation_batch_probe_report.csv":
            "runs/metrics/regional_moe_batch_probe_report.csv",
    }
    for attr in ("output", "report_output", "batch_profile", "quick_output"):
        val = getattr(args, attr, None)
        if val and val in remap:
            setattr(args, attr, remap[val])


if __name__ == "__main__":
    args = base.parse_args()
    normalize_default_outputs(args)
    args.func(args)
