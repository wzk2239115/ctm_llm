#!/usr/bin/env python3
import argparse
import csv
import glob
import math
import os
import shlex
import subprocess
import sys
import time
import urllib.request
import json


URL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

DEFAULT_CLUSTER_CONFIG = "infra/clusters/h100_4nodes.env"

BASE_ARGS = {
    "epochs": 1,
    "batch_size": 4,
    "accumulation_steps": 1,
    "num_hidden_layers": 16,
    "hidden_size": 768,
    "d_model": 512,
    "d_input": 256,
    "heads": 8,
    "n_synch_out": 512,
    "n_synch_action": 512,
    "iterations": 4,
    "memory_length": 10,
    "memory_hidden_dims": 4,
    "deep_nlms": 1,
    "synapse_depth": 3,
    "tick_loss_mode": "min_conf",
    "elf_horizon_mode": "none",
    "elf_max_horizon": 4,
    "tick_improve_weight": 0.0,
    "tick_improve_margin": 0.0,
    "tick_halt_mode": "none",
    "tick_halt_threshold": 0.65,
    "tick_halt_temperature": 0.25,
    "tick_compute_weight": 0.0,
    "cell_sparsity_mode": "none",
    "cell_topk": 512,
    "cell_sparsity_rescale": 1,
    "self_cond": 1,
    "cross_layer_state": 1,
    "max_seq_len": 512,
    "log_interval": 20,
    "save_interval": 1000,
    "max_steps": 1000,
    "dtype": "bfloat16",
}

PLAN_SIZES = ("core", "full", "wide")
REGIONAL_STAGES = (
    "iv00",
    "iv01",
    "iv02",
    "iv03",
    "iv04",
    "iv05",
    "iv06",
    "iv07",
    "iv08",
    "all",
)
REGIONAL_PREFIXES = tuple(f"{stage}_" for stage in REGIONAL_STAGES if stage != "all")


def merge_args(**overrides):
    data = dict(BASE_ARGS)
    data.update(overrides)
    return data


def experiment(name, question, args):
    args = dict(args)
    args["experiment_name"] = name
    args["swanlab_name"] = name
    args["save_weight"] = name
    return {"name": name, "question": question, "args": args}


def clone_experiment(exp, name, args_overrides, question=None):
    args = dict(exp["args"])
    args.update(args_overrides)
    args["experiment_name"] = name
    args["swanlab_name"] = name
    args["save_weight"] = name
    return {
        "name": name,
        "base_name": exp["name"],
        "question": question or exp["question"],
        "args": args,
    }


def include_plan_size(plan_size, min_size):
    return PLAN_SIZES.index(plan_size) >= PLAN_SIZES.index(min_size)


def ctm_name(prefix, **items):
    parts = [prefix]
    for key, value in items.items():
        parts.append(f"{key}{str(value).replace('.', 'p')}")
    return "_".join(parts)


def validate_plan(plan):
    for exp in plan:
        args = exp["args"]
        name = exp["name"]
        heads = int(args.get("heads", 1))
        hidden = int(args.get("hidden_size", 1))
        d_input = int(args.get("d_input", 1))
        d_model = int(args.get("d_model", 1))
        n_synch_out = int(args.get("n_synch_out", 1))
        n_synch_action = int(args.get("n_synch_action", 1))
        if args.get("model_type") == "transformer" and hidden % heads != 0:
            raise ValueError(
                f"{name}: hidden_size({hidden}) must be divisible by heads({heads})")
        if args.get("model_type") == "ctm":
            if d_input % heads != 0:
                raise ValueError(
                    f"{name}: d_input({d_input}) must be divisible by heads({heads})")
            if d_model < max(n_synch_out, n_synch_action):
                raise ValueError(
                    f"{name}: d_model({d_model}) must be >= n_synch_out({n_synch_out}) "
                    f"and n_synch_action({n_synch_action})")
    return plan


def sparse_args(
    d_model,
    *,
    topk=None,
    iterations=4,
    memory_hidden_dims=2,
    memory_length=8,
    synapse_depth=2,
    max_steps=1000,
    cell_sparsity_rescale=1,
    moe_routing_mode="none",
    moe_num_experts=1,
    moe_topk_experts=1,
    moe_shared_experts=0,
    moe_expert_size=0,
    moe_load_balance_weight=0.0,
    moe_router_entropy_weight=0.0,
    moe_router_z_loss_weight=0.0,
    moe_capacity_factor=1.0,
    moe_drop_tokens=0,
    moe_dispatch_mode="dense_mask",
    moe_topk_warmup_steps=0,
    moe_aux_loss_free_bias=0,
    moe_expert_dropout=0.0,
    moe_activation_passes=1,
    moe_region_diversity_weight=0.0,
    moe_mtp_mode="none",
    moe_mtp_horizons="",
    **overrides,
):
    sparsity = "topk" if topk is not None and topk < d_model else "none"
    cell_topk = topk if topk is not None else d_model
    return merge_args(
        model_type="ctm",
        d_model=d_model,
        n_synch_out=d_model,
        n_synch_action=d_model,
        iterations=iterations,
        memory_hidden_dims=memory_hidden_dims,
        memory_length=memory_length,
        synapse_depth=synapse_depth,
        cell_sparsity_mode=sparsity,
        cell_topk=cell_topk,
        cell_sparsity_rescale=cell_sparsity_rescale,
        moe_routing_mode=moe_routing_mode,
        moe_num_experts=moe_num_experts,
        moe_topk_experts=moe_topk_experts,
        moe_shared_experts=moe_shared_experts,
        moe_expert_size=moe_expert_size,
        moe_load_balance_weight=moe_load_balance_weight,
        moe_router_entropy_weight=moe_router_entropy_weight,
        moe_router_z_loss_weight=moe_router_z_loss_weight,
        moe_capacity_factor=moe_capacity_factor,
        moe_drop_tokens=moe_drop_tokens,
        moe_dispatch_mode=moe_dispatch_mode,
        moe_topk_warmup_steps=moe_topk_warmup_steps,
        moe_aux_loss_free_bias=moe_aux_loss_free_bias,
        moe_expert_dropout=moe_expert_dropout,
        moe_activation_passes=moe_activation_passes,
        moe_region_diversity_weight=moe_region_diversity_weight,
        moe_mtp_mode=moe_mtp_mode,
        moe_mtp_horizons=moe_mtp_horizons,
        max_steps=max_steps,
        **overrides,
    )


def add_sparse_experiment(plan, name, question, d_model, **kwargs):
    plan.append(experiment(name, question, sparse_args(d_model, **kwargs)))


def active_topk(expert_size, topk_experts, shared_experts=0):
    return expert_size * (topk_experts + shared_experts)


def add_moe_experiment(
    plan,
    name,
    question,
    *,
    num_experts,
    expert_size,
    topk_experts,
    shared_experts=0,
    routing="topk",
    dispatch="dense_mask",
    max_steps=1000,
    **kwargs,
):
    d_model = num_experts * expert_size
    topk = active_topk(expert_size, topk_experts, shared_experts)
    add_sparse_experiment(
        plan,
        name,
        question,
        d_model,
        topk=topk,
        moe_routing_mode=routing,
        moe_num_experts=num_experts,
        moe_topk_experts=topk_experts,
        moe_shared_experts=shared_experts,
        moe_expert_size=expert_size,
        moe_dispatch_mode=dispatch,
        max_steps=max_steps,
        **kwargs,
    )


def add_regional_experiment(
    plan,
    name,
    question,
    *,
    num_experts=16,
    expert_size=64,
    topk_experts=1,
    shared_experts=1,
    activation_passes=2,
    routing=None,
    diversity=0.0,
    balance=1e-2,
    max_steps=1000,
    **kwargs,
):
    routing = routing or ("regional_shared_topk" if shared_experts else "regional_topk")
    kwargs.setdefault("moe_dispatch_mode", "block_sparse")
    d_model = num_experts * expert_size
    topk = active_topk(expert_size, topk_experts, shared_experts)
    add_sparse_experiment(
        plan,
        name,
        question,
        d_model,
        topk=topk,
        moe_routing_mode=routing,
        moe_num_experts=num_experts,
        moe_topk_experts=topk_experts,
        moe_shared_experts=shared_experts,
        moe_expert_size=expert_size,
        moe_activation_passes=activation_passes,
        moe_region_diversity_weight=diversity,
        moe_load_balance_weight=balance,
        max_steps=max_steps,
        **kwargs,
    )


def build_plan(stage, plan_size="full"):
    plan = []

    if stage in ("iv00", "all"):
        add_sparse_experiment(
            plan,
            "iv00_dense_d512_tick2",
            "Dense d512/tick2 anchor for implementation-validation runs.",
            512,
            topk=512,
            iterations=2,
            synapse_depth=2,
            memory_hidden_dims=2,
            max_steps=120,
        )
        add_moe_experiment(
            plan,
            "iv00_singlepass_top2_e16_s64",
            "Single-pass MoE mask anchor to separate modeling from grouped sparse regional execution.",
            num_experts=16,
            expert_size=64,
            topk_experts=2,
            routing="topk",
            moe_load_balance_weight=1e-2,
            iterations=2,
            synapse_depth=2,
            memory_hidden_dims=2,
            max_steps=120,
        )
        add_regional_experiment(
            plan,
            "iv00_regional_sparse_p3_shared1_top1_d1024",
            "Smoke-check new sequential grouped sparse regional backend.",
            num_experts=16,
            expert_size=64,
            activation_passes=3,
            shared_experts=1,
            topk_experts=1,
            balance=1e-2,
            diversity=1e-3,
            iterations=2,
            synapse_depth=2,
            memory_hidden_dims=2,
            max_steps=120,
        )
        add_regional_experiment(
            plan,
            "iv00_halt_threshold_p3_shared1_top1_d1024",
            "Smoke-check real tick early-exit on top of regional sparse backend.",
            num_experts=16,
            expert_size=64,
            activation_passes=3,
            shared_experts=1,
            topk_experts=1,
            balance=1e-2,
            diversity=1e-3,
            tick_halt_mode="threshold",
            tick_halt_threshold=0.0,
            tick_compute_weight=1e-3,
            iterations=4,
            synapse_depth=2,
            memory_hidden_dims=2,
            max_steps=120,
        )
        add_regional_experiment(
            plan,
            "iv00_mtp_1_2_4_p3_shared1_top1_d1024",
            "Smoke-check real MTP multi-horizon loss with regional sparse backend.",
            num_experts=16,
            expert_size=64,
            activation_passes=3,
            shared_experts=1,
            topk_experts=1,
            balance=1e-2,
            diversity=1e-3,
            moe_mtp_mode="mtp_1_2_4",
            moe_mtp_horizons="1,2,4",
            iterations=2,
            synapse_depth=2,
            memory_hidden_dims=2,
            max_steps=120,
        )

    if stage in ("iv01", "all"):
        controls = [
            ("dense_d512", "dense", 512, 1, 512, 512, 0, 0),
            ("dense_d1024", "dense", 1024, 1, 1024, 1024, 0, 0),
            ("topk_d512_k256", "topk", 512, 1, 512, 512, 0, 256),
            ("singlepass_d1024_e16_top2", "single", 1024, 16, 64, 1024, 0, 128),
            ("regional_d512_p4_shared1_top1", "regional", 512, 16, 32, 512, 4, 64),
            ("regional_d1024_p4_shared1_top1", "regional", 1024, 16, 64, 1024, 4, 128),
        ]
        if include_plan_size(plan_size, "full"):
            controls.extend([
                ("regional_d768_p4_shared1_top1", "regional", 768, 24, 32, 768, 4, 64),
                ("regional_d1536_p3_shared1_top1", "regional", 1536, 24, 64, 1536, 3, 128),
            ])
        for tag, kind, d_model, num_experts, expert_size, dense_topk, passes, topk in controls:
            if kind == "dense":
                add_sparse_experiment(
                    plan, f"iv01_backend_{tag}",
                    "Dense backend anchor for quality/cost comparison.",
                    d_model, topk=dense_topk, iterations=2,
                    synapse_depth=2, memory_hidden_dims=2)
            elif kind == "topk":
                add_sparse_experiment(
                    plan, f"iv01_backend_{tag}",
                    "Post-activation top-k mask anchor for quality/cost comparison.",
                    d_model, topk=topk, iterations=2,
                    synapse_depth=2, memory_hidden_dims=2)
            elif kind == "single":
                add_moe_experiment(
                    plan, f"iv01_backend_{tag}",
                    "Single-pass MoE mask anchor for grouped sparse regional comparisons.",
                    num_experts=num_experts, expert_size=expert_size,
                    topk_experts=2, routing="topk",
                    moe_load_balance_weight=1e-2,
                    iterations=2, synapse_depth=2, memory_hidden_dims=2)
            else:
                add_regional_experiment(
                    plan, f"iv01_backend_{tag}",
                    "Sequential grouped sparse regional backend quality/cost candidate.",
                    num_experts=num_experts, expert_size=expert_size,
                    activation_passes=passes, shared_experts=1, topk_experts=1,
                    balance=1e-2, diversity=1e-3,
                    iterations=2, synapse_depth=2, memory_hidden_dims=2)

    if stage in ("iv02", "all"):
        pass_grid = [(1, 1), (2, 1), (3, 1), (4, 1)]
        if include_plan_size(plan_size, "full"):
            pass_grid.extend([(5, 1), (6, 1), (2, 2), (3, 2), (4, 2), (5, 2)])
        if include_plan_size(plan_size, "wide"):
            pass_grid.extend([(6, 2), (3, 3), (4, 3)])
        for passes, topk_experts in pass_grid:
            add_regional_experiment(
                plan,
                f"iv02_pass_p{passes}_shared1_top{topk_experts}_d1024",
                "Validate sequential regional pass count and routed top-k after grouped sparse implementation.",
                num_experts=16,
                expert_size=64,
                activation_passes=passes,
                shared_experts=1,
                topk_experts=topk_experts,
                balance=1e-2,
                diversity=1e-3 if passes >= 3 else 0.0,
                iterations=2,
                synapse_depth=2,
                memory_hidden_dims=2,
            )

    if stage in ("iv03", "all"):
        bases = [
            (512, 16, 32, 3, 1, 1),
            (512, 16, 32, 4, 1, 1),
            (768, 24, 32, 4, 1, 1),
            (1024, 16, 64, 3, 1, 1),
            (1024, 16, 64, 4, 1, 1),
            (1024, 16, 64, 4, 1, 2),
            (1536, 24, 64, 3, 1, 1),
        ]
        if include_plan_size(plan_size, "wide"):
            bases.extend([
                (1536, 24, 64, 4, 1, 1),
                (2048, 32, 64, 3, 1, 1),
                (2048, 32, 64, 4, 1, 1),
            ])
        for d_model, num_experts, expert_size, passes, shared_experts, topk_experts in bases:
            add_regional_experiment(
                plan,
                f"iv03_base_d{d_model}_p{passes}_shared{shared_experts}_top{topk_experts}",
                "Find best d_model/pass/top-k quality-cost point for the new regional backend.",
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

    if stage in ("iv04", "all"):
        halt_grid = [
            ("none_tick2", "none", 0.65, 0.0, 2),
            ("threshold0p00_tick4", "threshold", 0.00, 1e-3, 4),
            ("threshold0p15_tick4", "threshold", 0.15, 1e-3, 4),
            ("threshold0p30_tick4", "threshold", 0.30, 1e-3, 4),
            ("threshold0p45_tick4", "threshold", 0.45, 1e-3, 4),
            ("threshold0p60_tick4", "threshold", 0.60, 1e-3, 4),
            ("threshold0p30_tick6", "threshold", 0.30, 2e-3, 6),
        ]
        if include_plan_size(plan_size, "full"):
            halt_grid.extend([
                ("threshold0p45_tick6", "threshold", 0.45, 2e-3, 6),
                ("confidence0p25_tick4", "confidence", 0.30, 1e-3, 4),
                ("confidence0p15_tick6", "confidence", 0.30, 2e-3, 6),
            ])
        for tag, halt_mode, threshold, compute_weight, iterations in halt_grid:
            add_regional_experiment(
                plan,
                f"iv04_halt_{tag}_d512_p4",
                "Validate real tick early-exit by measuring quality, effective tick, throughput, and memory.",
                num_experts=16,
                expert_size=32,
                activation_passes=4,
                shared_experts=1,
                topk_experts=1,
                balance=1e-2,
                diversity=1e-3,
                tick_halt_mode=halt_mode,
                tick_halt_threshold=threshold,
                tick_compute_weight=compute_weight,
                iterations=iterations,
                synapse_depth=2,
                memory_hidden_dims=2,
            )

    if stage in ("iv05", "all"):
        mtp_grid = [
            ("none", "none", "", "none", 4, 0.0),
            ("elf_linear_h2", "none", "", "linear", 2, 0.01),
            ("elf_linear_h4", "none", "", "linear", 4, 0.03),
            ("mtp_1_2", "mtp_1_2", "1,2", "none", 4, 0.0),
            ("mtp_1_2_4", "mtp_1_2_4", "1,2,4", "none", 4, 0.0),
            ("mtp_1_2_3_4", "mtp_1_2_3_4", "1,2,3,4", "none", 4, 0.0),
            ("mtp_1_2_4_improve", "mtp_1_2_4", "1,2,4", "none", 4, 0.03),
        ]
        if include_plan_size(plan_size, "full"):
            mtp_grid.extend([
                ("mtp_1_2_4_8", "mtp_1_2_4_8", "1,2,4,8", "none", 8, 0.0),
                ("elf_pow2_h4", "none", "", "pow2", 4, 0.03),
                ("mtp_plus_elf_linear", "mtp_1_2_4", "1,2,4", "linear", 4, 0.03),
            ])
        for tag, mtp_mode, horizons, elf_mode, elf_max, improve_weight in mtp_grid:
            add_regional_experiment(
                plan,
                f"iv05_mtp_{tag}_d512_p4",
                "Validate real MTP multi-horizon loss against ELF and no-MTP controls.",
                num_experts=16,
                expert_size=32,
                activation_passes=4,
                shared_experts=1,
                topk_experts=1,
                balance=1e-2,
                diversity=1e-3,
                moe_mtp_mode=mtp_mode,
                moe_mtp_horizons=horizons,
                elf_horizon_mode=elf_mode,
                elf_max_horizon=elf_max,
                tick_improve_weight=improve_weight,
                iterations=2,
                synapse_depth=2,
                memory_hidden_dims=2,
            )

    if stage in ("iv06", "all"):
        combos = [
            ("d512_p4_plain", 16, 32, 4, "none", 0.65, "none", "", "none"),
            ("d512_p4_halt0p30", 16, 32, 4, "threshold", 0.30, "none", "", "none"),
            ("d512_p4_mtp124", 16, 32, 4, "none", 0.65, "mtp_1_2_4", "1,2,4", "none"),
            ("d512_p4_halt0p30_mtp124", 16, 32, 4, "threshold", 0.30, "mtp_1_2_4", "1,2,4", "none"),
            ("d1024_p4_plain", 16, 64, 4, "none", 0.65, "none", "", "none"),
            ("d1024_p4_halt0p30", 16, 64, 4, "threshold", 0.30, "none", "", "none"),
            ("d1024_p4_mtp124", 16, 64, 4, "none", 0.65, "mtp_1_2_4", "1,2,4", "none"),
            ("d1024_p4_halt0p30_mtp124", 16, 64, 4, "threshold", 0.30, "mtp_1_2_4", "1,2,4", "none"),
        ]
        if include_plan_size(plan_size, "wide"):
            combos.extend([
                ("d768_p4_halt0p30_mtp124", 24, 32, 4, "threshold", 0.30, "mtp_1_2_4", "1,2,4", "none"),
                ("d1536_p3_halt0p30_mtp124", 24, 64, 3, "threshold", 0.30, "mtp_1_2_4", "1,2,4", "none"),
            ])
        for tag, num_experts, expert_size, passes, halt_mode, threshold, mtp_mode, horizons, elf_mode in combos:
            add_regional_experiment(
                plan,
                f"iv06_combo_{tag}",
                "Test whether grouped sparse regional execution, real halt, and MTP compose constructively.",
                num_experts=num_experts,
                expert_size=expert_size,
                activation_passes=passes,
                shared_experts=1,
                topk_experts=1,
                balance=1e-2,
                diversity=1e-3,
                tick_halt_mode=halt_mode,
                tick_halt_threshold=threshold,
                tick_compute_weight=1e-3 if halt_mode != "none" else 0.0,
                moe_mtp_mode=mtp_mode,
                moe_mtp_horizons=horizons,
                elf_horizon_mode=elf_mode,
                iterations=4 if halt_mode != "none" else 2,
                synapse_depth=2,
                memory_hidden_dims=2,
            )

    if stage in ("iv07", "all"):
        confirms = [
            ("d512_p4_plain", 16, 32, 4, "none", 0.65, "none", "", 3000),
            ("d512_p4_halt0p30", 16, 32, 4, "threshold", 0.30, "none", "", 3000),
            ("d512_p4_mtp124", 16, 32, 4, "none", 0.65, "mtp_1_2_4", "1,2,4", 3000),
            ("d512_p4_halt0p30_mtp124", 16, 32, 4, "threshold", 0.30, "mtp_1_2_4", "1,2,4", 3000),
            ("d1024_p4_plain", 16, 64, 4, "none", 0.65, "none", "", 3000),
            ("d1024_p4_mtp124", 16, 64, 4, "none", 0.65, "mtp_1_2_4", "1,2,4", 3000),
        ]
        if include_plan_size(plan_size, "full"):
            confirms.extend([
                ("d1024_p4_halt0p30", 16, 64, 4, "threshold", 0.30, "none", "", 3000),
                ("d768_p4_plain", 24, 32, 4, "none", 0.65, "none", "", 3000),
                ("d768_p4_mtp124", 24, 32, 4, "none", 0.65, "mtp_1_2_4", "1,2,4", 3000),
            ])
        if include_plan_size(plan_size, "wide"):
            confirms.extend([
                ("d512_p4_plain_6000", 16, 32, 4, "none", 0.65, "none", "", 6000),
                ("d1024_p4_plain_6000", 16, 64, 4, "none", 0.65, "none", "", 6000),
            ])
        for tag, num_experts, expert_size, passes, halt_mode, threshold, mtp_mode, horizons, steps in confirms:
            add_regional_experiment(
                plan,
                f"iv07_confirm_{tag}",
                "Longer confirmation for implementation-validation winners.",
                num_experts=num_experts,
                expert_size=expert_size,
                activation_passes=passes,
                shared_experts=1,
                topk_experts=1,
                balance=1e-2,
                diversity=1e-3,
                tick_halt_mode=halt_mode,
                tick_halt_threshold=threshold,
                tick_compute_weight=1e-3 if halt_mode != "none" else 0.0,
                moe_mtp_mode=mtp_mode,
                moe_mtp_horizons=horizons,
                iterations=4 if halt_mode != "none" else 2,
                synapse_depth=2,
                memory_hidden_dims=2,
                max_steps=steps,
            )

    if stage in ("iv08", "all"):
        dispatches = [
            ("regional_densemask_label", "dense_mask", 1.0, 0),
            ("regional_block_sparse", "block_sparse", 1.0, 0),
            ("regional_capacity1p00", "capacity_drop", 1.0, 1),
            ("regional_capacity1p25", "capacity_drop", 1.25, 1),
        ]
        if include_plan_size(plan_size, "full"):
            dispatches.extend([
                ("regional_capacity0p75", "capacity_drop", 0.75, 1),
                ("regional_dropless", "dropless", 1.0, 0),
            ])
        for tag, dispatch, capacity, drop_tokens in dispatches:
            add_regional_experiment(
                plan,
                f"iv08_dispatch_{tag}_d512_p4",
                "Exercise dispatch/capacity flags now consumed by grouped sparse regional routing.",
                num_experts=16,
                expert_size=32,
                activation_passes=4,
                shared_experts=1,
                topk_experts=1,
                balance=1e-2,
                diversity=1e-3,
                moe_dispatch_mode=dispatch,
                moe_capacity_factor=capacity,
                moe_drop_tokens=drop_tokens,
                iterations=2,
                synapse_depth=2,
                memory_hidden_dims=2,
            )

    return validate_plan(plan)


def build_batch_tune_plan(stage, batch_sizes, max_steps, log_interval, plan_size="full"):
    plan = []
    for exp in build_plan(stage, plan_size):
        for batch_size in batch_sizes:
            name = f"bt__{exp['name']}__bs{batch_size}"
            plan.append(clone_experiment(
                exp,
                name,
                {
                    "batch_size": batch_size,
                    "max_steps": max_steps,
                    "log_interval": log_interval,
                    "save_interval": max_steps + 1000000,
                    "no_tensorboard": None,
                },
                question=f"Batch-size probe for {exp['name']} at batch_size={batch_size}.",
            ))
    return plan


def quick_probe_experiment(exp, batch_size, max_steps, log_interval):
    name = f"qp__{exp['name']}__bs{batch_size}"
    return clone_experiment(
        exp,
        name,
        {
            "batch_size": batch_size,
            "max_steps": max_steps,
            "log_interval": log_interval,
            "save_interval": max_steps + 1000000,
            "no_tensorboard": None,
        },
        question=f"Quick batch probe for {exp['name']} at batch_size={batch_size}.",
    )


def load_batch_profile(path):
    if not path:
        return {}
    if not os.path.exists(path):
        print(f"warning: batch profile not found, using default batches: {path}", file=sys.stderr)
        return {}
    profile = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row.get("experiment_name") or row.get("base_experiment") or row.get("name")
            batch = row.get("batch_size") or row.get("recommended_batch_size")
            if name and batch:
                profile[name] = int(float(batch))
    return profile


def apply_batch_profile(plan, profile):
    if not profile:
        return plan
    tuned = []
    for exp in plan:
        if exp["name"] in profile:
            exp = clone_experiment(exp, exp["name"], {"batch_size": profile[exp["name"]]})
        tuned.append(exp)
    return tuned


def arg_items(args):
    items = []
    for key, value in args.items():
        if value is None:
            items.append(f"--{key}")
        else:
            items.extend([f"--{key}", str(value)])
    return items


def submit_command(exp, config, master_addr=None, port=None, wait=None):
    cmd = ["./scripts/ctmctl", "pool", "submit", config]
    if master_addr:
        cmd.extend(["--master_addr", master_addr])
    if port:
        cmd.extend(["--port", str(port)])
    if wait is not None:
        cmd.extend(["--wait", str(wait)])
    if exp.get("node_addrs"):
        cmd.extend(["--nodes", ",".join(exp["node_addrs"])])
    cmd.extend(arg_items(exp["args"]))
    return " ".join(shlex.quote(x) for x in cmd)


def load_cluster_node_addrs(path):
    if not path or not os.path.exists(path):
        return []
    node_addrs = []
    in_nodes = False
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("NODE_ADDRS=("):
                in_nodes = True
                rest = line[len("NODE_ADDRS=("):].strip()
                if rest.endswith(")"):
                    rest = rest[:-1].strip()
                    in_nodes = False
                node_addrs.extend(shlex.split(rest))
                continue
            if in_nodes:
                if line == ")":
                    in_nodes = False
                    continue
                if line.endswith(")"):
                    line = line[:-1].strip()
                    in_nodes = False
                node_addrs.extend(shlex.split(line))
    return node_addrs


def write_manifest(plan, path, config, master_addr, port, wait):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        fields = ["index", "name", "base_name", "batch_size", "nodes", "question", "command"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, exp in enumerate(plan):
            writer.writerow({
                "index": i,
                "name": exp["name"],
                "base_name": exp.get("base_name", exp["name"]),
                "batch_size": exp["args"].get("batch_size", ""),
                "nodes": ",".join(exp.get("node_addrs") or []),
                "question": exp["question"],
                "command": submit_command(exp, config, master_addr, port, wait),
            })


def print_commands(args):
    plan = apply_batch_profile(
        build_plan(args.stage, args.plan_size),
        load_batch_profile(args.batch_profile),
    )
    if args.output:
        write_manifest(plan, args.output, args.config, args.master_addr, args.port, args.wait)
        print(f"wrote {len(plan)} experiments: {args.output}")
    for i, exp in enumerate(plan):
        print(f"\n# {i:02d} {exp['name']}")
        print(f"# {exp['question']}")
        print(submit_command(exp, args.config, args.master_addr, args.port, args.wait))


def print_batch_commands(args):
    plan = build_batch_tune_plan(
        args.stage, args.batch_sizes, args.tune_steps, args.tune_log_interval,
        args.plan_size)
    plan = assign_node_groups(plan, resolve_node_groups(args, "batch-commands"))
    if args.output:
        write_manifest(plan, args.output, args.config, args.master_addr, args.port, args.wait)
        print(f"wrote {len(plan)} batch probes: {args.output}")
    for i, exp in enumerate(plan):
        print(f"\n# {i:02d} {exp['name']}")
        print(f"# {exp['question']}")
        print(submit_command(exp, args.config, args.master_addr, args.port, args.wait))


def pool_status(master_addr, port):
    with URL_OPENER.open(f"http://{master_addr}:{port}/status", timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_json(url, payload, timeout=10):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with URL_OPENER.open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8") or "{}")


def parse_gpu_spec(spec):
    if not spec:
        return None
    gpus = []
    for part in str(spec).replace("+", " ").replace("|", " ").split():
        if "-" in part:
            start, end = part.split("-", 1)
            if start.isdigit() and end.isdigit():
                gpus.extend(range(int(start), int(end) + 1))
        elif part.isdigit():
            gpus.append(int(part))
    return sorted(set(gpus))


def parse_node_spec(spec):
    if ":" not in spec:
        return spec, None
    addr, gpu_spec = spec.split(":", 1)
    return addr, parse_gpu_spec(gpu_spec)


def format_gpu_spec(gpus):
    gpus = list(gpus)
    if not gpus:
        return ""
    ranges = []
    start = prev = gpus[0]
    for gpu in gpus[1:]:
        if gpu == prev + 1:
            prev = gpu
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = gpu
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return "+".join(ranges)


def expand_node_groups_to_gpu_lanes(groups, gpus_per_lane=None, gpus_per_node=8):
    if not gpus_per_lane:
        return groups
    lanes = []
    for group in groups:
        if len(group) != 1:
            lanes.append(group)
            continue
        addr, gpus = parse_node_spec(group[0])
        if gpus is not None:
            lanes.append(group)
            continue
        for start in range(0, gpus_per_node, gpus_per_lane):
            lane_gpus = list(range(start, min(start + gpus_per_lane, gpus_per_node)))
            if len(lane_gpus) == gpus_per_lane:
                lanes.append([f"{addr}:{format_gpu_spec(lane_gpus)}"])
    return lanes


def parse_node_groups(items, gpus_per_lane=None, gpus_per_node=8):
    groups = []
    for item in items or []:
        parts = [part.strip() for part in item.split(",") if part.strip()]
        if parts:
            groups.append(parts)
    return expand_node_groups_to_gpu_lanes(groups, gpus_per_lane, gpus_per_node)


def resolve_node_groups(args, required_for):
    items = args.node_groups or load_cluster_node_addrs(args.config)
    node_groups = parse_node_groups(
        items,
        getattr(args, "gpus_per_lane", None),
        getattr(args, "gpus_per_node", 8),
    )
    if not node_groups:
        raise SystemExit(
            f"--node_groups is required for {required_for} when NODE_ADDRS cannot be read from {args.config}"
        )
    print(
        f"using {len(node_groups)} lane(s) from "
        f"{'--node_groups' if args.node_groups else args.config}"
    )
    return node_groups


def assign_node_groups(plan, node_groups):
    if not node_groups:
        return plan
    assigned = []
    for i, exp in enumerate(plan):
        exp = dict(exp)
        exp["node_addrs"] = node_groups[i % len(node_groups)]
        assigned.append(exp)
    return assigned


def wait_until_idle(master_addr, port, task_id, poll_interval):
    while True:
        status = pool_status(master_addr, port)
        nodes = status.get("nodes", {})
        running = [
            addr for addr, node in nodes.items()
            if task_id in (node.get("running_tasks") or [])
            or str(node.get("status", "")).startswith(f"running:{task_id}")
        ]
        if not running:
            return
        time.sleep(poll_interval)


def node_group_idle(master_addr, port, node_group):
    status = pool_status(master_addr, port)
    return node_group_idle_from_status(status, node_group)


def node_group_idle_from_status(status, node_group):
    nodes = status.get("nodes", {})
    for spec in node_group:
        addr, requested_gpus = parse_node_spec(spec)
        node = nodes.get(addr, {})
        if requested_gpus is None:
            if node.get("running_tasks") or str(node.get("status", "")).startswith("running:"):
                return False
        elif set(requested_gpus) & set(node.get("busy_gpus") or []):
            return False
    return True


def submit_exp(args, exp, wait=30.0):
    extra_args = " ".join(shlex.quote(item) for item in arg_items(exp["args"]))
    payload = {
        "config": args.config,
        "extra_args": extra_args,
        "node_addrs": exp.get("node_addrs") or [],
    }
    resp = post_json(f"http://{args.master_addr}:{args.port}/submit", payload)
    task = resp["task"]
    print(f"submitted task {task['task_id']}: {exp['name']} nodes={payload['node_addrs'] or 'all'}")
    if wait > 0:
        expected = {parse_node_spec(spec)[0] for spec in payload["node_addrs"]}
        if expected:
            deadline = time.time() + wait
            seen = set()
            while time.time() < deadline:
                status = pool_status(args.master_addr, args.port)
                acks = status.get("acks", {}).get(task["task_id"], {})
                for addr, ack in sorted(acks.items()):
                    if addr not in seen:
                        seen.add(addr)
                        print(f"ack {addr}: {ack.get('status')} {ack.get('message', '')}")
                if expected.issubset(seen):
                    break
                time.sleep(1)
    return task


def run_plan(args):
    if getattr(args, "batch_tune", False):
        plan = build_batch_tune_plan(
            args.stage, args.batch_sizes, args.tune_steps, args.tune_log_interval,
            args.plan_size)
    else:
        plan = apply_batch_profile(
            build_plan(args.stage, args.plan_size),
            load_batch_profile(args.batch_profile),
        )
    for i, exp in enumerate(plan):
        cmd = submit_command(exp, args.config, args.master_addr, args.port, args.wait)
        print(f"\n[{i + 1}/{len(plan)}] {exp['name']}")
        print(cmd)
        proc = subprocess.run(shlex.split(cmd), check=False)
        if proc.returncode != 0:
            raise SystemExit(proc.returncode)
        status = pool_status(args.master_addr, args.port)
        task = status.get("task") or {}
        task_id = task.get("task_id")
        if task_id:
            time.sleep(args.startup_grace)
            wait_until_idle(args.master_addr, args.port, task_id, args.poll_interval)


def run_parallel(args):
    if getattr(args, "batch_tune", False):
        plan = build_batch_tune_plan(
            args.stage, args.batch_sizes, args.tune_steps, args.tune_log_interval,
            args.plan_size)
    else:
        plan = apply_batch_profile(
            build_plan(args.stage, args.plan_size),
            load_batch_profile(args.batch_profile),
        )

    node_groups = resolve_node_groups(args, "run-parallel")

    queue = list(plan)
    running = {}
    completed = 0
    while queue or running:
        status = pool_status(args.master_addr, args.port)
        for idx, group in enumerate(node_groups):
            if idx in running or not queue:
                continue
            if not node_group_idle_from_status(status, group):
                continue
            exp = dict(queue.pop(0))
            exp["node_addrs"] = group
            print(f"\n[{completed + len(running) + 1}/{len(plan)}] lane={idx} {exp['name']}")
            task = submit_exp(args, exp, wait=0)
            running[idx] = {
                "task_id": task["task_id"],
                "nodes": group,
                "name": exp["name"],
                "submitted_at": time.time(),
            }

        done = []
        status = pool_status(args.master_addr, args.port)
        for idx, item in running.items():
            if time.time() - item["submitted_at"] < args.startup_grace:
                continue
            if node_group_idle_from_status(status, item["nodes"]):
                done.append(idx)
        for idx in done:
            item = running.pop(idx)
            completed += 1
            print(f"[done {completed}/{len(plan)}] lane={idx} {item['name']} task={item['task_id']}")

        if queue or running:
            time.sleep(args.poll_interval)


def probe_result(exp_name, metrics_dir):
    metrics = latest_row_by_experiment(metrics_dir).get(exp_name)
    failure = failure_reports_by_experiment(metrics_dir).get(exp_name)
    if failure:
        return {
            "status": failure.get("status", "failed"),
            "peak_memory_mb": failure.get("peak_memory_mb", ""),
            "tokens_per_sec": "",
            "metrics_file": "",
            "failure_file": failure.get("failure_file", ""),
            "error_type": failure.get("error_type", ""),
            "error": failure.get("error", "")[:500],
        }
    if metrics:
        return {
            "status": "ok",
            "peak_memory_mb": metrics.get("peak_memory_mb", ""),
            "tokens_per_sec": metrics.get("tokens_per_sec", ""),
            "metrics_file": metrics.get("metrics_file", ""),
            "failure_file": "",
            "error_type": "",
            "error": "",
        }
    return {
        "status": "missing_metrics",
        "peak_memory_mb": "",
        "tokens_per_sec": "",
        "metrics_file": "",
        "failure_file": "",
        "error_type": "",
        "error": "",
    }


def probe_result_after_settle(exp_name, metrics_dir, settle_seconds):
    deadline = time.time() + max(0.0, settle_seconds)
    result = probe_result(exp_name, metrics_dir)
    while result["status"] == "missing_metrics" and time.time() < deadline:
        time.sleep(min(1.0, max(0.0, deadline - time.time())))
        result = probe_result(exp_name, metrics_dir)
    return result


def result_peak_memory_mb(result):
    try:
        value = result.get("peak_memory_mb", "")
        return float(value) if value != "" else None
    except (TypeError, ValueError):
        return None


def apply_oom_backoff(base, batch_size, ratio):
    if ratio <= 0 or ratio >= 1:
        return
    cutoff = max(1, int(math.floor(batch_size * ratio)))
    base["batches"] = [bs for bs in base["batches"] if bs <= cutoff]


def maybe_refine_quick_probe(base, ok_batch_size, peak_memory_mb, limit_mb, args):
    if not args.refine_bracket_after_ok:
        return None
    failed_batches = [
        item["batch_size"] for item in base.get("attempts", [])
        if item["status"] in {"oom", "over_memory"} and item["batch_size"] > ok_batch_size
    ]
    if not failed_batches:
        return None
    upper_failed = min(failed_batches)
    skipped = [
        batch_size for batch_size in sorted(set(args.batch_sizes), reverse=True)
        if ok_batch_size < batch_size < upper_failed
    ]
    if not skipped:
        return None
    candidate = skipped[-1]
    if peak_memory_mb is not None:
        predicted_mb = peak_memory_mb * (candidate / ok_batch_size)
        if predicted_mb > limit_mb * args.refine_memory_margin:
            return None
    return candidate


def maybe_retry_skipped_after_missing(base, args):
    attempted = {item["batch_size"] for item in base.get("attempts", [])}
    failed = [
        item["batch_size"] for item in base.get("attempts", [])
        if item["status"] in {"oom", "over_memory"}
    ]
    uncertain = [
        item["batch_size"] for item in base.get("attempts", [])
        if item["status"] == "missing_metrics"
    ]
    if not failed or not uncertain:
        return None
    upper_failed = min(failed)
    lower_uncertain = max((bs for bs in uncertain if bs < upper_failed), default=None)
    if lower_uncertain is None:
        return None
    skipped = [
        batch_size for batch_size in sorted(set(args.batch_sizes), reverse=True)
        if lower_uncertain < batch_size < upper_failed and batch_size not in attempted
    ]
    return skipped[-1] if skipped else None


def write_quick_outputs(args, selected, attempts):
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    profile_fields = [
        "experiment_name", "recommended_batch_size", "peak_memory_mb",
        "tokens_per_sec", "memory_limit_mb", "num_successful_probes",
        "selected_probe", "metrics_file",
    ]
    limit_mb = args.target_memory_gb * 1024 * args.memory_util
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=profile_fields)
        writer.writeheader()
        all_names = getattr(args, "_quick_base_names", [])
        names = sorted(set(all_names) | set(selected.keys()))
        for base_name in names:
            item = selected.get(base_name)
            if item is None:
                writer.writerow({
                    "experiment_name": base_name,
                    "recommended_batch_size": args.fallback_batch_size,
                    "peak_memory_mb": "",
                    "tokens_per_sec": "",
                    "memory_limit_mb": round(limit_mb, 3),
                    "num_successful_probes": 0,
                    "selected_probe": "fallback_unprobed",
                    "metrics_file": "",
                })
                continue
            writer.writerow({
                "experiment_name": base_name,
                "recommended_batch_size": item["batch_size"],
                "peak_memory_mb": item.get("peak_memory_mb", ""),
                "tokens_per_sec": item.get("tokens_per_sec", ""),
                "memory_limit_mb": round(limit_mb, 3),
                "num_successful_probes": 1,
                "selected_probe": item["probe_name"],
                "metrics_file": item.get("metrics_file", ""),
            })

    if args.report_output:
        os.makedirs(os.path.dirname(args.report_output), exist_ok=True)
        fields = [
            "base_experiment", "probe_experiment", "batch_size", "status",
            "peak_memory_mb", "memory_limit_mb", "tokens_per_sec", "metrics_file",
            "failure_file", "error_type", "error",
        ]
        with open(args.report_output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(attempts)


def run_quick_probe(args):
    base_plan = build_plan(args.stage, args.plan_size)
    batch_sizes = sorted(set(args.batch_sizes), reverse=True)
    if args.fallback_batch_size is None:
        args.fallback_batch_size = min(batch_sizes)
    args._quick_base_names = [exp["name"] for exp in base_plan]
    node_groups = resolve_node_groups(args, "quick-probe")

    pending = {
        exp["name"]: {
            "exp": exp,
            "batches": list(batch_sizes),
            "done": False,
            "attempts": [],
        }
        for exp in base_plan
    }
    queue = list(pending.keys())
    running = {}
    selected = {}
    attempts = []
    started_at = time.time()
    deadline = started_at + args.time_limit_min * 60 if args.time_limit_min > 0 else None
    limit_mb = args.target_memory_gb * 1024 * args.memory_util

    def next_probe():
        while queue:
            base_name = queue.pop(0)
            item = pending[base_name]
            if item["done"] or not item["batches"]:
                continue
            batch_size = item["batches"].pop(0)
            return base_name, quick_probe_experiment(
                item["exp"], batch_size, args.tune_steps, args.tune_log_interval)
        return None, None

    total = len(base_plan)
    while queue or running:
        now = time.time()
        scheduling_open = deadline is None or now < deadline
        status = pool_status(args.master_addr, args.port)
        if scheduling_open:
            for idx, group in enumerate(node_groups):
                if idx in running:
                    continue
                if not node_group_idle_from_status(status, group):
                    continue
                base_name, exp = next_probe()
                if exp is None:
                    continue
                exp["node_addrs"] = group
                batch_size = exp["args"]["batch_size"]
                print(
                    f"\n[quick {len(selected)}/{total}] lane={idx} "
                    f"{base_name} bs={batch_size}",
                    flush=True,
                )
                task = submit_exp(args, exp, wait=0)
                running[idx] = {
                    "task_id": task["task_id"],
                    "nodes": group,
                    "name": exp["name"],
                    "base_name": base_name,
                    "batch_size": batch_size,
                    "submitted_at": time.time(),
                }

        done = []
        status = pool_status(args.master_addr, args.port)
        for idx, item in running.items():
            if time.time() - item["submitted_at"] < args.startup_grace:
                continue
            if node_group_idle_from_status(status, item["nodes"]):
                done.append(idx)

        for idx in done:
            item = running.pop(idx)
            result = probe_result_after_settle(
                item["name"], args.metrics_dir, args.metrics_settle_seconds)
            peak_memory_mb = result_peak_memory_mb(result)
            status = result["status"]
            if status == "ok" and peak_memory_mb is not None and peak_memory_mb > limit_mb:
                status = "over_memory"
                result = dict(result)
                result["status"] = status
                result["error"] = (
                    f"peak_memory_mb={peak_memory_mb:.3f} exceeds "
                    f"limit_mb={limit_mb:.3f}"
                )
            attempt = {
                "base_experiment": item["base_name"],
                "probe_experiment": item["name"],
                "batch_size": item["batch_size"],
                "memory_limit_mb": round(limit_mb, 3),
                **result,
            }
            attempts.append(attempt)
            base = pending[item["base_name"]]
            base["attempts"].append({
                "batch_size": item["batch_size"],
                "status": status,
                "peak_memory_mb": peak_memory_mb,
            })
            if status == "ok":
                selected_item = {
                    "batch_size": item["batch_size"],
                    "probe_name": item["name"],
                    "peak_memory_mb": result.get("peak_memory_mb", ""),
                    "tokens_per_sec": result.get("tokens_per_sec", ""),
                    "metrics_file": result.get("metrics_file", ""),
                }
                refine_batch = maybe_refine_quick_probe(
                    base, item["batch_size"], peak_memory_mb, limit_mb, args)
                if refine_batch is not None:
                    selected_item["refine_fallback"] = True
                    selected[item["base_name"]] = selected_item
                    base["done"] = False
                    base["batches"] = [refine_batch]
                    queue.append(item["base_name"])
                else:
                    base["done"] = True
                    selected[item["base_name"]] = selected_item
                print(
                    f"[selected {len(selected)}/{total}] {item['base_name']} "
                    f"bs={item['batch_size']} mem={result.get('peak_memory_mb', '')}",
                    flush=True,
                )
            else:
                if status in {"oom", "over_memory"}:
                    apply_oom_backoff(base, item["batch_size"], args.oom_backoff_ratio)
                if base["batches"]:
                    queue.append(item["base_name"])
                else:
                    retry_batch = maybe_retry_skipped_after_missing(base, args)
                    if retry_batch is not None and item["base_name"] not in selected:
                        base["batches"] = [retry_batch]
                        queue.append(item["base_name"])
                    else:
                        base["done"] = True
                print(
                    f"[probe {result['status']}] {item['base_name']} "
                    f"bs={item['batch_size']} next={base['batches'][:1] or '-'}",
                    flush=True,
                )
            write_quick_outputs(args, selected, attempts)

        if deadline is not None and time.time() >= deadline and not running:
            break
        if deadline is not None and time.time() >= deadline and queue:
            queue = []
        if queue or running:
            time.sleep(args.poll_interval)

    write_quick_outputs(args, selected, attempts)
    missing = total - len(selected)
    print(
        f"quick probe done: selected={len(selected)} missing={missing} "
        f"profile={args.output} report={args.report_output}",
        flush=True,
    )


def latest_rows(metrics_dir):
    rows = []
    for path in glob.glob(os.path.join(metrics_dir, "*.csv")):
        with open(path, newline="", encoding="utf-8") as f:
            data = list(csv.DictReader(f))
        if not data:
            continue
        row = data[-1]
        row["metrics_file"] = path
        rows.append(row)
    return rows


def parse_float(row, key, default=math.nan):
    try:
        value = row.get(key, "")
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def batch_probe_base_name(experiment_name):
    if not (experiment_name.startswith("bt__") or experiment_name.startswith("qp__")):
        return None
    body = experiment_name.split("__", 1)[1]
    if "__bs" not in body:
        return None
    return body.rsplit("__bs", 1)[0]


def is_regional_experiment(name):
    return bool(name) and name.startswith(REGIONAL_PREFIXES)


def is_final_metrics_row(row):
    name = row.get("experiment_name", "")
    metrics_file = row.get("metrics_file", "")
    if not is_regional_experiment(name):
        return False
    return os.path.basename(metrics_file) == f"{name}.csv"


def recommend_batches(args):
    rows = latest_rows(args.metrics_dir)
    probes = []
    for row in rows:
        base = batch_probe_base_name(row.get("experiment_name", ""))
        if not base or not is_regional_experiment(base):
            continue
        peak = parse_float(row, "peak_memory_mb")
        batch = int(parse_float(row, "batch_size", 0))
        tokens_per_sec = parse_float(row, "tokens_per_sec")
        if batch <= 0 or math.isnan(peak):
            continue
        probes.append({
            "base_experiment": base,
            "experiment_name": row["experiment_name"],
            "batch_size": batch,
            "peak_memory_mb": peak,
            "tokens_per_sec": tokens_per_sec,
            "loss": row.get("loss", ""),
            "metrics_file": row.get("metrics_file", ""),
        })

    limit_mb = args.target_memory_gb * 1024 * args.memory_util
    grouped = {}
    for probe in probes:
        grouped.setdefault(probe["base_experiment"], []).append(probe)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fields = [
        "experiment_name", "recommended_batch_size", "peak_memory_mb",
        "tokens_per_sec", "memory_limit_mb", "num_successful_probes",
        "selected_probe", "metrics_file",
    ]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for base, items in sorted(grouped.items()):
            under = [x for x in items if x["peak_memory_mb"] <= limit_mb]
            if under:
                # Prefer highest stable batch, then highest throughput.
                selected = sorted(
                    under,
                    key=lambda x: (x["batch_size"], x["tokens_per_sec"]),
                    reverse=True,
                )[0]
            else:
                # Everything that succeeded exceeded the target utilization.
                selected = sorted(items, key=lambda x: (x["peak_memory_mb"], x["batch_size"]))[0]
            writer.writerow({
                "experiment_name": base,
                "recommended_batch_size": selected["batch_size"],
                "peak_memory_mb": round(selected["peak_memory_mb"], 3),
                "tokens_per_sec": round(selected["tokens_per_sec"], 3)
                if not math.isnan(selected["tokens_per_sec"]) else "",
                "memory_limit_mb": round(limit_mb, 3),
                "num_successful_probes": len(items),
                "selected_probe": selected["experiment_name"],
                "metrics_file": selected["metrics_file"],
            })
    print(f"wrote batch recommendations: {args.output}")


def latest_row_by_experiment(metrics_dir):
    by_name = {}
    for row in latest_rows(metrics_dir):
        name = row.get("experiment_name", "")
        if name:
            by_name[name] = row
    return by_name


def failure_reports_by_experiment(metrics_dir):
    def priority(report):
        status_score = 0 if report.get("status") == "oom" else 1
        rank = report.get("rank")
        try:
            rank_score = int(rank)
        except (TypeError, ValueError):
            rank_score = 0
        return (status_score, rank_score, report.get("failure_file", ""))

    grouped = {}
    for path in sorted(glob.glob(os.path.join(metrics_dir, "*.fail.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        name = data.get("experiment_name")
        if name:
            data["failure_file"] = path
            grouped.setdefault(name, []).append(data)
    return {
        name: sorted(items, key=priority)[0]
        for name, items in grouped.items()
    }


def batch_report(args):
    planned = build_batch_tune_plan(
        args.stage,
        args.batch_sizes,
        args.tune_steps,
        args.tune_log_interval,
        args.plan_size,
    )
    metrics = latest_row_by_experiment(args.metrics_dir)
    failures = failure_reports_by_experiment(args.metrics_dir)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fields = [
        "base_experiment", "probe_experiment", "batch_size", "status",
        "peak_memory_mb", "tokens_per_sec", "steps_per_sec", "loss",
        "global_step", "world_size", "metrics_file", "failure_file",
        "failure_rank", "error_type", "error", "question",
    ]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for exp in planned:
            row = metrics.get(exp["name"])
            failure = failures.get(exp["name"])
            status = failure.get("status", "failed") if failure else ("ok" if row else "missing_metrics")
            writer.writerow({
                "base_experiment": exp.get("base_name", exp["name"]),
                "probe_experiment": exp["name"],
                "batch_size": exp["args"].get("batch_size", ""),
                "status": status,
                "peak_memory_mb": (
                    failure.get("peak_memory_mb", "")
                    if failure else row.get("peak_memory_mb", "") if row else ""
                ),
                "tokens_per_sec": row.get("tokens_per_sec", "") if row else "",
                "steps_per_sec": row.get("steps_per_sec", "") if row else "",
                "loss": row.get("loss", "") if row else "",
                "global_step": row.get("global_step", "") if row else "",
                "world_size": (
                    failure.get("world_size", "")
                    if failure else row.get("world_size", "") if row else ""
                ),
                "metrics_file": row.get("metrics_file", "") if row else "",
                "failure_file": failure.get("failure_file", "") if failure else "",
                "failure_rank": failure.get("rank", "") if failure else "",
                "error_type": failure.get("error_type", "") if failure else "",
                "error": failure.get("error", "")[:500] if failure else "",
                "question": exp["question"],
            })
    print(f"wrote batch probe report: {args.output}")


def export_final_plan(args):
    profile = load_batch_profile(args.batch_profile)
    plan = apply_batch_profile(build_plan(args.stage, args.plan_size), profile)
    write_manifest(plan, args.output, args.config, args.master_addr, args.port, args.wait)
    print(f"wrote final execution plan: {args.output}")


def summarize(args):
    rows = [
        row for row in latest_rows(args.metrics_dir)
        if is_final_metrics_row(row)
    ]
    for row in rows:
        loss = parse_float(row, "loss")
        peak_memory_mb = parse_float(row, "peak_memory_mb")
        tokens_per_sec = parse_float(row, "tokens_per_sec")
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
        "moe_expert_dropout", "moe_activation_passes",
        "moe_region_diversity_weight", "moe_mtp_mode", "moe_mtp_horizons",
        "self_cond", "cross_layer_state", "global_step",
        "metrics_file",
    ]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote summary: {args.output}")


def parse_args():
    parser = argparse.ArgumentParser(description="CTM implementation-validation experiment plan helper")
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--stage", default="all", choices=REGIONAL_STAGES)
    common.add_argument("--config", default=DEFAULT_CLUSTER_CONFIG)
    common.add_argument("--master_addr", default=None)
    common.add_argument("--port", type=int, default=None)
    common.add_argument("--wait", type=float, default=30.0)
    common.add_argument("--plan_size", default="full", choices=PLAN_SIZES,
                        help="core=current compact matrix, full=default sufficient sweep, wide=extra exploratory runs.")

    p = sub.add_parser("commands", parents=[common])
    p.add_argument("--output", default="runs/experiment_plans/impl_validation_plan.csv")
    p.add_argument("--batch_profile", default=None)
    p.set_defaults(func=print_commands)

    p = sub.add_parser("final-plan", parents=[common])
    p.add_argument("--output", default="runs/experiment_plans/impl_validation_final_plan.csv")
    p.add_argument("--batch_profile", default="runs/metrics/impl_validation_batch_profile.csv")
    p.set_defaults(func=export_final_plan)

    p = sub.add_parser("batch-commands", parents=[common])
    p.add_argument("--batch_sizes", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12])
    p.add_argument("--tune_steps", type=int, default=80)
    p.add_argument("--tune_log_interval", type=int, default=20)
    p.add_argument("--node_groups", nargs="*", default=None,
                   help="Optional node groups assigned round-robin; omitted means NODE_ADDRS from --config.")
    p.add_argument("--gpus_per_lane", type=int, default=2,
                   help="Split bare single-node groups into GPU lanes; default 2 gives 16 two-GPU lanes for four 8-GPU nodes.")
    p.add_argument("--gpus_per_node", type=int, default=8)
    p.add_argument("--output", default="runs/experiment_plans/impl_validation_batch_tune_plan.csv")
    p.set_defaults(func=print_batch_commands)

    p = sub.add_parser("run", parents=[common])
    p.add_argument("--startup_grace", type=float, default=20.0)
    p.add_argument("--poll_interval", type=float, default=30.0)
    p.add_argument("--batch_profile", default=None)
    p.add_argument("--batch_tune", action="store_true")
    p.add_argument("--batch_sizes", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12])
    p.add_argument("--tune_steps", type=int, default=80)
    p.add_argument("--tune_log_interval", type=int, default=20)
    p.set_defaults(func=run_plan)

    p = sub.add_parser("run-parallel", parents=[common])
    p.add_argument("--startup_grace", type=float, default=20.0)
    p.add_argument("--poll_interval", type=float, default=30.0)
    p.add_argument("--batch_profile", default=None)
    p.add_argument("--batch_tune", action="store_true")
    p.add_argument("--batch_sizes", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12])
    p.add_argument("--tune_steps", type=int, default=80)
    p.add_argument("--tune_log_interval", type=int, default=20)
    p.add_argument("--node_groups", nargs="*", default=None,
                   help="Node groups for parallel lanes; omitted means NODE_ADDRS from --config.")
    p.add_argument("--gpus_per_lane", type=int, default=2,
                   help="Split bare single-node groups into GPU lanes; default 2 gives 16 two-GPU lanes for four 8-GPU nodes.")
    p.add_argument("--gpus_per_node", type=int, default=8)
    p.set_defaults(func=run_parallel)

    p = sub.add_parser("quick-probe", parents=[common])
    p.add_argument("--startup_grace", type=float, default=8.0)
    p.add_argument("--poll_interval", type=float, default=5.0)
    p.add_argument("--batch_sizes", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12])
    p.add_argument("--tune_steps", type=int, default=3)
    p.add_argument("--tune_log_interval", type=int, default=1)
    p.add_argument("--time_limit_min", type=float, default=15.0)
    p.add_argument("--metrics_settle_seconds", type=float, default=8.0,
                   help="Wait this long for metrics/failure files after a lane becomes idle.")
    p.add_argument("--oom_backoff_ratio", type=float, default=0.67,
                   help="After OOM/over_memory, only try remaining batches <= current_batch * ratio; set 1.0 to disable.")
    p.add_argument("--refine_bracket_after_ok", action=argparse.BooleanOptionalAction, default=True,
                   help="After a lower batch succeeds below an OOM/over_memory batch, try one skipped middle batch when memory looks safe.")
    p.add_argument("--refine_memory_margin", type=float, default=0.97,
                   help="Only run bracket refinement when linear memory estimate is below limit * margin.")
    p.add_argument("--metrics_dir", default="runs/metrics")
    p.add_argument("--output", default="runs/metrics/impl_validation_batch_profile_quick.csv")
    p.add_argument("--report_output", default="runs/metrics/impl_validation_quick_probe_report.csv")
    p.add_argument("--target_memory_gb", type=float, default=80.0)
    p.add_argument("--memory_util", type=float, default=0.90)
    p.add_argument("--fallback_batch_size", type=int, default=None,
                   help="Batch size used for experiments not resolved before the time limit; default=min(batch_sizes).")
    p.add_argument("--node_groups", nargs="*", default=None,
                   help="Node groups for quick probe lanes; omitted means NODE_ADDRS from --config.")
    p.add_argument("--gpus_per_lane", type=int, default=2,
                   help="Split bare single-node groups into GPU lanes; default 2 gives 16 two-GPU lanes for four 8-GPU nodes.")
    p.add_argument("--gpus_per_node", type=int, default=8)
    p.set_defaults(func=run_quick_probe)

    p = sub.add_parser("summarize")
    p.add_argument("--metrics_dir", default="runs/metrics")
    p.add_argument("--output", default="runs/metrics/impl_validation_summary.csv")
    p.set_defaults(func=summarize)

    p = sub.add_parser("recommend-batches")
    p.add_argument("--metrics_dir", default="runs/metrics")
    p.add_argument("--output", default="runs/metrics/impl_validation_batch_profile.csv")
    p.add_argument("--target_memory_gb", type=float, default=80.0)
    p.add_argument("--memory_util", type=float, default=0.90)
    p.set_defaults(func=recommend_batches)

    p = sub.add_parser("batch-report")
    p.add_argument("--stage", default="all", choices=REGIONAL_STAGES)
    p.add_argument("--plan_size", default="full", choices=PLAN_SIZES)
    p.add_argument("--metrics_dir", default="runs/metrics")
    p.add_argument("--output", default="runs/metrics/impl_validation_batch_probe_report.csv")
    p.add_argument("--batch_sizes", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12])
    p.add_argument("--tune_steps", type=int, default=80)
    p.add_argument("--tune_log_interval", type=int, default=20)
    p.set_defaults(func=batch_report)

    args = parser.parse_args()
    if args.cmd in ("run", "run-parallel", "quick-probe"):
        if args.master_addr is None:
            args.master_addr = "11.131.210.78"
        if args.port is None:
            args.port = 8765
    return args


if __name__ == "__main__":
    args = parse_args()
    args.func(args)
