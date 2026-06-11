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
REGIONAL_DISPATCH_BLOCK_SPARSE = True


class _PlanContext:
    def __init__(self):
        self.build_plan = None
        self.summarize_fn = None
        self.is_regional_experiment = None
        self.is_final_metrics_row = None
        self.stages = ()
        self.prefixes = ()
        self.metrics_prefix = "impl_validation"
        self.cluster_config = "infra/clusters/h100_4nodes.env"


_ctx = _PlanContext()

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

_ctx.stages = REGIONAL_STAGES
_ctx.prefixes = REGIONAL_PREFIXES
_ctx.metrics_prefix = "impl_validation"
_ctx.cluster_config = DEFAULT_CLUSTER_CONFIG


def metrics_path(name):
    return f"runs/metrics/{_ctx.metrics_prefix}_{name}"


def configure_plan_defaults(
    *,
    metrics_prefix=None,
    cluster_config=None,
    dispatch_block_sparse=None,
    build_plan=None,
    summarize_fn=None,
    is_regional_experiment=None,
    is_final_metrics_row=None,
    stages=None,
    prefixes=None,
):
    global DEFAULT_CLUSTER_CONFIG, REGIONAL_DISPATCH_BLOCK_SPARSE
    if metrics_prefix:
        _ctx.metrics_prefix = metrics_prefix
    if cluster_config:
        DEFAULT_CLUSTER_CONFIG = cluster_config
        _ctx.cluster_config = cluster_config
    if dispatch_block_sparse is not None:
        REGIONAL_DISPATCH_BLOCK_SPARSE = dispatch_block_sparse
    if build_plan is not None:
        _ctx.build_plan = build_plan
    if summarize_fn is not None:
        _ctx.summarize_fn = summarize_fn
    if is_regional_experiment is not None:
        _ctx.is_regional_experiment = is_regional_experiment
    if is_final_metrics_row is not None:
        _ctx.is_final_metrics_row = is_final_metrics_row
    if stages is not None:
        _ctx.stages = stages
    if prefixes is not None:
        _ctx.prefixes = prefixes


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
    if REGIONAL_DISPATCH_BLOCK_SPARSE:
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


_ctx.build_plan = build_plan


def build_batch_tune_plan(stage, batch_sizes, max_steps, log_interval, plan_size="full"):
    plan = []
    for exp in _ctx.build_plan(stage, plan_size):
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


def memory_limit_mb(args):
    return args.target_memory_gb * 1024 * args.memory_util


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


def load_batch_profile_meta(path):
    meta = {}
    if not path or not os.path.exists(path):
        return meta
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row.get("experiment_name") or row.get("base_experiment") or row.get("name")
            batch = row.get("batch_size") or row.get("recommended_batch_size")
            if not name or not batch:
                continue
            meta[name] = {
                "batch_size": int(float(batch)),
                "source": row.get("selected_probe", "profile_csv"),
                "peak_memory_mb": row.get("peak_memory_mb", ""),
                "metrics_file": row.get("metrics_file", ""),
            }
    return meta


def count_quick_probe_metrics(metrics_dir):
    pattern = os.path.join(metrics_dir, "qp__*.csv")
    return len(glob.glob(pattern))


def completed_final_experiments(metrics_dir):
    done = set()
    for name, row in latest_row_by_experiment(metrics_dir).items():
        if is_completed_experiment(row):
            done.add(name)
    return done


def experiment_memory_family(exp):
    args = exp["args"]
    d_model = int(args.get("d_model", 512))
    iterations = int(args.get("iterations", 4))
    memory_length = int(args.get("memory_length", 8))
    activation_passes = int(args.get("moe_activation_passes", 1))
    active_width = int(args.get("cell_topk", d_model))
    mtp = args.get("moe_mtp_mode", "none")
    elf = args.get("elf_horizon_mode", "none")
    fast_output = args.get("fast_output_mode", "none")
    diff_cell = args.get("diff_cell_mode", "none")

    if d_model <= 512:
        d_bucket = 512
    elif d_model <= 1024:
        d_bucket = 1024
    elif d_model <= 1536:
        d_bucket = 1536
    else:
        d_bucket = 2048

    if iterations <= 2:
        tick_bucket = "t2"
    elif iterations <= 6:
        tick_bucket = "t6"
    elif iterations <= 12:
        tick_bucket = "t12"
    else:
        tick_bucket = "t16"

    if memory_length <= 4:
        mem_bucket = "m4"
    elif memory_length <= 8:
        mem_bucket = "m8"
    elif memory_length <= 16:
        mem_bucket = "m16"
    else:
        mem_bucket = "m32"

    width_bucket = min(active_width, 2048)
    width_bucket = int(math.ceil(width_bucket / 64.0) * 64)

    return (
        f"d{d_bucket}_{tick_bucket}_{mem_bucket}_p{activation_passes}"
        f"_w{width_bucket}_mtp{int(mtp != 'none')}_elf{int(elf != 'none')}"
        f"_fast{int(fast_output != 'none')}_diff{int(diff_cell != 'none')}"
    )


def conservative_heuristic_batch(exp, batch_sizes, limit_mb, fallback):
    args = exp["args"]
    d_model = int(args.get("d_model", 512))
    iterations = int(args.get("iterations", 4))
    memory_length = int(args.get("memory_length", 8))
    activation_passes = int(args.get("moe_activation_passes", 1))
    active_width = int(args.get("cell_topk", d_model))
    plan_default = int(args.get("batch_size", fallback))

    if d_model <= 512:
        batch = 12
    elif d_model <= 768:
        batch = 10
    elif d_model <= 1024:
        batch = 8
    elif d_model <= 1536:
        batch = 6
    else:
        batch = 4

    if iterations >= 12:
        batch -= 2
    elif iterations >= 8:
        batch -= 1
    if memory_length >= 32:
        batch -= 2
    elif memory_length >= 16:
        batch -= 1
    if activation_passes >= 5:
        batch -= 2
    elif activation_passes >= 4:
        batch -= 1
    if active_width >= 512:
        batch -= 2
    elif active_width >= 256:
        batch -= 1
    if args.get("moe_mtp_mode", "none") != "none":
        batch -= 1
    if args.get("elf_horizon_mode", "none") != "none":
        batch -= 1
    if args.get("fast_output_mode", "none") != "none":
        batch -= 1
    if args.get("diff_cell_mode", "none") != "none":
        batch -= 1

    batch = max(min(batch_sizes), min(max(batch_sizes), batch, plan_default))
    return batch


def infer_batch_from_family(exp, family_samples, batch_sizes, limit_mb, fallback):
    if not family_samples:
        return conservative_heuristic_batch(exp, batch_sizes, limit_mb, fallback), "heuristic"

    valid = []
    for item in family_samples:
        if item.get("batch_size", 0) <= 0:
            continue
        peak = parse_float(item, "peak_memory_mb")
        if math.isnan(peak):
            continue
        valid.append({**item, "peak_memory_mb": peak})
    if not valid:
        return conservative_heuristic_batch(exp, batch_sizes, limit_mb, fallback), "heuristic"

    best = None
    for candidate in sorted(set(batch_sizes), reverse=True):
        predicted = []
        for sample in valid:
            peak = float(sample["peak_memory_mb"])
            sample_batch = int(sample["batch_size"])
            if sample_batch <= 0:
                continue
            predicted.append(peak * (candidate / sample_batch))
        if not predicted:
            continue
        worst = max(predicted)
        if worst <= limit_mb:
            best = candidate
            break
    if best is None:
        smallest = sorted(valid, key=lambda x: (float(x["peak_memory_mb"]), -int(x["batch_size"])))[0]
        best = max(min(batch_sizes), int(smallest["batch_size"]))
        return best, "family_conservative"
    return best, "family_inferred"


def load_probe_samples_from_metrics(metrics_dir, base_names, limit_mb):
    metrics = latest_row_by_experiment(metrics_dir)
    failures = failure_reports_by_experiment(metrics_dir)
    samples = {}
    for row in metrics.values():
        probe_name = row.get("experiment_name", "")
        base = batch_probe_base_name(probe_name)
        if not base or base not in base_names:
            continue
        peak = parse_float(row, "peak_memory_mb")
        batch = int(parse_float(row, "batch_size", 0))
        if batch <= 0 or math.isnan(peak):
            continue
        status = "ok" if peak <= limit_mb else "over_memory"
        samples.setdefault(base, []).append({
            "base_experiment": base,
            "batch_size": batch,
            "peak_memory_mb": peak,
            "status": status,
            "probe_name": probe_name,
            "metrics_file": row.get("metrics_file", ""),
            "source": "resume_metrics",
        })
    for probe_name, failure in failures.items():
        base = batch_probe_base_name(probe_name)
        if not base or base not in base_names:
            continue
        peak = failure.get("peak_memory_mb", "")
        try:
            peak = float(peak) if peak != "" else math.nan
        except (TypeError, ValueError):
            peak = math.nan
        batch = int(parse_float({"batch_size": failure.get("batch_size", "")}, "batch_size", 0))
        if batch <= 0:
            continue
        samples.setdefault(base, []).append({
            "base_experiment": base,
            "batch_size": batch,
            "peak_memory_mb": peak,
            "status": failure.get("status", "failed"),
            "probe_name": probe_name,
            "metrics_file": "",
            "source": "resume_failure",
        })
    return samples


def best_probe_sample(samples):
    under = [x for x in samples if x.get("status") == "ok" and not math.isnan(float(x["peak_memory_mb"]))]
    if under:
        return sorted(under, key=lambda x: (x["batch_size"], -float(x["peak_memory_mb"])), reverse=True)[0]
    valid = [x for x in samples if not math.isnan(float(x.get("peak_memory_mb", math.nan)))]
    if valid:
        return sorted(valid, key=lambda x: (float(x["peak_memory_mb"]), x["batch_size"]))[0]
    return None


def build_smart_profile(base_plan, args):
    batch_sizes = sorted(set(args.batch_sizes), reverse=True)
    if args.fallback_batch_size is None:
        args.fallback_batch_size = min(batch_sizes)
    limit_mb = memory_limit_mb(args)
    base_names = {exp["name"] for exp in base_plan}

    profile = {}
    meta = {}

    for path, source in (
        (args.batch_profile, "profile_csv"),
        (args.quick_output, "quick_csv"),
    ):
        for name, item in load_batch_profile_meta(path).items():
            if name in base_names and name not in profile:
                profile[name] = item["batch_size"]
                meta[name] = {**item, "source": source}

    probe_samples = load_probe_samples_from_metrics(args.metrics_dir, base_names, limit_mb)
    for base, items in probe_samples.items():
        if base in profile:
            continue
        selected = best_probe_sample(items)
        if selected and selected.get("status") == "ok":
            profile[base] = int(selected["batch_size"])
            meta[base] = {
                "batch_size": int(selected["batch_size"]),
                "source": "resume_metrics",
                "peak_memory_mb": selected.get("peak_memory_mb", ""),
                "metrics_file": selected.get("metrics_file", ""),
                "selected_probe": selected.get("probe_name", ""),
            }

    family_probed = {}
    exp_by_name = {exp["name"]: exp for exp in base_plan}
    for name in sorted(profile):
        exp = exp_by_name.get(name)
        if not exp:
            continue
        family = experiment_memory_family(exp)
        peak = meta.get(name, {}).get("peak_memory_mb", "")
        try:
            peak_val = float(peak) if peak != "" else math.nan
        except (TypeError, ValueError):
            peak_val = math.nan
        if math.isnan(peak_val):
            continue
        family_probed.setdefault(family, []).append({
            "base_experiment": name,
            "batch_size": profile[name],
            "peak_memory_mb": peak_val,
        })

    for exp in base_plan:
        name = exp["name"]
        if name in profile:
            continue
        family = experiment_memory_family(exp)
        batch, source = infer_batch_from_family(
            exp, family_probed.get(family, []), batch_sizes, limit_mb, args.fallback_batch_size)
        profile[name] = batch
        meta[name] = {
            "batch_size": batch,
            "source": source,
            "peak_memory_mb": "",
            "metrics_file": "",
            "selected_probe": source,
        }

    return profile, meta


def write_merged_batch_profile(path, base_plan, profile, meta, args):
    limit_mb = memory_limit_mb(args)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = [
        "experiment_name", "recommended_batch_size", "peak_memory_mb",
        "tokens_per_sec", "memory_limit_mb", "num_successful_probes",
        "selected_probe", "metrics_file",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for exp in base_plan:
            name = exp["name"]
            item = meta.get(name, {})
            writer.writerow({
                "experiment_name": name,
                "recommended_batch_size": profile[name],
                "peak_memory_mb": item.get("peak_memory_mb", ""),
                "tokens_per_sec": item.get("tokens_per_sec", ""),
                "memory_limit_mb": round(limit_mb, 3),
                "num_successful_probes": 1 if item.get("source") in {
                    "profile_csv", "quick_csv", "resume_metrics", "probed",
                } else 0,
                "selected_probe": item.get("selected_probe", item.get("source", "inferred")),
                "metrics_file": item.get("metrics_file", ""),
            })


def select_probe_targets(base_plan, profile_meta, args):
    if args.max_probe_experiments <= 0:
        return []

    candidates = []
    for exp in base_plan:
        name = exp["name"]
        item = profile_meta.get(name, {})
        source = item.get("source", "")
        if source in {"profile_csv", "quick_csv", "resume_metrics", "probed"}:
            continue
        candidates.append(exp)

    if not candidates:
        return []

    by_family = {}
    for exp in candidates:
        by_family.setdefault(experiment_memory_family(exp), []).append(exp)

    targets = []
    for family in sorted(by_family):
        targets.append(by_family[family][0]["name"])
        if len(targets) >= args.max_probe_experiments:
            return targets

    for exp in candidates:
        if exp["name"] not in targets:
            targets.append(exp["name"])
        if len(targets) >= args.max_probe_experiments:
            break
    return targets


def effective_probe_budget_min(args):
    probe_budget = getattr(args, "probe_time_budget_min", 0) or 0
    if probe_budget > 0:
        return probe_budget
    legacy = getattr(args, "time_limit_min", 0) or 0
    return legacy if legacy > 0 else 0


def prepare_probe_run_args(args):
    batch_sizes = getattr(args, "batch_sizes", None)
    if batch_sizes:
        batch_sizes = sorted(set(batch_sizes), reverse=True)
        if getattr(args, "fallback_batch_size", None) is None:
            args.fallback_batch_size = min(batch_sizes)
    if args.cmd in ("probe-and-run", "quick-probe") and args.master_addr is None:
        args.master_addr = "11.131.210.78"
    if args.cmd in ("probe-and-run", "quick-probe") and args.port is None:
        args.port = 8765


def print_execution_plan(args, base_plan, *, mode, probe_targets=None, profile=None, profile_meta=None):
    total = len(base_plan)
    done = completed_final_experiments(args.metrics_dir)
    pending_runs = total - len(done)
    qp_count = count_quick_probe_metrics(args.metrics_dir)
    has_profile = bool(args.batch_profile and os.path.exists(args.batch_profile))
    quick_output = getattr(args, "quick_output", None)
    has_quick = bool(quick_output and os.path.exists(quick_output))
    budget_min = effective_probe_budget_min(args)
    probe_targets = probe_targets or []
    profile = profile or {}
    profile_meta = profile_meta or {}

    probed = sum(1 for item in profile_meta.values() if item.get("source") in {
        "profile_csv", "quick_csv", "resume_metrics", "probed"})
    inferred = sum(1 for item in profile_meta.values() if item.get("source", "").startswith("family"))
    heuristic = sum(1 for item in profile_meta.values() if item.get("source") == "heuristic")

    print("\n=== execution plan ===", flush=True)
    print(f"mode: {mode}", flush=True)
    print(f"stage={args.stage} plan_size={args.plan_size} total_experiments={total}", flush=True)
    print(f"formal_runs_pending={pending_runs} already_completed={len(done)}", flush=True)
    print(f"existing_qp_metrics={qp_count}", flush=True)
    print(f"batch_profile={args.batch_profile} exists={has_profile}", flush=True)
    print(f"quick_profile={quick_output} exists={has_quick}", flush=True)
    if mode != "run-only":
        print(f"probe_targets={len(probe_targets)} max_probe_experiments={args.max_probe_experiments}", flush=True)
        print(f"max_probe_attempts={args.max_probe_attempts} probe_time_budget_min={budget_min or 'unlimited'}", flush=True)
        print(f"streaming={getattr(args, 'streaming', False)} max_probe_lanes={getattr(args, 'max_probe_lanes', 0)}", flush=True)
    if profile_meta:
        print(
            f"batch_sources: probed_or_cached={probed} family_inferred={inferred} "
            f"heuristic={heuristic}",
            flush=True,
        )
    if mode == "run-only":
        print("TIP: use run-only when batch_profile already exists to avoid repeating probe.", flush=True)
    elif has_profile and not getattr(args, "force_probe", False):
        print("TIP: batch_profile exists; default is run-only unless --force_probe.", flush=True)
    elif qp_count >= max(32, total // 2):
        print(
            f"WARNING: found {qp_count} qp__ metrics. Consider run-only:\n"
            f"  python ... run-parallel --stage {args.stage} "
            f"--batch_profile {args.batch_profile}",
            flush=True,
        )
    print("======================\n", flush=True)


def should_skip_probe(args, base_plan):
    if args.force_probe:
        return False
    if args.batch_profile and os.path.exists(args.batch_profile):
        profile = load_batch_profile(args.batch_profile)
        if profile:
            return True
    return False


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
        _ctx.build_plan(args.stage, args.plan_size),
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


def resolve_physical_node_groups(args):
    """One full node per group, with any GPU suffix stripped."""
    items = args.node_groups or load_cluster_node_addrs(args.config)
    seen = []
    for item in items or []:
        for part in str(item).split(","):
            part = part.strip()
            if not part:
                continue
            addr, _ = parse_node_spec(part)
            if addr not in seen:
                seen.append(addr)
    return [[addr] for addr in seen]


def group_slot_key(group):
    return tuple(group)


def should_use_tail_full_nodes(args, remaining, lane_groups, full_node_groups):
    if not getattr(args, "tail_full_nodes", True):
        return False
    if not getattr(args, "gpus_per_lane", None):
        return False
    if not full_node_groups or len(lane_groups) <= len(full_node_groups):
        return False
    threshold = getattr(args, "tail_full_nodes_threshold", None)
    if threshold is None:
        threshold = len(full_node_groups)
    return remaining <= threshold


def active_parallel_groups(args, remaining, lane_groups, full_node_groups):
    if should_use_tail_full_nodes(args, remaining, lane_groups, full_node_groups):
        return full_node_groups, "full-node"
    return lane_groups, "lane"


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
        if not task_running_in_status(status, task_id):
            return
        time.sleep(poll_interval)


def task_running_in_status(status, task_id):
    task_id = str(task_id)
    nodes = status.get("nodes", {})
    for node in nodes.values():
        running_tasks = {str(item) for item in (node.get("running_tasks") or [])}
        if task_id in running_tasks:
            return True
        if str(node.get("status", "")).startswith(f"running:{task_id}"):
            return True
    return False


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
            _ctx.build_plan(args.stage, args.plan_size),
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
            _ctx.build_plan(args.stage, args.plan_size),
            load_batch_profile(args.batch_profile),
        )

    lane_groups = resolve_node_groups(args, "run-parallel")
    full_node_groups = resolve_physical_node_groups(args)
    metrics_dir = getattr(args, "metrics_dir", "runs/metrics")
    if getattr(args, "resume", True):
        done = completed_final_experiments(metrics_dir)
        if done:
            before = len(plan)
            plan = [exp for exp in plan if exp["name"] not in done]
            print(f"resume: skipping {before - len(plan)} experiments with final metrics", flush=True)

    queue = list(plan)
    running = {}
    completed = 0
    schedule_mode = "lane"
    while queue or running:
        remaining = len(queue) + len(running)
        active_groups, new_mode = active_parallel_groups(
            args, remaining, lane_groups, full_node_groups)
        if new_mode != schedule_mode:
            print(
                f"[tail] switching scheduler to {new_mode} "
                f"({remaining} job(s) left, {len(full_node_groups)} full node(s))",
                flush=True,
            )
            schedule_mode = new_mode

        status = pool_status(args.master_addr, args.port)
        for group in active_groups:
            key = group_slot_key(group)
            if key in running or not queue:
                continue
            if not node_group_idle_from_status(status, group):
                continue
            exp = dict(queue.pop(0))
            exp["node_addrs"] = group
            slot_label = schedule_mode
            slot_id = ",".join(group)
            print(
                f"\n[{completed + len(running) + 1}/{len(plan)}] "
                f"{slot_label}={slot_id} {exp['name']}"
            )
            task = submit_exp(args, exp, wait=0)
            running[key] = {
                "task_id": task["task_id"],
                "nodes": group,
                "name": exp["name"],
                "submitted_at": time.time(),
                "schedule_mode": schedule_mode,
            }

        done = []
        status = pool_status(args.master_addr, args.port)
        for key, item in running.items():
            if time.time() - item["submitted_at"] < args.startup_grace:
                continue
            if (
                node_group_idle_from_status(status, item["nodes"])
                and not task_running_in_status(status, item["task_id"])
            ):
                done.append(key)
        for key in done:
            item = running.pop(key)
            completed += 1
            slot_id = ",".join(item["nodes"])
            print(
                f"[done {completed}/{len(plan)}] "
                f"{item.get('schedule_mode', 'lane')}={slot_id} "
                f"{item['name']} task={item['task_id']}"
            )

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


def probe_task_already_done(probe_name, metrics_dir):
    metrics = latest_row_by_experiment(metrics_dir)
    if probe_name in metrics:
        return True
    failures = failure_reports_by_experiment(metrics_dir)
    return probe_name in failures


def seed_quick_probe_state(base_plan, args, probe_targets=None):
    batch_sizes = sorted(set(args.batch_sizes), reverse=True)
    limit_mb = memory_limit_mb(args)
    base_names = {exp["name"] for exp in base_plan}
    target_set = set(probe_targets) if probe_targets is not None else None

    selected = {}
    for name, item in load_batch_profile_meta(args.quick_output).items():
        if name in base_names and (target_set is None or name in target_set):
            selected[name] = {
                "batch_size": item["batch_size"],
                "probe_name": item.get("selected_probe", "quick_csv"),
                "peak_memory_mb": item.get("peak_memory_mb", ""),
                "metrics_file": item.get("metrics_file", ""),
                "source": "quick_csv",
            }

    for base, samples in load_probe_samples_from_metrics(args.metrics_dir, base_names, limit_mb).items():
        if target_set is not None and base not in target_set:
            continue
        if base in selected:
            continue
        best = best_probe_sample(samples)
        if best and best.get("status") == "ok":
            selected[base] = {
                "batch_size": int(best["batch_size"]),
                "probe_name": best.get("probe_name", ""),
                "peak_memory_mb": best.get("peak_memory_mb", ""),
                "metrics_file": best.get("metrics_file", ""),
                "source": "resume_metrics",
            }

    pending = {}
    for exp in base_plan:
        name = exp["name"]
        if target_set is not None and name not in target_set:
            continue
        if name in selected:
            continue
        pending[name] = {
            "exp": exp,
            "batches": list(batch_sizes),
            "done": False,
            "attempts": [],
        }

    return pending, selected


def run_quick_probe(args):
    prepare_probe_run_args(args)
    base_plan = _ctx.build_plan(args.stage, args.plan_size)
    batch_sizes = sorted(set(args.batch_sizes), reverse=True)
    if args.fallback_batch_size is None:
        args.fallback_batch_size = min(batch_sizes)
    args._quick_base_names = [exp["name"] for exp in base_plan]
    node_groups = resolve_node_groups(args, "quick-probe")

    probe_targets = getattr(args, "_probe_targets", None)
    if probe_targets is None and args.max_probe_experiments > 0:
        profile, meta = build_smart_profile(base_plan, args)
        probe_targets = select_probe_targets(base_plan, meta, args)
    elif probe_targets is None:
        probe_targets = [exp["name"] for exp in base_plan]

    pending, selected = seed_quick_probe_state(base_plan, args, probe_targets)
    queue = list(pending.keys())
    running = {}
    attempts = []
    probe_attempts = 0
    started_at = time.time()
    budget_min = effective_probe_budget_min(args)
    deadline = started_at + budget_min * 60 if budget_min > 0 else None
    limit_mb = memory_limit_mb(args)
    max_attempts = max(0, getattr(args, "max_probe_attempts", 0))

    def next_probe():
        while queue:
            base_name = queue.pop(0)
            item = pending[base_name]
            if item["done"] or not item["batches"]:
                continue
            batch_size = item["batches"].pop(0)
            probe_exp = quick_probe_experiment(
                item["exp"], batch_size, args.tune_steps, args.tune_log_interval)
            if getattr(args, "resume", True) and probe_task_already_done(
                    probe_exp["name"], args.metrics_dir):
                item["attempts"].append({
                    "batch_size": batch_size,
                    "status": "resume_skip",
                    "peak_memory_mb": None,
                })
                continue
            return base_name, probe_exp
        return None, None

    total = len(probe_targets) if probe_targets else len(base_plan)
    while queue or running:
        now = time.time()
        attempts_open = max_attempts <= 0 or probe_attempts < max_attempts
        scheduling_open = attempts_open and (deadline is None or now < deadline)
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
                probe_attempts += 1
                print(
                    f"\n[quick {len(selected)}/{total}] attempt={probe_attempts} "
                    f"lane={idx} {base_name} bs={batch_size}",
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
            if (
                node_group_idle_from_status(status, item["nodes"])
                and not task_running_in_status(status, item["task_id"])
            ):
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
                    selected[item["base_name"]]["source"] = "probed"
                print(
                    f"[selected {len(selected)}/{total}] {item['base_name']} "
                    f"bs={item['batch_size']} mem={result.get('peak_memory_mb', '')}",
                    flush=True,
                )
                on_selected = getattr(args, "_on_probe_selected", None)
                if on_selected:
                    on_selected(item["base_name"], selected[item["base_name"]])
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

        if max_attempts > 0 and probe_attempts >= max_attempts and not running:
            queue = []
        if deadline is not None and time.time() >= deadline and not running:
            queue = []
        if deadline is not None and time.time() >= deadline and queue and not running:
            queue = []
        if queue or running:
            time.sleep(args.poll_interval)

    write_quick_outputs(args, selected, attempts)
    missing = total - len(selected)
    print(
        f"quick probe done: selected={len(selected)} missing={missing} "
        f"attempts={probe_attempts} profile={args.output} report={args.report_output}",
        flush=True,
    )
    return selected


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
    return bool(name) and name.startswith(_ctx.prefixes)


def is_final_metrics_row(row):
    name = row.get("experiment_name", "")
    metrics_file = row.get("metrics_file", "")
    if not _ctx.is_regional_experiment(name):
        return False
    return os.path.basename(metrics_file) == f"{name}.csv"


def is_completed_experiment(row):
    if not is_final_metrics_row(row):
        return False
    max_steps_raw = row.get("max_steps", "")
    if max_steps_raw in ("", None):
        # Legacy metrics rows without max_steps: keep old resume behavior.
        return True
    max_steps = parse_float(row, "max_steps", 0)
    global_step = parse_float(row, "global_step", 0)
    if max_steps > 0:
        return global_step >= max_steps
    return True


_ctx.is_regional_experiment = is_regional_experiment
_ctx.is_final_metrics_row = is_final_metrics_row


def recommend_batches(args):
    rows = latest_rows(args.metrics_dir)
    probes = []
    for row in rows:
        base = batch_probe_base_name(row.get("experiment_name", ""))
        if not base or not _ctx.is_regional_experiment(base):
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


def run_streaming_probe_and_parallel(args, base_plan, profile, profile_meta, probe_targets):
    prepare_probe_run_args(args)
    limit_mb = memory_limit_mb(args)
    done = completed_final_experiments(args.metrics_dir)
    run_plan = apply_batch_profile(base_plan, profile)
    run_queue = [exp for exp in run_plan if exp["name"] not in done]

    node_groups = resolve_node_groups(args, "probe-and-run")
    full_node_groups = resolve_physical_node_groups(args)
    max_probe_lanes = min(max(0, args.max_probe_lanes), len(node_groups))
    probe_lane_ids = set(range(max_probe_lanes))

    pending, selected = seed_quick_probe_state(base_plan, args, probe_targets)
    queue = list(pending.keys())
    running_probes = {}
    running_runs = {}
    completed_runs = 0
    total_runs = len(run_queue)
    run_schedule_mode = "lane"
    probe_attempts = 0
    attempts = []
    started_at = time.time()
    budget_min = effective_probe_budget_min(args)
    deadline = started_at + budget_min * 60 if budget_min > 0 else None
    max_attempts = max(0, args.max_probe_attempts)

    args.output = args.quick_output
    args._quick_base_names = [exp["name"] for exp in base_plan]

    def on_probe_selected(base_name, item):
        profile[base_name] = int(item["batch_size"])
        profile_meta[base_name] = {
            "batch_size": int(item["batch_size"]),
            "source": "probed",
            "peak_memory_mb": item.get("peak_memory_mb", ""),
            "metrics_file": item.get("metrics_file", ""),
            "selected_probe": item.get("probe_name", "probed"),
        }
        write_merged_batch_profile(args.batch_profile, base_plan, profile, profile_meta, args)

    args._on_probe_selected = on_probe_selected

    def next_probe():
        while queue:
            base_name = queue.pop(0)
            item = pending[base_name]
            if item["done"] or not item["batches"]:
                continue
            batch_size = item["batches"].pop(0)
            probe_exp = quick_probe_experiment(
                item["exp"], batch_size, args.tune_steps, args.tune_log_interval)
            if args.resume and probe_task_already_done(probe_exp["name"], args.metrics_dir):
                item["attempts"].append({
                    "batch_size": batch_size,
                    "status": "resume_skip",
                    "peak_memory_mb": None,
                })
                continue
            return base_name, probe_exp
        return None, None

    total_probe = len(probe_targets)
    write_merged_batch_profile(args.batch_profile, base_plan, profile, profile_meta, args)
    print(
        f"[streaming] starting formal runs ({total_runs}) while probing "
        f"{len(probe_targets)} representatives on {max_probe_lanes} lane(s)",
        flush=True,
    )

    while run_queue or running_runs or queue or running_probes:
        now = time.time()
        attempts_open = max_attempts <= 0 or probe_attempts < max_attempts
        probe_open = attempts_open and (deadline is None or now < deadline)
        status = pool_status(args.master_addr, args.port)

        run_remaining = len(run_queue) + len(running_runs)
        run_groups, new_run_mode = active_parallel_groups(
            args, run_remaining, node_groups, full_node_groups)
        if new_run_mode != run_schedule_mode:
            print(
                f"[tail] switching formal-run scheduler to {new_run_mode} "
                f"({run_remaining} job(s) left, {len(full_node_groups)} full node(s))",
                flush=True,
            )
            run_schedule_mode = new_run_mode

        for group in run_groups:
            key = group_slot_key(group)
            if key in running_runs or not run_queue:
                continue
            if not node_group_idle_from_status(status, group):
                continue
            exp = dict(run_queue.pop(0))
            exp["node_addrs"] = group
            slot_id = ",".join(group)
            print(
                f"\n[run {completed_runs + len(running_runs) + 1}/{total_runs}] "
                f"{run_schedule_mode}={slot_id} {exp['name']} "
                f"bs={exp['args'].get('batch_size', '')}",
                flush=True,
            )
            task = submit_exp(args, exp, wait=0)
            running_runs[key] = {
                "task_id": task["task_id"],
                "nodes": group,
                "name": exp["name"],
                "submitted_at": time.time(),
                "schedule_mode": run_schedule_mode,
            }

        for idx, group in enumerate(node_groups):
            if group_slot_key(group) in running_runs or idx in running_probes:
                continue
            if not node_group_idle_from_status(status, group):
                continue

            if probe_open and (idx in probe_lane_ids or not run_queue):
                base_name, exp = next_probe()
                if exp is None:
                    continue
                exp["node_addrs"] = group
                batch_size = exp["args"]["batch_size"]
                probe_attempts += 1
                print(
                    f"\n[quick {len(selected)}/{total_probe}] attempt={probe_attempts} "
                    f"lane={idx} {base_name} bs={batch_size}",
                    flush=True,
                )
                task = submit_exp(args, exp, wait=0)
                running_probes[idx] = {
                    "task_id": task["task_id"],
                    "nodes": group,
                    "name": exp["name"],
                    "base_name": base_name,
                    "batch_size": batch_size,
                    "submitted_at": time.time(),
                }

        done_runs = []
        status = pool_status(args.master_addr, args.port)
        for key, item in running_runs.items():
            if time.time() - item["submitted_at"] < args.startup_grace:
                continue
            if (
                node_group_idle_from_status(status, item["nodes"])
                and not task_running_in_status(status, item["task_id"])
            ):
                done_runs.append(key)
        for key in done_runs:
            item = running_runs.pop(key)
            completed_runs += 1
            slot_id = ",".join(item["nodes"])
            print(
                f"[done run {completed_runs}/{total_runs}] "
                f"{item.get('schedule_mode', 'lane')}={slot_id} "
                f"{item['name']} task={item['task_id']}",
                flush=True,
            )

        done_probes = []
        status = pool_status(args.master_addr, args.port)
        for idx, item in running_probes.items():
            if time.time() - item["submitted_at"] < args.startup_grace:
                continue
            if (
                node_group_idle_from_status(status, item["nodes"])
                and not task_running_in_status(status, item["task_id"])
            ):
                done_probes.append(idx)

        for idx in done_probes:
            item = running_probes.pop(idx)
            result = probe_result_after_settle(
                item["name"], args.metrics_dir, args.metrics_settle_seconds)
            peak_memory_mb = result_peak_memory_mb(result)
            status_name = result["status"]
            if status_name == "ok" and peak_memory_mb is not None and peak_memory_mb > limit_mb:
                status_name = "over_memory"
                result = dict(result)
                result["status"] = status_name
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
                "status": status_name,
                "peak_memory_mb": peak_memory_mb,
            })
            if status_name == "ok":
                selected_item = {
                    "batch_size": item["batch_size"],
                    "probe_name": item["name"],
                    "peak_memory_mb": result.get("peak_memory_mb", ""),
                    "tokens_per_sec": result.get("tokens_per_sec", ""),
                    "metrics_file": result.get("metrics_file", ""),
                    "source": "probed",
                }
                refine_batch = maybe_refine_quick_probe(
                    base, item["batch_size"], peak_memory_mb, limit_mb, args)
                if refine_batch is not None:
                    selected[item["base_name"]] = selected_item
                    base["done"] = False
                    base["batches"] = [refine_batch]
                    queue.append(item["base_name"])
                else:
                    base["done"] = True
                    selected[item["base_name"]] = selected_item
                    on_probe_selected(item["base_name"], selected_item)
            else:
                if status_name in {"oom", "over_memory"}:
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

            write_quick_outputs(args, selected, attempts)

        if max_attempts > 0 and probe_attempts >= max_attempts and not running_probes:
            queue = []
        if deadline is not None and time.time() >= deadline and not running_probes:
            queue = []

        if run_queue or running_runs or queue or running_probes:
            time.sleep(args.poll_interval)

    write_quick_outputs(args, selected, attempts)
    write_merged_batch_profile(args.batch_profile, base_plan, profile, profile_meta, args)
    print(
        f"[streaming] done: runs={completed_runs}/{total_runs} "
        f"probe_selected={len(selected)}/{total_probe} probe_attempts={probe_attempts}",
        flush=True,
    )


def run_only(args):
    prepare_probe_run_args(args)
    if not args.batch_profile or not os.path.exists(args.batch_profile):
        raise SystemExit(
            f"run-only requires an existing --batch_profile; not found: {args.batch_profile}"
        )
    base_plan = _ctx.build_plan(args.stage, args.plan_size)
    profile = load_batch_profile(args.batch_profile)
    print_execution_plan(
        args, base_plan, mode="run-only", profile=profile,
        profile_meta=load_batch_profile_meta(args.batch_profile),
    )
    args.batch_tune = False
    run_parallel(args)


def run_probe_and_parallel(args):
    prepare_probe_run_args(args)
    base_plan = _ctx.build_plan(args.stage, args.plan_size)

    if should_skip_probe(args, base_plan):
        profile = load_batch_profile(args.batch_profile)
        print_execution_plan(
            args, base_plan, mode="run-only (batch_profile exists)",
            profile=profile, profile_meta=load_batch_profile_meta(args.batch_profile),
        )
        print(
            f"[probe-and-run] skipping probe; using existing profile {args.batch_profile}",
            flush=True,
        )
        args.batch_tune = False
        run_parallel(args)
        return

    profile, profile_meta = build_smart_profile(base_plan, args)
    probe_targets = select_probe_targets(base_plan, profile_meta, args)
    print_execution_plan(
        args, base_plan, mode="probe-and-run", probe_targets=probe_targets,
        profile=profile, profile_meta=profile_meta,
    )

    write_merged_batch_profile(args.batch_profile, base_plan, profile, profile_meta, args)
    args.batch_tune = False

    if args.streaming:
        run_streaming_probe_and_parallel(args, base_plan, profile, profile_meta, probe_targets)
        return

    quick_output = args.quick_output
    batch_profile = args.batch_profile
    args._probe_targets = probe_targets
    print(f"[probe-and-run] limited probe -> {quick_output}", flush=True)
    args.output = quick_output
    run_quick_probe(args)

    print(f"[probe-and-run] merging recommendations -> {batch_profile}", flush=True)
    profile, profile_meta = build_smart_profile(base_plan, args)
    write_merged_batch_profile(batch_profile, base_plan, profile, profile_meta, args)

    print(f"[probe-and-run] starting run-parallel with {batch_profile}", flush=True)
    args.batch_profile = batch_profile
    run_parallel(args)


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
    plan = apply_batch_profile(_ctx.build_plan(args.stage, args.plan_size), profile)
    write_manifest(plan, args.output, args.config, args.master_addr, args.port, args.wait)
    print(f"wrote final execution plan: {args.output}")


def summarize(args):
    stage_prefix = getattr(args, "stage", "all")

    def stage_match(name):
        return stage_prefix == "all" or name.startswith(f"{stage_prefix}_")

    success_rows = [
        row for row in latest_rows(args.metrics_dir)
        if _ctx.is_final_metrics_row(row) and stage_match(row.get("experiment_name", ""))
    ]
    for row in success_rows:
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
    success_rows.sort(key=lambda r: r.get("experiment_name", ""))

    failures = failure_reports_by_experiment(args.metrics_dir)
    fail_rows = [
        report for name, report in failures.items()
        if stage_match(name)
    ]
    fail_rows.sort(key=lambda r: r.get("experiment_name", ""))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    success_fields = [
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
        writer = csv.DictWriter(f, fieldnames=success_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(success_rows)
    print(f"wrote summary ({len(success_rows)} ok): {args.output}")

    fail_output = args.output.replace(".csv", "_fail.csv")
    fail_fields = [
        "experiment_name", "status", "rank", "error_type", "error",
        "peak_memory_mb", "world_size", "global_step",
        "failure_file",
    ]
    with open(fail_output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fail_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(fail_rows)
    print(f"wrote fail summary ({len(fail_rows)} failed): {fail_output}")


_ctx.summarize_fn = summarize


def parse_args():
    parser = argparse.ArgumentParser(description="CTM implementation-validation experiment plan helper")
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--stage", default="all", choices=_ctx.stages)
    common.add_argument("--config", default=_ctx.cluster_config)
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
    p.add_argument("--output", default=None)
    p.add_argument("--batch_profile", default=None)
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
    p.add_argument("--metrics_dir", default="runs/metrics")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--batch_tune", action="store_true")
    p.add_argument("--batch_sizes", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12])
    p.add_argument("--tune_steps", type=int, default=80)
    p.add_argument("--tune_log_interval", type=int, default=20)
    p.add_argument("--node_groups", nargs="*", default=None,
                   help="Node groups for parallel lanes; omitted means NODE_ADDRS from --config.")
    p.add_argument("--gpus_per_lane", type=int, default=2,
                   help="Split bare single-node groups into GPU lanes; default 2 gives 16 two-GPU lanes for four 8-GPU nodes.")
    p.add_argument("--gpus_per_node", type=int, default=8)
    p.add_argument("--tail_full_nodes", action=argparse.BooleanOptionalAction, default=True,
                   help="When few jobs remain, schedule each on a full node instead of a GPU lane.")
    p.add_argument("--tail_full_nodes_threshold", type=int, default=None,
                   help="Switch to full-node scheduling when queue+running <= this; default=number of physical nodes.")
    p.set_defaults(func=run_parallel)

    probe_common = argparse.ArgumentParser(add_help=False)
    probe_common.add_argument("--startup_grace", type=float, default=60.0)
    probe_common.add_argument("--poll_interval", type=float, default=10.0)
    probe_common.add_argument("--batch_sizes", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12])
    probe_common.add_argument("--tune_steps", type=int, default=3)
    probe_common.add_argument("--tune_log_interval", type=int, default=1)
    probe_common.add_argument("--time_limit_min", type=float, default=15.0)
    probe_common.add_argument("--probe_time_budget_min", type=float, default=45.0,
                              help="Hard cap on probe phase wall time; 0 disables.")
    probe_common.add_argument("--max_probe_experiments", type=int, default=32,
                              help="Probe at most this many base experiments (family representatives); 0 disables live probe.")
    probe_common.add_argument("--max_probe_attempts", type=int, default=96,
                              help="Hard cap on total probe task submissions across all experiments; 0 disables.")
    probe_common.add_argument("--max_probe_lanes", type=int, default=2,
                              help="Lanes reserved for probe tasks during streaming; remaining lanes run formal jobs.")
    probe_common.add_argument("--metrics_settle_seconds", type=float, default=45.0,
                              help="Wait this long for metrics/failure files after a lane becomes idle.")
    probe_common.add_argument("--oom_backoff_ratio", type=float, default=0.67,
                              help="After OOM/over_memory, only try remaining batches <= current_batch * ratio; set 1.0 to disable.")
    probe_common.add_argument("--refine_bracket_after_ok", action=argparse.BooleanOptionalAction, default=True,
                              help="After a lower batch succeeds below an OOM/over_memory batch, try one skipped middle batch when memory looks safe.")
    probe_common.add_argument("--refine_memory_margin", type=float, default=0.97,
                              help="Only run bracket refinement when linear memory estimate is below limit * margin.")
    probe_common.add_argument("--metrics_dir", default="runs/metrics")
    probe_common.add_argument("--target_memory_gb", type=float, default=80.0)
    probe_common.add_argument("--memory_util", type=float, default=0.90)
    probe_common.add_argument("--fallback_batch_size", type=int, default=None,
                              help="Batch size used for experiments not resolved before limits; default=min(batch_sizes).")
    probe_common.add_argument("--node_groups", nargs="*", default=None,
                              help="Node groups for quick probe lanes; omitted means NODE_ADDRS from --config.")
    probe_common.add_argument("--gpus_per_lane", type=int, default=2,
                              help="Split bare single-node groups into GPU lanes; default 2 gives 16 two-GPU lanes for four 8-GPU nodes.")
    probe_common.add_argument("--gpus_per_node", type=int, default=8)
    probe_common.add_argument("--tail_full_nodes", action=argparse.BooleanOptionalAction, default=True,
                              help="When few formal jobs remain, schedule each on a full node instead of a GPU lane.")
    probe_common.add_argument("--tail_full_nodes_threshold", type=int, default=None,
                              help="Switch to full-node scheduling when queue+running <= this; default=number of physical nodes.")
    probe_common.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                              help="Reuse existing qp__ metrics and final og metrics instead of resubmitting.")

    p = sub.add_parser("quick-probe", parents=[common, probe_common])
    p.add_argument("--output", default=None)
    p.add_argument("--report_output", default=None)
    p.set_defaults(func=run_quick_probe)

    p = sub.add_parser("probe-and-run", parents=[common, probe_common])
    p.add_argument("--quick_output", default=None)
    p.add_argument("--report_output", default=None)
    p.add_argument("--batch_profile", default=None)
    p.add_argument("--force_probe", action="store_true",
                   help="Run probe even when batch_profile already exists.")
    p.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True,
                   help="Start formal runs immediately using inferred batches while limited probe runs in background.")
    p.set_defaults(func=run_probe_and_parallel)

    p = sub.add_parser("run-only", parents=[common])
    p.add_argument("--startup_grace", type=float, default=20.0)
    p.add_argument("--poll_interval", type=float, default=30.0)
    p.add_argument("--batch_profile", default=None)
    p.add_argument("--metrics_dir", default="runs/metrics")
    p.add_argument("--node_groups", nargs="*", default=None)
    p.add_argument("--gpus_per_lane", type=int, default=2)
    p.add_argument("--gpus_per_node", type=int, default=8)
    p.add_argument("--tail_full_nodes", action=argparse.BooleanOptionalAction, default=True,
                   help="When few jobs remain, schedule each on a full node instead of a GPU lane.")
    p.add_argument("--tail_full_nodes_threshold", type=int, default=None,
                   help="Switch to full-node scheduling when queue+running <= this; default=number of physical nodes.")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.set_defaults(func=run_only)

    p = sub.add_parser("summarize")
    p.add_argument("--stage", default="all",
                   choices=list(_ctx.stages) + ["all"],
                   help="Export only experiments matching this stage prefix (e.g. dr02); 'all' for no filter.")
    p.add_argument("--metrics_dir", default="runs/metrics")
    p.add_argument("--output", default=None)
    p.set_defaults(func=_ctx.summarize_fn)

    p = sub.add_parser("recommend-batches")
    p.add_argument("--metrics_dir", default="runs/metrics")
    p.add_argument("--output", default=None)
    p.add_argument("--target_memory_gb", type=float, default=80.0)
    p.add_argument("--memory_util", type=float, default=0.90)
    p.set_defaults(func=recommend_batches)

    p = sub.add_parser("batch-report")
    p.add_argument("--stage", default="all", choices=_ctx.stages)
    p.add_argument("--plan_size", default="full", choices=PLAN_SIZES)
    p.add_argument("--metrics_dir", default="runs/metrics")
    p.add_argument("--output", default="runs/metrics/impl_validation_batch_probe_report.csv")
    p.add_argument("--batch_sizes", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12])
    p.add_argument("--tune_steps", type=int, default=80)
    p.add_argument("--tune_log_interval", type=int, default=20)
    p.set_defaults(func=batch_report)

    args = parser.parse_args()
    if getattr(args, "output", None) is None:
        if args.cmd == "summarize":
            args.output = metrics_path("summary.csv")
        elif args.cmd == "recommend-batches":
            args.output = metrics_path("batch_profile.csv")
        elif args.cmd == "final-plan":
            args.output = f"runs/experiment_plans/{_ctx.metrics_prefix}_final_plan.csv"
        elif args.cmd == "quick-probe":
            args.output = metrics_path("batch_profile_quick.csv")
            args.report_output = metrics_path("quick_probe_report.csv")
    if args.cmd == "probe-and-run":
        if getattr(args, "quick_output", None) is None:
            args.quick_output = metrics_path("batch_profile_quick.csv")
        if getattr(args, "report_output", None) is None:
            args.report_output = metrics_path("quick_probe_report.csv")
        if getattr(args, "batch_profile", None) is None:
            args.batch_profile = metrics_path("batch_profile.csv")
    if args.cmd == "run-only":
        if getattr(args, "batch_profile", None) is None:
            args.batch_profile = metrics_path("batch_profile.csv")
    if args.cmd in ("run", "run-parallel", "quick-probe", "probe-and-run", "run-only"):
        if args.master_addr is None:
            args.master_addr = "11.131.210.78"
        if args.port is None:
            args.port = 8765
    if args.cmd == "final-plan" and getattr(args, "batch_profile", None) is None:
        args.batch_profile = metrics_path("batch_profile.csv")
    return args


if __name__ == "__main__":
    args = parse_args()
    args.func(args)
