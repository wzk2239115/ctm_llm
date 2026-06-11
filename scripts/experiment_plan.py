#!/usr/bin/env python3
import argparse
import csv
import os

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import experiment_plan_impl_validation as base


PLAN_STAGES = ["smoke", "compass", "ticks", "elf", "cells", "ablations", "all"]
PLAN_PREFIXES = tuple(f"{s}_" for s in PLAN_STAGES if s != "all")

METRICS_PREFIX = "plan"


def metrics_path(name):
    return f"runs/metrics/{METRICS_PREFIX}_{name}"


def build_plan(stage, plan_size="full"):
    plan = []
    if stage in ("smoke", "all"):
        smoke = dict(base.BASE_ARGS, max_steps=100, num_hidden_layers=4,
                     hidden_size=512, d_model=256, d_input=128, heads=4,
                     n_synch_out=256, n_synch_action=256,
                     iterations=3, max_seq_len=256, synapse_depth=2)
        plan.append(base.experiment(
            "s00_transformer_smoke",
            "Verify standard Transformer baseline path and metrics logging.",
            base.merge_args(**smoke, model_type="transformer")))
        plan.append(base.experiment(
            "s00_ctm_smoke",
            "Verify CTM path and per-tick metrics logging.",
            base.merge_args(**smoke, model_type="ctm")))

    if stage in ("compass", "all"):
        scales = [
            ("12l_h640", 12, 640, 384, 192, 8),
            ("16l_h768", 16, 768, 512, 256, 8),
        ]
        if base.include_plan_size(plan_size, "full"):
            scales.append(("24l_h896", 24, 896, 640, 320, 8))
        if base.include_plan_size(plan_size, "wide"):
            scales.append(("32l_h1024", 32, 1024, 768, 384, 8))
        for tag, layers, hidden, d_model, d_input, heads in scales:
            common = {
                "num_hidden_layers": layers,
                "hidden_size": hidden,
                "d_model": d_model,
                "d_input": d_input,
                "heads": heads,
                "n_synch_out": d_model,
                "n_synch_action": d_model,
            }
            plan.append(base.experiment(
                f"s01_transformer_{tag}",
                "Transformer scale baseline for CTM cost/loss comparison.",
                base.merge_args(model_type="transformer", iterations=1, **common)))
            plan.append(base.experiment(
                f"s01_ctm_{tag}_tick4",
                "CTM at matched outer layer/hidden scale with 4 ticks.",
                base.merge_args(model_type="ctm", iterations=4, **common)))
        plan.append(base.experiment(
            "s01_ctm_16l_tick1",
            "CTM with one tick to isolate cell-block overhead from iterative thinking.",
            base.merge_args(model_type="ctm", iterations=1)))

    if stage in ("ticks", "all"):
        tick_values = [1, 2, 4, 8, 16]
        if base.include_plan_size(plan_size, "full"):
            tick_values = [1, 2, 3, 4, 6, 8, 12, 16]
        if base.include_plan_size(plan_size, "wide"):
            tick_values += [24, 32]
        for ticks in tick_values:
            plan.append(base.experiment(
                f"s02_ctm_tick{ticks}",
                "Measure loss/cost curve as maximum CTM ticks increase.",
                base.merge_args(model_type="ctm", iterations=ticks)))
        if base.include_plan_size(plan_size, "full"):
            for mode in ["mean", "last", "min_conf"]:
                plan.append(base.experiment(
                    f"s02_ctm_tick8_loss_{mode}",
                    "Compare per-tick loss aggregation to see whether later ticks specialize.",
                    base.merge_args(model_type="ctm", iterations=8, tick_loss_mode=mode)))
        for key, overrides in [
            ("halt_conf_t8_c0", {"iterations": 8, "tick_halt_mode": "confidence",
                                 "tick_compute_weight": 0.0}),
            ("halt_conf_t8_c01", {"iterations": 8, "tick_halt_mode": "confidence",
                                  "tick_compute_weight": 0.01}),
            ("halt_conf_t16_c01", {"iterations": 16, "tick_halt_mode": "confidence",
                                   "tick_compute_weight": 0.01}),
            ("halt_thresh_t16", {"iterations": 16, "tick_halt_mode": "threshold",
                                 "tick_halt_threshold": 0.65}),
        ]:
            plan.append(base.experiment(
                f"s02_ctm_{key}",
                "Train CTM with adaptive tick halting signals and optional compute penalty.",
                base.merge_args(model_type="ctm", **overrides)))
        if base.include_plan_size(plan_size, "full"):
            for temp in [0.15, 0.35, 0.60]:
                plan.append(base.experiment(
                    base.ctm_name("s02_ctm_halt_conf_t16", temp=temp),
                    "Sweep confidence-halting temperature for natural early/late tick separation.",
                    base.merge_args(
                        model_type="ctm",
                        iterations=16,
                        tick_halt_mode="confidence",
                        tick_halt_temperature=temp,
                        tick_compute_weight=0.01,
                    )))
            for threshold in [0.55, 0.70, 0.85]:
                plan.append(base.experiment(
                    base.ctm_name("s02_ctm_halt_thresh_t16", th=threshold),
                    "Sweep hard confidence threshold used to estimate effective thinking depth.",
                    base.merge_args(
                        model_type="ctm",
                        iterations=16,
                        tick_halt_mode="threshold",
                        tick_halt_threshold=threshold,
                    )))

    if stage in ("elf", "all"):
        for key, overrides in [
            ("next", {"elf_horizon_mode": "none"}),
            ("linear_h4", {"elf_horizon_mode": "linear", "elf_max_horizon": 4}),
            ("pow2_h4", {"elf_horizon_mode": "pow2", "elf_max_horizon": 4}),
            ("linear_h8", {"elf_horizon_mode": "linear", "elf_max_horizon": 8,
                           "iterations": 8}),
            ("linear_h4_improve", {"elf_horizon_mode": "linear",
                                   "elf_max_horizon": 4,
                                   "tick_improve_weight": 0.1}),
        ]:
            plan.append(base.experiment(
                f"s03_elf_{key}",
                "Give different ticks different prediction horizons and measure whether tick roles separate.",
                base.merge_args(model_type="ctm", **overrides)))
        if base.include_plan_size(plan_size, "full"):
            for ticks in [8, 12, 16]:
                for horizon in [4, 8]:
                    plan.append(base.experiment(
                        f"s03_elf_linear_t{ticks}_h{horizon}",
                        "Cross ELF horizon with more internal ticks to test multi-token prediction utility.",
                        base.merge_args(
                            model_type="ctm",
                            iterations=ticks,
                            elf_horizon_mode="linear",
                            elf_max_horizon=horizon,
                        )))
            for weight in [0.03, 0.1, 0.3]:
                plan.append(base.experiment(
                    base.ctm_name("s03_elf_linear_h8_improve", w=weight),
                    "Sweep improvement regularization so later ticks earn their extra compute.",
                    base.merge_args(
                        model_type="ctm",
                        iterations=8,
                        elf_horizon_mode="linear",
                        elf_max_horizon=8,
                        tick_improve_weight=weight,
                    )))
        if base.include_plan_size(plan_size, "wide"):
            for mode in ["pow2", "linear"]:
                plan.append(base.experiment(
                    f"s03_elf_{mode}_t16_h16",
                    "Wide ELF run with long horizon and 16 ticks.",
                    base.merge_args(
                        model_type="ctm",
                        iterations=16,
                        elf_horizon_mode=mode,
                        elf_max_horizon=16,
                    )))

    if stage in ("cells", "all"):
        cell_grid = [
            (256, 4, 10, 3, "none", 256),
            (512, 4, 10, 3, "none", 512),
            (1024, 2, 8, 2, "none", 1024),
            (1024, 2, 8, 2, "topk", 512),
            (1536, 1, 6, 1, "none", 1536),
            (1536, 1, 6, 1, "topk", 512),
            (2048, 1, 6, 1, "topk", 512),
        ]
        if base.include_plan_size(plan_size, "full"):
            cell_grid += [
                (768, 3, 10, 2, "none", 768),
                (768, 3, 10, 2, "topk", 384),
                (1024, 1, 6, 1, "topk", 256),
                (1024, 2, 8, 2, "topk", 256),
                (1536, 1, 6, 1, "topk", 384),
                (2048, 1, 6, 1, "topk", 256),
                (2048, 1, 4, 1, "topk", 512),
            ]
        if base.include_plan_size(plan_size, "wide"):
            cell_grid += [
                (3072, 1, 4, 1, "topk", 256),
                (3072, 1, 4, 1, "topk", 512),
                (4096, 1, 4, 1, "topk", 512),
            ]
        for d_model, mem_h, mem_len, depth, sparsity, topk in cell_grid:
            sparse_tag = f"{sparsity}{topk}" if sparsity != "none" else "dense"
            plan.append(base.experiment(
                f"s04_cells_d{d_model}_mh{mem_h}_m{mem_len}_sd{depth}_{sparse_tag}",
                "Trade larger cell count against smaller per-cell memory/synapse models.",
                base.merge_args(
                    model_type="ctm",
                    d_model=d_model,
                    n_synch_out=d_model,
                    n_synch_action=d_model,
                    memory_hidden_dims=mem_h,
                    memory_length=mem_len,
                    synapse_depth=depth,
                    cell_sparsity_mode=sparsity,
                    cell_topk=topk,
                    iterations=4,
                )))
        if base.include_plan_size(plan_size, "full"):
            for rescale in [0, 1]:
                plan.append(base.experiment(
                    f"s04_cells_topk512_rescale{rescale}",
                    "Check whether sparse activation rescaling stabilizes top-k cells.",
                    base.merge_args(
                        model_type="ctm",
                        d_model=1536,
                        n_synch_out=1536,
                        n_synch_action=1536,
                        memory_hidden_dims=1,
                        memory_length=6,
                        synapse_depth=1,
                        cell_sparsity_mode="topk",
                        cell_topk=512,
                        cell_sparsity_rescale=rescale,
                    )))

    if stage in ("ablations", "all"):
        for key, overrides in [
            ("no_selfcond", {"self_cond": 0}),
            ("no_cross_state", {"cross_layer_state": 0}),
            ("shallow_synapse", {"synapse_depth": 1}),
            ("short_memory", {"memory_length": 6}),
            ("tiny_nlm", {"memory_hidden_dims": 1, "deep_nlms": 1}),
        ]:
            plan.append(base.experiment(
                f"s05_{key}",
                "Ablate CTM details to find elegant low-cost simplifications.",
                base.merge_args(model_type="ctm", iterations=4, **overrides)))
        if base.include_plan_size(plan_size, "full"):
            for key, overrides in [
                ("no_selfcond_no_cross", {"self_cond": 0, "cross_layer_state": 0}),
                ("dinput128_heads4", {"d_input": 128, "heads": 4}),
                ("dinput384_heads8", {"d_input": 384, "heads": 8}),
                ("memlen14", {"memory_length": 14}),
                ("synapse2_mh2", {"synapse_depth": 2, "memory_hidden_dims": 2}),
                ("deep_nlms0", {"deep_nlms": 0}),
            ]:
                plan.append(base.experiment(
                    f"s05_{key}",
                    "Search for low-cost CTM simplifications and sensitivity points.",
                    base.merge_args(model_type="ctm", iterations=4, **overrides)))
        if base.include_plan_size(plan_size, "wide"):
            for key, overrides in [
                ("short_memory_sparse", {
                    "memory_length": 4,
                    "memory_hidden_dims": 1,
                    "synapse_depth": 1,
                    "d_model": 1536,
                    "n_synch_out": 1536,
                    "n_synch_action": 1536,
                    "cell_sparsity_mode": "topk",
                    "cell_topk": 384,
                }),
                ("tick8_simple_cell", {
                    "iterations": 8,
                    "memory_hidden_dims": 1,
                    "synapse_depth": 1,
                }),
            ]:
                plan.append(base.experiment(
                    f"s05_{key}",
                    "Wide ablation for sparse/simple CTM variants.",
                    base.merge_args(model_type="ctm", **overrides)))

    return base.validate_plan(plan)


def summarize(args):
    stage_prefix = getattr(args, "stage", "all")
    rows = base.latest_rows(args.metrics_dir)
    if stage_prefix != "all":
        rows = [r for r in rows if r.get("experiment_name", "").startswith(f"{stage_prefix}_")]
    rows.sort(key=lambda r: r.get("experiment_name", ""))
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fields = [
        "experiment_name", "model_type", "loss", "tokens_per_sec",
        "peak_memory_mb", "best_tick", "conf_tick", "tick_count",
        "effective_tick", "active_cell_fraction",
        "losses_per_tick", "certainties_per_tick",
        "hidden_size", "num_hidden_layers", "d_model", "d_input",
        "iterations", "memory_length", "memory_hidden_dims", "deep_nlms",
        "synapse_depth", "tick_loss_mode", "elf_horizon_mode",
        "elf_max_horizon", "tick_improve_weight", "tick_improve_margin",
        "tick_halt_mode", "tick_halt_threshold", "tick_halt_temperature",
        "tick_compute_weight", "cell_sparsity_mode", "cell_topk",
        "cell_sparsity_rescale",
        "self_cond", "cross_layer_state", "global_step",
        "metrics_file",
    ]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote summary: {args.output}")


def is_plan_experiment(name):
    return bool(name) and name.startswith(PLAN_PREFIXES)


def is_final_metrics_row(row):
    name = row.get("experiment_name", "")
    metrics_file = row.get("metrics_file", "")
    if not is_plan_experiment(name):
        return False
    return os.path.basename(metrics_file) == f"{name}.csv"


base.configure_plan_defaults(
    metrics_prefix=METRICS_PREFIX,
    cluster_config="infra/clusters/h100_4nodes.env",
    dispatch_block_sparse=False,
    build_plan=build_plan,
    summarize_fn=summarize,
    is_regional_experiment=is_plan_experiment,
    is_final_metrics_row=is_final_metrics_row,
    stages=tuple(PLAN_STAGES),
    prefixes=PLAN_PREFIXES,
)


def normalize_default_outputs(args):
    remap = {
        "runs/experiment_plans/impl_validation_plan.csv":
            "runs/experiment_plans/experiment_plan.csv",
        "runs/experiment_plans/impl_validation_batch_tune_plan.csv":
            "runs/experiment_plans/experiment_plan_batch_tune_plan.csv",
        "runs/metrics/impl_validation_summary.csv":
            "runs/metrics/plan_summary.csv",
        "runs/metrics/impl_validation_batch_profile.csv":
            "runs/metrics/plan_batch_profile.csv",
        "runs/metrics/impl_validation_batch_profile_quick.csv":
            "runs/metrics/plan_batch_profile_quick.csv",
        "runs/metrics/impl_validation_quick_probe_report.csv":
            "runs/metrics/plan_quick_probe_report.csv",
        "runs/metrics/impl_validation_batch_probe_report.csv":
            "runs/metrics/plan_batch_probe_report.csv",
    }
    for attr in ("output", "report_output", "batch_profile", "quick_output"):
        val = getattr(args, attr, None)
        if val and val in remap:
            setattr(args, attr, remap[val])


if __name__ == "__main__":
    args = base.parse_args()
    normalize_default_outputs(args)
    args.func(args)
