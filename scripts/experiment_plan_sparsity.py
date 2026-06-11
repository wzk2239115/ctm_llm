#!/usr/bin/env python3
import csv
import math
import os

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import experiment_plan_impl_validation as base


SPARSITY_STAGES = (
    "sp00",
    "sp01",
    "sp02",
    "sp03",
    "sp04",
    "sp05",
    "all",
)
SPARSITY_PREFIXES = tuple(f"{stage}_" for stage in SPARSITY_STAGES if stage != "all")

METRICS_PREFIX = "sparsity"


def metrics_path(name):
    return f"runs/metrics/{METRICS_PREFIX}_{name}"


def build_plan(stage, plan_size="full"):
    plan = []

    if stage in ("sp00", "all"):
        base.add_sparse_experiment(
            plan,
            "sp00_sparse_d512_dense",
            "Smoke-check dense sparse-plan CTM path and metrics.",
            512,
            topk=512,
            max_steps=100,
        )
        base.add_sparse_experiment(
            plan,
            "sp00_sparse_d512_topk256",
            "Smoke-check top-k cell sparsity path and active-cell metrics.",
            512,
            topk=256,
            max_steps=100,
        )

    if stage in ("sp01", "all"):
        d_values = [256, 512, 1024] if plan_size == "core" else [256, 384, 512, 768, 1024]
        for d_model in d_values:
            base.add_sparse_experiment(
                plan,
                f"sp01_d{d_model}_dense",
                "Measure dense cell-size curve while shrinking/expanding cell volume.",
                d_model,
                topk=d_model,
            )
        for d_model in d_values:
            base.add_sparse_experiment(
                plan,
                f"sp01_d{d_model}_topk{d_model // 2}",
                "Measure half-active top-k cells at each cell size.",
                d_model,
                topk=d_model // 2,
            )

    if stage in ("sp02", "all"):
        d_values = [512, 1024] if plan_size == "core" else [512, 768, 1024]
        for d_model in d_values:
            for topk in [d_model, d_model // 2, d_model // 4, d_model // 8]:
                tag = "dense" if topk == d_model else f"topk{topk}"
                base.add_sparse_experiment(
                    plan,
                    f"sp02_d{d_model}_{tag}",
                    "Sweep active cell fraction to locate sparse quality/cost breakpoints.",
                    d_model,
                    topk=topk,
                )

    if stage in ("sp03", "all"):
        d_values = [512, 1024] if plan_size == "core" else [512, 768, 1024]
        for d_model in d_values:
            topk = d_model // 2
            for synapse_depth in [1, 2]:
                for memory_hidden_dims in [1, 2]:
                    base.add_sparse_experiment(
                        plan,
                        f"sp03_d{d_model}_sd{synapse_depth}_mh{memory_hidden_dims}_topk{topk}",
                        "Cross sparse cells with synapse depth and memory-hidden simplifications.",
                        d_model,
                        topk=topk,
                        synapse_depth=synapse_depth,
                        memory_hidden_dims=memory_hidden_dims,
                    )

    if stage in ("sp04", "all"):
        bs = [(512, 256), (1024, 512)] if plan_size == "core" else [
            (512, 256),
            (768, 384),
            (1024, 512),
        ]
        for d_model, topk in bs:
            for ticks in [1, 2, 4]:
                base.add_sparse_experiment(
                    plan,
                    f"sp04_d{d_model}_topk{topk}_tick{ticks}",
                    "Cross sparse cells with low tick counts to test whether sparsity shifts the tick optimum.",
                    d_model,
                    topk=topk,
                    iterations=ticks,
                )

    if stage in ("sp05", "all"):
        confirm = [
            ("sp05_confirm_d512_topk256_sd2_mh2_tick2", 512, 256, 2, 2, 2),
            ("sp05_confirm_d768_topk384_sd2_mh2_tick2", 768, 384, 2, 2, 2),
            ("sp05_confirm_d1024_topk512_sd2_mh2_tick2", 1024, 512, 2, 2, 2),
            ("sp05_confirm_d512_dense_sd2_mh2_tick2", 512, 512, 2, 2, 2),
        ]
        for name, d_model, topk, synapse_depth, memory_hidden_dims, ticks in confirm:
            base.add_sparse_experiment(
                plan,
                name,
                "Confirm sparse candidates with longer training to reduce short-run noise.",
                d_model,
                topk=topk,
                synapse_depth=synapse_depth,
                memory_hidden_dims=memory_hidden_dims,
                iterations=ticks,
                max_steps=2000,
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
        "loss_per_gb", "tokens_per_gb", "quality_cost_score",
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


base.configure_plan_defaults(
    metrics_prefix=METRICS_PREFIX,
    cluster_config="infra/clusters/h100_4nodes.env",
    dispatch_block_sparse=False,
    build_plan=build_plan,
    summarize_fn=summarize,
    stages=SPARSITY_STAGES,
    prefixes=SPARSITY_PREFIXES,
)


def normalize_default_outputs(args):
    remap = {
        "runs/experiment_plans/impl_validation_plan.csv":
            "runs/experiment_plans/sparsity_plan.csv",
        "runs/experiment_plans/impl_validation_batch_tune_plan.csv":
            "runs/experiment_plans/sparsity_batch_tune_plan.csv",
        "runs/metrics/impl_validation_summary.csv":
            "runs/metrics/sparsity_summary.csv",
        "runs/metrics/impl_validation_batch_profile.csv":
            "runs/metrics/sparsity_batch_profile.csv",
        "runs/metrics/impl_validation_batch_profile_quick.csv":
            "runs/metrics/sparsity_batch_profile_quick.csv",
        "runs/metrics/impl_validation_quick_probe_report.csv":
            "runs/metrics/sparsity_quick_probe_report.csv",
        "runs/metrics/impl_validation_batch_probe_report.csv":
            "runs/metrics/sparsity_batch_probe_report.csv",
    }
    for attr in ("output", "report_output", "batch_profile", "quick_output"):
        val = getattr(args, attr, None)
        if val and val in remap:
            setattr(args, attr, remap[val])


if __name__ == "__main__":
    args = base.parse_args()
    normalize_default_outputs(args)
    args.func(args)
