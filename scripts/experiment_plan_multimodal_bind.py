#!/usr/bin/env python3
"""
CTM-Bind: ImageBind-style multimodal self-supervised binding experiments.

Stages:
  mb00 — Smoke test: verify model creation + forward pass for each modality pair
  mb01 — Phase 1 image-text contrastive (Flickr30k) at small scale
  mb02 — Phase 1 scale sweep: hidden_size / layers / ticks / temperature
  mb03 — Phase 2a: image-thermal binding (TartanRGBT), freeze image encoder
  mb04 — Phase 2b: image-depth binding (TartanRGBT + MCAP), freeze image encoder
  mb05 — Phase 2c: video-text binding (MSR-VTT), freeze text encoder
  mb06 — Phase 2d: video-IMU binding (MCAP-Housing), freeze video encoder
  mb07 — Phase 3: joint fine-tune all modalities
  all  — Run all stages
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import experiment_plan_impl_validation as base


BIND_STAGES = (
    "mb00",
    "mb01",
    "mb02",
    "mb03",
    "mb04",
    "mb05",
    "mb06",
    "mb07",
    "all",
)
BIND_PREFIXES = tuple(f"{stage}_" for stage in BIND_STAGES if stage != "all")
METRICS_PREFIX = "multimodal_bind"

BIND_BASE_ARGS = {
    "train_module": "baseline.tasks.multimodal_bind.train",
    "model_type": "ctm_bind",
    "epochs": 1,
    "accumulation_steps": 4,
    "num_hidden_layers": 6,
    "hidden_size": 512,
    "d_model": 256,
    "d_input": 128,
    "heads": 4,
    "n_synch_out": 256,
    "n_synch_action": 256,
    "iterations": 10,
    "memory_length": 8,
    "memory_hidden_dims": 2,
    "deep_nlms": 1,
    "synapse_depth": 2,
    "tick_loss_mode": "last",
    "cell_sparsity_mode": "none",
    "cell_topk": 256,
    "cell_sparsity_rescale": 1,
    "self_cond": 1,
    "cross_layer_state": 1,
    "max_seq_len": 196,
    "log_interval": 10,
    "save_interval": 200,
    "max_steps": 500,
    "dtype": "bfloat16",
    "temperature": 0.07,
    "image_size": 128,
    "patch_size": 32,
    "embed_dim": 256,
    "pool_mode": "last_tick",
}


def bind_args(**overrides):
    data = dict(BIND_BASE_ARGS)
    data.update(overrides)
    return data


def build_plan(stage, plan_size="full"):
    plan = []

    if stage in ("mb00", "all"):
        _add_mb00_smoke(plan, plan_size)

    if stage in ("mb01", "all"):
        _add_mb01_image_text_base(plan, plan_size)

    if stage in ("mb02", "all"):
        _add_mb02_scale_sweep(plan, plan_size)

    if stage in ("mb03", "all"):
        _add_mb03_image_thermal(plan, plan_size)

    if stage in ("mb04", "all"):
        _add_mb04_image_depth(plan, plan_size)

    if stage in ("mb05", "all"):
        _add_mb05_video_text(plan, plan_size)

    if stage in ("mb06", "all"):
        _add_mb06_video_imu(plan, plan_size)

    if stage in ("mb07", "all"):
        _add_mb07_joint_finetune(plan, plan_size)

    return base.validate_plan(plan)


def _add_modality_pair_experiment(
    plan, name, question, *,
    modality_pair,
    dataset,
    freeze_modality=None,
    **overrides,
):
    args = bind_args(
        modality_pair=modality_pair,
        dataset=dataset,
        **overrides,
    )
    if freeze_modality:
        args["freeze_modality"] = freeze_modality
    plan.append(base.experiment(name, question, args))


def _add_mb00_smoke(plan, plan_size):
    pairs = [
        ("image_text", "flickr30k", "Smoke: image-text contrastive pair forward + backward."),
        ("image_thermal", "tartanrgbt", "Smoke: image-thermal contrastive pair forward + backward."),
        ("image_depth", "tartanrgbt", "Smoke: image-depth contrastive pair forward + backward."),
        ("video_text", "msrvtt", "Smoke: video-text contrastive pair forward + backward."),
        ("video_imu", "mcap_housing", "Smoke: video-IMU contrastive pair forward + backward."),
    ]
    for pair, ds, question in pairs:
        _add_modality_pair_experiment(
            plan,
            f"mb00_smoke_{pair}",
            question,
            modality_pair=pair,
            dataset=ds,
            max_steps=20,
            log_interval=5,
            batch_size=2,
        )


def _add_mb01_image_text_base(plan, plan_size):
    configs = [
        ("mb01_flickr_d256_t10_b4", 4, 10, 0.07,
         "Phase 1 baseline: Flickr30k image-text, small CTM, 10 ticks."),
    ]
    for name, bs, ticks, temp, question in configs:
        _add_modality_pair_experiment(
            plan,
            name,
            question,
            modality_pair="image_text",
            dataset="flickr30k",
            batch_size=bs,
            iterations=ticks,
            temperature=temp,
            max_steps=500,
        )

    if base.include_plan_size(plan_size, "full"):
        configs_ext = [
            ("mb01_flickr_d256_t10_b8", 8, 10, 0.07,
             "Phase 1: larger batch via accumulation."),
            ("mb01_flickr_d256_t15_b4", 4, 15, 0.07,
             "Phase 1: more ticks for deeper reasoning."),
            ("mb01_flickr_d256_t10_b4_temp005", 4, 10, 0.05,
             "Phase 1: sharper contrastive temperature."),
            ("mb01_flickr_d256_t10_b4_temp010", 4, 10, 0.10,
             "Phase 1: softer contrastive temperature."),
        ]
        for name, bs, ticks, temp, question in configs_ext:
            _add_modality_pair_experiment(
                plan,
                name,
                question,
                modality_pair="image_text",
                dataset="flickr30k",
                batch_size=bs,
                iterations=ticks,
                temperature=temp,
                max_steps=500,
            )

    if base.include_plan_size(plan_size, "wide"):
        configs_wide = [
            ("mb01_flickr_d256_t10_b4_pool_weighted", 4, 10, 0.07,
             "Phase 1: learnable tick-weighted pooling instead of last tick."),
            ("mb01_flickr_d256_t10_b4_pool_mean", 4, 10, 0.07,
             "Phase 1: mean-over-ticks pooling."),
        ]
        for name, bs, ticks, temp, question in configs_wide:
            _add_modality_pair_experiment(
                plan,
                name,
                question,
                modality_pair="image_text",
                dataset="flickr30k",
                batch_size=bs,
                iterations=ticks,
                temperature=temp,
                max_steps=500,
                pool_mode=name.split("_")[-1],
            )


def _add_mb02_scale_sweep(plan, plan_size):
    sweep = [
        ("mb02_d128_l3_t5", 128, 64, 32, 2, 3, 5,
         "Scale: tiny CTM for fast iteration."),
        ("mb02_d256_l6_t10", 256, 128, 64, 4, 6, 10,
         "Scale: small CTM baseline."),
        ("mb02_d384_l6_t10", 384, 192, 96, 6, 6, 10,
         "Scale: medium hidden_size."),
    ]
    for name, hs, dm, di, h, layers, ticks, question in sweep:
        _add_modality_pair_experiment(
            plan,
            name,
            question,
            modality_pair="image_text",
            dataset="flickr30k",
            hidden_size=hs,
            d_model=dm,
            d_input=di,
            heads=h,
            num_hidden_layers=layers,
            iterations=ticks,
            n_synch_out=dm,
            n_synch_action=dm,
            cell_topk=dm,
            batch_size=4,
            max_steps=500,
        )

    if base.include_plan_size(plan_size, "full"):
        sweep_ext = [
            ("mb02_d256_l6_t20", 256, 128, 64, 4, 6, 20,
             "Scale: more ticks at small size."),
            ("mb02_d512_l6_t10", 512, 256, 128, 4, 6, 10,
             "Scale: standard hidden_size."),
            ("mb02_d256_l12_t10", 256, 128, 64, 4, 12, 10,
             "Scale: deeper network."),
            ("mb02_d256_l6_t10_sparse128", 256, 128, 64, 4, 6, 10,
             "Scale: cell sparsity topk=128."),
        ]
        for name, hs, dm, di, h, layers, ticks, question in sweep_ext:
            overrides = {}
            if "sparse" in name:
                overrides["cell_sparsity_mode"] = "topk"
                overrides["cell_topk"] = 128
            else:
                overrides["cell_topk"] = dm
            _add_modality_pair_experiment(
                plan,
                name,
                question,
                modality_pair="image_text",
                dataset="flickr30k",
                hidden_size=hs,
                d_model=dm,
                d_input=di,
                heads=h,
                num_hidden_layers=layers,
                iterations=ticks,
                n_synch_out=dm,
                n_synch_action=dm,
                batch_size=4,
                max_steps=500,
                **overrides,
            )

    if base.include_plan_size(plan_size, "wide"):
        _add_modality_pair_experiment(
            plan,
            "mb02_d512_l6_t15",
            "Scale: larger model with more ticks.",
            modality_pair="image_text",
            dataset="flickr30k",
            hidden_size=512,
            d_model=256,
            d_input=128,
            heads=4,
            num_hidden_layers=6,
            iterations=15,
            n_synch_out=256,
            n_synch_action=256,
            batch_size=2,
            max_steps=1000,
        )
        _add_modality_pair_experiment(
            plan,
            "mb02_d256_l6_t10_accum16",
            "Scale: simulated large batch via heavy accumulation.",
            modality_pair="image_text",
            dataset="flickr30k",
            batch_size=2,
            accumulation_steps=16,
            max_steps=500,
        )


def _add_mb03_image_thermal(plan, plan_size):
    _add_modality_pair_experiment(
        plan,
        "mb03_thermal_d256_t10",
        "Phase 2a: image-thermal binding (TartanRGBT), freeze image encoder.",
        modality_pair="image_thermal",
        dataset="tartanrgbt",
        freeze_modality="image",
        max_steps=500,
    )

    if base.include_plan_size(plan_size, "full"):
        _add_modality_pair_experiment(
            plan,
            "mb03_thermal_d256_t15",
            "Phase 2a: more ticks for thermal encoder.",
            modality_pair="image_thermal",
            dataset="tartanrgbt",
            freeze_modality="image",
            iterations=15,
            max_steps=500,
        )
        _add_modality_pair_experiment(
            plan,
            "mb03_thermal_d256_t10_temp010",
            "Phase 2a: softer temperature for thermal binding.",
            modality_pair="image_thermal",
            dataset="tartanrgbt",
            freeze_modality="image",
            temperature=0.10,
            max_steps=500,
        )

    if base.include_plan_size(plan_size, "wide"):
        _add_modality_pair_experiment(
            plan,
            "mb03_thermal_d256_t10_nofreeze",
            "Phase 2a: unfrozen image encoder for comparison.",
            modality_pair="image_thermal",
            dataset="tartanrgbt",
            max_steps=500,
        )


def _add_mb04_image_depth(plan, plan_size):
    _add_modality_pair_experiment(
        plan,
        "mb04_depth_d256_t10",
        "Phase 2b: image-depth binding (TartanRGBT + MCAP), freeze image encoder.",
        modality_pair="image_depth",
        dataset="tartanrgbt",
        freeze_modality="image",
        max_steps=500,
    )

    if base.include_plan_size(plan_size, "full"):
        _add_modality_pair_experiment(
            plan,
            "mb04_depth_mcap_d256_t10",
            "Phase 2b: image-depth binding using MCAP data.",
            modality_pair="image_depth",
            dataset="mcap_housing",
            freeze_modality="image",
            max_steps=500,
        )
        _add_modality_pair_experiment(
            plan,
            "mb04_depth_d256_t10_sparse128",
            "Phase 2b: sparse depth encoder.",
            modality_pair="image_depth",
            dataset="tartanrgbt",
            freeze_modality="image",
            cell_sparsity_mode="topk",
            cell_topk=128,
            max_steps=500,
        )


def _add_mb05_video_text(plan, plan_size):
    _add_modality_pair_experiment(
        plan,
        "mb05_videotext_d256_t10",
        "Phase 2c: video-text binding (MSR-VTT), freeze text encoder.",
        modality_pair="video_text",
        dataset="msrvtt",
        freeze_modality="text",
        max_steps=500,
        num_video_frames=4,
    )

    if base.include_plan_size(plan_size, "full"):
        _add_modality_pair_experiment(
            plan,
            "mb05_videotext_d256_t10_f8",
            "Phase 2c: more video frames for temporal coverage.",
            modality_pair="video_text",
            dataset="msrvtt",
            freeze_modality="text",
            num_video_frames=8,
            max_steps=500,
        )
        _add_modality_pair_experiment(
            plan,
            "mb05_videotext_d256_t15",
            "Phase 2c: more ticks for video encoder.",
            modality_pair="video_text",
            dataset="msrvtt",
            freeze_modality="text",
            iterations=15,
            num_video_frames=4,
            max_steps=500,
        )


def _add_mb06_video_imu(plan, plan_size):
    _add_modality_pair_experiment(
        plan,
        "mb06_videoimu_d256_t10",
        "Phase 2d: video-IMU binding (MCAP-Housing), freeze video encoder.",
        modality_pair="video_imu",
        dataset="mcap_housing",
        freeze_modality="video",
        max_steps=500,
        imu_seq_len=250,
    )

    if base.include_plan_size(plan_size, "full"):
        _add_modality_pair_experiment(
            plan,
            "mb06_videoimu_d256_t5",
            "Phase 2d: fewer ticks for simple IMU signal.",
            modality_pair="video_imu",
            dataset="mcap_housing",
            freeze_modality="video",
            iterations=5,
            max_steps=500,
        )
        _add_modality_pair_experiment(
            plan,
            "mb06_videoimu_d256_t10_temp010",
            "Phase 2d: softer temperature for IMU binding.",
            modality_pair="video_imu",
            dataset="mcap_housing",
            freeze_modality="video",
            temperature=0.10,
            max_steps=500,
        )


def _add_mb07_joint_finetune(plan, plan_size):
    _add_modality_pair_experiment(
        plan,
        "mb07_joint_d256_t10",
        "Phase 3: joint fine-tune all modalities, unfrozen.",
        modality_pair="joint",
        dataset="all",
        max_steps=300,
    )

    if base.include_plan_size(plan_size, "full"):
        _add_modality_pair_experiment(
            plan,
            "mb07_joint_d256_t10_wd01",
            "Phase 3: joint with weight decay regularization.",
            modality_pair="joint",
            dataset="all",
            max_steps=300,
            weight_decay=0.01,
        )
        _add_modality_pair_experiment(
            plan,
            "mb07_joint_d256_t15",
            "Phase 3: more ticks during joint fine-tuning.",
            modality_pair="joint",
            dataset="all",
            iterations=15,
            max_steps=300,
        )


def summarize(args):
    rows = [
        row for row in base.latest_rows(args.metrics_dir)
        if _is_final_metrics_row(row)
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
    rows.sort(key=lambda r: r.get("experiment_name", ""))
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fields = [
        "experiment_name", "model_type", "modality_pair", "dataset",
        "loss", "contrastive_loss", "tokens_per_sec",
        "peak_memory_mb", "temperature", "pool_mode",
        "hidden_size", "num_hidden_layers", "d_model", "d_input",
        "iterations", "memory_length", "memory_hidden_dims",
        "synapse_depth", "image_size", "patch_size", "embed_dim",
        "batch_size", "accumulation_steps",
        "freeze_modality",
        "loss_per_gb",
        "global_step", "metrics_file",
    ]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote summary: {args.output}")


def _is_final_metrics_row(row):
    name = row.get("experiment_name", "")
    if name.startswith("qp__") or name.startswith("bt__"):
        return False
    loss = base.parse_float(row, "loss")
    return not math.isnan(loss)


base.configure_plan_defaults(
    metrics_prefix=METRICS_PREFIX,
    cluster_config="infra/clusters/h100_4nodes.env",
    dispatch_block_sparse=False,
    build_plan=build_plan,
    summarize_fn=summarize,
    is_regional_experiment=lambda exp: False,
    is_final_metrics_row=_is_final_metrics_row,
    stages=BIND_STAGES,
    prefixes=BIND_PREFIXES,
)


def normalize_default_outputs(args):
    remap = {
        "runs/experiment_plans/impl_validation_plan.csv":
            "runs/experiment_plans/multimodal_bind_plan.csv",
        "runs/experiment_plans/impl_validation_batch_tune_plan.csv":
            "runs/experiment_plans/multimodal_bind_batch_tune_plan.csv",
        "runs/metrics/impl_validation_summary.csv":
            "runs/metrics/multimodal_bind_summary.csv",
        "runs/metrics/impl_validation_batch_profile.csv":
            "runs/metrics/multimodal_bind_batch_profile.csv",
        "runs/metrics/impl_validation_batch_profile_quick.csv":
            "runs/metrics/multimodal_bind_batch_profile_quick.csv",
        "runs/metrics/impl_validation_quick_probe_report.csv":
            "runs/metrics/multimodal_bind_quick_probe_report.csv",
        "runs/metrics/impl_validation_batch_probe_report.csv":
            "runs/metrics/multimodal_bind_batch_probe_report.csv",
    }
    for attr in ("output", "report_output", "batch_profile", "quick_output"):
        val = getattr(args, attr, None)
        if val and val in remap:
            setattr(args, attr, remap[val])


if __name__ == "__main__":
    import csv
    args = base.parse_args()
    normalize_default_outputs(args)
    args.func(args)
