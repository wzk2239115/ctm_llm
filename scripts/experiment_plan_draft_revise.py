#!/usr/bin/env python3
"""Draft-and-revise CTM experiment plan.

This plan separates two kinds of runs:

- dr00/dr01/"real": runnable anchors and negative controls using the current
  training stack.
- dr02/dr02_elf/"real": parallel draft slots + ELF/MTP smoke matrix on regional dense_mask.
- dr07 studies a DINO-like learning-speed spectrum: not one fixed
  teacher/student pair, but several EMA targets with different update speeds
  supervising fast ticks, slow ticks, and draft slots.
- dr08+ studies residual compute for CTM: delta states, real conditional
  execution, cached/recursive tick internals, and full/residual/stop controllers.
- dr12 studies objective mismatch against ELF: current supervised next-token CE
  versus self-supervised denoising/flow objectives and hybrid decoder branches.

The shared CLI comes from experiment_plan_impl_validation.py, including
commands, quick-probe, probe-and-run, run-parallel, run-only, summarize, and
batch recommendation helpers for the 4-node / 32-GPU pool.
"""

import experiment_plan_impl_validation as base


DRAFT_STAGES = (
    "dr00",
    "dr01",
    "dr02",
    "dr02_elf",
    "dr03",
    "dr04",
    "dr05",
    "dr06",
    "dr07",
    "dr08",
    "dr09",
    "dr10",
    "dr11",
    "dr12",
    "real",
    "all",
)
DRAFT_PREFIXES = tuple(f"{stage}_" for stage in DRAFT_STAGES if stage != "all")


def bs_for(
    d_model,
    *,
    draft=False,
    async_tick=False,
    iterations=4,
    elf_horizon=4,
    fastslow=False,
):
    if d_model <= 512:
        batch = 10
    elif d_model <= 1024:
        batch = 8
    elif d_model <= 1536:
        batch = 6
    else:
        batch = 4
    if draft:
        batch = max(2, batch - 2)
    if async_tick:
        batch = max(2, batch - 2)
        if iterations >= 16 or fastslow:
            batch = max(2, batch - 2)
    if iterations >= 8:
        batch = max(2, batch - 2)
    if elf_horizon >= 8:
        batch = max(2, batch - 2)
    if async_tick and d_model >= 1024:
        batch = max(2, batch - 1)
    floor = 4 if d_model <= 512 else (3 if d_model <= 1024 else 2)
    return max(floor, batch)


def regional(
    plan,
    name,
    question,
    *,
    num_experts=16,
    expert_size=32,
    max_steps=3500,
    iterations=8,
    activation_passes=4,
    topk_experts=1,
    shared_experts=1,
    balance=1e-2,
    diversity=None,
    batch_size=None,
    **kwargs,
):
    d_model = num_experts * expert_size
    if diversity is None:
        diversity = 1e-3 if activation_passes >= 3 else 0.0
    if batch_size is None:
        fastslow = (
            kwargs.get("fast_output_mode", "none") != "none"
            or float(kwargs.get("habit_output_weight", 0.0)) > 0
            or float(kwargs.get("slow_output_weight", 0.0)) > 0
        )
        batch_size = bs_for(
            d_model,
            draft=(
                kwargs.get("draft_mode", "none") != "none"
                or kwargs.get("residual_compute_mode", "none") != "none"
            ),
            async_tick=kwargs.get("async_tick_mode", "none") != "none",
            iterations=iterations,
            elf_horizon=int(kwargs.get("elf_max_horizon", 4)),
            fastslow=fastslow,
        )
    base.add_regional_experiment(
        plan,
        name,
        question,
        num_experts=num_experts,
        expert_size=expert_size,
        activation_passes=activation_passes,
        shared_experts=shared_experts,
        topk_experts=topk_experts,
        balance=balance,
        diversity=diversity,
        iterations=iterations,
        max_steps=max_steps,
        batch_size=batch_size,
        **kwargs,
    )


def draft_args(
    *,
    mode="parallel",
    block=4,
    head="slot_adapter",
    attention="block_bidir",
    loss=0.2,
    revise=0.0,
    corrupt=0.0,
    commit=0.0,
    rounds=1,
    carry=0,
    curriculum="none",
    threshold=0.65,
):
    return {
        "draft_mode": mode,
        "draft_block_size": block,
        "draft_head_mode": head,
        "draft_slot_attention": attention,
        "draft_loss_weight": loss,
        "draft_revise_weight": revise,
        "draft_corrupt_prob": corrupt,
        "draft_commit_loss_weight": commit,
        "draft_num_revise": rounds,
        "draft_memory_carry": carry,
        "draft_curriculum": curriculum,
        "draft_commit_threshold": threshold,
    }


def speed_args(
    *,
    mode="ema_spectrum",
    decays="0.90,0.97,0.995",
    weight=0.05,
    targets="fast,mid,slow",
    center=1,
    center_momentum=0.90,
    student_temp=0.10,
    teacher_temp=0.04,
    warmup=500,
):
    return {
        "speed_spectrum_mode": mode,
        "speed_ema_decays": decays,
        "speed_distill_weight": weight,
        "speed_target_ticks": targets,
        "speed_centering": center,
        "speed_center_momentum": center_momentum,
        "speed_student_temperature": student_temp,
        "speed_teacher_temperature": teacher_temp,
        "speed_warmup_steps": warmup,
    }


def residual_args(
    *,
    mode="observe",
    synapse="dense_delta",
    nlm="full",
    attention="kv_cache",
    sync="recursive_pairs",
    groups=32,
    active_ratio=0.25,
    threshold=0.15,
    refresh=4,
    compute=0.01,
    delta_l1=0.0,
    controller="none",
    speed_cells="none",
):
    return {
        "residual_compute_mode": mode,
        "residual_synapse_mode": synapse,
        "residual_nlm_mode": nlm,
        "residual_attention_mode": attention,
        "residual_sync_mode": sync,
        "residual_num_groups": groups,
        "residual_active_ratio": active_ratio,
        "residual_gate_threshold": threshold,
        "residual_full_refresh_interval": refresh,
        "residual_compute_weight": compute,
        "residual_delta_l1_weight": delta_l1,
        "residual_tick_controller": controller,
        "residual_speed_cells": speed_cells,
        "residual_track_deltas": 1,
    }


def objective_args(
    *,
    mode="hybrid_flow_ce",
    denoise=0.5,
    ce=0.5,
    latent="token_embed",
    time="logit_normal",
    self_cond=0.5,
    decoder_noise=1.0,
    cond_drop=0.0,
):
    return {
        "objective_mode": mode,
        "objective_denoise_weight": denoise,
        "objective_ce_weight": ce,
        "objective_latent_space": latent,
        "objective_time_schedule": time,
        "objective_self_cond_prob": self_cond,
        "objective_decoder_noise_scale": decoder_noise,
        "objective_cond_drop_prob": cond_drop,
    }


def build_plan(stage, plan_size="full"):
    plan = []

    if stage == "real":
        for item in ("dr00", "dr01"):
            plan.extend(build_plan(item, plan_size))
        return base.validate_plan(plan)

    if stage in ("dr00", "all"):
        regional(
            plan,
            "dr00_anchor_regional_d512_tick4",
            "Runnable anchor: regional sparse CTM before any draft-revise module.",
            num_experts=16,
            expert_size=32,
            iterations=4,
            activation_passes=4,
            max_steps=2500,
        )
        regional(
            plan,
            "dr00_anchor_regional_d1024_tick4",
            "Larger runnable anchor for quality/cost comparison.",
            num_experts=16,
            expert_size=64,
            iterations=4,
            activation_passes=4,
            max_steps=2500,
        )
        regional(
            plan,
            "dr00_anchor_async_anytime_d512",
            "Runnable anchor: async clocks plus anytime output before draft slots.",
            num_experts=16,
            expert_size=32,
            iterations=16,
            activation_passes=4,
            async_tick_mode="banded",
            async_tick_periods="1,2,4,8",
            async_fast_output_weight=0.20,
            fast_output_mode="anytime",
            fast_output_weight=0.10,
            habit_output_weight=0.20,
            slow_output_weight=0.10,
            fast_output_ticks="1,4,8,12,16",
            fast_output_distill_weight=0.10,
            tick_halt_mode="threshold",
            tick_halt_threshold=0.30,
            tick_compute_weight=2e-3,
            max_steps=3500,
        )

    if stage in ("dr01", "all"):
        regional(
            plan,
            "dr01_current_mtp124_d512",
            "Current MTP negative control: shared lm_head over h=1,2,4.",
            num_experts=16,
            expert_size=32,
            iterations=4,
            activation_passes=4,
            moe_mtp_mode="mtp_1_2_4",
            moe_mtp_horizons="1,2,4",
            max_steps=3000,
        )
        regional(
            plan,
            "dr01_current_elf_linear_h4_d512",
            "Current ELF-style shifted CE control: linear horizons up to 4.",
            num_experts=16,
            expert_size=32,
            iterations=4,
            activation_passes=4,
            elf_horizon_mode="linear",
            elf_max_horizon=4,
            max_steps=3000,
        )
        regional(
            plan,
            "dr01_current_elf_linear_h8_tick8_d512",
            "Stress current shifted CE with more ticks and longer horizon.",
            num_experts=16,
            expert_size=32,
            iterations=8,
            activation_passes=4,
            elf_horizon_mode="linear",
            elf_max_horizon=8,
            tick_improve_weight=0.05,
            max_steps=3000,
        )
        if base.include_plan_size(plan_size, "full"):
            regional(
                plan,
                "dr01_current_async_mtp124_d512",
                "Current async+MTP control before adding true draft slots.",
                num_experts=16,
                expert_size=32,
                iterations=16,
                activation_passes=4,
                async_tick_mode="banded",
                async_tick_periods="1,2,4,8",
                async_fast_output_weight=0.20,
                fast_output_mode="anytime",
                fast_output_weight=0.10,
                habit_output_weight=0.20,
                slow_output_weight=0.10,
                fast_output_ticks="1,4,8,12,16",
                fast_output_distill_weight=0.10,
                tick_halt_mode="threshold",
                tick_halt_threshold=0.30,
                tick_compute_weight=2e-3,
                moe_mtp_mode="mtp_1_2_4",
                moe_mtp_horizons="1,2,4",
                max_steps=3500,
            )

    if stage in ("dr02", "all"):
        blocks = [2, 4] if plan_size == "core" else [2, 4, 8]
        heads = (
            ["shared", "slot_adapter"]
            if plan_size == "core"
            else ["shared", "slot_adapter", "slot_head"]
        )
        for block in blocks:
            for head in heads:
                regional(
                    plan,
                    f"dr02_parallel_b{block}_{head}_d512",
                    "Parallel draft slots with slot-aware heads on regional dense_mask.",
                    num_experts=16,
                    expert_size=32,
                    iterations=8,
                    activation_passes=4,
                    max_steps=3500,
                    **draft_args(mode="parallel", block=block, head=head, loss=0.2),
                )

    if stage in ("dr02_elf", "all"):
        elf_heads = ["shared", "slot_adapter"]
        if base.include_plan_size(plan_size, "full"):
            elf_heads.append("slot_head")
        for head in elf_heads:
            regional(
                plan,
                f"dr02_elf_parallel_b4_{head}_d512",
                "Draft slots + ELF linear h4 + tick_improve confirm run.",
                num_experts=16,
                expert_size=32,
                iterations=8,
                activation_passes=4,
                elf_horizon_mode="linear",
                elf_max_horizon=4,
                tick_improve_weight=0.05,
                max_steps=3500,
                **draft_args(mode="parallel", block=4, head=head, loss=0.2),
            )

    if stage in ("dr03", "all"):
        for corrupt in ([0.15, 0.30] if plan_size == "core" else [0.10, 0.20, 0.35, 0.50]):
            regional(
                plan,
                base.ctm_name("dr03_revise_b4", corrupt=corrupt),
                "Future module: corruption-trained overwrite/revise branch.",
                num_experts=16,
                expert_size=32,
                iterations=8,
                activation_passes=4,
                max_steps=4000,
                **draft_args(
                    mode="revise",
                    block=4,
                    head="slot_adapter",
                    loss=0.15,
                    revise=0.20,
                    corrupt=corrupt,
                ),
            )
        if base.include_plan_size(plan_size, "full"):
            for rounds in [1, 2, 3]:
                regional(
                    plan,
                    f"dr03_revise_rounds{rounds}_b8_d512",
                    "Future module: number of revise passes at fixed block size.",
                    num_experts=16,
                    expert_size=32,
                    iterations=8,
                    activation_passes=4,
                    max_steps=4000,
                    **draft_args(
                        mode="revise",
                        block=8,
                        head="slot_adapter",
                        loss=0.15,
                        revise=0.20,
                        corrupt=0.30,
                        rounds=rounds,
                    ),
                )

    if stage in ("dr04", "all"):
        for threshold in [0.50, 0.65, 0.80]:
            regional(
                plan,
                base.ctm_name("dr04_commit_b4", th=threshold),
                "Future module: confidence/commit head for safe contiguous prefix emission.",
                num_experts=16,
                expert_size=32,
                iterations=8,
                activation_passes=4,
                max_steps=4000,
                **draft_args(
                    mode="revise",
                    block=4,
                    head="slot_adapter",
                    loss=0.15,
                    revise=0.20,
                    corrupt=0.30,
                    commit=0.05,
                    threshold=threshold,
                ),
            )
        if base.include_plan_size(plan_size, "full"):
            for commit_w in [0.02, 0.05, 0.10]:
                regional(
                    plan,
                    base.ctm_name("dr04_commit_b8", w=commit_w),
                    "Future module: commit loss weight sweep for draft acceptance.",
                    num_experts=16,
                    expert_size=32,
                    iterations=8,
                    activation_passes=4,
                    max_steps=4500,
                    **draft_args(
                        mode="revise",
                        block=8,
                        head="slot_adapter",
                        loss=0.15,
                        revise=0.20,
                        corrupt=0.30,
                        commit=commit_w,
                        threshold=0.65,
                    ),
                )

    if stage in ("dr05", "all"):
        for carry in [0, 1]:
            for attention in ["causal_slots", "block_bidir"]:
                regional(
                    plan,
                    f"dr05_memory_carry{carry}_{attention}_b4_d512",
                    "Future module: draft-slot attention mask and CTM state carry.",
                    num_experts=16,
                    expert_size=32,
                    iterations=8,
                    activation_passes=4,
                    memory_length=8,
                    max_steps=4500,
                    **draft_args(
                        mode="revise",
                        block=4,
                        head="slot_adapter",
                        attention=attention,
                        loss=0.15,
                        revise=0.20,
                        corrupt=0.30,
                        commit=0.05,
                        carry=carry,
                    ),
                )
        if base.include_plan_size(plan_size, "wide"):
            for block in [8, 12]:
                regional(
                    plan,
                    f"dr05_memory_carry1_b{block}_d1024",
                    "Future larger-capacity draft memory-carry confirmation.",
                    num_experts=16,
                    expert_size=64,
                    iterations=8,
                    activation_passes=4,
                    memory_length=10,
                    max_steps=5000,
                    **draft_args(
                        mode="revise",
                        block=block,
                        head="slot_adapter",
                        loss=0.15,
                        revise=0.20,
                        corrupt=0.30,
                        commit=0.05,
                        carry=1,
                    ),
                )

    if stage in ("dr06", "all"):
        combos = [
            ("b4", 4, 8, "1,2,4,8", 0.20, 3500),
            ("b8", 8, 12, "1,2,4,8", 0.20, 4500),
        ]
        if base.include_plan_size(plan_size, "wide"):
            combos.append(("b12", 12, 16, "1,4,8,16", 0.25, 5500))
        for tag, block, ticks, periods, fast_w, steps in combos:
            regional(
                plan,
                f"dr06_full_async_sparse_revise_{tag}_d512",
                "Future full stack: sparse CTM + async clocks + draft/revise/commit.",
                num_experts=16,
                expert_size=32,
                iterations=ticks,
                activation_passes=4,
                async_tick_mode="banded",
                async_tick_periods=periods,
                async_fast_output_weight=fast_w,
                fast_output_mode="anytime",
                fast_output_weight=0.10,
                habit_output_weight=0.20,
                slow_output_weight=0.10,
                fast_output_ticks="1,4,8,12,16",
                fast_output_distill_weight=0.10,
                tick_halt_mode="threshold",
                tick_halt_threshold=0.30,
                tick_compute_weight=2e-3,
                max_steps=steps,
                batch_size=5 if block >= 8 else 6,
                **draft_args(
                    mode="revise",
                    block=block,
                    head="slot_adapter",
                    loss=0.15,
                    revise=0.20,
                    corrupt=0.30,
                    commit=0.05,
                    rounds=2,
                    carry=1,
                    curriculum="linear",
                ),
            )

    if stage in ("dr07", "all"):
        for decays in [
            "0.90,0.97,0.995",
            "0.80,0.95,0.99,0.997",
        ]:
            regional(
                plan,
                f"dr07_speed_spectrum_{decays.replace(',', '_').replace('.', 'p')}_d512",
                "Future module: DINO-like continuous learning-speed spectrum with multiple EMA targets.",
                num_experts=16,
                expert_size=32,
                iterations=8,
                activation_passes=4,
                max_steps=4000,
                **speed_args(decays=decays, weight=0.05),
            )
        for weight in ([0.02, 0.05] if plan_size == "core" else [0.01, 0.03, 0.07, 0.12]):
            regional(
                plan,
                base.ctm_name("dr07_speed_weight", w=weight),
                "Future module: speed-spectrum distillation strength sweep.",
                num_experts=16,
                expert_size=32,
                iterations=8,
                activation_passes=4,
                max_steps=4000,
                **speed_args(weight=weight),
            )
        if base.include_plan_size(plan_size, "full"):
            for center, teacher_temp in [
                (0, 0.04),
                (1, 0.07),
                (1, 0.10),
            ]:
                regional(
                    plan,
                    base.ctm_name("dr07_speed_center_temp", c=center, tt=teacher_temp),
                    "Future module: DINO centering and teacher-temperature ablation.",
                    num_experts=16,
                    expert_size=32,
                    iterations=8,
                    activation_passes=4,
                    max_steps=4000,
                    **speed_args(center=center, teacher_temp=teacher_temp),
                )
            regional(
                plan,
                "dr07_speed_plus_draft_revise_b4_d512",
                "Future combined stack: draft-revise block generator plus learning-speed spectrum.",
                num_experts=16,
                expert_size=32,
                iterations=12,
                activation_passes=4,
                async_tick_mode="banded",
                async_tick_periods="1,2,4,8",
                async_fast_output_weight=0.20,
                fast_output_mode="anytime",
                fast_output_weight=0.10,
                habit_output_weight=0.20,
                slow_output_weight=0.10,
                fast_output_ticks="1,4,8,12",
                fast_output_distill_weight=0.10,
                tick_halt_mode="threshold",
                tick_halt_threshold=0.30,
                tick_compute_weight=2e-3,
                max_steps=5000,
                batch_size=5,
                **draft_args(
                    mode="revise",
                    block=4,
                    head="slot_adapter",
                    loss=0.15,
                    revise=0.20,
                    corrupt=0.30,
                    commit=0.05,
                    rounds=2,
                    carry=1,
                    curriculum="linear",
                ),
                **speed_args(weight=0.05, targets="fast,mid,slow,draft"),
            )

    if stage in ("dr08", "all"):
        # Residual compute phase 1: keep semantics close to dense CTM, but add
        # delta accounting, K/V cache, recursive sync, and residual state paths.
        for tag, synapse, nlm, attention, sync in [
            ("observe", "dense_delta", "full", "kv_cache", "recursive_pairs"),
            ("syn_delta", "dense_delta", "output_delta", "kv_cache", "recursive_pairs"),
            ("attn_refresh", "dense_delta", "output_delta", "refresh_delta", "recursive_pairs"),
            ("sync_recursive", "dense_delta", "output_delta", "kv_cache", "recursive_pairs"),
        ]:
            regional(
                plan,
                f"dr08_residual_semantic_{tag}_d512",
                "Future module: residual compute without hard skipping; measure delta statistics and stability.",
                num_experts=16,
                expert_size=32,
                iterations=8,
                activation_passes=4,
                max_steps=3500,
                **residual_args(
                    mode="observe",
                    synapse=synapse,
                    nlm=nlm,
                    attention=attention,
                    sync=sync,
                    compute=0.0,
                    delta_l1=0.0,
                ),
            )
        if base.include_plan_size(plan_size, "full"):
            for refresh in [2, 4, 8]:
                regional(
                    plan,
                    f"dr08_attention_refresh{refresh}_d512",
                    "Future module: intermittent full attention refresh with residual/delta reuse.",
                    num_experts=16,
                    expert_size=32,
                    iterations=8,
                    activation_passes=4,
                    max_steps=3500,
                    **residual_args(
                        mode="observe",
                        synapse="dense_delta",
                        nlm="output_delta",
                        attention="refresh_delta",
                        refresh=refresh,
                        compute=0.0,
                    ),
                )

    if stage in ("dr09", "all"):
        # Phase 2: real synapse/block skipping. These knobs must control dispatch,
        # not merely multiply a dense result by a mask.
        group_grid = [
            (16, 0.25), (32, 0.25),
        ] if plan_size == "core" else [
            (16, 0.125), (16, 0.25), (32, 0.125), (32, 0.25), (64, 0.125), (64, 0.25),
        ]
        for groups, active_ratio in group_grid:
            regional(
                plan,
                f"dr09_synapse_skip_g{groups}_a{str(active_ratio).replace('.', 'p')}_d512",
                "Future module: block-level residual synapse update with true conditional execution.",
                num_experts=16,
                expert_size=32,
                iterations=8,
                activation_passes=4,
                max_steps=4000,
                **residual_args(
                    mode="block_skip",
                    synapse="block_delta_skip",
                    nlm="output_delta",
                    attention="kv_cache",
                    groups=groups,
                    active_ratio=active_ratio,
                    threshold=0.15,
                    refresh=4,
                    compute=0.01,
                    delta_l1=1e-4,
                ),
            )
        if base.include_plan_size(plan_size, "full"):
            for threshold in [0.05, 0.15, 0.30, 0.45]:
                regional(
                    plan,
                    base.ctm_name("dr09_synapse_gate", th=threshold),
                    "Future module: residual synapse novelty threshold sweep.",
                    num_experts=16,
                    expert_size=32,
                    iterations=8,
                    activation_passes=4,
                    max_steps=4000,
                    **residual_args(
                        mode="block_skip",
                        synapse="block_delta_skip",
                        nlm="output_delta",
                        active_ratio=0.25,
                        threshold=threshold,
                        refresh=4,
                        compute=0.01,
                        delta_l1=1e-4,
                    ),
                )

    if stage in ("dr10", "all"):
        # Phase 3: replace repeated full history-window NLM work with recursive
        # fast path, plus periodic or event-driven full refresh.
        for nlm_mode in ["recursive_fast", "hybrid_fast_full"]:
            for refresh in ([4, 8] if plan_size == "core" else [2, 4, 8, 16]):
                regional(
                    plan,
                    f"dr10_nlm_{nlm_mode}_refresh{refresh}_d512",
                    "Future module: recursive NLM fast path with periodic full-history correction.",
                    num_experts=16,
                    expert_size=32,
                    iterations=8,
                    activation_passes=4,
                    memory_length=10,
                    max_steps=4500,
                    **residual_args(
                        mode="nlm_recursive",
                        synapse="block_delta_skip",
                        nlm=nlm_mode,
                        attention="kv_cache",
                        active_ratio=0.25,
                        threshold=0.15,
                        refresh=refresh,
                        compute=0.015,
                        delta_l1=1e-4,
                    ),
                )
        if base.include_plan_size(plan_size, "wide"):
            for memory_length in [8, 16, 32]:
                regional(
                    plan,
                    f"dr10_nlm_recursive_mem{memory_length}_d1024",
                    "Future module: NLM recursive path at longer memory windows where caching should matter most.",
                    num_experts=16,
                    expert_size=64,
                    iterations=8,
                    activation_passes=4,
                    memory_length=memory_length,
                    max_steps=5000,
                    **residual_args(
                        mode="nlm_recursive",
                        synapse="block_delta_skip",
                        nlm="hybrid_fast_full",
                        attention="kv_cache",
                        active_ratio=0.25,
                        threshold=0.15,
                        refresh=8,
                        compute=0.015,
                        delta_l1=1e-4,
                    ),
                )

    if stage in ("dr11", "all"):
        # Phase 4: full/residual/stop controller and speed-cell scheduling.
        for controller, compute in [
            ("threshold", 0.01),
            ("learned", 0.02),
        ]:
            regional(
                plan,
                f"dr11_tick_controller_{controller}_d512",
                "Future module: three-way stop/residual/full tick controller with compute penalty.",
                num_experts=16,
                expert_size=32,
                iterations=12,
                activation_passes=4,
                max_steps=5000,
                **residual_args(
                    mode="tick_controller",
                    synapse="block_delta_skip",
                    nlm="hybrid_fast_full",
                    attention="refresh_delta",
                    active_ratio=0.25,
                    threshold=0.15,
                    refresh=4,
                    compute=compute,
                    delta_l1=1e-4,
                    controller=controller,
                ),
            )
        for speed_cells in ["fast_mid_slow", "event_teacher"]:
            regional(
                plan,
                f"dr11_speed_cells_{speed_cells}_d512",
                "Future module: event-triggered fast/mid/slow cell groups with residual execution frequency.",
                num_experts=16,
                expert_size=32,
                iterations=16,
                activation_passes=4,
                async_tick_mode="banded",
                async_tick_periods="1,2,4,8",
                async_fast_output_weight=0.20,
                max_steps=5000,
                batch_size=5,
                **residual_args(
                    mode="speed_cells",
                    synapse="block_delta_skip",
                    nlm="hybrid_fast_full",
                    attention="refresh_delta",
                    active_ratio=0.25,
                    threshold=0.15,
                    refresh=4,
                    compute=0.02,
                    delta_l1=1e-4,
                    controller="learned",
                    speed_cells=speed_cells,
                ),
                **speed_args(weight=0.05, targets="fast,mid,slow"),
            )
        if base.include_plan_size(plan_size, "full"):
            regional(
                plan,
                "dr11_full_draft_speed_residual_b4_d512",
                "Future full stack: draft-revise + speed spectrum + residual tick controller.",
                num_experts=16,
                expert_size=32,
                iterations=16,
                activation_passes=4,
                async_tick_mode="banded",
                async_tick_periods="1,2,4,8",
                async_fast_output_weight=0.20,
                fast_output_mode="anytime",
                fast_output_weight=0.10,
                habit_output_weight=0.20,
                slow_output_weight=0.10,
                fast_output_ticks="1,4,8,12,16",
                fast_output_distill_weight=0.10,
                max_steps=6000,
                batch_size=4,
                **draft_args(
                    mode="revise",
                    block=4,
                    head="slot_adapter",
                    loss=0.15,
                    revise=0.20,
                    corrupt=0.30,
                    commit=0.05,
                    rounds=2,
                    carry=1,
                    curriculum="linear",
                ),
                **speed_args(weight=0.05, targets="fast,mid,slow,draft"),
                **residual_args(
                    mode="tick_controller",
                    synapse="block_delta_skip",
                    nlm="hybrid_fast_full",
                    attention="refresh_delta",
                    active_ratio=0.25,
                    threshold=0.15,
                    refresh=4,
                    compute=0.02,
                    delta_l1=1e-4,
                    controller="learned",
                    speed_cells="event_teacher",
                ),
            )

    if stage in ("dr12", "all"):
        # ELF is mainly a self-supervised flow/denoising objective over text
        # latents, with a mixed decoder CE branch. This stage tests whether CTM
        # needs the same objective shift before draft/revise can work well.
        for mode, denoise, ce in [
            ("causal_ce", 0.0, 1.0),
            ("latent_denoise", 1.0, 0.0),
            ("hybrid_flow_ce", 0.5, 0.5),
        ]:
            regional(
                plan,
                f"dr12_objective_{mode}_d512",
                "Future module: compare supervised CE with ELF-like self-supervised latent denoising.",
                num_experts=16,
                expert_size=32,
                iterations=8,
                activation_passes=4,
                max_steps=4000,
                **objective_args(mode=mode, denoise=denoise, ce=ce),
            )
        if base.include_plan_size(plan_size, "full"):
            for denoise, ce in [(0.25, 0.75), (0.75, 0.25)]:
                regional(
                    plan,
                    base.ctm_name("dr12_hybrid_mix", denoise=denoise, ce=ce),
                    "Future module: hybrid denoise/decoder CE mixing ratio.",
                    num_experts=16,
                    expert_size=32,
                    iterations=8,
                    activation_passes=4,
                    max_steps=4000,
                    **objective_args(
                        mode="hybrid_flow_ce",
                        denoise=denoise,
                        ce=ce,
                    ),
                )
            for latent in ["token_embed", "ctm_hidden", "frozen_encoder"]:
                regional(
                    plan,
                    f"dr12_latent_{latent}_d512",
                    "Future module: latent space choice for CTM denoising objective.",
                    num_experts=16,
                    expert_size=32,
                    iterations=8,
                    activation_passes=4,
                    max_steps=4000,
                    **objective_args(
                        mode="hybrid_flow_ce",
                        denoise=0.5,
                        ce=0.5,
                        latent=latent,
                    ),
                )
            regional(
                plan,
                "dr12_objective_plus_draft_revise_b4_d512",
                "Future combined stack: ELF-like objective plus CTM draft-revise block generation.",
                num_experts=16,
                expert_size=32,
                iterations=12,
                activation_passes=4,
                max_steps=5000,
                batch_size=5,
                **objective_args(
                    mode="hybrid_flow_ce",
                    denoise=0.5,
                    ce=0.5,
                    latent="ctm_hidden",
                    self_cond=0.5,
                ),
                **draft_args(
                    mode="revise",
                    block=4,
                    head="slot_adapter",
                    loss=0.15,
                    revise=0.20,
                    corrupt=0.30,
                    commit=0.05,
                    rounds=2,
                    carry=1,
                    curriculum="linear",
                ),
            )

    return base.validate_plan(plan)


base.configure_plan_defaults(metrics_prefix="draft_revise", dispatch_block_sparse=False)
base.REGIONAL_STAGES = DRAFT_STAGES
base.REGIONAL_PREFIXES = DRAFT_PREFIXES
base.build_plan = build_plan


def normalize_default_outputs(args):
    default_outputs = {
        "runs/experiment_plans/impl_validation_plan.csv":
            "runs/experiment_plans/draft_revise_plan.csv",
        "runs/experiment_plans/impl_validation_batch_tune_plan.csv":
            "runs/experiment_plans/draft_revise_batch_tune_plan.csv",
        "runs/metrics/impl_validation_batch_probe_report.csv":
            "runs/metrics/draft_revise_batch_probe_report.csv",
    }
    output = getattr(args, "output", None)
    if output in default_outputs:
        args.output = default_outputs[output]


def audit_readiness():
    labels = {
        "dr00": "runnable: sparse/async anchors",
        "dr01": "runnable: current shared-head MTP/ELF negative controls",
        "dr02": "future: parallel draft slots + slot-aware heads",
        "dr03": "future: overwrite/revise corruption training",
        "dr04": "future: commit/confidence head",
        "dr05": "future: draft attention mask + state carry",
        "dr06": "future: full sparse async draft-revise stack",
        "dr07": "future: DINO-like continuous learning-speed spectrum",
        "dr08": "future: residual compute semantic-preserving caches/deltas",
        "dr09": "future: synapse block residual true skipping",
        "dr10": "future: recursive NLM fast path",
        "dr11": "future: full/residual/stop tick controller + speed cells",
        "dr12": "future: ELF-like self-supervised denoising objective",
    }
    for st in DRAFT_STAGES:
        if st in {"all", "real"}:
            continue
        plan = build_plan(st, "full")
        print(f"{st}: {len(plan):3d} {labels[st]}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "audit-readiness":
        audit_readiness()
    else:
        args = base.parse_args()
        normalize_default_outputs(args)
        args.func(args)
