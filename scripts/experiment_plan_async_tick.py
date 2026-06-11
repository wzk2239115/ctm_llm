#!/usr/bin/env python3
"""Async-tick / brain-like output experiment plan.

Research question: how can CTM guarantee usable output at a steady cadence
(like ~1 Hz conscious broadcast) while deeper thought continues asynchronously?

Current implementation status:
- REAL: resident reflex head, anytime tick-head supervision, tick halt,
  differentiated cell memory windows, slow-to-fast distillation.
- REAL (at07+): banded async tick clocks with per-band periods/phases and
  optional fast-band output blend.

This plan uses only runnable training knobs. Stages progress from anchors to
combined fast/slow/async-proxy stacks.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import experiment_plan_impl_validation as base


ASYNC_TICK_STAGES = (
    "at00",
    "at01",
    "at02",
    "at03",
    "at04",
    "at05",
    "at06",
    "at07",
    "real",
    "all",
)
ASYNC_TICK_PREFIXES = tuple(
    f"{stage}_" for stage in ASYNC_TICK_STAGES if stage != "all"
)

DIFF_COMMON = dict(
    diff_cell_mode="learned",
    diff_cell_temperature=0.7,
    diff_cell_capacity_weight=2e-3,
    diff_cell_memory_weight=1e-3,
    diff_cell_diversity_weight=1e-3,
)


def bs_for(
    d_model,
    *,
    halt=False,
    fastslow=False,
    async_tick=False,
    iterations=4,
    elf_horizon=4,
    mtp=False,
):
    if d_model <= 512:
        batch = 12
    elif d_model <= 768:
        batch = 10
    elif d_model <= 1024:
        batch = 8 if halt or fastslow else 10
    elif d_model <= 1536:
        batch = 6
    else:
        batch = 4
    if halt:
        batch = max(2, batch - 2)
    if fastslow or async_tick:
        batch = max(2, batch - 2)
    if mtp:
        batch = max(2, batch - 1)
    if iterations >= 8:
        batch = max(2, batch - 2)
    if iterations >= 16 and (fastslow or async_tick or mtp):
        batch = max(2, batch - 2)
    if elf_horizon >= 8:
        batch = max(2, batch - 2)
    floor = 4 if d_model <= 512 else (3 if d_model <= 1024 else 2)
    return max(floor, batch)


def regional(
    plan,
    name,
    question,
    *,
    num_experts=16,
    expert_size=32,
    max_steps=3000,
    iterations=8,
    activation_passes=4,
    topk_experts=1,
    shared_experts=1,
    halt_mode="none",
    halt_threshold=0.65,
    mtp_mode="none",
    mtp_horizons="",
    **kwargs,
):
    d_model = num_experts * expert_size
    halt = halt_mode != "none"
    fastslow = (
        kwargs.get("fast_output_mode", "none") != "none"
        or float(kwargs.get("habit_output_weight", 0.0)) > 0
        or float(kwargs.get("slow_output_weight", 0.0)) > 0
    )
    async_tick = kwargs.get("async_tick_mode", "none") != "none"
    mtp = (mtp_mode or kwargs.get("moe_mtp_mode", "none")) not in ("none", "")
    batch_size = kwargs.pop("batch_size", None)
    if batch_size is None:
        batch_size = bs_for(
            d_model,
            halt=halt,
            fastslow=fastslow,
            async_tick=async_tick,
            iterations=iterations,
            elf_horizon=int(kwargs.get("elf_max_horizon", 4)),
            mtp=mtp,
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
        balance=kwargs.pop("balance", 1e-2),
        diversity=kwargs.pop("diversity", 1e-3 if activation_passes >= 3 else 0.0),
        iterations=iterations,
        tick_halt_mode=halt_mode,
        tick_halt_threshold=halt_threshold,
        tick_compute_weight=kwargs.pop(
            "tick_compute_weight", 1e-3 if halt else 0.0),
        moe_mtp_mode=mtp_mode,
        moe_mtp_horizons=mtp_horizons,
        max_steps=max_steps,
        batch_size=batch_size,
        **kwargs,
    )


def fastslow(
    plan,
    name,
    question,
    *,
    tag,
    mode,
    fast_w,
    habit_w,
    slow_w,
    ticks,
    distill,
    iterations=8,
    num_experts=16,
    expert_size=32,
    max_steps=3500,
    **kwargs,
):
    regional(
        plan,
        name,
        question,
        num_experts=num_experts,
        expert_size=expert_size,
        iterations=iterations,
        fast_output_mode=mode,
        fast_output_weight=fast_w,
        habit_output_weight=habit_w,
        slow_output_weight=slow_w,
        fast_output_ticks=ticks,
        fast_output_distill_weight=distill,
        mtp_mode="mtp_1_2_4",
        mtp_horizons="1,2,4",
        max_steps=max_steps,
        **kwargs,
    )


def build_plan(stage, plan_size="full"):
    plan = []

    if stage in ("at00", "all"):
        # Sync-tick anchors: baseline before async/fast-slow stacks.
        base.add_sparse_experiment(
            plan,
            "at00_dense_d512_tick2",
            "Dense sync-tick anchor (no async output path).",
            512,
            topk=512,
            iterations=2,
            synapse_depth=2,
            memory_hidden_dims=2,
            max_steps=120,
        )
        regional(
            plan,
            "at00_regional_d512_tick4",
            "Regional sparse sync-tick anchor for async-tick comparisons.",
            num_experts=16,
            expert_size=32,
            iterations=4,
            activation_passes=3,
            max_steps=120,
        )
        regional(
            plan,
            "at00_regional_d1024_tick4",
            "Larger regional anchor.",
            num_experts=16,
            expert_size=64,
            iterations=4,
            activation_passes=3,
            max_steps=120,
        )

    if stage in ("at01", "all"):
        # Reflex path: zero-tick resident head for guaranteed immediate output.
        for dtag, num_experts, expert_size, ticks, mem in [
            ("d512", 16, 32, 1, 4),
            ("d512", 16, 32, 2, 4),
            ("d1024", 16, 64, 1, 4),
        ]:
            regional(
                plan,
                f"at01_reflex_{dtag}_tick{ticks}_mem{mem}",
                "Reflex-only proxy: shortest tick budget + short memory for sub-second output.",
                num_experts=num_experts,
                expert_size=expert_size,
                iterations=ticks,
                memory_length=mem,
                synapse_depth=2,
                memory_hidden_dims=2,
                activation_passes=1 if ticks == 1 else 2,
                shared_experts=0,
                max_steps=2500,
                batch_size=bs_for(num_experts * expert_size) + 2,
            )
        for tag, fast_w, distill in [
            ("light", 0.10, 0.05),
            ("strong", 0.20, 0.10),
        ]:
            fastslow(
                plan,
                f"at01_reflex_head_d512_{tag}",
                "Real resident reflex head: embed adapter bypasses tick loop for instant logits.",
                tag=tag,
                mode="reflex",
                fast_w=fast_w,
                habit_w=0.0,
                slow_w=0.0,
                ticks="1",
                distill=distill,
                iterations=4,
                max_steps=3000,
            )

    if stage in ("at02", "all"):
        # Anytime tick supervision: train early ticks to be independently usable.
        anytime_grid = [
            ("tick1_only", "1", 1, 0.0, 0.0),
            ("tick1_4", "1,4", 4, 0.10, 0.05),
            ("tick1_4_8", "1,4,8", 8, 0.15, 0.10),
            ("tick1_4_8_12", "1,4,8,12", 12, 0.20, 0.15),
        ]
        for dtag, num_experts, expert_size in [
            ("d512", 16, 32),
            ("d1024", 16, 64),
        ]:
            for tag, ticks, iters, habit_w, distill in anytime_grid:
                fastslow(
                    plan,
                    f"at02_anytime_{dtag}_{tag}",
                    "Anytime output: supervise habit ticks so each cadence can emit without waiting for last tick.",
                    tag=tag,
                    mode="anytime",
                    fast_w=0.10,
                    habit_w=habit_w,
                    slow_w=0.05 if iters >= 8 else 0.0,
                    ticks=ticks,
                    distill=distill,
                    iterations=iters,
                    num_experts=num_experts,
                    expert_size=expert_size,
                    max_steps=3500,
                )

        for tag, elf, elf_max, improve in [
            ("elf_h1", "linear", 1, 0.0),
            ("elf_h2", "linear", 2, 0.01),
            ("elf_h4", "linear", 4, 0.02),
        ]:
            regional(
                plan,
                f"at02_anytime_elf_d512_{tag}",
                "Pair anytime habit ticks with ELF horizons so early ticks predict nearer tokens.",
                iterations=8,
                elf_horizon_mode=elf,
                elf_max_horizon=elf_max,
                tick_improve_weight=improve,
                fast_output_mode="anytime",
                fast_output_weight=0.10,
                habit_output_weight=0.15,
                slow_output_weight=0.05,
                fast_output_ticks="1,4,8",
                fast_output_distill_weight=0.10,
                mtp_mode="mtp_1_2",
                mtp_horizons="1,2",
                max_steps=3500,
            )

    if stage in ("at03", "all"):
        # Dynamic tick budget: halt when confident instead of waiting for full loop.
        halt_grid = [
            ("none_tick8", "none", 0.65, 0.0, 8),
            ("thr0p00_tick8", "threshold", 0.00, 1e-3, 8),
            ("thr0p15_tick8", "threshold", 0.15, 1e-3, 8),
            ("thr0p30_tick8", "threshold", 0.30, 2e-3, 8),
            ("thr0p45_tick8", "threshold", 0.45, 2e-3, 8),
            ("conf0p25_tick8", "confidence", 0.30, 1e-3, 8),
            ("thr0p30_tick12", "threshold", 0.30, 3e-3, 12),
        ]
        if base.include_plan_size(plan_size, "full"):
            halt_grid.extend([
                ("thr0p30_tick16", "threshold", 0.30, 3e-3, 16),
                ("conf0p15_tick12", "confidence", 0.30, 2e-3, 12),
            ])
        for dtag, num_experts, expert_size in [
            ("d512", 16, 32),
            ("d1024", 16, 64),
        ]:
            for tag, halt_mode, threshold, compute_w, ticks in halt_grid:
                regional(
                    plan,
                    f"at03_halt_{dtag}_{tag}",
                    "Variable-depth ticks: early exit when confidence crosses threshold (async depth proxy).",
                    num_experts=num_experts,
                    expert_size=expert_size,
                    iterations=ticks,
                    halt_mode=halt_mode,
                    halt_threshold=threshold,
                    tick_compute_weight=compute_w,
                    max_steps=3000,
                )

        for tag, halt_mode, threshold in [
            ("halt_anytime_thr0p30", "threshold", 0.30),
            ("halt_anytime_conf0p25", "confidence", 0.30),
        ]:
            regional(
                plan,
                f"at03_{tag}_d512",
                "Combine tick halt with anytime habit supervision for cadence + variable depth.",
                iterations=12,
                halt_mode=halt_mode,
                halt_threshold=threshold,
                tick_compute_weight=2e-3,
                fast_output_mode="anytime",
                fast_output_weight=0.10,
                habit_output_weight=0.20,
                slow_output_weight=0.10,
                fast_output_ticks="1,4,8,12",
                fast_output_distill_weight=0.10,
                mtp_mode="mtp_1_2_4",
                mtp_horizons="1,2,4",
                max_steps=4000,
            )

    if stage in ("at04", "all"):
        # Async-clock proxy: differentiated memory windows per expert group.
        memory_paths = [
            ("reflex", 4, 2, 1),
            ("habit", 8, 4, 4),
            ("delib", 16, 8, 8),
            ("slow", 32, 12, 12),
        ]
        for dtag, num_experts, expert_size in [
            ("d512", 16, 32),
            ("d1024", 16, 64),
        ]:
            for path, memory_length, ticks, passes in memory_paths:
                regional(
                    plan,
                    f"at04_memory_{dtag}_{path}_tick{ticks}_mem{memory_length}",
                    "Memory-timescale proxy: fast vs slow cell traces emulate heterogeneous clocks.",
                    num_experts=num_experts,
                    expert_size=expert_size,
                    memory_length=memory_length,
                    iterations=ticks,
                    activation_passes=passes,
                    shared_experts=0 if ticks <= 2 else 1,
                    tick_compute_weight=2e-3 if ticks >= 8 else 1e-3,
                    max_steps=3000,
                )

        diff_grids = [
            ("d512", 16, 32, "8,16,32", "4,8", 8),
            ("d1024", 16, 64, "16,32,64", "4,8,16", 16),
        ]
        if base.include_plan_size(plan_size, "wide"):
            diff_grids.append(("d1536", 24, 64, "16,32,64", "4,8,16", 16))
        for dtag, num_experts, expert_size, widths, memories, memory_length in diff_grids:
            regional(
                plan,
                f"at04_diffcell_{dtag}_p4_sh1_top1",
                "Real learned differentiated cells: per-expert width/memory as async clock bands.",
                num_experts=num_experts,
                expert_size=expert_size,
                activation_passes=4,
                memory_length=memory_length,
                diff_cell_widths=widths,
                diff_cell_memory_lengths=memories,
                iterations=8,
                max_steps=4000,
                **DIFF_COMMON,
            )

    if stage in ("at05", "all"):
        # Combined brain-like stack: reflex + anytime + halt + diff cell + slow compile.
        stacks = [
            (
                "brain_light",
                "anytime",
                0.10,
                0.15,
                0.05,
                "1,4,8",
                0.05,
                4,
                "threshold",
                0.30,
                8,
                False,
            ),
            (
                "brain_balanced",
                "anytime",
                0.15,
                0.20,
                0.10,
                "1,4,8,12",
                0.10,
                4,
                "threshold",
                0.30,
                12,
                True,
            ),
            (
                "brain_strong",
                "anytime",
                0.20,
                0.25,
                0.15,
                "1,4,8,12,16",
                0.15,
                5,
                "confidence",
                0.30,
                16,
                True,
            ),
            (
                "brain_reflex_mix",
                "anytime",
                0.20,
                0.20,
                0.10,
                "1,4,8,12",
                0.15,
                4,
                "threshold",
                0.15,
                12,
                True,
            ),
        ]
        for dtag, num_experts, expert_size, widths, memories, memory_length in [
            ("d512", 16, 32, "8,16,32", "4,8", 8),
            ("d1024", 16, 64, "16,32,64", "4,8,16", 16),
        ]:
            for (
                tag,
                mode,
                fast_w,
                habit_w,
                slow_w,
                ticks,
                distill,
                passes,
                halt_mode,
                halt_thr,
                iters,
                use_diff,
            ) in stacks:
                kwargs = dict(
                    num_experts=num_experts,
                    expert_size=expert_size,
                    iterations=iters,
                    activation_passes=passes,
                    halt_mode=halt_mode,
                    halt_threshold=halt_thr,
                    tick_compute_weight=2e-3,
                    fast_output_mode=mode,
                    fast_output_weight=fast_w,
                    habit_output_weight=habit_w,
                    slow_output_weight=slow_w,
                    fast_output_ticks=ticks,
                    fast_output_distill_weight=distill,
                    mtp_mode="mtp_1_2_4",
                    mtp_horizons="1,2,4",
                    tick_improve_weight=0.03 if iters >= 8 else 0.01,
                    memory_length=memory_length,
                    max_steps=4500,
                )
                if use_diff:
                    kwargs.update(
                        diff_cell_widths=widths,
                        diff_cell_memory_lengths=memories,
                        **DIFF_COMMON,
                    )
                regional(
                    plan,
                    f"at05_brain_{dtag}_{tag}",
                    "Full async-output proxy: reflex/anytime heads + halt + multi-timescale cells.",
                    **kwargs,
                )

    if stage in ("at06", "all"):
        # Confirmation: pick best candidates from at01-at05 for longer runs.
        confirm = [
            (
                "confirm_reflex_d512",
                dict(
                    iterations=4,
                    memory_length=4,
                    fast_output_mode="reflex",
                    fast_output_weight=0.15,
                    fast_output_distill_weight=0.10,
                ),
            ),
            (
                "confirm_anytime_d512",
                dict(
                    iterations=12,
                    fast_output_mode="anytime",
                    fast_output_weight=0.15,
                    habit_output_weight=0.20,
                    slow_output_weight=0.10,
                    fast_output_ticks="1,4,8,12",
                    fast_output_distill_weight=0.10,
                    mtp_mode="mtp_1_2_4",
                    mtp_horizons="1,2,4",
                    tick_improve_weight=0.03,
                ),
            ),
            (
                "confirm_halt_anytime_d512",
                dict(
                    iterations=12,
                    halt_mode="threshold",
                    halt_threshold=0.30,
                    tick_compute_weight=2e-3,
                    fast_output_mode="anytime",
                    fast_output_weight=0.15,
                    habit_output_weight=0.20,
                    slow_output_weight=0.10,
                    fast_output_ticks="1,4,8,12",
                    fast_output_distill_weight=0.10,
                ),
            ),
            (
                "confirm_brain_d512",
                dict(
                    iterations=12,
                    activation_passes=4,
                    halt_mode="threshold",
                    halt_threshold=0.30,
                    tick_compute_weight=2e-3,
                    fast_output_mode="anytime",
                    fast_output_weight=0.15,
                    habit_output_weight=0.20,
                    slow_output_weight=0.10,
                    fast_output_ticks="1,4,8,12",
                    fast_output_distill_weight=0.10,
                    diff_cell_widths="8,16,32",
                    diff_cell_memory_lengths="4,8",
                    memory_length=8,
                    mtp_mode="mtp_1_2_4",
                    mtp_horizons="1,2,4",
                    tick_improve_weight=0.03,
                    **DIFF_COMMON,
                ),
            ),
        ]
        steps = 5000 if base.include_plan_size(plan_size, "full") else 3500
        for tag, overrides in confirm:
            regional(
                plan,
                f"at06_{tag}",
                "Long confirmation run for async-tick candidate.",
                num_experts=16,
                expert_size=32,
                max_steps=steps,
                batch_size=8,
                **overrides,
            )

    if stage in ("at07", "all"):
        async_grids = [
            ("p1248", "1,2,4,8", "", 0.0),
            ("p1248_fastblend", "1,2,4,8", "", 0.25),
            ("p14816", "1,4,8,16", "", 0.0),
            ("p1248_phased", "1,2,4,8", "0,0,1,2", 0.0),
        ]
        for dtag, num_experts, expert_size in [
            ("d512", 16, 32),
            ("d1024", 16, 64),
        ]:
            for tag, periods, phases, fast_blend in async_grids:
                regional(
                    plan,
                    f"at07_async_{dtag}_{tag}",
                    "Real banded async tick: fast band every tick, slow bands on longer periods.",
                    num_experts=num_experts,
                    expert_size=expert_size,
                    iterations=16,
                    activation_passes=4,
                    async_tick_mode="banded",
                    async_tick_periods=periods,
                    async_tick_phases=phases,
                    async_fast_band=0,
                    async_fast_output_weight=fast_blend,
                    max_steps=3500,
                )

        for tag, periods, fast_blend, habit_w in [
            ("async_anytime", "1,2,4,8", 0.15, 0.20),
            ("async_anytime_strong", "1,4,8,16", 0.25, 0.25),
        ]:
            regional(
                plan,
                f"at07_brain_{tag}_d512",
                "Banded async tick plus anytime habit supervision for steady cadence output.",
                num_experts=16,
                expert_size=32,
                iterations=16,
                activation_passes=4,
                async_tick_mode="banded",
                async_tick_periods=periods,
                async_fast_output_weight=fast_blend,
                fast_output_mode="anytime",
                fast_output_weight=0.10,
                habit_output_weight=habit_w,
                slow_output_weight=0.10,
                fast_output_ticks="1,4,8,12,16",
                fast_output_distill_weight=0.10,
                halt_mode="threshold",
                halt_threshold=0.30,
                tick_compute_weight=2e-3,
                mtp_mode="mtp_1_2_4",
                mtp_horizons="1,2,4",
                max_steps=4500,
            )

        regional(
            plan,
            "at07_confirm_async_brain_d512",
            "Long confirmation: banded async tick + anytime + halt.",
            num_experts=16,
            expert_size=32,
            iterations=16,
            activation_passes=4,
            async_tick_mode="banded",
            async_tick_periods="1,2,4,8",
            async_fast_output_weight=0.20,
            fast_output_mode="anytime",
            fast_output_weight=0.15,
            habit_output_weight=0.20,
            slow_output_weight=0.10,
            fast_output_ticks="1,4,8,12,16",
            fast_output_distill_weight=0.10,
            halt_mode="threshold",
            halt_threshold=0.30,
            tick_compute_weight=2e-3,
            diff_cell_widths="8,16,32",
            diff_cell_memory_lengths="4,8",
            memory_length=8,
            mtp_mode="mtp_1_2_4",
            mtp_horizons="1,2,4",
            max_steps=5000,
            **DIFF_COMMON,
        )

    return base.validate_plan(plan)


base.configure_plan_defaults(metrics_prefix="async_tick")
base.REGIONAL_STAGES = ASYNC_TICK_STAGES
base.REGIONAL_PREFIXES = ASYNC_TICK_PREFIXES
base.build_plan = build_plan


def audit_realism():
    labels = {
        "at00": "real: sync-tick anchors",
        "at01": "real: reflex path + resident reflex head",
        "at02": "real: anytime tick-head supervision + ELF pairing",
        "at03": "real: tick halt + halt+anytime combo",
        "at04": "mixed: memory proxy + real differentiated cells",
        "at05": "real: combined reflex/anytime/halt/diff-cell stack",
        "at06": "real: long confirmation candidates",
        "at07": "real: banded async tick clocks + brain combos",
    }
    for st in ASYNC_TICK_STAGES:
        if st in {"all", "real"}:
            continue
        plan = build_plan(st, "full")
        print(f"{st}: {len(plan):3d} {labels[st]}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "audit-realism":
        audit_realism()
    else:
        args = base.parse_args()
        args.func(args)
