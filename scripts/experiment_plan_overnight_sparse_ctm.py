#!/usr/bin/env python3
"""Large overnight CTM sparse-compute experiment plan.

This plan intentionally uses only knobs already implemented in the training
stack. Several pasted research ideas, such as heterogeneous nested cells,
Sinkhorn routing, Hebbian assemblies, and oscillatory gates, still require new
model code. The experiments below cover runnable proxies for the same
questions: capacity gradient, variable active compute, dynamic ticks, routing
regularization, dispatch capacity, and multi-horizon supervision.
"""

import experiment_plan_impl_validation as base


OVERNIGHT_STAGES = (
    "og00",
    "og01",
    "og02",
    "og03",
    "og04",
    "og05",
    "og06",
    "og07",
    "og08",
    "og09",
    "og10",
    "real",
    "all",
)
OVERNIGHT_PREFIXES = tuple(f"{stage}_" for stage in OVERNIGHT_STAGES if stage != "all")


def bs_for(d_model, *, halt=False, long=False):
    if d_model <= 512:
        batch = 12
    elif d_model <= 768:
        batch = 10
    elif d_model <= 1024:
        batch = 8 if halt else 10
    elif d_model <= 1536:
        batch = 6
    else:
        batch = 4
    if long and d_model >= 1536:
        batch = min(batch, 4)
    return batch


def regional(plan, name, question, *, num_experts, expert_size, max_steps=3500,
             iterations=2, activation_passes=4, topk_experts=1,
             shared_experts=1, halt_mode="none", halt_threshold=0.65,
             mtp_mode="none", mtp_horizons="", **kwargs):
    d_model = num_experts * expert_size
    halt = halt_mode != "none"
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
        tick_compute_weight=kwargs.pop("tick_compute_weight", 1e-3 if halt else 0.0),
        moe_mtp_mode=mtp_mode,
        moe_mtp_horizons=mtp_horizons,
        synapse_depth=kwargs.pop("synapse_depth", 2),
        memory_hidden_dims=kwargs.pop("memory_hidden_dims", 2),
        memory_length=kwargs.pop("memory_length", 8),
        max_steps=max_steps,
        batch_size=kwargs.pop("batch_size", bs_for(d_model, halt=halt, long=max_steps >= 4000)),
        **kwargs,
    )


def single_moe(plan, name, question, *, num_experts, expert_size, topk_experts,
               routing="topk", max_steps=3500, **kwargs):
    d_model = num_experts * expert_size
    base.add_moe_experiment(
        plan,
        name,
        question,
        num_experts=num_experts,
        expert_size=expert_size,
        topk_experts=topk_experts,
        routing=routing,
        dispatch=kwargs.pop("dispatch", "dense_mask"),
        moe_load_balance_weight=kwargs.pop("moe_load_balance_weight", 1e-2),
        iterations=kwargs.pop("iterations", 2),
        synapse_depth=kwargs.pop("synapse_depth", 2),
        memory_hidden_dims=kwargs.pop("memory_hidden_dims", 2),
        max_steps=max_steps,
        batch_size=kwargs.pop("batch_size", bs_for(d_model)),
        **kwargs,
    )


def build_plan(stage, plan_size="full"):
    plan = []
    real_stages = (
        "og00", "og01", "og02", "og03", "og04",
        "og05", "og06", "og07", "og10",
    )
    if stage == "real":
        for item in real_stages:
            plan.extend(build_plan(item, plan_size))
        return base.validate_plan(plan)

    if stage in ("og00", "all"):
        anchors = [
            ("dense_d512_tick2", 512, 512, 2, 2500),
            ("dense_d1024_tick2", 1024, 1024, 2, 2500),
            ("topk_d1024_k128", 1024, 128, 2, 6000),
            ("topk_d1024_k256", 1024, 256, 2, 6000),
        ]
        for tag, d_model, topk, iterations, steps in anchors:
            base.add_sparse_experiment(
                plan,
                f"og00_anchor_{tag}",
                "Anchor dense/post-activation sparse CTM for overnight comparisons.",
                d_model,
                topk=topk,
                iterations=iterations,
                synapse_depth=2,
                memory_hidden_dims=2,
                max_steps=steps,
                batch_size=bs_for(d_model),
            )
        regional(
            plan, "og00_anchor_regional_d1024_p4_shared1_top1",
            "Implementation-validation regional winner anchor.",
            num_experts=16, expert_size=64, activation_passes=4,
            shared_experts=1, topk_experts=1, max_steps=3500)
        regional(
            plan, "og00_anchor_regional_d512_p4_halt0p30_mtp124",
            "Composite dynamic tick plus MTP anchor.",
            num_experts=16, expert_size=32, activation_passes=4,
            shared_experts=1, topk_experts=1, iterations=4,
            halt_mode="threshold", halt_threshold=0.30,
            mtp_mode="mtp_1_2_4", mtp_horizons="1,2,4", max_steps=3500)

    if stage in ("og01", "all"):
        # Capacity-gradient proxy: same or larger d_model, different physical
        # expert granularity and active width. This is the closest runnable
        # proxy for heterogeneous/nested cells.
        grids = [
            (32, 16, 4, 1, 1), (32, 16, 4, 2, 1),
            (16, 32, 4, 1, 1), (16, 32, 4, 2, 1),
            (16, 64, 4, 1, 1), (16, 64, 4, 2, 1),
            (8, 128, 4, 1, 1), (8, 128, 4, 1, 2),
            (24, 64, 3, 1, 1), (24, 64, 4, 1, 1),
            (32, 64, 3, 1, 1), (32, 64, 4, 1, 1),
        ]
        for num_experts, expert_size, passes, topk, shared in grids:
            d_model = num_experts * expert_size
            regional(
                plan,
                f"og01_capacity_d{d_model}_e{num_experts}_s{expert_size}_p{passes}_sh{shared}_top{topk}",
                "Capacity-gradient proxy: compare expert granularity and active width.",
                num_experts=num_experts, expert_size=expert_size,
                activation_passes=passes, shared_experts=shared,
                topk_experts=topk, max_steps=3500)

    if stage in ("og02", "all"):
        # Variable active compute proxy: sweep active expert count from tiny to
        # wide at fixed model sizes.
        for dtag, num_experts, expert_size in [
            ("d512", 16, 32), ("d1024", 16, 64), ("d1536", 24, 64)
        ]:
            for shared, topk, passes in [
                (0, 1, 2), (1, 1, 3), (1, 1, 4), (2, 1, 4),
                (1, 2, 3), (1, 2, 4), (2, 2, 4), (1, 3, 3)
            ]:
                regional(
                    plan,
                    f"og02_active_{dtag}_sh{shared}_top{topk}_p{passes}",
                    "Variable Top-K active compute proxy at fixed capacity.",
                    num_experts=num_experts, expert_size=expert_size,
                    shared_experts=shared, topk_experts=topk,
                    activation_passes=passes, max_steps=3000)

    if stage in ("og03", "all"):
        for dtag, num_experts, expert_size in [("d512", 16, 32), ("d1024", 16, 64)]:
            for iterations in [4, 6, 8]:
                regional(
                    plan,
                    f"og03_halt_{dtag}_none_tick{iterations}",
                    "No-halt control for dynamic tick budget.",
                    num_experts=num_experts, expert_size=expert_size,
                    activation_passes=4, iterations=iterations,
                    max_steps=3000)
                for threshold in [0.15, 0.30, 0.45, 0.60]:
                    regional(
                        plan,
                        f"og03_halt_{dtag}_thr{str(threshold).replace('.', 'p')}_tick{iterations}",
                        "Dynamic tick halting threshold sweep.",
                        num_experts=num_experts, expert_size=expert_size,
                        activation_passes=4, iterations=iterations,
                        halt_mode="threshold", halt_threshold=threshold,
                        tick_compute_weight=2e-3 if iterations >= 6 else 1e-3,
                        max_steps=3000)

    if stage in ("og04", "all"):
        regs = [
            ("bal0", 0.0, 0.0, 0.0, 0.0, 0),
            ("bal1e2", 1e-2, 0.0, 0.0, 0.0, 0),
            ("bal3e2", 3e-2, 0.0, 0.0, 0.0, 0),
            ("ent1e3", 1e-2, 1e-3, 0.0, 0.0, 0),
            ("ent3e3", 1e-2, 3e-3, 0.0, 0.0, 0),
            ("z1e3", 1e-2, 0.0, 1e-3, 0.0, 0),
            ("div1e3", 1e-2, 0.0, 0.0, 1e-3, 0),
            ("div3e3", 1e-2, 0.0, 0.0, 3e-3, 0),
            ("drop5", 1e-2, 0.0, 0.0, 1e-3, 0.05),
            ("drop10", 1e-2, 0.0, 0.0, 1e-3, 0.10),
        ]
        for tag, balance, entropy, z_loss, diversity, dropout in regs:
            regional(
                plan,
                f"og04_router_reg_{tag}_d1024_p4",
                "Routing regularization and cell specialization proxy.",
                num_experts=16, expert_size=64,
                activation_passes=4, shared_experts=1, topk_experts=1,
                balance=balance, diversity=diversity,
                moe_router_entropy_weight=entropy,
                moe_router_z_loss_weight=z_loss,
                moe_expert_dropout=dropout,
                max_steps=3500)
        for routing in ["topk", "top1", "top2", "expert_choice", "hash", "topk_warmup"]:
            single_moe(
                plan,
                f"og04_router_single_{routing}_d1024",
                "Single-pass routing family control for specialization behavior.",
                num_experts=16, expert_size=64, topk_experts=2,
                routing=routing,
                moe_topk_warmup_steps=500 if routing == "topk_warmup" else 0,
                max_steps=3000)

    if stage in ("og05", "all"):
        dispatches = [
            ("block", "block_sparse", 1.0, 0),
            ("dropless", "dropless", 1.0, 0),
            ("cap075", "capacity_drop", 0.75, 1),
            ("cap100", "capacity_drop", 1.0, 1),
            ("cap125", "capacity_drop", 1.25, 1),
            ("cap150", "capacity_drop", 1.5, 1),
        ]
        for dtag, num_experts, expert_size in [("d512", 16, 32), ("d1024", 16, 64)]:
            for tag, dispatch, capacity, drop in dispatches:
                regional(
                    plan,
                    f"og05_dispatch_{dtag}_{tag}",
                    "Capacity/drop dispatch proxy for token-budgeted sparse compute.",
                    num_experts=num_experts, expert_size=expert_size,
                    activation_passes=4, shared_experts=1, topk_experts=1,
                    moe_dispatch_mode=dispatch,
                    moe_capacity_factor=capacity,
                    moe_drop_tokens=drop,
                    max_steps=3000)

    if stage in ("og06", "all"):
        mtp_grid = [
            ("none", "none", "", "none", 4, 0.0),
            ("elf_h2", "none", "", "linear", 2, 0.01),
            ("elf_h4", "none", "", "linear", 4, 0.03),
            ("elf_pow2_h4", "none", "", "pow2", 4, 0.03),
            ("mtp12", "mtp_1_2", "1,2", "none", 4, 0.0),
            ("mtp124", "mtp_1_2_4", "1,2,4", "none", 4, 0.0),
            ("mtp1234", "mtp_1_2_3_4", "1,2,3,4", "none", 4, 0.0),
            ("mtp1248", "mtp_1_2_4_8", "1,2,4,8", "none", 8, 0.0),
            ("mtp124_improve", "mtp_1_2_4", "1,2,4", "none", 4, 0.03),
            ("mtp124_elf", "mtp_1_2_4", "1,2,4", "linear", 4, 0.03),
        ]
        for dtag, num_experts, expert_size in [("d512", 16, 32), ("d1024", 16, 64)]:
            for tag, mtp, horizons, elf, elf_max, improve in mtp_grid:
                regional(
                    plan,
                    f"og06_mtp_{dtag}_{tag}",
                    "Multi-horizon supervision and improvement regularization sweep.",
                    num_experts=num_experts, expert_size=expert_size,
                    activation_passes=4, shared_experts=1, topk_experts=1,
                    mtp_mode=mtp, mtp_horizons=horizons,
                    elf_horizon_mode=elf, elf_max_horizon=elf_max,
                    tick_improve_weight=improve,
                    max_steps=3500)

    if stage in ("og07", "all"):
        confirms = [
            ("d512_p4_plain", 16, 32, 4, "none", 0.65, "none", "", 9000),
            ("d512_p4_halt030", 16, 32, 4, "threshold", 0.30, "none", "", 9000),
            ("d512_p4_mtp124", 16, 32, 4, "none", 0.65, "mtp_1_2_4", "1,2,4", 9000),
            ("d512_p4_halt030_mtp124", 16, 32, 4, "threshold", 0.30, "mtp_1_2_4", "1,2,4", 9000),
            ("d1024_p4_plain", 16, 64, 4, "none", 0.65, "none", "", 9000),
            ("d1024_p4_mtp124", 16, 64, 4, "none", 0.65, "mtp_1_2_4", "1,2,4", 9000),
            ("d1024_p4_halt030", 16, 64, 4, "threshold", 0.30, "none", "", 9000),
            ("d1536_p4_plain", 24, 64, 4, "none", 0.65, "none", "", 7500),
            ("d1536_p4_mtp124", 24, 64, 4, "none", 0.65, "mtp_1_2_4", "1,2,4", 7500),
            ("d2048_p3_plain", 32, 64, 3, "none", 0.65, "none", "", 6000),
        ]
        for tag, num_experts, expert_size, passes, halt, threshold, mtp, horizons, steps in confirms:
            regional(
                plan,
                f"og07_confirm_{tag}",
                "Long overnight confirmation run for strongest sparse-compute candidates.",
                num_experts=num_experts, expert_size=expert_size,
                activation_passes=passes, shared_experts=1,
                topk_experts=1, iterations=4 if halt != "none" else 2,
                halt_mode=halt, halt_threshold=threshold,
                mtp_mode=mtp, mtp_horizons=horizons,
                max_steps=steps)

    if stage in ("og08", "all"):
        # Sparse-Delta / Keyframe-Delta proxy. The actual delta cache, sticky
        # routing, and per-cell freezing need model code. These runs keep rich
        # internal time but restrict per-tick active compute with low top-k,
        # periodic-refresh-like pass counts, and halt/freezing-style thresholds.
        tick_grids = [
            ("d512", 16, 32, 8),
            ("d512", 16, 32, 12),
            ("d512", 16, 32, 16),
            ("d1024", 16, 64, 8),
            ("d1024", 16, 64, 12),
            ("d1024", 16, 64, 16),
        ]
        for dtag, num_experts, expert_size, ticks in tick_grids:
            for passes, shared, topk in [
                (1, 0, 1),
                (2, 0, 1),
                (3, 1, 1),
                (4, 1, 1),
            ]:
                regional(
                    plan,
                    f"og08_delta_time_{dtag}_tick{ticks}_p{passes}_sh{shared}_top{topk}",
                    "Keyframe-Delta proxy: preserve many ticks while limiting active cells per tick.",
                    num_experts=num_experts,
                    expert_size=expert_size,
                    activation_passes=passes,
                    shared_experts=shared,
                    topk_experts=topk,
                    iterations=ticks,
                    tick_compute_weight=2e-3 if ticks >= 12 else 1e-3,
                    max_steps=3000,
                    batch_size=8 if dtag == "d512" else 6,
                )
        for dtag, num_experts, expert_size, ticks in [
            ("d512", 16, 32, 12),
            ("d512", 16, 32, 16),
            ("d1024", 16, 64, 12),
            ("d1024", 16, 64, 16),
        ]:
            for threshold in [0.15, 0.30, 0.45]:
                regional(
                    plan,
                    f"og08_delta_halt_{dtag}_tick{ticks}_thr{str(threshold).replace('.', 'p')}",
                    "Cell-freezing proxy: long internal time plus early tick exit pressure.",
                    num_experts=num_experts,
                    expert_size=expert_size,
                    activation_passes=4,
                    shared_experts=1,
                    topk_experts=1,
                    iterations=ticks,
                    halt_mode="threshold",
                    halt_threshold=threshold,
                    tick_compute_weight=3e-3,
                    max_steps=3000,
                    batch_size=8 if dtag == "d512" else 6,
                )
        for dtag, num_experts, expert_size, ticks, mtp, horizons in [
            ("d512", 16, 32, 12, "mtp_1_2_4", "1,2,4"),
            ("d512", 16, 32, 16, "mtp_1_2_4_8", "1,2,4,8"),
            ("d1024", 16, 64, 12, "mtp_1_2_4", "1,2,4"),
            ("d1024", 16, 64, 16, "mtp_1_2_4_8", "1,2,4,8"),
        ]:
            regional(
                plan,
                f"og08_delta_mtp_{dtag}_tick{ticks}_{mtp.replace('_', '')}",
                "Cache-consistency proxy: long internal time with multi-horizon supervision.",
                num_experts=num_experts,
                expert_size=expert_size,
                activation_passes=4,
                shared_experts=1,
                topk_experts=1,
                iterations=ticks,
                mtp_mode=mtp,
                mtp_horizons=horizons,
                tick_improve_weight=0.03,
                max_steps=3500,
                batch_size=8 if dtag == "d512" else 6,
            )

    if stage in ("og09", "all"):
        # Differentiated Fast-Slow / Developmental CTM proxy. True learned cell
        # growth, lineage splitting, asynchronous clocks, resident reflex cells,
        # and slow-to-fast head distillation require model changes. These runs
        # use existing knobs to probe memory scale differentiation, anytime
        # latency targets, reflex/habit/deliberative compute, and counterfactual
        # high-compute recruitment.
        for d_model, topk, ticks, memory_length, tag in [
            (256, 256, 1, 4, "reflex_d256_tick1_mem4"),
            (512, 128, 1, 4, "reflex_d512_k128_tick1_mem4"),
            (512, 256, 2, 4, "reflex_d512_k256_tick2_mem4"),
            (512, 512, 2, 8, "fast_dense_d512_tick2_mem8"),
        ]:
            base.add_sparse_experiment(
                plan,
                f"og09_fastpath_{tag}",
                "Reflex-path proxy: very low latency with short memory and small active width.",
                d_model,
                topk=topk,
                iterations=ticks,
                memory_length=memory_length,
                synapse_depth=2,
                memory_hidden_dims=2,
                max_steps=3500,
                batch_size=12,
            )

        for dtag, num_experts, expert_size in [
            ("d512", 16, 32),
            ("d1024", 16, 64),
        ]:
            for memory_length, ticks, path in [
                (4, 2, "reflex"),
                (8, 4, "habit"),
                (16, 8, "delib"),
                (32, 12, "slow"),
                (64, 16, "very_slow"),
            ]:
                regional(
                    plan,
                    f"og09_memory_{dtag}_{path}_tick{ticks}_mem{memory_length}",
                    "Memory-timescale differentiation proxy across fast, habitual, and deliberative paths.",
                    num_experts=num_experts,
                    expert_size=expert_size,
                    activation_passes=2 if ticks <= 4 else 4,
                    shared_experts=0 if ticks <= 2 else 1,
                    topk_experts=1,
                    iterations=ticks,
                    memory_length=memory_length,
                    tick_compute_weight=3e-3 if ticks >= 12 else 1e-3,
                    max_steps=3000,
                    batch_size=8 if dtag == "d512" else 6,
                )

        anytime = [
            ("early_tick1", 1, "none", "", "none", 4, 0.0),
            ("habit_tick4_elf_h2", 4, "none", "", "linear", 2, 0.01),
            ("habit_tick4_mtp12", 4, "mtp_1_2", "1,2", "none", 4, 0.0),
            ("delib_tick8_mtp124", 8, "mtp_1_2_4", "1,2,4", "none", 4, 0.03),
            ("delib_tick12_mtp1248", 12, "mtp_1_2_4_8", "1,2,4,8", "none", 8, 0.03),
            ("delib_tick16_mtp1248", 16, "mtp_1_2_4_8", "1,2,4,8", "none", 8, 0.05),
        ]
        for dtag, num_experts, expert_size in [
            ("d512", 16, 32),
            ("d1024", 16, 64),
        ]:
            for tag, ticks, mtp, horizons, elf, elf_max, improve in anytime:
                regional(
                    plan,
                    f"og09_anytime_{dtag}_{tag}",
                    "Anytime-output proxy: supervise usable early ticks and slower corrective ticks.",
                    num_experts=num_experts,
                    expert_size=expert_size,
                    activation_passes=4 if ticks >= 4 else 1,
                    shared_experts=1 if ticks >= 4 else 0,
                    topk_experts=1,
                    iterations=ticks,
                    mtp_mode=mtp,
                    mtp_horizons=horizons,
                    elf_horizon_mode=elf,
                    elf_max_horizon=elf_max,
                    tick_improve_weight=improve,
                    tick_compute_weight=2e-3 if ticks >= 8 else 1e-3,
                    max_steps=3500,
                    batch_size=8 if dtag == "d512" else 6,
                )

        for tag, num_experts, expert_size, passes, shared, topk, ticks, diversity in [
            ("surprise_low_compute_d512", 16, 32, 2, 0, 1, 4, 0.0),
            ("surprise_recruit_d512", 16, 32, 4, 1, 2, 8, 1e-3),
            ("counterfactual_d512", 16, 32, 5, 2, 2, 12, 3e-3),
            ("surprise_low_compute_d1024", 16, 64, 2, 0, 1, 4, 0.0),
            ("surprise_recruit_d1024", 16, 64, 4, 1, 2, 8, 1e-3),
            ("counterfactual_d1024", 16, 64, 5, 2, 2, 12, 3e-3),
            ("competing_thoughts_d1536", 24, 64, 5, 2, 2, 12, 3e-3),
            ("competing_thoughts_d2048", 32, 64, 4, 2, 2, 12, 3e-3),
        ]:
            regional(
                plan,
                f"og09_recruit_{tag}",
                "Surprise/counterfactual recruitment proxy: compare low compute to larger competing assemblies.",
                num_experts=num_experts,
                expert_size=expert_size,
                activation_passes=passes,
                shared_experts=shared,
                topk_experts=topk,
                iterations=ticks,
                diversity=diversity,
                mtp_mode="mtp_1_2_4",
                mtp_horizons="1,2,4",
                tick_improve_weight=0.03,
                max_steps=3500,
                batch_size=bs_for(num_experts * expert_size, halt=False),
            )

    if stage in ("og10", "all"):
        # Real implementation runs: learned differentiated expert width/memory
        # and true fast/anytime output losses are enabled here.
        diff_common = dict(
            diff_cell_mode="learned",
            diff_cell_temperature=0.7,
            diff_cell_capacity_weight=2e-3,
            diff_cell_memory_weight=1e-3,
            diff_cell_diversity_weight=1e-3,
        )
        for dtag, num_experts, expert_size, widths, memories, memory_length, steps in [
            ("d512", 16, 32, "8,16,32", "4,8", 8, 4000),
            ("d1024", 16, 64, "16,32,64", "4,8,16", 16, 4000),
            ("d1536", 24, 64, "16,32,64", "4,8,16", 16, 3500),
        ]:
            for passes, shared, topk in [(3, 1, 1), (4, 1, 1), (4, 1, 2)]:
                regional(
                    plan,
                    f"og10_diffcell_{dtag}_p{passes}_sh{shared}_top{topk}",
                    "Real differentiated cells: learned expert width and memory window with true sparse execution.",
                    num_experts=num_experts,
                    expert_size=expert_size,
                    activation_passes=passes,
                    shared_experts=shared,
                    topk_experts=topk,
                    memory_length=memory_length,
                    diff_cell_widths=widths,
                    diff_cell_memory_lengths=memories,
                    max_steps=steps,
                    batch_size=bs_for(num_experts * expert_size),
                    **diff_common,
                )

        for tag, mode, fast_w, habit_w, slow_w, ticks, distill in [
            ("reflex_only", "reflex", 0.20, 0.0, 0.0, "1", 0.05),
            ("anytime_light", "anytime", 0.10, 0.10, 0.0, "1,4", 0.05),
            ("anytime_balanced", "anytime", 0.15, 0.20, 0.10, "1,4,8", 0.10),
            ("anytime_slow_compile", "anytime", 0.20, 0.25, 0.15, "1,4,8,12", 0.15),
        ]:
            regional(
                plan,
                f"og10_fastslow_d512_{tag}",
                "Real fast/slow output: resident reflex head and anytime tick-head supervision.",
                num_experts=16,
                expert_size=32,
                activation_passes=4,
                shared_experts=1,
                topk_experts=1,
                iterations=12 if "slow" in tag else 8,
                fast_output_mode=mode,
                fast_output_weight=fast_w,
                habit_output_weight=habit_w,
                slow_output_weight=slow_w,
                fast_output_ticks=ticks,
                fast_output_distill_weight=distill,
                mtp_mode="mtp_1_2_4",
                mtp_horizons="1,2,4",
                max_steps=4000,
                batch_size=8,
            )
            regional(
                plan,
                f"og10_fastslow_d1024_{tag}",
                "Real fast/slow output on larger sparse CTM.",
                num_experts=16,
                expert_size=64,
                activation_passes=4,
                shared_experts=1,
                topk_experts=1,
                iterations=12 if "slow" in tag else 8,
                fast_output_mode=mode,
                fast_output_weight=fast_w,
                habit_output_weight=habit_w,
                slow_output_weight=slow_w,
                fast_output_ticks=ticks,
                fast_output_distill_weight=distill,
                mtp_mode="mtp_1_2_4",
                mtp_horizons="1,2,4",
                max_steps=4000,
                batch_size=6,
            )

        for dtag, num_experts, expert_size, widths, memories, memory_length in [
            ("d512", 16, 32, "8,16,32", "4,8", 8),
            ("d1024", 16, 64, "16,32,64", "4,8,16", 16),
        ]:
            for tag, fast_w, habit_w, distill in [
                ("compile_light", 0.10, 0.10, 0.05),
                ("compile_strong", 0.20, 0.25, 0.15),
            ]:
                regional(
                    plan,
                    f"og10_diff_fastslow_{dtag}_{tag}",
                    "Combined real differentiated cells plus fast-to-slow skill compilation loss.",
                    num_experts=num_experts,
                    expert_size=expert_size,
                    activation_passes=4,
                    shared_experts=1,
                    topk_experts=1,
                    iterations=12,
                    memory_length=memory_length,
                    diff_cell_widths=widths,
                    diff_cell_memory_lengths=memories,
                    fast_output_mode="anytime",
                    fast_output_weight=fast_w,
                    habit_output_weight=habit_w,
                    slow_output_weight=0.10,
                    fast_output_ticks="1,4,8,12",
                    fast_output_distill_weight=distill,
                    mtp_mode="mtp_1_2_4",
                    mtp_horizons="1,2,4",
                    max_steps=4500,
                    batch_size=8 if dtag == "d512" else 6,
                    **diff_common,
                )

    return base.validate_plan(plan)


base.configure_plan_defaults(metrics_prefix="overnight_sparse_ctm")
base.REGIONAL_STAGES = OVERNIGHT_STAGES
base.REGIONAL_PREFIXES = OVERNIGHT_PREFIXES
base.build_plan = build_plan


def audit_realism():
    labels = {
        "og00": "real: dense/top-k/regional anchors",
        "og01": "real: physical expert granularity/active width sweep",
        "og02": "real: active expert count/pass sweep",
        "og03": "real: implemented tick halt sweep",
        "og04": "real: implemented routing regularizers/routing modes",
        "og05": "real: implemented dispatch/capacity/drop modes",
        "og06": "real: implemented ELF/MTP losses",
        "og07": "real: long confirmation runs",
        "og08": "proxy: delta-cache/sticky-routing not implemented",
        "og09": "proxy: developmental/asynchronous/skill-compilation concepts not fully implemented",
        "og10": "real: learned differentiated cells plus resident reflex/anytime outputs",
    }
    for stage in OVERNIGHT_STAGES:
        if stage in {"all", "real"}:
            continue
        plan = build_plan(stage, "full")
        print(f"{stage}: {len(plan):3d} {labels[stage]}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "audit-realism":
        audit_realism()
    else:
        args = base.parse_args()
        args.func(args)
