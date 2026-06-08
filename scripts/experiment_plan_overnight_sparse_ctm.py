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

    return base.validate_plan(plan)


base.REGIONAL_STAGES = OVERNIGHT_STAGES
base.REGIONAL_PREFIXES = OVERNIGHT_PREFIXES
base.build_plan = build_plan


if __name__ == "__main__":
    args = base.parse_args()
    args.func(args)
