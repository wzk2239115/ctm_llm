#!/usr/bin/env python3
"""Liquid-dynamics experiment plan (LTC-inspired).

Research question: can ideas from Liquid Time-Constant Networks improve CTM-LLM?

Two mechanisms are studied:

1. **Trajectory-length regularisation** (LTC paper Section 5).
   The arc-length of the tick-state path is penalised, encouraging compact
   inference trajectories.  Orthogonal to JEPA (which enforces predictability
   of the *next* tick) — this penalises the *total* path length.

2. **Liquid tick update** (LTC's state-dependent time constant).
   Instead of replacing the activation fully each tick, blend the new state
   with the previous one through a learned, *state-dependent* gate:
       h_{t+1} = (1 - gate(h)) * h_t + gate(h) * f(h_t)
   When the gate opens fully (→1) this degenerates to vanilla CTM; when it
   closes (→0) the state freezes.  The gate is initialised at a configurable
   bias so the model can learn how fast to update.

Stages:
  ld00 — anchors (no liquid / no traj)
  ld01 — trajectory-length regularisation sweep (mode × weight)
  ld02 — liquid tick update sweep (init bias)
  ld03 — liquid + trajectory combination
  ld04 — liquid + JEPA composition
  ld05 — liquid with deeper ticks
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment_plan_impl_validation as base


LIQUID_STAGES = ("ld00", "ld01", "ld02", "ld03", "ld04", "ld05", "all")
LIQUID_PREFIXES = tuple(f"{stage}_" for stage in LIQUID_STAGES if stage != "all")


def build_plan(stage, plan_size="full"):
    plan = []

    # ---- common configs ----------------------------------------------------
    d512 = dict(num_experts=16, expert_size=32, activation_passes=4,
                shared_experts=1, topk_experts=1, iterations=4, max_steps=1000,
                batch_size=6)
    d1024 = dict(num_experts=16, expert_size=64, activation_passes=4,
                 shared_experts=1, topk_experts=1, iterations=4, max_steps=1000,
                 batch_size=4)

    # =======================================================================
    # ld00 — anchors
    # =======================================================================
    if stage in ("ld00", "all"):
        base.add_regional_experiment(plan,
            "ld00_anchor_d512",
            "Dense regional d512 anchor (no liquid, no traj).",
            **d512, liquid_update_mode="none",
            trajectory_length_weight=0.0)
        base.add_regional_experiment(plan,
            "ld00_anchor_d1024",
            "Dense regional d1024 anchor (no liquid, no traj).",
            **d1024, liquid_update_mode="none",
            trajectory_length_weight=0.0)

    # =======================================================================
    # ld01 — trajectory-length regularisation sweep
    # =======================================================================
    if stage in ("ld01", "all"):
        for mode in ("l2", "l1", "cosine"):
            for w in (0.001, 0.01, 0.1):
                base.add_regional_experiment(plan,
                    f"ld01_traj_{mode}_w{str(w).replace('.', 'p')}_d512",
                    f"Trajectory-length {mode} regularisation, weight={w}.",
                    **d512, liquid_update_mode="none",
                    trajectory_length_weight=w,
                    trajectory_length_mode=mode)
        # best-mode confirmation at d1024
        base.add_regional_experiment(plan,
            "ld01_traj_l2_w0p01_d1024",
            "Trajectory-length l2 regularisation at d1024.",
            **d1024, liquid_update_mode="none",
            trajectory_length_weight=0.01,
            trajectory_length_mode="l2")

    # =======================================================================
    # ld02 — liquid tick update sweep
    # =======================================================================
    if stage in ("ld02", "all"):
        for init in (0.0, 1.0, 2.0, 4.0):
            base.add_regional_experiment(plan,
                f"ld02_liquid_init{str(init).replace('.', 'p')}_d512",
                f"Liquid tick update, gate init bias={init} "
                f"(sigmoid({init})={_sigmoid_str(init)}).",
                **d512, liquid_update_mode="gated",
                liquid_update_init=init,
                trajectory_length_weight=0.0)
        # d1024 best-init confirmation
        base.add_regional_experiment(plan,
            "ld02_liquid_init2p0_d1024",
            "Liquid tick update at d1024, init bias=2.0.",
            **d1024, liquid_update_mode="gated",
            liquid_update_init=2.0,
            trajectory_length_weight=0.0)

    # =======================================================================
    # ld03 — liquid + trajectory combination
    # =======================================================================
    if stage in ("ld03", "all"):
        for init in (1.0, 2.0):
            for tw in (0.01, 0.1):
                base.add_regional_experiment(plan,
                    f"ld03_liquid{str(init).replace('.', 'p')}"
                    f"_traj_w{str(tw).replace('.', 'p')}_d512",
                    f"Liquid(init={init}) + traj-l2(w={tw}) combination.",
                    **d512, liquid_update_mode="gated",
                    liquid_update_init=init,
                    trajectory_length_weight=tw,
                    trajectory_length_mode="l2")

    # =======================================================================
    # ld04 — liquid + JEPA composition
    # =======================================================================
    if stage in ("ld04", "all"):
        for init in (1.0, 2.0):
            base.add_regional_experiment(plan,
                f"ld04_liquid{str(init).replace('.', 'p')}_jepa_d512",
                f"Liquid(init={init}) + cross-tick JEPA(w=1.0).",
                **d512, liquid_update_mode="gated",
                liquid_update_init=init,
                cross_tick_jepa_weight=1.0,
                cross_tick_jepa_loss="cosine")

    # =======================================================================
    # ld05 — liquid with deeper ticks
    # =======================================================================
    if stage in ("ld05", "all"):
        for ticks in (8, 16):
            deep = dict(d512)
            deep = {**deep, "iterations": ticks, "batch_size": 4}
            base.add_regional_experiment(plan,
                f"ld05_liquid_init2p0_tick{ticks}_d512",
                f"Liquid(init=2.0) with {ticks} ticks — does liquid "
                f"scaling help deeper thought?",
                **deep, liquid_update_mode="gated",
                liquid_update_init=2.0,
                trajectory_length_weight=0.0)

    return base.validate_plan(plan)


def _sigmoid_str(x):
    val = 1.0 / (1.0 + pow(2.718281828, -x))
    return f"{val:.2f}"


base.configure_plan_defaults(
    metrics_prefix="liquid_dynamics",
    dispatch_block_sparse=False,
    build_plan=build_plan,
    stages=LIQUID_STAGES,
    prefixes=LIQUID_PREFIXES,
)
base.REGIONAL_STAGES = LIQUID_STAGES
base.REGIONAL_PREFIXES = LIQUID_PREFIXES


if __name__ == "__main__":
    base.parse_args().func(base.parse_args())
