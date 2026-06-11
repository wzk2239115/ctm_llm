#!/usr/bin/env python3
"""Cross-tick JEPA validation plan.

Tests whether tick_{i+1} → tick_i latent prediction improves
multi-tick representation quality vs baseline.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment_plan_impl_validation as base

JEPA_STAGES = ("jx00", "jx01", "all")
JEPA_PREFIXES = tuple(f"{stage}_" for stage in JEPA_STAGES if stage != "all")


def build_plan(stage, plan_size="full"):
    plan = []

    if stage in ("jx00", "all"):
        # Anchor: dense + top-k + regional baselines, no JEPA
        base.add_sparse_experiment(plan,
            "jx00_anchor_dense_tick4", "Dense tick4 anchor (no JEPA).",
            d_model=512, topk=512,
            iterations=4, synapse_depth=2, memory_hidden_dims=2, max_steps=1000,
            cross_tick_jepa_weight=0.0,
            batch_size=6,
        )
        # For the regional variant, we need to use the base.add_regional_experiment
        # since jepa_experiment uses add_sparse_experiment and doesn't know about moe_routing_mode
        base.add_regional_experiment(plan,
            "jx00_anchor_regional_d512_p4_tick4",
            "Regional p4 anchor (no JEPA).",
            num_experts=16, expert_size=32,
            activation_passes=4, shared_experts=1, topk_experts=1,
            iterations=4, max_steps=1000,
            cross_tick_jepa_weight=0.0,
            batch_size=6,
        )
        base.add_regional_experiment(plan,
            "jx00_anchor_regional_d1024_p4_tick4",
            "Regional d1024 anchor (no JEPA).",
            num_experts=16, expert_size=64,
            activation_passes=4, shared_experts=1, topk_experts=1,
            iterations=4, max_steps=1000,
            cross_tick_jepa_weight=0.0,
            batch_size=4,
        )

    if stage in ("jx01", "all"):
        # JEPA variants: add cross-tick prediction on top of same configs
        base.add_regional_experiment(plan,
            "jx01_jepa_cosine_d512_p4_tick4",
            "JEPA cosine: tick latent prediction with cosine loss.",
            num_experts=16, expert_size=32,
            activation_passes=4, shared_experts=1, topk_experts=1,
            iterations=4, max_steps=1000,
            cross_tick_jepa_weight=1.0,
            cross_tick_jepa_loss="cosine",
            batch_size=6,
        )
        base.add_regional_experiment(plan,
            "jx01_jepa_cosine_d1024_p4_tick4",
            "JEPA cosine d1024.",
            num_experts=16, expert_size=64,
            activation_passes=4, shared_experts=1, topk_experts=1,
            iterations=4, max_steps=1000,
            cross_tick_jepa_weight=1.0,
            cross_tick_jepa_loss="cosine",
            batch_size=4,
        )
        base.add_regional_experiment(plan,
            "jx01_jepa_cosine_w0p5_d512_p4_tick4",
            "JEPA cosine at lower weight.",
            num_experts=16, expert_size=32,
            activation_passes=4, shared_experts=1, topk_experts=1,
            iterations=4, max_steps=1000,
            cross_tick_jepa_weight=0.5,
            cross_tick_jepa_loss="cosine",
            batch_size=6,
        )
        base.add_regional_experiment(plan,
            "jx01_jepa_mse_d512_p4_tick4",
            "JEPA MSE loss variant.",
            num_experts=16, expert_size=32,
            activation_passes=4, shared_experts=1, topk_experts=1,
            iterations=4, max_steps=1000,
            cross_tick_jepa_weight=1.0,
            cross_tick_jepa_loss="mse",
            batch_size=6,
        )
        base.add_regional_experiment(plan,
            "jx01_jepa_no_stopgrad_d512_p4_tick4",
            "JEPA without stop-gradient on target.",
            num_experts=16, expert_size=32,
            activation_passes=4, shared_experts=1, topk_experts=1,
            iterations=4, max_steps=1000,
            cross_tick_jepa_weight=1.0,
            cross_tick_jepa_loss="cosine",
            cross_tick_jepa_target_stop_grad=False,
            batch_size=6,
        )
        base.add_regional_experiment(plan,
            "jx01_jepa_tick8_d512_p4",
            "JEPA with more ticks for richer cross-tick pairs.",
            num_experts=16, expert_size=32,
            activation_passes=4, shared_experts=1, topk_experts=1,
            iterations=8, max_steps=1000,
            cross_tick_jepa_weight=1.0,
            cross_tick_jepa_loss="cosine",
            batch_size=4,
        )

    return base.validate_plan(plan)


base.configure_plan_defaults(
    metrics_prefix="cross_tick_jepa",
    dispatch_block_sparse=False,
    build_plan=build_plan,
    stages=JEPA_STAGES,
    prefixes=JEPA_PREFIXES,
)
base.REGIONAL_STAGES = JEPA_STAGES
base.REGIONAL_PREFIXES = JEPA_PREFIXES


if __name__ == "__main__":
    base.parse_args().func(base.parse_args())
