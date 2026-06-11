#!/usr/bin/env python3
"""Failure-informed CTM experiment plan: fix what failed.

Each stage addresses one root-cause category from the failure taxonomy
(results_analysis/failure_taxonomy_all_sweeps.md).  The experiments are
designed with explicit awareness of CTM's recurrent tick/trace/synapse
architecture.

Stages:
  fx00 - Per-tick diversity loss (addresses s02, s03, dr11 -- "all ticks
         compute the same thing" because min_conf gives identical supervision).
  fx01 - Training-time halt with early-tick loss (addresses dr11, og03 --
         halt only checked at inference, never learned during training).
  fx02 - Corruption + multi-tick synergy (extends dr03 -- the only quality
         win, with per-tick diversity pressure on top).
  fx03 - Sparse MoE with routing z-loss tuned for CTM scale (addresses moe04
         z-loss too strong, plus block-sparse parity with proper trace init).
  fx04 - Content-aware routing for CTM (addresses moe01 hash/expert-choice
         failures -- routing must depend on activated state, not position).
  fx05 - Simplified fast/slow paths (addresses og09/og10 OOM -- strip to
         minimal reflex architecture, no extra heads).
  fx06 - Deep CTM: per-tick parameter diversity (addresses s01/s04 scaling
         failures -- ticks reuse same params, need inter-tick diversity).
  fx07 - DINO self-distillation combined with CTM synchronous activation.
         Three layers:
           fx07a - Layer 1: tick-sync self-distill (converged tick teaches
                     early ticks, no EMA teacher, zero extra compute).
           fx07b - Layer 2: sync-center DINO (EMA teacher with tick-mean
                     centering instead of hidden-state centering).
           fx07c - Layer 3: per-tick sync decay as adaptive temperature
                     (linear schedule from fast-forget to slow-accumulate).
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import experiment_plan_impl_validation as base

FIX_STAGES = (
    "fx00",
    "fx01",
    "fx02",
    "fx03",
    "fx04",
    "fx05",
    "fx06",
    "fx07",
    "real",
    "all",
)
FIX_PREFIXES = tuple(f"{s}_" for s in FIX_STAGES if s != "all")

DEFAULT_MAX_STEPS = 1000


def bs_for(d_model, *, iterations=4, halt_train=False, diversity=False):
    batch = 10 if d_model <= 512 else (8 if d_model <= 1024 else 6)
    if iterations >= 8:
        batch = max(2, batch - 2)
    if halt_train:
        batch = max(2, batch - 1)
    if diversity:
        batch = max(2, batch - 1)
    floor = 4 if d_model <= 512 else (3 if d_model <= 1024 else 2)
    return max(floor, batch)


def regional(plan, name, question, *,
             num_experts=16, expert_size=32, max_steps=DEFAULT_MAX_STEPS,
             iterations=4, activation_passes=1, topk_experts=1,
             shared_experts=1, balance=1e-2, diversity=None,
             batch_size=None, **kwargs):
    d_model = num_experts * expert_size
    if diversity is None:
        diversity = 1e-3 if activation_passes >= 3 else 0.0
    if batch_size is None:
        batch_size = bs_for(d_model, iterations=iterations)
    base.add_regional_experiment(
        plan, name, question,
        num_experts=num_experts, expert_size=expert_size,
        activation_passes=activation_passes, shared_experts=shared_experts,
        topk_experts=topk_experts, balance=balance, diversity=diversity,
        iterations=iterations, max_steps=max_steps, batch_size=batch_size,
        **kwargs,
    )


def build_plan(stage, plan_size="full"):
    plan = []

    # ========================================================================
    # fx00: Per-tick diversity loss
    # ========================================================================
    # Root cause: min_conf gives identical next-token supervision to every tick.
    # All ticks converge to the same computation within 2-3 iterations (s02).
    # Fix: penalize ticks for producing identical distributions via KL divergence.
    #
    # The diversity loss forces tick T to predict differently from tick T+gap.
    # Temperature controls sharpness: high temp = tolerate similar distributions,
    # low temp = demand more distinct predictions per tick.
    #
    # Key CTM insight: unlike MTP (which predicts DIFFERENT tokens from the same
    # hidden state), diversity loss forces the recurrent state to EVOLVE across
    # ticks.  This works WITH the synapse-activate-trace pipeline instead of
    # ignoring it.

    if stage in ("fx00", "all"):
        # fx00a: anchor -- no diversity (baseline for comparison)
        plan.append(base.experiment(
            "fx00a_anchor_d512",
            "Anchor: no diversity loss, iterations=4",
            base.merge_args(iterations=4, tick_diversity_mode="none",
                            batch_size=10, max_steps=DEFAULT_MAX_STEPS),
        ))
        # fx00b-f: sweep diversity temperature and weight
        for temp in ("0.05", "0.1", "0.2"):
            for weight in ("0.01", "0.05", "0.1"):
                plan.append(base.experiment(
                    f"fx00b_div_t{temp.replace('.','p')}_w{weight.replace('.','p')}_d512",
                    f"Diversity KL loss: temp={temp}, weight={weight}",
                    base.merge_args(
                        iterations=4,
                        tick_diversity_mode="kl",
                        tick_diversity_temperature=float(temp),
                        tick_diversity_weight=float(weight),
                        batch_size=bs_for(512, diversity=True),
                        max_steps=DEFAULT_MAX_STEPS,
                    ),
                ))
        if base.include_plan_size(plan_size, "full"):
            # fx00g: gap=2 (compare tick 0 vs tick 2, tick 2 vs tick 4)
            plan.append(base.experiment(
                "fx00g_div_gap2_d512",
                "Diversity KL with horizon_gap=2 (compare every 2nd tick)",
                base.merge_args(
                    iterations=8,
                    tick_diversity_mode="kl",
                    tick_diversity_temperature=0.1,
                    tick_diversity_weight=0.05,
                    tick_diversity_horizon_gap=2,
                    batch_size=bs_for(512, iterations=8, diversity=True),
                    max_steps=DEFAULT_MAX_STEPS,
                ),
            ))
        if base.include_plan_size(plan_size, "wide"):
            # fx00h: more ticks to give diversity room to work
            for iters in (8, 16):
                plan.append(base.experiment(
                    f"fx00h_div_iters{iters}_d512",
                    f"Diversity KL with {iters} ticks (more room for divergence)",
                    base.merge_args(
                        iterations=iters,
                        tick_diversity_mode="kl",
                        tick_diversity_temperature=0.1,
                        tick_diversity_weight=0.05,
                        batch_size=bs_for(512, iterations=iters, diversity=True),
                        max_steps=DEFAULT_MAX_STEPS,
                    ),
                ))

    # ========================================================================
    # fx01: Training-time halt with early-tick loss
    # ========================================================================
    # Root cause: halt only checked at inference (line 664, enable_tick_halt=False
    # during training). Model never PRACTICES early exit. dr11, og03 showed
    # effective_tick=1.0 because tick-1 is weakest when trained with 4 ticks.
    #
    # Fix: actually break the tick loop during training when confidence >=
    # threshold. Weight the loss toward the halted tick. Add early-tick
    # bonus loss to ensure tick-1/2/3 are well-trained.
    #
    # CTM insight: the recurrent state at early ticks is different from late
    # ticks (shorter trace history). By actually halting during training,
    # the synapse learns to produce useful output from a fresh trace, not
    # from a fully-converged one.

    if stage in ("fx01", "all"):
        # fx01a: anchor -- no halt, 4 ticks
        plan.append(base.experiment(
            "fx01a_anchor_d512",
            "Anchor: no halt, iterations=4",
            base.merge_args(iterations=4, tick_halt_mode="none",
                            batch_size=10, max_steps=DEFAULT_MAX_STEPS),
        ))
        # fx01b: old-style halt (inference-only, for comparison)
        plan.append(base.experiment(
            "fx01b_halt_infonly_t0p65_d512",
            "Old halt: inference-only, threshold=0.65",
            base.merge_args(iterations=8, tick_halt_mode="threshold",
                            tick_halt_threshold=0.65,
                            batch_size=8, max_steps=DEFAULT_MAX_STEPS),
        ))
        # fx01c-f: training-time halt with different thresholds
        for thresh in ("0.50", "0.65", "0.80"):
            plan.append(base.experiment(
                f"fx01c_trainhalt_t{thresh.replace('.','p')}_d512",
                f"Training-time halt: threshold={thresh}, 8 ticks",
                base.merge_args(
                    iterations=8,
                    tick_halt_train_mode="threshold",
                    tick_halt_train_threshold=float(thresh),
                    batch_size=bs_for(512, iterations=8, halt_train=True),
                    max_steps=DEFAULT_MAX_STEPS,
                ),
            ))
        if base.include_plan_size(plan_size, "full"):
            # fx01g: training halt + confidence penalty (encourage earlier halt)
            plan.append(base.experiment(
                "fx01g_trainhalt_t0p65_confw0p5_d512",
                "Training halt + confidence-weighted compute penalty",
                base.merge_args(
                    iterations=8,
                    tick_halt_train_mode="threshold",
                    tick_halt_train_threshold=0.65,
                    tick_halt_train_confidence_weight=0.5,
                    batch_size=bs_for(512, iterations=8, halt_train=True),
                    max_steps=DEFAULT_MAX_STEPS,
                ),
            ))
            # fx01h: training halt + early-tick bonus (ensure tick 1-3 are strong)
            plan.append(base.experiment(
                "fx01h_trainhalt_t0p65_early0p2_d512",
                "Training halt + early-tick loss bonus (tick 1-3)",
                base.merge_args(
                    iterations=8,
                    tick_halt_train_mode="threshold",
                    tick_halt_train_threshold=0.65,
                    tick_halt_train_early_loss_weight=0.2,
                    batch_size=bs_for(512, iterations=8, halt_train=True),
                    max_steps=DEFAULT_MAX_STEPS,
                ),
            ))
            # fx01i: both penalties
            plan.append(base.experiment(
                "fx01i_trainhalt_t0p65_confw0p3_early0p1_d512",
                "Training halt + compute penalty + early-tick bonus",
                base.merge_args(
                    iterations=8,
                    tick_halt_train_mode="threshold",
                    tick_halt_train_threshold=0.65,
                    tick_halt_train_confidence_weight=0.3,
                    tick_halt_train_early_loss_weight=0.1,
                    batch_size=bs_for(512, iterations=8, halt_train=True),
                    max_steps=DEFAULT_MAX_STEPS,
                ),
            ))

    # ========================================================================
    # fx02: Corruption + per-tick diversity (extends dr03)
    # ========================================================================
    # Root cause: dr03 (corruption 15%) was the only quality win. It works
    # because corruption provides novel per-tick supervision: later ticks must
    # correct corrupted tokens.
    #
    # Fix: combine corruption with diversity loss to amplify the per-tick
    # differentiation signal. The corruption forces ticks to differ in
    # WHAT they predict (correct vs corrupt), while diversity loss forces
    # ticks to differ in HOW they predict (distribution shape).
    #
    # CTM insight: corruption modifies the input at token level. In CTM,
    # a corrupted token changes the activated state at that position, which
    # propagates through the trace to neighboring positions across ticks.
    # This means corruption naturally creates per-position variation in the
    # recurrent state, which is exactly what the diversity loss needs to
    # operate on meaningfully.

    if stage in ("fx02", "all"):
        # fx02a: dr03 anchor (corruption 15%, no diversity)
        plan.append(base.experiment(
            "fx02a_corrupt_anchor_d512",
            "dr03 anchor: corruption 15%, no diversity",
            base.merge_args(
                draft_mode="revise", draft_corrupt_prob=0.15,
                draft_loss_weight=0.2, draft_revise_weight=0.2,
                iterations=4, tick_diversity_mode="none",
                batch_size=10, max_steps=DEFAULT_MAX_STEPS,
            ),
        ))
        # fx02b-d: corruption + diversity at different weights
        for div_w in ("0.01", "0.05", "0.1"):
            plan.append(base.experiment(
                f"fx02b_corrupt_div{div_w.replace('.','p')}_d512",
                f"Corruption 15% + diversity KL weight={div_w}",
                base.merge_args(
                    draft_mode="revise", draft_corrupt_prob=0.15,
                    draft_loss_weight=0.2, draft_revise_weight=0.2,
                    iterations=4,
                    tick_diversity_mode="kl",
                    tick_diversity_temperature=0.1,
                    tick_diversity_weight=float(div_w),
                    batch_size=bs_for(512, diversity=True),
                    max_steps=DEFAULT_MAX_STEPS,
                ),
            ))
        if base.include_plan_size(plan_size, "full"):
            # fx02e: corruption + diversity + 8 ticks (more room for correction)
            plan.append(base.experiment(
                "fx02e_corrupt_div0p05_8tick_d512",
                "Corruption + diversity with 8 ticks (more correction rounds)",
                base.merge_args(
                    draft_mode="revise", draft_corrupt_prob=0.15,
                    draft_loss_weight=0.2, draft_revise_weight=0.2,
                    iterations=8,
                    tick_diversity_mode="kl",
                    tick_diversity_temperature=0.1,
                    tick_diversity_weight=0.05,
                    batch_size=bs_for(512, iterations=8, diversity=True),
                    max_steps=DEFAULT_MAX_STEPS,
                ),
            ))
            # fx02f: corruption + training halt (correct quickly, then halt)
            plan.append(base.experiment(
                "fx02f_corrupt_trainhalt0p65_d512",
                "Corruption + training halt (correct and stop early)",
                base.merge_args(
                    draft_mode="revise", draft_corrupt_prob=0.15,
                    draft_loss_weight=0.2, draft_revise_weight=0.2,
                    iterations=8,
                    tick_halt_train_mode="threshold",
                    tick_halt_train_threshold=0.65,
                    tick_halt_train_early_loss_weight=0.1,
                    batch_size=bs_for(512, iterations=8, halt_train=True),
                    max_steps=DEFAULT_MAX_STEPS,
                ),
            ))

    # ========================================================================
    # fx03: Sparse MoE with proper z-loss scaling and dispatch parity
    # ========================================================================
    # Root cause: moe04 used z-loss=1e-3 (too strong for 16 experts at d=512).
    # iv73 showed block_sparse numerically diverges from dense_mask.
    #
    # Fix: (a) sweep z-loss at appropriate scale for this model.
    # (b) Use dense_mask dispatch (known stable) with regional routing.
    # (c) Vary expert count to find the sweet spot.
    #
    # CTM insight: z-loss regularizes routing logits to prevent collapse.
    # In CTM, routing depends on activated state magnitude, which evolves
    # per tick. Over-regularizing makes all experts equally likely, which
    # defeats the purpose of content-dependent routing. Under-regularizing
    # lets a few experts dominate, creating hotspots in the recurrent state.

    if stage in ("fx03", "all"):
        # fx03a: dense anchor (no sparsity)
        plan.append(base.experiment(
            "fx03a_dense_d512",
            "Dense anchor: no MoE, no sparsity",
            base.merge_args(
                d_model=512, n_synch_out=512, n_synch_action=512,
                cell_sparsity_mode="none", moe_routing_mode="none",
                batch_size=10, max_steps=DEFAULT_MAX_STEPS,
            ),
        ))
        # fx03b: regional MoE baseline (dense_mask, no z-loss)
        regional(
            plan, "fx03b_regional_p4_s1_t1_d512",
            "Regional MoE baseline: dense_mask, p4/shared1/top1",
            num_experts=16, expert_size=32, activation_passes=1,
            topk_experts=1, shared_experts=1,
        )
        # fx03c-f: z-loss sweep at CTM-appropriate scale
        for zl in ("1e-5", "5e-5", "1e-4", "5e-4"):
            regional(
                plan, f"fx03c_zloss{zl}_p4_s1_t1_d512",
                f"Regional MoE + z-loss={zl}",
                num_experts=16, expert_size=32, activation_passes=1,
                topk_experts=1, shared_experts=1,
                moe_router_z_loss_weight=float(zl),
            )
        if base.include_plan_size(plan_size, "full"):
            # fx03g: more experts, smaller expert_size
            regional(
                plan, "fx03g_p8_s1_t1_es16_d512",
                "Regional MoE: 32 experts x 16 dim (more routing granularity)",
                num_experts=32, expert_size=16, activation_passes=1,
                topk_experts=1, shared_experts=1,
            )
            # fx03h: top-2 routing with z-loss
            regional(
                plan, "fx03h_t2_zloss1e4_d512",
                "Regional MoE: top-2 routing + z-loss=1e-4",
                num_experts=16, expert_size=32, activation_passes=1,
                topk_experts=2, shared_experts=1,
                moe_router_z_loss_weight=1e-4,
            )
            # fx03i: multi-pass with z-loss
            regional(
                plan, "fx03i_p2_s1_t1_zloss1e4_d512",
                "Regional MoE: 2 passes + z-loss=1e-4",
                num_experts=16, expert_size=32, activation_passes=2,
                topk_experts=1, shared_experts=1,
                moe_router_z_loss_weight=1e-4,
            )

    # ========================================================================
    # fx04: Content-aware routing (CTM-native)
    # ========================================================================
    # Root cause: hash routing (position-based) and expert-choice (batch-mean)
    # both ignore CTM's activated state. CTM routing must be content-dependent.
    #
    # Fix: the existing topk routing already uses activated state magnitude,
    # which is correct. The issue is that magnitude alone is a weak signal.
    # We add router entropy weight to prevent collapse, and vary the routing
    # granularity to find the right resolution.
    #
    # CTM insight: the activated state in CTM carries recurrent information
    # accumulated across ticks. At tick 0, activated is close to the initial
    # state. At tick 3, activated reflects 3 rounds of synapse-activate-trace
    # processing. Content-aware routing should capture this temporal evolution.
    # Simple magnitude routing (current topk) does this implicitly since later
    # ticks have more refined activations. But we can improve it with entropy
    # regularization that prevents the router from always selecting the same
    # experts.

    if stage in ("fx04", "all"):
        # fx04a: topk routing anchor (current best)
        regional(
            plan, "fx04a_topk_anchor_d512",
            "Topk routing anchor: magnitude-based, no entropy reg",
            num_experts=16, expert_size=32, activation_passes=1,
            topk_experts=1, shared_experts=1,
        )
        # fx04b-d: entropy regularization sweep
        for ent_w in ("0.01", "0.05", "0.1"):
            regional(
                plan, f"fx04b_entropy{ent_w.replace('.','p')}_d512",
                f"Topk routing + entropy weight={ent_w} (prevent collapse)",
                num_experts=16, expert_size=32, activation_passes=1,
                topk_experts=1, shared_experts=1,
                moe_router_entropy_weight=float(ent_w),
            )
        if base.include_plan_size(plan_size, "full"):
            # fx04e: entropy + z-loss combo
            regional(
                plan, "fx04e_ent0p05_zloss1e4_d512",
                "Topk routing + entropy(0.05) + z-loss(1e-4)",
                num_experts=16, expert_size=32, activation_passes=1,
                topk_experts=1, shared_experts=1,
                moe_router_entropy_weight=0.05,
                moe_router_z_loss_weight=1e-4,
            )
            # fx04f: larger routing space (32 experts)
            regional(
                plan, "fx04f_e32_ent0p05_d512",
                "32-expert topk routing + entropy(0.05)",
                num_experts=32, expert_size=16, activation_passes=1,
                topk_experts=1, shared_experts=1,
                moe_router_entropy_weight=0.05,
            )

    # ========================================================================
    # fx05: Simplified fast/slow (reflex-only, no extra heads)
    # ========================================================================
    # Root cause: og09/og10 fast/slow paths added extra output heads, extra
    # loss terms, and OOM'd. The reflex path (og09_fastpath_reflex) worked
    # (loss 4.02, 8684 tok/s) because it REMOVED complexity.
    #
    # Fix: minimal reflex architecture: tick-1 output with short memory.
    # No extra heads. Just reduce iterations and memory_length.
    #
    # CTM insight: the reflex path works because CTM's tick-1 with short
    # memory is already a useful "fast" computation. The synapse runs once,
    # trace has only 1-2 entries, and NLM processes a shallow memory. This
    # is the minimum viable CTM computation. Adding complexity (extra heads,
    # MTP, halt) on top of this simple base breaks it.

    if stage in ("fx05", "all"):
        # fx05a: dense anchor (4 ticks, mem=10)
        plan.append(base.experiment(
            "fx05a_dense_anchor_d512",
            "Dense anchor: 4 ticks, memory=10",
            base.merge_args(iterations=4, memory_length=10,
                            batch_size=10, max_steps=DEFAULT_MAX_STEPS),
        ))
        # fx05b: reflex tick-1, short memory
        plan.append(base.experiment(
            "fx05b_reflex_t1_m2_d512",
            "Reflex: 1 tick, memory=2 (minimal CTM)",
            base.merge_args(iterations=1, memory_length=2,
                            batch_size=10, max_steps=DEFAULT_MAX_STEPS),
        ))
        # fx05c: reflex tick-1, memory=4
        plan.append(base.experiment(
            "fx05c_reflex_t1_m4_d512",
            "Reflex: 1 tick, memory=4",
            base.merge_args(iterations=1, memory_length=4,
                            batch_size=10, max_steps=DEFAULT_MAX_STEPS),
        ))
        # fx05d: reflex tick-2, short memory
        plan.append(base.experiment(
            "fx05d_reflex_t2_m4_d512",
            "Reflex: 2 ticks, memory=4 (slightly more compute)",
            base.merge_args(iterations=2, memory_length=4,
                            batch_size=10, max_steps=DEFAULT_MAX_STEPS),
        ))
        if base.include_plan_size(plan_size, "full"):
            # fx05e: regional reflex (sparse + short memory)
            regional(
                plan, "fx05e_regional_t1_m4_d512",
                "Regional MoE reflex: 1 tick, memory=4, p4/s1/t1",
                num_experts=16, expert_size=32, activation_passes=1,
                topk_experts=1, shared_experts=1,
                iterations=1, memory_length=4,
            )
            # fx05f: regional + 2 ticks
            regional(
                plan, "fx05f_regional_t2_m4_d512",
                "Regional MoE: 2 ticks, memory=4, p4/s1/t1",
                num_experts=16, expert_size=32, activation_passes=1,
                topk_experts=1, shared_experts=1,
                iterations=2, memory_length=4,
            )

    # ========================================================================
    # fx06: Deep CTM -- inter-tick parameter diversity
    # ========================================================================
    # Root cause: s01/s04 showed that scaling CTM by adding more ticks or
    # layers fails because ticks reuse the SAME layer parameters. Each tick
    # at the same layer is parameter-identical.
    #
    # Fix: this stage tests whether adding more ticks helps when each tick
    # gets its own parameters (deep CTM). Since the current architecture
    # doesn't support per-tick parameters natively, we approximate by
    # using different configurations per tick that expose different
    # aspects of the architecture.
    #
    # CTM insight: in a standard transformer, depth means different layers
    # learn different attention patterns. CTM achieves a similar effect if
    # the recurrent state evolves to be in different "phases" at different
    # ticks. The diversity loss (fx00) provides the evolutionary pressure.
    # Here we test whether that pressure alone, without architectural changes,
    # is sufficient to make deep ticks useful.
    #
    # Concretely: we train with many ticks + diversity loss, and check if
    # later ticks actually produce different (better) predictions.

    if stage in ("fx06", "all"):
        # fx06a: many ticks + diversity (no halt)
        plan.append(base.experiment(
            "fx06a_div_t8_w0p05_d512",
            "8 ticks + diversity KL (diversity pressure enables deep ticks)",
            base.merge_args(
                iterations=8,
                tick_diversity_mode="kl",
                tick_diversity_temperature=0.1,
                tick_diversity_weight=0.05,
                batch_size=bs_for(512, iterations=8, diversity=True),
                max_steps=DEFAULT_MAX_STEPS,
            ),
        ))
        # fx06b: many ticks + diversity + training halt
        plan.append(base.experiment(
            "fx06b_div_t8_halt0p80_d512",
            "8 ticks + diversity + training halt at 0.80",
            base.merge_args(
                iterations=8,
                tick_diversity_mode="kl",
                tick_diversity_temperature=0.1,
                tick_diversity_weight=0.05,
                tick_halt_train_mode="threshold",
                tick_halt_train_threshold=0.80,
                tick_halt_train_early_loss_weight=0.1,
                batch_size=bs_for(512, iterations=8, halt_train=True, diversity=True),
                max_steps=DEFAULT_MAX_STEPS,
            ),
        ))
        if base.include_plan_size(plan_size, "full"):
            # fx06c: 16 ticks + diversity (stress test)
            plan.append(base.experiment(
                "fx06c_div_t16_w0p03_d512",
                "16 ticks + diversity KL weight=0.03 (stress test)",
                base.merge_args(
                    iterations=16,
                    tick_diversity_mode="kl",
                    tick_diversity_temperature=0.1,
                    tick_diversity_weight=0.03,
                    batch_size=bs_for(512, iterations=16, diversity=True),
                    max_steps=DEFAULT_MAX_STEPS,
                ),
            ))
            # fx06d: corruption + diversity + deep ticks
            plan.append(base.experiment(
                "fx06d_corrupt_div_t8_d512",
                "Corruption + diversity + 8 ticks (max synergy)",
                base.merge_args(
                    draft_mode="revise", draft_corrupt_prob=0.15,
                    draft_loss_weight=0.2, draft_revise_weight=0.2,
                    iterations=8,
                    tick_diversity_mode="kl",
                    tick_diversity_temperature=0.1,
                    tick_diversity_weight=0.05,
                    batch_size=bs_for(512, iterations=8, diversity=True),
                    max_steps=DEFAULT_MAX_STEPS,
                ),
            ))

    # ========================================================================
    # fx07: DINO self-distillation + CTM synchronous activation
    # ========================================================================
    # Three layers of combining DINO with CTM's sync mechanism:
    #
    # Layer 1 (fx07a): tick-sync self-distill
    #   The converged tick (last tick) serves as teacher for all earlier ticks.
    #   KL loss from early ticks' distributions toward the converged distribution.
    #   No EMA teacher needed. Zero extra forward passes. The teacher signal
    #   comes from CTM's own recurrent convergence.
    #
    #   CTM insight: sync_o is the projection head of _compute_synch, which
    #   computes Welford running mean of pairwise neuron products. Each tick's
    #   sync_o reflects the accumulated synaptic state. The last tick's sync_o
    #   is the most "converged" representation. Distilling earlier ticks toward
    #   it forces the recurrent pipeline to produce useful output early.
    #
    # Layer 2 (fx07b): sync-center DINO
    #   EMA teacher model with centering on tick-mean (mean of all tick outputs)
    #   instead of single hidden-state centering.
    #
    #   CTM insight: standard DINO centers on proj(h_final) -- one point in
    #   hidden space. CTM's per-tick evolution means tick_outs span a trajectory.
    #   Centering on the trajectory mean captures CTM's temporal structure.
    #   The center becomes a "sync-level" aggregate instead of a point estimate.
    #
    # Layer 3 (fx07c): per-tick sync decay as adaptive temperature
    #   The _compute_synch decay rate (r = exp(-decay)) controls how much
    #   history the sync retains. Linearly schedule decay across ticks:
    #   early ticks get one decay, late ticks get another. This acts as an
    #   adaptive temperature -- fast-forgetting early ticks respond sharply
    #   to current activated state, slow-forgetting late ticks smooth over
    #   accumulated history.
    #
    #   CTM insight: sync decay is the only per-tick hyperparameter in CTM's
    #   forward pass. Making it tick-dependent gives each tick a different
    #   "temporal resolution": early ticks = high-resolution (focus on now),
    #   late ticks = low-resolution (integrate over history). This is CTM's
    #   native analogue of temperature annealing in knowledge distillation.

    if stage in ("fx07", "all"):
        # fx07a: Layer 1 -- tick-sync self-distill
        plan.append(base.experiment(
            "fx07a_sync_distill_w0p05_t0p1_d512",
            "Layer 1: tick-sync self-distill, weight=0.05, temp=0.1, 4 ticks",
            base.merge_args(
                iterations=4,
                tick_sync_distill_weight=0.05,
                tick_sync_distill_temperature=0.1,
                batch_size=10,
                max_steps=DEFAULT_MAX_STEPS,
            ),
        ))
        for w in ("0.01", "0.1", "0.2"):
            plan.append(base.experiment(
                f"fx07a_sync_distill_w{w.replace('.','p')}_d512",
                f"Layer 1: tick-sync self-distill, weight={w}, 4 ticks",
                base.merge_args(
                    iterations=4,
                    tick_sync_distill_weight=float(w),
                    tick_sync_distill_temperature=0.1,
                    batch_size=10,
                    max_steps=DEFAULT_MAX_STEPS,
                ),
            ))
        if base.include_plan_size(plan_size, "full"):
            plan.append(base.experiment(
                "fx07a_sync_distill_w0p05_t8_d512",
                "Layer 1: tick-sync self-distill, 8 ticks (more teacher signal)",
                base.merge_args(
                    iterations=8,
                    tick_sync_distill_weight=0.05,
                    tick_sync_distill_temperature=0.1,
                    batch_size=bs_for(512, iterations=8),
                    max_steps=DEFAULT_MAX_STEPS,
                ),
            ))

        # fx07b: Layer 2 -- sync-center DINO
        plan.append(base.experiment(
            "fx07b_sync_dino_w0p05_d512",
            "Layer 2: sync-center DINO (tick-mean centering), weight=0.05",
            base.merge_args(
                iterations=4,
                dino_self_supervised_weight=0.01,
                dino_out_dim=512,
                dino_hidden_dim=256,
                dino_bottleneck_dim=64,
                dino_student_temperature=0.10,
                dino_teacher_temperature=0.04,
                tick_sync_dino_mode="tick_center",
                tick_sync_dino_weight=0.05,
                batch_size=8,
                max_steps=DEFAULT_MAX_STEPS,
            ),
        ))
        for w in ("0.01", "0.1"):
            plan.append(base.experiment(
                f"fx07b_sync_dino_w{w.replace('.','p')}_d512",
                f"Layer 2: sync-center DINO, weight={w}",
                base.merge_args(
                    iterations=4,
                    dino_self_supervised_weight=0.01,
                    dino_out_dim=512,
                    dino_hidden_dim=256,
                    dino_bottleneck_dim=64,
                    dino_student_temperature=0.10,
                    dino_teacher_temperature=0.04,
                    tick_sync_dino_mode="tick_center",
                    tick_sync_dino_weight=float(w),
                    batch_size=8,
                    max_steps=DEFAULT_MAX_STEPS,
                ),
            ))

        # fx07c: Layer 3 -- per-tick sync decay
        for start, end in [("-1.0", "1.0"), ("0.0", "2.0"), ("1.0", "0.0")]:
            plan.append(base.experiment(
                f"fx07c_decay_s{start.replace('.','p')}_e{end.replace('.','p')}_d512",
                f"Layer 3: per-tick decay start={start}, end={end} (4 ticks)",
                base.merge_args(
                    iterations=4,
                    tick_sync_decay_schedule="linear",
                    tick_sync_decay_start=float(start),
                    tick_sync_decay_end=float(end),
                    batch_size=10,
                    max_steps=DEFAULT_MAX_STEPS,
                ),
            ))
        if base.include_plan_size(plan_size, "full"):
            for start, end in [("-2.0", "2.0"), ("0.0", "3.0")]:
                plan.append(base.experiment(
                    f"fx07c_decay_s{start.replace('.','p')}_e{end.replace('.','p')}_t8_d512",
                    f"Layer 3: per-tick decay start={start}, end={end}, 8 ticks",
                    base.merge_args(
                        iterations=8,
                        tick_sync_decay_schedule="linear",
                        tick_sync_decay_start=float(start),
                        tick_sync_decay_end=float(end),
                        batch_size=bs_for(512, iterations=8),
                        max_steps=DEFAULT_MAX_STEPS,
                    ),
                ))

        if base.include_plan_size(plan_size, "full"):
            # fx07d: Layer 1+3 combo (self-distill + per-tick decay)
            plan.append(base.experiment(
                "fx07d_syncdistill_decay_s0_e2_d512",
                "Layer 1+3: tick-sync self-distill + per-tick decay (0->2)",
                base.merge_args(
                    iterations=4,
                    tick_sync_distill_weight=0.05,
                    tick_sync_distill_temperature=0.1,
                    tick_sync_decay_schedule="linear",
                    tick_sync_decay_start=0.0,
                    tick_sync_decay_end=2.0,
                    batch_size=10,
                    max_steps=DEFAULT_MAX_STEPS,
                ),
            ))
            # fx07e: Layer 1+2 combo (self-distill + sync-center DINO)
            plan.append(base.experiment(
                "fx07e_syncdistill_dino_d512",
                "Layer 1+2: tick-sync self-distill + sync-center DINO",
                base.merge_args(
                    iterations=4,
                    tick_sync_distill_weight=0.05,
                    tick_sync_distill_temperature=0.1,
                    dino_self_supervised_weight=0.01,
                    dino_out_dim=512,
                    dino_hidden_dim=256,
                    dino_bottleneck_dim=64,
                    tick_sync_dino_mode="tick_center",
                    tick_sync_dino_weight=0.05,
                    batch_size=8,
                    max_steps=DEFAULT_MAX_STEPS,
                ),
            ))

    # ========================================================================
    # real: runnable subset (best configs from each stage)
    # ========================================================================
    if stage == "real":
        # Anchors
        plan.append(base.experiment(
            "real_anchor_d512",
            "Dense anchor: 4 ticks, d=512",
            base.merge_args(iterations=4, batch_size=10,
                            max_steps=DEFAULT_MAX_STEPS),
        ))
        # fx00 best: diversity
        plan.append(base.experiment(
            "real_fx00_div_t0p1_w0p05_d512",
            "fx00 best: diversity KL temp=0.1, weight=0.05",
            base.merge_args(
                iterations=4,
                tick_diversity_mode="kl",
                tick_diversity_temperature=0.1,
                tick_diversity_weight=0.05,
                batch_size=bs_for(512, diversity=True),
                max_steps=DEFAULT_MAX_STEPS,
            ),
        ))
        # fx01 best: training halt
        plan.append(base.experiment(
            "real_fx01_trainhalt_t0p65_early0p1_d512",
            "fx01 best: training halt + early-tick bonus",
            base.merge_args(
                iterations=8,
                tick_halt_train_mode="threshold",
                tick_halt_train_threshold=0.65,
                tick_halt_train_early_loss_weight=0.1,
                batch_size=bs_for(512, iterations=8, halt_train=True),
                max_steps=DEFAULT_MAX_STEPS,
            ),
        ))
        # fx02 best: corruption + diversity
        plan.append(base.experiment(
            "real_fx02_corrupt_div0p05_d512",
            "fx02 best: corruption + diversity",
            base.merge_args(
                draft_mode="revise", draft_corrupt_prob=0.15,
                draft_loss_weight=0.2, draft_revise_weight=0.2,
                iterations=4,
                tick_diversity_mode="kl",
                tick_diversity_temperature=0.1,
                tick_diversity_weight=0.05,
                batch_size=bs_for(512, diversity=True),
                max_steps=DEFAULT_MAX_STEPS,
            ),
        ))
        # fx05 best: reflex
        plan.append(base.experiment(
            "real_fx05_reflex_t1_m4_d512",
            "fx05 best: reflex 1 tick, memory=4",
            base.merge_args(iterations=1, memory_length=4,
                            batch_size=10, max_steps=DEFAULT_MAX_STEPS),
        ))
        # fx07 best: sync self-distill
        plan.append(base.experiment(
            "real_fx07a_sync_distill_w0p05_d512",
            "fx07 best: tick-sync self-distill",
            base.merge_args(
                iterations=4,
                tick_sync_distill_weight=0.05,
                tick_sync_distill_temperature=0.1,
                batch_size=10,
                max_steps=DEFAULT_MAX_STEPS,
            ),
        ))

    return base.validate_plan(plan)


base.configure_plan_defaults(
    metrics_prefix="fix_failure",
    build_plan=build_plan,
    stages=FIX_STAGES,
    prefixes=FIX_PREFIXES,
)
base.REGIONAL_STAGES = FIX_STAGES
base.REGIONAL_PREFIXES = FIX_PREFIXES

if __name__ == "__main__":
    args = base.parse_args()
    args.func(args)
