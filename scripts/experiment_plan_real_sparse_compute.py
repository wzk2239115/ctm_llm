#!/usr/bin/env python3
"""Experiment plan: real sparse-compute vs numerical sparsity for CTM-LLM.

This plan is designed to answer one question with wall-clock evidence:

    Does a sparsity knob actually reduce GPU cost (tok/s, peak memory, step time),
    or does it only reduce the active-cell fraction on paper?

Stages map to the research taxonomy discussed in the sparse-compute design note:

  rc00  smoke anchors (one per category)
  rc01  numerical mask vs structured block routing at matched active fraction
  rc02  dispatch mode comparison (dense_mask / block_sparse / dropless / capacity_drop)
  rc03  tick sparsity: fixed iterations vs adaptive early-exit
  rc04  block routing granularity (16/32/64 cell blocks)
  rc05  single-pass MoE (Switch-style dense dispatch) vs regional block sparse
  rc06  composite tick + block routing
  rc07  longer confirmation for throughput winners

Run locally (wall-clock only, no training):
  python scripts/benchmark_sparse_compute.py --variant all

Run on cluster (training + tok/s from trainer):
  python scripts/experiment_plan_real_sparse_compute.py commands --stage rc00
  python scripts/experiment_plan_real_sparse_compute.py probe-and-run --stage real
"""

import experiment_plan_impl_validation as base


REAL_SPARSE_STAGES = (
    "rc00",
    "rc01",
    "rc02",
    "rc03",
    "rc04",
    "rc05",
    "rc06",
    "rc07",
    "real",
    "all",
)
REAL_SPARSE_PREFIXES = tuple(f"{stage}_" for stage in REAL_SPARSE_STAGES if stage != "all")


def bs_for(d_model, *, halt=False):
    if d_model <= 512:
        return 10 if halt else 12
    if d_model <= 768:
        return 8 if halt else 10
    if d_model <= 1024:
        return 6 if halt else 8
    return 4


def _dense(plan, name, question, d_model=512, iterations=4, max_steps=600, **kwargs):
    base.add_sparse_experiment(
        plan, name, question, d_model,
        topk=d_model,
        iterations=iterations,
        synapse_depth=2,
        memory_hidden_dims=2,
        memory_length=8,
        max_steps=max_steps,
        batch_size=kwargs.pop("batch_size", bs_for(d_model)),
        **kwargs,
    )


def _numerical_mask(plan, name, question, d_model=512, topk=64, iterations=4,
                    max_steps=600, **kwargs):
    base.add_sparse_experiment(
        plan, name, question, d_model,
        topk=topk,
        iterations=iterations,
        synapse_depth=2,
        memory_hidden_dims=2,
        memory_length=8,
        max_steps=max_steps,
        batch_size=kwargs.pop("batch_size", bs_for(d_model)),
        **kwargs,
    )


def _regional(plan, name, question, *, num_experts=16, expert_size=32,
              activation_passes=4, shared_experts=1, topk_experts=1,
              dispatch="block_sparse", iterations=4, halt_mode="none",
              halt_threshold=0.30, max_steps=600, **kwargs):
    d_model = num_experts * expert_size
    halt = halt_mode != "none"
    base.add_regional_experiment(
        plan, name, question,
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
        synapse_depth=kwargs.pop("synapse_depth", 2),
        memory_hidden_dims=kwargs.pop("memory_hidden_dims", 2),
        memory_length=kwargs.pop("memory_length", 8),
        max_steps=max_steps,
        batch_size=kwargs.pop("batch_size", bs_for(d_model, halt=halt)),
        moe_dispatch_mode=dispatch,
        **kwargs,
    )


def _single_moe(plan, name, question, *, num_experts=16, expert_size=32,
                topk_experts=2, dispatch="dense_mask", iterations=4,
                max_steps=600, **kwargs):
    base.add_moe_experiment(
        plan, name, question,
        num_experts=num_experts,
        expert_size=expert_size,
        topk_experts=topk_experts,
        routing="topk",
        dispatch=dispatch,
        moe_load_balance_weight=kwargs.pop("moe_load_balance_weight", 1e-2),
        iterations=iterations,
        synapse_depth=kwargs.pop("synapse_depth", 2),
        memory_hidden_dims=kwargs.pop("memory_hidden_dims", 2),
        max_steps=max_steps,
        batch_size=kwargs.pop("batch_size", bs_for(num_experts * expert_size)),
        **kwargs,
    )


def build_plan(stage, plan_size="full"):
    plan = []
    real_stages = ("rc00", "rc01", "rc02", "rc03", "rc04", "rc05", "rc06")

    if stage == "real":
        for item in real_stages:
            plan.extend(build_plan(item, plan_size))
        return base.validate_plan(plan)

    # ------------------------------------------------------------------ rc00
    if stage in ("rc00", "all"):
        _dense(
            plan, "rc00_anchor_dense_d512_tick4",
            "Smoke: dense baseline for sparse-compute comparisons.",
            iterations=4, max_steps=120)
        _numerical_mask(
            plan, "rc00_anchor_numerical_k64",
            "Smoke: post-activation top-k mask (numerical sparsity only).",
            topk=64, max_steps=120)
        _regional(
            plan, "rc00_anchor_regional_blocksparse",
            "Smoke: grouped block-sparse regional backend.",
            dispatch="block_sparse", max_steps=120)
        _regional(
            plan, "rc00_anchor_regional_densemask",
            "Smoke: regional routing with dense_mask dispatch (full expert compute).",
            dispatch="dense_mask", max_steps=120)
        _regional(
            plan, "rc00_anchor_tick_halt_thr0p30",
            "Smoke: adaptive tick early-exit on block-sparse backend.",
            halt_mode="threshold", halt_threshold=0.30, iterations=8, max_steps=120)

    # ------------------------------------------------------------------ rc01
    # Matched ~12.5% active fraction: k64 / shared1+top1 on 16 experts.
    if stage in ("rc01", "all"):
        anchors = [
            ("dense_tick4", "dense", dict(iterations=4)),
            ("numerical_k64", "mask", dict(topk=64, iterations=4)),
            ("numerical_k128", "mask", dict(topk=128, iterations=4)),
            ("tick2_dense", "tick", dict(iterations=2)),
            ("tick8_dense", "tick", dict(iterations=8)),
        ]
        for tag, kind, opts in anchors:
            if kind == "dense" or kind == "tick":
                _dense(
                    plan, f"rc01_{tag}_d512",
                    "rc01 control: full compute reference for matched-active-fraction study.",
                    iterations=opts.get("iterations", 4), max_steps=800)
            else:
                _numerical_mask(
                    plan, f"rc01_{tag}_d512",
                    "rc01 numerical mask: same active fraction, full synapse GEMM.",
                    topk=opts["topk"], max_steps=800)

        for dispatch, label in [
            ("dense_mask", "densemask"),
            ("block_sparse", "blocksparse"),
        ]:
            _regional(
                plan, f"rc01_regional_{label}_d512_p4_sh1_top1",
                f"rc01 structured routing via {dispatch}: should differ in tok/s from numerical mask.",
                num_experts=16, expert_size=32, activation_passes=4,
                shared_experts=1, topk_experts=1, dispatch=dispatch, max_steps=800)

        if base.include_plan_size(plan_size, "full"):
            for dispatch, label in [("dense_mask", "densemask"), ("block_sparse", "blocksparse")]:
                _regional(
                    plan, f"rc01_regional_{label}_d1024_p4_sh1_top1",
                    f"rc01 d1024 structured routing via {dispatch}.",
                    num_experts=16, expert_size=64, activation_passes=4,
                    shared_experts=1, topk_experts=1, dispatch=dispatch, max_steps=800)

    # ------------------------------------------------------------------ rc02
    if stage in ("rc02", "all"):
        dispatches = [
            ("dense_mask", "full expert compute then mask"),
            ("block_sparse", "grouped sparse — only active blocks execute"),
            ("dropless", "dropless grouped dispatch"),
        ]
        if base.include_plan_size(plan_size, "full"):
            dispatches.extend([
                ("capacity_drop", "capacity-limited expert dispatch"),
            ])
        for dispatch, note in dispatches:
            extra = {}
            if dispatch == "capacity_drop":
                extra["moe_capacity_factor"] = 1.0
            _regional(
                plan, f"rc02_dispatch_{dispatch}_d512_p4",
                f"rc02 dispatch A/B: {note}.",
                dispatch=dispatch, max_steps=800, **extra)

        if base.include_plan_size(plan_size, "wide"):
            for cap in (0.75, 1.0, 1.25):
                tag = str(cap).replace(".", "p")
                _regional(
                    plan, f"rc02_capacity_{tag}_d512_p4",
                    f"rc02 capacity_drop sweep at capacity_factor={cap}.",
                    dispatch="capacity_drop",
                    moe_capacity_factor=cap,
                    max_steps=600)

    # ------------------------------------------------------------------ rc03
    if stage in ("rc03", "all"):
        for tick in (2, 4, 6, 8):
            _dense(
                plan, f"rc03_tick_fixed_{tick}_d512",
                f"rc03 true tick reduction: fixed {tick} iterations (uniform batch).",
                iterations=tick, max_steps=800)
            _regional(
                plan, f"rc03_tick_fixed_{tick}_d512_blocksparse",
                f"rc03 block-sparse + fixed {tick} ticks.",
                iterations=tick, dispatch="block_sparse", max_steps=800)

        halt_grid = [
            ("none_tick8", "none", 0.65, 0.0, 8),
            ("threshold0p15", "threshold", 0.15, 1e-3, 8),
            ("threshold0p30", "threshold", 0.30, 1e-3, 8),
            ("threshold0p45", "threshold", 0.45, 1e-3, 8),
            ("confidence0p25", "confidence", 0.30, 1e-3, 8),
        ]
        if base.include_plan_size(plan_size, "full"):
            halt_grid.append(("threshold0p60", "threshold", 0.60, 1e-3, 8))
        for tag, halt_mode, threshold, compute_w, iterations in halt_grid:
            _regional(
                plan, f"rc03_halt_{tag}_d512_blocksparse",
                "rc03 adaptive tick early-exit: measure effective_tick vs tok/s.",
                halt_mode=halt_mode,
                halt_threshold=threshold,
                tick_compute_weight=compute_w,
                iterations=iterations,
                dispatch="block_sparse",
                max_steps=800)

    # ------------------------------------------------------------------ rc04
    if stage in ("rc04", "all"):
        grids = [
            (16, 32, 4, 1, 1, "16x32"),
            (16, 64, 4, 1, 1, "16x64"),
            (32, 16, 4, 1, 1, "32x16"),
            (8, 64, 4, 1, 1, "8x64"),
        ]
        if base.include_plan_size(plan_size, "wide"):
            grids.extend([
                (16, 32, 2, 1, 1, "16x32_p2"),
                (16, 32, 1, 1, 1, "16x32_p1"),
                (16, 32, 4, 1, 2, "16x32_top2"),
            ])
        for num_experts, expert_size, passes, shared, topk, tag in grids:
            _regional(
                plan, f"rc04_block_{tag}_d{num_experts * expert_size}",
                f"rc04 block granularity {tag}: {num_experts} groups × {expert_size} cells.",
                num_experts=num_experts,
                expert_size=expert_size,
                activation_passes=passes,
                shared_experts=shared,
                topk_experts=topk,
                dispatch="block_sparse",
                max_steps=800)

    # ------------------------------------------------------------------ rc05
    if stage in ("rc05", "all"):
        for topk in (1, 2):
            _single_moe(
                plan, f"rc05_moe_densemask_top{topk}_d512",
                f"rc05 Switch-style MoE: dense_mask dispatch, top-{topk} experts.",
                num_experts=16, expert_size=32, topk_experts=topk,
                dispatch="dense_mask", max_steps=800)
        _regional(
            plan, "rc05_regional_blocksparse_d512_p4",
            "rc05 regional block-sparse reference at same d512 capacity.",
            num_experts=16, expert_size=32, activation_passes=4,
            dispatch="block_sparse", max_steps=800)
        if base.include_plan_size(plan_size, "full"):
            _single_moe(
                plan, "rc05_moe_densemask_top2_d1024",
                "rc05 d1024 single-pass MoE dense_mask baseline.",
                num_experts=16, expert_size=64, topk_experts=2,
                dispatch="dense_mask", max_steps=800)
            _regional(
                plan, "rc05_regional_blocksparse_d1024_p4",
                "rc05 d1024 regional block-sparse reference.",
                num_experts=16, expert_size=64, activation_passes=4,
                dispatch="block_sparse", max_steps=800)

    # ------------------------------------------------------------------ rc06
    if stage in ("rc06", "all"):
        composites = [
            ("tick2_p1", dict(iterations=2, activation_passes=1)),
            ("tick2_p4", dict(iterations=2, activation_passes=4)),
            ("tick4_p2", dict(iterations=4, activation_passes=2)),
            ("tick4_p4_halt0p30", dict(
                iterations=4, activation_passes=4,
                halt_mode="threshold", halt_threshold=0.30, tick_compute_weight=1e-3)),
            ("tick4_p4_halt0p45", dict(
                iterations=4, activation_passes=4,
                halt_mode="threshold", halt_threshold=0.45, tick_compute_weight=1e-3)),
        ]
        if base.include_plan_size(plan_size, "wide"):
            composites.append((
                "tick2_p1_halt0p30",
                dict(iterations=2, activation_passes=1,
                     halt_mode="threshold", halt_threshold=0.30, tick_compute_weight=1e-3),
            ))
        for tag, opts in composites:
            halt_mode = opts.pop("halt_mode", "none")
            halt_threshold = opts.pop("halt_threshold", 0.30)
            compute_w = opts.pop("tick_compute_weight", 0.0)
            _regional(
                plan, f"rc06_composite_{tag}_d512",
                f"rc06 composite sparse strategy: {tag}.",
                halt_mode=halt_mode,
                halt_threshold=halt_threshold,
                tick_compute_weight=compute_w,
                dispatch="block_sparse",
                max_steps=1000,
                **opts)

    # ------------------------------------------------------------------ rc07
    if stage in ("rc07", "all"):
        confirmations = [
            ("dense_tick4", lambda: _dense(
                plan, "rc07_confirm_dense_tick4_d512",
                "rc07 long confirm: dense baseline.",
                iterations=4, max_steps=2500)),
            ("numerical_k64", lambda: _numerical_mask(
                plan, "rc07_confirm_numerical_k64_d512",
                "rc07 long confirm: numerical mask should NOT beat dense on tok/s.",
                topk=64, max_steps=2500)),
            ("blocksparse_p4", lambda: _regional(
                plan, "rc07_confirm_blocksparse_p4_d512",
                "rc07 long confirm: block-sparse regional candidate.",
                activation_passes=4, dispatch="block_sparse", max_steps=2500)),
            ("tick2_blocksparse", lambda: _regional(
                plan, "rc07_confirm_tick2_blocksparse_d512",
                "rc07 long confirm: tick2 + block-sparse composite.",
                iterations=2, activation_passes=4,
                dispatch="block_sparse", max_steps=2500)),
            ("halt0p30_blocksparse", lambda: _regional(
                plan, "rc07_confirm_halt0p30_blocksparse_d512",
                "rc07 long confirm: adaptive halt + block-sparse.",
                halt_mode="threshold", halt_threshold=0.30,
                tick_compute_weight=1e-3, iterations=8,
                dispatch="block_sparse", max_steps=2500)),
        ]
        if base.include_plan_size(plan_size, "full"):
            confirmations.append((
                "densemask_p4",
                lambda: _regional(
                    plan, "rc07_confirm_densemask_p4_d512",
                    "rc07 long confirm: dense_mask regional (mask-only control).",
                    activation_passes=4, dispatch="dense_mask", max_steps=2500),
            ))
        for _, fn in confirmations:
            fn()

    return base.validate_plan(plan)


base.configure_plan_defaults(
    metrics_prefix="real_sparse_compute",
    cluster_config=base.DEFAULT_CLUSTER_CONFIG,
)
base.REGIONAL_STAGES = REAL_SPARSE_STAGES
base.REGIONAL_PREFIXES = REAL_SPARSE_PREFIXES
base.build_plan = build_plan


def audit_realism():
    labels = {
        "rc00": "smoke: one anchor per sparsity category",
        "rc01": "real: numerical mask vs block routing at matched active fraction",
        "rc02": "real: dispatch mode wall-clock comparison",
        "rc03": "real: fixed ticks vs adaptive early-exit",
        "rc04": "real: cell-block granularity sweep",
        "rc05": "real: single-pass MoE dense dispatch vs regional block sparse",
        "rc06": "real: composite tick + block routing",
        "rc07": "real: longer confirmation for throughput winners",
    }
    total = 0
    for st in REAL_SPARSE_STAGES:
        if st in {"all", "real"}:
            continue
        n = len(build_plan(st, "full"))
        total += n
        print(f"{st}: {n:3d}  {labels[st]}")
    print(f"all: {len(build_plan('all', 'full')):3d}  (deduplicated stage union)")
    print(f"\nLocal wall-clock benchmark (no cluster):")
    print("  python scripts/benchmark_sparse_compute.py --variant all")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "audit-realism":
        audit_realism()
    else:
        args = base.parse_args()
        args.func(args)
