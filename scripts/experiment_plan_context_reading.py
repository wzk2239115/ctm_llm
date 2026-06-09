#!/usr/bin/env python3
"""Plan CTM-LLM experiments where ticks read context, memory, retrieval, experts, and egram drafts."""
import argparse
import csv
import os
import shlex


BASE_ARGS = {
    "epochs": 1,
    "batch_size": 4,
    "accumulation_steps": 1,
    "num_hidden_layers": 16,
    "hidden_size": 768,
    "d_model": 512,
    "d_input": 256,
    "heads": 8,
    "n_synch_out": 512,
    "n_synch_action": 512,
    "iterations": 2,
    "memory_length": 8,
    "memory_hidden_dims": 2,
    "deep_nlms": 1,
    "synapse_depth": 2,
    "tick_loss_mode": "min_conf",
    "cell_sparsity_mode": "topk",
    "cell_topk": 512,
    "cell_sparsity_rescale": 1,
    "moe_routing_mode": "regional_shared_topk",
    "moe_num_experts": 16,
    "moe_topk_experts": 1,
    "moe_shared_experts": 1,
    "moe_expert_size": 32,
    "moe_load_balance_weight": 1e-2,
    "moe_activation_passes": 3,
    "moe_region_diversity_weight": 1e-3,
    "self_cond": 1,
    "cross_layer_state": 1,
    "max_seq_len": 512,
    "log_interval": 20,
    "save_interval": 1000,
    "max_steps": 1000,
    "dtype": "bfloat16",
    "context_reading_mode": "none",
    "context_reading_sources": "local,compressed,retrieval,expert,egram",
    "context_reading_gate_init": -2.0,
    "context_local_window": 32,
    "context_compressed_stride": 16,
    "context_retrieval_topk": 8,
    "context_expert_memory_slots": 4,
    "context_egram_decay": 0.75,
}


def merge_args(**overrides):
    args = dict(BASE_ARGS)
    args.update(overrides)
    return args


def experiment(name, question, **overrides):
    args = merge_args(**overrides)
    args["experiment_name"] = name
    args["swanlab_name"] = name
    args["save_weight"] = name
    return {"name": name, "question": question, "args": args}


def build_plan(stage, plan_size):
    plan = []

    if stage in ("smoke", "all"):
        plan.extend([
            experiment(
                "cr00_regional_tick2_no_context",
                "Regional MoE tick2 anchor without extra context-reading sources.",
                max_steps=120,
            ),
            experiment(
                "cr00_context_all_tick2",
                "Smoke-check all context-reading sources fused into each CTM tick.",
                context_reading_mode="fusion",
                max_steps=120,
            ),
        ])

    if stage in ("sources", "all"):
        source_questions = {
            "local": "Does a causal local-window read help common short-range next-token prediction?",
            "compressed": "Does compressed causal history give ticks a useful long-context summary?",
            "retrieval": "Does top-k token retrieval outperform repeating the same tick state?",
            "expert": "Can per-expert learned long-term memory slots stabilize regional routing?",
            "egram": "Does an EAGLE/egram-like draft trace improve later tick predictions?",
        }
        for source, question in source_questions.items():
            plan.append(experiment(
                f"cr01_source_{source}",
                question,
                context_reading_mode="fusion",
                context_reading_sources=source,
            ))

    if stage in ("combos", "all"):
        combos = [
            ("read_ctx", "local,compressed,retrieval",
             "Current-context, compressed-history, and retrieved-token reading without learned memories."),
            ("memory", "expert,egram",
             "Learned expert memory plus tick-to-tick egram draft without extra token retrieval."),
            ("ctx_egram", "local,compressed,retrieval,egram",
             "External context reads plus draft integration, excluding expert memory."),
            ("all", "local,compressed,retrieval,expert,egram",
             "Full recurrent reading: context slices, compressed history, retrieval, expert memory, and egram."),
        ]
        if plan_size in ("full", "wide"):
            combos.extend([
                ("retrieval_egram", "retrieval,egram",
                 "Pair sparse token retrieval with draft state to test DeepSeek-style draft refinement."),
                ("compressed_expert", "compressed,expert",
                 "Pair long-history summaries with expert memory slots."),
            ])
        for tag, sources, question in combos:
            plan.append(experiment(
                f"cr02_combo_{tag}",
                question,
                context_reading_mode="fusion",
                context_reading_sources=sources,
            ))

    if stage in ("longctx", "all"):
        seq_lens = [1024]
        if plan_size in ("full", "wide"):
            seq_lens.append(2048)
        if plan_size == "wide":
            seq_lens.append(4096)
        for seq_len in seq_lens:
            batch = 2 if seq_len <= 1024 else 1
            plan.append(experiment(
                f"cr03_longctx_seq{seq_len}_baseline",
                "Long-context anchor without recurrent reading.",
                max_seq_len=seq_len,
                batch_size=batch,
                max_steps=700,
            ))
            plan.append(experiment(
                f"cr03_longctx_seq{seq_len}_all",
                "Test whether recurrent reading becomes more useful as the context becomes longer.",
                max_seq_len=seq_len,
                batch_size=batch,
                max_steps=700,
                context_reading_mode="fusion",
                context_reading_sources="local,compressed,retrieval,expert,egram",
                context_local_window=min(128, seq_len),
                context_compressed_stride=32 if seq_len <= 1024 else 64,
                context_retrieval_topk=16,
            ))

    if stage in ("ticks", "all") and plan_size in ("full", "wide"):
        for ticks in [1, 2, 4]:
            plan.append(experiment(
                f"cr04_ticks{ticks}_all",
                "Check whether context-reading makes extra ticks useful again.",
                iterations=ticks,
                context_reading_mode="fusion",
                context_reading_sources="local,compressed,retrieval,expert,egram",
            ))

    return plan


def command_for(exp, python="python"):
    parts = [python, "trainer/train.py"]
    for key, value in exp["args"].items():
        parts.extend([f"--{key}", str(value)])
    return " ".join(shlex.quote(part) for part in parts)


def print_commands(args):
    plan = build_plan(args.stage, args.plan_size)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "question", "command"])
        writer.writeheader()
        for exp in plan:
            cmd = command_for(exp, python=args.python)
            writer.writerow({"name": exp["name"], "question": exp["question"], "command": cmd})
            print(cmd)
    print(f"\nWrote {len(plan)} commands to {args.output}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="all",
                        choices=["smoke", "sources", "combos", "longctx", "ticks", "all"])
    parser.add_argument("--plan_size", default="core", choices=["core", "full", "wide"])
    parser.add_argument("--output", default="runs/experiment_plans/context_reading_plan.csv")
    parser.add_argument("--python", default="python")
    return parser.parse_args()


if __name__ == "__main__":
    print_commands(parse_args())
