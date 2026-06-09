#!/usr/bin/env python3
"""Minimal parity checks for grouped sparse regional backend vs dense_mask."""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F

from model.config import CTMLLMConfig
from model.model_ctm_llm import CTMBlock, CTMForCausalLM


def parity_config(dispatch_mode, seed=42):
    return CTMLLMConfig(
        vocab_size=6400,
        hidden_size=768,
        num_hidden_layers=1,
        d_model=512,
        d_input=256,
        iterations=4,
        memory_length=10,
        memory_hidden_dims=4,
        deep_nlms=True,
        heads=8,
        n_synch_out=512,
        n_synch_action=512,
        synapse_depth=3,
        self_cond=True,
        cross_layer_state=False,
        dropout=0.0,
        cell_sparsity_mode="topk",
        cell_topk=512,
        cell_sparsity_rescale=True,
        moe_routing_mode="regional_shared_topk",
        moe_num_experts=16,
        moe_topk_experts=1,
        moe_shared_experts=1,
        moe_expert_size=32,
        moe_activation_passes=4,
        moe_dispatch_mode=dispatch_mode,
        tick_halt_mode="none",
        moe_mtp_mode="none",
        diff_cell_mode="none",
    )


def tie_group_weights_from_dense(block):
    """Copy dense synapse/trace weights into per-expert group modules (block-local)."""
    if block.group_synapses is None:
        return
    expert_size = block.moe_expert_size
    num_experts = block.moe_num_experts
    synapse = block.synapses
    if not isinstance(synapse, torch.nn.Sequential):
        return
    dense_linear = synapse[1]
    w = dense_linear.weight.data
    b = dense_linear.bias.data if dense_linear.bias is not None else None
    d_input = block.d_input
    d_model = block.d_model
    synapse_in = d_input + d_model + (d_model if block.self_cond else 0)
    for expert in range(num_experts):
        start = expert * expert_size
        end = start + expert_size
        group = block.group_synapses[expert]
        group_linear = group[1]
        group_in = d_input + expert_size + (expert_size if block.self_cond else 0)
        gw = group_linear.weight.data.zero_()
        if b is not None and group_linear.bias is not None:
            group_linear.bias.data.copy_(b[start * 2:end * 2])
        gw[:d_input, :expert_size * 2].copy_(w[start * 2:end * 2, :d_input])
        gw[d_input:d_input + expert_size, :expert_size * 2].copy_(
            w[start * 2:end * 2, d_input + start:d_input + end])
        if block.self_cond:
            off = d_input + d_model
            gw[d_input + expert_size:, :expert_size * 2].copy_(
                w[start * 2:end * 2, off + start:off + end])
        block.group_trace_processors[expert].load_state_dict(
            block.trace_processor.state_dict(), strict=False)


def collect_regional_masks_from_scores(block, routing_scores):
    B, T, num_experts = routing_scores.shape
    shared = min(block.moe_shared_experts, num_experts)
    routed_count = max(num_experts - shared, 1)
    topk = block._effective_moe_topk(routed_count)
    max_distinct_passes = max(1, math.ceil(routed_count / max(1, topk)))
    passes = min(max(1, block.moe_activation_passes), max_distinct_passes)
    selected = torch.zeros(B, T, routed_count, dtype=torch.bool, device=routing_scores.device)
    masks = []
    for _ in range(passes):
        expert_mask, _ = block._route_experts(
            routing_scores, shared, routed_count, topk, selected)
        selected = selected | expert_mask[:, :, shared:].bool()
        masks.append(expert_mask.clone())
    return torch.stack(masks, dim=0)


def collect_regional_masks_dense(block, activated):
    B, T, _ = activated.shape
    expert_size = block.moe_expert_size
    num_experts = block.moe_num_experts
    routing_scores = activated.view(B, T, num_experts, expert_size).abs().mean(dim=-1).detach()
    shared = min(block.moe_shared_experts, num_experts)
    routed_count = max(num_experts - shared, 1)
    topk = block._effective_moe_topk(routed_count)
    routed_scores = routing_scores[:, :, shared:]
    max_distinct_passes = max(1, math.ceil(routed_count / max(1, topk)))
    passes = min(max(1, block.moe_activation_passes), max_distinct_passes)
    selected = torch.zeros_like(routed_scores, dtype=torch.bool)
    masks = []
    for _ in range(passes):
        masked_scores = routed_scores.masked_fill(selected, float("-inf"))
        idx = torch.topk(masked_scores, topk, dim=-1).indices
        routed_mask = torch.zeros_like(routed_scores)
        routed_mask.scatter_(-1, idx, 1.0)
        selected = selected | routed_mask.bool()
        expert_mask = torch.zeros_like(routing_scores)
        expert_mask[:, :, shared:] = routed_mask
        if shared > 0:
            expert_mask[:, :, :shared] = 1.0
        masks.append(expert_mask.clone())
    return torch.stack(masks, dim=0)


@torch.no_grad()
def run_block_tick(block, x, seed=0):
    torch.manual_seed(seed)
    B, T, _ = x.shape
    device = x.device
    block.eval()
    normed = block.input_norm(x)
    kv = block.kv_proj(normed)
    k = v = kv
    activated = block.start_activated_state.view(1, 1, -1).expand(B, T, -1).contiguous()
    state_trace = block.start_trace.view(1, 1, block.d_model, block.memory_length) \
        .expand(B, T, -1, -1).contiguous()
    r_a = torch.exp(-block.decay_action).view(1, 1, -1).expand(B, T, -1)
    r_o = torch.exp(-block.decay_out).view(1, 1, -1).expand(B, T, -1)
    alpha_a = beta_a = alpha_o = beta_o = None
    sync_a, alpha_a, beta_a = block._compute_synch(
        activated, alpha_a, beta_a, r_a, block.action_left, block.action_right)
    q = block.q_proj(sync_a)
    q_mh = q.view(B, T, block.heads, block.head_dim).transpose(1, 2)
    k_mh = k.view(B, T, block.heads, block.head_dim).transpose(1, 2)
    v_mh = v.view(B, T, block.heads, block.head_dim).transpose(1, 2)
    attn_out = F.scaled_dot_product_attention(q_mh, k_mh, v_mh, is_causal=True)
    attn = block.o_proj(attn_out.transpose(1, 2).reshape(B, T, -1))
    prev_sync_o_activated = None

    if block._use_group_sparse_backend():
        out_act, out_trace = block._run_group_sparse_regional_tick(
            attn, activated, state_trace, prev_sync_o_activated)
    else:
        pre_syn_parts = [attn, activated]
        if block.self_cond:
            pre_syn_parts.append(torch.zeros(B, T, block.d_model, device=device))
        pre_syn = torch.cat(pre_syn_parts, dim=-1)
        state = block.synapses(pre_syn)
        state_trace = torch.cat([state_trace[:, :, :, 1:], state.unsqueeze(-1)], dim=-1)
        dense_act = block.trace_processor(state_trace)
        out_act = block._apply_cell_sparsity(dense_act)
        out_trace = state_trace

    routing_act, routing_trace = block._compute_dense_pre_mask_activation(
        attn, activated, state_trace, prev_sync_o_activated)
    return {
        "activated": out_act,
        "trace": out_trace,
        "routing_act": routing_act,
        "attn": attn,
        "state_trace_in": state_trace,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=8)
    args = parser.parse_args()
    device = args.device
    torch.manual_seed(args.seed)

    B, T = args.batch_size, args.seq_len
    x = torch.randn(B, T, 768, device=device)

    cfg_dense = parity_config("dense_mask")
    cfg_sparse = parity_config("block_sparse")
    block_dense = CTMBlock(0, cfg_dense).to(device)
    block_sparse = CTMBlock(0, cfg_sparse).to(device)
    block_sparse.load_state_dict(block_dense.state_dict(), strict=False)
    tie_group_weights_from_dense(block_sparse)

    out_dense = run_block_tick(block_dense, x, seed=args.seed)
    out_sparse = run_block_tick(block_sparse, x, seed=args.seed)

    dense_masks = collect_regional_masks_dense(block_dense, out_dense["routing_act"])
    routing_scores = out_dense["routing_act"].view(
        B, T, block_dense.moe_num_experts, block_dense.moe_expert_size
    ).abs().mean(dim=-1).detach()
    sparse_masks = collect_regional_masks_from_scores(block_sparse, routing_scores)

    mask_match = torch.equal(dense_masks, sparse_masks)
    act_diff = (out_dense["activated"] - out_sparse["activated"]).abs()
    print(f"routing masks match: {mask_match}")
    print(f"dense activated: mean={out_dense['activated'].mean():.4f} "
          f"std={out_dense['activated'].std():.4f}")
    print(f"sparse activated: mean={out_sparse['activated'].mean():.4f} "
          f"std={out_sparse['activated'].std():.4f}")
    print(f"activated max abs diff: {act_diff.max():.6f}")
    print(f"activated mean abs diff: {act_diff.mean():.6f}")
    assert torch.isfinite(out_sparse["activated"]).all(), "sparse activated has non-finite values"
    assert torch.isfinite(out_dense["activated"]).all(), "dense activated has non-finite values"

    cfg = parity_config("block_sparse")
    cfg.num_hidden_layers = 2
    model = CTMForCausalLM(cfg).to(device)
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    out = model(ids)
    logits = out["logits"]
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, cfg.vocab_size),
        ids[:, 1:].reshape(-1),
    )
    print(f"full model forward loss (random init): {loss.item():.4f}")
    assert torch.isfinite(loss), "loss is not finite"

    print("parity smoke checks passed")


if __name__ == "__main__":
    main()
