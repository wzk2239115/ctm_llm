#!/usr/bin/env python3
"""Smoke tests for banded async tick clocks."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.config import CTMLLMConfig
from model.model_ctm_async import AsyncCTMBlock, AsyncCTMForCausalLM


def block_config(**overrides):
    base = dict(
        hidden_size=768,
        d_model=512,
        d_input=256,
        iterations=8,
        memory_length=8,
        memory_hidden_dims=4,
        deep_nlms=True,
        heads=8,
        n_synch_out=512,
        n_synch_action=512,
        synapse_depth=2,
        self_cond=True,
        dropout=0.0,
        cell_sparsity_mode="topk",
        cell_topk=512,
        moe_routing_mode="regional_shared_topk",
        moe_num_experts=16,
        moe_topk_experts=1,
        moe_shared_experts=1,
        moe_expert_size=32,
        moe_activation_passes=2,
        moe_dispatch_mode="block_sparse",
    )
    base.update(overrides)
    return CTMLLMConfig(**base)


def test_clock_schedule():
    block = AsyncCTMBlock(0, block_config(async_tick_periods="1,2,4,8"))
    fired = [block.clock_engine.bands_fired_count(t) for t in range(8)]
    assert fired == [4, 1, 2, 1, 3, 1, 2, 1]
    print("clock schedule:", fired)


def test_forward_runs():
    torch.manual_seed(0)
    cfg = block_config(async_tick_periods="1,2,4,8")
    block = AsyncCTMBlock(0, cfg)
    x = torch.randn(2, 16, cfg.hidden_size)
    pos = torch.arange(16).unsqueeze(0).expand(2, -1)
    pos_emb = torch.randn(16, cfg.d_input)
    result = block(
        x, pos_emb=pos_emb, track=False, return_all_ticks=True, num_iters=8)
    assert result.hidden.shape == x.shape
    assert result.extras["tick_outputs"].shape[-1] == 8
    print("forward ok:", result.hidden.shape, result.extras["tick_outputs"].shape)


def test_local_ticks_diverge():
    block = AsyncCTMBlock(0, block_config(async_tick_periods="1,2,4,8", iterations=8))
    x = torch.randn(1, 4, block.config.hidden_size)
    pos_emb = torch.randn(4, block.config.d_input)
    block(x, pos_emb=pos_emb, num_iters=8)
    local = block.last_async_local_ticks
    assert local[0] == 8
    assert local[1] == 4
    assert local[2] == 2
    assert local[3] == 1
    print("local ticks:", local)


def test_fast_band_moves_every_tick():
    torch.manual_seed(1)
    cfg = block_config(async_tick_periods="1,2,4,8")
    block = AsyncCTMBlock(0, cfg)
    x = torch.randn(1, 8, cfg.hidden_size)
    pos_emb = torch.randn(8, cfg.d_input)
    result = block(
        x, pos_emb=pos_emb, return_all_ticks=True, num_iters=4)
    tick_outs = result.extras["tick_outputs"]
    diffs = [
        (tick_outs[..., t] - tick_outs[..., t - 1]).abs().mean().item()
        for t in range(1, 4)
    ]
    assert all(d > 0 for d in diffs), f"tick outputs should change each step: {diffs}"
    print("per-tick output deltas:", [round(d, 6) for d in diffs])


def test_train_forward():
    torch.manual_seed(2)
    cfg = block_config(
        num_hidden_layers=2,
        async_tick_periods="1,2,4",
        async_fast_output_weight=0.25,
        fast_output_mode="anytime",
        fast_output_weight=0.1,
        habit_output_weight=0.1,
        fast_output_ticks="1,4",
    )
    model = AsyncCTMForCausalLM(cfg)
    ids = torch.randint(0, 100, (2, 32))
    labels = ids.clone()
    loss, losses, certainties = model.forward_train(ids, labels, num_iters=4)
    assert torch.isfinite(loss)
    assert losses.shape == (2, 4)
    print("train loss:", float(loss))


def main():
    test_clock_schedule()
    test_forward_runs()
    test_local_ticks_diverge()
    test_fast_band_moves_every_tick()
    test_train_forward()
    print("all async tick tests passed")


if __name__ == "__main__":
    main()
