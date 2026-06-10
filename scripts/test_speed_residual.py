#!/usr/bin/env python3
"""Smoke tests for speed spectrum and residual-compute CLI wiring."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.config import CTMLLMConfig
from model.model_ctm_llm import CTMForCausalLM


def base_config(**overrides):
    cfg = dict(
        vocab_size=6400,
        hidden_size=768,
        num_hidden_layers=2,
        d_model=512,
        d_input=256,
        iterations=4,
        memory_length=8,
        memory_hidden_dims=4,
        deep_nlms=True,
        heads=8,
        n_synch_out=512,
        n_synch_action=512,
        synapse_depth=2,
        self_cond=True,
        cross_layer_state=False,
        moe_routing_mode="regional_shared_topk",
        moe_num_experts=16,
        moe_topk_experts=1,
        moe_shared_experts=1,
        moe_expert_size=32,
        moe_activation_passes=4,
        moe_dispatch_mode="dense_mask",
        tick_halt_mode="none",
    )
    cfg.update(overrides)
    return CTMLLMConfig(**cfg)


def test_speed_spectrum_finite():
    cfg = base_config(
        speed_spectrum_mode="ema_spectrum",
        speed_ema_decays="0.90,0.97,0.995",
        speed_distill_weight=0.05,
        speed_target_ticks="fast,mid,slow",
        speed_warmup_steps=0,
    )
    model = CTMForCausalLM(cfg)
    model.train()
    ids = torch.randint(0, cfg.vocab_size, (2, 12))
    loss, _, _ = model.forward_train(ids, ids)
    assert torch.isfinite(loss)
    assert model.last_speed_loss >= 0.0
    loss.backward()
    model.update_speed_teachers()
    print(f"speed spectrum smoke loss={loss.item():.4f} speed={model.last_speed_loss:.4f}")


def test_residual_observe_finite():
    cfg = base_config(
        residual_compute_mode="observe",
        residual_track_deltas=1,
        residual_synapse_mode="dense_delta",
        residual_nlm_mode="output_delta",
        residual_attention_mode="kv_cache",
        residual_sync_mode="recursive_pairs",
    )
    model = CTMForCausalLM(cfg)
    model.train()
    ids = torch.randint(0, cfg.vocab_size, (2, 12))
    loss, _, _ = model.forward_train(ids, ids)
    assert torch.isfinite(loss)
    assert model.last_residual_delta_l1 > 0.0
    loss.backward()
    print(
        f"residual observe smoke loss={loss.item():.4f} "
        f"delta_l1={model.last_residual_delta_l1:.4f}"
    )


def main():
    test_speed_spectrum_finite()
    test_residual_observe_finite()
    print("speed/residual smoke tests passed")


if __name__ == "__main__":
    main()
