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


def test_block_skip_finite():
    cfg = base_config(
        residual_compute_mode="block_skip",
        residual_synapse_mode="block_delta_skip",
        residual_nlm_mode="output_delta",
        residual_num_groups=16,
        residual_active_ratio=0.25,
        residual_gate_threshold=0.15,
        residual_full_refresh_interval=4,
        residual_compute_weight=0.01,
        residual_delta_l1_weight=1e-4,
        moe_routing_mode="regional_shared_topk",
        moe_num_experts=16,
        moe_expert_size=32,
        moe_activation_passes=4,
        moe_dispatch_mode="dense_mask",
    )
    model = CTMForCausalLM(cfg)
    model.train()
    ids = torch.randint(0, cfg.vocab_size, (2, 12))
    loss, _, _ = model.forward_train(ids, ids)
    assert torch.isfinite(loss)
    assert 0.0 <= model.last_residual_skip_ratio <= 1.0
    loss.backward()
    print(
        f"block skip smoke loss={loss.item():.4f} "
        f"skip={model.last_residual_skip_ratio:.3f} "
        f"delta={model.last_residual_delta_l1:.4f}"
    )


def test_nlm_recursive_finite():
    cfg = base_config(
        memory_length=10,
        iterations=8,
        residual_compute_mode="nlm_recursive",
        residual_synapse_mode="block_delta_skip",
        residual_nlm_mode="recursive_fast",
        residual_full_refresh_interval=4,
        residual_compute_weight=0.015,
        residual_delta_l1_weight=1e-4,
        moe_routing_mode="regional_shared_topk",
        moe_num_experts=16,
        moe_expert_size=32,
        moe_activation_passes=4,
        moe_dispatch_mode="dense_mask",
    )
    model = CTMForCausalLM(cfg)
    model.train()
    ids = torch.randint(0, cfg.vocab_size, (2, 12))
    loss, _, _ = model.forward_train(ids, ids, num_iters=cfg.iterations)
    assert torch.isfinite(loss)
    assert model.last_nlm_fast_ratio > 0.0
    loss.backward()
    print(
        f"nlm recursive smoke loss={loss.item():.4f} "
        f"fast={model.last_nlm_fast_ratio:.3f} skip={model.last_residual_skip_ratio:.3f}"
    )


def test_tick_controller_finite():
    cfg = base_config(
        iterations=12,
        residual_compute_mode="tick_controller",
        residual_synapse_mode="block_delta_skip",
        residual_nlm_mode="hybrid_fast_full",
        residual_tick_controller="threshold",
        residual_full_refresh_interval=4,
        residual_compute_weight=0.01,
        residual_delta_l1_weight=1e-4,
        moe_routing_mode="regional_shared_topk",
        moe_num_experts=16,
        moe_expert_size=32,
        moe_activation_passes=4,
        moe_dispatch_mode="dense_mask",
    )
    model = CTMForCausalLM(cfg)
    model.train()
    ids = torch.randint(0, cfg.vocab_size, (2, 12))
    loss, _, _ = model.forward_train(ids, ids, num_iters=cfg.iterations)
    assert torch.isfinite(loss)
    loss.backward()
    print(
        f"tick controller smoke loss={loss.item():.4f} "
        f"skip={model.last_residual_skip_ratio:.3f} "
        f"stop={getattr(model.model.layers[-1], 'last_controller_stop_ratio', 0.0):.3f}"
    )


def test_speed_cells_async_finite():
    cfg = base_config(
        iterations=8,
        async_tick_mode="banded",
        async_tick_periods="1,2,4",
        residual_compute_mode="speed_cells",
        residual_synapse_mode="block_delta_skip",
        residual_nlm_mode="hybrid_fast_full",
        residual_speed_cells="fast_mid_slow",
        residual_full_refresh_interval=4,
        residual_compute_weight=0.01,
        speed_spectrum_mode="ema_spectrum",
        speed_distill_weight=0.05,
        speed_warmup_steps=0,
    )
    from model.model_ctm_llm import build_ctm_for_causal_lm

    model = build_ctm_for_causal_lm(cfg)
    model.train()
    ids = torch.randint(0, cfg.vocab_size, (2, 12))
    loss, _, _ = model.forward_train(ids, ids, num_iters=cfg.iterations)
    assert torch.isfinite(loss)
    loss.backward()
    print(
        f"speed cells async smoke loss={loss.item():.4f} "
        f"speed={model.last_speed_loss:.4f} skip={model.last_residual_skip_ratio:.3f}"
    )


def test_objective_hybrid_finite():
    cfg = base_config(
        objective_mode="hybrid_flow_ce",
        objective_denoise_weight=0.5,
        objective_ce_weight=0.5,
        objective_latent_space="token_embed",
        objective_self_cond_prob=0.5,
    )
    model = CTMForCausalLM(cfg)
    model.train()
    ids = torch.randint(0, cfg.vocab_size, (2, 12))
    loss, _, _ = model.forward_train(ids, ids)
    assert torch.isfinite(loss)
    assert model.last_objective_denoise >= 0.0
    loss.backward()
    print(
        f"objective hybrid smoke loss={loss.item():.4f} "
        f"ce={model.last_objective_ce:.4f} denoise={model.last_objective_denoise:.4f}"
    )


def main():
    test_speed_spectrum_finite()
    test_residual_observe_finite()
    test_block_skip_finite()
    test_nlm_recursive_finite()
    test_tick_controller_finite()
    test_speed_cells_async_finite()
    test_objective_hybrid_finite()
    print("speed/residual smoke tests passed")


if __name__ == "__main__":
    main()
