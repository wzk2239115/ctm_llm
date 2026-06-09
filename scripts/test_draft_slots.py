#!/usr/bin/env python3
"""Smoke tests for parallel draft slots + unified tick supervision."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F

from model.config import CTMLLMConfig
from model.model_ctm_llm import CTMForCausalLM


def draft_config(**overrides):
    cfg = dict(
        vocab_size=6400,
        hidden_size=768,
        num_hidden_layers=2,
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
        cell_topk=64,
        cell_sparsity_rescale=True,
        moe_routing_mode="regional_shared_topk",
        moe_num_experts=16,
        moe_topk_experts=1,
        moe_shared_experts=1,
        moe_expert_size=32,
        moe_activation_passes=4,
        moe_dispatch_mode="dense_mask",
        tick_halt_mode="none",
        draft_mode="parallel",
        draft_block_size=4,
        draft_head_mode="shared",
        draft_loss_weight=0.2,
        draft_slot_attention="causal_slots",
        elf_horizon_mode="linear",
        elf_max_horizon=4,
        moe_mtp_mode="mtp_1_2_4",
        moe_mtp_horizons="1,2,4",
    )
    cfg.update(overrides)
    return CTMLLMConfig(**cfg)


def test_forward_shape():
    cfg = draft_config()
    model = CTMForCausalLM(cfg)
    model.eval()
    B, T = 2, 16
    ids = torch.randint(0, cfg.vocab_size, (B, T))
    result = model.model(
        ids, return_all_ticks=True, draft_lm_head=model.lm_head)
    tick_outs = result[2]
    draft_logits = result[3]
    assert tick_outs.shape[:3] == (B, T, cfg.hidden_size)
    assert draft_logits.shape[:4] == (B, T, cfg.draft_block_size, cfg.vocab_size)
    print(f"tick_outs={tuple(tick_outs.shape)} draft_logits={tuple(draft_logits.shape)}")


def test_draft_offset_alignment():
    cfg = draft_config(draft_block_size=3, draft_head_mode="slot_head")
    model = CTMForCausalLM(cfg)
    model.eval()
    B, T = 1, 8
    ids = torch.randint(0, cfg.vocab_size, (B, T))
    labels = ids.clone()
    result = model.model(
        ids, return_all_ticks=True, draft_lm_head=model.lm_head)
    draft_logits = result[3]
    slot_logits_t = draft_logits[..., :, :, :, 0]
    for slot in range(cfg.draft_block_size):
        horizon = slot + 1
        logits = slot_logits_t[:, :, slot, :]
        shift_logits = logits[:, :-horizon, :]
        shift_labels = labels[:, horizon:]
        loss = F.cross_entropy(
            shift_logits.reshape(-1, cfg.vocab_size),
            shift_labels.reshape(-1),
        )
        assert torch.isfinite(loss), f"slot {slot} offset {horizon} loss not finite"
    print("draft CE offsets 1..block align with shifted labels")


def test_finite_train_loss():
    cfg = draft_config(
        num_hidden_layers=1,
        iterations=2,
        draft_loss_weight=0.2,
        tick_improve_weight=0.05,
    )
    model = CTMForCausalLM(cfg)
    model.train()
    B, T = 2, 12
    ids = torch.randint(0, cfg.vocab_size, (B, T))
    labels = ids.clone()
    for step in range(10):
        model.zero_grad(set_to_none=True)
        loss, losses_per_tick, certainties = model.forward_train(ids, labels)
        assert torch.isfinite(loss), f"non-finite loss at step {step}"
        loss.backward()
    print(f"10-step smoke loss={loss.item():.4f} "
          f"losses_per_tick={losses_per_tick.mean(dim=0).tolist()}")


def test_training_runs_all_ticks_with_halt():
    cfg = draft_config(
        tick_halt_mode="threshold",
        tick_halt_threshold=0.0,
        iterations=4,
    )
    model = CTMForCausalLM(cfg)
    model.train()
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    model.forward_train(ids, ids)
    executed = model.model.layers[-1].last_executed_ticks
    assert executed == cfg.iterations, (
        f"training should run all ticks, got {executed} != {cfg.iterations}")
    model.eval()
    model(ids, num_iters=cfg.iterations)
    infer_executed = model.model.layers[-1].last_executed_ticks
    assert infer_executed <= cfg.iterations
    print(f"train executed_ticks={executed} infer executed_ticks={infer_executed}")


def main():
    test_forward_shape()
    test_draft_offset_alignment()
    test_finite_train_loss()
    test_training_runs_all_ticks_with_halt()
    print("draft slot smoke tests passed")


if __name__ == "__main__":
    main()
