import pytest
import torch
import math

from model.config import CTMLLMConfig
from model.model_ctm_llm import CTMForCausalLM


def _trainable_config(**overrides):
    cfg = dict(
        vocab_size=6400,
        hidden_size=128,
        num_hidden_layers=1,
        d_model=64,
        d_input=32,
        iterations=2,
        memory_length=4,
        memory_hidden_dims=4,
        deep_nlms=False,
        heads=4,
        n_synch_out=32,
        n_synch_action=32,
        synapse_depth=1,
        self_cond=False,
        cross_layer_state=False,
        dropout=0.0,
        tick_halt_mode="none",
    )
    cfg.update(overrides)
    return CTMLLMConfig(**cfg)


class TestGradientFlow:
    def test_all_params_get_gradients(self):
        cfg = _trainable_config()
        model = CTMForCausalLM(cfg)
        ids = torch.randint(0, cfg.vocab_size, (2, 8))
        loss, _, _ = model.forward_train(ids, ids, num_iters=2)
        loss.backward()
        non_frozen = [
            n for n, p in model.named_parameters()
            if p.requires_grad
        ]
        with_grad = [
            n for n, p in model.named_parameters()
            if p.requires_grad and p.grad is not None
        ]
        assert len(with_grad) > 0
        assert len(with_grad) >= len(non_frozen) * 0.8, (
            f"Too many params without gradients: {len(non_frozen) - len(with_grad)}/{len(non_frozen)}")

    def test_loss_decreases_over_steps(self):
        cfg = _trainable_config()
        model = CTMForCausalLM(cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        ids = torch.randint(0, cfg.vocab_size, (2, 8))
        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            loss, _, _ = model.forward_train(ids, ids, num_iters=2)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0], (
            f"loss should decrease: {losses[0]:.4f} -> {losses[-1]:.4f}")

    def test_no_gradient_leak_between_steps(self):
        cfg = _trainable_config()
        model = CTMForCausalLM(cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        ids = torch.randint(0, cfg.vocab_size, (2, 8))
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = model.forward_train(ids, ids, num_iters=2)
            loss.backward()
        grad_norms = [p.grad.norm().item() for p in model.parameters()
                      if p.grad is not None]
        assert all(math.isfinite(g) for g in grad_norms)


class TestEndToEndWithMoE:
    def test_train_with_regional_moe(self):
        cfg = _trainable_config(
            moe_routing_mode="regional_shared_topk",
            moe_num_experts=4,
            moe_topk_experts=1,
            moe_shared_experts=1,
            moe_expert_size=16,
            moe_load_balance_weight=0.01,
            moe_dispatch_mode="dense_mask",
        )
        model = CTMForCausalLM(cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        ids = torch.randint(0, cfg.vocab_size, (2, 8))
        for _ in range(3):
            optimizer.zero_grad()
            loss, _, _ = model.forward_train(ids, ids, num_iters=2)
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()


class TestEndToEndWithDraft:
    def test_train_with_draft_parallel(self):
        cfg = _trainable_config(
            draft_mode="parallel",
            draft_block_size=2,
            draft_head_mode="shared",
            draft_loss_weight=0.1,
        )
        model = CTMForCausalLM(cfg)
        ids = torch.randint(0, cfg.vocab_size, (2, 8))
        loss, losses, certainties = model.forward_train(ids, ids, num_iters=2)
        assert torch.isfinite(loss)
        loss.backward()

    def test_train_with_draft_revise(self):
        cfg = _trainable_config(
            draft_mode="revise",
            draft_block_size=2,
            draft_head_mode="slot_adapter",
            draft_loss_weight=0.1,
        )
        model = CTMForCausalLM(cfg)
        ids = torch.randint(0, cfg.vocab_size, (2, 8))
        loss, _, _ = model.forward_train(ids, ids, num_iters=2)
        assert torch.isfinite(loss)
        loss.backward()


class TestEndToEndMultiLayer:
    def test_two_layers(self):
        cfg = _trainable_config(num_hidden_layers=2, cross_layer_state=True)
        model = CTMForCausalLM(cfg)
        ids = torch.randint(0, cfg.vocab_size, (2, 8))
        loss, _, _ = model.forward_train(ids, ids, num_iters=2)
        assert torch.isfinite(loss)
        loss.backward()

    def test_generate_multi_layer(self):
        cfg = _trainable_config(num_hidden_layers=2)
        model = CTMForCausalLM(cfg)
        model.eval()
        ids = torch.randint(0, cfg.vocab_size, (1, 4))
        generated = model.generate(ids, max_new_tokens=4, num_iters=2)
        assert generated.shape == (1, 8)
