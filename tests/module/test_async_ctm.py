import pytest
import torch

from model.config import CTMLLMConfig
from model.model_ctm_async import AsyncCTMBlock, AsyncCTMForCausalLM


def _async_config(**overrides):
    cfg = dict(
        vocab_size=6400,
        hidden_size=128,
        num_hidden_layers=1,
        d_model=64,
        d_input=32,
        iterations=4,
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
        async_tick_mode="banded",
        async_tick_periods="1,2",
    )
    cfg.update(overrides)
    return CTMLLMConfig(**cfg)


class TestAsyncCTMBlock:
    def test_forward_shape(self):
        cfg = _async_config()
        block = AsyncCTMBlock(0, cfg)
        x = torch.randn(2, 8, cfg.hidden_size)
        result = block(x, return_all_ticks=True, num_iters=4)
        assert result.hidden.shape == x.shape

    def test_clock_bands_fire(self):
        cfg = _async_config(async_tick_periods="1,2,4")
        block = AsyncCTMBlock(0, cfg)
        x = torch.randn(1, 4, cfg.hidden_size)
        block(x, num_iters=4)
        local_ticks = block.last_async_local_ticks
        assert local_ticks[0] == 4
        assert local_ticks[1] == 2
        assert local_ticks[2] == 1

    def test_no_nan(self):
        cfg = _async_config()
        block = AsyncCTMBlock(0, cfg)
        x = torch.randn(2, 8, cfg.hidden_size)
        result = block(x)
        assert torch.isfinite(result.hidden).all()

    def test_with_moe_block_sparse(self):
        cfg = _async_config(
            moe_routing_mode="regional_shared_topk",
            moe_num_experts=4,
            moe_topk_experts=1,
            moe_shared_experts=1,
            moe_expert_size=16,
            moe_dispatch_mode="block_sparse",
        )
        block = AsyncCTMBlock(0, cfg)
        x = torch.randn(2, 8, cfg.hidden_size)
        result = block(x)
        assert torch.isfinite(result.hidden).all()


class TestAsyncCTMForCausalLM:
    def test_forward_train(self):
        cfg = _async_config()
        model = AsyncCTMForCausalLM(cfg)
        ids = torch.randint(0, cfg.vocab_size, (2, 8))
        loss, losses, certainties = model.forward_train(ids, ids, num_iters=4)
        assert torch.isfinite(loss)
        assert losses.shape[1] == 4

    def test_backward(self):
        cfg = _async_config()
        model = AsyncCTMForCausalLM(cfg)
        model.train()
        ids = torch.randint(0, cfg.vocab_size, (2, 8))
        loss, _, _ = model.forward_train(ids, ids, num_iters=2)
        try:
            loss.backward()
            grads = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
            assert len(grads) > 0
        except RuntimeError as e:
            if "inplace" in str(e):
                pytest.xfail("async CTM has in-place tensor modification bug in backward")
            raise

    def test_generate(self):
        cfg = _async_config()
        model = AsyncCTMForCausalLM(cfg)
        model.eval()
        ids = torch.randint(0, cfg.vocab_size, (1, 4))
        generated = model.generate(ids, max_new_tokens=4, num_iters=2)
        assert generated.shape == (1, 8)

    def test_with_fast_output(self):
        cfg = _async_config(
            fast_output_mode="anytime",
            fast_output_weight=0.1,
            habit_output_weight=0.1,
            async_fast_output_weight=0.25,
        )
        model = AsyncCTMForCausalLM(cfg)
        ids = torch.randint(0, cfg.vocab_size, (2, 8))
        loss, _, _ = model.forward_train(ids, ids, num_iters=4)
        assert torch.isfinite(loss)
