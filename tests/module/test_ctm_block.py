import pytest
import torch

from model.config import CTMLLMConfig
from model.model_ctm_llm import CTMBlock


def _block_config(**overrides):
    cfg = dict(
        hidden_size=128,
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
    )
    cfg.update(overrides)
    return CTMLLMConfig(**cfg)


class TestCTMBlockBasic:
    def test_forward_shape(self):
        cfg = _block_config()
        block = CTMBlock(0, cfg)
        x = torch.randn(2, 8, cfg.hidden_size)
        result = block(x)
        assert result.hidden.shape == x.shape

    def test_forward_no_extras_without_flags(self):
        cfg = _block_config()
        block = CTMBlock(0, cfg)
        x = torch.randn(1, 4, cfg.hidden_size)
        result = block(x)
        assert result.hidden is not None
        assert result.present_kv is not None or result.present_kv is None

    def test_return_all_ticks(self):
        cfg = _block_config(iterations=4)
        block = CTMBlock(0, cfg)
        x = torch.randn(1, 4, cfg.hidden_size)
        result = block(x, return_all_ticks=True, num_iters=4)
        assert "tick_outputs" in result.extras
        assert result.extras["tick_outputs"].shape[-1] == 4

    def test_tracking(self):
        cfg = _block_config(iterations=2)
        block = CTMBlock(0, cfg)
        x = torch.randn(1, 4, cfg.hidden_size)
        result = block(x, track=True, num_iters=2)
        assert "tracking" in result.extras
        assert "pre_activations" in result.extras["tracking"]

    def test_no_nan(self):
        cfg = _block_config()
        block = CTMBlock(0, cfg)
        x = torch.randn(2, 8, cfg.hidden_size)
        result = block(x)
        assert torch.isfinite(result.hidden).all()

    def test_kv_cache(self):
        cfg = _block_config()
        block = CTMBlock(0, cfg)
        x = torch.randn(1, 8, cfg.hidden_size)
        result1 = block(x, use_cache=True)
        present = result1.present_kv
        x2 = torch.randn(1, 1, cfg.hidden_size)
        result2 = block(x2, past_kv=present, use_cache=True)
        assert result2.hidden.shape == (1, 1, cfg.hidden_size)

    def test_self_cond(self):
        cfg = _block_config(self_cond=True)
        block = CTMBlock(0, cfg)
        x = torch.randn(1, 4, cfg.hidden_size)
        result = block(x)
        assert result.hidden.shape == x.shape

    def test_cross_layer_state(self):
        cfg = _block_config(cross_layer_state=True, iterations=2)
        block = CTMBlock(0, cfg)
        x = torch.randn(1, 4, cfg.hidden_size)
        result = block(x, return_all_ticks=True)
        assert "final_activated" in result.extras
        assert "final_trace" in result.extras

    def test_executed_ticks(self):
        cfg = _block_config(iterations=3)
        block = CTMBlock(0, cfg)
        x = torch.randn(1, 4, cfg.hidden_size)
        block(x, num_iters=3)
        assert block.last_executed_ticks == 3

    def test_num_iters_override(self):
        cfg = _block_config(iterations=8)
        block = CTMBlock(0, cfg)
        x = torch.randn(1, 4, cfg.hidden_size)
        result = block(x, return_all_ticks=True, num_iters=2)
        assert result.extras["tick_outputs"].shape[-1] == 2


class TestCTMBlockMoE:
    @pytest.mark.parametrize("routing", ["topk", "hash", "expert_choice"])
    def test_moe_routing_modes(self, routing):
        cfg = _block_config(
            moe_routing_mode=routing,
            moe_num_experts=4,
            moe_topk_experts=2,
            moe_expert_size=16,
            moe_dispatch_mode="dense_mask",
        )
        block = CTMBlock(0, cfg)
        x = torch.randn(2, 8, cfg.hidden_size)
        result = block(x)
        assert torch.isfinite(result.hidden).all()

    def test_regional_topk(self):
        cfg = _block_config(
            moe_routing_mode="regional_topk",
            moe_num_experts=4,
            moe_topk_experts=2,
            moe_expert_size=16,
            moe_dispatch_mode="dense_mask",
        )
        block = CTMBlock(0, cfg)
        x = torch.randn(2, 8, cfg.hidden_size)
        result = block(x)
        assert torch.isfinite(result.hidden).all()

    def test_regional_shared_topk(self):
        cfg = _block_config(
            moe_routing_mode="regional_shared_topk",
            moe_num_experts=4,
            moe_topk_experts=1,
            moe_shared_experts=1,
            moe_expert_size=16,
            moe_dispatch_mode="dense_mask",
        )
        block = CTMBlock(0, cfg)
        x = torch.randn(2, 8, cfg.hidden_size)
        result = block(x)
        assert torch.isfinite(result.hidden).all()

    def test_block_sparse_dispatch(self):
        cfg = _block_config(
            moe_routing_mode="regional_shared_topk",
            moe_num_experts=4,
            moe_topk_experts=1,
            moe_shared_experts=1,
            moe_expert_size=16,
            moe_dispatch_mode="block_sparse",
        )
        block = CTMBlock(0, cfg)
        x = torch.randn(2, 8, cfg.hidden_size)
        result = block(x)
        assert torch.isfinite(result.hidden).all()

    def test_activation_passes(self):
        cfg = _block_config(
            moe_routing_mode="regional_shared_topk",
            moe_num_experts=4,
            moe_topk_experts=1,
            moe_expert_size=16,
            moe_activation_passes=2,
            moe_dispatch_mode="dense_mask",
        )
        block = CTMBlock(0, cfg)
        x = torch.randn(2, 8, cfg.hidden_size)
        result = block(x)
        assert torch.isfinite(result.hidden).all()

    def test_moe_aux_loss_accumulated(self):
        cfg = _block_config(
            moe_routing_mode="regional_shared_topk",
            moe_num_experts=4,
            moe_topk_experts=1,
            moe_shared_experts=1,
            moe_expert_size=16,
            moe_load_balance_weight=0.1,
            moe_dispatch_mode="dense_mask",
            cell_sparsity_mode="topk",
            cell_topk=32,
        )
        block = CTMBlock(0, cfg)
        x = torch.randn(2, 8, cfg.hidden_size)
        block(x)
        assert block.moe_aux_loss is not None


class TestCTMBlockSparsity:
    def test_topk_sparsity(self):
        cfg = _block_config(
            cell_sparsity_mode="topk",
            cell_topk=16,
        )
        block = CTMBlock(0, cfg)
        x = torch.randn(2, 8, cfg.hidden_size)
        result = block(x)
        assert torch.isfinite(result.hidden).all()

    def test_no_sparsity(self):
        cfg = _block_config(cell_sparsity_mode="none")
        block = CTMBlock(0, cfg)
        x = torch.randn(2, 8, cfg.hidden_size)
        result = block(x)
        assert torch.isfinite(result.hidden).all()
