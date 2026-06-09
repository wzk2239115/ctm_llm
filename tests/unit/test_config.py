import pytest
import torch

from model.config import CTMLLMConfig


class TestCTMLLMConfig:
    def test_defaults(self):
        cfg = CTMLLMConfig()
        assert cfg.vocab_size == 6400
        assert cfg.hidden_size == 768
        assert cfg.d_model == 512
        assert cfg.tick_loss_mode == "min_conf"

    def test_override(self):
        cfg = CTMLLMConfig(
            hidden_size=256, d_model=64, d_input=32,
            n_synch_out=32, n_synch_action=32, iterations=5
        )
        assert cfg.hidden_size == 256
        assert cfg.d_model == 64
        assert cfg.iterations == 5

    def test_d_model_ge_synch(self):
        with pytest.raises(AssertionError, match="d_model"):
            CTMLLMConfig(d_model=4, n_synch_out=64, n_synch_action=64)

    def test_d_input_divisible_by_heads(self):
        with pytest.raises(AssertionError, match="d_input"):
            CTMLLMConfig(d_input=31, heads=8)

    def test_repr_contains_keys(self):
        cfg = CTMLLMConfig(hidden_size=99, d_model=64, n_synch_out=32, n_synch_action=32)
        r = repr(cfg)
        assert "hidden_size" in r
        assert "99" in r

    def test_cell_topk_defaults_to_d_model(self):
        cfg = CTMLLMConfig(d_model=64, n_synch_out=32, n_synch_action=32)
        assert cfg.cell_topk == 64

    def test_moe_defaults(self):
        cfg = CTMLLMConfig()
        assert cfg.moe_routing_mode == "none"
        assert cfg.moe_num_experts == 1

    def test_all_moe_fields_present(self):
        cfg = CTMLLMConfig()
        for attr in ["moe_routing_mode", "moe_num_experts", "moe_topk_experts",
                      "moe_shared_experts", "moe_expert_size", "moe_load_balance_weight",
                      "moe_dispatch_mode", "moe_activation_passes", "moe_region_diversity_weight"]:
            assert hasattr(cfg, attr)

    def test_async_tick_defaults(self):
        cfg = CTMLLMConfig()
        assert cfg.async_tick_mode == "none"
        assert cfg.async_tick_periods == "1,2,4,8"

    def test_dino_defaults_off(self):
        cfg = CTMLLMConfig()
        assert cfg.dino_self_supervised_weight == 0.0

    def test_context_reading_defaults_off(self):
        cfg = CTMLLMConfig()
        assert cfg.context_reading_mode == "none"
