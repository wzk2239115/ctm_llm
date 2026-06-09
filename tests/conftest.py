import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.config import CTMLLMConfig


def _base_config(**overrides):
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


def _moe_config(**overrides):
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
        moe_routing_mode="regional_shared_topk",
        moe_num_experts=4,
        moe_topk_experts=1,
        moe_shared_experts=1,
        moe_expert_size=16,
        moe_activation_passes=1,
        moe_dispatch_mode="dense_mask",
    )
    cfg.update(overrides)
    return CTMLLMConfig(**cfg)


@pytest.fixture
def base_config():
    return _base_config


@pytest.fixture
def moe_config():
    return _moe_config


@pytest.fixture
def tiny_ids():
    return torch.randint(0, 6400, (2, 8))
