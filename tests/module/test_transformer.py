import pytest
import torch

from model.config import CTMLLMConfig
from model.model_transformer import TransformerForCausalLM


def _tf_config(**overrides):
    cfg = dict(
        vocab_size=6400,
        hidden_size=128,
        num_hidden_layers=2,
        d_model=64,
        d_input=32,
        iterations=1,
        memory_length=4,
        memory_hidden_dims=4,
        deep_nlms=False,
        heads=4,
        n_synch_out=32,
        n_synch_action=32,
        synapse_depth=1,
        dropout=0.0,
    )
    cfg.update(overrides)
    return CTMLLMConfig(**cfg)


class TestTransformerBaseline:
    def test_forward_shape(self):
        cfg = _tf_config()
        model = TransformerForCausalLM(cfg)
        model.eval()
        ids = torch.randint(0, cfg.vocab_size, (2, 8))
        out = model(ids)
        assert "logits" in out
        assert out["logits"].shape == (2, 8, cfg.vocab_size)

    def test_loss_with_labels(self):
        cfg = _tf_config()
        model = TransformerForCausalLM(cfg)
        ids = torch.randint(0, cfg.vocab_size, (2, 8))
        out = model(ids, labels=ids)
        assert out["loss"] is not None
        assert torch.isfinite(out["loss"])

    def test_generate(self):
        cfg = _tf_config()
        model = TransformerForCausalLM(cfg)
        model.eval()
        ids = torch.randint(0, cfg.vocab_size, (1, 4))
        generated = model.generate(ids, max_new_tokens=8, num_iters=1)
        assert generated.shape[0] == 1
        assert generated.shape[1] >= 4

    def test_generate_with_cache(self):
        cfg = _tf_config()
        model = TransformerForCausalLM(cfg)
        model.eval()
        ids = torch.randint(0, cfg.vocab_size, (1, 4))
        generated = model.generate(ids, max_new_tokens=4, use_cache=True, num_iters=1)
        assert generated.shape == (1, 8)
