import pytest
import torch
import os
import math
import tempfile

from model.config import CTMLLMConfig
from model.model_ctm_llm import CTMForCausalLM
from trainer.trainer_utils import save_checkpoint, load_checkpoint, get_lr


def _tiny_config():
    return CTMLLMConfig(
        vocab_size=6400,
        hidden_size=64,
        num_hidden_layers=1,
        d_model=32,
        d_input=16,
        iterations=1,
        memory_length=2,
        memory_hidden_dims=4,
        deep_nlms=False,
        heads=4,
        n_synch_out=16,
        n_synch_action=16,
        synapse_depth=1,
        self_cond=False,
        dropout=0.0,
    )


class TestCheckpoint:
    def test_save_load_roundtrip(self):
        cfg = _tiny_config()
        model = CTMForCausalLM(cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_ckpt.pth")
            save_checkpoint(model, optimizer, epoch=3, step=50, save_path=path)
            loaded_epoch, loaded_step = load_checkpoint(
                path, model, optimizer, device="cpu")
            assert loaded_epoch == 3
            assert loaded_step == 50

    def test_half_save_preserves_dtype(self):
        cfg = _tiny_config()
        model = CTMForCausalLM(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "half_model.pth")
            torch.save(
                {k: v.half().cpu() for k, v in model.state_dict().items()}, path)
            state = torch.load(path, map_location="cpu", weights_only=False)
            for v in state.values():
                assert v.dtype == torch.float16

    def test_load_reproduces_forward(self):
        cfg = _tiny_config()
        model = CTMForCausalLM(cfg)
        model.eval()
        ids = torch.randint(0, cfg.vocab_size, (1, 4))
        with torch.no_grad():
            orig_logits = model(ids)["logits"].clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_ckpt.pth")
            save_checkpoint(model, optimizer, epoch=0, step=0, save_path=path)
            model2 = CTMForCausalLM(cfg)
            load_checkpoint(path, model2, device="cpu")
            model2.eval()
            loaded_logits = model2(ids)["logits"].float()
            assert torch.allclose(orig_logits.float(), loaded_logits, atol=0.05)


class TestLR:
    def test_cosine_schedule_start(self):
        lr = get_lr(0, 100, 1e-3)
        assert abs(lr - 1e-3) < 1e-5

    def test_schedule_monotonically_decreases(self):
        lrs = [get_lr(s, 100, 1e-3) for s in range(0, 100, 10)]
        for i in range(len(lrs) - 1):
            assert lrs[i + 1] <= lrs[i] + 1e-8

    def test_schedule_bounds(self):
        for step in [0, 25, 50, 75, 100]:
            lr = get_lr(step, 100, 1e-3)
            assert lr > 0
            assert lr <= 1e-3 + 1e-6


class TestModelFactory:
    def test_create_ctm(self):
        from trainer.trainer_utils import create_model
        cfg = _tiny_config()
        model = create_model(cfg, device="cpu")
        assert isinstance(model, CTMForCausalLM)

    def test_create_transformer(self):
        from trainer.trainer_utils import create_model
        from model.model_transformer import TransformerForCausalLM
        cfg = CTMLLMConfig(
            model_type="transformer",
            vocab_size=6400,
            hidden_size=64,
            num_hidden_layers=1,
            d_model=32,
            d_input=16,
            iterations=1,
            memory_length=2,
            memory_hidden_dims=4,
            deep_nlms=False,
            heads=4,
            n_synch_out=16,
            n_synch_action=16,
            synapse_depth=1,
            dropout=0.0,
        )
        model = create_model(cfg, device="cpu")
        assert isinstance(model, TransformerForCausalLM)
