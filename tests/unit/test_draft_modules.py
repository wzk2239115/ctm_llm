import pytest
import torch

from model.config import CTMLLMConfig
from model.draft_modules import DraftSlotHead


def _cfg(**overrides):
    defaults = dict(
        hidden_size=128,
        d_model=64,
        d_input=32,
        vocab_size=6400,
        iterations=2,
        memory_length=4,
        memory_hidden_dims=4,
        deep_nlms=False,
        heads=4,
        n_synch_out=32,
        n_synch_action=32,
        synapse_depth=1,
        self_cond=False,
        dropout=0.0,
        draft_block_size=4,
        draft_head_mode="shared",
        draft_slot_attention="causal_slots",
    )
    defaults.update(overrides)
    return CTMLLMConfig(**defaults)


class TestDraftSlotHead:
    def test_shared_mode_shape(self):
        cfg = _cfg(draft_head_mode="shared")
        head = DraftSlotHead(cfg, hidden_size=128, vocab_size=6400)
        tick_out = torch.randn(2, 8, 128)
        lm_head = torch.nn.Linear(128, 6400, bias=False)
        slot_hidden, slot_logits = head(tick_out, lm_head=lm_head)
        assert slot_logits.shape == (2, 8, 4, 6400)

    def test_slot_head_mode_shape(self):
        cfg = _cfg(draft_head_mode="slot_head")
        head = DraftSlotHead(cfg, hidden_size=128, vocab_size=6400)
        tick_out = torch.randn(2, 8, 128)
        lm_head = torch.nn.Linear(128, 6400, bias=False)
        slot_hidden, slot_logits = head(tick_out, lm_head=lm_head)
        assert slot_logits.shape == (2, 8, 4, 6400)

    def test_slot_adapter_mode_shape(self):
        cfg = _cfg(draft_head_mode="slot_adapter")
        head = DraftSlotHead(cfg, hidden_size=128, vocab_size=6400)
        tick_out = torch.randn(2, 8, 128)
        lm_head = torch.nn.Linear(128, 6400, bias=False)
        slot_hidden, slot_logits = head(tick_out, lm_head=lm_head)
        assert slot_logits.shape == (2, 8, 4, 6400)

    def test_slot_hidden_shape(self):
        cfg = _cfg(draft_head_mode="shared")
        head = DraftSlotHead(cfg, hidden_size=128, vocab_size=6400)
        tick_out = torch.randn(2, 8, 128)
        lm_head = torch.nn.Linear(128, 6400, bias=False)
        slot_hidden, _ = head(tick_out, lm_head=lm_head)
        assert slot_hidden.shape == (2, 8, 4, 128)

    def test_gradient_flows(self):
        cfg = _cfg(draft_head_mode="shared")
        head = DraftSlotHead(cfg, hidden_size=128, vocab_size=6400)
        tick_out = torch.randn(2, 8, 128, requires_grad=True)
        lm_head = torch.nn.Linear(128, 6400, bias=False)
        slot_hidden, slot_logits = head(tick_out, lm_head=lm_head)
        slot_logits.sum().backward()
        assert tick_out.grad is not None

    def test_block_size_1(self):
        cfg = _cfg(draft_block_size=1, draft_head_mode="shared")
        head = DraftSlotHead(cfg, hidden_size=128, vocab_size=6400)
        tick_out = torch.randn(2, 8, 128)
        lm_head = torch.nn.Linear(128, 6400, bias=False)
        _, slot_logits = head(tick_out, lm_head=lm_head)
        assert slot_logits.shape == (2, 8, 1, 6400)

    def test_block_bidir_attention(self):
        cfg = _cfg(draft_slot_attention="block_bidir")
        head = DraftSlotHead(cfg, hidden_size=128, vocab_size=6400)
        tick_out = torch.randn(2, 8, 128)
        lm_head = torch.nn.Linear(128, 6400, bias=False)
        _, slot_logits = head(tick_out, lm_head=lm_head)
        assert torch.isfinite(slot_logits).all()
