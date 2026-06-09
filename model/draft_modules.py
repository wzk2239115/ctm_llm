import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import CTMLLMConfig


class DraftSlotHead(nn.Module):
    """Parallel draft slots within a single tick (future token offsets 1..block_size)."""

    def __init__(self, config: CTMLLMConfig, hidden_size: int, vocab_size: int):
        super().__init__()
        self.block_size = max(1, int(config.draft_block_size))
        self.head_mode = config.draft_head_mode
        self.attention_mode = config.draft_slot_attention
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.slot_norm = nn.LayerNorm(hidden_size)
        self.slot_in = nn.Linear(hidden_size, hidden_size * self.block_size, bias=False)

        if self.head_mode == 'slot_adapter':
            mid = max(1, hidden_size // 2)
            self.slot_adapters = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_size, mid, bias=False),
                    nn.SiLU(),
                    nn.Linear(mid, hidden_size, bias=False),
                )
                for _ in range(self.block_size)
            ])
        elif self.head_mode == 'slot_head':
            self.slot_lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def _slot_attention(self, slots):
        B, T, S, H = slots.shape
        x = slots.reshape(B * T, S, H)
        q = x.unsqueeze(1)
        k = x.unsqueeze(1)
        v = x.unsqueeze(1)
        if self.attention_mode == 'block_bidir':
            attn_out = F.scaled_dot_product_attention(q, k, v).squeeze(1)
        else:
            mask = torch.triu(
                torch.full((S, S), float('-inf'), device=x.device, dtype=x.dtype),
                diagonal=1,
            )
            attn_out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask,
            ).squeeze(1)
        return attn_out.view(B, T, S, H)

    def forward(self, tick_hidden, lm_head=None):
        B, T, H = tick_hidden.shape
        slots = self.slot_in(self.slot_norm(tick_hidden)).view(B, T, self.block_size, H)
        slot_hidden = self._slot_attention(slots)

        if self.head_mode == 'slot_adapter':
            adapted = [
                self.slot_adapters[s](slot_hidden[:, :, s])
                for s in range(self.block_size)
            ]
            slot_hidden = torch.stack(adapted, dim=2)

        if self.head_mode == 'slot_head':
            slot_logits = self.slot_lm_head(slot_hidden)
        else:
            if lm_head is None:
                raise ValueError("shared/slot_adapter draft head requires lm_head")
            slot_logits = lm_head(slot_hidden)

        return slot_hidden, slot_logits
