import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


@dataclass
class BlockOutput:
    hidden: torch.Tensor
    present_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelOutput:
    hidden: torch.Tensor
    past_key_values: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = field(default_factory=list)
    tick_outputs: Optional[torch.Tensor] = None
    draft_slot_logits: Optional[torch.Tensor] = None
    tracking: Optional[Dict[str, Any]] = None
    executed_ticks: Optional[List[int]] = None


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        normed = x.float() * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * normed).type_as(x)


class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_ratio=4):
        super().__init__()
        intermediate = math.ceil(hidden_size * intermediate_ratio / 64) * 64
        self.gate_proj = nn.Linear(hidden_size, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def _parse_int_list(raw, *, max_value=None):
    values = []
    for item in str(raw or "").replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError:
            continue
        if value <= 0:
            continue
        if max_value is not None:
            value = min(value, max_value)
        if value not in values:
            values.append(value)
    return sorted(values)


def _parse_name_list(raw):
    values = []
    for item in str(raw or "").replace(";", ",").split(","):
        item = item.strip().lower()
        if item and item not in values:
            values.append(item)
    return values
