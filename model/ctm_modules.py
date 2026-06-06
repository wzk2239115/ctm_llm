import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Squeeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)


class TTTMLP(nn.Module):
    """
    Test-time learner used inside CTM ticks.

    This module is a small real-parameter model that transforms the CTM
    activated state. During normal training it behaves like a zero-init residual
    adapter; during test-time training its parameters can be updated with a
    self-supervised prefix loss, making the CTM hidden computation adaptive.
    """

    def __init__(self, d_model, hidden_mult=2, gate_init=-2.0, dropout=0.0):
        super().__init__()
        hidden = max(d_model, int(d_model * hidden_mult))
        self.norm = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        nn.init.zeros_(self.up.weight)

    def forward(self, activated):
        delta = self.up(self.dropout(F.silu(self.down(self.norm(activated)))))
        return torch.sigmoid(self.gate) * delta


class SuperLinear(nn.Module):
    """
    Neuron-Level Model (NLM) for CTM-LLM.
    Extends the original CTM's SuperLinear to support sequence input (B, T, D, M).

    Each of the N neurons has its own private linear transformation,
    applied independently via einsum over the memory/history dimension.
    """

    def __init__(self, in_dims, out_dims, N, T=1.0, dropout=0.0):
        super().__init__()
        self.in_dims = in_dims
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.register_parameter('w1', nn.Parameter(
            torch.empty((in_dims, out_dims, N)).uniform_(
                -1 / math.sqrt(in_dims + out_dims),
                1 / math.sqrt(in_dims + out_dims)
            ), requires_grad=True))
        self.register_parameter('b1', nn.Parameter(
            torch.zeros((1, 1, N, out_dims)), requires_grad=True))
        self.register_parameter('T_param', nn.Parameter(torch.Tensor([T])))

    def forward(self, x):
        # x: (B, T_seq, N, in_dims)
        out = self.dropout(x)
        out = torch.einsum('btdm,mhd->btdh', out, self.w1) + self.b1
        return out.squeeze(-1) / self.T_param


class SynapseUNET(nn.Module):
    """
    U-Net style synapse model for CTM-LLM, adapted for sequence input (B, T, in_dims).
    Down/up blocks with skip connections enable multi-level information mixing
    across the neuron population.
    """

    def __init__(self, in_dims, out_dims, depth, minimum_width=16, dropout=0.0):
        super().__init__()
        widths = np.linspace(out_dims, minimum_width, depth)

        self.first_projection = nn.Sequential(
            nn.Linear(in_dims, int(widths[0])),
            nn.LayerNorm(int(widths[0])),
            nn.SiLU()
        )

        self.down_projections = nn.ModuleList()
        self.up_projections = nn.ModuleList()
        self.skip_lns = nn.ModuleList()

        for i in range(len(widths) - 1):
            self.down_projections.append(nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(int(widths[i]), int(widths[i + 1])),
                nn.LayerNorm(int(widths[i + 1])),
                nn.SiLU()
            ))
            self.up_projections.append(nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(int(widths[i + 1]), int(widths[i])),
                nn.LayerNorm(int(widths[i])),
                nn.SiLU()
            ))
            self.skip_lns.append(nn.LayerNorm(int(widths[i])))

    def forward(self, x):
        out_first = self.first_projection(x)
        outs_down = [out_first]
        for layer in self.down_projections:
            outs_down.append(layer(outs_down[-1]))

        outs_up = outs_down[-1]
        num_blocks = len(self.up_projections)
        for i in range(num_blocks):
            up_idx = num_blocks - 1 - i
            out_up = self.up_projections[up_idx](outs_up)
            skip = outs_down[up_idx]
            outs_up = self.skip_lns[up_idx](out_up + skip)
        return outs_up
