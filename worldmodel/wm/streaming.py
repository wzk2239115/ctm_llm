"""Streaming CTM — a continuous ingest / think / emit latent-dynamics module.

This is the closed-loop form of CTM: instead of freezing the input and
deliberating for K internal ticks, the network maintains a *persistent* thought
state ``(activated_state, trace)`` and, at every tick, simultaneously

  1. **ingests feedback**  — the current action (the control / feedback signal),
  2. **thinks**            — synapse update + Neuron-Level Model over the trace,
  3. **emits**             — a latent (the predicted next state representation).

There is no fixed ``iterations`` and no per-sample reset: the recurrence is
unbounded and carries across the whole horizon / episode. Used here as the
*dynamics predictor* of a JEPA world model, it replaces the Markov
:class:`MLPPredictor` with a recurrent, always-thinking variant — isolating the
streaming-recurrence contribution while the encoder / solver / data stay fixed.

What is retained from native CTM: the Neuron-Level Model (per-neuron private
MLP over a pre-activation trace, via :class:`SuperLinear`) and an internal
``d_model`` thought space decoupled from the representation. What is dropped:
the frozen-input attention, the per-sample start state, and the fixed tick
budget. The synchronisation readout is a learnable linear head (a simplified
pairwise-sync) so the gradient is a short per-tick loop — which, unlike the
deep frozen-input encoder, does not collapse.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from baseline.models.modules import SuperLinear, Squeeze


class _NLM(nn.Module):
    """Per-neuron private MLP over the pre-activation trace (CTM trace_processor)."""

    def __init__(self, d_model: int, memory_length: int, hidden: int, do_norm: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            SuperLinear(in_dims=memory_length, out_dims=2 * hidden, N=d_model, do_norm=do_norm),
            nn.GLU(dim=-1),
            SuperLinear(in_dims=hidden, out_dims=2, N=d_model, do_norm=do_norm),
            nn.GLU(dim=-1),
            Squeeze(-1),
        )

    def forward(self, trace: torch.Tensor) -> torch.Tensor:
        # trace: (B, d_model, memory_length) -> activated_state (B, d_model)
        return self.net(trace)


class StreamingCTMPredictor(nn.Module):
    """Always-on recurrent latent-dynamics predictor.

    Args:
        latent_dim: representation dim in/out (matches the encoder).
        action_dim: control/feedback dim ingested every tick.
        d_model: internal thought space (CTM neuron count).
        memory_length: NLM trace history length.
        nlm_hidden: hidden width of the per-neuron MLPs.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        d_model: int = 128,
        memory_length: int = 8,
        nlm_hidden: int = 8,
        act_emb: int = 16,
        state_gate: str = "none",
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.d_model = int(d_model)
        self.memory_length = int(memory_length)
        self.state_gate = str(state_gate)

        self.in_proj = nn.Linear(latent_dim, d_model)          # seed thought from obs latent
        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, act_emb), nn.GELU(), nn.Linear(act_emb, act_emb),
        )
        self.synapse = nn.Sequential(                          # mix feedback + state -> pre-activation
            nn.Linear(act_emb + d_model, d_model * 2), nn.GLU(), nn.LayerNorm(d_model),
        )
        self.trace_processor = _NLM(d_model, memory_length, nlm_hidden)
        self.emit = nn.Sequential(                             # synchronisation readout -> latent
            nn.LayerNorm(d_model), nn.Linear(d_model, latent_dim),
        )
        # GRU-style gating lets `activated` accumulate across ticks instead of being
        # overwritten each step (state_gate='none' = legacy overwrite; 'gru' = gated residual).
        if self.state_gate == "gru":
            self.gate_linear = nn.Linear(d_model, d_model)

        # Persistent state (reset per rollout).
        self._trace: torch.Tensor | None = None
        self._activated: torch.Tensor | None = None

    def reset_state(self, init_latent: torch.Tensor) -> None:
        """Seed thought state from the initial observation latent."""
        self._activated = self.in_proj(init_latent)            # (..., d_model)
        shape = (*self._activated.shape[:-1], self.d_model, self.memory_length)
        self._trace = self._activated.unsqueeze(-1).expand(*shape).clone()

    def step(self, action: torch.Tensor) -> torch.Tensor:
        """One streaming tick: ingest action -> think -> emit latent."""
        lead_shape = self._activated.shape[:-1]  # arbitrary leading dims (..., )
        a = self.action_embed(action.float())                  # (..., act_emb)
        inp = torch.cat([a, self._activated], dim=-1)          # (..., act_emb + d_model)
        new = self.synapse(inp).unsqueeze(-1)                  # (..., d_model, 1)
        # shift the trace and append the newest pre-activation
        self._trace = torch.cat([self._trace[..., 1:], new], dim=-1)
        # SuperLinear operates on exactly 3D (B, d_model, mem_len); flatten leading dims.
        trace_flat = self._trace.reshape(-1, self.d_model, self.memory_length)
        act_flat = self.trace_processor(trace_flat)            # (B', d_model)
        if self.state_gate == "gru":
            # gated residual: keep part of the old thought so state accumulates across ticks
            old_flat = self._activated.reshape(-1, self.d_model)
            gate = torch.sigmoid(self.gate_linear(old_flat))   # (B', d_model)
            merged = gate * act_flat + (1.0 - gate) * old_flat
            self._activated = merged.reshape(*lead_shape, self.d_model)
        else:
            self._activated = act_flat.reshape(*lead_shape, self.d_model)
        return self.emit(self._activated)                      # (..., latent_dim)

    def rollout(self, init_latent: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Roll the persistent state over ``actions`` ``(..., H, A)``.

        Returns predicted latents ``(..., H, latent_dim)``. The state is carried
        across all H ticks (true streaming recurrence), not recomputed per step.
        """
        self.reset_state(init_latent)
        preds = []
        H = actions.shape[-2]
        for t in range(H):
            preds.append(self.step(actions[..., t, :]))
        return torch.stack(preds, dim=-2)

    # Stateless single-step form, for parity with MLPPredictor if ever needed.
    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        self.reset_state(z)
        return self.step(action)
