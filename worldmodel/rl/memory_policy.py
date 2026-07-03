"""Memory-policy backbones for the POMDP ablation (Route: JEPA encoder + memory policy).

All backbones share one interface so PPO trains them identically and only the
memory mechanism differs:
  - state is a tuple of tensors; snapshot()/recompute() carry it for recurrent PPO
  - step_stateless(x, state) -> (feat, new_state) is a pure fn
  - mask_reset zeroes the state where an episode ended (done)

Backbones:
  mlp          no memory (Markov baseline)
  ctm          CTM persistent NLM state (activated + pre-activation trace)
  lstm / gru   standard recurrent baselines (the comparison CTM must beat)
  transformer  causal self-attention over a latent history buffer

The encoder (obs+goal -> latent) is a learnable part of the policy (end-to-end
fine-tuned), NOT a frozen JEPA encoder — the pretrain experiment showed frozen
dynamics representations hurt the policy. A pretrained init can be plugged in
later via load_state_dict.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Normal

from worldmodel.wm.streaming import _NLM


class MemoryBackbone(nn.Module):
    """Interface. Subclass implements zero_state / mask_state / step_stateless."""
    feat_dim: int

    def zero_state(self, batch, device):
        raise NotImplementedError

    def mask_state(self, state, done_mask):
        raise NotImplementedError

    def step_stateless(self, x, state):
        raise NotImplementedError

    def init_state(self, batch, device):
        self._state = self.zero_state(batch, device)

    def detach_state(self):
        self._state = tuple(s.detach() for s in self._state)

    def snapshot(self):
        return tuple(s.detach() for s in self._state)

    def mask_reset(self, done_mask):
        self._state = self.mask_state(self._state, done_mask)

    def step(self, x):
        feat, ns = self.step_stateless(x, self._state)
        self._state = ns
        return feat


def _mask_parts(state, done_mask):
    """Zero each tensor in state where done_mask is True (per-env reset)."""
    m = done_mask
    out = []
    for s in state:
        # broadcast mask over leading batch dim
        view = [1] * s.dim()
        view[0] = -1
        out.append(s * (1.0 - m.to(s).view(view)))
    return tuple(out)


class MLPMemory(MemoryBackbone):
    """No memory — Markov baseline. Empty state tuple."""

    def __init__(self, latent_dim, d_model=128):
        super().__init__()
        self.feat_dim = d_model
        self.net = nn.Sequential(nn.Linear(latent_dim, d_model), nn.GELU(),
                                 nn.Linear(d_model, d_model), nn.GELU())

    def zero_state(self, batch, device):
        return ()

    def mask_state(self, state, done_mask):
        return ()

    def step_stateless(self, x, state):
        return self.net(x), ()


class CTMMemory(MemoryBackbone):
    """CTM persistent NLM: (activated, trace) carry across steps."""

    def __init__(self, latent_dim, d_model=128, memory_length=8, nlm_hidden=8,
                 state_gate="gru"):
        super().__init__()
        self.d_model = d_model
        self.memory_length = memory_length
        self.state_gate = state_gate
        self.feat_dim = d_model
        self.in_proj = nn.Linear(latent_dim, d_model)
        self.synapse = nn.Sequential(nn.Linear(2 * d_model, 2 * d_model),
                                     nn.GLU(), nn.LayerNorm(d_model))
        self.trace_processor = _NLM(d_model, memory_length, nlm_hidden)
        self.feat = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU())
        if state_gate == "gru":
            self.gate_linear = nn.Linear(d_model, d_model)

    def zero_state(self, batch, device):
        z = torch.zeros(batch, self.d_model, device=device)
        tr = z.unsqueeze(-1).expand(-1, self.d_model, self.memory_length).clone()
        return (z, tr)

    def mask_state(self, state, done_mask):
        return _mask_parts(state, done_mask)

    def step_stateless(self, z, state):
        activated, trace = state
        lead = activated.shape[:-1]
        e = self.in_proj(z)
        inp = torch.cat([e, activated], dim=-1)
        new = self.synapse(inp).unsqueeze(-1)
        trace2 = torch.cat([trace[..., 1:], new], dim=-1)
        act_flat = self.trace_processor(trace2.reshape(-1, self.d_model, self.memory_length))
        if self.state_gate == "gru":
            old = activated.reshape(-1, self.d_model)
            g = torch.sigmoid(self.gate_linear(old))
            new_act = (g * act_flat + (1 - g) * old).reshape(*lead, self.d_model)
        else:
            new_act = act_flat.reshape(*lead, self.d_model)
        return self.feat(new_act), (new_act, trace2)


class LSTMMemory(MemoryBackbone):
    def __init__(self, latent_dim, d_model=128):
        super().__init__()
        self.d_model = d_model
        self.feat_dim = d_model
        self.lstm = nn.LSTM(latent_dim, d_model, batch_first=True)
        self.feat = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())

    def zero_state(self, batch, device):
        z = torch.zeros(batch, self.d_model, device=device)
        return (z, z)

    def mask_state(self, state, done_mask):
        return _mask_parts(state, done_mask)

    def step_stateless(self, z, state):
        h, c = state
        out, (h2, c2) = self.lstm(z.unsqueeze(1), (h.unsqueeze(0), c.unsqueeze(0)))
        feat = self.feat(out.squeeze(1))
        return feat, (h2.squeeze(0), c2.squeeze(0))


class GRUMemory(MemoryBackbone):
    def __init__(self, latent_dim, d_model=128):
        super().__init__()
        self.d_model = d_model
        self.feat_dim = d_model
        self.gru = nn.GRU(latent_dim, d_model, batch_first=True)
        self.feat = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())

    def zero_state(self, batch, device):
        return (torch.zeros(batch, self.d_model, device=device),)

    def mask_state(self, state, done_mask):
        return _mask_parts(state, done_mask)

    def step_stateless(self, z, state):
        (h,) = state
        out, h2 = self.gru(z.unsqueeze(1), h.unsqueeze(0))  # nn.GRU returns (output, h_n tensor) — NOT a tuple like LSTM
        feat = self.feat(out.squeeze(1))
        return feat, (h2.squeeze(0),)


class TransformerMemory(MemoryBackbone):
    """Causal self-attention over a rolling latent history buffer."""

    def __init__(self, latent_dim, d_model=128, nhead=4, layers=2, max_hist=32):
        super().__init__()
        self.d_model = d_model
        self.feat_dim = d_model
        self.max_hist = max_hist
        self.in_proj = nn.Linear(latent_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.feat = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())
        self.pos = nn.Parameter(torch.zeros(1, max_hist + 1, d_model))

    def zero_state(self, batch, device):
        return (torch.zeros(batch, self.max_hist, self.d_model, device=device),)

    def mask_state(self, state, done_mask):
        m = done_mask.to(state[0]).view(-1, 1, 1)
        return (state[0] * (1.0 - m),)

    def step_stateless(self, z, state):
        (hist,) = state
        e = self.in_proj(z).unsqueeze(1)
        hist2 = torch.cat([hist, e], dim=1)[:, -self.max_hist:, :]
        T = hist2.shape[1]
        seq = hist2 + self.pos[:, -T:]
        mask = torch.triu(torch.full((T, T), float("-inf"), device=hist2.device), diagonal=1)
        out = self.encoder(seq, mask=mask)
        feat = self.feat(out[:, -1, :])
        return feat, (hist2,)


class GRUGate(nn.Module):
    """HRM-style gate (from baseline/utils/hrm_ideas.py): a 'deep' signal modulates a
    'shallow' state non-blockingly. bias init -2 => gate starts ~0.12, so shallow
    dominates first and deep enters gradually — exactly the multi-timescale behavior."""

    def __init__(self, d_model, d_input):
        super().__init__()
        self.gate_proj = nn.Linear(d_model + d_input, d_model)
        self.value_proj = nn.Linear(d_input, d_model)
        nn.init.zeros_(self.gate_proj.weight)
        self.gate_proj.bias.data.fill_(-2.0)

    def forward(self, deep, shallow, return_gate=False):
        z = torch.sigmoid(self.gate_proj(torch.cat([deep, shallow], dim=-1)))
        candidate = torch.relu(self.value_proj(deep))
        out = (1.0 - z) * shallow + z * candidate
        return (out, z) if return_gate else out


class FlashBrainBackbone(MemoryBackbone):
    """Multi-timescale 'Flash Brain': a shallow-fast path + a deep-slow CTM,
    fused by a GRUGate so deep thought never blocks shallow reflex.

      shallow (Linear, every step, low-latency)  -> fast feat
      deep    (CTM memory, accumulates belief)    -> deep feat
      gate    (GRUGate): feat = gate(deep, shallow)  # deep modulates shallow

    JEPA/encoder feeds z_t (perception); CTM is the memory-control component.
    """

    def __init__(self, latent_dim, d_model=128, memory_length=8, nlm_hidden=8,
                 state_gate="gru", gate_mode="learn"):
        super().__init__()
        self.feat_dim = d_model
        self.gate_mode = gate_mode  # learn | shallow (z=0) | deep (z=1)
        self.shallow = nn.Sequential(nn.Linear(latent_dim, d_model), nn.GELU())
        self.deep = CTMMemory(latent_dim, d_model, memory_length, nlm_hidden,
                              state_gate=state_gate)
        self.gate = GRUGate(d_model, d_model) if gate_mode == "learn" else None
        self.last_gate_z = 0.0  # diagnostics: mean gate opening (0=shallow, 1=deep)
        self.last_shallow_norm = 0.0  # diagnostics: shallow path output norm
        self.last_deep_norm = 0.0     # diagnostics: deep path output norm

    def zero_state(self, batch, device):
        return self.deep.zero_state(batch, device)

    def mask_state(self, state, done_mask):
        return self.deep.mask_state(state, done_mask)

    def step_stateless(self, z, state):
        shallow_feat = self.shallow(z)
        deep_feat, new_state = self.deep.step_stateless(z, state)
        self.last_shallow_norm = float(shallow_feat.norm(dim=-1).mean().detach().item())
        self.last_deep_norm = float(deep_feat.norm(dim=-1).mean().detach().item())
        if self.gate_mode == "shallow":
            self.last_gate_z = 0.0
            return shallow_feat, new_state
        if self.gate_mode == "deep":
            self.last_gate_z = 1.0
            return deep_feat, new_state
        feat, gz = self.gate(deep_feat, shallow_feat, return_gate=True)
        self.last_gate_z = float(gz.mean().detach().item())
        return feat, new_state


class _CNNImageEncoder(nn.Module):
    """Small CNN for image POMDP: input (B, 6, H, W) = stacked obs+goal frames."""
    def __init__(self, in_ch=6, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 2), nn.GELU(),
            nn.Conv2d(32, 32, 3, 2), nn.GELU(),
            nn.Conv2d(32, 32, 3, 2), nn.GELU(),
            nn.Flatten(), nn.Linear(32 * 3 * 3, latent_dim), nn.GELU())

    def forward(self, x):
        return self.net(x)


def build_encoder(obs_dim, goal_dim, latent_dim, image=False):
    if image:
        return _CNNImageEncoder(in_ch=6, latent_dim=latent_dim)
    inp = obs_dim + goal_dim
    return nn.Sequential(nn.Linear(inp, latent_dim), nn.GELU(),
                         nn.Linear(latent_dim, latent_dim))


def build_backbone(kind, latent_dim, d_model=128, memory_length=8, state_gate="gru",
                   nhead=4, layers=2, max_hist=32):
    if kind == "mlp":
        return MLPMemory(latent_dim, d_model)
    if kind == "ctm":
        return CTMMemory(latent_dim, d_model, memory_length, state_gate=state_gate)
    if kind == "lstm":
        return LSTMMemory(latent_dim, d_model)
    if kind == "gru":
        return GRUMemory(latent_dim, d_model)
    if kind == "transformer":
        return TransformerMemory(latent_dim, d_model, nhead, layers, max_hist)
    if kind == "flash":
        return FlashBrainBackbone(latent_dim, d_model, memory_length=memory_length,
                                  state_gate=state_gate, gate_mode="learn")
    if kind == "flash-shallow":
        return FlashBrainBackbone(latent_dim, d_model, memory_length=memory_length,
                                  state_gate=state_gate, gate_mode="shallow")
    if kind == "flash-deep":
        return FlashBrainBackbone(latent_dim, d_model, memory_length=memory_length,
                                  state_gate=state_gate, gate_mode="deep")
    raise KeyError(kind)


class MemoryPolicyNetwork(nn.Module):
    """encoder (obs+goal -> latent) + memory backbone + actor/critic.

    Exposes the interface PPOTrainer expects: init_state / detach_state /
    mask_reset / snapshot / forward / recompute.
    """

    def __init__(self, encoder, backbone, action_dim):
        super().__init__()
        self.encoder = encoder
        self.backbone = backbone
        self.actor_mean = nn.Linear(backbone.feat_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(backbone.feat_dim, 1)

    def init_state(self, batch, device):
        self.backbone.init_state(batch, device)

    def detach_state(self):
        self.backbone.detach_state()

    def mask_reset(self, done_mask):
        self.backbone.mask_reset(done_mask)

    def snapshot(self):
        return self.backbone.snapshot()

    def _heads(self, feat):
        mean = self.actor_mean(feat)
        std = self.actor_logstd.exp().expand_as(mean)
        return Normal(mean, std), self.critic(feat)

    def forward(self, x):
        z = self.encoder(x)
        feat = self.backbone.step(z)
        return self._heads(feat)

    def forward_feat(self, x):
        """Same as forward but also returns the backbone feature (for probing)."""
        z = self.encoder(x)
        feat = self.backbone.step(z)
        dist, value = self._heads(feat)
        return feat, dist, value

    def recompute(self, x, state):
        z = self.encoder(x)
        if state is None:
            state = ()
        feat, _ = self.backbone.step_stateless(z, state)
        return self._heads(feat)


def build_memory_policy(kind, obs_dim, goal_dim, action_dim, latent_dim=64,
                        d_model=128, image=False, memory_length=8, state_gate="gru"):
    enc = build_encoder(obs_dim, goal_dim, latent_dim, image=image)
    back = build_backbone(kind, latent_dim, d_model, memory_length=memory_length,
                          state_gate=state_gate)
    return MemoryPolicyNetwork(enc, back, action_dim)
