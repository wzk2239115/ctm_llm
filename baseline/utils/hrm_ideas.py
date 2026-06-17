"""
HRM-Inspired idea modules for CTM.

Techniques learned from HRM (Hierarchical Reasoning Model) and HRM-Text,
adapted for the CTM (Continuous Thought Machine) architecture.

Modules:
  - AdamATan2: atan2-based optimizer from HRM-Text
  - stablemax_cross_entropy: numerically stable CE from HRM
  - add_hrm_idea_args: CLI args for all HRM-inspired ideas
  - compute_bp_steps: BP warmup schedule
  - GRUGate: gated state injection
  - GatedAttention: sigmoid-gated attention wrapper
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim.optimizer import Optimizer, ParamsT


# ═══════════════════════════════════════════════════════════════
# Adam-atan2 Optimizer (from HRM-Text)
# ═══════════════════════════════════════════════════════════════

class AdamATan2(Optimizer):
    """Adam-atan2 optimizer from HRM-Text.

    Uses atan2(exp_avg, denom) instead of exp_avg / denom for the update.
    This bounds the per-parameter update to [-lr, +lr], improving stability.

    Reference: HRM-Text (arXiv:2605.20613)
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        ema: Optional[float] = None,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "ema": ema,
        }
        super().__init__(params, defaults)
        self._init_state()

    @torch.no_grad()
    def _init_state(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = torch.tensor(0.0, dtype=torch.float32, device=p.device)
                if group["betas"][0] > 0:
                    state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
                if group["ema"] is not None:
                    state["param_ema"] = torch.empty_like(p).copy_(p)

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is None, "Closure is not supported"

        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue

                state = self.state[param]
                grad = param.grad

                if group["weight_decay"] != 0:
                    param.mul_(1 - group["lr"] * group["weight_decay"])

                if "exp_avg" in state:
                    state["exp_avg"].lerp_(grad, 1 - group["betas"][0])
                state["exp_avg_sq"].mul_(group["betas"][1]).addcmul_(grad, grad, value=1 - group["betas"][1])

                state["step"] += 1
                bias_correction1 = 1 - group["betas"][0] ** state["step"]
                bias_correction2 = 1 - group["betas"][1] ** state["step"]
                step_size = group["lr"] / bias_correction1
                bias_correction2_sqrt = bias_correction2.sqrt()

                denom = state["exp_avg_sq"].sqrt() / bias_correction2_sqrt

                if "exp_avg" in state:
                    param.add_(torch.atan2(state["exp_avg"], denom), alpha=-step_size)
                else:
                    param.add_(torch.atan2(grad, denom), alpha=-group["lr"])

                if "param_ema" in state:
                    state["param_ema"].lerp_(param, 1 - group["ema"])

    @torch.no_grad()
    def swap_ema(self):
        for group in self.param_groups:
            for param in group["params"]:
                state = self.state[param]
                if "param_ema" in state:
                    temp = torch.empty_like(param).copy_(param)
                    param.copy_(state["param_ema"])
                    state["param_ema"].copy_(temp)


# ═══════════════════════════════════════════════════════════════
# Stablemax Cross-Entropy (from HRM)
# ═══════════════════════════════════════════════════════════════

def _stablemax_transform(x, epsilon=1e-30):
    """Stablemax: s(x) = 1/(1-x) for x<0, x+1 for x>=0."""
    return torch.where(x < 0, 1.0 / (1.0 - x + epsilon), x + 1.0)


def _log_stablemax(x, dim=-1):
    s_x = _stablemax_transform(x)
    return torch.log(s_x / torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index=-100):
    """Stablemax cross-entropy loss.

    More numerically stable than softmax CE for extreme logit values.
    Used by HRM for puzzle solving tasks.
    """
    logprobs = _log_stablemax(logits.to(torch.float64), dim=-1)
    valid_mask = labels != ignore_index
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(
        logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1
    ).squeeze(-1)
    return -torch.where(valid_mask, prediction_logprobs, 0.0)


# ═══════════════════════════════════════════════════════════════
# Gated Attention (from HRM-Text)
# ═══════════════════════════════════════════════════════════════

class GatedAttention(nn.Module):
    """Wraps an existing attention module with a sigmoid gate.

    HRM-Text splits QKV projection into gate+query+key+value and applies
    sigmoid(gate) * attn_output. This is a lightweight wrapper that adds
    a gating projection to the attention output.

    Usage:
        attn = GatedAttention(nn.MultiheadAttention(...), d_input)
        out, weights = attn(q, k, v)
    """

    def __init__(self, base_attention: nn.Module, d_input: int):
        super().__init__()
        self.base = base_attention
        self.gate_proj = nn.Linear(d_input, d_input)

        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    def forward(self, query, key, value, **kwargs):
        out, weights = self.base(query, key, value, **kwargs)
        gate = torch.sigmoid(self.gate_proj(out))
        return out * gate, weights


# ═══════════════════════════════════════════════════════════════
# GRU Gate for Input Injection
# ═══════════════════════════════════════════════════════════════

class GRUGate(nn.Module):
    """GRU-style gating for combining attention output with latent state.

    Instead of concatenating attn_out and activated_state (which doubles
    the synapse input dimension), use a learned GRU gate:

        z = sigmoid(W_z [attn_out, state])
        h = (1-z) * state + z * relu(W_h attn_out)

    This is parameter-efficient and allows selective information flow.
    """

    def __init__(self, d_model: int, d_input: int):
        super().__init__()
        self.d_model = d_model
        self.d_input = d_input
        self.gate_proj = nn.Linear(d_model + d_input, d_model)
        self.value_proj = nn.Linear(d_input, d_model)
        nn.init.zeros_(self.gate_proj.weight)
        self.gate_proj.bias.data.fill_(-2.0)

    def forward(self, attn_out: Tensor, state: Tensor) -> Tensor:
        gate_input = torch.cat([attn_out, state], dim=-1)
        z = torch.sigmoid(self.gate_proj(gate_input))
        candidate = F.relu(self.value_proj(attn_out))
        return (1 - z) * state + z * candidate


# ═══════════════════════════════════════════════════════════════
# BP Steps Schedule (from HRM-Text)
# ═══════════════════════════════════════════════════════════════

def compute_bp_steps(
    current_step: int,
    total_steps: int,
    bp_warmup_ratio: float,
    bp_min_steps: int,
    bp_max_steps: int,
) -> int:
    """Compute how many ticks to backprop through at the current training step.

    During warmup (first bp_warmup_ratio of training), ramp from bp_min_steps
    to bp_max_steps. After warmup, use bp_max_steps.

    This is HRM-Text's bp_warmup_ratio schedule.
    """
    warmup_steps = int(total_steps * bp_warmup_ratio)
    if warmup_steps <= 0:
        return bp_max_steps
    progress = min(1.0, current_step / warmup_steps)
    return bp_min_steps + int(progress * (bp_max_steps - bp_min_steps))


# ═══════════════════════════════════════════════════════════════
# EMA Weight Tracking
# ═══════════════════════════════════════════════════════════════

class EMATracker:
    """Simple EMA weight tracker for evaluation.

    Usage:
        ema = EMATracker(model, decay=0.9999)
        # after each optimizer.step():
        ema.update(model)
        # for evaluation:
        ema.swap(model)  # swap EMA weights into model
        ... evaluate ...
        ema.swap(model)  # swap back
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {name: p.data.clone() for name, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    @torch.no_grad()
    def swap(self, model: nn.Module):
        for name, p in model.named_parameters():
            if name in self.shadow:
                temp = p.data.clone()
                p.data.copy_(self.shadow[name])
                self.shadow[name].copy_(temp)


# ═══════════════════════════════════════════════════════════════
# CLI Args
# ═══════════════════════════════════════════════════════════════

def add_hrm_idea_args(parser):
    """Add CLI arguments for all HRM-inspired ideas."""
    # ─── Phase A: Gradient Control ───
    parser.add_argument("--bp_steps", type=int, default=0,
                        help="Truncated BPTT: only backprop through last N ticks. 0=full BPTT.")
    parser.add_argument("--bp_warmup_ratio", type=float, default=0.0,
                        help="BP warmup: ramp bp_steps from bp_min to bp_max over this fraction of training.")
    parser.add_argument("--bp_min_steps", type=int, default=2,
                        help="Minimum bp_steps during warmup.")
    parser.add_argument("--bp_max_steps", type=int, default=10,
                        help="Maximum bp_steps after warmup.")
    parser.add_argument("--detach_every", type=int, default=0,
                        help="Detach state every K ticks (chunked BPTT). 0=never.")

    # ─── Phase B: Optimization ───
    parser.add_argument("--optimizer_type", type=str, default="adam",
                        choices=["adam", "adam_atan2"],
                        help="Optimizer type.")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1.")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta2.")
    parser.add_argument("--loss_type", type=str, default="softmax_ce",
                        choices=["softmax_ce", "stablemax_ce"],
                        help="Loss function type for classification tasks.")

    # ─── Phase C: Attention ───
    parser.add_argument("--gated_attention", action="store_true", default=False,
                        help="Enable sigmoid gate on attention output (HRM-Text style).")
    parser.add_argument("--input_injection", type=str, default="concat",
                        choices=["concat", "additive", "gru_gate"],
                        help="How to combine attention output with latent state.")

    # ─── Phase D: Hierarchical Recurrence ───
    parser.add_argument("--h_cycles", type=int, default=1,
                        help="Outer loop cycles for hierarchical recurrence. 1=flat (default).")
    parser.add_argument("--l_cycles", type=int, default=0,
                        help="Inner loop cycles. 0 means l_cycles = iterations / h_cycles.")

    # ─── Phase E: Training ───
    parser.add_argument("--ema_decay", type=float, default=0.0,
                        help="EMA decay for model weights. 0=disabled.")

    # ─── Phase E: ACT Q-learning Halting ───
    parser.add_argument("--act_halt", action="store_true", default=False,
                        help="Enable ACT Q-learning halting (HRM style).")
    parser.add_argument("--halt_max_steps", type=int, default=50,
                        help="Max steps before forced halt.")
    parser.add_argument("--halt_exploration_prob", type=float, default=0.1,
                        help="Exploration probability for Q-learning halting.")
    parser.add_argument("--halt_q_weight", type=float, default=0.5,
                        help="Weight for Q-learning loss (0.5 = equal to task loss).")

    return parser


# ═══════════════════════════════════════════════════════════════
# ACT Q-Learning Halting Loss (from HRM)
# ═══════════════════════════════════════════════════════════════

def compute_act_q_loss(
    q_logits: Tensor,
    is_correct_per_tick: Tensor,
    weight: float = 0.5,
) -> Tensor:
    """Compute ACT Q-learning halting loss (PQN-style, no replay buffer).

    Follows HRM's approach:
    - Q-halt loss: BCE(q_halt_logit, is_correct) — predict if output is correct
    - Q-continue loss: bootstrapped target from next tick's Q-values

    Args:
        q_logits: (B, 2, T) — halt and continue logits per tick
        is_correct_per_tick: (B, T) — float, 1.0 if prediction at that tick is correct
        weight: loss weight

    Returns:
        Scalar loss tensor.
    """
    q_halt = q_logits[:, 0, :]       # (B, T)
    q_continue = q_logits[:, 1, :]    # (B, T)
    T = q_logits.size(-1)

    # Q-halt loss: predict correctness
    q_halt_loss = F.binary_cross_entropy_with_logits(q_halt, is_correct_per_tick, reduction='mean')

    # Q-continue loss: bootstrapped target
    with torch.no_grad():
        if T > 1:
            next_max = torch.maximum(q_halt[:, 1:], q_continue[:, 1:])  # (B, T-1)
            target_q = torch.sigmoid(next_max)
            target_q = torch.cat([target_q, torch.sigmoid(q_halt[:, -1:])], dim=-1)  # (B, T)
        else:
            target_q = torch.sigmoid(q_halt)

    q_continue_loss = F.binary_cross_entropy_with_logits(q_continue, target_q, reduction='mean')

    return weight * (q_halt_loss + q_continue_loss)


# ═══════════════════════════════════════════════════════════════
# Factory: build optimizer from args
# ═══════════════════════════════════════════════════════════════

def build_optimizer_from_args(model_params, args, lr=None):
    """Build optimizer based on args.optimizer_type.

    Falls back to standard AdamW if optimizer_type is not recognized.
    """
    lr = lr if lr is not None else getattr(args, "lr", 1e-3)
    wd = getattr(args, "weight_decay", 0.0)
    beta1 = getattr(args, "beta1", 0.9)
    beta2 = getattr(args, "beta2", 0.999)

    opt_type = getattr(args, "optimizer_type", "adam")
    if opt_type == "adam_atan2":
        return AdamATan2(
            model_params,
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=wd,
        )
    # Default: standard AdamW
    return torch.optim.AdamW(
        model_params,
        lr=lr,
        betas=(beta1, beta2),
        weight_decay=wd,
    )
