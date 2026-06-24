"""
Decoupled Thought Training (DTT) — One-Step Gradient for CTM.

Core insight: HRM's one-step gradient fails on CTM sort because CTC loss
needs gradient across ALL ticks. By reformulating the loss so each tick
independently predicts the full answer, one-step gradient becomes viable.

This module provides:
  - add_dtt_args(): CLI args for DTT
  - per_tick_sort_loss(): per-tick independent CE loss for sort
  - compute_progressive_weights(): tick-level loss weighting
  - compute_per_tick_accuracy(): evaluation for per-tick mode
  - get_sort_out_dims(): compute out_dims based on loss mode
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ═══════════════════════════════════════════════════════════════
# CLI Args
# ═══════════════════════════════════════════════════════════════

def add_dtt_args(parser):
    """Add CLI arguments for Decoupled Thought Training."""
    # ─── Loss Reformulation ───
    parser.add_argument("--sort_loss_mode", type=str, default="ctc",
                        choices=["ctc", "per_tick_ce"],
                        help="Sort loss: 'ctc' (sequence alignment, needs full BPTT) "
                             "or 'per_tick_ce' (each tick independently predicts full "
                             "ordering, compatible with one-step gradient).")

    # ─── Progressive Weighting ───
    parser.add_argument("--dtt_progressive_mode", type=str, default="none",
                        choices=["none", "linear", "certainty", "exp"],
                        help="Per-tick loss weighting mode. "
                             "none: equal weight. "
                             "linear: later ticks weighted higher (drafts→refined). "
                             "certainty: weight by certainty (like CTM's existing mechanism). "
                             "exp: exponential decay weight for early ticks.")
    parser.add_argument("--dtt_exp_decay", type=float, default=0.95,
                        help="Decay rate for exp progressive mode.")

    # ─── State Momentum Correction ───
    parser.add_argument("--dtt_momentum_weight", type=float, default=0.0,
                        help="State momentum correction weight. 0=disabled. "
                             "When >0, maintains EMA of synchronisation accumulators "
                             "and applies correction to prevent detached state drift.")
    parser.add_argument("--dtt_momentum_decay", type=float, default=0.99,
                        help="EMA decay rate for state momentum accumulators.")

    return parser


# ═══════════════════════════════════════════════════════════════
# Output Dimensions
# ═══════════════════════════════════════════════════════════════

def get_sort_out_dims(N_to_sort, sort_loss_mode):
    """Compute out_dims for the sort task based on loss mode.

    CTC mode: each tick outputs 1 distribution over N+1 classes (N + blank).
    Per-tick CE mode: each tick outputs N distributions, each over N classes.
    """
    if sort_loss_mode == "per_tick_ce":
        return N_to_sort * N_to_sort
    else:
        return N_to_sort + 1


# ═══════════════════════════════════════════════════════════════
# Progressive Weighting
# ═══════════════════════════════════════════════════════════════

def compute_progressive_weights(
    certainties: Optional[torch.Tensor],
    T: int,
    mode: str = "none",
    exp_decay: float = 0.95,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Compute per-tick loss weights.

    Args:
        certainties: (B, 2, T) certainty tensor, or None.
            certainties[:, 1] is "1 - normalized_entropy" (higher=more certain).
        T: number of ticks.
        mode: weighting mode.
        exp_decay: decay rate for 'exp' mode.

    Returns:
        weights: (T,) tensor, normalized to sum=1.
    """
    if mode == "none":
        return torch.ones(T, device=device) / T

    if mode == "linear":
        # Linear ramp: tick 0 gets weight 0.1, tick T-1 gets weight 1.0
        weights = torch.linspace(0.1, 1.0, T, device=device)

    elif mode == "exp":
        # Exponential: w[t] = exp_decay^(T-1-t)
        weights = torch.tensor(
            [exp_decay ** (T - 1 - t) for t in range(T)],
            device=device, dtype=torch.float32,
        )

    elif mode == "certainty":
        # Certainty-based: use the model's own certainty as weight
        if certainties is not None:
            cert = certainties[:, 1, :]  # (B, T)
            weights = F.softmin(cert.mean(dim=0), dim=0)  # (T,) — softmin so less certain ticks get MORE weight
        else:
            weights = torch.ones(T, device=device) / T
    else:
        weights = torch.ones(T, device=device) / T

    # Normalize
    weights = weights / weights.sum()
    return weights


# ═══════════════════════════════════════════════════════════════
# Per-Tick Sort Loss
# ═══════════════════════════════════════════════════════════════

def per_tick_sort_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    certainties: Optional[torch.Tensor] = None,
    N_to_sort: Optional[int] = None,
    progressive_mode: str = "none",
    exp_decay: float = 0.95,
    loss_type: str = "softmax_ce",
    reduction: str = "mean",
) -> torch.Tensor:
    """Per-tick independent CE loss for sort.

    Each tick independently predicts the FULL sorted ordering (all N positions).
    This decouples the loss from requiring gradient across all ticks,
    making it compatible with one-step gradient (bp_steps=1).

    Args:
        predictions: (B, N*N, T) — flattened N positions × N classes per tick.
            Produced by output_projector with out_dims = N*N.
        targets: (B, N) — sorted indices (argsort of input).
        certainties: (B, 2, T) — for progressive weighting.
        N_to_sort: N. If None, inferred from targets.
        progressive_mode: per-tick loss weighting mode.
        exp_decay: decay rate for 'exp' progressive mode.
        loss_type: 'softmax_ce' or 'stablemax_ce'.
        reduction: 'mean' or 'none'.

    Returns:
        Scalar loss tensor (if reduction='mean') or (B, T) per-tick losses.
    """
    B = predictions.size(0)
    T = predictions.size(-1)

    if N_to_sort is None:
        N_to_sort = targets.size(-1)
    N = N_to_sort

    # Reshape predictions: (B, N*N, T) → (B, N, N, T)
    pred_reshaped = predictions.reshape(B, N, N, T)

    # Compute per-tick CE loss
    # For each tick: CE over N positions, each predicting one of N classes
    per_tick_losses = torch.empty(B, T, device=predictions.device, dtype=torch.float32)

    for t in range(T):
        pred_t = pred_reshaped[..., t]  # (B, N, N)

        if loss_type == "stablemax_ce":
            from baseline.utils.hrm_ideas import stablemax_cross_entropy
            loss_t = stablemax_cross_entropy(
                pred_t.reshape(B * N, N),
                targets.reshape(B * N),
            ).reshape(B, N).mean(dim=1)  # (B,)
        else:
            loss_t = F.cross_entropy(
                pred_t.reshape(B * N, N),
                targets.reshape(B * N).long(),
                reduction="none",
            ).reshape(B, N).mean(dim=1)  # (B,)

        per_tick_losses[:, t] = loss_t

    if reduction == "none":
        return per_tick_losses

    # Progressive weighting
    weights = compute_progressive_weights(
        certainties, T, progressive_mode, exp_decay, predictions.device,
    )

    # Weighted average: sum over ticks, mean over batch
    weighted_loss = (per_tick_losses * weights.unsqueeze(0)).sum(dim=1).mean()
    return weighted_loss


# ═══════════════════════════════════════════════════════════════
# Per-Tick Accuracy (for evaluation)
# ═══════════════════════════════════════════════════════════════

def compute_per_tick_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    N_to_sort: Optional[int] = None,
    use_last_tick: bool = True,
) -> float:
    """Compute accuracy for per-tick CE sort predictions.

    Args:
        predictions: (B, N*N, T) — per-tick predictions.
        targets: (B, N) — sorted indices.
        N_to_sort: N. If None, inferred from targets.
        use_last_tick: if True, use the last tick's prediction.
                       if False, use the most certain tick.

    Returns:
        Accuracy (fraction of samples with fully correct ordering).
    """
    B = predictions.size(0)
    T = predictions.size(-1)

    if N_to_sort is None:
        N_to_sort = targets.size(-1)
    N = N_to_sort

    # Reshape: (B, N*N, T) → (B, N, N, T)
    pred_reshaped = predictions.reshape(B, N, N, T)

    # Use last tick (most refined)
    tick = T - 1 if use_last_tick else T - 1
    pred_final = pred_reshaped[..., tick]  # (B, N, N)

    # Decode: argmax over classes → predicted ordering (B, N)
    predicted_ordering = pred_final.argmax(dim=-1)  # (B, N)

    # Accuracy: exact match of full ordering
    correct = (predicted_ordering == targets).all(dim=1).float().mean().item()
    return correct


def compute_per_tick_fine_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    N_to_sort: Optional[int] = None,
) -> float:
    """Compute fine-grained accuracy (per-position, not per-sequence).

    Returns fraction of individual positions that are correct.
    """
    B = predictions.size(0)
    T = predictions.size(-1)

    if N_to_sort is None:
        N_to_sort = targets.size(-1)
    N = N_to_sort

    pred_reshaped = predictions.reshape(B, N, N, T)
    pred_final = pred_reshaped[..., T - 1]  # (B, N, N)
    predicted_ordering = pred_final.argmax(dim=-1)  # (B, N)

    accuracy = (predicted_ordering == targets).float().mean().item()
    return accuracy


# ═══════════════════════════════════════════════════════════════
# State Momentum Correction (for ctm_sort.py forward)
# ═══════════════════════════════════════════════════════════════

class StateMomentumTracker:
    """Maintains EMA of synchronisation accumulators to correct detached state drift.

    Problem: with bp_steps=1, the decay_alpha and decay_beta from no_grad ticks
    are computed with the CURRENT model parameters. But these accumulators were
    built from MANY ticks of forward passes. As training updates parameters,
    the detached accumulators become stale.

    Solution: maintain an EMA of the accumulators. During grad ticks, add a
    correction term that pulls the synchronisation toward the EMA.

    Usage in CTM forward:
        tracker = StateMomentumTracker(momentum_weight=0.3, decay=0.99)

        for stepi in range(iterations):
            grad_enabled = stepi >= iterations - bp_steps

            # ... compute synchronisation ...
            synch_out, decay_alpha, decay_beta = self.compute_synchronisation(...)

            if not grad_enabled:
                # no_grad tick: update EMA
                tracker.update(decay_alpha.detach(), decay_beta.detach())
            else:
                # grad tick: apply correction
                correction = tracker.compute_correction(decay_alpha, decay_beta)
                synch_out = synch_out + correction
    """

    def __init__(self, momentum_weight: float = 0.3, decay: float = 0.99):
        self.momentum_weight = momentum_weight
        self.decay = decay
        self.alpha_ema = None
        self.beta_ema = None
        self._initialized = False

    def update(self, decay_alpha: torch.Tensor, decay_beta: torch.Tensor):
        """Update EMA with current tick's accumulators (called during no_grad ticks)."""
        if not self._initialized:
            self.alpha_ema = decay_alpha.clone()
            self.beta_ema = decay_beta.clone()
            self._initialized = True
        else:
            self.alpha_ema.mul_(self.decay).add_(decay_alpha, alpha=1 - self.decay)
            self.beta_ema.mul_(self.decay).add_(decay_beta, alpha=1 - self.decay)

    def compute_correction(self, decay_alpha: torch.Tensor, decay_beta: torch.Tensor) -> torch.Tensor:
        """Compute synchronisation correction term (called during grad ticks).

        The corrected synchronisation is:
            s_corrected = alpha / sqrt(beta) + w * (alpha_ema / sqrt(beta_ema) - alpha / sqrt(beta))

        This pulls the current synchronisation toward the EMA "consensus" value.
        """
        if not self._initialized or self.momentum_weight <= 0:
            return 0.0

        with torch.no_grad():
            current_synch = decay_alpha / (torch.sqrt(decay_beta) + 1e-8)
            ema_synch = self.alpha_ema / (torch.sqrt(self.beta_ema) + 1e-8)
            correction = self.momentum_weight * (ema_synch - current_synch)

        return correction

    def reset(self):
        """Reset EMA accumulators (call at the start of each forward pass)."""
        self._initialized = False
        self.alpha_ema = None
        self.beta_ema = None
