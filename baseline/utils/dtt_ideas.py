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
                        choices=["ctc", "per_tick_ce", "per_tick_sinkhorn"],
                        help="Sort loss: 'ctc' (sequence alignment, needs full BPTT) "
                             "or 'per_tick_ce' (each tick independently predicts full "
                             "ordering, compatible with one-step gradient) "
                             "or 'per_tick_sinkhorn' (per-tick prediction + Sinkhorn "
                             "doubly-stochastic projection; truncation-native AND "
                             "enforces valid permutation structure).")

    # ─── Sinkhorn Permutation Structure (for per_tick_sinkhorn mode) ───
    parser.add_argument("--sinkhorn_iters", type=int, default=5,
                        help="Sinkhorn normalization iterations. More -> closer to a "
                             "true permutation but costlier. 0 disables (== per_tick_ce).")
    parser.add_argument("--sinkhorn_tau", type=float, default=1.0,
                        help="Sinkhorn temperature at start of training. Higher = softer "
                             "(more uniform), lower = sharper (closer to hard permutation).")
    parser.add_argument("--sinkhorn_tau_min", type=float, default=0.1,
                        help="Sinkhorn temperature annealed toward this by end of training. "
                             "Set == sinkhorn_tau to disable annealing.")
    parser.add_argument("--sinkhorn_anneal", type=str, default="linear",
                        choices=["none", "linear"],
                        help="Temperature anneal schedule. 'none' keeps tau constant.")

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
                        help="State momentum correction weight. 0=disabled.")
    parser.add_argument("--dtt_momentum_decay", type=float, default=0.99,
                        help="EMA decay rate for state momentum accumulators.")

    # ─── Multi-Scale Hierarchy (MSH) ───
    parser.add_argument("--msh_levels", type=str, default="",
                        help="N-level hierarchy periods, comma-separated innermost→outermost. "
                             "e.g. '10,5,1' = 3-level: 10 fast × 5 medium × 1 slow = 50 total. "
                             "Empty = flat (no hierarchy). "
                             "Nested mode: product must equal --iterations. "
                             "Coprime mode: periods should be coprime (primes); "
                             "T is independent, full resonance at lcm(periods). "
                             "Learnable mode: periods used for gate initialization only.")
    parser.add_argument("--msh_mode", type=str, default="nested",
                        choices=["nested", "coprime", "learnable"],
                        help="Hierarchy mode: 'nested' = cumulative-product boundaries; "
                             "'coprime' = independent prime periods via CRT; "
                             "'learnable' = soft gates learned during training "
                             "(init from coprime pattern, can drift to task-optimal).")
    parser.add_argument("--msh_sn_scale", type=float, default=0.0,
                        help="Spectral norm scale for level synapses. 0=disabled.")
    parser.add_argument("--msh_gate_init", type=str, default="coprime",
                        choices=["coprime", "random", "uniform"],
                        help="Learnable mode: gate initialization. "
                             "'coprime' = init from msh_levels pattern; "
                             "'random' = random normal; "
                             "'uniform' = all zeros (50% update probability).")
    parser.add_argument("--msh_gate_sparsity", type=float, default=0.0,
                        help="Learnable mode: sparsity regularization weight for gates. "
                             "Penalizes high average gate activation to encourage sparse updates.")

    return parser


# ═══════════════════════════════════════════════════════════════
# Output Dimensions
# ═══════════════════════════════════════════════════════════════

def get_sort_out_dims(N_to_sort, sort_loss_mode):
    """Compute out_dims for the sort task based on loss mode.

    CTC mode: each tick outputs 1 distribution over N+1 classes (N + blank).
    Per-tick CE / Sinkhorn mode: each tick outputs N distributions, each over N classes.
    """
    if sort_loss_mode in ("per_tick_ce", "per_tick_sinkhorn"):
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
# Sinkhorn Permutation Structure (truncation-native sort loss)
# ═══════════════════════════════════════════════════════════════

def sinkhorn_normalize(scores, n_iters=5, tau=1.0):
    """Differentiable Sinkhorn normalization → doubly-stochastic matrix.

    A doubly-stochastic matrix is the continuous relaxation of a permutation
    matrix: each row and each column sums to 1. This enforces the bijection
    constraint that per_tick_ce lacks (no two positions can claim the same element).

    Operates in log-space for numerical stability. Each call is LOCAL to one tick
    (no cross-tick dependency) → remains truncation-native (bp_steps=1 compatible).

    Args:
        scores: (..., N, N) raw logits. dim[-2] = position, dim[-1] = element.
        n_iters: Sinkhorn iterations (alternating row/col normalization).
        tau: temperature. Higher = softer (uniform), lower = sharper (permutation).

    Returns:
        (..., N, N) doubly-stochastic matrix (probabilities).
    """
    log_alpha = scores / max(tau, 1e-6)
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
    return log_alpha.exp()


def _current_tau(args_tau, args_tau_min, cur_step, total_steps, anneal="linear"):
    """Resolve the effective Sinkhorn temperature, with optional annealing."""
    if anneal == "none" or args_tau_min >= args_tau or total_steps is None or total_steps <= 0 or cur_step is None:
        return args_tau
    frac = min(1.0, max(0.0, float(cur_step) / float(total_steps)))
    return args_tau + (args_tau_min - args_tau) * frac


def per_tick_sinkhorn_sort_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    certainties: Optional[torch.Tensor] = None,
    N_to_sort: Optional[int] = None,
    progressive_mode: str = "none",
    exp_decay: float = 0.95,
    sinkhorn_iters: int = 5,
    sinkhorn_tau: float = 1.0,
    sinkhorn_tau_min: float = 0.1,
    anneal: str = "linear",
    cur_step: Optional[int] = None,
    total_steps: Optional[int] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Truncation-native sort loss with enforced permutation structure.

    Each tick independently predicts the full N-position ordering (truncation-native,
    like per_tick_ce), but the N×N score matrix is projected through Sinkhorn to a
    doubly-stochastic matrix before the loss. This restores the bijection constraint
    that lets sort scale to large N, WITHOUT any cross-tick dependency (so one-step
    gradient / bp_steps=1 stays valid).

    Args:
        predictions: (B, N*N, T) — N positions × N classes per tick.
        targets: (B, N) — sorted indices. targets[b, j] = element at position j.
    """
    B = predictions.size(0)
    T = predictions.size(-1)
    N = N_to_sort if N_to_sort is not None else targets.size(-1)

    # Vectorized: move tick dim next to batch so Sinkhorn runs on all (B*T)
    # matrices at once (last two dims = position x element). No Python loop →
    # far fewer kernel launches → much higher GPU util on small models.
    pred = predictions.reshape(B, N, N, T).permute(0, 3, 1, 2).contiguous()  # (B, T, N, N)
    tgt = targets.long()                                                     # (B, N)

    tau = _current_tau(sinkhorn_tau, sinkhorn_tau_min, cur_step, total_steps, anneal)

    P = sinkhorn_normalize(pred, sinkhorn_iters, tau)                        # (B, T, N, N)
    tgt_exp = tgt.unsqueeze(1).unsqueeze(-1).expand(B, T, N, 1)              # (B, T, N, 1)
    p_correct = P.gather(3, tgt_exp).squeeze(-1)                             # (B, T, N)
    per_tick_losses = -torch.log(p_correct + 1e-9).mean(dim=-1)              # (B, T)

    if reduction == "none":
        return per_tick_losses

    weights = compute_progressive_weights(
        certainties, T, progressive_mode, exp_decay, predictions.device,
    )
    return (per_tick_losses * weights.unsqueeze(0)).sum(dim=1).mean()


def decode_permutation_hungarian(predictions, N_to_sort=None, tick=None):
    """Decode a valid permutation via exact Hungarian assignment (eval only).

    Guarantees a valid bijection, unlike argmax. Runs on CPU.

    Returns:
        (B, N) numpy array: predicted element index at each position.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    B = predictions.size(0)
    T = predictions.size(-1)
    N = N_to_sort if N_to_sort is not None else int(round(predictions.size(1) ** 0.5))
    t = tick if tick is not None else T - 1
    scores = predictions.reshape(B, N, N, T)[..., t].detach().float().cpu().numpy()

    out = np.empty((B, N), dtype=np.int64)
    for b in range(B):
        row, col = linear_sum_assignment(-scores[b])
        out[b] = col[np.argsort(row)]
    return out


def compute_sinkhorn_accuracy(predictions, targets, N_to_sort=None):
    """Return (fine_acc, full_acc) using Hungarian decoding on the last tick."""
    import numpy as np
    decoded = decode_permutation_hungarian(predictions, N_to_sort)
    tgt = targets.detach().cpu().numpy()
    fine = float((decoded == tgt).mean())
    full = float((decoded == tgt).all(axis=1).mean())
    return fine, full


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


# ═══════════════════════════════════════════════════════════════
# Multi-Scale Hierarchy (MSH) Helpers
# ═══════════════════════════════════════════════════════════════

def parse_msh_levels(levels_str):
    """Parse MSH levels string into a list of ints.

    '10,5,1' → [10, 5, 1]
    Product of levels = total iterations.
    levels[0] = innermost (fastest), levels[-1] = outermost (slowest).
    Gradient path = levels[-1].
    """
    if not levels_str:
        return None
    levels = [int(x.strip()) for x in levels_str.split(',')]
    assert all(l > 0 for l in levels), f"Level periods must be positive: {levels}"
    return levels


def compute_level_boundaries(levels):
    """Compute the step boundaries at which each macro level updates.

    For levels = [10, 5, 1]:
      Level 0 (fastest macro): updates every 10 steps → boundaries = [9, 19, 29, 39, 49]
      Level 1 (slowest macro): updates every 50 steps → boundaries = [49]

    Returns:
        list of sets: boundaries[i] = set of step indices where level i updates.
    """
    n_macro = len(levels) - 1
    total = 1
    for l in levels:
        total *= l

    boundaries = []
    cumulative = 1
    for i in range(n_macro):
        cumulative *= levels[i]
        period = cumulative
        bset = set()
        for stepi in range(total):
            if (stepi + 1) % period == 0:
                bset.add(stepi)
        boundaries.append(bset)

    return boundaries


def build_msh_synapses(levels, d_model, sn_scale=0.0, device='cpu'):
    """Build N-1 level synapse modules for the MSH hierarchy.

    Each level synapse maps activated_state → state update (additive).
    Structure: LayerNorm → Linear → GLU → LayerNorm (same as HRM's h_synapse).

    Args:
        levels: list of level periods (e.g., [10, 5, 1]).
        d_model: model dimension.
        sn_scale: spectral norm scale. 0 = disabled.

    Returns:
        nn.ModuleList of N-1 synapse modules.
    """
    import torch.nn as nn

    n_macro = len(levels) - 1
    synapses = []
    for i in range(n_macro):
        syn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GLU(),
            nn.LayerNorm(d_model),
        )
        if sn_scale > 0:
            for m in syn:
                if isinstance(m, nn.Linear):
                    torch.nn.utils.parametrizations.spectral_norm(m)
        synapses.append(syn)

    result = nn.ModuleList(synapses).to(device)
    return result


def should_update_level(stepi, levels, level_idx):
    """Check if macro level `level_idx` should update at step `stepi`.

    Level i updates every (product of levels[0:i+1]) steps.
    """
    cumulative = 1
    for j in range(level_idx + 1):
        cumulative *= levels[j]
    return (stepi + 1) % cumulative == 0


# ═══════════════════════════════════════════════════════════════
# Coprime Mode Helpers
# ═══════════════════════════════════════════════════════════════

def should_update_level_coprime(stepi, periods, level_idx):
    """Check if macro level should update at step (coprime mode).

    In coprime mode, each level updates independently at its own period.
    No cumulative dependency between levels.

    Args:
        stepi: current step (0-indexed).
        periods: list of coprime periods, e.g., [2, 3, 5].
        level_idx: which level to check.

    Returns:
        True if level should update at this step.
    """
    return (stepi + 1) % periods[level_idx] == 0


def get_resonance_patterns(periods, T):
    """Enumerate all synchronization patterns for coprime periods over T steps.

    For periods [2, 3, 5], T=30:
      Returns dict mapping frozenset(active_levels) → list of step indices.

    This reveals the "resonance spectrum": which levels co-activate when.
    """
    from math import gcd
    from functools import reduce

    def lcm(a, b):
        return a * b // gcd(a, b)

    full_cycle = reduce(lcm, periods)

    patterns = {}
    for stepi in range(min(T, full_cycle)):
        active = frozenset(
            i for i, p in enumerate(periods)
            if (stepi + 1) % p == 0
        )
        if active:
            patterns.setdefault(active, []).append(stepi)

    return patterns


def print_resonance_report(periods, T):
    """Print a human-readable resonance pattern report."""
    patterns = get_resonance_patterns(periods, T)

    from math import gcd
    from functools import reduce
    def lcm(a, b):
        return a * b // gcd(a, b)
    full_cycle = reduce(lcm, periods)

    print(f"\n  Coprime periods: {periods}")
    print(f"  Full resonance cycle: lcm = {full_cycle}")
    print(f"  Distinct sync patterns: {len(patterns)}")
    print(f"  Pattern breakdown:")

    for active in sorted(patterns, key=lambda s: (len(s), sorted(s))):
        steps = patterns[active]
        labels = "+".join(f"L{i}" for i in sorted(active))
        density = len(steps) / min(T, full_cycle)
        full = " ★ FULL RESONANCE" if len(active) == len(periods) else ""
        print(f"    {{{labels}}}: {len(steps)} times "
              f"(density {density:.3f}){full}")


# ═══════════════════════════════════════════════════════════════
# Learnable Gate Helpers
# ═══════════════════════════════════════════════════════════════

def init_gate_logits(n_macro, iterations, init_mode='coprime', periods=None):
    """Initialize learnable gate logits for MSH learnable mode.

    Args:
        n_macro: number of macro levels.
        iterations: total number of ticks (T).
        init_mode: 'coprime' (init from periods pattern), 'random', or 'uniform'.
        periods: list of periods for coprime init (e.g., [2, 3, 5]).

    Returns:
        Tensor of shape (n_macro, iterations) — raw logits before sigmoid.
    """
    import torch

    if init_mode == 'coprime' and periods is not None:
        logits = torch.full((n_macro, iterations), -5.0)
        for level_idx in range(min(n_macro, len(periods))):
            period = periods[level_idx]
            for t in range(iterations):
                if (t + 1) % period == 0:
                    logits[level_idx, t] = 5.0
        return logits
    elif init_mode == 'uniform':
        return torch.zeros(n_macro, iterations)
    else:
        return torch.randn(n_macro, iterations) * 2.0


def compute_gate_sparsity_loss(gate_logits):
    """Sparsity regularization for learnable gates.

    Encourages gates to be sparse (most ticks: gate ≈ 0, few ticks: gate ≈ 1).
    Uses L1 on sigmoid(gate_logits).

    Returns:
        Scalar loss tensor.
    """
    import torch
    return torch.sigmoid(gate_logits).mean()


def visualize_learned_gates(gate_logits, iterations):
    """Print a text visualization of learned gate patterns.

    Args:
        gate_logits: (n_macro, iterations) learned parameter.
        iterations: T.
    """
    import torch

    gates = torch.sigmoid(gate_logits).detach().cpu()
    n_macro = gates.size(0)

    print(f"\n  Learned gate patterns ({n_macro} levels × {iterations} ticks):")
    for level_idx in range(n_macro):
        row = gates[level_idx]
        active = (row > 0.5).sum().item()
        density = active / iterations
        bar = ''.join('█' if g > 0.5 else ('▒' if g > 0.1 else '·') for g in row)
        print(f"    L{level_idx}: [{bar}] active={active}/{iterations} ({density:.1%})")

    # Check pairwise coprimality of learned periods
    print(f"  Inferred approximate periods:")
    for level_idx in range(n_macro):
        row = gates[level_idx]
        active_steps = (row > 0.5).nonzero(as_tuple=True)[0]
        if len(active_steps) > 1:
            diffs = active_steps[1:] - active_steps[:-1]
            median_period = diffs.float().median().item()
            print(f"    L{level_idx}: ~period {median_period:.1f} "
                  f"(steps: {active_steps[:10].tolist()}{'...' if len(active_steps) > 10 else ''})")
        else:
            print(f"    L{level_idx}: ~period N/A (too few activations)")

