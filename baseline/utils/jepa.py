"""Cross-tick JEPA utilities for CTM baseline tasks.

Each tick produces a synchronisation representation (synch_out).
tick_{i+1}'s synch is predicted from tick_i's synch via a lightweight
predictor. Cosine/MSE loss + stop-gradient on target prevents collapse.

Adaptive weight modes (avoid hand-tuning cross_tick_jepa_weight):
  fixed        : static weight (legacy)
  balance  (A) : loss-magnitude balancing — JEPA contributes a fixed
                 fraction of the main loss magnitude, so the base weight
                 becomes "JEPA = N% of main" instead of a sensitive scalar.
  gate     (B) : acc-gated sigmoid — JEPA only opens up once the main task
                 has actually started learning (acc EMA crosses threshold).
  uncertainty(C): learnable sigma (Kendall 2018) — the optimizer finds the
                 relative weight automatically via a log-variance param.
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossTickJEPAPredictor(nn.Module):
    """Lightweight MLP that maps synch[t] -> predicted synch[t+1]."""
    def __init__(self, synch_dim, hidden_dim=512, depth=2, dropout=0.1):
        super().__init__()
        layers = []
        dims = [synch_dim] + [hidden_dim] * (depth - 1) + [synch_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=False))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
        self.predictor = nn.Sequential(*layers)

    def forward(self, x):
        return self.predictor(x)


class AdaptiveJEPAController(nn.Module):
    """Wraps a CrossTickJEPAPredictor with an adaptive loss-weight scheme.

    Replaces the static ``cross_tick_jepa_weight`` multiplier with one of
    three schemes (see module docstring). ``fixed`` reproduces legacy
    behaviour. Exposes ``.predictor`` so it drops into the existing
    ``model.cross_tick_predictor`` slot; being an nn.Module attached before
    optimizer construction, its parameters (incl. the ``log_sigma`` of
    scheme C) are collected automatically by ``model.parameters()``.
    """

    def __init__(self, synch_dim, args):
        super().__init__()
        self.predictor = CrossTickJEPAPredictor(
            synch_dim,
            int(args.cross_tick_jepa_hidden_dim),
            int(args.cross_tick_jepa_predictor_depth),
            float(args.cross_tick_jepa_dropout),
        )
        self.mode = str(getattr(args, 'jepa_weight_mode', 'fixed'))
        self.base_weight = float(args.cross_tick_jepa_weight)
        self.loss_type = str(args.cross_tick_jepa_loss)
        self.target_stop_grad = bool(args.cross_tick_jepa_target_stop_grad)
        # (A) balance
        self.balance_ratio = float(getattr(args, 'jepa_balance_ratio', 0.3))
        lo, hi = getattr(args, 'jepa_balance_clamp', (0.0, 10.0))
        self.balance_lo = float(lo)
        self.balance_hi = float(hi)
        # (B) gate
        self.gate_threshold = float(getattr(args, 'jepa_gate_threshold', 0.7))
        self.gate_temp = float(getattr(args, 'jepa_gate_temp', 0.05))
        self.gate_ema = float(getattr(args, 'jepa_gate_ema', 0.99))
        self.register_buffer('acc_ema', torch.tensor(0.5))
        # (C) uncertainty
        self.log_sigma = None
        if self.mode == 'uncertainty':
            self.log_sigma = nn.Parameter(
                torch.tensor(float(getattr(args, 'jepa_log_sigma_init', 0.0))))
        # diagnostics
        self.last_raw_jepa = 0.0
        self.last_effective_weight = 0.0

    def update_gate_acc(self, acc):
        """EMA-update the gate's running accuracy (no-op unless mode=='gate').

        Call once per train step after the batch accuracy is known, so the
        next loss computation sees an up-to-date main-task-progress signal.
        """
        if self.mode != 'gate' or acc is None:
            return
        with torch.no_grad():
            a = torch.as_tensor(float(acc), device=self.acc_ema.device,
                                dtype=self.acc_ema.dtype)
            self.acc_ema.mul_(self.gate_ema).add_(a, alpha=1.0 - self.gate_ema)

    def _raw_jepa(self, synch_per_tick):
        return _raw_jepa_loss(self.predictor, synch_per_tick,
                              self.loss_type, self.target_stop_grad)

    def compute_loss(self, synch_per_tick, main_loss=None, acc=None):
        """Return the (scalar) JEPA loss term to add to the total loss."""
        jepa_raw = self._raw_jepa(synch_per_tick)
        self.last_raw_jepa = float(jepa_raw.detach().item())

        if self.mode == 'uncertainty' and self.log_sigma is not None:
            # Kendall multi-task: L = exp(-s)*L_jepa + s, s = log(sigma^2)
            precision = torch.exp(-self.log_sigma)
            self.last_effective_weight = float(precision.detach().item())
            return precision * jepa_raw + self.log_sigma

        if self.mode == 'balance':
            if main_loss is None:
                w = torch.as_tensor(self.base_weight, device=jepa_raw.device,
                                    dtype=jepa_raw.dtype)
            else:
                ref = main_loss.detach()
                denom = jepa_raw.detach().clamp(min=1e-6)
                w = (self.balance_ratio * ref / denom).clamp(
                    self.balance_lo, self.balance_hi)
            self.last_effective_weight = float(w.detach().item())
            return w * jepa_raw

        if self.mode == 'gate':
            if acc is not None:
                self.update_gate_acc(acc)
            mult = torch.sigmoid((self.acc_ema - self.gate_threshold) / self.gate_temp)
            w = self.base_weight * mult
            self.last_effective_weight = float(w.detach().item())
            return w * jepa_raw

        # fixed (fallback)
        self.last_effective_weight = self.base_weight
        return self.base_weight * jepa_raw


def _raw_jepa_loss(predictor, synch_per_tick, loss_type='cosine',
                   target_stop_grad=True):
    """Unweighted cross-tick JEPA loss over adjacent tick pairs."""
    num_ticks = synch_per_tick.size(-1)
    if num_ticks < 2 or predictor is None:
        return synch_per_tick.new_zeros(())
    total = synch_per_tick.new_zeros(())
    count = 0
    for t in range(num_ticks - 1):
        src = synch_per_tick[..., t]       # (B, synch_dim)
        tgt = synch_per_tick[..., t + 1]   # (B, synch_dim)
        if target_stop_grad:
            tgt = tgt.detach()
        pred = predictor(src)
        if loss_type == 'cosine':
            pred = F.normalize(pred, dim=-1)
            tgt = F.normalize(tgt, dim=-1)
            total = total + (1 - (pred * tgt).sum(dim=-1)).mean()
        else:
            total = total + F.mse_loss(pred, tgt)
        count += 1
    return total / count if count > 0 else total


def add_jepa_args(parser):
    """Add cross-tick JEPA CLI arguments to an argument parser."""
    group = parser.add_argument_group('Cross-Tick JEPA')
    group.add_argument('--cross_tick_jepa_weight', type=float, default=0.0,
                       help='Weight for cross-tick JEPA loss (0 = disabled). Acts as '
                            'the base weight for fixed/gate modes; balance/uncertainty '
                            'derive the effective weight adaptively.')
    group.add_argument('--cross_tick_jepa_loss', type=str, default='cosine',
                       choices=['cosine', 'mse'],
                       help='JEPA loss type.')
    group.add_argument('--cross_tick_jepa_hidden_dim', type=int, default=512,
                       help='Predictor hidden dimension.')
    group.add_argument('--cross_tick_jepa_predictor_depth', type=int, default=2,
                       help='Number of predictor MLP layers.')
    group.add_argument('--cross_tick_jepa_dropout', type=float, default=0.1,
                       help='Predictor dropout.')
    group.add_argument('--cross_tick_jepa_target_stop_grad',
                       action=argparse.BooleanOptionalAction, default=True,
                       help='Stop gradient on target synch.')

    g2 = parser.add_argument_group('Adaptive JEPA weight')
    g2.add_argument('--jepa_weight_mode', type=str, default='fixed',
                    choices=['fixed', 'balance', 'gate', 'uncertainty'],
                    help='fixed=static weight (legacy); balance=A loss-magnitude '
                         'balancing; gate=B acc-gated sigmoid; uncertainty=C '
                         'learnable sigma (Kendall 2018). When != fixed, '
                         'cross_tick_jepa_weight still must be >0 to enable JEPA.')
    g2.add_argument('--jepa_balance_ratio', type=float, default=0.3,
                    help='[balance] target JEPA-to-main loss ratio '
                         '(effective weight = ratio*L_main/L_jepa).')
    g2.add_argument('--jepa_balance_clamp', type=float, nargs=2,
                    default=(0.0, 10.0), metavar=('LO', 'HI'),
                    help='[balance] clamp range for the adaptive weight.')
    g2.add_argument('--jepa_gate_threshold', type=float, default=0.7,
                    help='[gate] acc EMA at which JEPA opens to half-strength.')
    g2.add_argument('--jepa_gate_temp', type=float, default=0.05,
                    help='[gate] sigmoid temperature (smaller=sharper).')
    g2.add_argument('--jepa_gate_ema', type=float, default=0.99,
                    help='[gate] EMA decay for the running accuracy.')
    g2.add_argument('--jepa_log_sigma_init', type=float, default=0.0,
                    help='[uncertainty] initial log(sigma^2) (0 => weight=1).')
    return parser


def build_jepa_predictor(synch_dim, args):
    """Create a JEPA predictor / adaptive controller from parsed args.

    Returns None when JEPA is disabled (weight <= 0). For
    ``jepa_weight_mode != 'fixed'`` an AdaptiveJEPAController is returned
    (wraps the predictor + adaptive weight logic); its parameters are
    collected automatically when attached to the model before optimizer
    construction.
    """
    if float(args.cross_tick_jepa_weight) <= 0:
        return None
    mode = str(getattr(args, 'jepa_weight_mode', 'fixed'))
    if mode != 'fixed':
        return AdaptiveJEPAController(synch_dim, args)
    return CrossTickJEPAPredictor(
        synch_dim,
        hidden_dim=int(args.cross_tick_jepa_hidden_dim),
        depth=int(args.cross_tick_jepa_predictor_depth),
        dropout=float(args.cross_tick_jepa_dropout),
    )


def compute_jepa_loss(predictor, synch_per_tick, weight, loss_type='cosine',
                      target_stop_grad=True, main_loss=None, acc=None):
    """Compute cross-tick JEPA loss from per-tick synch representations.

    Backward compatible: if ``predictor`` is an AdaptiveJEPAController the
    adaptive scheme chooses the effective weight (the static ``weight``,
    ``loss_type``, ``target_stop_grad`` args are then ignored). Otherwise the
    legacy static-weight path is used.

    Args:
        predictor: CrossTickJEPAPredictor or AdaptiveJEPAController.
        synch_per_tick: (B, synch_dim, num_ticks) tensor.
        weight: static weight (used only for the legacy/fixed path).
        loss_type: 'cosine' or 'mse'.
        target_stop_grad: detach target synch.
        main_loss: detached scalar of the main task loss (used by 'balance').
        acc: current-step main task accuracy as float (used by 'gate').

    Returns:
        jepa_loss: scalar tensor (0 if no adjacent pairs).
    """
    if hasattr(predictor, 'compute_loss'):
        return predictor.compute_loss(synch_per_tick, main_loss, acc)
    raw = _raw_jepa_loss(predictor, synch_per_tick, loss_type, target_stop_grad)
    return raw * weight


def update_jepa_gate_acc(predictor, acc):
    """Update the gate controller's running accuracy; no-op for other modes.

    Call once per train step after computing the batch accuracy, so the next
    compute_jepa_loss call sees an up-to-date main-task-progress signal.
    Safe to call with predictor=None or a plain CrossTickJEPAPredictor.
    """
    if predictor is not None and hasattr(predictor, 'update_gate_acc'):
        predictor.update_gate_acc(acc)
