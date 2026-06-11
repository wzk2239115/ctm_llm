"""Cross-tick JEPA utilities for CTM baseline tasks.

Each tick produces a synchronisation representation (synch_out).
tick_{i+1}'s synch is predicted from tick_i's synch via a lightweight
predictor. Cosine/MSE loss + stop-gradient on target prevents collapse.
"""

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


def add_jepa_args(parser):
    """Add cross-tick JEPA CLI arguments to an argument parser."""
    group = parser.add_argument_group('Cross-Tick JEPA')
    group.add_argument('--cross_tick_jepa_weight', type=float, default=0.0,
                       help='Weight for cross-tick JEPA loss (0 = disabled).')
    group.add_argument('--cross_tick_jepa_loss', type=str, default='cosine',
                       choices=['cosine', 'mse'],
                       help='JEPA loss type.')
    group.add_argument('--cross_tick_jepa_hidden_dim', type=int, default=512,
                       help='Predictor hidden dimension.')
    group.add_argument('--cross_tick_jepa_predictor_depth', type=int, default=2,
                       help='Number of predictor MLP layers.')
    group.add_argument('--cross_tick_jepa_dropout', type=float, default=0.1,
                       help='Predictor dropout.')
    group.add_argument('--cross_tick_jepa_target_stop_grad', action='store_true', default=True,
                       help='Stop gradient on target synch.')
    return parser


def build_jepa_predictor(synch_dim, args):
    """Create a CrossTickJEPAPredictor from parsed args."""
    if args.cross_tick_jepa_weight <= 0:
        return None
    return CrossTickJEPAPredictor(
        synch_dim,
        hidden_dim=int(args.cross_tick_jepa_hidden_dim),
        depth=int(args.cross_tick_jepa_predictor_depth),
        dropout=float(args.cross_tick_jepa_dropout),
    )


def compute_jepa_loss(predictor, synch_per_tick, weight, loss_type='cosine',
                      target_stop_grad=True):
    """Compute cross-tick JEPA loss from per-tick synch representations.

    Args:
        predictor: CrossTickJEPAPredictor instance.
        synch_per_tick: (B, synch_dim, num_ticks) tensor.
        weight: JEPA loss weight.
        loss_type: 'cosine' or 'mse'.
        target_stop_grad: detach target synch.

    Returns:
        jepa_loss: scalar tensor (0 if no adjacent pairs).
    """
    num_ticks = synch_per_tick.size(-1)
    if num_ticks < 2 or predictor is None or weight <= 0:
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

    return (total / count) * weight if count > 0 else synch_per_tick.new_zeros(())
