import torch

from model.config import CTMLLMConfig


def residual_enabled(config: CTMLLMConfig):
    return (
        config.residual_compute_mode != 'none'
        or bool(int(getattr(config, 'residual_track_deltas', 0)))
        or float(getattr(config, 'residual_delta_l1_weight', 0.0)) > 0
        or float(getattr(config, 'residual_compute_weight', 0.0)) > 0
    )


def compute_residual_metrics(tick_outs, final_hidden, config: CTMLLMConfig):
    """Track tick-to-tick deltas; optional L1 regularizer for residual-compute runs."""
    zero = tick_outs.new_zeros(())
    if tick_outs is None or tick_outs.size(-1) < 2:
        return zero, 0.0

    tick_deltas = tick_outs[..., 1:] - tick_outs[..., :-1]
    tick_delta_l1 = tick_deltas.abs().mean()
    hidden_delta_l1 = zero
    if final_hidden is not None and tick_outs.size(-1) >= 1:
        hidden_delta_l1 = (final_hidden - tick_outs[..., -1]).abs().mean()

    delta_l1 = 0.5 * (tick_delta_l1 + hidden_delta_l1)
    metric = float(delta_l1.detach().float().item())

    penalty = zero
    delta_weight = float(config.residual_delta_l1_weight)
    if delta_weight > 0:
        penalty = penalty + delta_weight * delta_l1

    compute_weight = float(config.residual_compute_weight)
    if compute_weight > 0 and config.residual_compute_mode not in ('none', 'observe'):
        penalty = penalty + compute_weight * delta_l1

    return penalty, metric
