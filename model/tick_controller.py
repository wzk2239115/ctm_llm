from model.config import CTMLLMConfig


def tick_controller_enabled(config: CTMLLMConfig) -> bool:
    return config.residual_compute_mode in ('tick_controller', 'speed_cells')


def resolve_tick_exec_mode(
    config: CTMLLMConfig,
    tick_idx: int,
    num_iters: int,
    prev_confidence=None,
):
    """Return one of: full, residual, stop."""
    refresh = max(1, int(config.residual_full_refresh_interval))

    if config.residual_compute_mode == 'speed_cells':
        cells = config.residual_speed_cells
        if cells == 'event_teacher':
            if (tick_idx + 1) % refresh == 0:
                return 'full'
            return 'residual'
        if cells == 'fast_mid_slow':
            third = max(1, num_iters // 3)
            if tick_idx < third:
                return 'full'
            return 'residual'
        return 'full'

    if config.residual_compute_mode != 'tick_controller':
        return 'full'

    controller = config.residual_tick_controller
    threshold = float(config.residual_gate_threshold)

    if controller in ('threshold', 'learned') and prev_confidence is not None:
        stop_threshold = threshold + (0.10 if controller == 'learned' else 0.0)
        if tick_idx > 0 and float(prev_confidence) >= stop_threshold:
            return 'stop'

    if (tick_idx + 1) % refresh == 0:
        return 'full'
    return 'residual'
