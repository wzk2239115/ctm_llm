import torch

from model.config import CTMLLMConfig
from model.tick_controller import tick_controller_enabled


def nlm_recursive_enabled(config: CTMLLMConfig) -> bool:
    if config.residual_nlm_mode not in (
        'recursive_fast', 'hybrid_fast_full', 'output_delta',
    ):
        return False
    return config.residual_compute_mode in (
        'nlm_recursive', 'tick_controller', 'speed_cells',
    )


def apply_recursive_nlm(trace_module, state_trace, cache, config: CTMLLMConfig, tick_idx: int):
    """Recursive NLM fast path with periodic full-history refresh."""
    if not nlm_recursive_enabled(config):
        activated = trace_module(state_trace)
        return activated, cache, 0.0

    mem_len = state_trace.size(-1)
    refresh = max(1, int(config.residual_full_refresh_interval))
    if tick_controller_enabled(config):
        tick_exec_mode = getattr(config, '_tick_exec_mode', 'full')
        if tick_exec_mode == 'stop' and cache is not None:
            carried = cache['activated'].to(dtype=state_trace.dtype, device=state_trace.device)
            slot_delta = state_trace[..., -1] - cache['trace'][..., -1].to(state_trace.dtype)
            activated = carried + slot_delta
            new_cache = {
                'activated': activated.detach(),
                'trace': state_trace.detach(),
            }
            return activated, new_cache, 1.0
        force_full = (
            tick_exec_mode == 'full'
            or cache is None
            or (tick_idx + 1) % refresh == 0
        )
    else:
        force_full = cache is None or (tick_idx + 1) % refresh == 0

    if force_full:
        activated = trace_module(state_trace)
        new_cache = {
            'activated': activated.detach(),
            'trace': state_trace.detach(),
        }
        return activated, new_cache, 0.0

    carried = cache['activated'].to(dtype=state_trace.dtype, device=state_trace.device)
    slot_delta = state_trace[..., -1] - cache['trace'][..., -1].to(state_trace.dtype)

    if config.residual_nlm_mode == 'hybrid_fast_full':
        tail_len = min(2, mem_len)
        prefix_len = mem_len - tail_len
        prefix = cache['trace'][..., :prefix_len].to(
            dtype=state_trace.dtype, device=state_trace.device)
        window = torch.cat([prefix.detach(), state_trace[..., -tail_len:]], dim=-1)
        fast = trace_module(window)
        activated = 0.5 * fast + 0.5 * (carried + slot_delta)
    else:
        # recursive_fast / output_delta: skip full-history NLM, carry activation forward.
        activated = carried + slot_delta

    new_cache = {
        'activated': activated.detach(),
        'trace': state_trace.detach(),
    }
    return activated, new_cache, 1.0
