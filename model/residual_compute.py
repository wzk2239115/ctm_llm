import math

import torch

from model.config import CTMLLMConfig


def residual_enabled(config: CTMLLMConfig):
    return (
        config.residual_compute_mode != 'none'
        or bool(int(getattr(config, 'residual_track_deltas', 0)))
        or float(getattr(config, 'residual_delta_l1_weight', 0.0)) > 0
        or float(getattr(config, 'residual_compute_weight', 0.0)) > 0
    )


def block_skip_enabled(config: CTMLLMConfig):
    if config.residual_synapse_mode not in ('block_delta_skip', 'sparse_delta'):
        return False
    return config.residual_compute_mode in (
        'block_skip', 'gate', 'skip', 'nlm_recursive',
    )


def _novelty_score(pre_syn, cached_pre):
    if cached_pre is None:
        return pre_syn.new_tensor(float('inf'))
    return (pre_syn - cached_pre).abs().mean()


def _force_full_refresh(config: CTMLLMConfig, tick_idx: int) -> bool:
    refresh = max(0, int(config.residual_full_refresh_interval))
    if refresh <= 0:
        return False
    return (tick_idx + 1) % refresh == 0


def plan_block_skip_keys(blocks, config: CTMLLMConfig, tick_idx: int):
    """Choose which synapse blocks run a full forward pass this tick."""
    if not blocks:
        return set()
    if _force_full_refresh(config, tick_idx):
        return {block['key'] for block in blocks}

    threshold = float(config.residual_gate_threshold)
    active_ratio = float(config.residual_active_ratio)
    max_active = max(1, int(math.ceil(active_ratio * len(blocks))))

    ranked = sorted(
        blocks, key=lambda item: float(item['novelty'].detach().item()), reverse=True)
    run_keys = set()
    for block in ranked:
        if len(run_keys) >= max_active:
            break
        if float(block['novelty']) >= threshold or len(run_keys) < max(1, max_active // 2):
            run_keys.add(block['key'])
    if not run_keys:
        run_keys.add(ranked[0]['key'])
    return run_keys


def run_block_delta_synapse(pre_syn, synapse_module, cache_entry, should_run_full: bool):
    """Conditional synapse dispatch: full synapse or reuse cached state without synapse forward."""
    if should_run_full or cache_entry is None:
        state = synapse_module(pre_syn)
        skipped = False
    else:
        state = cache_entry['state'].to(dtype=pre_syn.dtype, device=pre_syn.device)
        skipped = True
    new_cache = {
        'pre_syn': pre_syn.detach(),
        'state': state.detach(),
    }
    return state, new_cache, skipped


def run_grouped_block_delta_synapse(pre_syn, synapse_module, cache, config, tick_idx, group_prefix):
    """Sequence-chunked block skip for dense synapse paths."""
    if not block_skip_enabled(config):
        state = synapse_module(pre_syn)
        return state, cache, 0.0

    B, T, _ = pre_syn.shape
    num_groups = max(1, int(config.residual_num_groups))
    group_size = max(1, int(math.ceil(T / float(num_groups))))

    blocks = []
    for group_idx in range(num_groups):
        start = group_idx * group_size
        end = min(T, start + group_size)
        if start >= end:
            break
        key = f'{group_prefix}:g{group_idx}'
        chunk_pre = pre_syn[:, start:end, :]
        cache_entry = cache.get(key)
        blocks.append({
            'key': key,
            'start': start,
            'end': end,
            'pre_syn': chunk_pre,
            'cache_entry': cache_entry,
            'novelty': _novelty_score(chunk_pre, None if cache_entry is None else cache_entry['pre_syn']),
        })

    run_keys = plan_block_skip_keys(blocks, config, tick_idx)
    outputs = []
    skipped = 0
    total = 0
    merged = None
    for block in blocks:
        total += 1
        state, new_cache, did_skip = run_block_delta_synapse(
            block['pre_syn'],
            synapse_module,
            block['cache_entry'],
            block['key'] in run_keys,
        )
        cache[block['key']] = new_cache
        if did_skip:
            skipped += 1
        if merged is None:
            merged = state.new_zeros(pre_syn.size(0), pre_syn.size(1), state.size(-1))
        outputs.append((block['start'], block['end'], state))

    for start, end, state in outputs:
        merged[:, start:end, :] = state
    skip_ratio = skipped / max(total, 1)
    return merged, cache, skip_ratio


def compute_residual_metrics(
    tick_outs,
    final_hidden,
    config: CTMLLMConfig,
    skip_ratio=0.0,
    nlm_fast_ratio=0.0,
):
    """Track tick deltas and optional regularizers for residual-compute runs."""
    zero = tick_outs.new_zeros(()) if tick_outs is not None else final_hidden.new_zeros(())
    metric = float(skip_ratio)
    delta_l1 = zero

    if tick_outs is not None and tick_outs.size(-1) >= 2:
        tick_deltas = tick_outs[..., 1:] - tick_outs[..., :-1]
        tick_delta_l1 = tick_deltas.abs().mean()
        hidden_delta_l1 = zero
        if final_hidden is not None and tick_outs.size(-1) >= 1:
            hidden_delta_l1 = (final_hidden - tick_outs[..., -1]).abs().mean()
        delta_l1 = 0.5 * (tick_delta_l1 + hidden_delta_l1)
        metric = 0.5 * (
            float(delta_l1.detach().float().item())
            + float(skip_ratio)
            + float(nlm_fast_ratio)
        )

    penalty = zero
    delta_weight = float(config.residual_delta_l1_weight)
    if delta_weight > 0 and tick_outs is not None:
        penalty = penalty + delta_weight * delta_l1

    compute_weight = float(config.residual_compute_weight)
    if compute_weight > 0 and config.residual_compute_mode not in ('none', 'observe'):
        ref = tick_outs if tick_outs is not None else final_hidden
        if ref is None:
            return penalty, metric
        executed_ratio = max(
            0.0,
            1.0 - 0.5 * float(skip_ratio) - 0.5 * float(nlm_fast_ratio),
        )
        if block_skip_enabled(config) or config.residual_compute_mode == 'nlm_recursive':
            penalty = penalty + compute_weight * ref.new_tensor(executed_ratio)
        else:
            penalty = penalty + compute_weight * delta_l1

    return penalty, metric
