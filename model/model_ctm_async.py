"""Async-first CTM backend.

Each clock band owns its own local tick counter, cell slice, trace carry rules,
and output projection. Fast bands (period=1) fire every global tick and anchor
guaranteed emission; slow bands keep their last projection in the mix while they
wait for their next fire.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.config import CTMLLMConfig
from model.building_blocks import RMSNorm, FeedForward, _parse_int_list, BlockOutput, ModelOutput
from model.ctm_modules import SuperLinear, SynapseUNET, Squeeze, TTTMLP
from model.moe_regional import RegionalMoEMixin
from model.base_causal_lm import BaseCTMForCausalLM


class AsyncClockBand:
    """One asynchronous oscillator lane inside a CTM block."""

    __slots__ = (
        'band_id', 'period', 'phase', 'd_start', 'd_end',
        'expert_start', 'expert_end', 'use_expert_layout',
        'local_tick', 'last_tick_out', 'stale_weight',
    )

    def __init__(
        self,
        band_id,
        period,
        phase,
        d_start,
        d_end,
        *,
        expert_start=0,
        expert_end=0,
        use_expert_layout=False,
        stale_weight=0.35,
    ):
        self.band_id = band_id
        self.period = max(1, int(period))
        self.phase = int(phase)
        self.d_start = d_start
        self.d_end = d_end
        self.expert_start = expert_start
        self.expert_end = expert_end
        self.use_expert_layout = use_expert_layout
        self.local_tick = 0
        self.last_tick_out = None
        self.stale_weight = float(stale_weight)

    def fires(self, global_tick):
        return (global_tick + self.phase) % self.period == 0

    @property
    def is_fast(self):
        return self.period == 1

    def reset_runtime(self):
        self.local_tick = 0
        self.last_tick_out = None

    def mask_activated(self, activated):
        masked = torch.zeros_like(activated)
        masked[:, :, self.d_start:self.d_end] = activated[:, :, self.d_start:self.d_end]
        return masked

    def emission_weight(self, global_tick):
        if self.is_fast or self.fires(global_tick):
            return 1.0
        return self.stale_weight


class AsyncClockEngine(nn.Module):
    """Drives per-band clocks, carry, and fused emission inside AsyncCTMBlock."""

    def __init__(self, block, config):
        super().__init__()
        periods = _parse_int_list(config.async_tick_periods)
        phases = _parse_int_list(config.async_tick_phases)
        if not periods:
            periods = [1]
        self.num_bands = len(periods)
        self.fast_band = min(
            max(0, int(config.async_fast_band)),
            self.num_bands - 1)
        stale = float(config.async_stale_band_weight)
        while len(phases) < self.num_bands:
            phases.append(0)
        phases = phases[:self.num_bands]

        self.bands = []
        if block._async_expert_layout():
            per_band = max(1, math.ceil(block.moe_num_experts / self.num_bands))
            for band_id, period in enumerate(periods):
                expert_start = band_id * per_band
                expert_end = (
                    block.moe_num_experts
                    if band_id == self.num_bands - 1
                    else min((band_id + 1) * per_band, block.moe_num_experts))
                d_start = expert_start * block.moe_expert_size
                d_end = expert_end * block.moe_expert_size
                self.bands.append(AsyncClockBand(
                    band_id, period, phases[band_id], d_start, d_end,
                    expert_start=expert_start, expert_end=expert_end,
                    use_expert_layout=True, stale_weight=stale))
        else:
            per_band = max(1, math.ceil(block.d_model / self.num_bands))
            for band_id, period in enumerate(periods):
                d_start = band_id * per_band
                d_end = block.d_model if band_id == self.num_bands - 1 else (band_id + 1) * per_band
                self.bands.append(AsyncClockBand(
                    band_id, period, phases[band_id], d_start, d_end,
                    stale_weight=stale))

        self.band_output_projs = nn.ModuleList([
            nn.Linear(block.synch_repr_out, config.hidden_size, bias=False)
            for _ in range(self.num_bands)
        ])
        self.fuse_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.global_output_proj = block.output_proj
        self.fast_output_weight = float(config.async_fast_output_weight)

    def reset_runtime(self):
        for band in self.bands:
            band.reset_runtime()

    def bands_fired_count(self, global_tick):
        return sum(1 for band in self.bands if band.fires(global_tick))

    def band_for_expert(self, expert):
        for band in self.bands:
            if band.use_expert_layout and band.expert_start <= expert < band.expert_end:
                return band
        return self.bands[-1]

    def apply_carry(self, activated, activated_prev, state_trace, trace_prev, global_tick):
        for band in self.bands:
            if band.fires(global_tick):
                continue
            mask_a = torch.zeros(activated.shape[-1], device=activated.device, dtype=torch.bool)
            mask_a[band.d_start:band.d_end] = True
            activated = torch.where(mask_a.view(1, 1, -1), activated_prev, activated)
            mask_t = torch.zeros(state_trace.shape[-2], device=state_trace.device, dtype=torch.bool)
            mask_t[band.d_start:band.d_end] = True
            state_trace = torch.where(mask_t.view(1, 1, -1, 1), trace_prev, state_trace)
        return activated, state_trace

    def emit_tick_output(self, block, activated, alpha_o, beta_o, r_o, global_tick):
        weighted = []
        weights = []
        for band, proj in zip(self.bands, self.band_output_projs):
            masked = band.mask_activated(activated)
            sync_o, _, _ = block._compute_synch(
                masked, alpha_o, beta_o, r_o, block.out_left, block.out_right)
            band_out = proj(sync_o)
            band.last_tick_out = band_out
            w = band.emission_weight(global_tick)
            weighted.append(band_out * w)
            weights.append(w)

        total_w = max(sum(weights), 1e-6)
        tick_out = torch.stack(weighted, dim=0).sum(dim=0) / total_w
        tick_out = self.fuse_norm(tick_out)

        fast_w = self.fast_output_weight
        if fast_w > 0:
            fast_band = self.bands[self.fast_band]
            if fast_band.last_tick_out is not None:
                tick_out = tick_out * (1.0 - fast_w) + fast_band.last_tick_out * fast_w
        return tick_out

    def local_tick_vector(self):
        return [band.local_tick for band in self.bands]


class AsyncCTMBlock(RegionalMoEMixin, nn.Module):
    def __init__(self, layer_id: int, config: CTMLLMConfig):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.d_model = config.d_model
        self.d_input = config.d_input
        self.iterations = config.iterations
        self.memory_length = config.memory_length
        self.heads = config.heads
        self.head_dim = config.d_input // config.heads
        self.neuron_select_type = config.neuron_select_type
        self.self_cond = config.self_cond
        self.cell_sparsity_mode = config.cell_sparsity_mode
        self.cell_topk = min(max(1, int(config.cell_topk)), config.d_model)
        self.cell_sparsity_rescale = bool(config.cell_sparsity_rescale)
        self.moe_routing_mode = config.moe_routing_mode
        self.moe_num_experts = max(1, int(config.moe_num_experts))
        self.moe_topk_experts = max(1, int(config.moe_topk_experts))
        self.moe_shared_experts = max(0, int(config.moe_shared_experts))
        self.moe_expert_size = int(config.moe_expert_size)
        if self.moe_expert_size <= 0 and self.moe_num_experts > 0:
            self.moe_expert_size = config.d_model // self.moe_num_experts
        self.moe_expert_dropout = float(config.moe_expert_dropout)
        self.moe_activation_passes = max(1, int(
            config.moe_activation_passes))
        self.moe_region_diversity_weight = float(
            config.moe_region_diversity_weight)
        self.moe_load_balance_weight = float(
            config.moe_load_balance_weight)
        self.moe_router_entropy_weight = float(
            config.moe_router_entropy_weight)
        self.moe_router_z_loss_weight = float(
            config.moe_router_z_loss_weight)
        self.moe_dispatch_mode = config.moe_dispatch_mode
        self.moe_capacity_factor = float(config.moe_capacity_factor)
        self.moe_drop_tokens = bool(config.moe_drop_tokens)
        self.moe_aux_loss_free_bias = bool(
            config.moe_aux_loss_free_bias)
        self.moe_aux_loss = None
        self.last_executed_ticks = 0
        self.last_async_bands_fired = 0
        self.last_async_local_ticks = []
        self.diff_cell_mode = config.diff_cell_mode
        self.diff_cell_temperature = float(
            config.diff_cell_temperature)
        self.diff_cell_capacity_weight = float(
            config.diff_cell_capacity_weight)
        self.diff_cell_memory_weight = float(
            config.diff_cell_memory_weight)
        self.diff_cell_diversity_weight = float(
            config.diff_cell_diversity_weight)
        self.diff_cell_widths = _parse_int_list(
            config.diff_cell_widths, max_value=self.moe_expert_size)
        self.diff_cell_memory_lengths = _parse_int_list(
            config.diff_cell_memory_lengths, max_value=config.memory_length)
        if self.moe_expert_size > 0 and self.moe_expert_size not in self.diff_cell_widths:
            self.diff_cell_widths.append(self.moe_expert_size)
            self.diff_cell_widths = sorted(set(self.diff_cell_widths))
        if config.memory_length not in self.diff_cell_memory_lengths:
            self.diff_cell_memory_lengths.append(config.memory_length)
            self.diff_cell_memory_lengths = sorted(set(self.diff_cell_memory_lengths))

        self.input_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.kv_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.d_input, bias=False),
            nn.LayerNorm(config.d_input)
        )

        self._calc_sizes(config)
        self.q_proj = nn.Linear(self.synch_repr_action, config.d_input, bias=False)
        self.o_proj = nn.Linear(config.d_input, config.d_input, bias=False)
        self.attn_drop = nn.Dropout(config.dropout)

        synapse_in = config.d_input + config.d_model
        if config.self_cond:
            synapse_in += config.d_model
            self.self_cond_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        if config.synapse_depth == 1:
            self.synapses = nn.Sequential(
                nn.Dropout(config.dropout),
                nn.Linear(synapse_in, config.d_model * 2),
                nn.GLU(),
                nn.LayerNorm(config.d_model)
            )
        else:
            self.synapses = SynapseUNET(
                synapse_in, config.d_model, config.synapse_depth, dropout=config.dropout)

        self.trace_processor = self._build_nlms(config)
        self.group_synapses = None
        self.group_trace_processors = None
        self.diff_group_synapses = None
        self.diff_group_trace_processors = None
        self.diff_width_logits = None
        self.diff_memory_logits = None
        if self._can_build_group_sparse_backend():
            self.group_synapses = nn.ModuleList([
                self._build_group_synapse(config, self.moe_expert_size)
                for _ in range(self.moe_num_experts)
            ])
            self.group_trace_processors = nn.ModuleList([
                self._build_nlms_for_size(config, self.moe_expert_size)
                for _ in range(self.moe_num_experts)
            ])
            if self._differentiated_cells_enabled():
                self._init_differentiated_cells(config)
        self.ttt_layer = None
        if config.ttt_layer:
            self.ttt_layer = TTTMLP(
                config.d_model,
                hidden_mult=config.ttt_hidden_mult,
                gate_init=config.ttt_gate_init,
                dropout=config.dropout,
            )

        self.start_activated_state = nn.Parameter(
            torch.zeros(config.d_model).uniform_(
                -math.sqrt(1 / config.d_model), math.sqrt(1 / config.d_model)))
        self.start_trace = nn.Parameter(
            torch.zeros(config.d_model, config.memory_length).uniform_(
                -math.sqrt(1 / (config.d_model + config.memory_length)),
                math.sqrt(1 / (config.d_model + config.memory_length))))

        self._init_synch(config)

        self.output_proj = nn.Linear(self.synch_repr_out, config.hidden_size, bias=False)
        self.clock_engine = AsyncClockEngine(self, config)

        self.post_ctm_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config.hidden_size)
        self.resid_drop = nn.Dropout(config.dropout)

    def _async_expert_layout(self):
        return (
            self.moe_num_experts > 1
            and self.moe_expert_size > 0
            and self.moe_expert_size * self.moe_num_experts == self.d_model
        )

    def _calc_sizes(self, config):
        if config.neuron_select_type == 'random-pairing':
            self.synch_repr_action = config.n_synch_action
            self.synch_repr_out = config.n_synch_out
        else:
            self.synch_repr_action = (config.n_synch_action * (config.n_synch_action + 1)) // 2
            self.synch_repr_out = (config.n_synch_out * (config.n_synch_out + 1)) // 2

    def _build_nlms(self, config):
        return self._build_nlms_for_size(config, config.d_model)

    def _build_nlms_for_size(self, config, d_model, memory_length=None):
        memory_length = memory_length or config.memory_length
        if config.deep_nlms:
            return nn.Sequential(
                SuperLinear(memory_length, 2 * config.memory_hidden_dims,
                            d_model, dropout=config.dropout),
                nn.GLU(),
                SuperLinear(config.memory_hidden_dims, 2,
                            d_model, dropout=config.dropout),
                nn.GLU(),
                Squeeze(-1))
        return nn.Sequential(
            SuperLinear(memory_length, 2, d_model, dropout=config.dropout),
            nn.GLU(),
            Squeeze(-1))

    def _can_build_group_sparse_backend(self):
        if self.moe_routing_mode not in ('regional_topk', 'regional_shared_topk'):
            return False
        if self.moe_dispatch_mode == 'dense_mask':
            return False
        if self.moe_num_experts <= 1 or self.moe_expert_size <= 0:
            return False
        return self.moe_expert_size * self.moe_num_experts == self.d_model

    def _build_group_synapse(self, config, expert_size):
        synapse_in = config.d_input + expert_size
        if config.self_cond:
            synapse_in += expert_size
        if config.synapse_depth == 1:
            return nn.Sequential(
                nn.Dropout(config.dropout),
                nn.Linear(synapse_in, expert_size * 2),
                nn.GLU(),
                nn.LayerNorm(expert_size)
            )
        return SynapseUNET(
            synapse_in, expert_size, config.synapse_depth, dropout=config.dropout)

    def _differentiated_cells_enabled(self):
        return (
            self.diff_cell_mode == 'learned'
            and self.moe_num_experts > 1
            and self.moe_expert_size > 0
            and len(self.diff_cell_widths) > 1
            and len(self.diff_cell_memory_lengths) >= 1
        )

    def _init_differentiated_cells(self, config):
        self.diff_width_logits = nn.Parameter(
            torch.zeros(self.moe_num_experts, len(self.diff_cell_widths)))
        self.diff_memory_logits = nn.Parameter(
            torch.zeros(self.moe_num_experts, len(self.diff_cell_memory_lengths)))
        if len(self.diff_cell_widths) > 1:
            with torch.no_grad():
                self.diff_width_logits[:, 0] = 0.5
        if len(self.diff_cell_memory_lengths) > 1:
            with torch.no_grad():
                self.diff_memory_logits[:, 0] = 0.5

        self.diff_group_synapses = nn.ModuleList()
        self.diff_group_trace_processors = nn.ModuleList()
        for _ in range(self.moe_num_experts):
            synapses = nn.ModuleDict()
            traces = nn.ModuleDict()
            for width in self.diff_cell_widths:
                synapses[str(width)] = self._build_group_synapse(config, width)
                for mem_len in self.diff_cell_memory_lengths:
                    traces[f"{width}x{mem_len}"] = self._build_nlms_for_size(
                        config, width, memory_length=mem_len)
            self.diff_group_synapses.append(synapses)
            self.diff_group_trace_processors.append(traces)

    def _select_diff_level(self, logits, values, expert):
        tau = max(self.diff_cell_temperature, 1e-4)
        if self.training:
            gate = F.gumbel_softmax(logits[expert].float(), tau=tau, hard=True)
        else:
            idx = logits[expert].argmax(dim=-1)
            gate = F.one_hot(idx, num_classes=len(values)).float()
        idx = int(gate.argmax(dim=-1).item())
        value = values[idx]
        # Forward is 1.0; backward carries the straight-through gate gradient.
        scale = gate[idx].type_as(logits) / gate[idx].detach().clamp(min=1e-6).type_as(logits)
        return value, scale

    def _update_diff_cell_aux_loss(self):
        if not self._differentiated_cells_enabled():
            return
        if (
            self.diff_cell_capacity_weight <= 0
            and self.diff_cell_memory_weight <= 0
            and self.diff_cell_diversity_weight <= 0
        ):
            return
        aux = self.diff_width_logits.new_zeros(())
        width_probs = F.softmax(self.diff_width_logits.float(), dim=-1)
        mem_probs = F.softmax(self.diff_memory_logits.float(), dim=-1)
        width_values = torch.tensor(
            self.diff_cell_widths, device=width_probs.device, dtype=width_probs.dtype)
        mem_values = torch.tensor(
            self.diff_cell_memory_lengths, device=mem_probs.device, dtype=mem_probs.dtype)
        if self.diff_cell_capacity_weight > 0:
            expected_width = (width_probs * width_values.view(1, -1)).sum(dim=-1)
            aux = aux + self.diff_cell_capacity_weight * (
                expected_width / max(1, self.moe_expert_size)).mean()
        if self.diff_cell_memory_weight > 0:
            expected_mem = (mem_probs * mem_values.view(1, -1)).sum(dim=-1)
            aux = aux + self.diff_cell_memory_weight * (
                expected_mem / max(1, self.memory_length)).mean()
        if self.diff_cell_diversity_weight > 0:
            width_hist = width_probs.mean(dim=0)
            mem_hist = mem_probs.mean(dim=0)
            width_entropy = -(width_hist * torch.log(width_hist.clamp(min=1e-12))).sum()
            mem_entropy = -(mem_hist * torch.log(mem_hist.clamp(min=1e-12))).sum()
            norm = math.log(max(2, width_hist.numel())) + math.log(max(2, mem_hist.numel()))
            aux = aux - self.diff_cell_diversity_weight * (width_entropy + mem_entropy) / norm
        self.moe_aux_loss = aux if self.moe_aux_loss is None else self.moe_aux_loss + aux

    def _init_synch(self, config):
        d = config.d_model
        la, ra = self._make_pairs(d, config.n_synch_action, config.n_random_pairing_self)
        self.register_buffer('action_left', la)
        self.register_buffer('action_right', ra)
        self.register_parameter('decay_action',
                                nn.Parameter(torch.zeros(self.synch_repr_action)))

        lo, ro = self._make_pairs(d, config.n_synch_out, config.n_random_pairing_self)
        self.register_buffer('out_left', lo)
        self.register_buffer('out_right', ro)
        self.register_parameter('decay_out',
                                nn.Parameter(torch.zeros(self.synch_repr_out)))

    @staticmethod
    def _make_pairs(d_model, n_synch, n_self):
        left = torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch))
        right = torch.cat([
            left[:n_self],
            torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch - n_self))
        ])
        return left, right

    def _compute_synch(self, activated, alpha, beta, r, left, right):
        l = activated[:, :, left]
        r_sel = activated[:, :, right]

        if self.neuron_select_type in ('first-last', 'random'):
            outer = l.unsqueeze(3) * r_sel.unsqueeze(2)
            n = l.size(-1)
            i, j = torch.triu_indices(n, n, device=outer.device)
            pp = outer[:, :, i, j]
        else:
            pp = l * r_sel

        if alpha is None:
            alpha = pp
            beta = torch.ones_like(pp)
        else:
            alpha = r * alpha + pp
            beta = r * beta + 1
        return alpha / torch.sqrt(beta), alpha, beta

    def _apply_cell_sparsity(self, activated):
        if self.cell_sparsity_mode == 'none' or self.cell_topk >= self.d_model:
            return activated
        if self.moe_routing_mode != 'none' and self.moe_num_experts > 1:
            routed = self._apply_moe_group_sparsity(activated)
            if routed is not None:
                return routed
        if self.cell_sparsity_mode != 'topk':
            raise ValueError(f"Unknown cell_sparsity_mode: {self.cell_sparsity_mode}")

        scores = activated.detach().abs()
        idx = torch.topk(scores, self.cell_topk, dim=-1).indices
        mask = torch.zeros_like(activated).scatter_(-1, idx, 1.0)
        if self.cell_sparsity_rescale:
            mask = mask * (self.d_model / self.cell_topk)
        return activated * mask

    def _run_group_sparse_regional_tick(
        self, attn, activated, state_trace, prev_sync_o_activated, global_tick=0,
    ):
        B, T, _ = activated.shape
        expert_size = self.moe_expert_size
        num_experts = self.moe_num_experts
        shared = min(self.moe_shared_experts, num_experts)
        routed_count = max(num_experts - shared, 1)
        topk = self._effective_moe_topk(routed_count)
        max_distinct_passes = max(1, math.ceil(routed_count / max(1, topk)))
        passes = min(max(1, self.moe_activation_passes), max_distinct_passes)
        selected = torch.zeros(
            B, T, routed_count, dtype=torch.bool, device=activated.device)
        pass_masks = []

        routing_activated, _ = self._compute_dense_pre_mask_activation(
            attn, activated, state_trace, prev_sync_o_activated)
        routing_x = routing_activated.view(B, T, num_experts, expert_size)
        raw_scores = routing_x.abs().mean(dim=-1)
        routing_scores = raw_scores.detach()

        flat_attn = attn.reshape(B * T, self.d_input)
        base_flat_active = activated.reshape(B * T, self.d_model)
        base_flat_trace = state_trace.reshape(
            B * T, self.d_model, self.memory_length)
        flat_self = None
        if self.self_cond:
            if prev_sync_o_activated is None:
                flat_self = torch.zeros(
                    B * T, self.d_model, device=activated.device, dtype=activated.dtype)
            else:
                flat_self = self.self_cond_proj(
                    prev_sync_o_activated).reshape(B * T, self.d_model)

        pass_outputs = []
        merged_trace = base_flat_trace.clone()

        for _ in range(passes):
            expert_mask, _ = self._route_experts(
                routing_scores, shared, routed_count, topk, selected)
            selected = selected | expert_mask[:, :, shared:].bool()
            pass_masks.append(expert_mask[:, :, shared:])
            flat_mask = expert_mask.reshape(B * T, num_experts).bool()

            flat_active = base_flat_active.clone()
            flat_trace = base_flat_trace.clone()

            for expert in range(num_experts):
                band = self.clock_engine.band_for_expert(expert)
                if not band.fires(global_tick):
                    continue
                token_idx = flat_mask[:, expert].nonzero(as_tuple=False).squeeze(-1)
                if token_idx.numel() == 0:
                    continue
                start = expert * expert_size
                width = expert_size
                mem_len = self.memory_length
                gate_scale = flat_active.new_ones(())
                synapse_module = self.group_synapses[expert]
                trace_module = self.group_trace_processors[expert]
                if self._differentiated_cells_enabled():
                    width, width_scale = self._select_diff_level(
                        self.diff_width_logits, self.diff_cell_widths, expert)
                    mem_len, mem_scale = self._select_diff_level(
                        self.diff_memory_logits, self.diff_cell_memory_lengths, expert)
                    gate_scale = width_scale.to(flat_active.dtype) * mem_scale.to(flat_active.dtype)
                    synapse_module = self.diff_group_synapses[expert][str(width)]
                    trace_module = self.diff_group_trace_processors[expert][f"{width}x{mem_len}"]
                end = start + width
                parts = [
                    flat_attn[token_idx],
                    flat_active[token_idx, start:end],
                ]
                if self.self_cond:
                    parts.append(flat_self[token_idx, start:end])
                pre_syn = torch.cat(parts, dim=-1)
                state = synapse_module(pre_syn)
                prev_trace = flat_trace[token_idx, start:end, -mem_len:]
                new_trace = torch.cat(
                    [prev_trace[:, :, 1:], state.unsqueeze(-1)], dim=-1)
                group_active = trace_module(
                    new_trace.unsqueeze(1)).squeeze(1)
                flat_trace[token_idx, start:end, -mem_len:] = new_trace
                flat_active[token_idx, start:end] = group_active * gate_scale
                merged_trace[token_idx, start:end, -mem_len:] = new_trace

            pass_x = flat_active.view(B, T, num_experts, expert_size)
            active_cells = expert_mask.sum(dim=-1, keepdim=True).clamp(min=1) * expert_size
            mask = expert_mask.unsqueeze(-1).to(pass_x.dtype)
            if self.cell_sparsity_rescale:
                mask = mask * (self.d_model / active_cells.unsqueeze(-1))
            pass_outputs.append((pass_x * mask).reshape(B, T, self.d_model))

        self._update_moe_aux_loss(raw_scores[:, :, shared:])
        self._update_region_diversity_loss(pass_masks)
        self._update_diff_cell_aux_loss()
        activated = torch.stack(pass_outputs, dim=0).mean(dim=0)
        state_trace = merged_trace.view(B, T, self.d_model, self.memory_length)
        return activated, state_trace

    def forward(self, x, pos_emb=None, past_kv=None, use_cache=False, track=False,
                num_iters=None, return_all_ticks=False,
                prev_activated=None, prev_trace=None,
                halt_lm_head=None, enable_tick_halt=False):
        B, T, _ = x.shape
        device = x.device
        self.moe_aux_loss = None

        normed = self.input_norm(x)
        kv = self.kv_proj(normed)
        if pos_emb is not None:
            offset = past_kv[0].size(1) if past_kv is not None else 0
            kv = kv + pos_emb[offset:offset + T].unsqueeze(0)

        k = v = kv
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=1)
            v = torch.cat([past_kv[1], v], dim=1)
        present_kv = (k, v) if use_cache else None
        S = k.size(1)

        if prev_trace is not None:
            state_trace = prev_trace
        else:
            state_trace = self.start_trace.view(1, 1, self.d_model, self.memory_length) \
                .expand(B, T, -1, -1).contiguous()
        if prev_activated is not None:
            activated = prev_activated
        else:
            activated = self.start_activated_state.view(1, 1, self.d_model) \
                .expand(B, T, -1).contiguous()

        with torch.no_grad():
            self.decay_action.clamp_(0, 15)
            self.decay_out.clamp_(0, 15)
        r_a = torch.exp(-self.decay_action).view(1, 1, -1).expand(B, T, -1)
        r_o = torch.exp(-self.decay_out).view(1, 1, -1).expand(B, T, -1)

        alpha_a = beta_a = None
        _, alpha_o, beta_o = self._compute_synch(
            activated, None, None, r_o, self.out_left, self.out_right)

        tracking = None
        if track:
            tracking = {
                'pre_activations': [],
                'post_activations': [],
                'sync_action': [],
                'state_trace': [],
                'decay_action': r_a[0, 0].detach().cpu().numpy(),
                'decay_output': r_o[0, 0].detach().cpu().numpy(),
            }

        num_iters = num_iters if num_iters is not None else self.iterations
        all_tick_outs = [] if (track or return_all_ticks) else None
        prev_sync_o_activated = None
        last_sync_o = None
        self.last_executed_ticks = 0
        self.clock_engine.reset_runtime()

        for tick in range(num_iters):
            activated_prev = activated
            trace_prev = state_trace
            self.last_async_bands_fired = self.clock_engine.bands_fired_count(tick)

            sync_a, alpha_a, beta_a = self._compute_synch(
                activated, alpha_a, beta_a, r_a, self.action_left, self.action_right)

            q = self.q_proj(sync_a)
            q_mh = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
            k_mh = k.view(B, S, self.heads, self.head_dim).transpose(1, 2)
            v_mh = v.view(B, S, self.heads, self.head_dim).transpose(1, 2)

            if T > 1 and past_kv is None:
                attn_out = F.scaled_dot_product_attention(
                    q_mh, k_mh, v_mh, is_causal=True, attn_mask=None)
            else:
                attn_out = F.scaled_dot_product_attention(
                    q_mh, k_mh, v_mh)

            attn = self.attn_drop(
                self.o_proj(attn_out.transpose(1, 2).reshape(B, T, -1)))

            if self._use_group_sparse_backend():
                activated, state_trace = self._run_group_sparse_regional_tick(
                    attn, activated, state_trace, prev_sync_o_activated, global_tick=tick)
                state = activated
            else:
                pre_syn_parts = [attn, activated]
                if self.self_cond:
                    if prev_sync_o_activated is not None:
                        pre_syn_parts.append(self.self_cond_proj(prev_sync_o_activated))
                    else:
                        pre_syn_parts.append(torch.zeros(B, T, self.d_model, device=device))
                pre_syn = torch.cat(pre_syn_parts, dim=-1)
                state = self.synapses(pre_syn)

                state_trace = torch.cat(
                    [state_trace[:, :, :, 1:], state.unsqueeze(-1)], dim=-1)

                activated = self.trace_processor(state_trace)
            if self.ttt_layer is not None:
                activated = activated + self.ttt_layer(activated)
            if not self._use_group_sparse_backend():
                activated = self._apply_cell_sparsity(activated)
            activated, state_trace = self.clock_engine.apply_carry(
                activated, activated_prev, state_trace, trace_prev, tick)
            for band in self.clock_engine.bands:
                if band.fires(tick):
                    band.local_tick += 1

            sync_o, alpha_o, beta_o = self._compute_synch(
                activated, alpha_o, beta_o, r_o, self.out_left, self.out_right)
            last_sync_o = sync_o
            prev_sync_o_activated = activated
            tick_out = self.clock_engine.emit_tick_output(
                self, activated, alpha_o, beta_o, r_o, tick)
            self.last_executed_ticks = tick + 1
            self.last_async_local_ticks = self.clock_engine.local_tick_vector()
            if all_tick_outs is not None:
                all_tick_outs.append(tick_out)

            if track:
                tracking['pre_activations'].append(state[0].detach().cpu().numpy())
                tracking['post_activations'].append(activated[0].detach().cpu().numpy())
                tracking['sync_action'].append(sync_a[0].detach().cpu().numpy())
                tracking['state_trace'].append(state_trace[0].detach().cpu().numpy())

            if (
                enable_tick_halt
                and halt_lm_head is not None
                and self.config.tick_halt_mode != 'none'
                and tick + 1 < num_iters
            ):
                with torch.no_grad():
                    logits = halt_lm_head(tick_out.detach())
                    probs = F.softmax(logits, dim=-1)
                    entropy = -(probs * torch.log(probs.clamp(min=1e-12))).sum(-1)
                    confidence = 1 - entropy / math.log(logits.size(-1))
                    if confidence.mean() >= float(self.config.tick_halt_threshold):
                        break

        ctm_out = all_tick_outs[-1] if all_tick_outs is not None else \
            self.output_proj(last_sync_o)
        x = x + self.resid_drop(ctm_out)
        x = x + self.mlp(self.post_ctm_norm(x))

        extras = {}
        if all_tick_outs is not None:
            extras['tick_outputs'] = torch.stack(all_tick_outs, dim=-1)
        if track:
            for k_arr in ['pre_activations', 'post_activations', 'sync_action', 'state_trace']:
                tracking[k_arr] = np.array(tracking[k_arr])
            extras['tracking'] = tracking

        extras['final_activated'] = activated
        extras['final_trace'] = state_trace

        return BlockOutput(hidden=x, present_kv=present_kv, extras=extras)


class AsyncCTMModel(nn.Module):
    def __init__(self, config: CTMLLMConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embed = nn.Embedding(config.max_position_embeddings, config.d_input)
        self.drop = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [AsyncCTMBlock(i, config) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, past_key_values=None, use_cache=False, track=False,
                num_iters=None, return_all_ticks=False,
                halt_lm_head=None, enable_tick_halt=False):
        B, T = input_ids.shape
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        h = self.drop(self.embed_tokens(input_ids))
        pos_emb = self.pos_embed.weight
        presents = []
        tracking_all = {}
        last_tick_outs = None
        executed_ticks = []

        prev_activated = None
        prev_trace = None

        for layer, past_kv in zip(self.layers, past_key_values):
            is_last = layer.layer_id == len(self.layers) - 1
            layer_kwargs = dict(
                pos_emb=pos_emb, past_kv=past_kv,
                use_cache=use_cache, num_iters=num_iters,
                prev_activated=prev_activated if self.config.cross_layer_state else None,
                prev_trace=prev_trace if self.config.cross_layer_state else None,
                halt_lm_head=halt_lm_head if is_last else None,
                enable_tick_halt=enable_tick_halt and is_last,
            )

            if track and not is_last:
                result = layer(h, track=True, return_all_ticks=False, **layer_kwargs)
                h = result.hidden
                present = result.present_kv
                tracking_all[f'layer_{layer.layer_id}'] = result.extras['tracking']
            elif is_last and (track or return_all_ticks):
                result = layer(h, track=track, return_all_ticks=True, **layer_kwargs)
                h = result.hidden
                present = result.present_kv
                if track:
                    tracking_all[f'layer_{layer.layer_id}'] = result.extras.get('tracking', {})
                if return_all_ticks:
                    last_tick_outs = result.extras.get('tick_outputs')
            else:
                result = layer(h, **layer_kwargs)
                h = result.hidden
                present = result.present_kv
                extras = result.extras

            if self.config.cross_layer_state and isinstance(extras, dict):
                prev_activated = extras.get('final_activated', prev_activated)
                prev_trace = extras.get('final_trace', prev_trace)

            executed_ticks.append(getattr(layer, 'last_executed_ticks', 0))
            presents.append(present)

        h = self.norm(h)

        return ModelOutput(
            hidden=h,
            past_key_values=presents,
            tick_outputs=last_tick_outs if return_all_ticks else None,
            tracking=tracking_all if track else None,
            executed_ticks=executed_ticks if enable_tick_halt else None,
        )


class AsyncCTMForCausalLM(BaseCTMForCausalLM):
    def __init__(self, config: CTMLLMConfig):
        super().__init__()
        self.config = config
        self.model = AsyncCTMModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.reflex_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.reflex_adapter = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2, bias=False),
            nn.SiLU(),
            nn.Linear(config.hidden_size // 2, config.hidden_size, bias=False),
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, past_key_values=None, use_cache=False, labels=None,
                num_iters=None):
        enable_halt = self.config.tick_halt_mode != 'none'
        result = self.model(
            input_ids, past_key_values, use_cache, track=False, num_iters=num_iters,
            halt_lm_head=self.lm_head, enable_tick_halt=enable_halt)
        h, past_key_values = result.hidden, result.past_key_values
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            bs = self.config.block_size
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1), ignore_index=-100)
        return {'loss': loss, 'logits': logits, 'past_key_values': past_key_values}

    def forward_train(self, input_ids, labels, num_iters=None):
        B = input_ids.size(0)
        enable_halt = self.config.tick_halt_mode != 'none'
        result = self.model(
            input_ids, track=False, num_iters=num_iters, return_all_ticks=True,
            halt_lm_head=self.lm_head, enable_tick_halt=enable_halt)
        h = result.hidden
        tick_outs = result.tick_outputs
        num_ticks = tick_outs.size(-1)

        final_logits = self.lm_head(h)
        final_loss = self._lm_loss_from_logits(final_logits, labels)

        losses = []
        next_losses = []
        certainties = []
        mtp_horizons = self._mtp_horizons()
        for t in range(num_ticks):
            logits_t = self.lm_head(tick_outs[..., t])
            if mtp_horizons:
                horizon_losses = []
                label_mask = None
                for horizon in mtp_horizons:
                    horizon_loss, horizon_mask = self._per_sample_lm_loss(
                        logits_t, labels, horizon=horizon)
                    horizon_losses.append(horizon_loss)
                    if horizon == 1 or label_mask is None:
                        label_mask = horizon_mask
                per_sample_loss = torch.stack(horizon_losses, dim=1).mean(dim=1)
            else:
                horizon = self._tick_horizon(t, num_ticks)
                per_sample_loss, label_mask = self._per_sample_lm_loss(
                    logits_t, labels, horizon=horizon)
            losses.append(per_sample_loss)

            next_loss, next_mask = self._per_sample_lm_loss(
                logits_t, labels, horizon=1)
            next_losses.append(next_loss)
            certainties.append(self._per_sample_entropy(logits_t, next_mask, horizon=1))

        losses = torch.stack(losses, dim=1)
        next_losses = torch.stack(next_losses, dim=1)
        certainties = torch.stack(certainties, dim=1)

        tick_loss = self._combine_tick_losses(losses, certainties)
        if self.config.tick_improve_weight > 0 and num_ticks > 1:
            margin = float(self.config.tick_improve_margin)
            improve_loss = F.relu(next_losses[:, 1:] - next_losses[:, :-1] + margin).mean()
            tick_loss = tick_loss + self.config.tick_improve_weight * improve_loss
        slow_weight = float(self.config.slow_output_weight)
        base_w = float(self.config.tick_loss_base_weight)
        loss = (base_w + slow_weight) * final_loss + base_w * tick_loss
        fast_slow_aux = self._fast_slow_output_loss(
            input_ids, labels, tick_outs, final_logits)
        loss = loss + fast_slow_aux
        moe_aux_loss = self._moe_aux_loss()
        if moe_aux_loss is not None:
            loss = loss + moe_aux_loss

        return loss, losses, certainties
