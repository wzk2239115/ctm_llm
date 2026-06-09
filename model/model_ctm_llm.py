import math
import copy
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.config import CTMLLMConfig
from model.ctm_modules import SuperLinear, SynapseUNET, Squeeze, TTTMLP
from model.draft_modules import DraftSlotHead
from model.moe_regional import RegionalMoEMixin


def _parse_int_list(raw, *, max_value=None):
    values = []
    for item in str(raw or "").replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError:
            continue
        if value <= 0:
            continue
        if max_value is not None:
            value = min(value, max_value)
        if value not in values:
            values.append(value)
    return sorted(values)


def _parse_name_list(raw):
    values = []
    for item in str(raw or "").replace(";", ",").split(","):
        item = item.strip().lower()
        if item and item not in values:
            values.append(item)
    return values


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        normed = x.float() * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * normed).type_as(x)


class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_ratio=4):
        super().__init__()
        intermediate = math.ceil(hidden_size * intermediate_ratio / 64) * 64
        self.gate_proj = nn.Linear(hidden_size, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DINOProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, bottleneck_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.utils.parametrizations.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False))

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        return self.last_layer(x)


class CTMBlock(RegionalMoEMixin, nn.Module):
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
        self.moe_routing_mode = getattr(config, 'moe_routing_mode', 'none')
        self.moe_num_experts = max(1, int(getattr(config, 'moe_num_experts', 1)))
        self.moe_topk_experts = max(1, int(getattr(config, 'moe_topk_experts', 1)))
        self.moe_shared_experts = max(0, int(getattr(config, 'moe_shared_experts', 0)))
        self.moe_expert_size = int(getattr(config, 'moe_expert_size', 0))
        if self.moe_expert_size <= 0 and self.moe_num_experts > 0:
            self.moe_expert_size = config.d_model // self.moe_num_experts
        self.moe_expert_dropout = float(getattr(config, 'moe_expert_dropout', 0.0))
        self.moe_activation_passes = max(1, int(
            getattr(config, 'moe_activation_passes', 1)))
        self.moe_region_diversity_weight = float(
            getattr(config, 'moe_region_diversity_weight', 0.0))
        self.moe_load_balance_weight = float(
            getattr(config, 'moe_load_balance_weight', 0.0))
        self.moe_router_entropy_weight = float(
            getattr(config, 'moe_router_entropy_weight', 0.0))
        self.moe_router_z_loss_weight = float(
            getattr(config, 'moe_router_z_loss_weight', 0.0))
        self.moe_dispatch_mode = getattr(config, 'moe_dispatch_mode', 'dense_mask')
        self.moe_capacity_factor = float(getattr(config, 'moe_capacity_factor', 1.0))
        self.moe_drop_tokens = bool(getattr(config, 'moe_drop_tokens', False))
        self.moe_aux_loss_free_bias = bool(
            getattr(config, 'moe_aux_loss_free_bias', False))
        self.moe_aux_loss = None
        self.last_executed_ticks = 0
        self.diff_cell_mode = getattr(config, 'diff_cell_mode', 'none')
        self.diff_cell_temperature = float(
            getattr(config, 'diff_cell_temperature', 1.0))
        self.diff_cell_capacity_weight = float(
            getattr(config, 'diff_cell_capacity_weight', 0.0))
        self.diff_cell_memory_weight = float(
            getattr(config, 'diff_cell_memory_weight', 0.0))
        self.diff_cell_diversity_weight = float(
            getattr(config, 'diff_cell_diversity_weight', 0.0))
        self.diff_cell_widths = _parse_int_list(
            getattr(config, 'diff_cell_widths', ''), max_value=self.moe_expert_size)
        self.diff_cell_memory_lengths = _parse_int_list(
            getattr(config, 'diff_cell_memory_lengths', ''), max_value=config.memory_length)
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

        self.context_reading_mode = getattr(config, 'context_reading_mode', 'none')
        self.context_source_names = [
            name for name in _parse_name_list(getattr(
                config, 'context_reading_sources',
                'local,compressed,retrieval,expert,egram'))
            if name in ('local', 'compressed', 'retrieval', 'expert', 'egram')
        ]
        self.context_reading_enabled = (
            self.context_reading_mode != 'none' and len(self.context_source_names) > 0)
        self.context_source_to_idx = {
            name: idx for idx, name in enumerate(self.context_source_names)}
        self.context_local_window = max(1, int(getattr(config, 'context_local_window', 32)))
        self.context_compressed_stride = max(
            1, int(getattr(config, 'context_compressed_stride', 16)))
        self.context_retrieval_topk = max(
            1, int(getattr(config, 'context_retrieval_topk', 8)))
        self.context_expert_memory_slots = max(
            1, int(getattr(config, 'context_expert_memory_slots', 4)))
        self.context_egram_decay = min(
            max(float(getattr(config, 'context_egram_decay', 0.75)), 0.0), 0.99)
        if self.context_reading_enabled:
            self.context_source_gate = nn.Linear(
                self.synch_repr_action, len(self.context_source_names), bias=False)
            self.context_fusion_gate = nn.Parameter(torch.tensor(
                float(getattr(config, 'context_reading_gate_init', -2.0))))
            self.context_egram_proj = nn.Sequential(
                nn.LayerNorm(config.d_model),
                nn.Linear(config.d_model, config.d_input, bias=False),
            )
            num_memory_experts = max(1, self.moe_num_experts)
            self.context_expert_memory = nn.Parameter(torch.empty(
                num_memory_experts, self.context_expert_memory_slots, config.d_input))
            nn.init.normal_(self.context_expert_memory, mean=0.0,
                            std=1.0 / math.sqrt(config.d_input))

        self.start_activated_state = nn.Parameter(
            torch.zeros(config.d_model).uniform_(
                -math.sqrt(1 / config.d_model), math.sqrt(1 / config.d_model)))
        self.start_trace = nn.Parameter(
            torch.zeros(config.d_model, config.memory_length).uniform_(
                -math.sqrt(1 / (config.d_model + config.memory_length)),
                math.sqrt(1 / (config.d_model + config.memory_length))))

        self._init_synch(config)

        self.output_proj = nn.Linear(self.synch_repr_out, config.hidden_size, bias=False)

        self.draft_mode = getattr(config, 'draft_mode', 'none')
        self.draft_slot_head = None
        if self.draft_mode != 'none':
            self.draft_slot_head = DraftSlotHead(
                config, config.hidden_size, config.vocab_size)

        self.post_ctm_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config.hidden_size)
        self.resid_drop = nn.Dropout(config.dropout)

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

    def _causal_mask(self, T, S, device):
        query_pos = torch.arange(S - T, S, device=device).view(T, 1)
        key_pos = torch.arange(S, device=device).view(1, S)
        return key_pos <= query_pos

    def _local_context(self, v, T):
        S = v.size(1)
        window = min(self.context_local_window, S)
        prefix = F.pad(v.cumsum(dim=1), (0, 0, 1, 0))
        end = torch.arange(S - T + 1, S + 1, device=v.device)
        start = (end - window).clamp(min=0)
        sums = prefix[:, end] - prefix[:, start]
        counts = (end - start).to(v.dtype).view(1, T, 1).clamp(min=1)
        return sums / counts

    def _compressed_context(self, q, v, T):
        S = v.size(1)
        D = v.size(-1)
        stride = min(self.context_compressed_stride, S)
        prefix = F.pad(v.cumsum(dim=1), (0, 0, 1, 0))
        end = torch.arange(1, S + 1, device=v.device)
        start = (end - stride).clamp(min=0)
        memory = (prefix[:, end] - prefix[:, start]) / \
            (end - start).to(v.dtype).view(1, S, 1).clamp(min=1)
        scores = torch.matmul(q, memory.transpose(1, 2)) / math.sqrt(D)
        mask = self._causal_mask(T, S, q.device).view(1, T, S)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores.float(), dim=-1).type_as(q)
        return torch.matmul(weights, memory)

    def _retrieval_context(self, q, k, v, T):
        S = k.size(1)
        topk = min(self.context_retrieval_topk, S)
        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(k.size(-1))
        mask = self._causal_mask(T, S, q.device).view(1, T, S)
        scores = scores.masked_fill(~mask, float("-inf"))
        vals, idx = torch.topk(scores, topk, dim=-1)
        weights = F.softmax(vals.float(), dim=-1).type_as(q)
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, -1, v.size(-1))
        gathered = v.unsqueeze(1).expand(-1, T, -1, -1).gather(2, gather_idx)
        return (weights.unsqueeze(-1) * gathered).sum(dim=2)

    def _expert_memory_context(self, q, activated):
        memory = self.context_expert_memory
        E, M, D = memory.shape
        scores = torch.einsum('btd,emd->btem', q, memory) / math.sqrt(D)
        slot_weights = F.softmax(scores.float().flatten(-2), dim=-1).type_as(q)
        slot_values = memory.reshape(E * M, D)
        context = torch.matmul(slot_weights, slot_values)

        if self.moe_num_experts > 1 and self.moe_expert_size * self.moe_num_experts == self.d_model:
            expert_scores = activated.detach().view(
                activated.size(0), activated.size(1),
                self.moe_num_experts, self.moe_expert_size).abs().mean(dim=-1)
            expert_weights = F.softmax(expert_scores.float(), dim=-1).type_as(q)
            expert_values = memory.mean(dim=1)
            context = 0.5 * context + 0.5 * torch.matmul(expert_weights, expert_values)
        return context

    def _context_reading(self, q, k, v, sync_a, activated, egram_state):
        if not self.context_reading_enabled:
            return q.new_zeros(q.shape), egram_state
        T = q.size(1)
        source_values = []
        for name in self.context_source_names:
            if name == 'local':
                source_values.append(self._local_context(v, T))
            elif name == 'compressed':
                source_values.append(self._compressed_context(q, v, T))
            elif name == 'retrieval':
                source_values.append(self._retrieval_context(q, k, v, T))
            elif name == 'expert':
                source_values.append(self._expert_memory_context(q, activated))
            elif name == 'egram':
                draft = self.context_egram_proj(activated)
                if egram_state is None:
                    egram_state = draft
                else:
                    egram_state = self.context_egram_decay * egram_state + \
                        (1.0 - self.context_egram_decay) * draft
                source_values.append(egram_state)
        stacked = torch.stack(source_values, dim=-2)
        gates = F.softmax(self.context_source_gate(sync_a).float(), dim=-1).type_as(q)
        context = (stacked * gates.unsqueeze(-1)).sum(dim=-2)
        return torch.sigmoid(self.context_fusion_gate).type_as(q) * context, egram_state

    def forward(self, x, pos_emb=None, past_kv=None, use_cache=False, track=False,
                num_iters=None, return_all_ticks=False,
                prev_activated=None, prev_trace=None,
                halt_lm_head=None, enable_tick_halt=False, draft_lm_head=None):
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
        all_draft_slot_logits = [] if (
            return_all_ticks and self.draft_slot_head is not None) else None
        prev_sync_o_activated = None
        last_sync_o = None
        self.last_executed_ticks = 0
        egram_state = None
        allow_halt_break = (
            enable_tick_halt
            and not self.training
            and halt_lm_head is not None
            and self.config.tick_halt_mode != 'none'
        )

        for tick in range(num_iters):
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

            context_extra, egram_state = self._context_reading(
                q, k, v, sync_a, activated, egram_state)
            attn = self.attn_drop(
                self.o_proj(attn_out.transpose(1, 2).reshape(B, T, -1) + context_extra))

            if self._use_group_sparse_backend():
                activated, state_trace = self._run_group_sparse_regional_tick(
                    attn, activated, state_trace, prev_sync_o_activated)
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

            sync_o, alpha_o, beta_o = self._compute_synch(
                activated, alpha_o, beta_o, r_o, self.out_left, self.out_right)
            last_sync_o = sync_o
            prev_sync_o_activated = activated
            tick_out = self.output_proj(sync_o)
            self.last_executed_ticks = tick + 1
            if all_tick_outs is not None:
                all_tick_outs.append(tick_out)
            if all_draft_slot_logits is not None:
                _, slot_logits = self.draft_slot_head(tick_out, lm_head=draft_lm_head)
                all_draft_slot_logits.append(slot_logits)

            if track:
                tracking['pre_activations'].append(state[0].detach().cpu().numpy())
                tracking['post_activations'].append(activated[0].detach().cpu().numpy())
                tracking['sync_action'].append(sync_a[0].detach().cpu().numpy())
                tracking['state_trace'].append(state_trace[0].detach().cpu().numpy())

            if allow_halt_break and tick + 1 < num_iters:
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
        if all_draft_slot_logits is not None:
            extras['draft_slot_logits'] = torch.stack(all_draft_slot_logits, dim=-1)
        if track:
            for k_arr in ['pre_activations', 'post_activations', 'sync_action', 'state_trace']:
                tracking[k_arr] = np.array(tracking[k_arr])
            extras['tracking'] = tracking

        extras['final_activated'] = activated
        extras['final_trace'] = state_trace

        if extras:
            return x, present_kv, extras
        return x, present_kv


class CTMModel(nn.Module):
    def __init__(self, config: CTMLLMConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embed = nn.Embedding(config.max_position_embeddings, config.d_input)
        self.drop = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [CTMBlock(i, config) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, past_key_values=None, use_cache=False, track=False,
                num_iters=None, return_all_ticks=False,
                halt_lm_head=None, enable_tick_halt=False, draft_lm_head=None):
        B, T = input_ids.shape
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        h = self.drop(self.embed_tokens(input_ids))
        pos_emb = self.pos_embed.weight
        presents = []
        tracking_all = {}
        last_tick_outs = None
        last_draft_slot_logits = None
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
                draft_lm_head=draft_lm_head if is_last else None,
            )

            if track and not is_last:
                result = layer(h, track=True, return_all_ticks=False, **layer_kwargs)
                h, present, extras = result
                tracking_all[f'layer_{layer.layer_id}'] = extras['tracking']
            elif is_last and (track or return_all_ticks):
                result = layer(h, track=track, return_all_ticks=True, **layer_kwargs)
                h, present, extras = result
                if track:
                    tracking_all[f'layer_{layer.layer_id}'] = extras.get('tracking', {})
                if return_all_ticks:
                    last_tick_outs = extras.get('tick_outputs')
                    last_draft_slot_logits = extras.get('draft_slot_logits')
            else:
                result = layer(h, **layer_kwargs)
                if len(result) == 3:
                    h, present, extras = result
                else:
                    h, present = result
                    extras = {}

            if self.config.cross_layer_state and isinstance(extras, dict):
                prev_activated = extras.get('final_activated', prev_activated)
                prev_trace = extras.get('final_trace', prev_trace)

            executed_ticks.append(getattr(layer, 'last_executed_ticks', 0))
            presents.append(present)

        h = self.norm(h)

        outputs = [h, presents]
        if return_all_ticks:
            outputs.append(last_tick_outs)
            outputs.append(last_draft_slot_logits)
        if track:
            outputs.append(tracking_all)
        if enable_tick_halt:
            outputs.append(executed_ticks)

        if len(outputs) == 2:
            return tuple(outputs)
        return tuple(outputs)


class CTMForCausalLM(nn.Module):
    def __init__(self, config: CTMLLMConfig):
        super().__init__()
        self.config = config
        self.model = CTMModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.reflex_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.reflex_adapter = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2, bias=False),
            nn.SiLU(),
            nn.Linear(config.hidden_size // 2, config.hidden_size, bias=False),
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.dino_enabled = float(getattr(config, 'dino_self_supervised_weight', 0.0)) > 0
        self.dino_student_head = None
        self.dino_teacher_model = None
        self.dino_teacher_head = None
        self.last_dino_loss = 0.0
        if self.dino_enabled:
            self.dino_student_head = DINOProjectionHead(
                config.hidden_size,
                int(config.dino_hidden_dim),
                int(config.dino_bottleneck_dim),
                int(config.dino_out_dim),
            )
            self.dino_teacher_model = copy.deepcopy(self.model)
            self.dino_teacher_head = copy.deepcopy(self.dino_student_head)
            for module in (self.dino_teacher_model, self.dino_teacher_head):
                module.eval()
                for param in module.parameters():
                    param.requires_grad_(False)
            self.register_buffer(
                'dino_center',
                torch.zeros(1, 1, int(config.dino_out_dim)),
                persistent=True,
            )
        else:
            self.register_buffer('dino_center', torch.zeros(1, 1, 1), persistent=False)

    def _moe_aux_loss(self):
        aux = None
        for layer in self.model.layers:
            value = getattr(layer, 'moe_aux_loss', None)
            if value is None:
                continue
            aux = value if aux is None else aux + value
        return aux

    def _tick_horizon(self, tick, num_ticks):
        mode = self.config.elf_horizon_mode
        max_horizon = max(1, int(self.config.elf_max_horizon))
        if mode == 'none':
            return 1
        if mode == 'linear':
            return min(tick + 1, max_horizon)
        if mode == 'pow2':
            return min(2 ** tick, max_horizon)
        raise ValueError(f"Unknown elf_horizon_mode: {mode}")

    def _mtp_horizons(self):
        mode = getattr(self.config, 'moe_mtp_mode', 'none')
        raw = str(getattr(self.config, 'moe_mtp_horizons', '') or '')
        if mode == 'none' or not raw.strip():
            return []
        horizons = []
        for item in raw.split(','):
            item = item.strip()
            if not item:
                continue
            try:
                horizon = int(item)
            except ValueError:
                continue
            if horizon > 0 and horizon not in horizons:
                horizons.append(horizon)
        return horizons

    def _draft_mtp_horizons(self):
        horizons = self._mtp_horizons()
        if getattr(self.config, 'draft_mode', 'none') == 'none':
            return horizons
        block = max(1, int(getattr(self.config, 'draft_block_size', 1)))
        return [h for h in horizons if h > block]

    def _draft_slot_tick_loss(self, slot_logits, labels):
        block = slot_logits.size(2)
        losses = []
        for slot in range(block):
            horizon = slot + 1
            slot_loss, _ = self._per_sample_lm_loss(
                slot_logits[:, :, slot, :], labels, horizon=horizon)
            losses.append(slot_loss)
        return torch.stack(losses, dim=1).mean(dim=1)

    def _fast_output_ticks(self):
        ticks = _parse_int_list(getattr(self.config, 'fast_output_ticks', '1,4'))
        return [tick for tick in ticks if tick > 0]

    def _reflex_logits(self, input_ids):
        h = self.model.embed_tokens(input_ids)
        h = h + self.reflex_adapter(h)
        return self.lm_head(self.reflex_norm(h))

    @staticmethod
    def _lm_loss_from_logits(logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1), ignore_index=-100)

    @staticmethod
    def _distill_loss(student_logits, teacher_logits, labels):
        if labels.size(1) <= 1:
            return student_logits.new_zeros(())
        student = student_logits[..., :-1, :].float()
        teacher = teacher_logits[..., :-1, :].detach().float()
        mask = labels[..., 1:] != -100
        if not mask.any():
            return student_logits.new_zeros(())
        log_p = F.log_softmax(student, dim=-1)
        q = F.softmax(teacher, dim=-1)
        kl = F.kl_div(log_p, q, reduction='none').sum(dim=-1)
        return (kl * mask).sum() / mask.sum().clamp(min=1)

    def update_dino_teacher(self):
        if not self.dino_enabled:
            return
        momentum = float(getattr(self.config, 'dino_teacher_momentum', 0.996))
        with torch.no_grad():
            for student, teacher in zip(
                self.model.parameters(), self.dino_teacher_model.parameters()
            ):
                teacher.data.mul_(momentum).add_(student.data, alpha=1.0 - momentum)
            for student, teacher in zip(
                self.dino_student_head.parameters(), self.dino_teacher_head.parameters()
            ):
                teacher.data.mul_(momentum).add_(student.data, alpha=1.0 - momentum)

    def reset_dino_teacher(self):
        if not self.dino_enabled:
            return
        self.dino_teacher_model.load_state_dict(self.model.state_dict(), strict=True)
        self.dino_teacher_head.load_state_dict(self.dino_student_head.state_dict(), strict=True)
        self.dino_teacher_model.eval()
        self.dino_teacher_head.eval()

    def _dino_token_mask(self, input_ids, labels):
        pad_id = int(getattr(self.config, 'dino_pad_token_id', 0))
        mask = input_ids != pad_id
        if labels is not None:
            mask = mask | (labels != -100)
        return mask

    def _dino_self_supervised_loss(self, input_ids, labels, student_hidden, num_iters):
        if not self.dino_enabled:
            return student_hidden.new_zeros(())

        token_mask = self._dino_token_mask(input_ids, labels)
        if not token_mask.any():
            return student_hidden.new_zeros(())

        student_temp = max(float(self.config.dino_student_temperature), 1e-4)
        teacher_temp = max(float(self.config.dino_teacher_temperature), 1e-4)
        center_momentum = float(self.config.dino_center_momentum)

        student_logits = self.dino_student_head(student_hidden.float())
        with torch.no_grad():
            self.dino_teacher_model.eval()
            self.dino_teacher_head.eval()
            teacher_hidden = self.dino_teacher_model(
                input_ids, use_cache=False, track=False, num_iters=num_iters,
                return_all_ticks=False,
            )[0]
            teacher_logits = self.dino_teacher_head(teacher_hidden.float())
            teacher_probs = F.softmax(
                (teacher_logits - self.dino_center.to(teacher_logits.device)) / teacher_temp,
                dim=-1,
            )
            batch_center = teacher_logits[token_mask].mean(dim=0, keepdim=True).view(1, 1, -1)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(batch_center)
                batch_center = batch_center / dist.get_world_size()
            self.dino_center.mul_(center_momentum).add_(
                batch_center.to(self.dino_center.device),
                alpha=1.0 - center_momentum,
            )

        student_log_probs = F.log_softmax(student_logits / student_temp, dim=-1)
        per_token_loss = -(teacher_probs.detach() * student_log_probs).sum(dim=-1)
        return per_token_loss[token_mask].mean()

    def _fast_slow_output_loss(self, input_ids, labels, tick_outs, final_logits):
        mode = getattr(self.config, 'fast_output_mode', 'none')
        if mode == 'none':
            return final_logits.new_zeros(())
        aux = final_logits.new_zeros(())
        distill_weight = float(getattr(self.config, 'fast_output_distill_weight', 0.0))
        fast_weight = float(getattr(self.config, 'fast_output_weight', 0.0))
        habit_weight = float(getattr(self.config, 'habit_output_weight', 0.0))
        if fast_weight > 0:
            reflex_logits = self._reflex_logits(input_ids)
            aux = aux + fast_weight * self._lm_loss_from_logits(reflex_logits, labels)
            if distill_weight > 0:
                aux = aux + fast_weight * distill_weight * self._distill_loss(
                    reflex_logits, final_logits, labels)
        if habit_weight > 0 and tick_outs is not None:
            num_ticks = tick_outs.size(-1)
            tick_losses = []
            distill_losses = []
            for tick in self._fast_output_ticks():
                idx = min(max(tick - 1, 0), num_ticks - 1)
                logits_t = self.lm_head(tick_outs[..., idx])
                tick_losses.append(self._lm_loss_from_logits(logits_t, labels))
                if distill_weight > 0:
                    distill_losses.append(self._distill_loss(logits_t, final_logits, labels))
            if tick_losses:
                aux = aux + habit_weight * torch.stack(tick_losses).mean()
            if distill_losses:
                aux = aux + habit_weight * distill_weight * torch.stack(distill_losses).mean()
        return aux

    @staticmethod
    def _per_sample_lm_loss(logits, labels, horizon):
        B = labels.size(0)
        if labels.size(1) <= horizon:
            return logits.new_zeros(B), logits.new_zeros(B, 0, dtype=torch.bool)
        shift_logits = logits[..., :-horizon, :].contiguous()
        shift_labels = labels[..., horizon:].contiguous()
        label_mask = shift_labels != -100
        per_token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1), ignore_index=-100, reduction='none')
        per_token_loss = per_token_loss.view(B, -1)
        per_sample_loss = (
            per_token_loss * label_mask).sum(dim=1) / label_mask.sum(dim=1).clamp(min=1)
        return per_sample_loss, label_mask

    @staticmethod
    def _per_sample_entropy(logits, label_mask, horizon):
        if label_mask.numel() == 0:
            return logits.new_zeros(logits.size(0))
        valid_logits = logits[..., :-horizon, :]
        probs = F.softmax(valid_logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp(min=1e-12))).sum(-1)
        norm_ent = entropy / math.log(logits.size(-1))
        return (norm_ent * label_mask).sum(dim=1) / label_mask.sum(dim=1).clamp(min=1)

    def _combine_tick_losses(self, losses, certainties):
        mode = self.config.tick_loss_mode
        if self.config.tick_halt_mode != 'none':
            tick_loss, _ = self._halt_weighted_tick_loss(losses, certainties)
            return tick_loss
        if mode == 'min_conf':
            confidence = 1 - certainties
            loss_min = losses.min(dim=1).values.mean()
            best_conf_tick = confidence.argmax(dim=1)
            batch_idx = torch.arange(losses.size(0), device=losses.device)
            loss_conf = losses[batch_idx, best_conf_tick].mean()
            return (loss_min + loss_conf) / 2.0
        if mode == 'mean':
            return losses.mean()
        if mode == 'last':
            return losses[:, -1].mean()
        raise ValueError(f"Unknown tick_loss_mode: {mode}")

    def _halt_weighted_tick_loss(self, losses, certainties):
        mode = self.config.tick_halt_mode
        confidence = 1 - certainties
        if mode == 'confidence':
            temp = max(float(self.config.tick_halt_temperature), 1e-4)
            weights = torch.softmax(confidence / temp, dim=1)
        elif mode == 'threshold':
            threshold = float(self.config.tick_halt_threshold)
            hit = confidence >= threshold
            any_hit = hit.any(dim=1)
            first_hit = hit.float().argmax(dim=1)
            last_tick = torch.full_like(first_hit, losses.size(1) - 1)
            selected = torch.where(any_hit, first_hit, last_tick)
            weights = F.one_hot(selected, num_classes=losses.size(1)).type_as(losses)
        else:
            raise ValueError(f"Unknown tick_halt_mode: {mode}")

        tick_loss = (losses * weights).sum(dim=1).mean()
        if self.config.tick_compute_weight > 0:
            tick_ids = torch.arange(1, losses.size(1) + 1, device=losses.device,
                                    dtype=losses.dtype)
            expected_tick = (weights * tick_ids.view(1, -1)).sum(dim=1)
            compute_penalty = expected_tick.mean() / losses.size(1)
            tick_loss = tick_loss + self.config.tick_compute_weight * compute_penalty
        return tick_loss, weights

    def forward(self, input_ids, past_key_values=None, use_cache=False, labels=None,
                num_iters=None):
        enable_halt = self.config.tick_halt_mode != 'none' and not self.training
        result = self.model(
            input_ids, past_key_values, use_cache, track=False, num_iters=num_iters,
            halt_lm_head=self.lm_head, enable_tick_halt=enable_halt)
        h, past_key_values = result[0], result[1]
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
        result = self.model(
            input_ids, track=False, num_iters=num_iters, return_all_ticks=True,
            halt_lm_head=self.lm_head, enable_tick_halt=False,
            draft_lm_head=self.lm_head)
        h, tick_outs = result[0], result[2]
        draft_slot_logits = result[3] if len(result) > 3 else None
        num_ticks = tick_outs.size(-1)

        final_logits = self.lm_head(h)
        final_loss = self._lm_loss_from_logits(final_logits, labels)

        draft_enabled = getattr(self.config, 'draft_mode', 'none') != 'none'
        draft_block = max(1, int(getattr(self.config, 'draft_block_size', 1)))
        draft_weight = float(getattr(self.config, 'draft_loss_weight', 0.0))
        mtp_horizons = self._draft_mtp_horizons()

        losses = []
        next_losses = []
        certainties = []
        for t in range(num_ticks):
            logits_t = self.lm_head(tick_outs[..., t])
            tick_components = []

            next_loss, next_mask = self._per_sample_lm_loss(
                logits_t, labels, horizon=1)
            tick_components.append(next_loss)

            if mtp_horizons:
                mtp_losses = []
                for horizon in mtp_horizons:
                    horizon_loss, _ = self._per_sample_lm_loss(
                        logits_t, labels, horizon=horizon)
                    mtp_losses.append(horizon_loss)
                tick_components.append(torch.stack(mtp_losses, dim=1).mean(dim=1))
            else:
                elf_h = self._tick_horizon(t, num_ticks)
                if self.config.elf_horizon_mode != 'none':
                    if not (draft_enabled and elf_h <= draft_block):
                        elf_loss, _ = self._per_sample_lm_loss(
                            logits_t, labels, horizon=elf_h)
                        tick_components.append(elf_loss)
                elif elf_h > 1 and not (draft_enabled and elf_h <= draft_block):
                    horizon_loss, _ = self._per_sample_lm_loss(
                        logits_t, labels, horizon=elf_h)
                    tick_components.append(horizon_loss)

            if draft_slot_logits is not None and draft_weight > 0:
                slot_logits_t = draft_slot_logits[..., :, :, :, t]
                draft_loss = self._draft_slot_tick_loss(slot_logits_t, labels)
                tick_components.append(draft_weight * draft_loss)

            per_sample_loss = torch.stack(tick_components, dim=1).mean(dim=1)
            losses.append(per_sample_loss)

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
        slow_weight = float(getattr(self.config, 'slow_output_weight', 0.0))
        loss = (0.5 + slow_weight) * final_loss + 0.5 * tick_loss
        fast_slow_aux = self._fast_slow_output_loss(
            input_ids, labels, tick_outs, final_logits)
        loss = loss + fast_slow_aux
        dino_weight = float(getattr(self.config, 'dino_self_supervised_weight', 0.0))
        if dino_weight > 0:
            dino_loss = self._dino_self_supervised_loss(
                input_ids, labels, h, num_iters)
            self.last_dino_loss = float(dino_loss.detach().float().item())
            loss = loss + dino_weight * dino_loss
        else:
            self.last_dino_loss = 0.0
        moe_aux_loss = self._moe_aux_loss()
        if moe_aux_loss is not None:
            loss = loss + moe_aux_loss

        return loss, losses, certainties

    @torch.inference_mode()
    def forward_track(self, input_ids, num_iters=None):
        result = self.model(input_ids, track=True, num_iters=num_iters,
                            return_all_ticks=False)
        if len(result) == 3:
            h, _, tracking = result
        else:
            h, _, _, tracking = result
        logits = self.lm_head(h)
        probs = torch.softmax(logits, dim=-1)
        return tracking, logits, probs

    @torch.inference_mode()
    def generate(self, input_ids, max_new_tokens=512, temperature=0.85,
                 top_p=0.85, top_k=50, eos_token_id=2, use_cache=True,
                 repetition_penalty=1.0, num_iters=None):
        past_kv = None
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

        for _ in range(max_new_tokens):
            inp = input_ids if past_kv is None else input_ids[:, -1:]
            out = self.forward(inp, past_key_values=past_kv, use_cache=use_cache, num_iters=num_iters)
            token_logits = out['logits'][:, -1, :] / temperature

            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[i])
                    score = token_logits[i, seen]
                    token_logits[i, seen] = torch.where(
                        score > 0, score / repetition_penalty, score * repetition_penalty)

            if top_k > 0:
                top_k_eff = min(top_k, token_logits.size(-1))
                topk_val = torch.topk(token_logits, top_k_eff)[0][..., -1, None]
                token_logits[token_logits < topk_val] = float('-inf')

            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(token_logits, descending=True)
                cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cum_probs > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False
                token_logits[mask.scatter(1, sorted_idx, mask)] = float('-inf')

            probs = torch.softmax(token_logits, dim=-1)
            new_tokens = torch.multinomial(probs, num_samples=1)

            if eos_token_id is not None:
                new_tokens = torch.where(
                    finished.unsqueeze(-1),
                    new_tokens.new_full(new_tokens.shape, eos_token_id),
                    new_tokens)

            input_ids = torch.cat([input_ids, new_tokens], dim=-1)
            past_kv = out['past_key_values'] if use_cache else None

            if eos_token_id is not None:
                finished |= new_tokens.squeeze(1).eq(eos_token_id)
                if finished.all():
                    break

        return input_ids

    def compute_certainties(self, logits_seq):
        probs = F.softmax(logits_seq, dim=-1)
        ent = -(probs * torch.log(probs.clamp(min=1e-12))).sum(-1)
        norm_ent = ent / math.log(logits_seq.size(-1))
        return norm_ent


def build_ctm_for_causal_lm(config: CTMLLMConfig):
    """Return sync CTM by default; async backend when banded clocks are enabled."""
    if getattr(config, 'async_tick_mode', 'none') == 'banded':
        from model.model_ctm_async import AsyncCTMForCausalLM
        return AsyncCTMForCausalLM(config)
    return CTMForCausalLM(config)
