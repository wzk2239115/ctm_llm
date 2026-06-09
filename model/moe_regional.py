import math

import torch
import torch.nn.functional as F


class RegionalMoEMixin:
    """Shared regional MoE routing/sparsity helpers for CTM blocks."""

    def _apply_moe_group_sparsity(self, activated):
        expert_size = self.moe_expert_size
        num_experts = self.moe_num_experts
        if expert_size <= 0 or num_experts <= 0:
            return None
        if expert_size * num_experts != self.d_model:
            return None

        B, T, _ = activated.shape
        x = activated.view(B, T, num_experts, expert_size)
        raw_scores = x.abs().mean(dim=-1)
        scores = raw_scores.detach()

        shared = min(self.moe_shared_experts, num_experts)
        routed_count = max(num_experts - shared, 1)
        topk = self._effective_moe_topk(routed_count)
        routed_scores = scores[:, :, shared:]
        routed_raw_scores = raw_scores[:, :, shared:]
        mode = self.moe_routing_mode

        if mode in ('regional_topk', 'regional_shared_topk') or self.moe_activation_passes > 1:
            return self._apply_regional_moe_sparsity(
                activated, x, scores, raw_scores, shared, routed_count, topk,
                routed_scores, routed_raw_scores)

        if mode == 'hash':
            pos = torch.arange(T, device=activated.device).view(1, T, 1)
            offsets = torch.arange(topk, device=activated.device).view(1, 1, topk)
            idx = (pos + offsets + self.layer_id) % routed_count
            idx = idx.expand(B, -1, -1)
        elif mode == 'expert_choice':
            mean_scores = routed_scores.mean(dim=(0, 1), keepdim=True)
            idx = torch.topk(mean_scores.expand(B, T, -1), topk, dim=-1).indices
        else:
            idx = torch.topk(routed_scores, topk, dim=-1).indices

        expert_mask = torch.zeros_like(scores)
        expert_mask[:, :, shared:].scatter_(-1, idx, 1.0)
        if shared > 0:
            expert_mask[:, :, :shared] = 1.0

        if self.training and self.moe_expert_dropout > 0:
            keep = torch.rand_like(expert_mask) >= self.moe_expert_dropout
            if shared > 0:
                keep[:, :, :shared] = True
            expert_mask = expert_mask * keep.type_as(expert_mask)

        active_cells = expert_mask.sum(dim=-1, keepdim=True).clamp(min=1) * expert_size
        mask = expert_mask.unsqueeze(-1).expand_as(x).reshape_as(activated)
        if self.cell_sparsity_rescale:
            mask = mask * (self.d_model / active_cells)
        self._update_moe_aux_loss(routed_raw_scores)
        return activated * mask

    def _effective_moe_topk(self, routed_count):
        target = min(self.moe_topk_experts, routed_count)
        warmup_steps = max(0, int(getattr(self.config, 'moe_topk_warmup_steps', 0)))
        if warmup_steps <= 0:
            return target
        current_step = max(0, int(getattr(self.config, 'global_step', 0)))
        progress = min(1.0, current_step / max(1, warmup_steps))
        warm_topk = int(math.ceil(target + (routed_count - target) * (1.0 - progress)))
        return min(max(target, warm_topk), routed_count)

    def _apply_regional_moe_sparsity(
        self, activated, x, scores, raw_scores, shared, routed_count, topk,
        routed_scores, routed_raw_scores,
    ):
        max_distinct_passes = max(1, math.ceil(routed_count / max(1, topk)))
        passes = min(max(1, self.moe_activation_passes), max_distinct_passes)
        pass_outputs = []
        pass_masks = []
        selected = torch.zeros_like(routed_scores, dtype=torch.bool)

        for _ in range(passes):
            masked_scores = routed_scores.masked_fill(selected, float("-inf"))
            idx = torch.topk(masked_scores, topk, dim=-1).indices
            routed_mask = torch.zeros_like(routed_scores)
            routed_mask.scatter_(-1, idx, 1.0)
            selected = selected | routed_mask.bool()

            expert_mask = torch.zeros_like(scores)
            expert_mask[:, :, shared:] = routed_mask
            if shared > 0:
                expert_mask[:, :, :shared] = 1.0

            if self.training and self.moe_expert_dropout > 0:
                keep = torch.rand_like(expert_mask) >= self.moe_expert_dropout
                if shared > 0:
                    keep[:, :, :shared] = True
                expert_mask = expert_mask * keep.type_as(expert_mask)

            active_cells = expert_mask.sum(dim=-1, keepdim=True).clamp(min=1) * self.moe_expert_size
            mask = expert_mask.unsqueeze(-1).expand_as(x).reshape_as(activated)
            if self.cell_sparsity_rescale:
                mask = mask * (self.d_model / active_cells)
            pass_outputs.append(activated * mask)
            pass_masks.append(expert_mask[:, :, shared:])

        self._update_moe_aux_loss(routed_raw_scores)
        self._update_region_diversity_loss(pass_masks)
        return torch.stack(pass_outputs, dim=0).mean(dim=0)

    def _update_region_diversity_loss(self, pass_masks):
        if self.moe_region_diversity_weight <= 0 or len(pass_masks) <= 1:
            return
        overlaps = []
        for i in range(len(pass_masks)):
            for j in range(i + 1, len(pass_masks)):
                overlaps.append((pass_masks[i] * pass_masks[j]).sum(dim=-1).mean())
        if not overlaps:
            return
        diversity_loss = torch.stack(overlaps).mean() / max(1, self.moe_topk_experts)
        aux = self.moe_region_diversity_weight * diversity_loss
        self.moe_aux_loss = aux if self.moe_aux_loss is None else self.moe_aux_loss + aux

    def _update_moe_aux_loss(self, routed_scores):
        if (
            self.moe_load_balance_weight <= 0
            and self.moe_router_entropy_weight <= 0
            and self.moe_router_z_loss_weight <= 0
        ):
            return
        if routed_scores.size(-1) <= 1:
            return
        probs = F.softmax(routed_scores.float(), dim=-1)
        aux = routed_scores.new_zeros(())
        if self.moe_load_balance_weight > 0:
            load = probs.mean(dim=(0, 1))
            target = torch.full_like(load, 1.0 / load.numel())
            balance_loss = ((load - target) ** 2).mean() * load.numel()
            aux = aux + self.moe_load_balance_weight * balance_loss
        if self.moe_router_entropy_weight > 0:
            entropy = -(probs * torch.log(probs.clamp(min=1e-12))).sum(dim=-1).mean()
            norm_entropy = entropy / math.log(probs.size(-1))
            aux = aux - self.moe_router_entropy_weight * norm_entropy
        if self.moe_router_z_loss_weight > 0:
            z_loss = torch.logsumexp(routed_scores.float(), dim=-1).pow(2).mean()
            aux = aux + self.moe_router_z_loss_weight * z_loss
        self.moe_aux_loss = aux if self.moe_aux_loss is None else self.moe_aux_loss + aux

    def _use_group_sparse_backend(self):
        if self.moe_routing_mode not in ('regional_topk', 'regional_shared_topk'):
            return False
        if self.moe_dispatch_mode == 'dense_mask':
            return False
        return self.group_synapses is not None and self.group_trace_processors is not None

    def _route_experts(self, scores, shared, routed_count, topk, selected=None):
        routed_scores = scores[:, :, shared:]
        if selected is not None:
            routed_scores = routed_scores.masked_fill(selected, float("-inf"))
        idx = torch.topk(routed_scores, topk, dim=-1).indices
        expert_mask = torch.zeros_like(scores)
        expert_mask[:, :, shared:].scatter_(-1, idx, 1.0)
        if shared > 0:
            expert_mask[:, :, :shared] = 1.0
        if self.training and self.moe_expert_dropout > 0:
            keep = torch.rand_like(expert_mask) >= self.moe_expert_dropout
            if shared > 0:
                keep[:, :, :shared] = True
            expert_mask = expert_mask * keep.type_as(expert_mask)
        if self.moe_dispatch_mode == 'capacity_drop' or self.moe_drop_tokens:
            expert_mask = self._apply_expert_capacity(expert_mask, scores, shared)
        return expert_mask, idx

    def _apply_expert_capacity(self, expert_mask, scores, shared=0):
        B, T, E = expert_mask.shape
        active_per_token = expert_mask.sum(dim=-1).float().mean().clamp(min=1.0)
        capacity = int(math.ceil(self.moe_capacity_factor * B * T * active_per_token / E))
        capacity = max(1, capacity)
        flat_mask = expert_mask.reshape(B * T, E)
        flat_scores = scores.reshape(B * T, E)
        kept = torch.zeros_like(flat_mask)
        for expert in range(E):
            if expert < shared:
                kept[:, expert] = flat_mask[:, expert]
                continue
            active = flat_mask[:, expert].bool()
            if not active.any():
                continue
            active_idx = active.nonzero(as_tuple=False).squeeze(-1)
            if active_idx.numel() > capacity:
                expert_scores = flat_scores[active_idx, expert]
                active_idx = active_idx[torch.topk(expert_scores, capacity).indices]
            kept[active_idx, expert] = 1.0
        return kept.view_as(expert_mask)

    def _compute_dense_pre_mask_activation(
        self, attn, activated, state_trace, prev_sync_o_activated,
    ):
        B, T, _ = activated.shape
        device = activated.device
        pre_syn_parts = [attn, activated]
        if self.self_cond:
            if prev_sync_o_activated is not None:
                pre_syn_parts.append(self.self_cond_proj(prev_sync_o_activated))
            else:
                pre_syn_parts.append(torch.zeros(B, T, self.d_model, device=device))
        pre_syn = torch.cat(pre_syn_parts, dim=-1)
        state = self.synapses(pre_syn)
        new_trace = torch.cat(
            [state_trace[:, :, :, 1:], state.unsqueeze(-1)], dim=-1)
        return self.trace_processor(new_trace), new_trace

    def _run_group_sparse_regional_tick(
        self, attn, activated, state_trace, prev_sync_o_activated,
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
