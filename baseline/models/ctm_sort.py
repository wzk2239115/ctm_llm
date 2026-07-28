import torch
import numpy as np
from baseline.models.ctm import ContinuousThoughtMachine
from baseline.utils.ctm_model_ideas import apply_topk_sparsity, get_async_tick_mask, should_halt

class ContinuousThoughtMachineSORT(ContinuousThoughtMachine):
    """
    Slight adaption of the CTM to work with the sort task.
    """                               

    def __init__(self,
                 iterations,
                 d_model,
                 d_input,
                 heads,
                 n_synch_out,
                 n_synch_action,
                 synapse_depth,
                 memory_length,
                 deep_nlms,
                 memory_hidden_dims,
                 do_layernorm_nlm,
                 backbone_type,
                 positional_embedding_type,
                 out_dims,
                 prediction_reshaper=[-1],
                 dropout=0,
                 dropout_nlm=None,
                 neuron_select_type='random-pairing',  
                 n_random_pairing_self=0,
                 synch_gate_mode='fixed',
                 synch_gate_temp=1.0,
                 ):
        super().__init__(
            iterations=iterations,
            d_model=d_model,
            d_input=d_input,
            heads=0,
            n_synch_out=n_synch_out,
            n_synch_action=0,
            synapse_depth=synapse_depth,
            memory_length=memory_length,
            deep_nlms=deep_nlms,
            memory_hidden_dims=memory_hidden_dims,
            do_layernorm_nlm=do_layernorm_nlm,
            backbone_type='none',
            positional_embedding_type='none',
            out_dims=out_dims,
            prediction_reshaper=prediction_reshaper,
            dropout=dropout,
            dropout_nlm=dropout_nlm,
            neuron_select_type=neuron_select_type,
            n_random_pairing_self=n_random_pairing_self,
            synch_gate_mode=synch_gate_mode,
            synch_gate_temp=synch_gate_temp,
        )

        # --- Use a minimal CTM w/out input (action) synch ---
        self.neuron_select_type_action = None
        self.synch_representation_size_action = None

        self.attention = None  # Should already be None because super(... heads=0... ) 
        self.q_proj = None  # Should already be None because super(... heads=0... ) 
        self.kv_proj = None  # Should already be None because super(... heads=0... ) 




    def forward(self, x, track=False, return_per_tick_synch=False):
        B = x.size(0)
        device = x.device

        topk = getattr(self, 'topk_neurons', 1.0)
        async_mode = getattr(self, 'async_tick_mode', 'none')
        async_periods = getattr(self, 'async_tick_periods', None)
        async_phases = getattr(self, 'async_tick_phases', None)
        halt_mode = getattr(self, 'tick_halt_mode', 'none')
        halt_threshold = getattr(self, 'tick_halt_threshold', 0.0)
        min_ticks = getattr(self, 'tick_min_ticks', 1)
        use_reflex = hasattr(self, 'reflex_head') and self.reflex_head is not None
        nlm_diff = getattr(self, 'nlm_differentiated', None)

        # --- HRM-inspired config ---
        bp_steps_cfg = getattr(self, 'bp_steps', 0)
        detach_every = getattr(self, 'detach_every', 0)
        effective_bp_steps = min(bp_steps_cfg, self.iterations) if bp_steps_cfg > 0 else self.iterations
        h_cycles = getattr(self, 'h_cycles', 1)
        l_cycles = getattr(self, 'l_cycles', 0)
        h_synapse = getattr(self, 'h_synapse', None)
        use_hierarchical = h_cycles > 1 and h_synapse is not None
        q_head = getattr(self, 'q_head', None)
        act_halt = getattr(self, 'act_halt', False)
        halt_max_steps = getattr(self, 'halt_max_steps', self.iterations)
        halt_exploration_prob = getattr(self, 'halt_exploration_prob', 0.0)

        # --- Multi-Scale Hierarchy (N-level) ---
        from baseline.utils.dtt_ideas import parse_msh_levels, should_update_level, should_update_level_coprime
        msh_levels_str = getattr(self, 'msh_levels', '')
        msh_levels = parse_msh_levels(msh_levels_str)
        msh_mode = getattr(self, 'msh_mode', 'nested')
        use_msh = msh_levels is not None and hasattr(self, 'msh_synapses')
        msh_sn_scale = getattr(self, 'msh_sn_scale', 0.0)

        # --- Tracking Initialization ---
        pre_activations_tracking = []
        post_activations_tracking = []
        synch_out_tracking = []
        attention_tracking = []

        # --- Per-tick synch tracking ---
        if return_per_tick_synch:
            synch_per_tick = []

        # --- Initialise Recurrent State ---
        state_trace = self.start_trace.unsqueeze(0).expand(B, -1, -1)
        activated_state = self.start_activated_state.unsqueeze(0).expand(B, -1)
        d_model = activated_state.size(-1)

        predictions = torch.empty(B, self.out_dims, self.iterations, device=device, dtype=x.dtype)
        certainties = torch.empty(B, 2, self.iterations, device=device, dtype=x.dtype)
        n_steps_used = self.iterations
        reflex_preds = []
        draft_pred = None
        draft_mode = getattr(self, 'draft_mode', 'none')
        q_logits_all = [] if (act_halt and q_head is not None) else None
        halted_mask = torch.zeros(B, dtype=torch.bool, device=device) if act_halt else None
        halt_step = torch.full((B,), self.iterations, dtype=torch.long, device=device) if act_halt else None

        r_out = torch.exp(-torch.clamp(self.decay_params_out, 0, 15)).unsqueeze(0).repeat(B, 1)
        _, decay_alpha_out, decay_beta_out = self.compute_synchronisation(activated_state, None, None, r_out, synch_type='out')

        # --- Hierarchical: initialize H-level state ---
        z_H = torch.zeros_like(activated_state) if use_hierarchical else None
        eff_l_cycles = l_cycles if l_cycles > 0 else max(1, self.iterations // max(1, h_cycles))

        # --- MSH: initialize macro states ---
        # Nested mode: n_macro = len(levels) - 1 (innermost level IS the tick loop)
        # Coprime mode: n_macro = len(levels) (all levels are independent overlay states)
        if use_msh:
            if msh_mode == 'coprime':
                n_macro = len(msh_levels)
            else:
                n_macro = len(msh_levels) - 1
            msh_states = [torch.zeros_like(activated_state) for _ in range(n_macro)]

        for stepi in range(self.iterations):

            # --- Truncated BPTT + State detach ---
            if detach_every > 0 and stepi > 0 and stepi % detach_every == 0:
                state_trace = state_trace.detach()
                activated_state = activated_state.detach()
            grad_enabled_this_tick = stepi >= self.iterations - effective_bp_steps
            with torch.set_grad_enabled(torch.is_grad_enabled() and grad_enabled_this_tick):

                async_mask = None
                if async_mode == 'banded' and async_periods is not None:
                    periods = [int(p) for p in async_periods.split(',')]
                    phases = [int(p) for p in async_phases.split(',')] if async_phases else None
                    async_mask = get_async_tick_mask(stepi, d_model, periods, phases, device=device)

                # --- Hierarchical / MSH: inject macro-level state(s) ---
                if use_msh:
                    state_for_syn = activated_state
                    for ms in msh_states:
                        state_for_syn = state_for_syn + ms
                elif z_H is not None:
                    state_for_syn = activated_state + z_H
                else:
                    state_for_syn = activated_state
                if async_mask is not None:
                    state_for_syn = state_for_syn * async_mask.unsqueeze(0).float()
                pre_synapse_input = torch.concatenate((x, state_for_syn), dim=-1)

                state = self.synapses(pre_synapse_input)
                if async_mask is not None:
                    state = state * async_mask.unsqueeze(0).float()
                state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)

                if nlm_diff is not None:
                    activated_state = nlm_diff(state_trace)
                else:
                    activated_state = self.trace_processor(state_trace)
                if async_mask is not None:
                    activated_state = activated_state * async_mask.unsqueeze(0).float()

                if topk < 1.0:
                    activated_state = apply_topk_sparsity(activated_state, topk, stepi)

                # --- Hierarchical: H-level update every l_cycles ticks ---
                if z_H is not None and (stepi + 1) % eff_l_cycles == 0:
                    z_H = z_H + h_synapse(activated_state)

                # --- MSH: update macro states at level boundaries ---
                if use_msh:
                    for level_idx in range(n_macro):
                        if msh_mode == 'learnable':
                            gate_logits = getattr(self, 'msh_gate_logits', None)
                            if gate_logits is not None:
                                gate = torch.sigmoid(gate_logits[level_idx, stepi])
                                update = self.msh_synapses[level_idx](activated_state)
                                if msh_sn_scale > 0:
                                    update = update * msh_sn_scale
                                msh_states[level_idx] = msh_states[level_idx] + gate * update
                        elif msh_mode == 'coprime':
                            if should_update_level_coprime(stepi, msh_levels, level_idx):
                                update = self.msh_synapses[level_idx](activated_state)
                                if msh_sn_scale > 0:
                                    update = update * msh_sn_scale
                                msh_states[level_idx] = msh_states[level_idx] + update
                        else:
                            if should_update_level(stepi, msh_levels, level_idx):
                                update = self.msh_synapses[level_idx](activated_state)
                                if msh_sn_scale > 0:
                                    update = update * msh_sn_scale
                                msh_states[level_idx] = msh_states[level_idx] + update

                synchronisation_out, decay_alpha_out, decay_beta_out = self.compute_synchronisation(activated_state, decay_alpha_out, decay_beta_out, r_out, synch_type='out')

                current_prediction = self.output_projector(synchronisation_out)
                current_certainty = self.compute_certainty(current_prediction)

                predictions[..., stepi] = current_prediction
                certainties[..., stepi] = current_certainty

                # --- ACT Q-learning: compute Q-head logits ---
                if q_logits_all is not None:
                    q_logits_t = q_head(synchronisation_out)  # (B, 2)
                    q_logits_all.append(q_logits_t)

                    # --- Q-learning halting decision (training only) ---
                    if self.training and halt_max_steps > 1:
                        q_halt_logit = q_logits_t[:, 0]
                        q_continue_logit = q_logits_t[:, 1]
                        halt_decision = q_halt_logit > q_continue_logit

                        if halt_exploration_prob > 0:
                            min_halt = torch.randint(2, halt_max_steps + 1, (B,), device=device)
                            explore = (torch.rand(B, device=device) < halt_exploration_prob) & (stepi >= min_halt)
                            halt_decision = halt_decision | explore

                        halt_decision = halt_decision | (stepi >= halt_max_steps - 1)

                        newly_halted = halt_decision & (~halted_mask)
                        halted_mask = halted_mask | newly_halted
                        halt_step[newly_halted] = stepi

                # Draft-revise: save draft at block boundary, corrupt state
                if draft_mode == 'revise':
                    from baseline.utils.ctm_model_ideas import apply_draft_revise_corruption
                    draft_block_size = getattr(self, 'draft_block_size', 2)
                    corrupt_prob = getattr(self, 'draft_corrupt_prob', 0.0)
                    _saved, activated_state = apply_draft_revise_corruption(
                        stepi, draft_block_size, activated_state, corrupt_prob)
                    if _saved:
                        draft_pred = current_prediction

                if return_per_tick_synch:
                    synch_per_tick.append(synchronisation_out)

                if use_reflex and stepi < getattr(self, 'reflex_ticks', 1):
                    rp = self.reflex_head(synchronisation_out)
                    reflex_preds.append(rp)

                # --- Tick Halt ---
                if act_halt and halted_mask is not None:
                    if self.training and halted_mask.all():
                        n_steps_used = stepi + 1
                        if stepi + 1 < self.iterations:
                            predictions[..., stepi+1:] = 0
                        break
                elif should_halt(certainties, stepi, min_ticks, halt_threshold, halt_mode):
                    n_steps_used = stepi + 1
                    if stepi + 1 < self.iterations:
                        predictions[..., stepi+1:] = 0
                    break

                if track:
                    pre_activations_tracking.append(state_trace[:,:,-1].detach().cpu().numpy())
                    post_activations_tracking.append(activated_state.detach().cpu().numpy())
                    synch_out_tracking.append(synchronisation_out.detach().cpu().numpy())

        extras = {}
        if return_per_tick_synch:
            extras['synch_per_tick'] = torch.stack(synch_per_tick, dim=-1)
        if use_reflex and reflex_preds:
            extras['reflex_preds'] = torch.stack(reflex_preds, dim=-1)
        if draft_pred is not None:
            extras['draft_prediction'] = draft_pred
        if q_logits_all is not None:
            extras['q_logits'] = torch.stack(q_logits_all, dim=-1)
        if halt_step is not None:
            extras['halt_step'] = halt_step
            extras['halted_mask'] = halted_mask
        if n_steps_used < self.iterations:
            extras['n_steps_used'] = n_steps_used

        if track:
            base = (predictions, certainties, np.array(synch_out_tracking),
                    np.array(pre_activations_tracking), np.array(post_activations_tracking), np.array(attention_tracking))
            return base + (extras,)

        if extras:
            return predictions, certainties, synchronisation_out, extras
        return predictions, certainties, synchronisation_out
