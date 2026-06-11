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

        r_out = torch.exp(-torch.clamp(self.decay_params_out, 0, 15)).unsqueeze(0).repeat(B, 1)
        _, decay_alpha_out, decay_beta_out = self.compute_synchronisation(activated_state, None, None, r_out, synch_type='out')

        for stepi in range(self.iterations):

            async_mask = None
            if async_mode == 'banded' and async_periods is not None:
                periods = [int(p) for p in async_periods.split(',')]
                phases = [int(p) for p in async_phases.split(',')] if async_phases else None
                async_mask = get_async_tick_mask(stepi, d_model, periods, phases, device=device)

            pre_synapse_input = torch.concatenate((x, activated_state), dim=-1)
            if async_mask is not None:
                pre_synapse_input = torch.concatenate((x, activated_state * async_mask.unsqueeze(0).float()), dim=-1)

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

            synchronisation_out, decay_alpha_out, decay_beta_out = self.compute_synchronisation(activated_state, decay_alpha_out, decay_beta_out, r_out, synch_type='out')

            current_prediction = self.output_projector(synchronisation_out)
            current_certainty = self.compute_certainty(current_prediction)

            predictions[..., stepi] = current_prediction
            certainties[..., stepi] = current_certainty

            # Draft-revise: save draft at block boundary, corrupt state
            if draft_mode == 'revise':
                from baseline.utils.ctm_model_ideas import apply_draft_revise_corruption
                draft_block_size = getattr(self, 'draft_block_size', 2)
                corrupt_prob = getattr(self, 'draft_corrupt_prob', 0.0)
                _saved, activated_state = apply_draft_revise_corruption(
                    stepi, draft_block_size, activated_state, corrupt_prob)
                if _saved:
                    draft_pred = current_prediction.detach()

            if return_per_tick_synch:
                synch_per_tick.append(synchronisation_out)

            if use_reflex and stepi < getattr(self, 'reflex_ticks', 1):
                rp = self.reflex_head(synchronisation_out)
                reflex_preds.append(rp)

            if should_halt(certainties, stepi, min_ticks, halt_threshold, halt_mode):
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
        if n_steps_used < self.iterations:
            extras['n_steps_used'] = n_steps_used

        if track:
            base = (predictions, certainties, np.array(synch_out_tracking),
                    np.array(pre_activations_tracking), np.array(post_activations_tracking), np.array(attention_tracking))
            return base + (extras,)

        if extras:
            return predictions, certainties, synchronisation_out, extras
        return predictions, certainties, synchronisation_out
