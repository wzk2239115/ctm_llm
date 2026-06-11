import torch
import numpy as np
from baseline.models.ctm import ContinuousThoughtMachine
from baseline.models.modules import MNISTBackbone, QAMNISTIndexEmbeddings, QAMNISTOperatorEmbeddings
from baseline.utils.ctm_model_ideas import apply_topk_sparsity, get_async_tick_mask, should_halt

class ContinuousThoughtMachineQAMNIST(ContinuousThoughtMachine):
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
                 out_dims,
                 iterations_per_digit,
                 iterations_per_question_part,
                 iterations_for_answering,
                 prediction_reshaper=[-1],
                 dropout=0,
                 neuron_select_type='first-last',
                 n_random_pairing_self=256
                 ):
        super().__init__(
            iterations=iterations,
            d_model=d_model,
            d_input=d_input,
            heads=heads,
            n_synch_out=n_synch_out,
            n_synch_action=n_synch_action,
            synapse_depth=synapse_depth,
            memory_length=memory_length,
            deep_nlms=deep_nlms,
            memory_hidden_dims=memory_hidden_dims,
            do_layernorm_nlm=do_layernorm_nlm,
            out_dims=out_dims,
            prediction_reshaper=prediction_reshaper,
            dropout=dropout,
            neuron_select_type=neuron_select_type,
            n_random_pairing_self=n_random_pairing_self,
            backbone_type='none',
            positional_embedding_type='none',
        )

        # --- Core Parameters ---
        self.iterations_per_digit = iterations_per_digit
        self.iterations_per_question_part = iterations_per_question_part
        self.iterations_for_answering = iterations_for_answering

    # --- Setup Methods ---

    def set_initial_rgb(self):
        """Set the initial RGB values for the backbone."""
        return None

    def get_d_backbone(self):
        """Get the dimensionality of the backbone output."""
        return self.d_input

    def set_backbone(self):
        """Set the backbone module based on the specified type."""
        self.backbone_digit = MNISTBackbone(self.d_input)
        self.index_backbone = QAMNISTIndexEmbeddings(50, self.d_input)
        self.operator_backbone = QAMNISTOperatorEmbeddings(2, self.d_input)
        pass

    # --- Utilty Methods ---

    def determine_step_type(self, total_iterations_for_digits, total_iterations_for_question, stepi: int):
        """Determine whether the current step is for digits, questions, or answers."""
        is_digit_step = stepi < total_iterations_for_digits
        is_question_step = total_iterations_for_digits <= stepi < total_iterations_for_digits + total_iterations_for_question
        is_answer_step = stepi >= total_iterations_for_digits + total_iterations_for_question
        return is_digit_step, is_question_step, is_answer_step

    def determine_index_operator_step_type(self, total_iterations_for_digits, stepi: int):
        """Determine whether the current step is for index or operator."""
        step_within_questions = stepi - total_iterations_for_digits
        if step_within_questions % (2 * self.iterations_per_question_part) < self.iterations_per_question_part:
            is_index_step = True
            is_operator_step = False
        else:
            is_index_step = False
            is_operator_step = True
        return is_index_step, is_operator_step

    def get_kv_for_step(self, total_iterations_for_digits, total_iterations_for_question, stepi, x, z, prev_input=None, prev_kv=None):
        """Get the key-value for the current step."""
        is_digit_step, is_question_step, is_answer_step = self.determine_step_type(total_iterations_for_digits, total_iterations_for_question, stepi)

        if is_digit_step:
            current_input = x[:, stepi]
            if prev_input is not None and torch.equal(current_input, prev_input):
                return prev_kv, prev_input
            kv = self.kv_proj(self.backbone_digit(current_input).flatten(2).permute(0, 2, 1))

        elif is_question_step:
            offset = stepi - total_iterations_for_digits
            current_input = z[:, offset]
            if prev_input is not None and torch.equal(current_input, prev_input):
                return prev_kv, prev_input
            is_index_step, is_operator_step = self.determine_index_operator_step_type(total_iterations_for_digits, stepi)
            if is_index_step:
                kv = self.index_backbone(current_input)
            elif is_operator_step:
                kv = self.operator_backbone(current_input)
            else:
                raise ValueError("Invalid step type for question processing.")

        elif is_answer_step:
            current_input = None
            kv = torch.zeros((x.size(0), self.d_input), device=x.device)

        else:
            raise ValueError("Invalid step type.")

        return kv, current_input




    def forward(self, x, z, track=False, return_per_tick_synch=False):
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
        attention_tracking = []
        embedding_tracking = []

        # --- Per-tick synch tracking ---
        if return_per_tick_synch:
            synch_per_tick = []

        total_iterations_for_digits = x.size(1)
        total_iterations_for_question = z.size(1)
        total_iterations = total_iterations_for_digits + total_iterations_for_question + self.iterations_for_answering

        # --- Initialise Recurrent State ---
        state_trace = self.start_trace.unsqueeze(0).expand(B, -1, -1)
        activated_state = self.start_activated_state.unsqueeze(0).expand(B, -1)
        d_model = activated_state.size(-1)

        # --- Storage for outputs per iteration ---
        predictions = torch.empty(B, self.out_dims, total_iterations, device=device, dtype=x.dtype)
        certainties = torch.empty(B, 2, total_iterations, device=device, dtype=x.dtype)
        n_steps_used = total_iterations
        reflex_preds = []
        draft_pred = None
        draft_mode = getattr(self, 'draft_mode', 'none')

        # --- Initialise Recurrent Synch Values  ---
        decay_alpha_action, decay_beta_action = None, None
        self.decay_params_action.data = torch.clamp(self.decay_params_action, 0, 15)
        self.decay_params_out.data = torch.clamp(self.decay_params_out, 0, 15)
        r_action, r_out = torch.exp(-self.decay_params_action).unsqueeze(0).repeat(B, 1), torch.exp(-self.decay_params_out).unsqueeze(0).repeat(B, 1)

        _, decay_alpha_out, decay_beta_out = self.compute_synchronisation(activated_state, None, None, r_out, synch_type='out')

        prev_input = None
        prev_kv = None

        # --- Recurrent Loop  ---
        for stepi in range(total_iterations):
            is_digit_step, is_question_step, is_answer_step = self.determine_step_type(total_iterations_for_digits, total_iterations_for_question, stepi)

            kv, prev_input = self.get_kv_for_step(total_iterations_for_digits, total_iterations_for_question, stepi, x, z, prev_input, prev_kv)
            prev_kv = kv

            async_mask = None
            if async_mode == 'banded' and async_periods is not None:
                periods = [int(p) for p in async_periods.split(',')]
                phases = [int(p) for p in async_phases.split(',')] if async_phases else None
                async_mask = get_async_tick_mask(stepi, d_model, periods, phases)

            synchronization_action, decay_alpha_action, decay_beta_action = self.compute_synchronisation(activated_state, decay_alpha_action, decay_beta_action, r_action, synch_type='action')

            # --- Interact with Data via Attention ---
            attn_weights = None
            if is_digit_step:
                q = self.q_proj(synchronization_action).unsqueeze(1)
                attn_out, attn_weights = self.attention(q, kv, kv, average_attn_weights=False, need_weights=True)
                attn_out = attn_out.squeeze(1)
                base_input = attn_out
            else:
                kv_sq = kv.squeeze(1)
                base_input = kv_sq

            state_input = activated_state
            if async_mask is not None:
                state_input = activated_state * async_mask.unsqueeze(0).float()
            pre_synapse_input = torch.concatenate((base_input, state_input), dim=-1)

            # --- Apply Synapses ---
            state = self.synapses(pre_synapse_input)
            if async_mask is not None:
                state = state * async_mask.unsqueeze(0).float()
            state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)

            # --- Apply NLMs ---
            if nlm_diff is not None:
                activated_state = nlm_diff(state_trace)
            else:
                activated_state = self.trace_processor(state_trace)
            if async_mask is not None:
                activated_state = activated_state * async_mask.unsqueeze(0).float()

            # --- Top-k Sparsity ---
            if topk < 1.0:
                activated_state = apply_topk_sparsity(activated_state, topk, stepi)

            # --- Calculate Synchronisation for Output Predictions ---
            synchronization_out, decay_alpha_out, decay_beta_out = self.compute_synchronisation(activated_state, decay_alpha_out, decay_beta_out, r_out, synch_type='out')

            # --- Get Predictions and Certainties ---
            current_prediction = self.output_projector(synchronization_out)
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

            # --- Per-tick synch tracking ---
            if return_per_tick_synch:
                synch_per_tick.append(synchronization_out)

            # --- Reflex head ---
            if use_reflex and stepi < getattr(self, 'reflex_ticks', 1):
                rp = self.reflex_head(synchronization_out)
                reflex_preds.append(rp)

            # --- Tick halt ---
            if should_halt(certainties, stepi, min_ticks, halt_threshold, halt_mode):
                n_steps_used = stepi + 1
                if stepi + 1 < total_iterations:
                    predictions[..., stepi+1:] = 0
                break

            # --- Tracking ---
            if track:
                pre_activations_tracking.append(state_trace[:,:,-1].detach().cpu().numpy())
                post_activations_tracking.append(activated_state.detach().cpu().numpy())
                if attn_weights is not None:
                    attention_tracking.append(attn_weights.detach().cpu().numpy())
                if is_question_step:
                    embedding_tracking.append(kv.detach().cpu().numpy())

        # --- Return Values ---
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
            base = (predictions, certainties, synchronization_out,
                    np.array(pre_activations_tracking), np.array(post_activations_tracking),
                    np.array(attention_tracking), np.array(embedding_tracking))
            return base + (extras,) if extras else base

        if extras:
            return predictions, certainties, synchronization_out, extras
        return predictions, certainties, synchronization_out