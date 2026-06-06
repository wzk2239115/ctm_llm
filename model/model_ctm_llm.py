import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.config import CTMLLMConfig
from model.ctm_modules import SuperLinear, SynapseUNET, Squeeze, TTTMLP


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


class CTMBlock(nn.Module):
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
        if config.deep_nlms:
            return nn.Sequential(
                SuperLinear(config.memory_length, 2 * config.memory_hidden_dims,
                            config.d_model, dropout=config.dropout),
                nn.GLU(),
                SuperLinear(config.memory_hidden_dims, 2,
                            config.d_model, dropout=config.dropout),
                nn.GLU(),
                Squeeze(-1))
        return nn.Sequential(
            SuperLinear(config.memory_length, 2, config.d_model, dropout=config.dropout),
            nn.GLU(),
            Squeeze(-1))

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

    def forward(self, x, pos_emb=None, past_kv=None, use_cache=False, track=False,
                num_iters=None, return_all_ticks=False,
                prev_activated=None, prev_trace=None):
        B, T, _ = x.shape
        device = x.device

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

            attn = self.attn_drop(
                self.o_proj(attn_out.transpose(1, 2).reshape(B, T, -1)))

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

            sync_o, alpha_o, beta_o = self._compute_synch(
                activated, alpha_o, beta_o, r_o, self.out_left, self.out_right)
            prev_sync_o_activated = activated
            tick_out = self.output_proj(sync_o)
            if all_tick_outs is not None:
                all_tick_outs.append(tick_out)

            if track:
                tracking['pre_activations'].append(state[0].detach().cpu().numpy())
                tracking['post_activations'].append(activated[0].detach().cpu().numpy())
                tracking['sync_action'].append(sync_a[0].detach().cpu().numpy())
                tracking['state_trace'].append(state_trace[0].detach().cpu().numpy())

        ctm_out = all_tick_outs[-1] if all_tick_outs is not None else \
            self.output_proj(self._compute_synch(
                activated, alpha_o, beta_o, r_o, self.out_left, self.out_right)[0])
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
                num_iters=None, return_all_ticks=False):
        B, T = input_ids.shape
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        h = self.drop(self.embed_tokens(input_ids))
        pos_emb = self.pos_embed.weight
        presents = []
        tracking_all = {}
        last_tick_outs = None

        prev_activated = None
        prev_trace = None

        for layer, past_kv in zip(self.layers, past_key_values):
            is_last = layer.layer_id == len(self.layers) - 1
            layer_kwargs = dict(
                pos_emb=pos_emb, past_kv=past_kv,
                use_cache=use_cache, num_iters=num_iters,
                prev_activated=prev_activated if self.config.cross_layer_state else None,
                prev_trace=prev_trace if self.config.cross_layer_state else None,
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

            presents.append(present)

        h = self.norm(h)

        outputs = [h, presents]
        if return_all_ticks:
            outputs.append(last_tick_outs)
        if track:
            outputs.append(tracking_all)

        if len(outputs) == 2:
            return tuple(outputs)
        return tuple(outputs)


class CTMForCausalLM(nn.Module):
    def __init__(self, config: CTMLLMConfig):
        super().__init__()
        self.config = config
        self.model = CTMModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, past_key_values=None, use_cache=False, labels=None,
                num_iters=None):
        h, past_key_values = self.model(
            input_ids, past_key_values, use_cache, track=False, num_iters=num_iters)
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
        h, _, tick_outs = self.model(
            input_ids, track=False, num_iters=num_iters, return_all_ticks=True)
        num_ticks = tick_outs.size(-1)

        shift_labels = labels[..., 1:].contiguous()
        label_mask = (shift_labels != -100)

        final_logits = self.lm_head(h)
        final_shift_logits = final_logits[..., :-1, :].contiguous()
        final_loss = F.cross_entropy(
            final_shift_logits.view(-1, final_shift_logits.size(-1)),
            shift_labels.view(-1), ignore_index=-100)

        losses = []
        certainties = []
        for t in range(num_ticks):
            logits_t = self.lm_head(tick_outs[..., t])
            shift_logits = logits_t[..., :-1, :].contiguous()
            per_token_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1), ignore_index=-100, reduction='none')
            per_token_loss = per_token_loss.view(B, -1)
            per_sample_loss = (per_token_loss * label_mask).sum(dim=1) / label_mask.sum(dim=1).clamp(min=1)
            losses.append(per_sample_loss)

            probs = F.softmax(logits_t, dim=-1)
            entropy = -(probs * torch.log(probs.clamp(min=1e-12))).sum(-1)
            norm_ent = entropy / math.log(logits_t.size(-1))
            norm_ent_valid = (norm_ent[..., :-1] * label_mask).sum(dim=1) / label_mask.sum(dim=1).clamp(min=1)
            certainties.append(norm_ent_valid)

        losses = torch.stack(losses, dim=1)
        certainties = torch.stack(certainties, dim=1)
        confidence = 1 - certainties

        loss_min = losses.min(dim=1).values.mean()
        best_conf_tick = confidence.argmax(dim=1)
        batch_idx = torch.arange(B, device=losses.device)
        loss_conf = losses[batch_idx, best_conf_tick].mean()

        tick_loss = (loss_min + loss_conf) / 2.0
        loss = 0.5 * final_loss + 0.5 * tick_loss

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
