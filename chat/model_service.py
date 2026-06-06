import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

import math
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
from model.config import CTMLLMConfig
from model.model_ctm_llm import CTMForCausalLM


class CTMChatService:
    def __init__(self, weight_path, tokenizer_path='./model_tokenizer', device='cuda:0',
                 hidden_size=1024, num_hidden_layers=16, d_model=512, d_input=256,
                 iterations=4, memory_length=5, heads=8,
                 n_synch_out=512, n_synch_action=512, synapse_depth=2,
                 self_cond=True, cross_layer_state=True, block_size=4,
                 ttt_layer=False, ttt_hidden_mult=2, ttt_gate_init=-2.0,
                 num_iters=None):
        self.device = device
        self.num_iters = num_iters
        self.iterations = num_iters if num_iters is not None else iterations

        config = CTMLLMConfig(
            vocab_size=6400,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            d_model=d_model,
            d_input=d_input,
            iterations=iterations,
            memory_length=memory_length,
            heads=heads,
            n_synch_out=n_synch_out,
            n_synch_action=n_synch_action,
            synapse_depth=synapse_depth,
            self_cond=self_cond,
            cross_layer_state=cross_layer_state,
            block_size=block_size,
            ttt_layer=ttt_layer,
            ttt_hidden_mult=ttt_hidden_mult,
            ttt_gate_init=ttt_gate_init,
        )

        self.model = CTMForCausalLM(config).to(device)
        ckpt = torch.load(weight_path, map_location=device, weights_only=False)
        state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.config = config
        self.log_vocab = math.log(config.vocab_size)
        print(f'[CTMChatService] Loaded: {weight_path}')
        print(f'  iterations={self.iterations}, '
              f'layers={config.num_hidden_layers}, d_model={config.d_model}')

    def build_prompt_ids(self, messages):
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        ids = self.tokenizer.encode(text)
        return torch.tensor([ids], dtype=torch.long, device=self.device), text

    def _per_tick_confidence(self, tick_outs_last_pos):
        tick_outs_last_pos: torch.Tensor  # (B=1, hidden_size, num_ticks)
        num_ticks = tick_outs_last_pos.size(-1)
        confidences = []
        logits_list = []
        for t in range(num_ticks):
            h_t = tick_outs_last_pos[:, :, t]
            logits_t = self.model.lm_head(h_t.unsqueeze(1))[:, 0, :]
            probs = F.softmax(logits_t, dim=-1)
            entropy = -(probs * torch.log(probs.clamp(min=1e-12))).sum(dim=-1)
            norm_ent = entropy / self.log_vocab
            conf = (1.0 - norm_ent).squeeze(0).item()
            confidences.append(conf)
            logits_list.append(logits_t.squeeze(0))
        return confidences, logits_list

    def _select_tick(self, tick_outs, confidence_threshold):
        if tick_outs is None:
            return 0, [0.0], None

        num_ticks = tick_outs.size(-1)
        last_pos_outs = tick_outs[:, -1, :, :]  # (1, hidden, num_ticks)
        confidences, logits_list = self._per_tick_confidence(last_pos_outs)

        selected = num_ticks - 1
        for t in range(num_ticks):
            if confidences[t] >= confidence_threshold:
                selected = t
                break

        return selected, confidences, logits_list[selected]

    def _apply_sampling(self, token_logits, input_ids, temperature, top_k, top_p,
                        repetition_penalty):
        token_logits = token_logits / temperature

        if repetition_penalty != 1.0:
            seen = torch.unique(input_ids[0])
            score = token_logits[0, seen]
            token_logits[0, seen] = torch.where(
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
        return torch.multinomial(probs, num_samples=1)

    def _compute_tick_metrics(self, tick_outs, labels):
        num_ticks = tick_outs.size(-1)
        B = tick_outs.size(0)
        shift_labels = labels[..., 1:].contiguous()
        label_mask = (shift_labels != -100)

        tick_losses = []
        tick_entropies = []
        tick_confidences = []
        tick_top_tokens = []

        for t in range(num_ticks):
            logits_t = self.model.lm_head(tick_outs[..., t])
            shift_logits = logits_t[..., :-1, :].contiguous()

            per_token_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1), ignore_index=-100, reduction='none')
            per_token_loss = per_token_loss.view(B, -1)
            per_sample_loss = (per_token_loss * label_mask).sum(1) / label_mask.sum(1).clamp(min=1)
            tick_losses.append(per_sample_loss.mean().item())

            probs = F.softmax(logits_t, dim=-1)
            entropy = -(probs * torch.log(probs.clamp(min=1e-12))).sum(-1)
            norm_ent = entropy / self.log_vocab
            norm_ent_valid = (norm_ent[..., :-1] * label_mask).sum(1) / label_mask.sum(1).clamp(min=1)
            mean_ent = norm_ent_valid.mean().item()
            tick_entropies.append(mean_ent)
            tick_confidences.append(1.0 - mean_ent)

            last_logits = logits_t[0, -1]
            top5 = torch.topk(last_logits, 5)
            top_tokens = []
            for idx, score in zip(top5.indices.tolist(), top5.values.tolist()):
                tok = self.tokenizer.decode([idx]).strip()
                top_tokens.append({'token': tok, 'score': round(score, 3)})
            tick_top_tokens.append(top_tokens)

        return tick_losses, tick_entropies, tick_confidences, tick_top_tokens

    def _extract_attention_info(self, tracking_all, input_ids_tensor):
        attention_data = {}
        for layer_name, tracking in tracking_all.items():
            if 'post_activations' not in tracking:
                continue
            post_act = tracking['post_activations']
            num_ticks = len(post_act)
            layer_id = int(layer_name.split('_')[1])

            act_variance = []
            for t in range(num_ticks):
                act = post_act[t]
                variance = float(np.var(act, axis=-1).mean())
                act_variance.append(round(variance, 6))

            sync_action = tracking.get('sync_action', [])
            sync_norms = []
            for t in range(min(len(sync_action), num_ticks)):
                sa = sync_action[t]
                sync_norms.append(round(float(np.linalg.norm(sa[-1])), 4))

            attention_data[layer_name] = {
                'activation_variance': act_variance,
                'sync_norm': sync_norms,
                'num_ticks': num_ticks,
                'layer_id': layer_id,
            }

        return attention_data

    def _extract_neuron_firing(self, tracking_all, last_n=32):
        last_layer = f'layer_{self.config.num_hidden_layers - 1}'
        if last_layer not in tracking_all:
            return None
        tracking = tracking_all[last_layer]
        post_act = tracking.get('post_activations')
        if post_act is None:
            return None

        num_ticks = len(post_act)
        last_pos_act = post_act[num_ticks - 1]
        neuron_acts = last_pos_act[-1]
        top_indices = np.argsort(np.abs(neuron_acts))[-last_n:]
        top_indices = np.sort(top_indices)

        firing_grid = np.zeros((num_ticks, len(top_indices)))
        for t in range(num_ticks):
            act_t = post_act[t][-1]
            firing_grid[t] = act_t[top_indices]

        return {
            'neuron_ids': top_indices.tolist(),
            'firing_grid': firing_grid.tolist(),
            'num_ticks': num_ticks,
        }

    def _extract_all_neuron_activations(self, tracking_all):
        last_layer = f'layer_{self.config.num_hidden_layers - 1}'
        if last_layer not in tracking_all:
            return None
        tracking = tracking_all[last_layer]
        post_act = tracking.get('post_activations')
        if post_act is None:
            return None

        num_ticks = len(post_act)
        d_model = self.config.d_model

        activations = np.zeros((num_ticks, d_model), dtype=np.float32)
        for t in range(num_ticks):
            act_t = post_act[t][-1]
            activations[t] = act_t

        all_vals = activations.flatten()
        vmin = float(np.percentile(np.abs(all_vals), 5))
        vmax = float(np.percentile(np.abs(all_vals), 95))
        if vmax <= vmin:
            vmax = vmin + 1e-6

        sync_pairs_left = []
        sync_pairs_right = []
        if hasattr(tracking_all.get(last_layer, {}), 'get'):
            pass
        layer_obj = self.model.model.layers[self.config.num_hidden_layers - 1]
        n_show = min(64, layer_obj.out_left.size(0))
        sync_pairs_left = layer_obj.out_left[:n_show].tolist()
        sync_pairs_right = layer_obj.out_right[:n_show].tolist()

        return {
            'activations': (activations / vmax).clip(-1, 1).tolist(),
            'num_ticks': num_ticks,
            'd_model': d_model,
            'sync_pairs_left': sync_pairs_left,
            'sync_pairs_right': sync_pairs_right,
        }

    def _extract_trace_memory(self, tracking_all, neuron_indices=None):
        last_layer = f'layer_{self.config.num_hidden_layers - 1}'
        if last_layer not in tracking_all:
            return None
        tracking = tracking_all[last_layer]
        state_traces = tracking.get('state_trace')
        if state_traces is None:
            return None

        num_ticks = len(state_traces)
        last_trace = state_traces[num_ticks - 1]
        trace_last_pos = last_trace[-1]

        if neuron_indices is None:
            variance_per_neuron = np.var(trace_last_pos, axis=-1)
            top_n = min(16, len(variance_per_neuron))
            neuron_indices = np.argsort(variance_per_neuron)[-top_n:]
            neuron_indices = np.sort(neuron_indices)

        memory_data = []
        for n_idx in neuron_indices:
            memory_data.append({
                'neuron_id': int(n_idx),
                'final_trace': trace_last_pos[n_idx].tolist(),
            })

        return {
            'neurons': memory_data,
            'memory_length': self.config.memory_length,
        }

    def _get_tick_outs_from_model(self, inp, past_kv_list, use_cache=True):
        result = self.model.model(
            inp, past_key_values=past_kv_list, use_cache=use_cache,
            num_iters=self.num_iters, return_all_ticks=True)

        if len(result) == 4:
            h, past_kv, tick_outs, tracking = result
            return h, past_kv, tick_outs
        elif len(result) == 3:
            h, past_kv, third = result
            if isinstance(third, torch.Tensor) and third.dim() == 4:
                return h, past_kv, third
            return h, past_kv, None
        return result[0], result[1], None

    @torch.inference_mode()
    def generate_stream(self, messages, max_new_tokens=128, temperature=0.3,
                        top_p=0.8, top_k=40, repetition_penalty=1.08,
                        confidence_threshold=0.8):
        input_ids, prompt_text = self.build_prompt_ids(messages)
        prompt_len = input_ids.size(1)
        num_total_ticks = self.iterations

        result = self.model.model(
            input_ids, track=True, num_iters=self.num_iters,
            return_all_ticks=True, use_cache=True)

        if len(result) == 4:
            h, past_kv, tick_outs, tracking_all = result
        elif len(result) == 3:
            h, past_kv, extra = result
            if isinstance(extra, torch.Tensor) and extra.dim() == 4:
                tick_outs = extra
                tracking_all = {}
            else:
                tick_outs = None
                tracking_all = extra if isinstance(extra, dict) else {}
        else:
            h, past_kv = result[:2]
            tick_outs = None
            tracking_all = {}

        tick_metrics = None
        if tick_outs is not None:
            labels = input_ids.clone()
            tick_losses, tick_entropies, tick_confidences, tick_top_tokens = \
                self._compute_tick_metrics(tick_outs, labels)
            tick_metrics = {
                'losses': tick_losses,
                'entropies': tick_entropies,
                'confidences': tick_confidences,
                'top_tokens': tick_top_tokens,
            }

        attention_data = self._extract_attention_info(tracking_all, input_ids)
        neuron_firing = self._extract_neuron_firing(tracking_all)
        trace_memory = self._extract_trace_memory(tracking_all)
        neuron_activations = self._extract_all_neuron_activations(tracking_all)

        if neuron_activations is None:
            print(f'[WARN] neuron_activations is None, tracking_all keys: {list(tracking_all.keys())}')
        else:
            print(f'[OK] neuron_activations: d_model={neuron_activations["d_model"]}, '
                  f'ticks={neuron_activations["num_ticks"]}, '
                  f'sync_pairs={len(neuron_activations["sync_pairs_left"])}')

        full_tracking = {
            'attention': attention_data,
            'neuron_firing': neuron_firing,
            'trace_memory': trace_memory,
            'tick_metrics': tick_metrics,
            'neuron_activations': neuron_activations,
        }

        selected_tick, tick_confs, selected_logits = self._select_tick(
            tick_outs, confidence_threshold)

        if selected_logits is not None:
            token_logits = selected_logits.unsqueeze(0)
        else:
            logits = self.model.lm_head(h)
            token_logits = logits[:, -1:, :]

        new_token = self._apply_sampling(
            token_logits, input_ids, temperature, top_k, top_p, repetition_penalty)

        input_ids = torch.cat([input_ids, new_token], dim=-1)
        first_text = self.tokenizer.decode(new_token[0].tolist(), skip_special_tokens=True)

        yield {
            'type': 'metadata',
            'tick_metrics': tick_metrics,
            'tracking': full_tracking,
            'prompt_tokens': prompt_len,
            'num_ticks': num_total_ticks,
            'confidence_threshold': confidence_threshold,
        }

        yield {
            'type': 'token',
            'text': first_text,
            'tick': selected_tick,
            'confidence': tick_confs[selected_tick] if tick_confs else 0,
            'tick_confidences': tick_confs,
            'early_stop': selected_tick < num_total_ticks - 1,
        }

        past_kv_list = list(past_kv) if past_kv else [None] * self.config.num_hidden_layers

        for step in range(max_new_tokens - 1):
            inp = input_ids[:, -1:]
            _, new_past_kv, tick_outs_step = self._get_tick_outs_from_model(
                inp, past_kv_list, use_cache=True)

            selected_tick, tick_confs, selected_logits = self._select_tick(
                tick_outs_step, confidence_threshold)

            if selected_logits is not None:
                token_logits = selected_logits.unsqueeze(0)
            else:
                out = self.model.forward(inp, past_key_values=past_kv_list,
                                         use_cache=True, num_iters=self.num_iters)
                token_logits = out['logits'][:, -1:, :]
                new_past_kv = out['past_key_values']

            new_token = self._apply_sampling(
                token_logits, input_ids, temperature, top_k, top_p, repetition_penalty)

            if new_token.item() == self.tokenizer.eos_token_id:
                yield {
                    'type': 'token',
                    'text': '',
                    'tick': selected_tick,
                    'confidence': tick_confs[selected_tick] if tick_confs else 0,
                    'tick_confidences': tick_confs,
                    'early_stop': selected_tick < num_total_ticks - 1,
                    'eos': True,
                }
                break

            input_ids = torch.cat([input_ids, new_token], dim=-1)
            past_kv_list = new_past_kv

            tok_text = self.tokenizer.decode(new_token[0].tolist(), skip_special_tokens=True)
            yield {
                'type': 'token',
                'text': tok_text,
                'tick': selected_tick,
                'confidence': tick_confs[selected_tick] if tick_confs else 0,
                'tick_confidences': tick_confs,
                'early_stop': selected_tick < num_total_ticks - 1,
            }

        yield {'type': 'done'}

    def clean_response(self, text):
        for marker in ('</think', '<think', '💬', '💭'):
            if marker in text:
                idx = text.rfind(marker)
                close = text.find('\n', idx)
                if close != -1:
                    text = text[close + 1:]
                else:
                    text = text[idx + len(marker):]
        return text.strip()
