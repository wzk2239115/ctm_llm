import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import CTMLLMConfig
from model.model_ctm_llm import FeedForward, RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, config: CTMLLMConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.heads = config.heads
        assert config.hidden_size % config.heads == 0, \
            f"hidden_size({config.hidden_size}) must be divisible by heads({config.heads})"
        self.head_dim = config.hidden_size // config.heads

        self.input_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.attn_drop = nn.Dropout(config.dropout)

        self.post_attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config.hidden_size)
        self.resid_drop = nn.Dropout(config.dropout)

    def forward(self, x, past_kv=None, use_cache=False):
        B, T, C = x.shape
        normed = self.input_norm(x)
        q = self.q_proj(normed)
        k = self.k_proj(normed)
        v = self.v_proj(normed)
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=1)
            v = torch.cat([past_kv[1], v], dim=1)
        present_kv = (k, v) if use_cache else None
        S = k.size(1)

        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.heads, self.head_dim).transpose(1, 2)

        if T > 1 and past_kv is None:
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, T, C)
        x = x + self.resid_drop(self.attn_drop(self.o_proj(attn)))
        x = x + self.mlp(self.post_attn_norm(x))
        return x, present_kv


class TransformerModel(nn.Module):
    def __init__(self, config: CTMLLMConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embed = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.drop = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, past_key_values=None, use_cache=False, **_):
        B, T = input_ids.shape
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        offset = past_key_values[0][0].size(1) if past_key_values[0] is not None else 0
        pos = torch.arange(offset, offset + T, device=input_ids.device)
        h = self.embed_tokens(input_ids) + self.pos_embed(pos).unsqueeze(0)
        h = self.drop(h)
        presents = []
        for layer, past_kv in zip(self.layers, past_key_values):
            h, present = layer(h, past_kv=past_kv, use_cache=use_cache)
            presents.append(present)
        return self.norm(h), presents


class TransformerForCausalLM(nn.Module):
    def __init__(self, config: CTMLLMConfig):
        super().__init__()
        self.config = config
        self.model = TransformerModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, past_key_values=None, use_cache=False, labels=None,
                num_iters=None):
        h, past_key_values = self.model(input_ids, past_key_values, use_cache)
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1), ignore_index=-100)
        return {'loss': loss, 'logits': logits, 'past_key_values': past_key_values}

    def forward_train(self, input_ids, labels, num_iters=None):
        out = self.forward(input_ids, labels=labels)
        return out["loss"], None, None

    @torch.inference_mode()
    def generate(self, input_ids, max_new_tokens=512, temperature=0.85,
                 top_p=0.85, top_k=50, eos_token_id=2, use_cache=True,
                 repetition_penalty=1.0, num_iters=None):
        past_kv = None
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

        for _ in range(max_new_tokens):
            inp = input_ids if past_kv is None else input_ids[:, -1:]
            out = self.forward(inp, past_key_values=past_kv, use_cache=use_cache)
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
        return ent / math.log(logits_seq.size(-1))
