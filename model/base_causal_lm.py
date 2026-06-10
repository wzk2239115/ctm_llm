import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.building_blocks import RMSNorm, FeedForward


class BaseCTMForCausalLM(nn.Module):
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
        mode = self.config.moe_mtp_mode
        raw = str(self.config.moe_mtp_horizons or '')
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

    def _fast_output_ticks(self):
        from model.building_blocks import _parse_int_list
        ticks = _parse_int_list(self.config.fast_output_ticks)
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
        kl_sum = student.new_zeros(())
        valid = mask.sum().clamp(min=1)
        chunk = 64
        for start in range(0, student.size(1), chunk):
            end = min(start + chunk, student.size(1))
            chunk_mask = mask[:, start:end]
            if not chunk_mask.any():
                continue
            log_p = F.log_softmax(student[:, start:end, :], dim=-1)
            q = F.softmax(teacher[:, start:end, :], dim=-1)
            kl = F.kl_div(log_p, q, reduction='none').sum(dim=-1)
            kl_sum = kl_sum + (kl * chunk_mask).sum()
        return kl_sum / valid

    def _fast_slow_output_loss(self, input_ids, labels, tick_outs, final_logits):
        mode = self.config.fast_output_mode
        if mode == 'none':
            return final_logits.new_zeros(())
        aux = final_logits.new_zeros(())
        distill_weight = float(self.config.fast_output_distill_weight)
        fast_weight = float(self.config.fast_output_weight)
        habit_weight = float(self.config.habit_output_weight)
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

    _ENTROPY_CHUNK = 128

    @staticmethod
    @torch.no_grad()
    def _halt_confidence_mean(logits):
        """Mean 1-normalized_entropy over all tokens; used for infer-time tick halt."""
        vocab_log = math.log(logits.size(-1))
        conf_parts = []
        chunk = BaseCTMForCausalLM._ENTROPY_CHUNK
        for start in range(0, logits.size(1), chunk):
            end = min(start + chunk, logits.size(1))
            chunk_logits = logits[:, start:end, :]
            probs = F.softmax(chunk_logits, dim=-1)
            entropy = -(probs * torch.log(probs.clamp(min=1e-12))).sum(-1)
            conf_parts.append(1 - entropy / vocab_log)
        return torch.cat(conf_parts, dim=1).mean()

    @staticmethod
    @torch.no_grad()
    def _per_sample_entropy(logits, label_mask, horizon):
        """Normalized token entropy for tick selection; no grad (argmax-only use)."""
        if label_mask.numel() == 0:
            return logits.new_zeros(logits.size(0))
        valid_logits = logits[..., :-horizon, :]
        vocab_log = math.log(logits.size(-1))
        ent_sums = logits.new_zeros(logits.size(0))
        chunk = BaseCTMForCausalLM._ENTROPY_CHUNK
        for start in range(0, valid_logits.size(1), chunk):
            end = min(start + chunk, valid_logits.size(1))
            chunk_logits = valid_logits[:, start:end, :]
            probs = F.softmax(chunk_logits, dim=-1)
            entropy = -(probs * torch.log(probs.clamp(min=1e-12))).sum(-1)
            ent_sums += (entropy * label_mask[:, start:end]).sum(dim=1)
        norm_ent = ent_sums / label_mask.sum(dim=1).clamp(min=1) / vocab_log
        return norm_ent

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

    @torch.inference_mode()
    def forward_track(self, input_ids, num_iters=None):
        result = self.model(input_ids, track=True, num_iters=num_iters,
                            return_all_ticks=False)
        h = result.hidden
        tracking = result.tracking
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
