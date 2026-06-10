import copy

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from model.config import CTMLLMConfig


def parse_speed_decays(raw):
    decays = []
    for item in str(raw).split(','):
        item = item.strip()
        if not item:
            continue
        decays.append(float(item))
    return decays or [0.996]


def parse_speed_targets(raw):
    names = [item.strip() for item in str(raw).split(',') if item.strip()]
    return names or ['fast', 'mid', 'slow']


def resolve_speed_tick_index(name, num_ticks):
    if num_ticks <= 0:
        return 0
    if name == 'fast':
        return 0
    if name == 'mid':
        return max(0, num_ticks // 2 - 1)
    if name in ('slow', 'draft'):
        return num_ticks - 1
    return num_ticks - 1


class SpeedSpectrumDistiller(nn.Module):
    """Multi-EMA projection teachers supervising tick-band student states."""

    def __init__(self, config: CTMLLMConfig, backbone: nn.Module, hidden_size: int, head_cls):
        super().__init__()
        self.config = config
        self.decays = parse_speed_decays(config.speed_ema_decays)
        self.targets = parse_speed_targets(config.speed_target_ticks)
        out_dim = int(config.dino_out_dim)
        hidden_dim = int(config.dino_hidden_dim)
        bottleneck_dim = int(config.dino_bottleneck_dim)

        self.student_head = head_cls(hidden_size, hidden_dim, bottleneck_dim, out_dim)
        self.teacher_heads = nn.ModuleList([
            copy.deepcopy(self.student_head) for _ in self.decays
        ])
        self.teacher_model = copy.deepcopy(backbone)
        for module in (self.teacher_model, *self.teacher_heads):
            module.eval()
            for param in module.parameters():
                param.requires_grad_(False)

        self.register_buffer(
            'speed_center',
            torch.zeros(1, 1, out_dim),
            persistent=True,
        )

    def reset_teachers(self, backbone):
        self.teacher_model.load_state_dict(backbone.state_dict(), strict=True)
        for teacher_head in self.teacher_heads:
            teacher_head.load_state_dict(self.student_head.state_dict(), strict=True)
        self.teacher_model.eval()
        for teacher_head in self.teacher_heads:
            teacher_head.eval()

    @torch.no_grad()
    def update_teachers(self, backbone):
        for teacher_head, momentum in zip(self.teacher_heads, self.decays):
            for student, teacher in zip(self.student_head.parameters(), teacher_head.parameters()):
                teacher.data.mul_(momentum).add_(student.data, alpha=1.0 - momentum)
        slow_momentum = max(self.decays)
        for student, teacher in zip(backbone.parameters(), self.teacher_model.parameters()):
            teacher.data.mul_(slow_momentum).add_(student.data, alpha=1.0 - slow_momentum)

    def _token_mask(self, input_ids, labels):
        pad_id = int(self.config.dino_pad_token_id)
        mask = input_ids != pad_id
        if labels is not None:
            mask = mask | (labels != -100)
        return mask

    def _warmup_scale(self):
        warmup = max(0, int(self.config.speed_warmup_steps))
        if warmup <= 0:
            return 1.0
        step = max(0, int(getattr(self.config, 'global_step', 0)))
        return min(1.0, step / float(warmup))

    def forward(self, input_ids, labels, tick_outs, num_iters):
        token_mask = self._token_mask(input_ids, labels)
        if not token_mask.any():
            return tick_outs.new_zeros(())

        num_ticks = tick_outs.size(-1)
        student_temp = max(float(self.config.speed_student_temperature), 1e-4)
        teacher_temp = max(float(self.config.speed_teacher_temperature), 1e-4)
        center_momentum = float(self.config.speed_center_momentum)
        use_center = bool(int(self.config.speed_centering))

        with torch.no_grad():
            self.teacher_model.eval()
            for teacher_head in self.teacher_heads:
                teacher_head.eval()
            teacher_hidden = self.teacher_model(
                input_ids, track=False, num_iters=num_iters,
                return_all_ticks=False,
            ).hidden

        losses = []
        count = min(len(self.targets), len(self.teacher_heads))
        for i in range(count):
            target_name = self.targets[i]
            teacher_head = self.teacher_heads[i]
            tick_idx = resolve_speed_tick_index(target_name, num_ticks)
            student_hidden = tick_outs[..., tick_idx]
            student_logits = self.student_head(student_hidden.float())
            teacher_logits = teacher_head(teacher_hidden.float())
            if use_center:
                batch_center = teacher_logits[token_mask].mean(dim=0, keepdim=True).view(1, 1, -1)
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(batch_center)
                    batch_center = batch_center / dist.get_world_size()
                self.speed_center.mul_(center_momentum).add_(
                    batch_center.to(self.speed_center.device),
                    alpha=1.0 - center_momentum,
                )
                teacher_logits = teacher_logits - self.speed_center.to(teacher_logits.device)
            teacher_probs = F.softmax(teacher_logits / teacher_temp, dim=-1).detach()
            student_log_probs = F.log_softmax(student_logits / student_temp, dim=-1)
            per_token_loss = -(teacher_probs * student_log_probs).sum(dim=-1)
            losses.append(per_token_loss[token_mask].mean())

        if not losses:
            return tick_outs.new_zeros(())
        return torch.stack(losses).mean() * self._warmup_scale()
