import copy
import torch
import torch.nn.functional as F


def compute_multi_tick_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    task_loss_fn,
    mode: str = "last",
    certainties: torch.Tensor = None,
    weights: str = None,
) -> torch.Tensor:
    T = predictions.size(-1)
    total_loss = 0.0

    if mode == 'last':
        return task_loss_fn(predictions[..., -1:], targets)

    for t in range(T):
        p = predictions[..., t:t+1]
        loss_t = task_loss_fn(p, targets)
        if mode == 'mean':
            total_loss = total_loss + loss_t / T
        elif mode == 'min_conf' and certainties is not None:
            if certainties.dim() == 3:
                cert = certainties[:, 0, t]
            else:
                cert = certainties[:, t]
            weight = F.softmin(cert, dim=0).mean() if cert.numel() > 1 else 1.0
            total_loss = total_loss + loss_t * weight.detach()
        elif mode == 'weighted' and weights is not None:
            w = weights[t]
            total_loss = total_loss + loss_t * w

    if mode == 'min_conf' and certainties is not None:
        total_loss = total_loss / T

    return total_loss


def compute_tick_penalty(
    steps_used: int,
    max_steps: int,
    weight: float = 0.0,
) -> torch.Tensor:
    if weight <= 0 or max_steps <= 1:
        return torch.tensor(0.0)
    return weight * (steps_used / max_steps)


def compute_reflex_distillation_loss(
    reflex_predictions: torch.Tensor,
    main_predictions: torch.Tensor,
    temperature: float = 1.0,
    weight: float = 0.1,
) -> torch.Tensor:
    if weight <= 0:
        return torch.tensor(0.0)
    main_soft = F.log_softmax(main_predictions / temperature, dim=-1)
    reflex_soft = F.softmax(reflex_predictions / temperature, dim=-1)
    kl = F.kl_div(main_soft, reflex_soft, reduction='batchmean', log_target=False)
    return weight * (temperature ** 2) * kl


def add_train_idea_args(parser):
    parser.add_argument('--tick_loss_mode', type=str, default='last',
                        choices=['last', 'mean', 'min_conf'],
                        help='How to aggregate per-tick losses')
    parser.add_argument('--tick_loss_weights', type=str, default=None,
                        help='Comma-separated weights for each tick')
    parser.add_argument('--tick_compute_weight', type=float, default=0.0,
                        help='Penalty weight for using more ticks')
    parser.add_argument('--reflex_distill_weight', type=float, default=0.0,
                        help='Weight for reflex head distillation loss')
    parser.add_argument('--reflex_distill_temp', type=float, default=1.0,
                        help='Temperature for reflex distillation')
    parser.add_argument('--ema_speed_mode', type=str, default='none',
                        choices=['none', 'ema_spectrum'],
                        help='EMA speed spectrum mode')
    parser.add_argument('--ema_speed_decays', type=str, default='0.90,0.97,0.995',
                        help='Comma-separated EMA decay rates')
    parser.add_argument('--ema_distill_weight', type=float, default=0.0,
                        help='Weight for EMA distillation loss')
    parser.add_argument('--ema_warmup_steps', type=int, default=500,
                        help='Steps before EMA distillation starts')
    return parser


def build_ema_teachers(model, args):
    if args.ema_speed_mode != 'ema_spectrum' or args.ema_distill_weight <= 0:
        return None, None
    decays = [float(d) for d in args.ema_speed_decays.split(',')]
    teachers = []
    for decay in decays:
        teacher = copy.deepcopy(model)
        for p in teacher.parameters():
            p.requires_grad = False
        teacher.train()
        teachers.append((teacher, decay))
    return teachers, decays


@torch.no_grad()
def update_ema_teachers(teachers, student_model, step):
    if teachers is None:
        return
    for teacher, decay in teachers:
        for t_param, s_param in zip(teacher.parameters(), student_model.parameters()):
            t_param.data.mul_(decay).add_(s_param.data, alpha=1 - decay)


def compute_ema_distillation_loss(
    student_synch: torch.Tensor,
    teachers,
    teacher_synch_fn,
    inputs,
    step: int,
    warmup_steps: int = 500,
    weight: float = 0.0,
    temperature: float = 0.1,
    center_momentum: float = 0.9,
) -> torch.Tensor:
    if teachers is None or weight <= 0 or step < warmup_steps:
        return torch.tensor(0.0)
    total_loss = 0.0
    for teacher, decay in teachers:
        with torch.no_grad():
            teacher_synch = teacher_synch_fn(teacher, inputs)
        student_norm = F.normalize(student_synch, dim=-1)
        teacher_norm = F.normalize(teacher_synch, dim=-1)
        loss = 1 - (student_norm * teacher_norm).sum(dim=-1).mean()
        total_loss = total_loss + loss
    return weight * total_loss / len(teachers)
