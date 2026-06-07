import os
import sys
import argparse
import csv
import json
import subprocess
import time
import traceback
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from model.config import CTMLLMConfig
from dataset.text_dataset import TextDataset
from trainer.trainer_utils import (
    Logger, get_lr, setup_seed, create_model, load_tokenizer,
    save_checkpoint, load_checkpoint
)

warnings.filterwarnings('ignore')


def format_time(seconds):
    if seconds < 60:
        return f'{seconds:.0f}s'
    if seconds < 3600:
        return f'{seconds / 60:.1f}min'
    return f'{seconds / 3600:.2f}h'


def setup_ddp():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def count_valid_tokens(labels):
    return int((labels[..., 1:] != -100).sum().item())


def cuda_memory_mb(device):
    if not torch.cuda.is_available() or 'cuda' not in str(device):
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def append_metrics_csv(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def failure_path_from_metrics(path, rank=None):
    stem = os.path.splitext(path)[0]
    if rank is None:
        return stem + '.fail.json'
    return f'{stem}.rank{rank}.fail.json'


def write_failure_report(args, exc, rank=0):
    if not getattr(args, 'metrics_path', None):
        return
    err_text = str(exc)
    status = 'oom' if isinstance(exc, torch.OutOfMemoryError) or 'out of memory' in err_text.lower() else 'failed'
    payload = {
        'experiment_name': args.experiment_name,
        'status': status,
        'rank': rank,
        'error_type': type(exc).__name__,
        'error': err_text[-4000:],
        'traceback': traceback.format_exc()[-12000:],
        'model_type': args.model_type,
        'batch_size': args.batch_size,
        'world_size': getattr(args, 'world_size', 1),
        'hidden_size': args.hidden_size,
        'num_hidden_layers': args.num_hidden_layers,
        'd_model': args.d_model,
        'd_input': args.d_input,
        'iterations': args.iterations,
        'memory_length': args.memory_length,
        'memory_hidden_dims': args.memory_hidden_dims,
        'synapse_depth': args.synapse_depth,
        'peak_memory_mb': cuda_memory_mb(args.device),
        'time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'git_commit': getattr(args, 'git_commit', 'unknown'),
    }
    path = failure_path_from_metrics(args.metrics_path, rank=rank)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    Logger(f'Failure report: {path}')
    if rank == 0:
        legacy_path = failure_path_from_metrics(args.metrics_path)
        with open(legacy_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


def install_failure_hook(args, rank=0):
    default_hook = sys.excepthook

    def hook(exc_type, exc, tb):
        if not issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
            write_failure_report(args, exc, rank=rank)
        default_hook(exc_type, exc, tb)

    sys.excepthook = hook


def effective_tick_from_certainties(certainties, args):
    if certainties is None:
        return -1.0
    if args.tick_halt_mode == 'confidence':
        confidence = 1 - certainties
        temp = max(float(args.tick_halt_temperature), 1e-4)
        weights = torch.softmax(confidence / temp, dim=1)
        tick_ids = torch.arange(
            1, certainties.size(1) + 1, device=certainties.device,
            dtype=certainties.dtype)
        return (weights * tick_ids.view(1, -1)).sum(dim=1).mean().item()
    if args.tick_halt_mode == 'threshold':
        confidence = 1 - certainties
        hit = confidence >= args.tick_halt_threshold
        any_hit = hit.any(dim=1)
        first_hit = hit.float().argmax(dim=1) + 1
        last_tick = torch.full_like(first_hit, certainties.size(1))
        return torch.where(any_hit, first_hit, last_tick).float().mean().item()
    return -1.0


def active_cell_fraction(args):
    if args.model_type != 'ctm' or args.cell_sparsity_mode == 'none':
        return 1.0
    return min(max(args.cell_topk, 1), args.d_model) / max(args.d_model, 1)


def train_epoch(epoch, loader, iters, model, optimizer, scaler, autocast_ctx, args,
                tb_writer, swanlab, start_step=0, rank=0):
    model.train()
    epoch_start = time.time()
    total_loss = 0.0
    total_steps = 0
    total_tokens = 0
    interval_start = time.time()

    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        valid_tokens = count_valid_tokens(labels)

        global_step = epoch * iters + step
        total_global = args.epochs * iters
        lr = get_lr(global_step, total_global, args.learning_rate)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        with autocast_ctx:
            loss, losses_per_tick, certainties = raw_model.forward_train(
                input_ids, labels, num_iters=args.iterations)
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * args.accumulation_steps
        total_steps += 1
        total_tokens += valid_tokens

        if rank == 0 and (step % args.log_interval == 0 or step == iters):
            avg_loss = total_loss / total_steps
            if losses_per_tick is not None and certainties is not None:
                best_tick_mean = losses_per_tick.argmin(dim=1).float().mean().item()
                conf_tick_mean = (1 - certainties).argmax(dim=1).float().mean().item()
                tick_count = losses_per_tick.size(1)
                losses_tick_mean = [
                    round(x, 6) for x in losses_per_tick.mean(dim=0).detach().float().cpu().tolist()]
                certainties_tick_mean = [
                    round(x, 6) for x in certainties.mean(dim=0).detach().float().cpu().tolist()]
                effective_tick_mean = effective_tick_from_certainties(certainties, args)
            else:
                best_tick_mean = -1.0
                conf_tick_mean = -1.0
                tick_count = 1
                losses_tick_mean = []
                certainties_tick_mean = []
                effective_tick_mean = -1.0
            elapsed = time.time() - epoch_start
            interval_elapsed = time.time() - interval_start
            steps_done = step - start_step
            steps_left = iters - step
            speed = elapsed / max(steps_done, 1)
            steps_per_sec = total_steps / max(interval_elapsed, 1e-9)
            tokens_per_sec = total_tokens / max(interval_elapsed, 1e-9)
            epoch_eta = speed * steps_left
            global_done = global_step
            global_eta = speed * (total_global - global_done)
            pct = global_done / total_global * 100
            peak_mem_mb = cuda_memory_mb(args.device)

            Logger(
                f'Epoch[{epoch + 1}/{args.epochs}]({step}/{iters}) | '
                f'loss:{avg_loss:.4f} lr:{lr:.2e} | '
                f'best_tick:{best_tick_mean:.1f} conf_tick:{conf_tick_mean:.1f} | '
                f'eff_tick:{effective_tick_mean:.1f} | '
                f'tok/s:{tokens_per_sec:.0f} mem:{peak_mem_mb:.0f}MB | '
                f'elapsed:{format_time(elapsed)} epoch_eta:{format_time(epoch_eta)} | '
                f'total:{format_time(elapsed + global_eta)}({pct:.1f}%)'
            )

            metrics = {
                'loss': avg_loss, 'lr': lr,
                'best_tick': best_tick_mean,
                'conf_tick': conf_tick_mean,
                'tick_count': tick_count,
                'effective_tick': effective_tick_mean,
                'active_cell_fraction': active_cell_fraction(args),
                'tokens_per_sec': tokens_per_sec,
                'steps_per_sec': steps_per_sec,
                'peak_memory_mb': peak_mem_mb,
                'epoch': epoch + step / iters,
                'global_step': global_step,
                'progress_pct': pct,
            }
            if tb_writer:
                for key, value in metrics.items():
                    tb_writer.add_scalar(key, value, global_step)
            if swanlab:
                swanlab.log(metrics, step=global_step)
            if args.metrics_path:
                append_metrics_csv(args.metrics_path, {
                    'experiment_name': args.experiment_name,
                    'model_type': args.model_type,
                    'global_step': global_step,
                    'epoch': epoch + 1,
                    'step': step,
                    'loss': avg_loss,
                    'lr': lr,
                    'best_tick': best_tick_mean,
                    'conf_tick': conf_tick_mean,
                    'tick_count': tick_count,
                    'effective_tick': effective_tick_mean,
                    'active_cell_fraction': active_cell_fraction(args),
                    'losses_per_tick': json.dumps(losses_tick_mean),
                    'certainties_per_tick': json.dumps(certainties_tick_mean),
                    'tokens': total_tokens,
                    'tokens_per_sec': tokens_per_sec,
                    'steps_per_sec': steps_per_sec,
                    'peak_memory_mb': peak_mem_mb,
                    'world_size': args.world_size,
                    'hidden_size': args.hidden_size,
                    'num_hidden_layers': args.num_hidden_layers,
                    'd_model': args.d_model,
                    'd_input': args.d_input,
                    'iterations': args.iterations,
                    'memory_length': args.memory_length,
                    'memory_hidden_dims': args.memory_hidden_dims,
                    'deep_nlms': args.deep_nlms,
                    'synapse_depth': args.synapse_depth,
                    'tick_loss_mode': args.tick_loss_mode,
                    'elf_horizon_mode': args.elf_horizon_mode,
                    'elf_max_horizon': args.elf_max_horizon,
                    'tick_improve_weight': args.tick_improve_weight,
                    'tick_improve_margin': args.tick_improve_margin,
                    'tick_halt_mode': args.tick_halt_mode,
                    'tick_halt_threshold': args.tick_halt_threshold,
                    'tick_halt_temperature': args.tick_halt_temperature,
                    'tick_compute_weight': args.tick_compute_weight,
                    'cell_sparsity_mode': args.cell_sparsity_mode,
                    'cell_topk': args.cell_topk,
                    'cell_sparsity_rescale': int(args.cell_sparsity_rescale),
                    'self_cond': args.self_cond,
                    'cross_layer_state': args.cross_layer_state,
                    'max_seq_len': args.max_seq_len,
                    'batch_size': args.batch_size,
                    'accumulation_steps': args.accumulation_steps,
                    'elapsed_sec': elapsed,
                    'git_commit': args.git_commit,
                })

            total_loss = 0.0
            total_steps = 0
            total_tokens = 0
            interval_start = time.time()

        if rank == 0 and (step % args.save_interval == 0 or step == iters):
            save_path = os.path.join(args.save_dir, f'{args.save_weight}_{args.hidden_size}.pth')
            resume_path = os.path.join(args.save_dir, f'{args.save_weight}_{args.hidden_size}_resume.pth')
            raw = raw_model._orig_mod if hasattr(raw_model, '_orig_mod') else raw_model
            save_checkpoint(raw, optimizer, epoch, step, resume_path, scaler)
            torch.save(
                {k: v.half().cpu() for k, v in raw.state_dict().items()}, save_path)

        if args.max_steps > 0 and global_step >= args.max_steps:
            if rank == 0:
                Logger(f'Max steps reached: {args.max_steps}')
            break

    if rank == 0:
        epoch_time = time.time() - epoch_start
        Logger(f'Epoch {epoch + 1} done in {format_time(epoch_time)}')
    return step


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CTM-LLM Training')
    parser.add_argument('--save_dir', type=str, default='out')
    parser.add_argument('--save_weight', type=str, default='ctm_llm')
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--metrics_dir', type=str, default='runs/metrics')
    parser.add_argument('--metrics_path', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--dtype', type=str, default='bfloat16')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--accumulation_steps', type=int, default=4)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--save_interval', type=int, default=500)
    parser.add_argument('--max_steps', type=int, default=0,
                        help='Stop training after this many global steps; 0 disables.')
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--num_hidden_layers', type=int, default=12)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_input', type=int, default=256)
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--memory_length', type=int, default=10)
    parser.add_argument('--memory_hidden_dims', type=int, default=4)
    parser.add_argument('--deep_nlms', type=int, default=1, choices=[0, 1])
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--n_synch_out', type=int, default=512)
    parser.add_argument('--n_synch_action', type=int, default=512)
    parser.add_argument('--synapse_depth', type=int, default=3)
    parser.add_argument('--self_cond', type=int, default=1, choices=[0, 1])
    parser.add_argument('--cross_layer_state', type=int, default=1, choices=[0, 1])
    parser.add_argument('--block_size', type=int, default=4)
    parser.add_argument('--tick_loss_mode', type=str, default='min_conf',
                        choices=['min_conf', 'mean', 'last'])
    parser.add_argument('--elf_horizon_mode', type=str, default='none',
                        choices=['none', 'linear', 'pow2'])
    parser.add_argument('--elf_max_horizon', type=int, default=4)
    parser.add_argument('--tick_improve_weight', type=float, default=0.0)
    parser.add_argument('--tick_improve_margin', type=float, default=0.0)
    parser.add_argument('--tick_halt_mode', type=str, default='none',
                        choices=['none', 'confidence', 'threshold'])
    parser.add_argument('--tick_halt_threshold', type=float, default=0.65)
    parser.add_argument('--tick_halt_temperature', type=float, default=0.25)
    parser.add_argument('--tick_compute_weight', type=float, default=0.0)
    parser.add_argument('--cell_sparsity_mode', type=str, default='none',
                        choices=['none', 'topk'])
    parser.add_argument('--cell_topk', type=int, default=512)
    parser.add_argument('--cell_sparsity_rescale', type=int, default=1, choices=[0, 1])
    parser.add_argument('--ttt_layer', type=int, default=0, choices=[0, 1])
    parser.add_argument('--ttt_hidden_mult', type=int, default=2)
    parser.add_argument('--ttt_gate_init', type=float, default=-2.0)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--data_path', type=str,
                        default='dataset_data/sft_t2a_mini.parquet')
    parser.add_argument('--tokenizer_path', type=str,
                        default='model_tokenizer')
    parser.add_argument('--from_weight', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_tensorboard', action='store_true')
    parser.add_argument('--tensorboard_log_dir', type=str, default='runs')
    parser.add_argument('--use_swanlab', action='store_true')
    parser.add_argument('--swanlab_project', type=str, default='CTM-LLM')
    parser.add_argument('--swanlab_name', type=str, default=None)
    parser.add_argument('--use_compile', type=int, default=0, choices=[0, 1])
    parser.add_argument('--model_type', type=str, default='ctm',
                        choices=['ctm', 'transformer'])
    args = parser.parse_args()

    ddp = int(os.environ.get('WORLD_SIZE', 1)) > 1
    if ddp:
        local_rank, rank, world_size = setup_ddp()
        args.device = f'cuda:{local_rank}'
    else:
        local_rank, rank, world_size = 0, 0, 1
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args.world_size = world_size

    os.makedirs(args.save_dir, exist_ok=True)
    setup_seed(args.seed)
    try:
        args.git_commit = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        args.git_commit = 'unknown'
    if args.experiment_name is None:
        args.experiment_name = args.swanlab_name or args.save_weight
    if args.metrics_path is None:
        safe_name = ''.join(
            c if c.isalnum() or c in '-_.' else '_' for c in args.experiment_name)
        args.metrics_path = os.path.join(args.metrics_dir, f'{safe_name}.csv')
    install_failure_hook(args, rank=rank)

    config = CTMLLMConfig(
        model_type=args.model_type,
        vocab_size=6400,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        d_model=args.d_model,
        d_input=args.d_input,
        iterations=args.iterations,
        memory_length=args.memory_length,
        memory_hidden_dims=args.memory_hidden_dims,
        deep_nlms=bool(args.deep_nlms),
        heads=args.heads,
        n_synch_out=args.n_synch_out,
        n_synch_action=args.n_synch_action,
        synapse_depth=args.synapse_depth,
        self_cond=bool(args.self_cond),
        cross_layer_state=bool(args.cross_layer_state),
        block_size=args.block_size,
        tick_loss_mode=args.tick_loss_mode,
        elf_horizon_mode=args.elf_horizon_mode,
        elf_max_horizon=args.elf_max_horizon,
        tick_improve_weight=args.tick_improve_weight,
        tick_improve_margin=args.tick_improve_margin,
        tick_halt_mode=args.tick_halt_mode,
        tick_halt_threshold=args.tick_halt_threshold,
        tick_halt_temperature=args.tick_halt_temperature,
        tick_compute_weight=args.tick_compute_weight,
        cell_sparsity_mode=args.cell_sparsity_mode,
        cell_topk=args.cell_topk,
        cell_sparsity_rescale=bool(args.cell_sparsity_rescale),
        ttt_layer=bool(args.ttt_layer),
        ttt_hidden_mult=args.ttt_hidden_mult,
        ttt_gate_init=args.ttt_gate_init,
    )
    if rank == 0:
        Logger(f'Config: {config}')
        Logger(f'Experiment: {args.experiment_name}')
        Logger(f'Metrics CSV: {args.metrics_path}')
        manifest_path = os.path.splitext(args.metrics_path)[0] + '.json'
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=2)

    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
    autocast_ctx = torch.amp.autocast(device_type, dtype=dtype) if device_type == 'cuda' else torch.amp.autocast('cpu')

    model = create_model(config, args.device)
    if args.use_compile:
        model = torch.compile(model)
        if rank == 0:
            Logger('torch.compile enabled')
    if ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    raw_model = model.module if hasattr(model, 'module') else model

    tokenizer = load_tokenizer(args.tokenizer_path)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    use_fp16 = args.dtype == 'float16'
    scaler = torch.amp.GradScaler(device_type, enabled=use_fp16)

    run_name = args.swanlab_name or args.experiment_name or \
        f'CTM-LLM-{args.hidden_size}d-{args.d_model}m-{args.iterations}iter'

    tb_writer = None
    if not args.no_tensorboard and rank == 0:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_log_dir = os.path.join(args.tensorboard_log_dir, run_name)
            tb_writer = SummaryWriter(log_dir=tb_log_dir)
            Logger(f'TensorBoard logging: {tb_log_dir}')
        except ImportError:
            Logger('TensorBoard not installed, continue without TensorBoard logging.')

    swanlab = None
    if args.use_swanlab and rank == 0:
        try:
            import swanlab
            swanlab = swanlab.init(project=args.swanlab_project, name=run_name, config=vars(args))
        except ImportError:
            Logger('SwanLab not installed, continue without SwanLab logging.')

    start_epoch, start_step = 0, 0
    if args.from_weight:
        weight_path = os.path.join(args.save_dir,
                                   f'{args.from_weight}_{args.hidden_size}.pth')
        if os.path.exists(weight_path):
            resume_path = os.path.join(args.save_dir,
                                       f'{args.from_weight}_{args.hidden_size}_resume.pth')
            if os.path.exists(resume_path):
                start_epoch, start_step = load_checkpoint(
                    resume_path, model, optimizer, scaler, args.device)
            else:
                state = torch.load(weight_path, map_location=args.device, weights_only=False)
                model.load_state_dict(state, strict=False)
                if rank == 0:
                    Logger(f'Loaded weights: {weight_path}')

    dataset = TextDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    if rank == 0:
        Logger(f'Dataset: {len(dataset)} samples')

    train_start = time.time()
    try:
        for epoch in range(start_epoch, args.epochs):
            setup_seed(args.seed + epoch)
            if ddp:
                sampler = DistributedSampler(dataset, shuffle=True, num_replicas=world_size, rank=rank)
                sampler.set_epoch(epoch)
                loader = DataLoader(
                    dataset, batch_size=args.batch_size, sampler=sampler,
                    num_workers=args.num_workers, pin_memory=True, drop_last=True)
            else:
                loader = DataLoader(
                    dataset, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True, drop_last=True)

            iters = len(loader)
            skip = start_step if epoch == start_epoch and start_step > 0 else 0
            if skip > 0 and rank == 0:
                Logger(f'Epoch[{epoch + 1}]: skip first {skip} steps')
            last_step = train_epoch(
                epoch, loader, iters, model, optimizer, scaler, autocast_ctx, args,
                tb_writer, swanlab,
                start_step=skip, rank=rank)
            if args.max_steps > 0 and epoch * iters + last_step >= args.max_steps:
                break
            start_step = 0

        if rank == 0:
            total_time = time.time() - train_start
            Logger(f'Training complete! Total: {format_time(total_time)}')
    except Exception as exc:
        write_failure_report(args, exc, rank=rank)
        raise
    finally:
        if rank == 0:
            if swanlab:
                swanlab.finish()
            if tb_writer:
                tb_writer.close()
        if ddp:
            cleanup_ddp()
