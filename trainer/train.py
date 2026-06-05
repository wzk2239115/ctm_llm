import os
import sys
import argparse
import time
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


def train_epoch(epoch, loader, iters, model, optimizer, scaler, autocast_ctx, args,
                tb_writer, swanlab, start_step=0, rank=0):
    model.train()
    epoch_start = time.time()
    total_loss = 0.0
    total_steps = 0

    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)

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

        if rank == 0 and (step % args.log_interval == 0 or step == iters):
            avg_loss = total_loss / total_steps
            best_tick_mean = losses_per_tick.argmin(dim=1).float().mean().item()
            conf_tick_mean = (1 - certainties).argmax(dim=1).float().mean().item()
            elapsed = time.time() - epoch_start
            steps_done = step - start_step
            steps_left = iters - step
            speed = elapsed / max(steps_done, 1)
            epoch_eta = speed * steps_left
            global_done = global_step
            global_eta = speed * (total_global - global_done)
            pct = global_done / total_global * 100

            Logger(
                f'Epoch[{epoch + 1}/{args.epochs}]({step}/{iters}) | '
                f'loss:{avg_loss:.4f} lr:{lr:.2e} | '
                f'best_tick:{best_tick_mean:.1f} conf_tick:{conf_tick_mean:.1f} | '
                f'elapsed:{format_time(elapsed)} epoch_eta:{format_time(epoch_eta)} | '
                f'total:{format_time(elapsed + global_eta)}({pct:.1f}%)'
            )

            metrics = {
                'loss': avg_loss, 'lr': lr,
                'best_tick': best_tick_mean,
                'conf_tick': conf_tick_mean,
                'epoch': epoch + step / iters,
                'global_step': global_step,
                'progress_pct': pct,
            }
            if tb_writer:
                for key, value in metrics.items():
                    tb_writer.add_scalar(key, value, global_step)
            if swanlab:
                swanlab.log(metrics, step=global_step)

            total_loss = 0.0
            total_steps = 0

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
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--n_synch_out', type=int, default=512)
    parser.add_argument('--n_synch_action', type=int, default=512)
    parser.add_argument('--synapse_depth', type=int, default=3)
    parser.add_argument('--self_cond', type=int, default=1, choices=[0, 1])
    parser.add_argument('--cross_layer_state', type=int, default=1, choices=[0, 1])
    parser.add_argument('--block_size', type=int, default=4)
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
    args = parser.parse_args()

    ddp = int(os.environ.get('WORLD_SIZE', 1)) > 1
    if ddp:
        local_rank, rank, world_size = setup_ddp()
        args.device = f'cuda:{local_rank}'
    else:
        local_rank, rank, world_size = 0, 0, 1
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    os.makedirs(args.save_dir, exist_ok=True)
    setup_seed(args.seed)

    config = CTMLLMConfig(
        vocab_size=6400,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        d_model=args.d_model,
        d_input=args.d_input,
        iterations=args.iterations,
        memory_length=args.memory_length,
        heads=args.heads,
        n_synch_out=args.n_synch_out,
        n_synch_action=args.n_synch_action,
        synapse_depth=args.synapse_depth,
        self_cond=bool(args.self_cond),
        cross_layer_state=bool(args.cross_layer_state),
        block_size=args.block_size,
    )
    if rank == 0:
        Logger(f'Config: {config}')

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

    run_name = args.swanlab_name or f'CTM-LLM-{args.hidden_size}d-{args.d_model}m-{args.iterations}iter'

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
        if swanlab:
            swanlab.finish()
        if tb_writer:
            tb_writer.close()

    if ddp:
        cleanup_ddp()
