import argparse
import datetime
import glob
import math
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer

from model.config import CTMLLMConfig
from model.model_ctm_llm import CTMForCausalLM
from trainer.trainer_utils import Logger, get_lr, setup_seed


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.2f}h"


def setup_ddp(timeout_minutes=60):
    timeout = datetime.timedelta(minutes=timeout_minutes)
    dist.init_process_group(backend="nccl", timeout=timeout)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def build_config(args):
    return CTMLLMConfig(
        vocab_size=args.vocab_size,
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
        ttt_layer=True,
        ttt_hidden_mult=args.ttt_hidden_mult,
        ttt_gate_init=args.ttt_gate_init,
    )


def load_model(args):
    model = CTMForCausalLM(build_config(args)).to(args.device)
    ckpt = torch.load(args.from_weight, map_location=args.device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        Logger(f"load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            Logger(f"  missing: {missing[:10]}")
        if unexpected:
            Logger(f"  unexpected: {unexpected[:10]}")
    return model


def target_filter(name, args):
    last_layer = f"model.layers.{args.num_hidden_layers - 1}."
    if args.ttt_target == "ttt_layers":
        return ".ttt_layer." in name
    if args.ttt_target == "last_ttt_layer":
        return name.startswith(last_layer + "ttt_layer.")
    if args.ttt_target == "last_block":
        return name.startswith(last_layer)
    if args.ttt_target == "last_output":
        return name.startswith(last_layer + "output_proj") or \
            name.startswith(last_layer + "post_ctm_norm")
    return False


def select_trainable_params(model, args):
    selected_names = []
    for name, param in model.named_parameters():
        trainable = target_filter(name, args)
        param.requires_grad_(trainable)
        if trainable:
            selected_names.append(name)
    params = [p for p in model.parameters() if p.requires_grad]
    return params, selected_names


def save_model_only(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw = model.module if hasattr(model, "module") else model
    raw = raw._orig_mod if hasattr(raw, "_orig_mod") else raw
    tmp = path + ".tmp"
    torch.save({k: v.half().cpu() for k, v in raw.state_dict().items()}, tmp)
    os.replace(tmp, path)
    Logger(f"Checkpoint saved: {path}")


def local_data_files(path):
    patterns = [
        "**/*.jsonl",
        "**/*.json",
        "**/*.jsonl.gz",
        "**/*.json.gz",
        "**/*.jsonl.zst",
        "**/*.json.zst",
        "**/*.parquet",
    ]
    files = []
    if os.path.isfile(path):
        files.append(path)
    else:
        for pattern in patterns:
            files.extend(glob.glob(os.path.join(path, pattern), recursive=True))
    return sorted(set(files))


def local_loader_name(files):
    if not files:
        return None
    suffixes = {name.lower() for path in files for name in [os.path.basename(path)]}
    if all(item.endswith(".parquet") for item in suffixes):
        return "parquet"
    return "json"


def load_streaming_dataset(args, rank, world_size):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install datasets first: pip install datasets") from exc

    config = args.dataset_config if args.dataset_config else None
    if args.dataset_path:
        if not os.path.exists(args.dataset_path):
            raise FileNotFoundError(args.dataset_path)
        try:
            dataset = load_dataset(
                args.dataset_name,
                config,
                data_dir=args.dataset_path,
                split=args.dataset_split,
                streaming=True,
            )
        except Exception as exc:
            files = local_data_files(args.dataset_path)
            loader = local_loader_name(files)
            if not loader:
                raise RuntimeError(
                    f"No supported data files found under {args.dataset_path}"
                ) from exc
            Logger(
                f"Falling back to local {loader} loader with {len(files)} file(s) "
                f"from {args.dataset_path}"
            )
            dataset = load_dataset(
                loader,
                data_files={args.dataset_split: files},
                split=args.dataset_split,
                streaming=True,
            )
    else:
        dataset = load_dataset(
            args.dataset_name,
            config,
            split=args.dataset_split,
            streaming=True,
        )
    if world_size > 1:
        dataset = dataset.shard(num_shards=world_size, index=rank)
    if args.shuffle_buffer > 0:
        dataset = dataset.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed + rank)
    return dataset


def iter_token_batches(dataset, tokenizer, args, device):
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    langs = None
    if args.lang_filter:
        langs = {item.strip() for item in args.lang_filter.split(",") if item.strip()}

    buffer = []
    batch = []
    for row in dataset:
        if langs is not None and str(row.get("lang", "")) not in langs:
            continue
        text = row.get(args.text_field)
        if not isinstance(text, str) or len(text) < args.min_chars:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            continue
        if args.add_eos:
            ids.append(eos_id)
        buffer.extend(ids)

        while len(buffer) >= args.max_seq_len:
            chunk = buffer[:args.max_seq_len]
            buffer = buffer[args.max_seq_len:]
            batch.append(torch.tensor(chunk, dtype=torch.long))
            if len(batch) == args.batch_size:
                input_ids = torch.stack(batch, dim=0).to(device, non_blocking=True)
                labels = input_ids.clone()
                yield input_ids, labels
                batch = []


def train(args):
    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if ddp:
        local_rank, rank, world_size = setup_ddp(args.ddp_timeout_minutes)
        args.device = f"cuda:{local_rank}"
    else:
        local_rank, rank, world_size = 0, 0, 1
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    setup_seed(args.seed + rank)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = load_model(args)
    params, names = select_trainable_params(model, args)
    if not params:
        raise ValueError(f"no parameters selected for --ttt_target {args.ttt_target}")

    if rank == 0:
        total = sum(p.numel() for p in model.parameters()) / 1e6
        trainable = sum(p.numel() for p in params) / 1e6
        Logger(f"Model params: {total:.2f}M total, {trainable:.2f}M trainable")
        Logger(f"TTT target: {args.ttt_target}")
        Logger("First trainable params:")
        for name in names[:20]:
            Logger(f"  {name}")

    if ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    scaler = torch.amp.GradScaler(device_type, enabled=(args.dtype == "float16"))

    tb_writer = None
    if rank == 0 and not args.no_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = os.path.join(args.tensorboard_log_dir, args.run_name)
            tb_writer = SummaryWriter(log_dir=tb_dir)
            Logger(f"TensorBoard logging: {tb_dir}")
        except ImportError:
            Logger("TensorBoard not installed, continue without TensorBoard logging.")

    dataset = load_streaming_dataset(args, rank, world_size)
    batches = iter_token_batches(dataset, tokenizer, args, args.device)

    model.train()
    os.makedirs(args.save_dir, exist_ok=True)
    start = time.time()
    log_loss = 0.0
    log_steps = 0
    optimizer.zero_grad(set_to_none=True)

    for step, (input_ids, labels) in enumerate(batches, start=1):
        lr = get_lr(step, args.max_steps, args.learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = lr

        with torch.amp.autocast(device_type, dtype=dtype):
            out = model(input_ids, labels=labels, num_iters=args.num_iters)
            loss = out["loss"] / args.accumulation_steps

        scaler.scale(loss).backward()
        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        log_loss += loss.item() * args.accumulation_steps
        log_steps += 1

        if rank == 0 and (step % args.log_interval == 0 or step == args.max_steps):
            avg_loss = log_loss / max(1, log_steps)
            elapsed = time.time() - start
            speed = elapsed / max(1, step)
            eta = speed * max(0, args.max_steps - step)
            pct = step / args.max_steps * 100
            ppl = math.exp(min(avg_loss, 20))
            Logger(
                f"Step[{step}/{args.max_steps}] | loss:{avg_loss:.4f} "
                f"ppl:{ppl:.2f} lr:{lr:.2e} | elapsed:{format_time(elapsed)} "
                f"eta:{format_time(eta)}({pct:.1f}%)"
            )
            if tb_writer:
                tb_writer.add_scalar("loss", avg_loss, step)
                tb_writer.add_scalar("ppl", ppl, step)
                tb_writer.add_scalar("lr", lr, step)
            log_loss = 0.0
            log_steps = 0

        should_save = step % args.save_interval == 0 or step == args.max_steps
        if ddp and should_save:
            dist.barrier()
        if rank == 0 and should_save:
            save_path = os.path.join(args.save_dir, f"{args.save_weight}_{args.hidden_size}.pth")
            save_model_only(model, save_path)
        if ddp and should_save:
            dist.barrier()

        if step >= args.max_steps:
            break

    if tb_writer:
        tb_writer.close()
    if rank == 0:
        Logger(f"TTT stream training complete! Total: {format_time(time.time() - start)}")
    cleanup_ddp()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stream code data and continue-train CTM TTT Layers only."
    )
    parser.add_argument("--from_weight", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="out")
    parser.add_argument("--save_weight", type=str, default="ctm_ttt_code")
    parser.add_argument("--run_name", type=str, default="ctm-ttt-code-stream")
    parser.add_argument("--tokenizer_path", type=str, default="model_tokenizer")

    parser.add_argument("--dataset_name", type=str, default="OpenCoder-LLM/opc-annealing-corpus")
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--dataset_config", type=str, default="algorithmic_corpus")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--text_field", type=str, default="text")
    parser.add_argument("--lang_filter", type=str, default="")
    parser.add_argument("--min_chars", type=int, default=64)
    parser.add_argument("--shuffle_buffer", type=int, default=10000)
    parser.add_argument("--add_eos", type=int, default=1, choices=[0, 1])

    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--num_iters", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ddp_timeout_minutes", type=int, default=60)
    parser.add_argument("--no_tensorboard", action="store_true")
    parser.add_argument("--tensorboard_log_dir", type=str, default="runs")

    parser.add_argument("--ttt_target", type=str, default="ttt_layers",
                        choices=["ttt_layers", "last_ttt_layer", "last_block", "last_output"])
    parser.add_argument("--ttt_hidden_mult", type=int, default=2)
    parser.add_argument("--ttt_gate_init", type=float, default=-2.0)

    parser.add_argument("--vocab_size", type=int, default=6400)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=12)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_input", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--memory_length", type=int, default=10)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--n_synch_out", type=int, default=512)
    parser.add_argument("--n_synch_action", type=int, default=512)
    parser.add_argument("--synapse_depth", type=int, default=3)
    parser.add_argument("--self_cond", type=int, default=1, choices=[0, 1])
    parser.add_argument("--cross_layer_state", type=int, default=1, choices=[0, 1])
    parser.add_argument("--block_size", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=512)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
