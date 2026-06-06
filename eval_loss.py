import argparse
import math
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from dataset.text_dataset import TextDataset
from model.config import CTMLLMConfig
from model.model_ctm_llm import CTMForCausalLM


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
    )


def load_model(args):
    model = CTMForCausalLM(build_config(args)).to(args.device)
    ckpt = torch.load(args.weight, map_location=args.device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print("  missing:", missing[:10])
        if unexpected:
            print("  unexpected:", unexpected[:10])
    model.eval()
    return model


def make_subset(dataset, args):
    indices = list(range(len(dataset)))
    rng = random.Random(args.seed)
    rng.shuffle(indices)

    if args.split == "head":
        indices = list(range(len(dataset)))
    elif args.split == "tail":
        indices = list(range(len(dataset) - 1, -1, -1))

    if args.samples > 0:
        indices = indices[: min(args.samples, len(indices))]
    return Subset(dataset, indices)


@torch.inference_mode()
def evaluate(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    dataset = TextDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    subset = make_subset(dataset, args)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=("cuda" in args.device),
        drop_last=False,
    )

    model = load_model(args)
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    total_loss = 0.0
    total_tokens = 0
    total_batches = 0

    for step, (input_ids, labels) in enumerate(loader, start=1):
        input_ids = input_ids.to(args.device, non_blocking=True)
        labels = labels.to(args.device, non_blocking=True)
        valid = labels[..., 1:] != -100
        token_count = valid.sum().item()
        if token_count == 0:
            continue

        with torch.amp.autocast(device_type, dtype=dtype):
            out = model(input_ids, labels=labels, num_iters=args.num_iters)

        total_loss += out["loss"].item() * token_count
        total_tokens += token_count
        total_batches += 1

        if args.log_interval > 0 and step % args.log_interval == 0:
            mean_loss = total_loss / max(1, total_tokens)
            print(
                f"step={step} loss={mean_loss:.4f} ppl={math.exp(min(mean_loss, 20)):.2f} "
                f"tokens={total_tokens}",
                flush=True,
            )

    mean_loss = total_loss / max(1, total_tokens)
    print("\nEval complete")
    print(f"  weight  : {args.weight}")
    print(f"  data    : {args.data_path}")
    print(f"  split   : {args.split}")
    print(f"  samples : {len(subset)}")
    print(f"  batches : {total_batches}")
    print(f"  tokens  : {total_tokens}")
    print(f"  loss    : {mean_loss:.6f}")
    print(f"  ppl     : {math.exp(min(mean_loss, 20)):.3f}")


def parse_args():
    parser = argparse.ArgumentParser(description="CTM-LLM loss/perplexity eval")
    parser.add_argument("--weight", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="dataset_data/sft_t2a_mini.parquet")
    parser.add_argument("--tokenizer_path", type=str, default="model_tokenizer")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--split", type=str, default="random", choices=["random", "head", "tail"])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log_interval", type=int, default=20)

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
    parser.add_argument("--num_iters", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
