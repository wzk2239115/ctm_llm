import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
from transformers import AutoTokenizer

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
        ttt_layer=bool(args.ttt_layer),
        ttt_hidden_mult=args.ttt_hidden_mult,
        ttt_gate_init=args.ttt_gate_init,
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


@torch.inference_mode()
def complete(model, tokenizer, prompt, args):
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([ids], dtype=torch.long, device=args.device)
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    with torch.amp.autocast(device_type, dtype=dtype):
        output = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=args.repetition_penalty,
            num_iters=args.num_iters,
        )

    generated = tokenizer.decode(output[0].tolist()[len(ids):], skip_special_tokens=True)
    return generated


def parse_args():
    parser = argparse.ArgumentParser(description="Raw prefix completion eval for CTM-LLM")
    parser.add_argument("--weight", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default="./model_tokenizer")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--prompt", type=str, default="def is_palindrome(s):\n    ")
    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.75)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--repetition_penalty", type=float, default=1.12)
    parser.add_argument("--num_iters", type=int, default=None)

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
    parser.add_argument("--ttt_layer", type=int, default=0, choices=[0, 1])
    parser.add_argument("--ttt_hidden_mult", type=int, default=2)
    parser.add_argument("--ttt_gate_init", type=float, default=-2.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = load_model(args)
    print(f"Loaded: {args.weight}")
    print("\nPrompt:")
    print(args.prompt, end="")
    generated = complete(model, tokenizer, args.prompt, args)
    print(generated)
