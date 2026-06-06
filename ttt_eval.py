import argparse
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
import torch.nn.functional as F
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
    return model


def build_prompt_ids(tokenizer, prompt, device):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(text)
    return torch.tensor([ids], dtype=torch.long, device=device), text


def clean_response(text):
    for marker in ("</think", "<think", "💬", "💭"):
        if marker in text:
            idx = text.rfind(marker)
            close = text.find("\n", idx)
            text = text[close + 1:] if close != -1 else text[idx + len(marker):]
    return text.strip()


def causal_lm_loss(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def last_token_entropy(logits):
    probs = F.softmax(logits[:, -1, :], dim=-1)
    entropy = -(probs * torch.log(probs.clamp(min=1e-12))).sum(dim=-1)
    return entropy.mean()


def last_token_kl(base_logits, adapted_logits):
    base = base_logits[:, -1, :]
    adapted = adapted_logits[:, -1, :]
    return F.kl_div(
        F.log_softmax(adapted, dim=-1),
        F.softmax(base, dim=-1),
        reduction="batchmean",
    )


def target_filter(name, args):
    last_layer = f"model.layers.{args.num_hidden_layers - 1}."

    if args.ttt_target == "last_output":
        return name.startswith(last_layer + "output_proj") or \
            name.startswith(last_layer + "post_ctm_norm")
    if args.ttt_target == "ttt_layers":
        return ".ttt_layer." in name
    if args.ttt_target == "last_ttt_layer":
        return name.startswith(last_layer + "ttt_layer.")
    if args.ttt_target == "last_mlp":
        return name.startswith(last_layer + "mlp") or \
            name.startswith(last_layer + "post_ctm_norm")
    if args.ttt_target == "last_block":
        return name.startswith(last_layer)
    if args.ttt_target == "all_norms":
        return name.endswith(".weight") and (
            ".input_norm." in name or ".post_ctm_norm." in name or name == "model.norm.weight"
        )
    if args.ttt_target == "lm_head":
        return name == "lm_head.weight" or name == "model.embed_tokens.weight"
    if args.ttt_target == "all":
        return True
    return False


def select_ttt_params(model, args):
    selected = []
    selected_names = []
    for name, param in model.named_parameters():
        enable = target_filter(name, args)
        param.requires_grad_(enable)
        if enable:
            selected.append(param)
            selected_names.append(name)
    return selected, selected_names


def snapshot_params(params):
    return [p.detach().clone() for p in params]


def restore_params(params, snapshot):
    with torch.no_grad():
        for param, saved in zip(params, snapshot):
            param.copy_(saved)


def save_model_only(model, path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save({k: v.half().cpu() for k, v in raw.state_dict().items()}, path)


def ttt_adapt(model, input_ids, args):
    params, names = select_ttt_params(model, args)
    if not params:
        raise ValueError(f"no parameters selected for --ttt_target {args.ttt_target}")

    model.train()
    base_snapshot = snapshot_params(params)
    optimizer = torch.optim.AdamW(params, lr=args.ttt_lr, weight_decay=args.ttt_weight_decay)
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    labels = input_ids.clone()

    with torch.no_grad(), torch.amp.autocast(device_type, dtype=dtype):
        base_out = model(input_ids, labels=None, num_iters=args.num_iters)
        base_logits = base_out["logits"].detach()
        base_loss = causal_lm_loss(base_logits.float(), labels).detach()
        base_entropy = last_token_entropy(base_logits.float()).detach()

    step_logs = []
    for step in range(1, args.ttt_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type, dtype=dtype):
            out = model(input_ids, labels=None, num_iters=args.num_iters)
            loss = causal_lm_loss(out["logits"].float(), labels)

        loss.backward()
        if args.ttt_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, args.ttt_grad_clip)
        optimizer.step()

        with torch.no_grad(), torch.amp.autocast(device_type, dtype=dtype):
            check_out = model(input_ids, labels=None, num_iters=args.num_iters)
            check_logits = check_out["logits"].detach()
            check_loss = causal_lm_loss(check_logits.float(), labels).detach()
            check_entropy = last_token_entropy(check_logits.float()).detach()
            kl = last_token_kl(base_logits.float(), check_logits.float()).detach()
        step_logs.append({
            "step": step,
            "loss": check_loss.item(),
            "entropy": check_entropy.item(),
            "kl": kl.item(),
        })
        print(
            f"ttt step {step}: loss={check_loss.item():.6f} "
            f"entropy={check_entropy.item():.6f} kl={kl.item():.6f}",
            flush=True,
        )

    final = step_logs[-1]
    accepted = (
        final["loss"] <= base_loss.item() - args.ttt_min_loss_drop and
        final["kl"] <= args.ttt_kl_budget
    )

    if args.ttt_rollback and not accepted:
        restore_params(params, base_snapshot)

    model.eval()
    return {
        "accepted": accepted,
        "rolled_back": bool(args.ttt_rollback and not accepted),
        "target": args.ttt_target,
        "num_params": sum(p.numel() for p in params),
        "param_names": names[:20],
        "base_loss": base_loss.item(),
        "base_entropy": base_entropy.item(),
        "final_loss": final["loss"],
        "final_entropy": final["entropy"],
        "final_kl": final["kl"],
        "steps": step_logs,
    }


@torch.inference_mode()
def generate_response(model, tokenizer, input_ids, prompt_len, args):
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
    new_ids = output[0].tolist()[prompt_len:]
    return clean_response(tokenizer.decode(new_ids, skip_special_tokens=True))


def parse_args():
    parser = argparse.ArgumentParser(
        description="CTM-LLM test-time training eval with real parameter updates."
    )
    parser.add_argument("--weight", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default="./model_tokenizer")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--prompt", type=str, default="Explain what a neural network is in simple words.")
    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--repetition_penalty", type=float, default=1.08)
    parser.add_argument("--num_iters", type=int, default=None)

    parser.add_argument("--ttt_target", type=str, default="ttt_layers",
                        choices=[
                            "ttt_layers", "last_ttt_layer", "last_output", "last_mlp",
                            "last_block", "all_norms", "lm_head", "all",
                        ])
    parser.add_argument("--ttt_steps", type=int, default=1)
    parser.add_argument("--ttt_lr", type=float, default=1e-5)
    parser.add_argument("--ttt_weight_decay", type=float, default=0.0)
    parser.add_argument("--ttt_grad_clip", type=float, default=1.0)
    parser.add_argument("--ttt_kl_budget", type=float, default=0.25)
    parser.add_argument("--ttt_min_loss_drop", type=float, default=0.0)
    parser.add_argument("--no_ttt_rollback", dest="ttt_rollback", action="store_false")
    parser.set_defaults(ttt_rollback=True)
    parser.add_argument("--save_ttt_weight", type=str, default=None)

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
    parser.add_argument("--ttt_layer", type=int, default=1, choices=[0, 1])
    parser.add_argument("--ttt_hidden_mult", type=int, default=2)
    parser.add_argument("--ttt_gate_init", type=float, default=-2.0)
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    input_ids, prompt_text = build_prompt_ids(tokenizer, args.prompt, args.device)
    if input_ids.size(1) < 3:
        raise ValueError("prompt is too short for prefix TTT")

    model = load_model(args)
    model.eval()
    print(f"Loaded: {args.weight}")
    print(f"Prompt tokens: {input_ids.size(1)}")
    print(f"TTT target: {args.ttt_target}")

    report = ttt_adapt(model, input_ids, args)
    print("\nTTT report")
    for key in [
        "accepted", "rolled_back", "target", "num_params", "base_loss",
        "final_loss", "base_entropy", "final_entropy", "final_kl",
    ]:
        print(f"  {key}: {report[key]}")
    print("  first_params:")
    for name in report["param_names"]:
        print(f"    {name}")

    if args.save_ttt_weight and report["accepted"]:
        save_model_only(model, args.save_ttt_weight)
        print(f"Saved accepted TTT-adapted weight: {args.save_ttt_weight}")

    response = generate_response(model, tokenizer, input_ids, input_ids.size(1), args)
    print(f"\nUser: {args.prompt}")
    print(f"Assistant: {response}")


if __name__ == "__main__":
    main()
