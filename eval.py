import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformers import AutoTokenizer
from model.config import CTMLLMConfig
from model.model_ctm_llm import CTMForCausalLM


def clean_response(text):
    for marker in ('</think', '<think', '💬', '💭'):
        if marker in text:
            idx = text.rfind(marker)
            close = text.find('\n', idx)
            if close != -1:
                text = text[close + 1:]
            else:
                text = text[idx + len(marker):]
    return text.strip()


def chat(model, tokenizer, prompt, device, max_new_tokens=256,
         temperature=0.85, top_p=0.85, top_k=50, repetition_penalty=1.0,
         num_iters=None):
    messages = [{'role': 'user', 'content': prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(text)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        output = model.generate(
            input_ids, max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=top_p, top_k=top_k,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=repetition_penalty, num_iters=num_iters)

    new_ids = output[0].tolist()[len(ids):]
    response = tokenizer.decode(new_ids, skip_special_tokens=True)
    return clean_response(response)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CTM-LLM Eval')
    parser.add_argument('--weight', type=str, required=True, help='Model weight file')
    parser.add_argument('--tokenizer_path', type=str, default='./model_tokenizer')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--num_hidden_layers', type=int, default=8)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--d_input', type=int, default=128)
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--memory_length', type=int, default=5)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--n_synch_out', type=int, default=256)
    parser.add_argument('--n_synch_action', type=int, default=256)
    parser.add_argument('--synapse_depth', type=int, default=2)
    parser.add_argument('--self_cond', type=int, default=1, choices=[0, 1])
    parser.add_argument('--cross_layer_state', type=int, default=1, choices=[0, 1])
    parser.add_argument('--block_size', type=int, default=4)
    parser.add_argument('--prompt', type=str, default='Hello! How are you today?')
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.85)
    parser.add_argument('--top_p', type=float, default=0.85)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--num_iters', type=int, default=None)
    args = parser.parse_args()

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

    model = CTMForCausalLM(config).to(args.device)
    ckpt = torch.load(args.weight, map_location=args.device, weights_only=False)
    state = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f'Loaded: {args.weight}')

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    response = chat(
        model, tokenizer, args.prompt, args.device, args.max_new_tokens,
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        repetition_penalty=args.repetition_penalty, num_iters=args.num_iters)
    print(f'\nUser: {args.prompt}')
    print(f'Assistant: {response}')

    while True:
        try:
            prompt = input('\nYou: ')
            if prompt.strip() in ('quit', 'exit', 'q'):
                break
            response = chat(
                model, tokenizer, prompt, args.device, args.max_new_tokens,
                temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                repetition_penalty=args.repetition_penalty, num_iters=args.num_iters)
            print(f'Assistant: {response}')
        except (EOFError, KeyboardInterrupt):
            break
