import json
import os
import random
import math
import numpy as np
import torch
from transformers import AutoTokenizer
from model.config import CTMLLMConfig
from model.model_ctm_llm import CTMForCausalLM
from model.model_transformer import TransformerForCausalLM


def Logger(content):
    print(content, flush=True)


def get_lr(current_step, total_steps, base_lr):
    return base_lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_model_params(model):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    Logger(f'Model Params: {total:.2f}M total, {trainable:.2f}M trainable')


def create_model(config: CTMLLMConfig, device='cpu'):
    model_type = getattr(config, "model_type", "ctm")
    if model_type == "ctm":
        model = CTMForCausalLM(config).to(device)
    elif model_type == "transformer":
        model = TransformerForCausalLM(config).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    log_model_params(model)
    return model


def load_tokenizer(tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.chat_template is None:
        config_path = os.path.join(tokenizer_path, 'tokenizer_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            if 'chat_template' in config:
                tokenizer.chat_template = config['chat_template']
                Logger(f'chat_template loaded from {config_path}')
    return tokenizer


def save_checkpoint(model, optimizer, epoch, step, save_path, scaler=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    state = {
        'model': {k: v.half().cpu() for k, v in model.state_dict().items()},
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
    }
    if scaler is not None:
        state['scaler'] = scaler.state_dict()
    tmp = save_path + '.tmp'
    torch.save(state, tmp)
    os.replace(tmp, save_path)
    Logger(f'Checkpoint saved: {save_path}')


def load_checkpoint(save_path, model, optimizer=None, scaler=None, device='cpu'):
    if not os.path.exists(save_path):
        return 0, 0
    ckpt = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'], strict=False)
    if optimizer and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scaler and 'scaler' in ckpt:
        scaler.load_state_dict(ckpt['scaler'])
    Logger(f'Checkpoint loaded: {save_path}')
    return ckpt.get('epoch', 0), ckpt.get('step', 0)
