import json
import torch
import pandas as pd
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """
    Text-only dataset loaded from parquet files with a 'conversations' column.
    Conversations are formatted via the tokenizer's chat template.
    Labels are masked (-100) except for the last assistant response.
    """

    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = pd.read_parquet(data_path)
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        convs = row['conversations']
        if isinstance(convs, str):
            convs = json.loads(convs)

        messages = [{'role': c['role'], 'content': c['content']} for c in convs]

        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)
        full_ids = self.tokenizer.encode(full_text)[:self.max_length]

        input_ids = torch.tensor(
            full_ids + [self.pad_id] * max(0, self.max_length - len(full_ids)),
            dtype=torch.long)
        if len(full_ids) < self.max_length:
            input_ids[len(full_ids):] = self.pad_id

        labels = input_ids.clone()
        labels[:] = -100

        if convs and convs[-1]['role'] == 'assistant':
            prompt_conv = convs[:-1]
            if prompt_conv:
                prompt_text = self.tokenizer.apply_chat_template(
                    prompt_conv, tokenize=False, add_generation_prompt=True)
            else:
                prompt_text = self.tokenizer.apply_chat_template(
                    [], tokenize=False, add_generation_prompt=True)
            prompt_ids = self.tokenizer.encode(prompt_text)
            start = min(len(prompt_ids), len(full_ids))
            for i in range(start, len(full_ids)):
                labels[i] = full_ids[i]

        return input_ids[:self.max_length], labels[:self.max_length]
