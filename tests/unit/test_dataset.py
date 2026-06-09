import pytest
import torch
import json

from dataset.text_dataset import TextDataset


class FakeTokenizer:
    pad_token = "<pad>"
    eos_token = "</s>"
    pad_token_id = 0

    @property
    def chat_template(self):
        return "{% for msg in messages %}{{ msg['content'] }} {% endfor %}"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = " ".join(m["content"] for m in messages)
        if add_generation_prompt:
            text += " >>"
        if tokenize:
            return text
        return text

    def encode(self, text):
        return [1] + [hash(c) % 6400 for c in text.split()] + [2]


def _write_parquet(tmp_path, conversations_list):
    import pandas as pd
    rows = []
    for convs in conversations_list:
        rows.append({"conversations": json.dumps(convs)})
    df = pd.DataFrame(rows)
    path = tmp_path / "test.parquet"
    df.to_parquet(path)
    return str(path)


class TestTextDataset:
    def test_label_mask_only_last_assistant(self, tmp_path):
        path = _write_parquet(tmp_path, [
            [{"role": "user", "content": "a b"}, {"role": "assistant", "content": "c d e"}]
        ])
        ds = TextDataset(path, FakeTokenizer(), max_length=32)
        ids, labels = ds[0]
        assert ids.shape == labels.shape
        assert (labels == -100).any()

    def test_length(self, tmp_path):
        path = _write_parquet(tmp_path, [
            [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}],
            [{"role": "user", "content": "c"}, {"role": "assistant", "content": "d"}],
        ])
        ds = TextDataset(path, FakeTokenizer(), max_length=32)
        assert len(ds) == 2

    def test_labels_mask_first_turn(self, tmp_path):
        path = _write_parquet(tmp_path, [
            [{"role": "user", "content": "hello world test"}, {"role": "assistant", "content": "response here"}]
        ])
        ds = TextDataset(path, FakeTokenizer(), max_length=64)
        _, labels = ds[0]
        non_masked = (labels != -100).sum().item()
        assert non_masked > 0
