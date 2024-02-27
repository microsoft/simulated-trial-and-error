# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import copy
import json
import os
import torch

from sentencepiece import SentencePieceProcessor
from torch.utils.data import Dataset
from typing import List, Literal, Optional, Tuple, TypedDict, Union

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


def format_tokens(dialogs, tokenizer):
    return [format_tokens_single(dialog, tokenizer) for dialog in dialogs]


def format_tokens_single(dialog, tokenizer):

    dialog_history, last_msg = format_texts_single(dialog)
    
    return format_tokens_split(dialog_history, last_msg, tokenizer)


def format_tokens_split(dialog_history, last_msg, tokenizer):
    dialog_tokens: List[int] = sum(
        [tokenizer.encode(temp) for temp in dialog_history],
        [],
    )

    dialog_tokens += tokenizer.encode(last_msg)

    return dialog_tokens


def format_texts_single(dialog):
    if dialog[0]["role"] != "system":
        dialog = [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
            }
        ] + dialog
    dialog = [
        {
            "role": dialog[1]["role"],
            "content": B_SYS
            + dialog[0]["content"]
            + E_SYS
            + dialog[1]["content"],
        }
    ] + dialog[2:]
    assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
        [msg["role"] == "assistant" for msg in dialog[1::2]]
    ), (
        "model only supports 'system','user' and 'assistant' roles, "
        "starting with user and alternating (u/a/u/a/u...)"
    )
    """
    Please verify that yout tokenizer support adding "[INST]", "[/INST]" to your inputs.
    Here, we are adding it manually.
    """
    dialog_history = [
        f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} " for prompt, answer in zip(dialog[::2], dialog[1::2])
    ]

    assert (
        dialog[-1]["role"] == "user"
    ), f"Last message must be from user, got {dialog[-1]['role']}"

    last_msg = f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}"

    return dialog_history, last_msg


class InstructionDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=30):
        with open(dataset_config.data_path, "r", encoding='utf-8') as f:
            self.ann = json.load(f)
        if partition == "train":
            self.ann = self.ann
        else:
            self.ann = self.ann

        self.max_words = max_words
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        prompt = format_tokens_single(ann['dialog_history'], tokenizer=self.tokenizer)
        output = self.tokenizer.encode(ann['response'])

        assert output[0] == self.tokenizer.bos_token_id
        del output[0]
        example = prompt + output
        example.append(self.tokenizer.eos_token_id)

        IGNORE_INDEX = -100
        prompt = torch.tensor(prompt, dtype=torch.int64)
        example = torch.tensor(example, dtype=torch.int64)
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = IGNORE_INDEX

        padding = self.max_words - example.shape[0]

        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
            labels = torch.cat((labels, IGNORE_INDEX*torch.ones(padding, dtype=torch.int64)))

        elif padding < 0:
            # truncate
            example = example[: self.max_words]
            labels = labels[: self.max_words]
        
        example_mask = example.ge(0)
        example[~example_mask] = 0
        example_mask = example_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask": example_mask,
        }
