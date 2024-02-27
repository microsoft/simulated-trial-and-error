# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Literal, Optional, Tuple, TypedDict, Union
import json

Role = Literal["user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


Dialog = List[Message]

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
        f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} " for prompt, answer in
        zip(dialog[::2], dialog[1::2])
    ]

    assert (
            dialog[-1]["role"] == "user"
    ), f"Last message must be from user, got {dialog[-1]['role']}"

    last_msg = f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}"

    return dialog_history, last_msg


def read_dialogs_from_file(file_path):
    with open(file_path, 'r') as file:
        dialogs = json.load(file)
    return dialogs
