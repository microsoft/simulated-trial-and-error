import torch

from ft_datasets import (
    get_tool_dataset,
)


def get_preprocessed_dataset(
    tokenizer, train_config, split: str = "train"
) -> torch.utils.data.Dataset:

    return get_tool_dataset(
        train_config,
        tokenizer,
        split,
        max_words=train_config.max_words_dataset,
    )

