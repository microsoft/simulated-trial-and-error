# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import torch.distributed as dist
import torch.nn as nn
import torch

from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer

from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    BackwardPrefetch,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

import functools
from typing import Type


def get_size_policy(min_params=1e8):
    num_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=min_params
    )
    return num_wrap_policy


def get_llama_wrapper(model_type='llama'):
    """we register our main layer class and use the fsdp transformer wrapping policy
    ensures embedding layers are in the root fsdp unit for shared access and that fsdp units map to transformer layers
    """
    # ====   use new transformer wrapper
    if model_type == 'llama':
        wrap_layer = LlamaDecoderLayer
    elif model_type == 'mistral':
        wrap_layer = MistralDecoderLayer
    else:
        assert False

    llama_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            wrap_layer,
        },
    )

    return llama_auto_wrap_policy
