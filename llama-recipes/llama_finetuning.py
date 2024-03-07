# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import sys
from typing import List, Union

import fire
import torch
import transformers
from datasets import load_dataset
import os.path as osp
from tqdm import tqdm
import json

# Unused imports removed
from utils import fsdp_auto_wrap_policy
from transformers import (
    LlamaForCausalLM,
    MistralForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    default_data_collator,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
import torch.distributed as dist

# Unused imports removed
from utils.train_utils import (
    set_tokenizer_params,
    train,
    evaluation,
    freeze_transformer_layers,
    check_frozen_layers_peft_model,
    setup,
    setup_environ_flags,
    cleanup,
    clear_gpu_cache,
    get_parameter_dtypes,
    print_model_size,
    get_policies  
)

from utils.dataset_utils import get_preprocessed_dataset

from utils.config_utils import (
    update_config,
    generate_peft_config,
    # generate_dataset_config,
)
from peft import get_peft_model, TaskType, prepare_model_for_int8_training
import configs
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)
from torch.utils.data import DistributedSampler
import policies
from policies import AnyPrecisionAdamW
from configs import fsdp_config, train_config
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from pkg_resources import packaging
import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer


def main(**kwargs):

    print("cuda version:", torch.version.cuda)

    # Update the configuration for the training and sharding process
    update_config((train_config, fsdp_config), **kwargs)

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)

    if train_config.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        torch.cuda.set_device(rank)
        setup_environ_flags(rank)
    
    # Calculate gradient accumulation steps
    gradient_accumulation_steps = train_config.batch_size_training // train_config.micro_batch_size
    
    print("***model type***:", train_config.model_type)
    if train_config.model_type == 'llama':
        _model_class = LlamaForCausalLM
        _wrap_layer = LlamaDecoderLayer
    elif train_config.model_type == 'mistral':
        _model_class = MistralForCausalLM
        _wrap_layer = MistralDecoderLayer
    else:
         assert False
    # Load the pre-trained model and setup its configuration
    model = _model_class.from_pretrained(
        train_config.model_name,
        load_in_8bit=True if train_config.quantization else None,
        device_map="auto" if train_config.quantization else None,
    )

    # Load the tokenizer and add special tokens
    tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name)
    tokenizer.add_special_tokens(
        {
            "pad_token": '<PAD>',
        }
    )

    # add tokens
    print("# tokens before:", len(tokenizer))
    if train_config.add_token_list != 'None':
        assert train_config.add_token_list.endswith(".json")
        with open(train_config.add_token_list, "r", encoding='utf-8') as f:
            add_token_list = json.load(f)
            assert type(add_token_list) == list
        print("adding tokens:")
        print(add_token_list)
        tokenizer.add_tokens(add_token_list)
        print("# tokens after:", len(tokenizer))
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

    print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)
    
    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        model = prepare_model_for_int8_training(model)
        
    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)

    if train_config.use_peft:
        peft_config = generate_peft_config(train_config, kwargs)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    # setting up FSDP if enable_fsdp is enabled
    if train_config.enable_fsdp:
        if not train_config.use_peft and train_config.freeze_layers:
            freeze_transformer_layers(train_config.num_freeze_layers)

        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank, model_type=train_config.model_type)
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, _wrap_layer)
   
        model = FSDP(
            model,
            auto_wrap_policy= my_auto_wrapping_policy if train_config.use_peft else wrapping_policy,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=False,
        )
        if fsdp_config.fsdp_activation_checkpointing:
            policies.apply_fsdp_checkpointing(model)
    elif not train_config.quantization and not train_config.enable_fsdp:
        model.to("cuda")

    # dataset_config = generate_dataset_config(train_config, kwargs)
    
    # Load and preprocess the dataset for training and validation
    dataset_train = get_preprocessed_dataset(
        tokenizer,
        # dataset_config,
        train_config,
        split="train",
    )
    
    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Training Set Length = {len(dataset_train)}")

    dataset_val = get_preprocessed_dataset(
        tokenizer,
        # dataset_config,
        train_config,
        split="test",
    )
    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Validation Set Length = {len(dataset_val)}")

    train_sampler = None
    val_sampler = None
    train_dataloader = None
    eval_dataloader = None
    if train_config.enable_fsdp:
        train_sampler = DistributedSampler(
            dataset_train,
            rank=dist.get_rank(),
            num_replicas=dist.get_world_size(),
            shuffle=True,
        )
        if train_config.run_validation:
            val_sampler = DistributedSampler(
                dataset_val,
                rank=dist.get_rank(),
                num_replicas=dist.get_world_size(),

            )

    if rank == 0:
        print("**************************************")
        print("world size:", dist.get_world_size())
        print("batch size:", train_config.batch_size_training)
        print("val batch size:", train_config.val_batch_size)
        print("acc_steps:", gradient_accumulation_steps)
            
    # Create DataLoaders for the training and validation dataset
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        # batch_size=train_config.batch_size_training,
        batch_size=train_config.micro_batch_size,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        sampler=train_sampler if train_sampler else None,
        drop_last=True,
        collate_fn=default_data_collator,
    )

    print("train_dataloader len:", len(train_dataloader))

    if train_config.run_validation:
        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=train_config.val_batch_size,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            sampler=val_sampler if val_sampler else None,
            drop_last=True,
            collate_fn=default_data_collator,
        )
    
        print("eval_dataloader len:", len(eval_dataloader))

    # Initialize the optimizer and learning rate scheduler
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            model.parameters(),
            lr=train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=0.0,
        )
    # scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)
    num_total_steps = len(train_dataloader) // gradient_accumulation_steps * train_config.num_epochs
    warmup_steps = int(num_total_steps * train_config.warmup_ratio)

    print("**********************total steps:", num_total_steps, "warmup steps:", warmup_steps)

    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_total_steps)

    # Start the training process
    results = train(
        model,
        train_dataloader,
        eval_dataloader, 
        tokenizer,
        optimizer,
        scheduler,
        gradient_accumulation_steps,
        train_config,
        fsdp_config if train_config.enable_fsdp else None,
        local_rank if train_config.enable_fsdp else None,
        rank if train_config.enable_fsdp else None,
    )
    if not train_config.enable_fsdp or rank==0:
        [print(f'Key: {k}, Value: {v}') for k, v in results.items()]


if __name__ == "__main__":
    fire.Fire(main)
