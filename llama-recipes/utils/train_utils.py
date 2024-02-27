# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import sys
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset
from tqdm import tqdm

from torch.nn import functional as F
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer
from torch.distributed.fsdp import StateDictType
import torch.distributed as dist
from pkg_resources import packaging
from .memory_utils import MemoryTrace
import model_checkpointing
import torch.cuda.nccl as nccl
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from policies import bfSixteen, fpSixteen, bfSixteen_mixed, get_llama_wrapper


def set_tokenizer_params(tokenizer: LlamaTokenizer):
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"


# Converting Bytes to Megabytes
def byte2mb(x):
    return int(x / 2 ** 20)


def train(model, train_dataloader, eval_dataloader, tokenizer, optimizer, lr_scheduler, gradient_accumulation_steps,
          train_config, fsdp_config=None, local_rank=None, rank=None):

    # Create a gradient scaler for fp16
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.cuda.amp.GradScaler()

    train_prep = []
    train_loss = []
    val_prep = []
    val_loss = []
    results = {}
    best_val_loss = float("inf")
    for epoch in range(train_config.num_epochs):

        train_dataloader.sampler.set_epoch(epoch)

        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            total_loss = 0.0
            data_set_len = 0

            for step, batch in enumerate(tqdm(train_dataloader, colour="blue", desc=f"Training Epoch{epoch}")):
                for key in batch.keys():
                    # print("key:", key, "| val:", type(batch[key]), len(batch[key]))
                    if train_config.enable_fsdp:
                        batch[key] = batch[key].to(local_rank)
                    elif not train_config.quantization:
                        batch[key] = batch[key].to('cuda')
                loss = model(**batch).loss
                loss = loss / gradient_accumulation_steps
                total_loss += loss.detach().float()
                first_key = next(iter(batch))
                data_set_len += len(batch[first_key])
                if train_config.use_fp16:
                    # if fp16 is enabled, use gradient scaler to handle gradient update
                    scaler.scale(loss).backward()
                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        lr_scheduler.step()
                else:
                    # regular backpropagation when fp16 is not used
                    loss.backward()
                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        optimizer.step()
                        optimizer.zero_grad()
                        lr_scheduler.step()

                print(f"\n step {step} is completed and loss is {loss.detach().float()}")

        # Reducing total_loss across all devices if there's more than one CUDA device
        if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        train_epoch_loss = total_loss / data_set_len
        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(train_perplexity.item())
        train_loss.append(train_epoch_loss.item())

        print(f"Max CUDA memory allocated was {memtrace.peak} GB")
        print(f"Max CUDA memory reserved was {memtrace.max_reserved} GB")
        print(f"Cuda Malloc retires : {memtrace.cuda_malloc_retires}")
        print(f"CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB")

        if train_config.run_validation:
            eval_ppl, eval_epoch_loss = evaluation(model, train_config, eval_dataloader, rank, tokenizer)
            if local_rank == 0 and eval_epoch_loss < best_val_loss:
                best_val_loss = eval_epoch_loss
                print(f"best eval loss on epoch {epoch} is {best_val_loss}")
            val_loss.append(eval_epoch_loss.item())
            val_prep.append(eval_ppl.item())

        if train_config.save_model:
            if ((not train_config.save_only_at_last) and (epoch + 1) % train_config.save_epoch_interval == 0) or (
                    epoch + 1 == train_config.num_epochs):
                if train_config.use_peft:

                    print(f"we are in the saving the PEFT modules")
                    model.save_pretrained(train_config.output_dir)
                    print(f"PEFT modules are saved in {train_config.output_dir} directory")

                else:
                    if train_config.save_with_hf:
                        print("saving model checkpoint w/ hf")
                        model_checkpointing.save_model_checkpoint_hf(
                            model,
                            tokenizer,
                            rank,
                            train_config,
                            epoch=epoch + 1,
                        )

                    elif fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:

                        model_checkpointing.save_model_checkpoint(
                            model, optimizer, rank, train_config, epoch=epoch  # epoch=1
                        )
                    elif fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
                        print(" we are about to save the models *******")

                        model_checkpointing.save_model_and_optimizer_sharded(model, rank, train_config)
                        if train_config.save_optimizer:
                            model_checkpointing.save_model_and_optimizer_sharded(model, rank, train_config,
                                                                                 optim=optimizer)

                    if train_config.save_optimizer:
                        model_checkpointing.save_optimizer_checkpoint(
                            model, optimizer, rank, train_config, epoch=epoch  # epoch=1
                        )

        print(f"Epoch {epoch + 1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}")

    results['train_prep'] = train_prep
    results['train_loss'] = train_loss
    if train_config.run_validation:
        results['val_prep'] = val_prep
        results['val_loss'] = val_loss

    return results


def evaluation(model, train_config, eval_dataloader, local_rank, tokenizer):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions

    Returns: eval_ppl, eval_epoch_loss
    """
    model.eval()
    eval_preds = []
    eval_loss = 0.0  # Initialize evaluation loss
    eval_dataset_len = 0
    with MemoryTrace() as memtrace:
        for step, batch in enumerate(tqdm(eval_dataloader, colour="green", desc="evaluating Epoch")):
            for key in batch.keys():
                if train_config.enable_fsdp:
                    batch[key] = batch[key].to(local_rank)
                else:
                    batch[key] = batch[key].to('cuda')
            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                # Forward pass and compute loss
                outputs = model(**batch)
                loss = outputs.loss
                eval_loss += loss.detach().float()
                first_key = next(iter(batch))
                eval_dataset_len += len(batch[first_key])

            # Decode predictions and add to evaluation predictions list
            preds = torch.argmax(outputs.logits, -1)
            eval_preds.extend(
                tokenizer.batch_decode(preds.detach().cpu().numpy(), skip_special_tokens=True)
            )

    # If there's more than one CUDA device, reduce evaluation loss across all devices
    if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)

    # Compute average loss and perplexity
    eval_epoch_loss = eval_loss / eval_dataset_len
    eval_ppl = torch.exp(eval_epoch_loss)

    # Print evaluation metrics
    print(f" {eval_ppl=} {eval_epoch_loss=}")
    return eval_ppl, eval_epoch_loss


def freeze_transformer_layers(model, num_layer):
    for i, layer in enumerate(model.model.layers):
        if i < num_layer:
            for param in layer.parameters():
                param.requires_grad = False


def check_frozen_layers_peft_model(model):
    for i, layer in enumerate(model.base_model.model.model.layers):
        for name, param in layer.named_parameters():
            print(f"Layer {i}, parameter {name}: requires_grad = {param.requires_grad}")


def setup():
    """Initialize the process group for distributed training"""
    dist.init_process_group("nccl")


def setup_environ_flags(rank):
    """Set environment flags for debugging purposes"""
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    if rank == 0:
        print(f"--> Running with torch dist debug set to detail")


def cleanup():
    """Clean up the process group after training"""
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        print(f"Clearing GPU cache for all ranks")
    torch.cuda.empty_cache()


def get_parameter_dtypes(model):
    """Get the data types of model parameters"""
    parameter_dtypes = {}
    for name, parameter in model.named_parameters():
        parameter_dtypes[name] = parameter.dtype
    return parameter_dtypes


def print_model_size(model, config, rank: int = 0) -> None:
    """
    Print model name, the number of trainable parameters and initialization time.

    Args:
        model: The PyTorch model.
        model_name (str): Name of the model.
        init_time_start (float): Initialization start time.
        init_time_end (float): Initialization end time.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        print(f"--> Model {config.model_name}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> {config.model_name} has {total_params / 1e6} Million params\n")


def get_policies(cfg, rank, model_type='llama'):
    """Get the policies for mixed precision and fsdp wrapping"""

    verify_bfloat_support = (
            torch.version.cuda
            and torch.cuda.is_bf16_supported()
            and packaging.version.parse(torch.version.cuda).release >= (11, 0)
            and dist.is_nccl_available()
            and nccl.version() >= (2, 10)
    )

    mixed_precision_policy = None
    wrapping_policy = None

    # Mixed precision
    if cfg.mixed_precision:
        bf16_ready = verify_bfloat_support

        if bf16_ready and not cfg.use_fp16:
            mixed_precision_policy = bfSixteen_mixed
            if rank == 0:
                print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif cfg.use_fp16:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print(f"FP16 enabled")
        else:
            print(f"bFloat16 support not present. Using FP32, and not mixed precision")
    wrapping_policy = get_llama_wrapper(model_type=model_type)
    return mixed_precision_policy, wrapping_policy
