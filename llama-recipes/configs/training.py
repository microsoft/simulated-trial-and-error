# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
from dataclasses import dataclass
from typing import ClassVar


@dataclass
class train_config:
    model_name: str = ""
    data_path = "?"
    max_words_dataset = 256
    add_token_list: str = 'None'
    enable_fsdp: bool = False
    run_validation: bool = False
    batch_size_training: int = 4
    num_epochs: int = 1
    num_workers_dataloader: int = 1
    lr: float = 1e-4
    weight_decay: float = 0.0
    gamma: float = 0.85
    seed: int = 42
    use_fp16: bool = False
    mixed_precision: bool = True
    val_batch_size: int = 1
    micro_batch_size: int = 4
    peft_method: str = "lora"  # None , llama_adapter, prefix
    use_peft: bool = False
    output_dir: str = "PATH/to/save/PEFT/model"
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: bool = False
    one_gpu: bool = False
    save_model: bool = True
    dist_checkpoint_root_folder: str = "PATH/to/save/FSDP/model"  # will be used if using FSDP
    dist_checkpoint_folder: str = "fine-tuned"  # will be used if using FSDP
    save_optimizer: bool = False  # will be used if using FSDP
    checkpoint_folder: str = "ckpts"  # checkpoint save dir w/ hf
    save_with_hf: bool = True  # save in hf format (save_pretrained)
    save_only_at_last: bool = False
    warmup_ratio: float = 0.03
    save_epoch_interval: int = 4
    model_type: str = 'llama'  # / 'mistral'

