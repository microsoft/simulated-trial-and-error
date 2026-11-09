# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from pathlib import Path

from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaConfig, MistralForCausalLM

# Function to load the main model for text generation
def load_model(model_name, quantization, model_type='llama'):
    if model_type == 'llama':
        _class = LlamaForCausalLM
    elif model_type == 'mistral':
        _class = MistralForCausalLM

    model = _class.from_pretrained(
        model_name,
        return_dict=True,
        load_in_8bit=quantization,
        device_map="auto",
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    return model


# Function to load the PeftModel for performance optimization
def load_peft_model(model, peft_model):
    adapter_path = Path(peft_model)
    if not adapter_path.is_dir():
        adapter_path = Path(snapshot_download(
            repo_id=peft_model,
            allow_patterns=["*.json", "*.safetensors"],
        ))
    if not (adapter_path / "adapter_model.safetensors").is_file():
        raise ValueError(
            f"Adapter {peft_model} must contain adapter_model.safetensors; "
            "pickle-based adapter weights are not supported."
        )
    return PeftModel.from_pretrained(model, str(adapter_path))

# Loading the model from config to load FSDP checkpoints into that
def load_llama_from_config(config_path):
    model_config = LlamaConfig.from_pretrained(config_path) 
    model = LlamaForCausalLM(config=model_config)
    return model
    
    