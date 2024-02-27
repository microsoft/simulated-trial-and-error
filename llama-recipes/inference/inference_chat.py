import fire
import torch
import os
import sys
import warnings
from typing import List
import json
import numpy as np

from peft import PeftModel, PeftConfig
from transformers import LlamaTokenizer, AutoTokenizer
from model_utils import load_model, load_peft_model
from chat_utils import read_dialogs_from_file, format_tokens, format_tokens_single
from copy import deepcopy

import time
from transformers import StoppingCriteria, StoppingCriteriaList


def up_sam(l, factor):
    return_l = []
    for _ in range(factor):
        return_l = return_l + l
    return return_l


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False


def main(
    model_name,
    data_path,  # should contain either
                #   1) a list of dictionary items, each has a key 'dialog_history' OR 'query' (specified by the param item_type)
                #   2) a dictionary each mapping to a list of items as above
    save_path,  # location to which the output will be stored
    item_type,  # 'dialog' / 'query'
    sys_msg_dir: str = 'None',       # if set, add/overwrite the system message
    model_type: str = 'llama',
    do_ds: bool = False,  # whether do down-sampling
    ds_ratio=0.1,         # down-sampling ratio
    save_key: str = 'model_output',  # the key which model output will be saved onto
    auto_stop: bool = True,  # stop when generating "Observation"
    peft_model: str = None,
    quantization: bool = False,
    max_new_tokens=96,  # The maximum numbers of tokens to generate
    seed: int = 42,
    do_sample: bool = False,
    use_cache: bool = True,
    top_p: float = 1.0,
    temperature: float = 1.0,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    length_penalty: int = 1,
):
    with open(data_path, "r", encoding='utf-8') as f:
        test_data = json.load(f)

    if sys_msg_dir != 'None':
        with open(sys_msg_dir, "r", encoding='utf-8') as f:
            sys_msg = json.load(f)

    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = load_model(model_name, quantization, model_type=model_type)
    if peft_model:
        model = load_peft_model(model, peft_model)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens(
        {

            "pad_token": "<PAD>",
        }
    )

    if auto_stop:
        stop_tokens = [torch.tensor([21651, 362]), torch.tensor([6039, 2140, 362])]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_tokens)])
        kwargs = {"stopping_criteria": stopping_criteria}
    else:
        kwargs = {}

    # start inference
    is_list = False
    if type(test_data) == list:
        is_list = True
        test_data = {0: test_data}
        print("starting inference (list)")
    else:
        assert type(test_data) == dict
        print("starting inference (dict), total # of keys:", len(test_data))

    for key in test_data:
        examples = test_data[key]

        if do_ds:
            inds = np.random.choice(len(examples), round(ds_ratio * len(examples)), replace=False).tolist()
            examples = [examples[i] for i in inds]

        for i in range(len(examples)):
            print(key, i)
            item = examples[i]
            if item_type == 'dialog':
                dialog = item['dialog_history']
            elif item_type == 'query':
                dialog = [{"role": "user", "content": item['query']}]
            else:
                assert False

            # insert/override sys msg if given
            if sys_msg_dir != 'None':
                if dialog[0]['role'] == 'system':
                    dialog[0]['content'] = sys_msg
                elif dialog[0]['role'] == 'user':
                    dialog = [{"role": "system", "content": sys_msg}] + dialog
                else:
                    assert False

            chat = format_tokens_single(dialog, tokenizer)
            input_len = len(chat)
            start = time.perf_counter()
            with torch.no_grad():
                tokens = torch.tensor(chat).long()
                tokens = tokens.unsqueeze(0)
                tokens = tokens.to("cuda:0")
                outputs = model.generate(
                    tokens,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    top_p=top_p,
                    temperature=temperature,
                    use_cache=use_cache,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    **kwargs
                )
                e2e_inference_time = (time.perf_counter() - start) * 1000
                print(f"the inference time is {e2e_inference_time} ms")
                output_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

                item[save_key] = output_text
            examples[i] = item
        test_data[key] = examples

    if is_list:
        test_data = test_data[0]

    with open(save_path, "w", encoding='utf-8') as f:
        json.dump(test_data, f)


if __name__ == "__main__":
    fire.Fire(main)


