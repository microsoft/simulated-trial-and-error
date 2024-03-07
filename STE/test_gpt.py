import json
import os
import matplotlib.pyplot as plt
import numpy as np
import random
from utils import find_reverse, random_choose, parse_response, strip_end
from my_llm import chat_my, visualize_messages, get_chat_completion_my
import fire


def main(
    model_ckpt: str,
    save_name: str,
    setting: str = 'default',
    if_visualize: bool = True,
):
    with open("tool_metadata/API_descriptions.json", "r", encoding='utf-8') as f:
        API_descriptions = json.load(f)

    if setting == 'default':
        with open("../llama-recipes/ft_datasets/tool_test.json", "r", encoding='utf-8') as f:
            dataset = json.load(f)
    elif setting == 'ICL':
        with open("../llama-recipes/ft_datasets/tool_test_with_demo.json", "r", encoding='utf-8') as f:
            dataset = json.load(f)
    else:
        assert False

    API_name_list = list(dataset.keys())
    random.seed(0)
    random.shuffle(API_name_list)

    api_descriptions = "\n\n".join(["API_name: {}\nDescription: {}".format(API_name, API_descriptions[API_name]) for API_name in API_name_list])

    with open("prompts/prompt_template.txt", "r", encoding='utf-8') as f:
        prompt_template = f.read().strip()

    prompt = prompt_template.format(api_descriptions=api_descriptions, api_names="\n".join(API_name_list))

    for ground_truth_API_name in dataset:
        print("now testing ground_truth_API_name =", ground_truth_API_name)

        item_list = dataset[ground_truth_API_name]
        for i in range(len(item_list)):
            item = item_list[i]
            print(i, end=" ")

            messages = [
                {"role": "system", "content": "You are a helpful assistant."}
            ]

            if setting == 'default':
                prompt_ = prompt + "\n\nUser Query: " + item['query']
            else:
                demo_examples = item['demo']
                prompt_ = prompt + "\n\nBelow are some examples:\n\n" + \
                    "---\n".join(["User Query: {}\nAction: {}\nAction Input: {}\n".format(demo['query'], demo['action'], demo['action_input']) for demo in demo_examples]) + \
                    "Now it's your turn.\n\nUser Query: " + item['query']

            messages = chat_my(messages, prompt_, temp=1.0, stop="Observation:", visualize=if_visualize, max_tokens=256, model=model_ckpt)
            parsed_response = parse_response(messages[-1]['content'], API_name_list, api_descriptions)

            item['parsed_result'] = parsed_response

            item_list[i] = item
        dataset[ground_truth_API_name] = item_list

    with open("saved_results/{}.json".format(save_name), "w", encoding='utf-8') as f:
        json.dump(dataset, f)


if __name__ == '__main__':
    fire.Fire(main)
