import json
import os
import matplotlib.pyplot as plt
import numpy as np
import time
from ms_llm import chat_my, visualize_messages, get_chat_completion_my    # azure openai
from copy import deepcopy
import fire
import random


def main(
    directory: str = "results/",
    filter_model_ckpts: str = 'gpt-4-8k',
    paraphrase_model_ckpts: str = 'gpt-35-turbo-16k',
    target_num_train_per_API: int = 150,
    num_para_train_max: int = 6,
    save_file_name: str = 'tool_data_train.json',
    if_visualize: bool = True,
):
    with open(os.path.join(directory, "data_dict.json"), "r", encoding='utf-8') as f:
        data_dict = json.load(f)

    # filtering
    with open("prompts/prompt_filtering.txt", "r", encoding='utf-8') as f:
        prompt_filtering_template = f.read().strip()

    dataset = dict()
    for API_name in data_dict:
        data = data_dict[API_name]
        examples = []

        for session_id in range(len(data)):
            session = data[session_id]
            for i in range(len(session['item_list'])):
                item = session['item_list'][i]

                last_step = item['chains'][-1]
                if not last_step['finish']:
                    continue

                temp = []
                last_API_call_step = None
                chains_ = item['chains']
                for jj in range(len(chains_) - 2, -1, -1):
                    step = chains_[jj]
                    if step['parse_successful']:
                        temp.append("API name: {}\nAPI args (in json string format): {}\nExecution result: {}\n".format(
                            step['action'], step['action_input'], step['observation']))
                        last_API_call_step = step
                        break
                if len(temp) == 0:
                    continue

                prompt_criticize = prompt_filtering_template.format(
                    api_descriptions=last_step['api_descriptions'],
                    query=item['query'],
                    chains="----".join(temp),
                    final_ans=last_step['final_ans']
                )

                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                ]

                item['judgment'] = chat_my(messages, prompt_criticize, temp=1.0, stop="Thought:",
                                                visualize=if_visualize, max_tokens=512, model=filter_model_ckpts)[-1]['content']

                session['item_list'][i] = item

                if "No." in item['judgment']:
                    continue

                examples.append({
                    "query": item['query'],
                    "API_name_list": last_API_call_step['API_name_list'],
                    "api_descriptions": last_API_call_step['api_descriptions'],
                    "action": last_API_call_step['action'],
                    "action_input": last_API_call_step['action_input'],
                    "observation": last_API_call_step['observation'],
                    "final_ans": chains_[-1]['final_ans'],
                })

            data[session_id] = session
        data_dict[API_name] = data
        dataset[API_name] = examples

    # paraphrase
    dataset_paraphrased = dict()
    for key in dataset:
        examples = dataset[key]
        num_paraphrase = min(round(target_num_train_per_API/(len(examples)+0.001)) - 1, num_para_train_max)

        paraphrased_examples = []
        for example in examples:
            # expand it into a list of examples
            examples_l = [example]

            messages = [
                {"role": "system", "content": "You are a helpful assistant."}
            ]

            messages = chat_my(messages, "Below you will be given a user query. Try to paraphrase it in a different way while preserving its meaning. The query is:\n\n{}\n\nYour paraphrase of the query:".format(example['query']),
                               temp=1.0, stop="Thought:", visualize=if_visualize, max_tokens=256, model=paraphrase_model_ckpts)
            examples_l.append({"query": messages[-1]['content']})

            for _ in range(num_paraphrase - 1):
                messages = chat_my(messages, "Can you try to paraphrase it again in a new way? Avoid coming up with something too close to your previous ones. Your paraphrase:",
                               temp=1.0, stop="Thought:", visualize=if_visualize, max_tokens=256, model=paraphrase_model_ckpts)
                examples_l.append({"query": messages[-1]['content']})

            paraphrased_examples.append(examples_l)
            print("----------------------")
        print("=============")

        dataset_paraphrased[key] = paraphrased_examples

    tool_data_train = []

    for key in dataset_paraphrased:
        val = dataset_paraphrased[key]
        for examples_l in val:
            if len(examples_l) == 0:
                continue
            seed_example = examples_l[0]
            tool_data_train.append(seed_example)
            for i in range(1, len(examples_l)):
                temp = deepcopy(seed_example)
                temp['query'] = examples_l[i]['query']
                tool_data_train.append(temp)

    random.shuffle(tool_data_train)

    with open(os.path.join(directory, save_file_name), "w", encoding='utf-8') as f:
        json.dump(tool_data_train, f)


if __name__ == '__main__':
    fire.Fire(main)
