import openai
from termcolor import colored
from copy import deepcopy
import time

with open("../../api_key.txt", "r") as f:
    openai.api_key = f.read().strip()


def get_chat_completion_my(messages, model='gpt-3.5-turbo', max_tokens=512, temp=0, n=1,
                        stop="EndOfResponse", return_raw=False):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temp,
        max_tokens=max_tokens,
        n=n,
        stop=stop
    )
    if return_raw:
        return response
    if n == 1:
        return response['choices'][0]['message']['content'].strip()

    return [response['choices'][i]['message']['content'].strip() for i in range(n)]


def visualize_messages(messages):
    role2color = {'system': 'red', 'assistant': 'green', 'user': 'cyan'}
    for entry in messages:
        assert entry['role'] in role2color.keys()
        if entry['content'].strip() != "":
            print(colored(entry['content'], role2color[entry['role']]))
        else:
            print(colored("<no content>", role2color[entry['role']]))


def chat_my(messages, new_message, visualize=True, **params):
    messages = deepcopy(messages)
    messages.append({"role": "user", "content": new_message})
    response = get_chat_completion_my(messages, **params)
    messages.append({"role": "assistant", "content": response})
    if visualize:
        visualize_messages(messages[-2:])
    return messages
