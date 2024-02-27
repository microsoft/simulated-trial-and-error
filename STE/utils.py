import numpy as np
import json
from difflib import get_close_matches


def find_reverse(str_a, ch):
    assert type(str_a) == type(ch) == str
    for i in range(len(str_a) - 1, -1, -1):
        if str_a[i] == ch:
            return i
    return -1


def random_choose(l, num):
    if len(l) <= num:
        return l
    inds = np.random.choice(len(l), num, replace=False).tolist()
    return [l[i] for i in inds]


def strip_end(a, b):
    while a.endswith(b):
        a = a[:len(a) - len(b)]
    return a


def parse_response(response, API_name_list, api_descriptions,
                   proc_thought=False, proc_toolken=False, check_API_name=True, ground_API=False):
    item = dict()
    item['API_name_list'] = API_name_list
    item['api_descriptions'] = api_descriptions

    item['parse_successful'] = True

    if "Final Answer:" in response:
        temp = response.split("Final Answer:")
        response, final_ans = temp[0].strip(), temp[1].strip()
        if "Action Input:" not in response:
            item['final_ans'] = final_ans
            item['finish'] = True
            return item

    item['finish'] = False
    if "Action Input:" not in response:
        item['parse_successful'] = False
        item[
            'parse_error_msg'] = "If you have already got enough information for the final answer, say \"Final Answer:\" followed by your answer. Otherwise, please specify your API call via \"Action:\" and API arguments via \"Action Input:\" followed by a json string. If there are no arguments, use \"Action Input: {}\". Do NOT start your response with \"Observation:\"; there is no need to repeat it."
        return item

    if response.count("Action Input:") > 1:
        item['parse_successful'] = False
        item['parse_error_msg'] = "Please use only one \"Action Input:\" in your response."
        return item

    action, action_input = response.split("Action Input:")
    action, action_input = strip_end(action.strip(), "\\n").strip(), strip_end(action_input.strip(), "\\n").strip()

    # get action
    if "Action:" not in action:
        item['parse_successful'] = False
        item[
            'parse_error_msg'] = "Please specify the API name you would like to call via \"Action:\" followed by the name. Remember that you should only call one API at a time, and the API name should be one of the following: {}. If you have already got the final answer, say \"Final Answer:\" followed by your final answer.".format(
            ", ".join(API_name_list))
        return item

    if action.count("Action:") > 1:
        item['parse_successful'] = False
        item['parse_error_msg'] = "Please use only one \"Action:\" in your response."
        return item

    thought, action = action.split("Action:")
    thought, action = strip_end(thought.strip(), "\\n").strip(), strip_end(action.strip(), "\\n").strip()

    if proc_toolken:
        action = action.replace("<tool_", "").strip("<>")

    if check_API_name and (action not in API_name_list):

        if ground_API:
            # find the closest API that is supported
            action = get_close_matches(action, API_name_list, n=1, cutoff=0.001)[0]
        else:
            item['parse_successful'] = False
            item['parse_error_msg'] = "Please only use exactly one of the following APIs: {}.".format(
                ", ".join(API_name_list))
            return item

    if proc_thought:
        if "Thought:" not in thought:
            item['parse_successful'] = False
            item['parse_error_msg'] = "Your thought should begin with \"Thought:\"."
            return item

        if thought.count("Thought:") > 1:
            item['parse_successful'] = False
            item['parse_error_msg'] = "Please use only one \"Thought:\" in your response."
            return item

        thought = thought.split("Thought:")[-1].strip()

    # get action input
    left_bracket_pos = action_input.find('{')
    if left_bracket_pos == -1:
        item['parse_successful'] = False
        item['parse_error_msg'] = "the Action Input is in json string format, and should begin with \"{\""
        return item
    right_bracket_pos = find_reverse(action_input, '}')
    if right_bracket_pos == -1:
        item['parse_successful'] = False
        item[
            'parse_error_msg'] = "the Action Input is in json string format, and should end with \"}\". Do NOT say anything else after \"}\""
        return item

    if left_bracket_pos >= right_bracket_pos:
        item['parse_successful'] = False
        item['parse_error_msg'] = "Your action input cannot be parsed as a json string. Please try again."
        return item

    # keep only within {}
    action_input = action_input[left_bracket_pos: right_bracket_pos + 1]
    action_input = "{" + action_input.strip("{}") + "}"

    if action_input.startswith("{{"):
        item['parse_successful'] = False
        item[
            'parse_error_msg'] = "the Action Input is in json string format, and should begin with only one \"{\", not two or more."
        return item
    if action_input.endswith("}}"):
        item['parse_successful'] = False
        item[
            'parse_error_msg'] = "the Action Input is in json string format, and should end with only one \"}\". Do NOT say anything else after \"}\""
        return item

    action_input = action_input.strip()

    item['parse_successful'] = True
    if proc_thought:
        item['thought'] = thought
    item['action'] = action
    item['action_input'] = action_input
    return item


if __name__ == '__main__':
    print()

