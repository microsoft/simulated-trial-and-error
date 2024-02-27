import json
import numpy as np
import fire


def prepare_tool_data(tool_data_, ds_ratio=None, ds_seed=None, add_tool_response=False):

    with open("sysmsg_dir/sysmsg_tool.json") as f:
        system_msg_tool = json.load(f)

    if not (ds_ratio is None):
        assert type(ds_seed) == int
        np.random.seed(ds_seed)
        rand_inds = np.random.choice(len(tool_data_), round(ds_ratio*len(tool_data_)), replace=False).tolist()
        tool_data = [tool_data_[i] for i in rand_inds]
    else:
        tool_data = tool_data_

    return_l = []
    # add tool data
    for item in tool_data:

        # tool call
        response = "Action: {}\nAction Input: {}".format("<tool_{}>".format(item["action"]), item["action_input"])
        diag = [{"role": "system", "content": system_msg_tool}, {"role": "user", "content": item['query']}]
        return_l.append({
            "dialog_history": diag,
            "response": response,
        })

        if add_tool_response:
            # response generation
            temp = diag + [
                {"role": "assistant", 'content': response},
                {"role": "user", "content": "Observation: " + item['observation']}
            ]
            return_l.append({
                "dialog_history": temp,
                "response": "Final Answer: " + item['final_ans'],
            })
    
    return return_l


def main(
    tool_file: str,
    data_save_dir: str,
    batch_id: int = -1,      # id of the current training round. smaller ids will be history batches
    general_data_file: str = 'ft_datasets/flan_v2_2k.json',
    add_tool_response: bool = False,
    ds_ratio_past: float = 0.1,
    no_replay: bool = False,
    no_general: bool = False,
):
    if tool_file == 'empty':
        tool_data_batches = []
    else:
        with open(tool_file, "r") as f:
            tool_data_batches = json.load(f)
    
    if type(tool_data_batches) == dict:
        print(tool_data_batches.keys())
    elif type(tool_data_batches) == list:
        assert batch_id == -1
    else:
        assert False

    merged_data = []   # each item should have 'dialog_history', 'response'

    if type(tool_data_batches) == list:
        tool_data = tool_data_batches
        merged_data = merged_data + prepare_tool_data(tool_data, ds_ratio=None, ds_seed=None, add_tool_response=add_tool_response)
        print("adding batch -1 data (# {}), no downsampling".format(len(tool_data)))
    else:
        if no_replay:
            # current batch
            tool_data = tool_data_batches["batch_"+str(batch_id)]
            merged_data = merged_data + prepare_tool_data(tool_data, ds_ratio=None, ds_seed=None, add_tool_response=add_tool_response)
            print("adding batch {} data (# {}), no downsampling".format(batch_id, len(tool_data)))

        else:
            for i in range(batch_id, -1, -1):
                tool_data = tool_data_batches["batch_"+str(i)]
                if i == batch_id:
                    # main batch
                    merged_data = merged_data + prepare_tool_data(tool_data, ds_ratio=None, ds_seed=None, add_tool_response=add_tool_response)
                    print("adding batch {} data (# {}), no downsampling".format(i, len(tool_data)))

                else:
                    # replay
                    merged_data = merged_data + prepare_tool_data(tool_data, ds_ratio=ds_ratio_past, ds_seed=i, add_tool_response=add_tool_response)
                    print("adding batch {} data (# {}), ds ratio: {}".format(i, len(tool_data), ds_ratio_past))
        
    if not no_general:
        # add general pretraining data
        with open("sysmsg_dir/sysmsg_normal.json") as f:
            system_msg_no_tool = json.load(f)
        
        with open(general_data_file, "r", encoding='utf-8') as f:
            general_data = json.load(f)
            print("# samples for general data:", len(general_data))

        for item in general_data:
            merged_data.append({
                "dialog_history": [{"role": "system", "content": system_msg_no_tool}] + item['dialog_history'],
                "response": "Final Answer: " + item['response'],
            })

    print("total # of samples:", len(merged_data))
    with open(data_save_dir, "w", encoding='utf-8') as f:
        json.dump(merged_data, f)


if __name__ == '__main__':
    fire.Fire(main)
