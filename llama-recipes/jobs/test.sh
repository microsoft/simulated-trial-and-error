 CUDA_VISIBLE_DEVICES=0 python inference/inference_chat.py \
     --model_name <YOUR_MODEL_DIRECTORY> \
     --data_path ft_datasets/tool_test_OTR.json \
     --save_path <YOUR_SAVE_DIRECTORY> \
     --item_type dialog \
     --sys_msg_dir sys_msg_dir/sysmsg_tool.json \
     --quantization