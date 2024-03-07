python data_proc_format.py \
    --tool_file empty \
    --data_save_dir ft_datasets/merged_data_flan.json \
    --general_data_file ft_datasets/flan_v2_2k.json \
    --add_tool_response \

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes 1 --nproc_per_node 4 llama_finetuning.py \
    --enable_fsdp \
    --model_name <YOUR_MODEL_DIRECTORY> \
    --num_epochs 2 \
    --batch_size_training 16 \
    --micro_batch_size 1 \
    --val_batch_size 8 \
    --lr 2e-5 \
    --num_workers_dataloader 1 \
    --seed 42 \
    --data_path ft_datasets/merged_data_flan.json \
    --max_words_dataset 2048 \
    --checkpoint_folder <DIRECTORY_TO_SAVE> \
    --save_with_hf \
    --warmup_ratio 0.03 \
    --save_epoch_interval 1 \
    --add_token_list ft_datasets/toolken_list_50.json