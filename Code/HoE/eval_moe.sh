base_sft_model_path=yourpath/model/assistant_sft_llama
base_moe_model_path=yourpath/model/...
reward_names="helpful,harmless"
exp_type=assistant
#CUDA_VISIBLE_DEVICES
wandb_name="helpful3,harmless3"
accelerate launch --main_process_port 29504 eval_moe_copy.py --reward_names $reward_names --exp_type ${exp_type} --wandb_name $wandb_name --base_sft_model_path $base_sft_model_path --base_moe_model_path $base_moe_model_path --save_directory ./logs_MOEv2_${exp_type}_eval

