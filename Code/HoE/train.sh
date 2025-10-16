base_moe_model_path=yourpath/HoE/moe_model/humor,harmless
reward_names="helpful,humor"
exp_type=assistant
#CUDA_VISIBLE_DEVICES
accelerate launch --main_process_port 29503 Mymorlhf_ref.py --init_kl_coef 1.5 --reward_names $reward_names --exp_type ${exp_type} --wandb_name $reward_names --base_sft_model_path yourpath/HoE/model/assistant_sft_llama --base_moe_model_path $base_moe_model_path --save_directory ./logs_train_moev3_4