
base_model_path=yourpath/model/summary_sft_llama
base_rs_model_path=yourpath/model/model4rs/...
base_moe_model_path=yourpath/model/moe_model/...
reward_name="summary,deberta"
exp_type=summary
#CUDA_VISIBLE_DEVICES=5,6,7 
accelerate launch --main_process_port 29502 morlhf.py --preference 0.5 --targets_reward 0.70,2.20 --reward_name $reward_name --base_model_path $base_model_path --base_rs_model_path $base_rs_model_path --base_moe_model_path $base_moe_model_path --exp_type $exp_type --wandb_name train_sude
