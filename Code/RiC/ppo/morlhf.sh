
#base_model_path= yourpath/workspace/RiC/RiC/ppo/logs_ppo_assistant/train_helpful1/batch_43 
#base_model_path= yourpath/model/summary_sft_llama
base_model_path= yourpath/model/summary_sft_llama2
#base_model_path= yourpath/model/helpful_llama
#reward_name="summary"
reward_name="deberta,faithful"
#exp_type=assistant
exp_type=summary

#CUDA_VISIBLE_DEVICES=5,6,7
accelerate launch --main_process_port 29502 morlhf.py --preference 0.5 --reward_name $reward_name --base_model_name $base_model_path  --exp_type $exp_type --wandb_name train4moe_defa