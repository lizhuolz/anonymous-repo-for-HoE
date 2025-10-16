
save_directory=./logs_ppo_assistant
#save_directory=./logs_ppo_summary
wandb_name=eval_3.2_1B
#reward_names="reward,cost"
#reward_names="harmless,helpful,humor"
reward_names="harmless,helpful"
#reward_names="summary,deberta,faithful"
#reward_names="steer_5reward"
#exp_type=steer2
#exp_type=psoups
exp_type=assistant
#exp_type=steer

#base_model_name=/yourpath/public/model
#base_model_name=yourpath/model/Meta-Llama-3-8B
#base_model_name= yourpath/workspace/RiC/RiC/ppo/logs_ppo_assistant/train4moe_helpful_princeton/epoch_2_batch_30
#base_model_name= yourpath/workspace/pcbmerging/helpful-sft_llama_to_llama-2-7b-chat-hf
#CUDA_VISIBLE_DEVICES=0,1,2,3 

#tokenizer_name=/yourpath/public/model
tokenizer_name=/yourpath/ /model/assistant_sft_llama
#tokenizer_name=yourpath/model/opt-1.3b

#accelerate launch --main_process_port 29505 eval_ppo_single_model.py --save_directory $save_directory --wandb_name $wandb_name --reward_names $reward_names --exp_type $exp_type --base_model_name $base_model_name



base_model_name=/yourpath/ /model/assistant_sft_llama
base_model_name=yourpath/model/opt-1.3b
base_model_name= yourpath/workspace/RiC/RiC/mydpo/logs_dpo_assistant/train_helpful_0.2/checkpoint-10000
base_model_name=/data4/public/model/Llama-3.2-1B
base_model_name=/yourpath/public/model/opt-350m

tokenizer_name=/data4/public/model/Llama-3.2-1B
tokenizer_name=/yourpath/public/model/opt-350m

accelerate launch --main_process_port 29503 eval_ppo_single_model.py --batch_size 20 --tokenizer_name $tokenizer_name --save_directory $save_directory --wandb_name $wandb_name --reward_names $reward_names --exp_type $exp_type --base_model_name $base_model_name
