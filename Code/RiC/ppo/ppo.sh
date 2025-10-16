
#base_model_path= yourpath/workspace/RiC/RiC/ppo/logs_ppo_assistant/train_helpful1/batch_43 
#base_model_path= yourpath/model/summary_sft_llama2
#base_model_path=yourpath/model/llama-2-7b-chat-hf
#base_model_path=/yourpath/ /model/summary_sft_llama2
base_model_path=/data4/public/model/Llama-3.2-1B-Instruct
#base_model_path=/yourpath/public/model/
#base_model_path= yourpath/workspace/RiC/RiC/ppo/logs_ppo_assistant/train4moe3_humor_epoch_0_batch_461
#base_model_path= yourpath/workspace/RiC/RiC/ppo/logs_ppo_summary/train4moe3_summary/epoch_0_batch_163
#base_model_path= yourpath/model/assistant_sft_llama
#checkpoint_name= yourpath/workspace/RiC/RiC/ppo/logs_ppo_assistant/train4moe_helpful_princeton/epoch_0_batch_60
reward_name="Armo"
#reward_name="humor"
exp_type=IF
#exp_type=summary
#export NCCL_P2P_DISABLE=1
#export NCCL_IB_DISABLE=1
#CUDA_VISIBLE_DEVICES=5,6,7

#accelerate launch  --num_processes 2 --main_process_port 29503 --config_file= yourpath/workspace/RiC/RiC/ppo/accelerate.yaml \
# ppo.py --reward_name $reward_name --base_model_name $base_model_path  --exp_type $exp_type --wandb_name test_deepspeed

#accelerate launch  --num_processes 2 --main_process_port 29503 \
accelerate launch  --num_processes 3 --main_process_port 29503 --config_file= yourpath/workspace/RiC/RiC/ppo/accelerate.yaml \
 ppo_template.py  --reward_name $reward_name --base_model_name $base_model_path  --exp_type $exp_type --wandb_name train_Armo_llama1B3.2Inst



#accelerate launch --main_process_port 29502 ppo.py --checkpoint_name $checkpoint_name --reward_name $reward_name --base_model_name $base_model_path  --exp_type $exp_type --wandb_name train_Armo_llama1B3.2Inst

#deepspeed ppo.py --deepspeed --reward_name $reward_name --base_model_name $base_model_path  --exp_type $exp_type --wandb_name test_deepspeed
