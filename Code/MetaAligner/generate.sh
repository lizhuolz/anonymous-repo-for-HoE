base_model_name=/data1/public/models/Llama-2-7b-chat-hf
reward_names="helpful,humor,harmless"
exp_type=assistant
wandb_name="assistant"
batch_size=10
#CUDA_VISIBLE_DEVICES

accelerate launch --main_process_port 29502 ../eval_ppo_single_model.py --batch_size $batch_size --reward_names $reward_names --exp_type ${exp_type} --wandb_name $wandb_name --base_model_name $base_model_name --save_directory ./eval_llama_chat