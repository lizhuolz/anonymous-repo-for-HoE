train_dataset_path= yourpath/workspace/RiC/RiC/ric/datasets/train_rewardcost.hf
#train_dataset_path= yourpath/workspace/RiC/RiC/ric/datasets/train_sumryfaith.hf
exp_type='beaver'
reward_names='reward,cost'
#reward_names='summary,faithful'
#base_model_name=/yourpath/ /model/assistant_sft_llama
base_model_name=/yourpath/ /model/beaver_sft_llama
#base_model_name=/yourpath/ /model/summary_sft_llama2
#peft_name= yourpath/workspace/RiC/RiC/ric/logs_trl/harmhelp_offline20000_onlineiter1/model_iter1

#weighting=0.15,0.15,0.7
#CUDA_VISIBLE_DEVICES=0,1,2,3 
accelerate launch --main_process_port 29502 main.py --train_dataset_path $train_dataset_path --exp_type $exp_type --reward_names $reward_names --training_steps 20000 --online_training_steps 4000 --num_online_iterations 2 --wandb_name 'rewardcost_onlineiter2_fullora' --batch_size 4 --gradient_accumulation_steps 1 --load_in_8bit True --base_model_name $base_model_name
#accelerate launch --main_process_port 29502 main.py --train_dataset_path $train_dataset_path --peft_name $peft_name --exp_type $exp_type --reward_names $reward_names --training_steps 0 --online_training_steps 4000 --num_online_iterations 1 --wandb_name 'help_onlineiter2' --batch_size 4 --gradient_accumulation_steps 2 --load_in_8bit True --base_model_name $base_model_name
#accelerate launch --main_process_port 29502 main.py --train_dataset_path $train_dataset_path --weighting $weighting --peft_name $peft_name --exp_type $exp_type --reward_names $reward_names --training_steps 0 --online_training_steps 4000 --num_online_iterations 2 --wandb_name assistant_harmhelphumor_offline20000_onlineiter2_${weighting} --batch_size 4 --gradient_accumulation_steps 2 --load_in_8bit True --base_model_name $base_model_name