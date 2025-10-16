
exp_type='beaver'
#reward_names='harmless,helpful,humor'
reward_names='reward,cost'
base_model_name=/yourpath/ /model/beaver_sft_llama
#base_model_name= yourpath/model/harmhelphumor_rs820_280_pref0.5_0.5
peft_name= yourpath/workspace/RiC/RiC/ric/logs_trl/rewardcost_onlineiter2_fullora/model_iter2
#CUDA_VISIBLE_DEVICES=0,1,2,3 
accelerate launch --main_process_port 29502 evaluation.py --peft_name $peft_name --reward_names $reward_names --exp_type $exp_type --wandb_name rewardcost_onlineiter2 --base_model_name $base_model_name
#accelerate launch --main_process_port 29503 evaluation.py  --reward_names $reward_names --exp_type $exp_type --wandb_name harmhelphumor_rs820_280_pref0.5_0.5 --base_model_name $base_model_name
