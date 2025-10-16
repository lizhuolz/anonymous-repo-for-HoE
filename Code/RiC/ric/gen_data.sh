#CUDA_VISIBLE_DEVICES=4 
#accelerate launch prepare_dataset_with_rewards.py --reward_names 'harmless,humor' --exp_type 'assistant' --save_directory './datasets/train_harmhumor.hf'
#accelerate launch prepare_dataset_with_rewards.py --reward_names 'helpful,humor' --exp_type 'assistant' --save_directory './datasets/train_helphumor.hf'
#accelerate launch prepare_dataset_with_rewards.py --reward_names 'summary,faithful' --exp_type 'summary' --save_directory './datasets/train_sumryfaith.hf'
#accelerate launch prepare_dataset_with_rewards.py --reward_names 'summary,deberta' --exp_type 'summary' --save_directory './datasets/train_sumrydebta.hf'
#accelerate launch prepare_dataset_with_rewards.py --reward_names 'deberta,faithful' --exp_type 'summary' --save_directory './datasets/train_debtafaith.hf'
accelerate launch --main_process_port 39502 prepare_dataset_with_rewards.py --reward_names 'reward,cost' --exp_type 'beaver' --save_directory './datasets/train_rewardcost.hf'