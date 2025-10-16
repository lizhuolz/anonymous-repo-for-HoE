reward_names="helpful,humor,harmless"
aspects_names="humor,harmless"
exp_type=assistant
batch_size=10
base_eval_data_path=yourpath/eval_data.csv
#from="llama3bp"
from="sft"

#CUDA_VISIBLE_DEVICES
accelerate launch --main_process_port 29502 ../eval_MetaAligner.py --batch_size $batch_size --aspects_names $aspects_names --reward_names $reward_names --exp_type ${exp_type} --base_eval_data_path $base_eval_data_path --wandb_name $aspects_names --save_directory ./eval_MetaAligner_${from}_${exp_type}



