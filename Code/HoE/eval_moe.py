import os
import gc
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
#import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import HfArgumentParser
from transformers import AutoModelForCausalLM, DataCollatorWithPadding
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from mod_utils.utils import get_clean_data, load_main_tokenizer, save_configs, print_trainable_parameters, \
                  merge_weights_with_preference, Instructions, Instructions_summary, build_dataset_eval, build_dataset_summary_eval
#from mod_utils.util_decode import LogitsFusionModel
from mod_utils.multi_reward_models import RewardModels
#from peft import LoraConfig


import os
import sys
#sys.path.append("yourpath/HoE/workspace/MyMoLA/src")

import fire
import torch
import transformers
#from src.mola_trainer_hacked import Trainer
from datasets import load_dataset
from datasets import load_from_disk

from peft import (
    prepare_model_for_int8_training,
)
from src.mola_mapping_hacked import get_peft_model
from src.mola_lora_hacked import LoraConfig
from src.mola_peft_model_hacked import set_peft_model_state_dict_moe

from transformers import LlamaTokenizer, AutoConfig
from src.mola_modeling_llama_hacked import LlamaForCausalLM_d
from utils.prompter import Prompter

os.environ["WANDB_DISABLED"] = "true"

import random

seed = 10
random.seed(seed)
torch.manual_seed(0)



tqdm.pandas()

# define paths for two datasets
#hhrlhf_dataset_path = 'Anthropic/hh-rlhf'
hhrlhf_dataset_path = "yourpath/HoE/dataset/hh-rlhf"
summary_dataset_path = 'openai/summarize_from_feedback'
tokenizer_path = "yourpath/model/llama-2-7b-hf"


base_model = "yourpath/HoE/model/assistant_sft_llama"  # the only required argument
device_map = "auto"
data_path = "yourpath/HoE/dataset/hh-rlhf" # ../datasets/CL_biology_scienceq_train_all.hf
#base_model_list = ["yourpath/HoE/workspace/RiC/RiC/ppo/logs_ppo_assistant/train4moe_helpful/epoch_1_batch_554",
#                   "yourpath/HoE/workspace/RiC/RiC/ppo/logs_ppo_assistant/train4moe_harmless/epoch_0_batch_50"]
# training hyperparams
batch_size: int = 128,
cutoff_len: int = 256,
val_set_size: int = 2,
# lora hyperparams
lora_r = "64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64"
lora_alpha = 128
lora_dropout = 0.05
lora_target_modules = "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
lora_target_modules = "q_proj,v_proj"
# mola hyperparams
number_experts = "2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
top_k = "2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
# llm hyperparams
train_on_inputs: bool = True,  # if False, masks out inputs in loss
add_eos_token: bool = True,
group_by_length: bool = False,  # faster, but produces an odd training loss curve
# wandb params

resume_from_checkpoint: str = './step2_biology256r_8mbs_no8bit_scale10',  # either training checkpoint or final adapter
prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
obalance = False






@dataclass
class ScriptArguments:
    save_directory: Optional[str] = field(default='./logs_ours_summary_eval')
    mini_batch_size: Optional[int] = field(default=10, metadata={"help": "minibatch size for eval"})
    wandb_name: Optional[str] = field(default='eval_pposoups_klreg0.2_harmless_helpful', metadata={"help": "Name for this experiment"})
    reward_names:Optional[str] = field(default='harmless,helpful')
    base_sft_model_path: Optional[str]=field(default='')
    base_moe_model_path: Optional[str]=field(default='')
    exp_type: Optional[str] = field(default='assistant', metadata={"help": "exp type, 'summary' or 'assistant' "})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
exp_type = script_args.exp_type
base_sft_model_name = script_args.base_sft_model_path
base_moe_model_name = script_args.base_moe_model_path
tokenier_name = tokenizer_path

reward_names = [x.strip() for x in script_args.reward_names.split(',')]
reward_path_tokenizer_dict = {
    'harmless': ["yourpath/HoE/model/gpt2-large-harmless-reward_model"],
    'helpful': ["yourpath/HoE/model/gpt2-large-helpful-reward_model"],
    'deberta': ['OpenAssistant/reward-model-deberta-v3-large-v2'],
    'summary': ['Tristan/gpt2_reward_summarization'],
    'faithful':['CogComp/bart-faithful-summary-detector'],
    'humor': ['yourpath/HoE/model/humor-no-humor'],
}
reward_model_path_list = []
rm_tokenizer_path_list = []
for name in reward_names:
    if name not in reward_path_tokenizer_dict.keys():
        raise NotImplementedError
    reward_model_path_list.append(reward_path_tokenizer_dict[name][0])
    rm_tokenizer_path_list.append(reward_path_tokenizer_dict[name][0])
os.makedirs(os.path.join(script_args.save_directory, script_args.wandb_name), exist_ok=True)
save_info = {
    "base_sft_model_name": base_sft_model_name,
    'base_moe_model_name': base_moe_model_name,
    'reward_peft_path1': reward_model_path_list[0],
    'reward_peft_path2': reward_model_path_list[1],
    'reward_peft_path3': reward_model_path_list[2] if len(reward_model_path_list)==3 else ".",
    'tokenier_name': tokenier_name
}
for i in range(len(reward_model_path_list)):
    save_info['reward_peft_path{}'.format(i+1)] = reward_model_path_list[i]
save_configs(save_info, os.path.join(script_args.save_directory, script_args.wandb_name))


accelerator = Accelerator()
process_id = Accelerator().local_process_index 
gpu_id = process_id
print('process: {}, model gpu id: {}'.format(process_id, gpu_id))
reward_models = RewardModels(reward_model_path_list, rm_tokenizer_path_list, gpu_id) 


current_device = Accelerator().local_process_index
print(current_device)


tokenizer = load_main_tokenizer(tokenier_name)
if exp_type == 'assistant':
    valid_dataset = build_dataset_eval(hhrlhf_dataset_path, tokenizer, reward_models.rm_tokenizers, split='test') 
    instructions = Instructions()
else:
    valid_dataset = build_dataset_summary_eval(summary_dataset_path, tokenizer, reward_models.rm_tokenizers, split='test')
    instructions = Instructions_summary()
valid_batch_size = 1
#valid_dataset = valid_dataset.remove_columns(['prompt', 'query'])
for key in ['key', 'text', 'response']:
    if key in valid_dataset.column_names:
        valid_dataset = valid_dataset.remove_columns(key)
print(f"Size of the validation set: {len(valid_dataset)}")

def evaluate_model(model, tokenizer, valid_dataset):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    valid_data_loader = DataLoader(valid_dataset, batch_size=script_args.mini_batch_size, drop_last=True, collate_fn=data_collator)
    ### load_merged_model
    model.resize_token_embeddings(len(tokenizer))

    accelerator = Accelerator()
    model, valid_data_loader = accelerator.prepare(model, valid_data_loader)

    """
    generation_kwargs = {
        "max_new_tokens": 128 if exp_type == 'assistant' else 48, 
        "min_length": -1,
        "do_sample": False,
        "num_beams": 1
    }
    """
    generation_kwargs = {
        "max_new_tokens": 128 if exp_type == 'assistant' else 48,
        'min_length': -1, 
        "top_k": 0.0,
        "top_p": 1.0, 
        "do_sample": True,
        "temperature": 0.7,
    }
    full_responses = []
    full_prompts = []
    pbar = tqdm(total=len(valid_dataset) // script_args.mini_batch_size // accelerator.num_processes)
    with torch.no_grad():
        for i, batch in enumerate(valid_data_loader):
            response_tensors = accelerator.unwrap_model(model).generate(input_ids=batch["input_ids"], **generation_kwargs) #length_sampler=output_length_sampler, 
            full_responses.extend(response_tensors)
            full_prompts.extend(batch['input_ids'])
            pbar.update(1)
    
    full_responses = tokenizer.batch_decode(full_responses)
    full_prompts = tokenizer.batch_decode(full_prompts)
    # clean data
    full_prompts, full_responses = get_clean_data(full_responses, full_prompts)

    queries_responses = [
        (instructions.get_input(text), instructions.get_response(text))
        for text in full_responses
    ]
    if hasattr(instructions, 'get_post'):
        rewards_list = reward_models.get_reward_model_scores(queries_responses, instructions.get_post)
    else:
        rewards_list = reward_models.get_reward_model_scores(queries_responses)
    
    ### error here may because of old version of transformers/accelerate/peft
    all_rewards = []
    for i in range(reward_models.num_rewards):
        all_rewards.append(accelerator.gather_for_metrics(rewards_list[i]))
    all_full_prompts = accelerator.gather_for_metrics(full_prompts)
    all_full_responses = accelerator.gather_for_metrics(full_responses)
    return all_rewards, all_full_prompts, all_full_responses


print("Evaluating........")
tokenizer.padding_side = "left"
## preference list
if reward_models.num_rewards == 3:
    preferences = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.1, 0.1, 0.8],
        [0.1, 0.8, 0.1],
        [0.2, 0.2, 0.6],
        [0.2, 0.4, 0.4],
        [0.2, 0.6, 0.2],
        [0.33, 0.33, 0.33],
        [0.4, 0.4, 0.2],
        [0.4, 0.2, 0.4], 
        [0.6, 0.2, 0.2],
        [0.8, 0.1, 0.1], 
        [1.0, 0.0, 0.0], 
        ])
elif reward_models.num_rewards == 2:
    M = 5
    sample_num = M+1
    beg = -0.0
    end = 1.0
    preferences = np.zeros((M+1, 2))
    preferences[:, 0] = np.arange(beg, end+(end-beg)/M,(end-beg)/M)
    preferences[:, 1] = 1 - preferences[:, 0]
    preferences = np.round(preferences, 1)
else:
    raise NotImplementedError



#################################################

import os
import sys
sys.path.append("yourpath/HoE/workspace/MyMoLA/src/")
sys.path.append("yourpath/HoE/workspace/MyMoLA/mod_utils/")
import argparse
import random

import torch
from src.mola_peft_model_hacked import PeftModel
from transformers import GenerationConfig, LlamaTokenizer, AutoConfig
import sys
from src.mola_modeling_llama_hacked import LlamaForCausalLM_d
from mod_utils.My_util_decode import ConditionedMOEModel

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

seed = 10
random.seed(seed)  # random seed
torch.manual_seed(0)

base_model = base_sft_model_name
mola_weights = base_moe_model_name
lora_target_modules = "q_proj,v_proj"
number_experts = "2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
top_k = "2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
number_experts = "3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3"
#top_k = "3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3"


lora_target_modules = lora_target_modules.split(",")
lora_target_modules = [str(lr) for lr in lora_target_modules]
number_experts = number_experts.split(",")
number_experts = [int(lr) for lr in number_experts]
top_k = top_k.split(",")
top_k = [int(lr) for lr in top_k]

load_8bit = False

tokenizer = LlamaTokenizer.from_pretrained(base_model, padding_side='left')
config = AutoConfig.from_pretrained(base_model)
config.lora_target_modules = lora_target_modules
if device == "cuda":
    model = LlamaForCausalLM_d.from_pretrained(
        base_model,
        config=config,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model = ConditionedMOEModel.from_pretrained(
        model,
        mola_weights,
        torch_dtype=torch.float16,
        number_experts=number_experts,
        top_k=top_k,
    )
else:
    model = ConditionedMOEModel.from_pretrained(
        base_model, config=config, device_map={"": device}, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(
        model,
        mola_weights,
        device_map={"": device},
    )
obalance = False
model.get_new_parameters(number_experts, top_k, obalance)



from peft import prepare_model_for_int8_training
if not load_8bit:
    model = prepare_model_for_int8_training(model) 

model.train(False)
model.eval()



for k in range(0, len(preferences)):
    preference = preferences[k, :]
    model.dynamic_weights.set_dynamic_weights(torch.tensor(preference).unsqueeze(0))
    
    #sample_model = LogitsFusionModel(model=sft_model, weights=preference, f_type='reverse_kl', peft_config=lora_config)

    accelerator.wait_for_everyone()
    gc.collect()
    torch.cuda.empty_cache()
    print(k, preference)
    all_rewards, all_full_prompts, all_full_responses = evaluate_model(model, tokenizer, valid_dataset)
    gc.collect()
    torch.cuda.empty_cache()

    if process_id == 0:
        evaluation_result = {
            'prompt': all_full_prompts,
            'response': all_full_responses,
        }
        for i in range(reward_models.num_rewards):
            evaluation_result['obtained_score{}'.format(i+1)] = all_rewards[i]
            print('total average obtained score {}: {}'.format(i+1, np.mean(evaluation_result['obtained_score{}'.format(i+1)])))

        dataframe = pd.DataFrame(evaluation_result)
        if len(preference) == 2:
            dataframe.to_csv(os.path.join(script_args.save_directory, script_args.wandb_name,'eval_data_pref{}_{}.csv'.format(preference[0], preference[1])), escapechar='\\')
        else:
            dataframe.to_csv(os.path.join(script_args.save_directory, script_args.wandb_name,'eval_data_pref{}_{}_{}.csv'.format(preference[0], preference[1], preference[2])), escapechar='\\')




