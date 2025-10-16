import os
import gc
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
#import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser
from transformers import AutoModelForCausalLM, DataCollatorWithPadding
from trl import AutoModelForCausalLMWithValueHead, PPOConfig
import importlib
from trl.trainer.My_ppo_trainer import MyPPOTrainer
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from mod_utils.utils import get_clean_data, load_main_tokenizer, save_configs, print_trainable_parameters, \
                  merge_weights_with_preference, Instructions, Instructions_summary, build_dataset_eval, build_dataset_summary_eval, build_dataset_beaver
#from mod_utils.util_decode import LogitsFusionModel
from mod_utils.multi_reward_models import RewardModels
#from peft import LoraConfig

import sys
sys.path.append("yourpath/HoE/workspace/MyMoLA/src/")
sys.path.append("yourpath/HoE/workspace/MyMoLA/mod_utils/")

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

from transformers import GenerationConfig, LlamaTokenizer, AutoConfig
from src.mola_modeling_llama_hacked import LlamaForCausalLM_d
from utils.prompter import Prompter

os.environ["WANDB_DISABLED"] = "true"

import random

seed = 10
random.seed(seed)
torch.manual_seed(0)

import argparse
from src.mola_peft_model_hacked import PeftModel
from src.mola_modeling_llama_hacked import LlamaForCausalLM_d
from mod_utils.My_util_decode import ConditionedMOEModel

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

os.environ["WANDB_DISABLED"] = "true"
tqdm.pandas()

# define paths for two datasets
#hhrlhf_dataset_path = 'Anthropic/hh-rlhf'
hhrlhf_dataset_path = 'yourpath/HoE/dataset/hh-rlhf'
summary_dataset_path = 'yourpath/HoE/dataset/summarize_from_feedback/comparisons'
beaver_dataset_path = "yourpath/HoE/dataset/PKU-SafeRLHF-10K"
tokenizer_path = "yourpath/model/llama-2-7b-hf"

@dataclass
class ScriptArguments:
    log_with: Optional[str] = field(default='wandb', metadata={"help": "use 'wandb' to log with wandb"})
    disable_wandb: Optional[str] = field(default=True, metadata={'help': 'Whether to disable wandb or not.'})
    save_directory: Optional[str] = field(default='./logs_Mymorlhf/')
    epochs: Optional[int] = field(default=3, metadata={'help': "Number of training epoches"})
    learning_rate: Optional[float] = field(default=5e-6, metadata={"help": "the learning rate"})
    mini_batch_size: Optional[int] = field(default=4, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=20, metadata={"help": "the batch size64"})
    gradient_accumulation_steps: Optional[int] = field(default=1, metadata={"help": "the number of gradient accumulation steps"})
    early_stopping: Optional[bool] = field(default=True, metadata={"help": "whether to early stop"})
    target: Optional[float] = field(default=3, metadata={"help": "target kl divergence of adaptive control"})
    init_kl_coef: Optional[float] = field(default=0.2,metadata={"help": "0.05 Initial KL penalty coefficient (used for adaptive and linear control)"},)
    kl_penalty: Optional[str] = field(default='abs', metadata={"help": "kl, abs, mse, full"})
    max_grad_norm: Optional[float] = field(default=0.5, metadata={"help": "Maximum gradient norm for gradient clipping"})
    load_in_8bit: Optional[bool] = field(default=True, metadata={"help": "loading model in 8 bit or bfloat16"})
    preference: Optional[float] = field(default=0.5, metadata={"help": "the weight for reward 1"})
    wandb_name: Optional[str] = field(default='Mymorlhf_llamma2_klreg0.2', metadata={"help": "Name for this experiment"})
    #base_model_name: Optional[str] = field(default='./merged_sft_summary', metadata={'help':"the path to the sft model; need to merge if using lora"})
    reward_names:Optional[str] = field(default='harmless,helpful,humor') 
    exp_type: Optional[str] = field(default='assistant', metadata={"help": "exp type, 'summary' or 'assistant' "})
    base_sft_model_path: Optional[str]=field(default='')
    base_moe_model_path: Optional[str]=field(default='')


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
exp_type = script_args.exp_type
base_sft_model_name = script_args.base_sft_model_path
base_moe_model_name = script_args.base_moe_model_path
tokenier_name = tokenizer_path
print('base model: ', base_sft_model_name)

if script_args.disable_wandb: # if you don't need the wandb log
    os.environ['WANDB_DISABLED'] = 'true' 

reward_names = [x.strip() for x in script_args.reward_names.split(',')]
num_rewards = len(reward_names)
print('number of rewards: {}'.format(num_rewards))
#######################################################################
reward_path_tokenizer_dict = {
    'harmless': ['yourpath/HoE/model/gpt2-large-harmless-reward_model'],
    'helpful': ['yourpath/HoE/model/gpt2-large-helpful-reward_model'],
    'deberta': ['yourpath/HoE/model/reward-model-deberta-v3-large-v2'],
    'summary': ['yourpath/HoE/model/gpt2_reward_summarization'],
    'faithful':['yourpath/HoE/model/bart-faithful-summary-detector'],
    'humor': ['yourpath/HoE/model/humor-no-humor'],
    'cost': ["yourpath/HoE/model/beaver-7b-v1.0-cost"],
    'reward': ["yourpath/HoE/model/beaver-7b-v1.0-reward"],
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


config = PPOConfig(
    model_name=base_sft_model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    mini_batch_size=script_args.mini_batch_size,
    batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    early_stopping=script_args.early_stopping,
    target=script_args.target,
    max_grad_norm=script_args.max_grad_norm,
    optimize_cuda_cache=True,
    init_kl_coef=script_args.init_kl_coef,
    kl_penalty=script_args.kl_penalty,
    tracker_project_name='Mymorlhf',
    tracker_kwargs={"wandb":{"name":script_args.wandb_name}},
    ppo_epochs=3,
)

accelerator = Accelerator()
process_id = Accelerator().local_process_index 
gpu_id = process_id
print('process: {}, model gpu id: {}'.format(process_id, gpu_id))


############## load reward models
reward_model = RewardModels(reward_model_path_list, rm_tokenizer_path_list, gpu_id)

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

current_device = Accelerator().local_process_index
print(current_device)

lora_config = LoraConfig(
    r=64, 
    lora_alpha=128, 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

tokenizer = load_main_tokenizer(tokenier_name)
#dataset_name = "yourpath/HoE/dataset/assistant"
dataset_name = None
from datasets import load_from_disk, Dataset, disable_caching
disable_caching()
#config.HF_DATASETS_CACHE = 'yourpath/HoE/.cache'
if dataset_name is None:
    if exp_type == 'assistant':
        dataset = build_dataset_eval(hhrlhf_dataset_path, tokenizer, reward_model.rm_tokenizers, split='test')
        instructions = Instructions()
    elif exp_type == 'summary':
        dataset = build_dataset_summary_eval(summary_dataset_path, tokenizer, reward_model.rm_tokenizers, split='test')
        instructions = Instructions_summary()
    #dataset.save_to_disk("yourpath/HoE/dataset/assistant80k-160k")
    else:
        dataset = build_dataset_beaver(beaver_dataset_path, tokenizer, reward_model.rm_tokenizers[0], split='test')
        instructions = Instructions()
else: 
    dataset = load_from_disk(dataset_name)
    if exp_type == 'assistant':
        instructions = Instructions()
    else:
        instructions = Instructions_summary()
#train_dataset = dataset.shuffle()
#dataset = dataset.select(range(script_args.batch_size*6))
print(f"Size of the train set: {len(dataset)}")



base_model = base_sft_model_name
mola_weights = base_moe_model_name
#lora_target_modules = "q_proj,v_proj"
lora_target_modules = "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
number_experts = "2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
top_k = "2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
#number_experts = "3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3"
#top_k = "3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3"

lora_target_modules = lora_target_modules.split(",")
lora_target_modules = [str(lr) for lr in lora_target_modules]
number_experts = number_experts.split(",")
number_experts = [int(lr) for lr in number_experts]
top_k = top_k.split(",")
top_k = [int(lr) for lr in top_k]

load_8bit = False

tokenizer = LlamaTokenizer.from_pretrained(base_model, padding_side='left')
model_config = AutoConfig.from_pretrained(base_model)
model_config.lora_target_modules = lora_target_modules
model_config.use_cache = True
if device == "cuda":
    model = LlamaForCausalLM_d.from_pretrained(
        base_model,
        config=model_config,
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
    model = LlamaForCausalLM_d.from_pretrained(
        base_model, config=model_config, device_map={"": device}, low_cpu_mem_usage=True
    )
    model = ConditionedMOEModel.from_pretrained(
        model,
        mola_weights,
        device_map={"": device},
    )
obalance = False
model.get_new_parameters(number_experts, top_k, obalance)

from peft import prepare_model_for_int8_training
if not load_8bit:
    model = prepare_model_for_int8_training(model)  

from mod_utils.My_util_decode import ConditionedMOEModelWithValueHead
#helpful_v_head = "yourpath/HoE/workspace/MOD/experiment-PPO/ppo/logs_Mymorlhf_train5_posttrain/train3/epoch_0_batch_10/v_head1.pt"
#harmless_v_head = "yourpath/HoE/workspace/MOD/experiment-PPO/ppo/logs_Mymorlhf_train5_posttrain/train3/epoch_0_batch_10/v_head0.pt"
#model = ConditionedMOEModelWithValueHead(model=model, v_heads_init=[helpful_v_head, harmless_v_head])
model = ConditionedMOEModelWithValueHead(model=model)

model.train(True)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)


ppo_trainer = MyPPOTrainer(
    config, model, ref_model=model, tokenizer=tokenizer, dataset=dataset, data_collator=collator, optimizer=optimizer
)


generation_kwargs = {
        "max_new_tokens": 48 if exp_type == 'summary' else 128, 
        "min_length": -1,
        "do_sample": False,
        "num_beams": 1
    }

print("Testing........")
model.train(False)
model.eval()
model.model.base_model.config.use_cache = True
model.model.config.use_cache = True
epochs = script_args.epochs
import gc

M = 20
sample_num = M+1
beg = 0.0
end = 1.0
preferences = np.zeros((M+1, 2))
preferences[:, 1] = np.arange(beg, end+(end-beg)/M,(end-beg)/M)
preferences[:, 0] = 1 - preferences[:, 1]
preferences = np.round(preferences, 1)

for preference in preferences:
    print(preference)

for preference in preferences:
    print(preference)

    full_responses = []
    full_queries = []
    model.dynamic_weights.set_dynamic_weights(torch.tensor(preference).unsqueeze(0))
    
    pbar = tqdm(total=(len(dataset) // script_args.batch_size // accelerator.num_processes))
    for i, batch in enumerate(ppo_trainer.dataloader):
        query_tensors = batch["input_ids"]
        weights = [torch.tensor(preference)]*script_args.batch_size
        
        with torch.no_grad():
            response_tensors = ppo_trainer.generate(query_tensors, weights, batch_size=script_args.batch_size, return_prompt=False, **generation_kwargs)
        
        full_responses.extend(tokenizer.batch_decode(response_tensors))
        full_queries.extend(tokenizer.batch_decode(query_tensors))
        pbar.update(1)
    #full_responses = tokenizer.batch_decode(response_tensors)

    full_responses_clean = []
    for _, response in enumerate(full_responses):
        response = response.strip('[PAD] ')
        response = response.strip('<unk>')
        temp_resp = response.strip('<s>').strip('</s>')
        temp_resp = temp_resp.split('\n\nHuman:')[0].strip()
        temp_resp = temp_resp.split('\nHuman:')[0].strip()
        temp_resp = temp_resp.split('\n\nAssistant:')[0].strip()
        temp_resp = temp_resp.split('\nAssistant:')[0].strip()
        temp_resp = temp_resp.split('\n\n\n')[0].strip()
        temp_resp = temp_resp.split('Instruction:')[0].strip()
        temp_resp = temp_resp.split('// end of')[0].strip()
        temp_resp = temp_resp.split('// End of')[0].strip()
        temp_resp = temp_resp.split('###')[0].strip()
        full_responses_clean.append(temp_resp)

    # Compute score
    accelerator.wait_for_everyone()
    gc.collect()
    torch.cuda.empty_cache()

    texts_merge = [q + r for q, r in zip(full_queries, full_responses_clean)]
    queries_responses = [
        (instructions.get_input(text), instructions.get_response(text))
        for text in texts_merge
    ]

    if hasattr(instructions, 'get_post'):
        rewards_list = reward_model.get_reward_model_scores(queries_responses, instructions.get_post)
    else:
        rewards_list = reward_model.get_reward_model_scores(queries_responses)
    
    accelerator.wait_for_everyone()
    gc.collect()
    torch.cuda.empty_cache()

    all_rewards = []
    for i in range(num_rewards):
        all_rewards.append(accelerator.gather_for_metrics(rewards_list[i]))
    all_full_prompts = accelerator.gather_for_metrics(full_queries)
    all_full_responses = accelerator.gather_for_metrics(full_responses_clean)

    if process_id == 0:
        evaluation_result = {
            'prompt': all_full_prompts,
            'response': all_full_responses,
        }
        for i in range(num_rewards):
            evaluation_result['obtained_score{}'.format(i+1)] = all_rewards[i]
            print('total average obtained score {}: {}'.format(i+1, np.mean(evaluation_result['obtained_score{}'.format(i+1)])))

        dataframe = pd.DataFrame(evaluation_result)
        if len(preference) == 2:
            dataframe.to_csv(os.path.join(script_args.save_directory, script_args.wandb_name,'eval_data_pref{}_{}.csv'.format(preference[0], preference[1])))
        else:
            dataframe.to_csv(os.path.join(script_args.save_directory, script_args.wandb_name,'eval_data_pref{}_{}_{}.csv'.format(preference[0], preference[1], preference[2])))


    # wait for the main process
    accelerator.wait_for_everyone()
    gc.collect()
    torch.cuda.empty_cache()

        