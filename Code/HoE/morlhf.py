import os
import time
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser
from trl import PPOConfig#, set_seed
import importlib
from trl.trainer.My_ppo_trainer import MyPPOTrainer 

import sys
sys.path.append("yourpath/HoE/workspace/MyMoLA/src/")
sys.path.append("yourpath/HoE/workspace/MyMoLA/mod_utils/")

import numpy as np
import pandas as pd
from mod_utils.utils import print_trainable_parameters, load_main_tokenizer, Instructions, Instructions_summary, \
                  build_dataset, build_dataset_summary, save_configs, build_dataset_beaver         
from mod_utils.multi_reward_models import RewardModels
tqdm.pandas()
import matplotlib.pyplot as plt

os.environ['HF_HOME'] = "yourpath/HoE/.cache"

from trl.trainer.MOppo_trainer import MOPPOTrainer 
from morl_utils.My_util_decode import CasualLMWithValueHeads
from mod_utils.My_util_decode import ConditionedMOEModel
############################################################
import os

import argparse
import random

import torch
from src.mola_peft_model_hacked import PeftModel
from transformers import GenerationConfig, LlamaTokenizer, AutoConfig
import sys
from src.mola_modeling_llama_hacked import LlamaForCausalLM_d

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

seed = 10
random.seed(seed)  # random seed
torch.manual_seed(0)
######################################################



# define paths for two datasets
hhrlhf_dataset_path = 'yourpath/HoE/dataset/hh-rlhf'
summary_dataset_path = 'yourpath/HoE/dataset/summarize_from_feedback/comparisons'
beaver_dataset_path = "yourpath/HoE/dataset/PKU-SafeRLHF-10K"

@dataclass
class ScriptArguments:
    log_with: Optional[str] = field(default='wandb', metadata={"help": "use 'wandb' to log with wandb"})
    disable_wandb: Optional[str] = field(default=True, metadata={'help': 'Whether to disable wandb or not.'})
    save_directory: Optional[str] = field(default='./logs_morlhf/')
    epochs: Optional[int] = field(default=4, metadata={'help': "Number of training epoches"})
    learning_rate: Optional[float] = field(default=1e-5, metadata={"help": "the learning rate"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=120, metadata={"help": "the batch size64"})
    gradient_accumulation_steps: Optional[int] = field(default=10, metadata={"help": "the number of gradient accumulation steps"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    target: Optional[float] = field(default=1, metadata={"help": "target kl divergence of adaptive control"})
    init_kl_coef: Optional[float] = field(default=0.4,metadata={"help": "0.05 Initial KL penalty coefficient (used for adaptive and linear control)"},)
    max_grad_norm: Optional[float] = field(default=0.5, metadata={"help": "Maximum gradient norm for gradient clipping"})
    load_in_8bit: Optional[bool] = field(default=True, metadata={"help": "loading model in 8 bit or bfloat16"})
    preference: Optional[float] = field(default=0.5, metadata={"help": "the weight for reward 1"})
    targets_reward: Optional[str] = field(default="1.75, 1.2")
    wandb_name: Optional[str] = field(default='morlhf_llamma2_klreg0.2', metadata={"help": "Name for this experiment"})
    reward_names:Optional[str] = field(default='harmless,helpful,humor') 
    exp_type: Optional[str] = field(default='assistant', metadata={"help": "exp type, 'summary' or 'assistant' "})
    base_rs_model_path: Optional[str]=field(default='')
    base_moe_model_path: Optional[str]=field(default='')
    base_model_path: Optional[str]=field(default='')

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
targets_reward = list(map(float, script_args.targets_reward.split(",")))
exp_type = script_args.exp_type
preference = [round(script_args.preference, 1), round(1 - script_args.preference, 1)]
script_args.wandb_name = script_args.wandb_name + '_pref{}_{}'.format(preference[0], preference[1])

tokenier_name = script_args.base_model_path
base_rs_model_name = script_args.base_rs_model_path
base_model_name = script_args.base_model_path
base_moe_model_name = script_args.base_moe_model_path

if script_args.disable_wandb: # if you don't need the wandb log
    os.environ['WANDB_DISABLED'] = 'true' 

reward_names = [x.strip() for x in script_args.reward_names.split(',')]
num_rewards = len(reward_names)
print('number of rewards: {}'.format(num_rewards))
#######################################################################
if num_rewards == 3:
    preference = [round(1 / num_rewards, 2) for _ in range(num_rewards)]
print('preference: {}'.format(preference))
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
    "base_sft_model_name": base_model_name,
    'base_moe_model_name': base_moe_model_name,
    'reward_peft_path1': reward_model_path_list[0],
    'reward_peft_path2': reward_model_path_list[1],
    'reward_peft_path3': reward_model_path_list[2] if num_rewards==3 else ".",
    'tokenier_name': tokenier_name,
    'target': str(targets_reward),
}
for i in range(len(reward_model_path_list)):
    save_info['reward_peft_path{}'.format(i+1)] = reward_model_path_list[i]
save_configs(save_info, os.path.join(script_args.save_directory, script_args.wandb_name))

config = PPOConfig(
    model_name=base_moe_model_name,
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
    tracker_project_name='morlhf',
    tracker_kwargs={"wandb":{"name":script_args.wandb_name}},
    horizon=8000
)
mo_config = {
    "num_rewards": num_rewards,
    "weightings": preference,
    "targets": torch.tensor(targets_reward),
    "horizon": 2000,
    #init_mo_coefs
}

accelerator = Accelerator()
process_id = Accelerator().local_process_index 
gpu_id = process_id
print('process: {}, model gpu id: {}'.format(process_id, gpu_id))


############## load reward models
#reward_model = RewardModels(reward_model_path_list, rm_tokenizer_path_list, gpu_id)
reward_model = RewardModels(reward_model_path_list, rm_tokenizer_path_list, gpu_id)
rm_tokenizer = AutoTokenizer.from_pretrained(rm_tokenizer_path_list[0])

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

#set_seed(8888)
current_device = Accelerator().local_process_index
print(current_device)


tokenizer = load_main_tokenizer(tokenier_name)
if exp_type == 'assistant':
    #dataset = build_dataset(hhrlhf_dataset_path, tokenizer, rm_tokenizer, split='train')
    from datasets import load_from_disk
    #dataset.save_to_disk("yourpath/HoE/dataset/assistant")
    dataset = load_from_disk("yourpath/HoE/dataset/assistant")
    instructions = Instructions()
elif exp_type == 'summary':
    dataset = build_dataset_summary(summary_dataset_path, tokenizer, rm_tokenizer, split='train')
    instructions = Instructions_summary()
else:
    dataset = build_dataset_beaver(beaver_dataset_path, tokenizer, rm_tokenizer, split='train')
    instructions = Instructions()
train_dataset = dataset.shuffle()
print(f"Size of the train set: {len(train_dataset)}")


#################################################################################
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

from transformers import AutoModelForCausalLM, LlamaForCausalLM
ref_model = AutoModelForCausalLM.from_pretrained(
    base_rs_model_name,
    torch_dtype=torch.float16,
    #load_in_8bit=True,
    #quantization_config=bnb_config,
    device_map='cuda',
)
ref_model.config.update({
    "use_cache": True,
    "pad_token_id": ref_model.config.eos_token_id 
})
ref_model.resize_token_embeddings(len(tokenizer))
#ref_model.to("cuda")
#################################################################################


lora_target_modules = "q_proj,v_proj"
lora_target_modules = "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
number_experts = "2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
top_k = "2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"


lora_target_modules = lora_target_modules.split(",")
lora_target_modules = [str(lr) for lr in lora_target_modules]
number_experts = number_experts.split(",")
number_experts = [int(lr) for lr in number_experts]
top_k = top_k.split(",")
top_k = [int(lr) for lr in top_k]

load_8bit = False

#tokenizer = LlamaTokenizer.from_pretrained(base_sft_model_name, padding_side='left')
model_config = AutoConfig.from_pretrained(base_model_name)
model_config.lora_target_modules = lora_target_modules
if device == "cuda":
    model = LlamaForCausalLM_d.from_pretrained(
        base_model_name,
        config=model_config,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model = ConditionedMOEModel.from_pretrained(
        model,
        base_moe_model_name,
        torch_dtype=torch.float16,
        number_experts=number_experts,
        top_k=top_k,
    )
else:
    model = LlamaForCausalLM_d.from_pretrained(
        base_model_name,
        config=model_config, 
        #device_map={"": device}, 
        low_cpu_mem_usage=True
    )
    model = ConditionedMOEModel.from_pretrained(
        model,
        base_moe_model_name,
        #device_map={"": device},
    )
obalance = False
model.get_new_parameters(number_experts, top_k, obalance)


print_trainable_parameters(model)
model.resize_token_embeddings(len(tokenizer))

from peft import prepare_model_for_int8_training
if not load_8bit:
    model = prepare_model_for_int8_training(model) 

v_head_init_dict = {
    "helpful": "yourpath/HoE/workspace/MOD/experiment-PPO/ppo/logs_Mymorlhf_train5_posttrain/train3/epoch_0_batch_10/v_head1.pt",
    "humor": "yourpath/HoE/workspace/MOD/experiment-PPO/ppo/logs_Mymorlhf_train5/train1/epoch_1_batch_28/v_head1.pt",
    "harmless": "yourpath/HoE/workspace/MOD/experiment-PPO/ppo/logs_Mymorlhf_train5_posttrain/train3/epoch_0_batch_10/v_head0.pt",
    "reward": "yourpath/HoE/model/v_head0_4.pt",
    "cost": "yourpath/HoE/model/v_head1_4.pt",
    "summary": "yourpath/HoE/model/summary_summary_v_head.pt", 
    "faithful": "yourpath/HoE/model/summary_faithful_v_head.pt", 
    "deberta": "yourpath/HoE/model/summary_deberta_v_head.pt", 
}
v_heads_init = []
for name in reward_names:
    v_heads_init.append(v_head_init_dict[name])
wrappedmodel = CasualLMWithValueHeads(model=model, num_rewards=num_rewards, v_heads_init=v_heads_init)
wrappedmodel.train(True)

########################################################################
########################################################################

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, wrappedmodel.parameters()), lr=config.learning_rate)

ppo_trainer = MOPPOTrainer(
    config, model=wrappedmodel, ref_model=ref_model,  mo_config=mo_config, tokenizer=tokenizer, dataset=dataset, data_collator=collator, optimizer=optimizer
)


from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from transformers.generation.logits_process import InfNanRemoveLogitsProcessor, MinLengthLogitsProcessor

logits_processor = LogitsProcessorList()
logits_processor.append(InfNanRemoveLogitsProcessor())
generation_kwargs = {
    "max_new_tokens": 48 if exp_type == 'summary' else 128,
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0, 
    "do_sample": True,
    "temperature": 0.7,
    "pad_token_id": tokenizer.eos_token_id,
    "begin_suppress_tokens": [tokenizer.eos_token_id] ,
    "logits_processor": logits_processor,
    "repetition_penalty": 1.2
}


print("Training........")
wrappedmodel.model.gradient_checkpointing_disable()
wrappedmodel.model.config.use_cache = True
epochs = script_args.epochs
mean_scores = []
std_scores = []
save_data = {
    'kl_mean': [],
    #'reward_mean': [],
    #'reward_std': [],
    'text_sample':[],
    #'batch_time':[],
    #'total_time':[],
    "reward0_mean": [],
    "reward1_mean": [],
    "reward0_std": [],
    "reward1_std": [],
    'val_error': [],
    'mo_coef': [],
}
t_start = time.time()
for epoch in range(epochs):
    pbar = tqdm(total=len(train_dataset) // script_args.batch_size // accelerator.num_processes)
    for i, batch in enumerate(ppo_trainer.dataloader):
        t_epoch_start = time.time()
        print('epoch {}, batch {}'.format(epoch, i))
        query_tensors = batch["input_ids"]

        wrappedmodel.model.gradient_checkpointing_disable()
        wrappedmodel.model.config.use_cache = True
            
        with torch.no_grad():
            response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False, batch_size=15, **generation_kwargs)

        full_responses = tokenizer.batch_decode(response_tensors)
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
            temp_resp = temp_resp.split('###')[0].strip()
            full_responses_clean.append(temp_resp)

        clean_texts = full_responses_clean
        clean_response_tensors = [tokenizer.encode(text) for text in clean_texts]
        
        lengths = [len(clean_response_tensors[j]) for j in range(len(clean_response_tensors))]
        response_tensors = [response_tensors[j][:np.max([lengths[j], 2])] for j in range(len(response_tensors))]
        batch['response'] = clean_texts

        # Compute score
        texts_merge = [q + r for q, r in zip(batch['query'], batch['response'])]
        queries_responses = [
            (instructions.get_input(text), instructions.get_response(text))
            for text in texts_merge
        ]
        if hasattr(instructions, 'get_post'):
            rewards_list = reward_model.get_reward_model_scores(queries_responses, instructions.get_post)
        else:
            rewards_list = reward_model.get_reward_model_scores(queries_responses)
        rewards = []
        for j in range(len(queries_responses)):
            rewards.append(np.round( np.array([rewards_list[k][j] for k in range(num_rewards)]), 2))
        rewards_tensor = [torch.tensor(r).to(gpu_id) for r in rewards]
        print("iter {}, batch {}, mean score: {}".format(epoch, i, torch.mean(torch.tensor(rewards)).item()))

        wrappedmodel.model.gradient_checkpointing_enable()
        wrappedmodel.model.config.use_cache = False
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards_tensor)
        policy_kl = [stats["objective/kl"]]
        val_error = [stats["ppo/val/error"]]
        mo_coef = stats["My/mo_coef"]
        
        val_error = torch.stack([torch.tensor(_) for _ in val_error]).to('cuda')
        multi_rewards = torch.stack([torch.tensor(_) for _ in rewards_list]).to('cuda')  #tensor(num_rewards, batch)
        multi_rewards = multi_rewards.T  #tensor(batch, num_rewards)
        #ppo_trainer.log_stats(stats, batch, rewards)

        all_rewards = accelerator.gather_for_metrics(rewards)
        all_multi_rewards = accelerator.gather_for_metrics(multi_rewards)   #tensor(batch, num_rewards)
        
        all_policy_kl = accelerator.gather_for_metrics(policy_kl)
        all_val_error = accelerator.gather_for_metrics(val_error)

        all_val_error = np.array(all_val_error.cpu().clone())
        all_multi_rewards = np.array(all_multi_rewards.cpu().clone())
        if process_id == 0:
            mean_multi_rewards = np.mean(all_multi_rewards, axis=0)   #array(num_rewards)
            std_multi_rewards = np.std(all_multi_rewards, axis=0)   #array(num_rewards)
            
            save_path = os.path.join(script_args.save_directory, script_args.wandb_name, 'scores.png')
            #plt.plot(mean_scores)
            #plt.fill_between(np.arange(len(mean_scores)), np.array(mean_scores)- np.array(std_scores), np.array(mean_scores) + np.array(std_scores), alpha=0.5)
            plt.savefig(save_path)
            t_epoch_end = time.time()
            #save_data['batch_time'].append(t_epoch_end - t_epoch_start)
            #save_data['total_time'].append(t_epoch_end - t_start)
            save_data['kl_mean'].append(np.mean(all_policy_kl))
            #save_data['reward_mean'] = mean_scores
            #ave_data['reward_std'] = std_scores
            save_data['text_sample'].append(texts_merge[0])
            

            for k in range(num_rewards):
                save_data[f'reward{k}_mean'].append(mean_multi_rewards[k]) 
                save_data[f'reward{k}_std'].append(std_multi_rewards[k]) 
            save_data['mo_coef'].append(np.array(mo_coef.cpu().clone()))
            save_data['val_error'].append(np.mean(all_val_error))


            #plt.figure(figsize=(6, 6))  
            x = np.array(save_data['reward0_mean'])
            y = np.array(save_data['reward1_mean'])
            plt.plot(x, y, marker='o', linestyle='-', color='b')  
            #plt.scatter(x, y, color='r')  
            plt.scatter(targets_reward[0], targets_reward[1], color='gold', marker='*', s=200)
            
            for m in range(0, len(x), 10):
                start_x = save_data['reward0_mean'][m].item()
                start_y = save_data['reward1_mean'][m].item()
                vector_x = save_data['mo_coef'][m][0].item()
                vector_y = save_data['mo_coef'][m][1].item() 
                plt.quiver(
                    start_x, start_y,  
                    vector_x, vector_y, 
                    angles='xy', scale_units='xy', scale=5, color='r', alpha=0.2, linestyle='--',
                )


            line_x = np.linspace(x[0].item(),  targets_reward[0]+0.5, 100)  
            line_y = preference[0] / preference[1] * (line_x - targets_reward[0]) + targets_reward[1] 
            plt.plot(line_x, line_y, linestyle='--', color='green', alpha=0.5) 


            plt.xlabel(reward_names[0])
            plt.ylabel(reward_names[1])
            plt.legend()
            plt.savefig(save_path)

            dataframe = pd.DataFrame(save_data)
            dataframe.to_csv(os.path.join(script_args.save_directory, script_args.wandb_name,'data.csv'))
            print("iter {}, batch {}: log finish".format(epoch, i))

        # wait for the main process
        accelerator.wait_for_everyone()
        pbar.update(1)

        # save model
        if ppo_trainer.accelerator.is_main_process and i % 1 == 0 and i != 0:
            save_path = os.path.join(script_args.save_directory, script_args.wandb_name, 'epoch_{}_batch_{}'.format(epoch, i))
            ppo_trainer.save_pretrained(save_path)
            print("iter {}, batch {}: model saved".format(epoch, i))
    
    # save model
    if ppo_trainer.accelerator.is_main_process:
        save_path = os.path.join(script_args.save_directory, script_args.wandb_name, 'epoch_{}_batch_{}'.format(epoch, i))
        ppo_trainer.save_pretrained(save_path)
        print("iter {}, batch {}: model saved".format(epoch, i))
        