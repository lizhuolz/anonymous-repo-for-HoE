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
                  build_dataset, build_dataset_summary, save_configs          
from mod_utils.multi_reward_models import RewardModels
tqdm.pandas()
import matplotlib.pyplot as plt

os.environ['HF_HOME'] = "yourpath/HoE/.cache"

############################################################
import os

import argparse
import random

import torch
from src.mola_peft_model_hacked import PeftModel
from transformers import GenerationConfig, LlamaTokenizer, AutoConfig
import sys
from src.mola_modeling_llama_hacked import LlamaForCausalLM_d
from mod_utils.My_util_decode import ConditionedMOEModel, WeightSampler

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

seed = 10
random.seed(seed)  # random seed
torch.manual_seed(0)
######################################################



# define paths for two datasets
#hhrlhf_dataset_path = 'Anthropic/hh-rlhf'
hhrlhf_dataset_path = 'yourpath/HoE/dataset/hh-rlhf'
summary_dataset_path = 'yourpath/HoE/dataset/summarize_from_feedback/comparisons'

@dataclass
class ScriptArguments:
    log_with: Optional[str] = field(default='wandb', metadata={"help": "use 'wandb' to log with wandb"})
    disable_wandb: Optional[str] = field(default=True, metadata={'help': 'Whether to disable wandb or not.'})
    save_directory: Optional[str] = field(default='./logs_Mymorlhf/')
    epochs: Optional[int] = field(default=1, metadata={'help': "Number of training epoches"})
    learning_rate: Optional[float] = field(default=5e-6, metadata={"help": "the learning rate"})
    mini_batch_size: Optional[int] = field(default=2, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=128, metadata={"help": "the batch size64"})
    gradient_accumulation_steps: Optional[int] = field(default=8, metadata={"help": "the number of gradient accumulation steps"})
    early_stopping: Optional[bool] = field(default=True, metadata={"help": "whether to early stop"})
    target: Optional[float] = field(default=9, metadata={"help": "target kl divergence of adaptive control"})
    init_kl_coef: Optional[float] = field(default=0.2,metadata={"help": "0.05 Initial KL penalty coefficient (used for adaptive and linear control)"},)
    kl_penalty: Optional[str] = field(default='abs', metadata={"help": "kl, abs, mse, full"})
    max_grad_norm: Optional[float] = field(default=0.5, metadata={"help": "Maximum gradient norm for gradient clipping"})
    load_in_8bit: Optional[bool] = field(default=True, metadata={"help": "loading model in 8 bit or bfloat16"})
    #preference: Optional[float] = field(default=0.5, metadata={"help": "the weight for reward 1"})
    wandb_name: Optional[str] = field(default='Mymorlhf_llamma2_klreg0.2', metadata={"help": "Name for this experiment"})
    #base_model_name: Optional[str] = field(default='./merged_sft_summary', metadata={'help':"the path to the sft model; need to merge if using lora"})
    reward_names:Optional[str] = field(default='harmless,helpful,humor') 
    exp_type: Optional[str] = field(default='assistant', metadata={"help": "exp type, 'summary' or 'assistant' "})
    base_sft_model_path: Optional[str]=field(default='')
    base_moe_model_path: Optional[str]=field(default='')


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
exp_type = script_args.exp_type
#preference = [round(script_args.preference, 1), round(1 - script_args.preference, 1)]
#preference = [-0.4, 1.4]
#script_args.wandb_name = script_args.wandb_name + '_pref{}_{}'.format(preference[0], preference[1])

tokenier_name = script_args.base_sft_model_path
base_sft_model_name = script_args.base_sft_model_path
base_moe_model_name = script_args.base_moe_model_path
#sft_adapter_name = "yourpath/HoE/workspace/examples/task6/sft_llama/final_checkpoint"
print('base model: ', base_sft_model_name)

if script_args.disable_wandb: # if you don't need the wandb log
    os.environ['WANDB_DISABLED'] = 'true' 

reward_names = [x.strip() for x in script_args.reward_names.split(',')]
num_rewards = len(reward_names)
print('number of rewards: {}'.format(num_rewards))
#######################################################################
#######################################################################
reward_path_tokenizer_dict = {
    'harmless': ['yourpath/HoE/model/gpt2-large-harmless-reward_model'],
    'helpful': ['yourpath/HoE/model/gpt2-large-helpful-reward_model'],
    'deberta': ['yourpath/HoE/model/reward-model-deberta-v3-large-v2'],
    'summary': ['yourpath/HoE/model/gpt2_reward_summarization'],
    'faithful':['yourpath/HoE/model/bart-faithful-summary-detector'],
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
    'reward_peft_path3': reward_model_path_list[2] if num_rewards==3 else ".",
    'tokenier_name': tokenier_name
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
rm_tokenizer = AutoTokenizer.from_pretrained(rm_tokenizer_path_list[0])

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

#set_seed(8888)
current_device = Accelerator().local_process_index
print(current_device)


#################################################################################

tokenizer = load_main_tokenizer(tokenier_name)
dataset_name = "yourpath/HoE/dataset/assistant"
#dataset_name = None
from datasets import load_from_disk, Dataset, disable_caching
#disable_caching()
#config.HF_DATASETS_CACHE = 'yourpath/HoE/.cache'
if dataset_name is None:
    if exp_type == 'assistant':
        dataset = build_dataset(hhrlhf_dataset_path, tokenizer, rm_tokenizer, split='test')
        instructions = Instructions()
    else:
        dataset = build_dataset_summary(summary_dataset_path, tokenizer, rm_tokenizer, split='train', size=30000)
        instructions = Instructions_summary()
    #dataset.save_to_disk("yourpath/HoE/dataset/assistant80k-160k")
else: 
    dataset = load_from_disk(dataset_name)
    if exp_type == 'assistant':
        instructions = Instructions()
    else:
        instructions = Instructions_summary()
#train_dataset = dataset.shuffle()
print(f"Size of the train set: {len(dataset)}")


#################################################################################

from transformers import AutoModelForCausalLM
ref_model = AutoModelForCausalLM.from_pretrained(
    base_sft_model_name,
    torch_dtype=torch.float16,
    #device_map='cuda',
)
ref_model.config.update({
    "use_cache": True,
    "pad_token_id": ref_model.config.eos_token_id 
})

#################################################################################

lora_target_modules = "q_proj,v_proj"
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
model_config = AutoConfig.from_pretrained(base_sft_model_name)
model_config.lora_target_modules = lora_target_modules
if device == "cuda":
    model = LlamaForCausalLM_d.from_pretrained(
        base_sft_model_name,
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
        base_sft_model_name,
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




from mod_utils.My_util_decode import ConditionedMOEModelWithValueHead
helpful_v_head = "yourpath/HoE/workspace/MOD/experiment-PPO/ppo/logs_Mymorlhf_train5_posttrain/train3/epoch_0_batch_10/v_head1.pt"
harmless_v_head = "yourpath/HoE/workspace/MOD/experiment-PPO/ppo/logs_Mymorlhf_train5_posttrain/train3/epoch_0_batch_10/v_head0.pt"
moemodel = ConditionedMOEModelWithValueHead(model=model, v_heads_init=[helpful_v_head, harmless_v_head], trainable_module="all")

from peft import prepare_model_for_int8_training
if not load_8bit:
    moemodel.model = prepare_model_for_int8_training(moemodel.model) 


#print_trainable_parameters(model)
ref_model.resize_token_embeddings(len(tokenizer))
moemodel.model.resize_token_embeddings(len(tokenizer))
moemodel.train()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, moemodel.parameters()), lr=config.learning_rate)
print("optimizer")
moemodel.test_grad()


ppo_trainer = MyPPOTrainer(
    config, moemodel, ref_model=ref_model, tokenizer=tokenizer, dataset=dataset, data_collator=collator, optimizer=optimizer
)
weight_sampler = WeightSampler(num_rewards, uniform_ratio=1.0)

generation_kwargs = {
        "max_new_tokens": 128 if exp_type == 'assistant' else 48, 
        "min_length": -1,
        "do_sample": False,
        "num_beams": 1,
        "repetition_penalty": 1.2
    }


print("Training........")
moemodel.test_grad()
#model.gradient_checkpointing_disable()
#model.pretrained_model.config.use_cache = True
epochs = script_args.epochs
mean_scores = []
mean_score0 = []
mean_score1 = []
std_scores = []
save_data = {
    'kl_mean': [],
    'reward_mean': [],
    'reward_std': [],
    'batch_time':[],
    #'total_time':[],
    'text_sample':[],
    'weights': [],
    #'reward0_mean': [], 
    #'reward1_mean': [],
    #'gen_time': [],
    #'original_text_sample': [], 
    'reward0_sample':[],
    'reward1_sample':[],
    #"responses_len_mean": [],
    "reward0_mean": [],
    "reward1_mean": [],
    "reward0_partition": [],
    "reward1_partition": [],
    'val_error': [],
}
for k in range(num_rewards):
    save_data[f'reward{k}_sample'] = list()
    save_data[f'reward{k}_mean'] = list()
    save_data[f'reward{k}_partition'] = list()




t_start = time.time()
checkpoint=0
for epoch in range(epochs):
    pbar = tqdm(total=(len(dataset) // script_args.batch_size // accelerator.num_processes)-checkpoint)
    for i, batch in enumerate(ppo_trainer.dataloader):
        if i < checkpoint:
            continue
        t_epoch_start = time.time()
        print('epoch {}, batch {}'.format(epoch, i))
        query_tensors = batch["input_ids"]

        #model.gradient_checkpointing_disable()
        #model.pretrained_model.config.use_cache = True
        

        # randomly generate weights
        """
        min_lambda = preference[0]
        max_lambda = preference[1]
        weights = np.zeros((script_args.batch_size, 2))
        weights[:, 0] = np.random.rand(script_args.batch_size)*(max_lambda-min_lambda)+ min_lambda
        weights[:, 1] = 1 - weights[:, 0]
        weights = torch.tensor(np.round(weights, 2))
        weights = [weight for weight in weights]"""
        
        weights = weight_sampler.generate_weights(script_args.batch_size)

        
        moemodel.model.config.use_cache = True
        with torch.no_grad():
            response_tensors = ppo_trainer.generate(query_tensors, weights, batch_size=8, return_prompt=False, **generation_kwargs)
        #t_gen_end = time.time()
        #full_responses = []
        #for response in response_tensors:
        #    full_responses.append(tokenizer.decode(response))
        full_responses = tokenizer.batch_decode(response_tensors)

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
            temp_resp = temp_resp.split('###')[0].strip()
            full_responses_clean.append(temp_resp)

        clean_texts = full_responses_clean
        clean_response_tensors = [tokenizer.encode(text) for text in clean_texts]
        
        lengths = [len(clean_response_tensors[j]) for j in range(len(clean_response_tensors))]
        #response_tensors = [response_tensors[j][:np.max([lengths[j], 2])] for j in range(len(response_tensors))]
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

        if num_rewards == 2:
            reward_ratio = [1.0, 1.0]
        else:
            reward_ratio = [1.0, 1.0, 1.0]
        rewards_list = [ [reward*ratio for reward in rewards_list[k]]  for k, ratio in enumerate(reward_ratio)]

        #rewards_list = [[reward*1.0 for reward in rewards_list[0]], [reward*1.0 for reward in rewards_list[1]]]
        rewards = []
        for j in range(len(queries_responses)):
            #rewards.append(sum([weights[j][k] * rewards_list[k][j] for k in range(num_rewards)]))
            reward_a = sum([weights[j][k] * rewards_list[k][j] for k in range(num_rewards)])
            reward_c = sum([torch.tensor(0.5)*rewards_list[k][j] for k in range(num_rewards)])
            rewards.append(0.8 * reward_a + 0.2 * reward_c)
        rewards_tensor = [torch.tensor(r).to(gpu_id) for r in rewards]
        print("iter {}, batch {}, mean score: {}".format(epoch, i, torch.mean(torch.tensor(rewards)).item()))


        moemodel.model.config.use_cache = False
        #model.gradient_checkpointing_enable()
        #model.pretrained_model.config.use_cache = False
        #print("query_tensors: ",  query_tensors)
        #print("response_tensors: ", response_tensors)
        #print("weights: ", weights)
        #print("rewards_tensor: ", rewards_tensor)
        stats = ppo_trainer.step(query_tensors, weights, response_tensors, rewards_tensor)
        policy_kl = [stats["objective/kl"]]
        val_error = [stats["ppo/val/error"]]
        
        #ppo_trainer.log_stats(stats, batch, rewards)  #ValueError: autodetected range of [nan, nan] is not finite
        #responses_len = [stats["tokens/responses_len_mean"]]


        #print(rewards)
        rewards = torch.stack(rewards).to('cuda')
        policy_kl = torch.stack([torch.tensor(_) for _ in policy_kl]).to('cuda')
        val_error = torch.stack([torch.tensor(_) for _ in val_error]).to('cuda')
        multi_rewards = torch.stack([torch.tensor(_) for _ in rewards_list]).to('cuda')
        multi_rewards = multi_rewards.T 
        weights_tensor = torch.stack([torch.tensor(_) for _ in weights]).to('cuda')
        #responses_len = torch.stack([torch.tensor(_) for _ in responses_len]).to('cuda')
        all_rewards = accelerator.gather_for_metrics(rewards)
        all_policy_kl = accelerator.gather_for_metrics(policy_kl)
        all_val_error = accelerator.gather_for_metrics(val_error)
        all_multi_rewards = accelerator.gather_for_metrics(multi_rewards)
        all_weights = accelerator.gather_for_metrics(weights_tensor)
        #all_responses_len = accelerator.gather_for_metrics(responses_len)
        all_weights = all_weights.cpu().clone()
        
        all_rewards = np.array(all_rewards.cpu().clone())
        all_policy_kl = np.array(all_policy_kl.cpu().clone())
        all_val_error = np.array(all_val_error.cpu().clone())
        all_multi_rewards = all_multi_rewards.cpu().clone()
        #all_responses_len = np.array(all_responses_len.cpu().clone())
        #print("all_weights", all_weights.shape)
        #print("all_multi_rewards", all_multi_rewards.shape)

        if process_id == 0:
            reward_partition = np.mean(np.array(torch.einsum("bn, bn-> bn", all_weights , all_multi_rewards)), axis=0)
            mean_multi_rewards = np.mean(np.array(all_multi_rewards), axis=0)
            #print("mean_multi_rewards : ", mean_multi_rewards)
            #mean_score0.append(mean_multi_rewards[0])
            #mean_score1.append(mean_multi_rewards[1])
            mean_scores.append(torch.mean(torch.tensor(all_rewards)).item())
            std_scores.append(torch.std(torch.tensor(all_rewards)).item())
            save_path = os.path.join(script_args.save_directory, script_args.wandb_name, 'scores.png')
            plt.plot(mean_scores)
            plt.fill_between(np.arange(len(mean_scores)), np.array(mean_scores)- np.array(std_scores), np.array(mean_scores) + np.array(std_scores), alpha=0.5)
            plt.savefig(save_path)
            t_epoch_end = time.time()
            save_data['batch_time'].append(t_epoch_end - t_epoch_start)
            #save_data['total_time'].append(t_epoch_end - t_start)
            save_data['kl_mean'].append(np.mean(all_policy_kl))
            save_data['reward_mean'] = mean_scores
            save_data['reward_std'] = std_scores
            save_data['text_sample'].append(batch['response'][0:3])
            save_data['weights'].append(np.array(weights[0:3]))
            #save_data['original_text_sample'].append(full_responses[0:3])

            for k in range(num_rewards):
                save_data[f'reward{k}_sample'].append(np.array(rewards_list)[k, 0:3])
                save_data[f'reward{k}_mean'].append(mean_multi_rewards[k]) 
                save_data[f'reward{k}_partition'].append(reward_partition[k]) 
            """
            save_data['reward0_sample'].append(np.array(rewards_list)[0, 0:3])
            save_data['reward1_sample'].append(np.array(rewards_list)[1, 0:3])
            
            save_data['reward0_mean'].append(mean_multi_rewards[0]) 
            save_data['reward1_mean'].append(mean_multi_rewards[1]) 
            save_data['reward0_partition'].append(reward_partition[0]) 
            save_data['reward1_partition'].append(reward_partition[1]) 
            """
            save_data['val_error'].append(np.mean(all_val_error))
            #save_data["responses_len_mean"].append(np.mean(all_responses_len))

            dataframe = pd.DataFrame(save_data)
            dataframe.to_csv(os.path.join(script_args.save_directory, script_args.wandb_name,'data.csv'))
            print("iter {}, batch {}: log finish".format(epoch, i))

        # wait for the main process
        accelerator.wait_for_everyone()
        pbar.update(1)
        #weight_sampler.update()
        # save model
        if ppo_trainer.accelerator.is_main_process and i % 5 == 0 and i != 0:
            save_path = os.path.join(script_args.save_directory, script_args.wandb_name, 'epoch_{}_batch_{}'.format(epoch, i))
            ppo_trainer.save_pretrained(save_path)
            print("iter {}, batch {}: model saved".format(epoch, i))
    
    # save model
    if ppo_trainer.accelerator.is_main_process:
        save_path = os.path.join(script_args.save_directory, script_args.wandb_name, 'epoch_{}_batch_{}'.format(epoch, i))
        ppo_trainer.save_pretrained(save_path)
        print("iter {}, batch {}: model saved".format(epoch, i))
        