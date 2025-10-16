import os
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
import torch
from tqdm import tqdm
from transformers import HfArgumentParser
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
import numpy as np
import pandas as pd
from utils import print_trainable_parameters, load_main_tokenizer, Instructions, Instructions_summary, \
                  build_dataset, build_dataset_summary, build_dataset_beaver, build_dataset_steer             
from multi_reward_models import RewardModels
tqdm.pandas()
from peft import LoraConfig
import matplotlib.pyplot as plt

# define paths for two datasets
hhrlhf_dataset_path = ' yourpath/dataset/hh-rlhf'
summary_dataset_path = ' yourpath/dataset/summarize_from_feedback/comparisons'
beaver_dataset_path = " yourpath/dataset/PKU-SafeRLHF-10K"
steer_dataset_path = " yourpath/dataset/HelpSteer"
steer2_dataset_path = " yourpath/dataset/HelpSteer2"
@dataclass
class ScriptArguments:
    log_with: Optional[str] = field(default='wandb', metadata={"help": "use 'wandb' to log with wandb"})
    disable_wandb: Optional[str] = field(default=True, metadata={'help': 'Whether to disable wandb or not.'})
    save_directory: Optional[str] = field(default='./logs_ppo_summary/')
    epochs: Optional[int] = field(default=7, metadata={'help': "Number of training epoches"})
    learning_rate: Optional[float] = field(default=1e-5, metadata={"help": "the learning rate"})
    mini_batch_size: Optional[int] = field(default=6, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=120, metadata={"help": "the batch size"})
    load_in_8bit: Optional[bool] = field(default=True, metadata={"help": "loading model in 8 bit or bfloat16"})
    gradient_accumulation_steps: Optional[int] = field(default=1, metadata={"help": "the number of gradient accumulation steps"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    target: Optional[float] = field(default=12, metadata={"help": "target kl divergence of adaptive control"})
    init_kl_coef: Optional[float] = field(default=0.85,metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},)
    max_grad_norm: Optional[float] = field(default=0.5, metadata={"help": "Maximum gradient norm for gradient clipping"})
    wandb_name: Optional[str] = field(default='ppo_llamma2_klreg0.2_summary_faithfulrm', metadata={"help": "Name for this experiment"})
    exp_type: Optional[str] = field(default='summary', metadata={"help": "exp type: 'assistant" or 'summary'}) 
    base_model_name: Optional[str] = field(default='./merged_sft_summary', metadata={'help':"the path to the sft model; need to merge if using lora"})
    reward_name: Optional[str] = field(default='faithful')
    checkpoint_name: Optional[str] = field(default='')

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
exp_type = script_args.exp_type
checkpoint_name = script_args.checkpoint_name
# Remember to use a merged sft model if using lora 
base_model_name = script_args.base_model_name
tokenier_name = script_args.base_model_name
print('base model: ', base_model_name)

if script_args.disable_wandb: # if you don't need the wandb log
    os.environ['WANDB_DISABLED'] = 'true' 

if exp_type == 'assistant':
    script_args.save_directory = './logs_ppo_assistant/'
elif exp_type == 'beaver':
    script_args.save_directory = './logs_ppo_beaver/'
elif exp_type == 'steer':
    script_args.save_directory = './logs_ppo_steer/'
elif exp_type == 'steer2':
    script_args.save_directory = './logs_ppo_steer2/'
else:
    script_args.save_directory = './logs_ppo_summary/'

reward_name = script_args.reward_name
reward_path_tokenizer_dict = {
    'harmless': [' yourpath/model/gpt2-large-harmless-reward_model'],
    'helpful': [' yourpath/model/gpt2-large-helpful-reward_model'],
    'deberta': [' yourpath/model/reward-model-deberta-v3-large-v2'],
    'summary': [' yourpath/model/gpt2_reward_summarization'],
    'faithful':[' yourpath/model/bart-faithful-summary-detector'],
    'humor': [' yourpath/model/humor-no-humor'],
    'cost': [" yourpath/model/beaver-7b-v1.0-cost"],
    'reward': [" yourpath/model/beaver-7b-v1.0-reward"],
    'steer_helpful': [" yourpath/model/steer_helpful/RewardModel-Mistral-7B-for-DPA-v1"],
    'steer_correct': [" yourpath/model/steer_correct/RewardModel-Mistral-7B-for-DPA-v1"],
    'steer_coherence': [" yourpath/model/steer_coherence/RewardModel-Mistral-7B-for-DPA-v1"],
    'steer_complex': [" yourpath/model/steer_complex/RewardModel-Mistral-7B-for-DPA-v1"],
    'steer_verbosity': [" yourpath/model/steer_verbosity/RewardModel-Mistral-7B-for-DPA-v1"],
    'Armo': [" yourpath/model/ArmoRM-Llama3-8B-v0.1"]
}
reward_peft_path = reward_path_tokenizer_dict[reward_name][0]
rm_tokenizer_path = reward_peft_path
os.makedirs(os.path.join(script_args.save_directory, script_args.wandb_name), exist_ok=True)


config = PPOConfig(
    model_name=base_model_name,
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
    tracker_project_name='ppo',
    tracker_kwargs={"wandb":{"name":script_args.wandb_name}},
    horizon=20000
)

accelerator = Accelerator()
process_id = Accelerator().local_process_index 
gpu_id = process_id
print('process: {}'.format(process_id))
reward_model = RewardModels([reward_peft_path], [rm_tokenizer_path], gpu_id)
rm_tokenizer = reward_model.rm_tokenizers[0] 


# set seed before initializing value head for deterministic eval
set_seed(8888)
current_device = Accelerator().local_process_index
print(current_device)

lora_config = LoraConfig(
    r=64, 
    lora_alpha=128, 
    lora_dropout=0.05,
    bias="none",
    target_modules=[
    "down_proj",
    "v_proj",
    "k_proj",
    "up_proj",
    "o_proj",
    "gate_proj",
    "q_proj"
   ],
    task_type="CAUSAL_LM",
)

tokenizer = load_main_tokenizer(tokenier_name)
if exp_type == 'assistant':
    dataset = build_dataset(hhrlhf_dataset_path, tokenizer, rm_tokenizer, split='train')
    instructions = Instructions()
elif exp_type == 'summary':
    dataset = build_dataset_summary(summary_dataset_path, tokenizer, rm_tokenizer, split='train')
    instructions = Instructions_summary()
elif exp_type == 'beaver':
    dataset = build_dataset_beaver(beaver_dataset_path, tokenizer, rm_tokenizer, split='train')
    instructions = Instructions()
elif exp_type == 'steer':
    dataset = build_dataset_steer(steer_dataset_path, tokenizer, rm_tokenizer, split='train')
    instructions = Instructions()
elif exp_type == 'steer2':
    dataset = build_dataset_steer(steer2_dataset_path, tokenizer, rm_tokenizer, split='train')
    instructions = Instructions()
else: 
    raise NotImplementedError
train_dataset = dataset.shuffle()
print(f"Size of the train set: {len(train_dataset)}.")

if script_args.load_in_8bit:
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        base_model_name,
        load_in_8bit=True,
        peft_config=lora_config,
        #device_map=gpu_id,
        device_map="cpu",
    )
else:
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        peft_config=lora_config,
        #device_map=gpu_id,
        device_map="cpu",
    )


#if len(checkpoint_name) > 0:
#checkpoint_name = " yourpath/workspace/RiC/RiC/ppo/logs_ppo_summary/train4moe3_summary/epoch_0_batch_163"
if len(checkpoint_name) > 1:
    print("checkpoint_name", checkpoint_name)

    from transformers import AutoModelForCausalLM
    sft_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        load_in_8bit = True,
        device_map=gpu_id,
    )
    from transformers import LlamaForCausalLM, LlamaTokenizer
    from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model
    checkpoint_model = PeftModel.from_pretrained(sft_model, checkpoint_name)

    model_state_dict = model.pretrained_model.state_dict()
    for name, param in checkpoint_model.named_parameters():
        if "lora" in name and name in model_state_dict:
            print("replacing: ", name)
            model_state_dict[name].copy_(param)
        else:
            print(f"Warning: {name} not found in the target model. Skipping.")
    model.pretrained_model.load_state_dict(model_state_dict)
    del checkpoint_model, sft_model

    value_head_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        checkpoint_name,
        load_in_8bit=True,
        peft_config=lora_config,
        device_map=gpu_id,
    )
    with torch.no_grad():
        model.v_head.summary.weight.copy_(value_head_model.v_head.summary.weight)
        model.v_head.summary.bias.copy_(value_head_model.v_head.summary.bias)
    del value_head_model

for name,param in model.named_parameters():
    if "lora" in name:
        param.requires_grad=True
    elif "v_head" in name:
        print(name, param)
        param.requires_grad=True
    else:
        param.requires_grad=False

print_trainable_parameters(model)
model.pretrained_model.resize_token_embeddings(len(tokenizer))
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)
def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

ppo_trainer = PPOTrainer(
    config, model, tokenizer=tokenizer, dataset=dataset, data_collator=collator, optimizer=optimizer
)

generation_kwargs = {
    "max_new_tokens": 48 if exp_type == 'summary' else 128,
    'min_length': -1, 
    "top_k": 0.0,
    "top_p": 0.95, 
    "do_sample": True,
    "temperature": 0.7,
    "pad_token_id": tokenizer.eos_token_id,
    "begin_suppress_tokens": [tokenizer.eos_token_id],
    "no_repeat_ngram_size": 5
}

print("Training........")
model.gradient_checkpointing_disable()
model.pretrained_model.config.use_cache = True

epochs = script_args.epochs
mean_scores = []
std_scores = []
save_data = {
    'kl_mean': [],
    'kl_std': [],
    'reward_mean': [],
    'reward_std': [],
    'text_sample':[],
}
for epoch in range(epochs):
    pbar = tqdm(total=len(train_dataset) // script_args.batch_size // accelerator.num_processes)
    for i, batch in enumerate(ppo_trainer.dataloader):
        #if epoch==0 and i<50 and reward_name=="humor":
            #ppo_trainer.kl_ctl.value = 1.0 - (0.15/50) * i
        print("ppo_trainer.kl_ctl.value", ppo_trainer.kl_ctl.value)
        #generation_kwargs["temperature"] = max(0.05, generation_kwargs["temperature"]-0.001)
        print('epoch {}, batch {}'.format(epoch, i))
        query_tensors = batch["input_ids"]

        model.gradient_checkpointing_disable()
        model.pretrained_model.config.use_cache = True
            
        with torch.no_grad():
            response_tensors = ppo_trainer.generate(query_tensors, batch_size=30, return_prompt=False, **generation_kwargs) 

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
            temp_resp = temp_resp.split('###')[0].strip()
            temp_resp = temp_resp.split('Instruction:')[0].strip()
            temp_resp = temp_resp.split('// end of')[0].strip()
            temp_resp = temp_resp.split('// End of')[0].strip()
            if '</s>' in temp_resp:
                temp_resp = temp_resp[:temp_resp.rindex('</s>')]
            temp_resp = temp_resp.split('<|eot_id|>')[0].strip()     
            temp_resp = temp_resp.split('<|start_header_id|>')[0].strip()     
            temp_resp = temp_resp.split('<|end_header_id|>')[0].strip()     
            temp_resp = temp_resp.split('<|end_of_text|>')[0].strip()     
            temp_resp = temp_resp.split('<|end_of_text|')[0].strip()     
            temp_resp = temp_resp.split('\n\nuser:')[0].strip()
            temp_resp = temp_resp.split('\nuser:')[0].strip()
            temp_resp = temp_resp.split('\n\n\n')[0].strip()
            full_responses_clean.append(temp_resp)

        clean_texts = full_responses_clean
        clean_response_tensors = [tokenizer.encode(text) for text in clean_texts]
        
        lengths = [len(clean_response_tensors[j]) for j in range(len(clean_response_tensors))]
        print(lengths)
        response_tensors = [response_tensors[j][:np.max([lengths[j], 2])] for j in range(len(response_tensors))]
        batch['response'] = clean_texts
 
        # Compute score
        texts_merge = [q.replace("\n\nuser: ", '\n\nHuman: ') + r for q, r in zip(batch['query'], batch['response'])]
        queries_responses = [
            (instructions.get_input(text), instructions.get_response(text))
            for text in texts_merge
        ]
        if hasattr(instructions, 'get_post'):
            rewards = reward_model.get_reward_model_scores(queries_responses, instructions.get_post)[0]
        else:
            rewards = reward_model.get_reward_model_scores(queries_responses)[0]
        rewards_tensor = [torch.tensor(r).to(gpu_id) for r in rewards]
        print("iter {}, batch {}: mean score: {}".format(epoch, i, torch.mean(torch.tensor(rewards)).item()))

        model.gradient_checkpointing_enable()
        model.pretrained_model.config.use_cache = False
        ppo_trainer.config.batch_size = len(query_tensors)
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards_tensor)
        ppo_trainer.log_stats(stats, batch, rewards)
        policy_kl = [stats["objective/kl"]]

        all_rewards = accelerator.gather_for_metrics(rewards)
        all_policy_kl = accelerator.gather_for_metrics(policy_kl)
        if ppo_trainer.accelerator.is_main_process:
            mean_scores.append(torch.mean(torch.tensor(rewards)).item())
            std_scores.append(torch.std(torch.tensor(rewards)).item())
            save_path = os.path.join(script_args.save_directory, script_args.wandb_name, 'scores.png')
            plt.plot(mean_scores)
            plt.fill_between(np.arange(len(mean_scores)), np.array(mean_scores) - np.array(std_scores), np.array(mean_scores) + np.array(std_scores), alpha=0.5)
            plt.savefig(save_path)

            save_data['kl_mean'].append(np.mean(all_policy_kl))
            save_data['kl_std'].append(np.std(all_policy_kl))
            save_data['reward_mean'] = mean_scores
            save_data['reward_std'] = std_scores
            save_data['text_sample'].append(texts_merge[0])
            dataframe = pd.DataFrame(save_data)
            dataframe.to_csv(os.path.join(script_args.save_directory, script_args.wandb_name,'data.csv'), sep='\t', escapechar='\\')
            print("iter {}, batch {}: log finish".format(epoch, i))

        # wait for the main process
        accelerator.wait_for_everyone()
        pbar.update(1)

        # save model
        if ppo_trainer.accelerator.is_main_process and i % 30 == 0 and i != 0:
            save_path = os.path.join(script_args.save_directory, script_args.wandb_name, 'epoch_{}_batch_{}'.format(epoch, i))
            ppo_trainer.save_pretrained(save_path)
            print("iter {}, batch {}: model saved".format(epoch, i))

    # save model
    if ppo_trainer.accelerator.is_main_process:
        save_path = os.path.join(script_args.save_directory, script_args.wandb_name, 'epoch_{}_batch_{}'.format(epoch, i))
        ppo_trainer.save_pretrained(save_path)
        print("iter {}, batch {}: model saved".format(epoch, i))
            