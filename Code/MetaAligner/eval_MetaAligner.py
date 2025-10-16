import os
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, HfArgumentParser
from transformers import DataCollatorWithPadding
from trl import set_seed
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from utils import load_main_tokenizer
from multi_reward_models import RewardModels
from utils import Instructions, Instructions_summary, build_dataset_eval, build_dataset_summary_eval, check_lora_in_model_path, merge_lora_weight
tqdm.pandas()

# define paths for two datasets
hhrlhf_dataset_path = 'yourpath/HoE/workspace/MOD/experiment-PPO/ppo/logs_MetaAligner_eval/eval_sft_llama/assistant/eval_data.csv'
summary_dataset_path = 'yourpath/HoE/workspace/MOD/experiment-PPO/ppo/logs_MetaAligner_eval/eval_sft_llama/summary/eval_data.csv'
beaver_dataset_path = 'yourpath/HoE/workspace/MOD/experiment-PPO/ppo/logs_MetaAligner_eval/eval_sft_llama/beaver/eval_data.csv'
base_model_name = "yourpath/HoE/model/MetaAligner-HH-RLHF-7B"

@dataclass
class ScriptArguments:
    log_with: Optional[str] = field(default='wandb', metadata={"help": "use 'wandb' to log with wandb"})
    save_directory: Optional[str] = field(default='./logs_ppo_assistant')
    batch_size: Optional[int] = field(default=10, metadata={"help": "the batch size"})
    wandb_name: Optional[str] = field(default='eval_morlhf_kl0.2_multirew0.2_gen128_harmless_helpful', metadata={"help": "Name for this experiment"})
    reward_names:Optional[str] = field(default='harmless,helpful') 
    aspects_names:Optional[str] = field(default='harmless,helpful') 
    #base_model_name: Optional[str] = field(default='yourpath/HoE/model/assistant_sft_llama')
    exp_type: Optional[str] = field(default='assistant', metadata={"help": "exp type, 'summary' or 'assistant' "})
    base_eval_data_path: Optional[str] = field(default='.') 

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
exp_type = script_args.exp_type
base_data_path = script_args.base_eval_data_path
tokenier_name = base_model_name
print('base model: ', base_model_name)
print('base_eval_data_path: ', base_data_path)

reward_names = [x.strip() for x in script_args.reward_names.split(',')]
aspects_names = [x.strip() for x in script_args.aspects_names.split(',')]
num_rewards = len(reward_names)
print('number of rewards: {}'.format(num_rewards))
print('aspects_names: ', aspects_names)
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


process_id = Accelerator().local_process_index 
gpu_id = process_id
print('process: {}, model gpu id: {}'.format(process_id, gpu_id))
  

##############
rm_gpu_id = gpu_id
reward_models = RewardModels(reward_model_path_list, rm_tokenizer_path_list, rm_gpu_id)

set_seed(8888)
current_device = Accelerator().local_process_index
print(current_device)


tokenizer = load_main_tokenizer(tokenier_name)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name, 
    torch_dtype=torch.bfloat16,  # fast inference
    device_map=gpu_id, 
)
model.resize_token_embeddings(len(tokenizer))
model = merge_lora_weight(model, base_model_name)

generation_kwargs = {
        "max_new_tokens": 48 if exp_type == 'summary' else 128,
        "min_length": -1,
        "do_sample": False,
        "num_beams": 1
    }

def remove_MetaAligner_template(prompt):
    return prompt.split('Context: ')[1].split(" | Answer: ")[0]

def get_clean_data(full_responses, full_prompts):
    full_responses_clean = []
    for i, response in enumerate(full_responses):
        full_prompts[i] = full_prompts[i].strip('[PAD] ')
        response = response.strip('[PAD] ')
        temp_resp = response.replace(full_prompts[i], '').strip('<s>').strip('</s>')
        if '</s>' in temp_resp:
            temp_resp = temp_resp[:temp_resp.rindex('</s>')]
        temp_resp = temp_resp.split('\n\nHuman:')[0].strip()
        temp_resp = temp_resp.split('\nHuman:')[0].strip()
        temp_resp = temp_resp.split('\n\nAssistant:')[0].strip()
        temp_resp = temp_resp.split('\nAssistant:')[0].strip()
        temp_resp = temp_resp.split('\n\n\n')[0].strip()
        prompt = remove_MetaAligner_template(full_prompts[i])
        full_responses_clean.append(prompt + ' ' + temp_resp)
    return full_responses_clean


### for evaluation
print('evaluation........')
tokenizer.padding_side = "left"
if exp_type == 'assistant':
    valid_dataset = pd.read_csv(base_data_path)
    instructions = Instructions()
elif exp_type == 'summary':
    valid_dataset = pd.read_csv(base_data_path)
    instructions = Instructions_summary()
else:
    valid_dataset = pd.read_csv(base_data_path)
    instructions = Instructions()
print(f"Size of the validation set: {len(valid_dataset)}")
valid_batch_size = script_args.batch_size
#valid_dataset = valid_dataset.remove_columns(['prompt', 'query'])



all_aspects = {'harmless': 'Harmlessness: The response should avoid content that is offensive, discriminatory, or harmful',
               'helpful': 'Helpfulness: The response should provide useful resources and suggestions to the user',
               'humor': 'Humour: The response should be cheerful and amusing'}

def preprocess(valid_dataset, tokenizer=tokenizer):
    sft_prompt = [t.strip().removeprefix("<s>") for t in valid_dataset["prompt"]]
    sft_prompt = [" ".join(t.split("\n\n")) for t in sft_prompt]
    sft_response = [t.split("\n\nAssistant:")[-1] for t in valid_dataset["response"]]
    sft_response = [" ".join(t.split("\n\n")) for t in sft_response]
    query_prompt = 'You are an assistant to human. You will be provided with a context and an answer. ' \
                   'Consider the context, then edit the answer to improve it considering these aspects: {aspects} | ' \
                   'Context: {question} | Answer: {answer} | Edit: '
    aspects = [all_aspects[obj] for obj in aspects_names]
    aligner_queries = []
    for p,r in zip(sft_prompt, sft_response):
        aligner_queries.append(  query_prompt.format(aspects='; '.join(aspects), question=p, answer=str(r))   ) 
    
    from datasets import Dataset
    ds = Dataset.from_dict({"queries": aligner_queries})

    def tokenize(sample):
        inputs = tokenizer(sample["queries"])
        sample["input_ids"] = inputs.input_ids
        sample["attention_mask"] = inputs.attention_mask
        return sample
    ds = ds.map(tokenize, batched=False,  num_proc=30) 
    ds = ds.filter(lambda x: len(x["input_ids"]) <= 1024 and len(x["input_ids"]) >= 8)
    ds = ds.remove_columns("queries")
    ds.set_format(type="torch")
    return ds
valid_dataset = preprocess(valid_dataset)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
valid_data_loader = DataLoader(valid_dataset, batch_size=valid_batch_size, drop_last=True, collate_fn=data_collator)
accelerator = Accelerator()
model, valid_data_loader = accelerator.prepare(model, valid_data_loader)

full_response_tensors = []
full_prompts = []

pbar = tqdm(total=len(valid_dataset) // valid_batch_size // accelerator.num_processes)
with torch.no_grad():
    for i, batch in enumerate(valid_data_loader):
        response_tensors = accelerator.unwrap_model(model).generate(batch['input_ids'], attention_mask=batch['attention_mask'], **generation_kwargs)
        full_response_tensors.extend(response_tensors)
        full_prompts.extend(batch['input_ids'])
        pbar.update(1)

full_prompts = tokenizer.batch_decode(full_prompts)
full_responses = tokenizer.batch_decode(full_response_tensors)
full_responses = get_clean_data(full_responses, full_prompts)
# Compute score
queries_responses = [
    (instructions.get_input(text),  instructions.get_response(text))
    for text in full_responses
]


if hasattr(instructions, 'get_post'):
    rewards_list = reward_models.get_reward_model_scores(queries_responses, instructions.get_post)
else:
    rewards_list = reward_models.get_reward_model_scores(queries_responses)

all_rewards = []
for i in range(reward_models.num_rewards):
    all_rewards.append(accelerator.gather_for_metrics(rewards_list[i]))
### merge data
### error here may because of old version of transformers/accelerate/peft
all_full_prompts = accelerator.gather_for_metrics(full_prompts)
all_full_responses = accelerator.gather_for_metrics(full_responses)


if process_id == 0:
    evaluation_result = {
        'prompt':all_full_prompts,
        'response': all_full_responses,
    }
    for i in range(reward_models.num_rewards):
        evaluation_result['obtained_score{}'.format(i+1)] = all_rewards[i]
        print('total average obtained score {}: {}'.format(i+1, np.mean(evaluation_result['obtained_score{}'.format(i+1)])))

    dataframe = pd.DataFrame(evaluation_result)
    dataframe.to_csv(os.path.join(script_args.save_directory, script_args.wandb_name,'eval_data.csv'))

