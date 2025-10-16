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
from accelerate import Accelerator
accelerator  = Accelerator()

seed = 10
random.seed(seed)  # random seed
torch.manual_seed(0)

base_model = "yourpath/HoE/model/assistant_sft_llama"
base_model = "yourpath/model/llama-2-7b-chat-hf"
save_directory = "model/train4or/train1+"
use_dataset = "beaver"

mola_weights = "yourpath/HoE/workspace/MyMoLA/model/initv5/unsafe,chat-fulllora"
mola_weights = "yourpath/HoE/workspace/MyMoLA/model/train4or/train1/epoch2-steps34500"
lora_target_modules = "q_proj,v_proj"
lora_target_modules = "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
number_experts = "2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
top_k = "2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2"
#number_experts = "3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3"



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

model = LlamaForCausalLM_d.from_pretrained(
    base_model,
    config=config,
    load_in_8bit=load_8bit,
    torch_dtype=torch.float16,
    #device_map="cuda:7",
)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))
model = ConditionedMOEModel.from_pretrained(
    model,
    mola_weights,
    torch_dtype=torch.float16,
    number_experts=number_experts,
    #device_map="cuda:7",
    top_k=top_k,
)
print(model)
obalance = False
model.get_new_parameters(number_experts, top_k, obalance)

from peft import (
    prepare_model_for_int8_training,
)
if not load_8bit:
    model = prepare_model_for_int8_training(model)
model.base_model.model.model.config.output_router_logits = True

model.train()
for name, param in model.named_parameters():
    if "router" in name:
        param.requires_grad=True

#---------------------------------------------------------------------------------------------

from datasets import load_dataset
if use_dataset == "assistant":
    harm_ds = load_dataset("yourpath/HoE/dataset/hh-rlhf/harmless-base")["train"]
    help_ds = load_dataset("yourpath/HoE/dataset/hh-rlhf/helpful-base")["train"]
else: 
    beaver_ds = load_dataset("yourpath/HoE/dataset/PKU-SafeRLHF-10K")["train"]
#from transformers import AutoTokenizer
#tokenizer = AutoTokenizer.from_pretrained(base_model)

def label_tokens_between_substrings(input_ids, start_substring, end_substring, target_id = 0):
    text = tokenizer.decode(input_ids)
    
    outputs = tokenizer(text, add_special_tokens=False)
    labels = [-100] * len(outputs['input_ids'])
    current_pos = 0
    
    while True:
        start_idx = text.find(start_substring, current_pos)
        if start_idx == -1:
            break
        end_idx = text.find(end_substring, start_idx + len(start_substring))

        start_token_idx = len(tokenizer.tokenize(text[:start_idx], add_special_tokens=False))   
        if end_idx == -1:
            end_token_idx = len(outputs['input_ids'])
        else:
            end_token_idx = len(tokenizer.tokenize(text[:end_idx], add_special_tokens=False))  
        for i in range(start_token_idx, end_token_idx):
            labels[i] = target_id
        if end_idx == -1:
            break
        current_pos = end_idx + len(end_substring)
    outputs["router_labels"] = labels
    return outputs

def my_data_collator(tokenizer, features):
    first = features[0]

    batch = {}
    for k, v in first.items():
        if isinstance(v, str) and k=="text":
            text = [f[k] for f in features]
            inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
            #print(input_ids['input_ids'].shape)
            batch['input_ids'] = inputs['input_ids']
            batch['attention_mask'] = inputs['attention_mask']
        elif isinstance(v, str) and k=="category":
            batch["target_id"] = torch.tensor([0 if f[k]=="helpful" else 1 for f in features]).view(-1,1)
        else:
            raise NotImplementedError
    
    _batch = []
    for input_ids, target_id, attention_mask in zip(batch['input_ids'], batch["target_id"], batch["attention_mask"]):
        outputs = label_tokens_between_substrings(input_ids, start_substring="\n\nAssistant:", end_substring="\n\nHuman:", target_id=target_id)
        outputs['attention_mask'] = attention_mask
        #print(outputs)
        #assert 0==1
        _batch.append(outputs)

    first = _batch[0]
    batch = {}
    for k, v in first.items():
        if isinstance(v, list):
            batch[k] = torch.stack([torch.tensor(f[k]) for f in _batch])
        elif isinstance(v, torch.Tensor):
            batch[k] = torch.stack([f[k] for f in _batch])
        else:
            raise NotImplementedError
        
    return batch

import pandas as pd
from datasets import Dataset
def preprocess(ds, category="helpful"):
    df = pd.DataFrame(ds)
    df_chosen = df[['chosen']].copy()
    df_chosen = df_chosen.rename(columns={'chosen': 'text'})

    df_rejected = df[['rejected']].copy()
    df_rejected = df_rejected.rename(columns={'rejected': 'text'})
    df_final = pd.concat([df_chosen, df_rejected], ignore_index=True)

    df_final.insert(df_final.shape[1], 'category', category)
    return df_final

def preprocess2(ds):
    df = pd.DataFrame(ds)
    safe_entries = []
    unsafe_entries = []
    
    for _, row in df.iterrows():
        row_0 = {'text': "\n\nHuman: "+row['prompt']+"\n\nAssistant: "+row['response_0']}
        row_1 = {'text': "\n\nHuman: "+row['prompt']+"\n\nAssistant: "+row['response_1']}
        if row['is_response_0_safe'] and row['is_response_1_safe']:
            row_0.update({'category': "helpful"})
            row_1.update({'category': "helpful"})
            safe_entries.append(row_0)
            safe_entries.append(row_1)
        if not row['is_response_0_safe'] and not row['is_response_1_safe']:
            row_0.update({'category': "harmless"})
            row_1.update({'category': "harmless"})
            unsafe_entries.append(row_0)
            unsafe_entries.append(row_1)


    # Convert lists to DataFrames
    safe_ds = pd.DataFrame(safe_entries)
    unsafe_ds = pd.DataFrame(unsafe_entries)
    return [safe_ds, unsafe_ds]

if use_dataset == "assistant":
    df_all = pd.concat([preprocess(help_ds, "helpful"), preprocess(harm_ds, "harmless")], ignore_index=True)
else:
    df_all = pd.concat(preprocess2(beaver_ds), ignore_index=True)
ds = Dataset.from_pandas(df_all)
#ds = ds.filter(lambda x: len(tokenizer.tokenize(x['text'])) <= 400)
ds = ds.shuffle(seed=88)
print(f"num of dataset: {len(ds)}")

#-------------------------------------------------------------------------------------------
import numpy as np
def generate_weights(n_samples):
    num_rewards = 2
    uniform = np.random.uniform(0, 1, n_samples)
    _lambda = np.round(uniform, 2)

    weights = np.zeros((n_samples, num_rewards))
    weights[:, 0] = _lambda
    weights[:, 1] = 1 - weights[:, 0]
    weights = torch.tensor(weights)
    #weights = [weight for weight in weights]
    return weights

import gc
def clean_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()


#from transformers import Trainer
from transformers import Trainer, TrainerCallback
#from src.mola_trainer_hacked import Trainer
#from trl import SFTTrainer
import numpy as np
from torch.nn import CrossEntropyLoss, NLLLoss
from transformers import Trainer

class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        #inputs: dict{input_ids, attention_mask, router_labels}
        labels = inputs.pop("router_labels")
        bs = inputs["input_ids"].shape[0]
        seqlen0 = inputs["input_ids"].shape[1]

        accelerator.unwrap_model(model).dynamic_weights.set_dynamic_weights(generate_weights(bs*seqlen0), batch_size=bs*seqlen0)
        accelerator.unwrap_model(model).dynamic_weights.set_dynamic_labels(labels)
        #inputs = {k:v.to(model.device) for k,v in inputs.items()}

        # output[1] : tuple(tensor(modules*batch_size*seq_len, num_rewards) *32 )
        # router_logits : tensor(32, modules*batch_size*seq_len, num_rewards)
        #output = model.base_model.model.model(**inputs)
        output = accelerator.unwrap_model(model).model.model(**inputs)
        #output = accelerator.unwrap_model(model).forward4or(**inputs)
        router_logits = torch.stack(output[1]) 

        layers, modules_batchsize_seqlen, num_rewards = router_logits.shape
        batchsize, seqlen = labels.shape
        assert bs == batchsize
        assert seqlen0 == seqlen
        assert layers == 32
        assert num_rewards == 2
        assert modules_batchsize_seqlen % (batchsize*seqlen) == 0
        modules =  modules_batchsize_seqlen // (batchsize*seqlen)
        assert modules == len(lora_target_modules)

        #labels : tensor(batch_size, seq_len)
        labels = labels.view(-1)
        target = labels.repeat(32*modules)
        #target : tensor(32*modules*batch_size*seq_len)
        router_logits = router_logits.view(-1, num_rewards)
        # router_logits : tensor(32*modules*batch_size*seq_len, num_rewards)

        loss_fn = torch.nn.NLLLoss(ignore_index=-100)
        #loss = loss_fn(router_logits.log(), target.to(model.device))
        loss = loss_fn(router_logits.log(), target)
        return loss
    


from transformers import TrainingArguments
#from trl import SFTConfig
import wandb 
training_args = TrainingArguments(
    per_device_train_batch_size= 2,
    gradient_checkpointing= False,
    gradient_accumulation_steps= 2,
    learning_rate= 1e-5,
    lr_scheduler_type= "cosine", 
    save_strategy= "no", 
    #save_steps= 500,
    do_eval= False,
    logging_steps= 100,
    fp16= False, 
    bf16= True,
    #optim= "paged_adamw_32bit",
    optim= "adamw_torch",
    warmup_steps= 500,
    #report_to= "wandb",
    output_dir= save_directory, 
    #run_name= "ft-mistral-7b-irca-dpo-pairs",
    use_cpu= False,
    torch_empty_cache_steps= 20,
    num_train_epochs= 3.0,
    #max_steps= 100,
    #label_names= ["label_ids"],
    remove_unused_columns= False,
    max_grad_norm = 1.0
)



# Create a custom callback
class PeriodicSaveCallback(TrainerCallback):
    def __init__(self, accelerator, model, save_directory):
        self.accelerator = accelerator
        self.model = model
        self.save_directory = save_directory

    def on_step_end(self, args, state, control, **kwargs):
        # Save every 100 steps
        if state.global_step % 500 == 0 and state.global_step > 0:
            save_path = os.path.join(self.save_directory, f"epoch{int(state.epoch)}-steps{state.global_step}")
            # Unwrap the model from accelerator for saving and save it
            self.accelerator.unwrap_model(self.model).save_pretrained(save_path)
            print(f"Model saved at step {state.global_step} to {save_path}")
        return control

periodic_save_callback = PeriodicSaveCallback(accelerator, model, save_directory)


from transformers import DataCollatorForTokenClassification, DataCollatorWithPadding
trainer = MyTrainer(
    model= model,
    args= training_args,
    train_dataset= ds,
    #max_prompt_length= 1024,
    #max_length= 1536,
    data_collator = lambda x: my_data_collator(tokenizer, x),
    callbacks=[periodic_save_callback]
)

trainer.train()
