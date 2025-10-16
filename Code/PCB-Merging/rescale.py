import os
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser

@dataclass
class ScriptArguments:
    sft_model_path: Optional[str] = field(default='')
    model_path: Optional[str] = field(default='')
    rescale: Optional[str] = field(default=1.0)

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
rescale = float(script_args.rescale)
print("rescale", rescale)
sft_model_path = script_args.sft_model_path
model_path  = script_args.model_path


import torch
from transformers import AutoModelForCausalLM
sft_model = AutoModelForCausalLM.from_pretrained(
    sft_model_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model

peft_model = PeftModel.from_pretrained(sft_model, model_path)
peft_model_dict = peft_model.state_dict()

for name, param in peft_model_dict.items():
    if "lora_A" in name:
        print("replacing: ", name)
        peft_model_dict[name].copy_(rescale*param)
peft_model.load_state_dict(peft_model_dict)

save_directory = f"{model_path}_{rescale}"
print("save_directory", save_directory)
peft_model.save_pretrained(save_directory)