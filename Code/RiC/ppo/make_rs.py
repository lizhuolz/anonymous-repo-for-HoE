import os
import gc
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import HfArgumentParser
from transformers import AutoModelForCausalLM, DataCollatorWithPadding
from torch.utils.data import DataLoader
from trl import set_seed
import numpy as np
import pandas as pd
from utils import get_clean_data, load_main_tokenizer, save_configs, print_trainable_parameters, \
                  merge_weights_with_preference, Instructions, Instructions_summary, build_dataset_eval, build_dataset_summary_eval
from multi_reward_models import RewardModels
tqdm.pandas()


# define paths for two datasets
tokenizer_path = 'meta-llama/Llama-2-7b-hf'
tokenizer_path = '/yourpath/public/model'

@dataclass
class ScriptArguments:
    temp_save_path:Optional[str] = field(default="./temp_models/")
    base_model_path1: Optional[str]=field(default='./ppo_llamma2_klreg0.2_harmless/batch_200')
    base_model_path2: Optional[str]=field(default='./ppo_llamma2_klreg0.2_helpful/batch_200')
    base_model_path3: Optional[str]=field(default='')

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
temp_save_path = script_args.temp_save_path
base_model_name_1 = script_args.base_model_path1
base_model_name_2 = script_args.base_model_path2
base_model_name_3 = script_args.base_model_path3
tokenier_name = tokenizer_path

preference = [0.5,0.5]
#preference = [0.333,0.333,0.334]
if len(preference) == 3:
    base_model_list = [base_model_name_1, base_model_name_2, base_model_name_3]
else:
    base_model_list = [base_model_name_1, base_model_name_2]
merge_weights_with_preference(base_model_list, preference, temp_save_path)
