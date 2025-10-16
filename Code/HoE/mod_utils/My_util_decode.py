import numpy as np
import torch
from transformers import T5ForConditionalGeneration, T5Config
from transformers.models.t5.modeling_t5 import T5Stack
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import sys
import time
sys.path.append(".")
from transformers.utils.generic import ModelOutput
import torch.distributed as dist
from torch import nn
from transformers.modeling_utils import PreTrainedModel
from peft import PeftModel, LoraConfig
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
import copy, inspect, warnings
from collections import UserDict



import sys
sys.path.append('yourpath/HoE')

import os


import fire
import torch
import torch.nn as nn
import torch.nn.functional as F
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



class ValueHead(nn.Module):
    r"""
    The ValueHead class implements a head for GPT2 that returns a scalar for each output token.
    """

    def __init__(self, config, **kwargs):
        super().__init__()
        if not hasattr(config, "summary_dropout_prob"):
            summary_dropout_prob = kwargs.pop("summary_dropout_prob", 0.1)
        else:
            summary_dropout_prob = config.summary_dropout_prob

        self.dropout = nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity()

        # some models such as OPT have a projection layer before the word embeddings - e.g. OPT-350m
        if hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size
        if hasattr(config, "word_embed_proj_dim"):
            hidden_size = config.word_embed_proj_dim
        elif hasattr(config, "is_encoder_decoder"):
            if config.is_encoder_decoder and hasattr(config, "decoder"):
                if hasattr(config.decoder, "hidden_size"):
                    hidden_size = config.decoder.hidden_size

        self.summary = nn.Linear(hidden_size, 1)

        self.flatten = nn.Flatten()

    def forward(self, hidden_states):
        output = self.dropout(hidden_states)

        # For now force upcast in fp32 if needed. Let's keep the
        # output in fp32 for numerical stability.
        if output.dtype != self.summary.weight.dtype:
            output = output.to(self.summary.weight.dtype)

        output = self.summary(output)
        return output


class DynamicWeights:
    def __init__(self, num_rewards=2):
        self.num_rewards = num_rewards
        self.weights = None  
        self.labels = None

    def set_dynamic_weights(self, weights, batch_size=1):
        assert weights.shape == torch.Size([batch_size, self.num_rewards])
        self.weights = weights  

    def set_dynamic_labels(self, labels, batch_size=1):
        #assert labels.shape[0] == torch.Size([batch_size, self.num_rewards])
        self.labels = labels  


class CustomLinearv0(nn.Linear):  
    def __init__(self, in_features, out_features, dynamic_weights: DynamicWeights ):
        super(CustomLinearv0, self).__init__(in_features, out_features, bias=True)
        nn.init.constant_(self.bias, 1.0)
        nn.init.constant_(self.weight, 0.0)
        #self.bias.data = torch.tensor([0.7, 1.3])

    def forward(self, input):
        output = super(CustomLinearv0, self).forward(input.to(torch.float32)) #train
        return output
    
class CustomLinearv0_plus(nn.Module): 
    def __init__(self, in_features, out_features, dynamic_weights: DynamicWeights ):

        super(CustomLinearv0_plus, self).__init__()
        hidden_dim = 12
        self.linear1 = nn.Linear(in_features, hidden_dim, bias=True)
        self.linear2 = nn.Linear(hidden_dim, out_features, bias=True)

        nn.init.constant_(self.linear1.weight, 0.0)
        nn.init.constant_(self.linear1.bias, 0.0)
        nn.init.constant_(self.linear2.weight, 0.0)
        nn.init.constant_(self.linear2.bias, 1.0)

    def forward(self, input):
        x0 = self.linear1(input.to(torch.float32))
        x0 = torch.relu(x0)
        output = self.linear2(x0)
        return output

class CustomLinearv1(nn.Linear):  
    def __init__(self, in_features, out_features, dynamic_weights: DynamicWeights ):
        
        num_rewards = dynamic_weights.num_rewards
        assert num_rewards == out_features

        self.dynamic_weights = dynamic_weights
        super(CustomLinearv1, self).__init__(in_features + num_rewards, out_features, bias=False)
        self._init_weights()

    def _init_weights(self):
        num_rewards = self.out_features
        init_w = torch.zeros((self.out_features, self.in_features))
        for idx in range(num_rewards):
            init_w[idx, -num_rewards+idx] = 1.0
        self.weight.data = init_w

    def forward(self, input):
        
        weights = self.dynamic_weights.weights
        assert weights is not None
        #input: tensor(batch_size*sequence_length, hidden_dim)
        #weights: tensor(batch_size, num_reward)
        assert input.shape[0] % weights.shape[0] == 0
        extra_input = weights.unsqueeze(1).repeat(1, int(input.shape[0]/weights.shape[0]), 1)
        extra_input = extra_input.view(-1, weights.size(1)).to(input.device)

        #("input", input.shape)
        #print("extra_input", extra_input.shape)
        new_input = torch.concat([input, extra_input], dim=-1).to(input.device)
        #print("new_input: ", new_input.shape)
        #print(self.weight.shape)

        output = super(CustomLinearv1, self).forward(new_input.to(torch.float32)) #train
        #print("output: ", output)
        return output


class CustomLinearv2(nn.Linear):    
    def __init__(self, in_features, out_features, dynamic_weights: DynamicWeights ):
        
        num_rewards = dynamic_weights.num_rewards
        assert num_rewards == out_features

        self.dynamic_weights = dynamic_weights
        super(CustomLinearv2, self).__init__(in_features + num_rewards, out_features, bias=True)
        self._init_weights()

    def _init_weights(self):
        self.weight.data = torch.zeros((self.out_features, self.in_features))
        self.bias.data = torch.ones_like(self.bias.data)

    def forward(self, input):
        
        weights = self.dynamic_weights.weights
        assert weights is not None
        #input: tensor(batch_size*sequence_length, hidden_dim)
        #weights: tensor(batch_size, num_reward)
        assert input.shape[0] % weights.shape[0] == 0
        extra_input = weights.unsqueeze(1).repeat(1, int(input.shape[0]/weights.shape[0]), 1)
        extra_input = extra_input.view(-1, weights.size(1)).to(input.device)

        new_input = torch.concat([input, extra_input], dim=-1).to(input.device)
        
        output = super(CustomLinearv2, self).forward(new_input.to(torch.float32)) 
        output = torch.einsum('ln,ln->ln', output, extra_input)
        #print("final_output: ", output)
        return output

class CustomLinearv3(nn.Module): 
    def __init__(self, in_features, out_features, dynamic_weights: DynamicWeights ):
        
        num_rewards = dynamic_weights.num_rewards
        assert num_rewards == out_features

        self.dynamic_weights = dynamic_weights

        super(CustomLinearv3, self).__init__()

        hidden_dim = 32
        self.linear1 = nn.Linear(in_features + num_rewards, hidden_dim, bias=True)
        self.linear2 = nn.Linear(hidden_dim, out_features, bias=False)
        self.linear3 = nn.Linear(in_features + num_rewards, hidden_dim, bias=True)
        self.linear4 = nn.Linear(hidden_dim, out_features, bias=False)
        self._init_weights()

    def _init_weights(self):
        initializer_range = 0.001
        self.linear1.weight.data.normal_(mean=0.0, std=initializer_range)
        self.linear1.bias.data.zero_()
        self.linear2.weight.data.normal_(mean=0.0, std=initializer_range)
        self.linear3.weight.data.normal_(mean=0.0, std=initializer_range)
        self.linear3.bias.data.zero_()
        self.linear4.weight.data.normal_(mean=0.0, std=initializer_range)

    def forward(self, input):
        
        weights = self.dynamic_weights.weights
        assert weights is not None
        #input: tensor(batch_size*sequence_length, hidden_dim)
        #weights: tensor(batch_size, num_reward)
        assert input.shape[0] % weights.shape[0] == 0
        extra_input = weights.unsqueeze(1).repeat(1, int(input.shape[0]/weights.shape[0]), 1)
        extra_input = extra_input.view(-1, weights.size(1)).to(input.device)
        new_input = torch.concat([input, extra_input], dim=-1).to(input.device)
        new_input = new_input.to(torch.float32)

        x0 = self.linear1(new_input)
        x0 = torch.relu(x0)
        x0 = self.linear2(x0)
        x0 = x0 + torch.ones_like(x0.detach())

        x1 = self.linear3(new_input)
        x1 = torch.relu(x1)
        x1 = self.linear4(x1)

        output = torch.einsum('ln,ln->ln', x0, extra_input) + x1
        return output

class CustomLinearv4(nn.Linear):    
    def __init__(self, in_features, out_features, dynamic_weights: DynamicWeights ):
        
        num_rewards = dynamic_weights.num_rewards
        #assert num_rewards == out_features

        self.dynamic_weights = dynamic_weights

        super(CustomLinearv4, self).__init__(in_features, out_features)
        #super(CustomLinearv4, self).__init__(in_features + num_rewards, out_features, bias=False)
        self._init_weights()

    def _init_weights(self):
       pass

    def forward(self, input):
        weights = self.dynamic_weights.weights
        assert weights is not None
        #input: tensor(batch_size*sequence_length, hidden_dim)
        #weights: tensor(batch_size, num_reward)
        assert input.shape[0] % weights.shape[0] == 0
        extra_input = weights.unsqueeze(1).repeat(1, int(input.shape[0]/weights.shape[0]), 1)
        extra_input = extra_input.view(-1, weights.size(1)).to(input.device)

        #helpful",
        #harmless", 
        output = torch.zeros((input.shape[0], 3))
        first_threshold = 0.5
        second_threshold = 1 - first_threshold
        for i, weight in enumerate(extra_input):
            helpful_score = weight[0]
            if  helpful_score >= first_threshold:
                output[i, :] = torch.tensor([(helpful_score-first_threshold)/second_threshold, 0, (1-helpful_score)/second_threshold])
            else:
                output[i, :] = torch.tensor([0, (first_threshold-helpful_score)/first_threshold, helpful_score/first_threshold])
        #print("logit", output[0])
        return output.to(input.device)

class CustomLinearv5(nn.Linear):   # for router alignment
    def __init__(self, in_features, out_features, dynamic_weights: DynamicWeights ):
        
        num_rewards = dynamic_weights.num_rewards
        assert num_rewards == out_features

        self.dynamic_weights = dynamic_weights
        super(CustomLinearv5, self).__init__(in_features, out_features, bias=True)
        self._init_weights()

    def _init_weights(self):
        self.weight.data = torch.zeros((self.out_features, self.in_features))
        self.bias.data = torch.zeros_like(self.bias.data)
    def register_labels(self, labels):
        self.labels = labels

    def forward(self, input):
        weights = self.dynamic_weights.weights
        labels = self.dynamic_weights.labels

        if weights is not None and labels is not None: 
            #input: tensor(batch_size*sequence_length, hidden_dim)
            #weights: tensor(batch_size, num_reward)
            assert input.shape[0] % weights.shape[0] == 0
            weight_output = weights.unsqueeze(1).repeat(1, int(input.shape[0]/weights.shape[0]), 1)
            weight_output = weight_output.view(-1, weights.size(1)).to(input.device)
            
            router_output = super(CustomLinearv5, self).forward(input.to(torch.float32)) 
            router_output = F.softmax(router_output*3, dim=-1)

            # labels : tensor(batch_size, sequence_length)
            labels = labels.view(-1,1).repeat(1, weights.shape[1])
            # labels : tensor(batch_size* sequence_length, num_reward)


            output = torch.where(labels==-100, weight_output, router_output) 
            #print("final_output: ", output)
            return output
        elif weights is None and labels is None:
            router_output = super(CustomLinearv5, self).forward(input.to(torch.float32)) 
            router_output = F.softmax(router_output*3, dim=-1)
            return router_output
        else:
            raise NotImplementedError


class CustomLinearv6forOneLayer(nn.Linear):   
    def __init__(self, in_features, out_features):
        super(CustomLinearv6forOneLayer, self).__init__(in_features, out_features, bias=True)
        self.hidden_states = None
        self.router_output = None
        self._init_weights()
    def _init_weights(self):
        self.weight.data = torch.zeros((self.out_features, self.in_features))
        self.bias.data = torch.zeros_like(self.bias.data)
    
    def forward(self, input, hidden_states = None):
        #input: tensor(batch_size*sequence_length, hidden_dim)
        #hidden_state: tensor(batch_size*sequence_length, hidden_dim)
        #router_output: tensor(batch_size*sequence_length, out_features)

        if self.router_output is None:
            self.hidden_states = hidden_states
            self.router_output = super(CustomLinearv6forOneLayer, self).forward(hidden_states.to(torch.float32)) 
            return self.router_output
        else: #self.router_output is not None
            return self.router_output

    def clean(self):
        if self.hidden_states is not None or self.router_output is not None:
            del self.hidden_states
            del self.router_output
            self.hidden_states = None
            self.router_output = None


     

class CustomLinearv6forOneAdapter(nn.Linear):  
    def __init__(self, in_features, out_features, dynamic_weights: DynamicWeights ):
        num_rewards = dynamic_weights.num_rewards
        assert num_rewards == out_features

        self.dynamic_weights = dynamic_weights
        super(CustomLinearv6forOneAdapter, self).__init__(in_features, out_features, bias=True)
        self._init_weights()

        #self.

    def _init_weights(self):
        self.weight.data = torch.zeros((self.out_features, self.in_features))
        self.bias.data = torch.zeros_like(self.bias.data)
    def register_labels(self, labels):
        self.labels = labels

    def forward(self, input):
        weights = self.dynamic_weights.weights
        labels = self.dynamic_weights.labels

        if weights is not None and labels is not None: 
            #input: tensor(batch_size*sequence_length, hidden_dim)
            #weights: tensor(batch_size, num_reward)
            assert input.shape[0] % weights.shape[0] == 0
            weight_output = weights.unsqueeze(1).repeat(1, int(input.shape[0]/weights.shape[0]), 1)
            weight_output = weight_output.view(-1, weights.size(1)).to(input.device)
            
            router_output = super(CustomLinearv6forOneAdapter, self).forward(input.to(torch.float32)) 
            router_output = F.softmax(router_output*3, dim=-1)

            # labels : tensor(batch_size, sequence_length)
            labels = labels.view(-1,1).repeat(1, weights.shape[1])
            # labels : tensor(batch_size* sequence_length, num_reward)


            output = torch.where(labels==-100, weight_output, router_output) 
            #print("final_output: ", output)
            return output
        elif weights is None and labels is None:
            router_output = super(CustomLinearv6forOneAdapter, self).forward(input.to(torch.float32)) 
            router_output = F.softmax(router_output*3, dim=-1)
            return router_output
        else:
            raise NotImplementedError


class WeightingRouter(nn.Module): 
    def __init__(self, quantile_list, logit_list, in_features, out_features):
        self.quantile_list = quantile_list
        self.logit_list = logit_list
        self.num_obj= in_features
        self.num_router= out_features
        super(WeightingRouter, self).__init__()
        assert self.num_obj==2
        assert self.num_router==3
        if len(quantile_list) > 1:
            raise NotImplementedError
        assert quantile_list[0].shape[0] == self.num_obj
        assert len(quantile_list) + self.num_obj == self.num_router

    def _init_weights(self):
       pass

    def forward(self, input): #input: weights tensor(batch_size, num_reward)
        weights = input
        assert input is not None
        assert input.shape[1] == self.num_obj
        #input: tensor(batch_size*sequence_length, hidden_dim)
        #weights: tensor(batch_size, num_reward)
        assert input.shape[0] % weights.shape[0] == 0
        extra_input = weights.unsqueeze(1).repeat(1, int(input.shape[0]/weights.shape[0]), 1)
        extra_input = extra_input.view(-1, weights.size(1)).to(input.device)

        #helpful",
        #harmless", 
        output = torch.zeros((input.shape[0], self.num_router))
        first_threshold = self.quantile_list[0][0].item()
        second_threshold = 1 - first_threshold
        for i, weight in enumerate(extra_input):
            helpful_score = weight[0]
            if  helpful_score >= first_threshold:
                output[i, :] = torch.tensor([(helpful_score-first_threshold)/second_threshold, 0, (1-helpful_score)/second_threshold])
            else:
                output[i, :] = torch.tensor([0, (first_threshold-helpful_score)/first_threshold, helpful_score/first_threshold])
        #print("logit", output[0])
        return output.to(input.device)


class WeightingRouter(nn.Module): 
    def __init__(self, quantile_list, in_features, out_features):
        self.quantile_list = quantile_list
        self.num_obj= in_features
        self.num_router= out_features
        super(WeightingRouter, self).__init__()
        assert self.num_obj==2
        assert self.num_router==3
        if len(quantile_list) > 1:
            raise NotImplementedError
        assert quantile_list[0].shape[0] == self.num_obj
        assert len(quantile_list) + self.num_obj == self.num_router

    def _init_weights(self):
       pass

    def forward(self, input): #input: weights tensor(batch_size, num_reward)
        weights = input
        assert input is not None
        assert input.shape[1] == self.num_obj
        #input: tensor(batch_size*sequence_length, hidden_dim)
        #weights: tensor(batch_size, num_reward)
        assert input.shape[0] % weights.shape[0] == 0
        extra_input = weights.unsqueeze(1).repeat(1, int(input.shape[0]/weights.shape[0]), 1)
        extra_input = extra_input.view(-1, weights.size(1)).to(input.device)

        #helpful",
        #harmless", 
        output = torch.zeros((input.shape[0], self.num_router))
        first_threshold = self.quantile_list[0][0].item()
        second_threshold = 1 - first_threshold
        for i, weight in enumerate(extra_input):
            helpful_score = weight[0]
            if  helpful_score >= first_threshold:
                output[i, :] = torch.tensor([(helpful_score-first_threshold)/second_threshold, 0, (1-helpful_score)/second_threshold])
            else:
                output[i, :] = torch.tensor([0, (first_threshold-helpful_score)/first_threshold, helpful_score/first_threshold])
        #print("logit", output[0])
        return output.to(input.device)

class CustomRouterRouter(nn.Module):  
    r"""
        e: num_experts
        n: num_obj
        k: num_router
        h: hidden_state
        WeightingRouter:  MLP(n -> k)    lambda (num_obj),       router logits
        Router: CustomLinearv0(h -> e)    hidden_state,       expert logits

        step0: foward("MLP, n-> bk", WeightingRouter, lambda) -> router_logits_logits

        step1: einsum("khe, h -> ke", List(Router), hidden_state)  -> router_logits

        step2: einsum("ke, k -> e", router_logits, router_logits_logits) -> output_logits   

          batch 
        b: batch_size
        q: seq_len
        'bq': batch_size* seq_len
        step0: foward("MLP, 'bq'n-> 'bq'k", WeightingRouter, lambda) -> router_logits_logits
        step1: einsum("khe, 'bq'h -> k'bq'e", List(Router), hidden_state)  -> router_logits
        step2: einsum("k'bq'e, 'bq'k -> 'bq'e", router_logits, router_logits_logits) -> output_logits  

    """

    def __init__(self, 
                 in_features, #h
                 out_features, #e
                 dynamic_weights: DynamicWeights, 
                 quantile_list = [torch.tensor([0.5, 0.5])], ##List[tensor(num_obj) *（k-num_obj）]
                 ):
        super(CustomRouterRouter, self).__init__()
        
        self.h = in_features
        self.e = out_features
        self.n = dynamic_weights.num_rewards
        self.k = len(quantile_list)+self.n
        #assert num_rewards == out_features   
        self.dynamic_weights = dynamic_weights
        if self.e != self.n:
            raise NotImplementedError
        if len(quantile_list) > 1:
            raise NotImplementedError
        assert self.n == quantile_list[0].shape[0]

        self.router_router = WeightingRouter(quantile_list=quantile_list, in_features=self.n, out_features=self.k)
        self.router_list = nn.ModuleList([CustomLinearv0(in_features, out_features, None) for i in range(len(quantile_list))])
        self.quantile_list = quantile_list
        self._init_weights()

    def _init_weights(self):
       pass

    def forward(self, input):
        t0 = time()

        weights = self.dynamic_weights.weights
        assert weights is not None
        #input: tensor(batch_size*sequence_length, hidden_dim)
        #weights: tensor(batch_size, num_reward)
        #extra_input: tensor(batch_size*sequence_length, num_reward)
        assert input.shape[0] % weights.shape[0] == 0
        extra_input = weights.unsqueeze(1).repeat(1, int(input.shape[0]/weights.shape[0]), 1)
        extra_input = extra_input.view(-1, weights.size(1)).to(input.device)
        
        t1 = time()
        
        r"""step0: foward("MLP, 'bq'n-> 'bq'k", WeightingRouter, lambda) -> router_router_logits"""
        router_router_logits = self.router_router(extra_input)

        t2 = time()

        r"""step1: einsum("khe, 'bq'h -> k'bq'e", List(Router), hidden_state)  -> all_router_logits"""
        all_router_logits = []
        for i in range(self.n):  
            singleobj_router_logits = torch.zeros((input.shape[0], self.e)).to(input.device)
            singleobj_router_logits[:, i] = 1.0
            all_router_logits.append(singleobj_router_logits)
        for each_router in self.router_list: #k-n
            each_router_logits = each_router(input.to(torch.float32))  #tensor(b*q, e)
            each_router_logits = each_router_logits /  each_router_logits.sum(dim=-1, keepdim=True)
            all_router_logits.append(each_router_logits)
        all_router_logits = torch.stack(all_router_logits)

        t3 = time()

        r"""step2: einsum("k'bq'e, 'bq'k -> 'bq'e", router_logits, router_logits_logits) -> output_logits """
        output = torch.einsum("kle, lk -> le", all_router_logits, router_router_logits)

        t4 = time()
        print(f"assert: {t1- t0}")
        print(f"step0: {t2- t1}")
        print(f"step1: {t3- t2}")
        print(f"step2: {t4- t3}")
        
        return output.to(input.device)
   

#from trl import PreTrainedModelWrapper
#from transformers import AutoModelForCausalLM


from src.mola_peft_model_hacked import PeftModel
class ConditionedMOEModel(PeftModel):
    def __init__(self, model):
        self.dynamic_weights = DynamicWeights()
        #print(self.dynamic_weights)
        self.replace_all_router_layers(model)
        #print("children Init")
        #super().__init__(model, peft_config, **config)      
    
    @classmethod
    def from_pretrained(cls, model, model_id, **kwargs):
        instance = super().from_pretrained(model, model_id, **kwargs)
        return instance
    
    @classmethod
    def init(cls, model, CustomLinearType=CustomLinearv1):
        dynamic_weights = DynamicWeights()
        cls.replace_all_router_layers(model, dynamic_weights, CustomLinearType)
        return dynamic_weights


    @classmethod
    def replace_all_router_layers(cls, model, dynamic_weights, CustomLinearType):
        for name, module in model.named_children():
            if "router" in name and isinstance(module, nn.Linear):
                in_features = module.in_features
                out_features = module.out_features
                print("replace...")
                setattr(model, name, CustomLinearType(in_features, out_features, dynamic_weights).to(module.weight.device))
            else:
                cls.replace_all_router_layers(module, dynamic_weights, CustomLinearType)
    
    def forward4or(self, **inputs):
        return self.base_model.model.model(**inputs)
    



class ConditionedMOEModelWithValueHead(nn.Module):
    
    #transformers_parent_class = AutoModelForCausalLM
    lm_head_namings = ["lm_head", "embed_out"]
    supported_args = (
        "summary_dropout_prob",
        "v_head_initializer_range",
        "v_head_init_strategy",
    )
    
    def __init__(self, model, v_heads_init=None, trainable_module=["router"]):
        r"""
        Initializes the model.

        Args:
            model (`transformers.PreTrainedModel`):
                The model to wrap. It should be a causal language model such as GPT2.
                or any model mapped inside the `AutoModelForCausalLM` class.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the `ValueHead` class.
            v_head (List(`MultiValueHead`)):
        """
        super().__init__() 
        self.model = model
        self.trainable_module = trainable_module

        v_head_kwargs = {"v_head_init_strategy": "normal"}
        

        self.dynamic_weights = self.model.dynamic_weights   
        self.num_models = self.dynamic_weights.num_rewards


        if v_heads_init is None:
            self.v_heads = [ValueHead(self.model.config, **v_head_kwargs).to(model.device) for idx in range(self.num_models)]  #'List[ValueHead]'
            self._init_weights(**v_head_kwargs)
        else:
            self.v_heads = [torch.load(v_heads_init[i]).to(model.device)  for i in range(self.num_models)]

        
    
    def _init_weights(self, **kwargs):
        r"""
        Initializes the weights of the value head. The default initialization strategy is random.
        Users can pass a different initialization strategy by passing the `v_head_init_strategy` argument
        when calling `.from_pretrained`. Supported strategies are:
        - `normal`: initializes the weights with a normal distribution.

        Args:
            **kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the `ValueHead` class. These arguments
                can contain the `v_head_init_strategy` argument as well as the `v_head_initializer_range`
                argument.
        """
        initializer_range = kwargs.pop("v_head_initializer_range", 0.2)
        # random init by default
        init_strategy = kwargs.pop("v_head_init_strategy", None)
        if init_strategy is None:
            # do nothing
            pass
        elif init_strategy == "normal":
            for v_head in self.v_heads:
                v_head.summary.weight.data.normal_(mean=0.0, std=initializer_range)
                v_head.summary.bias.data.zero_()
        
    def forward(
        self,
        weights : torch.Tensor, 
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        r"""
        Applies a forward pass to the wrapped model and returns the logits of the value head.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            past_key_values (`tuple(tuple(torch.FloatTensor))`, `optional`):
                Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
                (see `past_key_values` input) to speed up sequential decoding.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the wrapped model.
        """

        kwargs["output_hidden_states"] = True  # this had already been set in the LORA / PEFT examples
        kwargs["past_key_values"] = past_key_values 

        self.dynamic_weights.set_dynamic_weights(weights=weights, batch_size=input_ids.shape[0])
        
        model_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        last_hidden_state = model_output.hidden_states[-1]
        lm_logits = model_output.logits
        #loss = base_model_output.loss

        if last_hidden_state.device != self.v_heads[0].summary.weight.device:
            last_hidden_state = last_hidden_state.to(self.v_heads[0].summary.weight.device)

        #value = self.v_head(last_hidden_state).squeeze(-1)
        value_for_each_adapter = [] #List[tensor(batch, prompt_len)]
        for idx in range(self.num_models):
            value_for_each_adapter.append(self.v_heads[idx](last_hidden_state).squeeze(-1)) 
        #self.v_heads[idx](last_hidden_state[idx]: tensor(batch, prompt_len, 1)
        value = self.value_sum_fn(value_for_each_adapter, weights)

        # force upcast in fp32 if logits are in half-precision
        if lm_logits.dtype != torch.float32:
            lm_logits = lm_logits.float()
    
        return {
            "lm_logits": lm_logits, # tensor(batch, seq_len, vocal_size)
            "value": value,
        }
    
    def value_sum_fn(self, value_for_each_adapter, weights):
        r"""  
           value_for_each_adapter : List[tensor(batch, seq_len)]
           weights: tensor(batch, adapter_num)
        """
        value_tensor = torch.stack(value_for_each_adapter)
        value = torch.einsum('bn,nbq->bq', weights.to(torch.float32), value_tensor[:, :, :])
        return value
    
    def parameters(self, recurse=True):
        base_params = list(self.model.parameters(recurse=recurse))
        vhead_params = []
        for idx in range(self.num_models):
            head_params = list(self.v_heads[idx].parameters(recurse=recurse))
            vhead_params = vhead_params + head_params

        #head_params = list(self.v_head.parameters(recurse=recurse))
        return iter(base_params + vhead_params)
    
    def named_parameters(self, prefix='', recurse=True):
        base_params = list(self.model.named_parameters(prefix=prefix, recurse=recurse))
        vhead_params = []
        for idx in range(self.num_models):
            head_params = list(self.v_heads[idx].named_parameters(prefix=prefix + f'v_head{idx}', recurse=recurse))
            vhead_params = vhead_params + head_params
        return iter(base_params + vhead_params)
    
    def train(self, mode=True, param="router"): 
        self.model.train(mode)
        for v_head in self.v_heads:
            v_head.train(mode)
        if mode == True:
            for name, param in self.model.named_parameters():
                param.requires_grad = False
                for module in self.trainable_module:
                    if module in name:
                        param.requires_grad = True
        elif mode == False:
            for name, param in self.model.named_parameters():
                param.requires_grad = False


    def eval(self):
        self.model.eval()

    def test_grad(self):
        test_name_list = ["router", "v_head"]
        for test_name in test_name_list:
            for name, param in self.named_parameters():
                if test_name in name:
                    print(f"name:{test_name}, requires_grad:{param.requires_grad}, grad:{param.grad}")
                    break

    def generate(self, weights, *args, **kwargs):
        r"""
        A simple wrapper around the `generate` method of the wrapped model.
        Please refer to the [`generate`](https://huggingface.co/docs/transformers/internal/generation_utils)
        method of the wrapped model for more information about the supported arguments.

        Args:
            *args (`list`, *optional*):
                Positional arguments passed to the `generate` method of the wrapped model.
            **kwargs (`dict`, *optional*):
                Keyword arguments passed to the `generate` method of the wrapped model.
        """
        batch_size = kwargs["input_ids"].shape[0]
        self.dynamic_weights.set_dynamic_weights(weights=weights, batch_size=batch_size)
        return self.model.generate(*args, **kwargs)
    
    def state_dict(self, *args, **kwargs):
        r"""
        Returns the state dictionary of the model. We add the state dictionary of the value head
        to the state dictionary of the wrapped model by prepending the key with `v_head.`.
        """
        if not self.is_peft_model:
            pretrained_model_state_dict = self.pretrained_model.state_dict(*args, **kwargs)
        else:
            # if it is a peft model, only save the v_head
            pretrained_model_state_dict = {}

        """
        v_head_state_dict = self.v_head.state_dict(*args, **kwargs)
        for k, v in v_head_state_dict.items():
            pretrained_model_state_dict[f"v_head.{k}"] = v
        """
        for idx in range(self.num_models):
            v_head_state_dict = self.v_heads[idx].state_dict(*args, **kwargs)
            for k, v in v_head_state_dict.items():
                pretrained_model_state_dict[f"v_head{idx}.{k}"] = v
        
        return pretrained_model_state_dict
    
    def save_pretrained(self, save_directory: str): 
        self.model.save_pretrained(save_directory)
        for idx in range(self.num_models):
            torch.save(self.v_heads[idx], os.path.join(save_directory, f"v_head{idx}.pt"))
        #torch.save(self.v_head, os.path.join(save_directory, "v_head.pt"))

    def can_generate(self):
        return True

class WeightSampler:
    def __init__(self, reward_dim=2, uniform_ratio=0.0):
        self.reward_dim = reward_dim
        if reward_dim==3:
            raise NotImplementedError
        self.uniform_ratio = uniform_ratio
        self.step=0
    def sample_from_linear_uniform(self, beg, end):
        k = -0.25
        c = (1- k*(end**2 - beg**2)/2)/ (end - beg)
        #c = -0.1
        #k = (2 * (1 - c * (end - beg))) / (end**2 - beg**2)
        u = np.random.uniform(0, 1)
        a = k / 2
        b = c
        constant = -(u + c * beg + (k / 2) * beg**2)
        x_prime = (-b + np.sqrt(b**2 - 4 * a * constant)) / (2 * a)
        return x_prime

    def sample_from_bimodal_gaussian(self, n_samples, temperature=0.001, beg=-0.366, end=1.366):
        """
        Sample from a bimodal Gaussian-like distribution with peaks at 0 and 1.
        
        Parameters:
        - n_samples: Number of samples to generate.
        - temperature: Controls the width of the peaks. Lower temperature results in samples closer to 0 or 1.
        
        Returns:
        - samples: Array of sampled values.
        """
        # Define the two Gaussians with means at 0 and 1, and variance controlled by temperature
        gaussian_0 = np.random.normal(0, np.sqrt(temperature), n_samples)
        gaussian_1 = np.random.normal(1, np.sqrt(temperature), n_samples)
        uniform = np.random.uniform(beg, end, n_samples)
        #uniform = np.array([self.sample_from_linear_uniform(beg, end) for i in range(n_samples)])

        # Mix the two Gaussians, each with equal probability (0.5 for each peak)
        mix_prob = np.random.rand(n_samples)
        mix_prob2 = np.random.rand(n_samples)

        #samples = np.where(mix_prob < 0.7, gaussian_0, gaussian_1)
        samples = np.where(mix_prob < 0.5, 0.0, 1.0)
        samples = np.where(mix_prob2 < self.uniform_ratio, uniform, samples)
        return samples

    def generate_weights(self, n_samples):
        weights = np.zeros((n_samples, self.reward_dim))
        _lambda = np.round(self.sample_from_bimodal_gaussian(n_samples, beg=0.0, end=1.0), 2)
        #_lambda = np.round(self.sample_from_bimodal_gaussian(n_samples), 2)

        weights[:, 0] = _lambda
        weights[:, 1] = 1 - weights[:, 0]
        weights = torch.tensor(weights)
        weights = [weight for weight in weights]
        
        return weights  # List(tensor)
    def log(self, reward_list, weights):
        pass
        return 
    def update(self):
        self.step = self.step +1
        if self.step==50:
            self.uniform_ratio = 0.3
        if self.step==100:
            self.uniform_ratio = 0.5
        if self.step==150:
            self.uniform_ratio = 0.5







