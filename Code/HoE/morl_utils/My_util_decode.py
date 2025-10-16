import numpy as np
import torch
from torch import nn
from peft import PeftModel, LoraConfig
import copy, inspect, warnings
from collections import UserDict


import os
from peft import PeftModel
  
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

class CasualLMWithValueHeads(nn.Module):
    
    #transformers_parent_class = AutoModelForCausalLM
    lm_head_namings = ["lm_head", "embed_out"]
    supported_args = (
        "summary_dropout_prob",
        "v_head_initializer_range",
        "v_head_init_strategy",
    )
    
    def __init__(self, model, num_rewards=2, v_heads_init=None):
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
        v_head_kwargs = {"v_head_init_strategy": "normal"}
        
        self.num_rewards = num_rewards

        if v_heads_init is None:
            self.v_heads = [ValueHead(self.model.config, **v_head_kwargs).to(model.device) for idx in range(self.num_rewards)]  #'List[ValueHead]'
            self._init_weights(**v_head_kwargs)
        else:
            self.v_heads = [torch.load(v_heads_init[i]).to(model.device)  for i in range(self.num_rewards)]

        
    
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
        value_for_each_obj = [] #List[tensor(batch, prompt_len)]
        for idx in range(self.num_rewards):
            value_for_each_obj.append(self.v_heads[idx](last_hidden_state).squeeze(-1)) 
        #self.v_heads[idx](last_hidden_state[idx]: tensor(batch, prompt_len, 1)

        #value = self.value_sum_fn(value_for_each_obj, weights)

        # force upcast in fp32 if logits are in half-precision
        if lm_logits.dtype != torch.float32:
            lm_logits = lm_logits.float()
    
        return {
            "lm_logits": lm_logits, # tensor(batch, seq_len, vocal_size)
            "value": torch.stack(value_for_each_obj), #List[tensor(batch, seq_len)] -> tensor(num_obj, batch, seq_len)
        }
    
    def value_sum_fn(self, value_for_each_obj, weights):
        r"""  
           value_for_each_obj : List[tensor(batch, seq_len)]
           weights: tensor(batch, adapter_num)
        """
        value_tensor = torch.stack(value_for_each_obj)
        value = torch.einsum('bn,nbq->bq', weights.to(torch.float32), value_tensor[:, :, :])
        return value
    
    def parameters(self, recurse=True):
        base_params = list(self.model.parameters(recurse=recurse))
        vhead_params = []
        for idx in range(self.num_rewards):
            head_params = list(self.v_heads[idx].parameters(recurse=recurse))
            vhead_params = vhead_params + head_params

        #head_params = list(self.v_head.parameters(recurse=recurse))
        return iter(base_params + vhead_params)
    
    def named_parameters(self, prefix='', recurse=True):
        base_params = list(self.model.named_parameters(prefix=prefix, recurse=recurse))
        vhead_params = []
        for idx in range(self.num_rewards):
            head_params = list(self.v_heads[idx].named_parameters(prefix=prefix + f'v_head{idx}', recurse=recurse))
            vhead_params = vhead_params + head_params
        #head_params = list(self.v_head.named_parameters(prefix=prefix + f'v_head', recurse=recurse))
        return iter(base_params + vhead_params)
    
    def train(self, mode=True): 
        self.model.train(mode)
        for v_head in self.v_heads:
            v_head.train(mode)
        if mode == True:
            for name, param in self.model.named_parameters():
                if "router" in name: 
                    param.requires_grad = True
                else:
                    param.requires_grad = False
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

    def generate(self, *args, **kwargs):
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
        #batch_size = kwargs["input_ids"].shape[0]
        #self.dynamic_weights.set_dynamic_weights(weights=weights, batch_size=batch_size)
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
        for idx in range(self.num_rewards):
            v_head_state_dict = self.v_heads[idx].state_dict(*args, **kwargs)
            for k, v in v_head_state_dict.items():
                pretrained_model_state_dict[f"v_head{idx}.{k}"] = v
        
        return pretrained_model_state_dict
    
    def save_pretrained(self, save_directory: str): 
        self.model.save_pretrained(save_directory)
        for idx in range(self.num_rewards):
            torch.save(self.v_heads[idx], os.path.join(save_directory, f"v_head{idx}.pt"))
        #torch.save(self.v_head, os.path.join(save_directory, "v_head.pt"))

    def can_generate(self):
        return True


        

