import torch

##############################################################
def normalize(x, dim=0):
    min_values, _ = torch.min(x, dim=dim, keepdim=True)
    max_values, _ = torch.max(x, dim=dim, keepdim=True)
    y = (x - min_values) / (max_values - min_values)
    return y


def clamp(x, min_ratio=0, max_ratio=0):
    if len(x.size())==1:
        d = x.size(0)
        min, _ = torch.kthvalue(x, int(d * min_ratio), dim=1)
        max, _ = torch.kthvalue(x, int(d * (1-max_ratio)-1), dim=1)
    else:
        d = x.size(1)
        min, _ = torch.kthvalue(x, int(d * min_ratio), dim=1)
        max, _ = torch.kthvalue(x, int(d * (1-max_ratio)-1), dim=1)
        min=min.unsqueeze(1)
        max=max.unsqueeze(1)
    clamped_x= torch.clamp(x, min, max)
    print("min", min, "max", max)
    return clamped_x

def act(x):
    y = torch.tanh(x)  # x**7; torch.relu(x)
    return y

def task_para_att_merge(flat_task_checks, att_ratio=0.1):
    all_checks = flat_task_checks.clone()
    n, d = all_checks.shape  
    all_checks_abs = clamp(torch.abs(all_checks), min_ratio=0.01, max_ratio=0.01)
    print("#clamp finish")
    clamped_all_checks = torch.sign(all_checks)*all_checks_abs
    print("#clip finish")
    self_att = normalize(all_checks_abs, 1)**2
    print("#normalize finish")
    self_att_act = torch.exp(n*self_att)
    print("#exp finish")
    cross_att = all_checks * torch.sum(all_checks, dim=0)
    print("#sum finish")
    cross_att_act = act(cross_att)
    print("#act finish")
    task_att = self_att_act * cross_att_act
    print("#* finish")
    del self_att_act, cross_att_act, cross_att, all_checks, self_att, all_checks_abs, flat_task_checks
    scale = normalize(clamp(task_att, 1-att_ratio, 0), dim=1)
    del task_att
    print("#scale finish")
    tvs = clamped_all_checks
    merged_tv = torch.sum(tvs * scale, dim=0) / torch.clamp(torch.sum(scale, dim=0), min=1e-12)
    print("#all finish")
    return merged_tv


##############################################################
#sft_model_path = " yourpath/model/summary_sft_llama2"
#sft_model_path = " yourpath/model/assistant_sft_llama"
sft_model_path = "/yourpath/public/model"
model0_path  = " yourpath/model/ppo_models/helpful_ppo_llama3.1"
model1_path  = " yourpath/model/ppo_models/harmless_ppo_llama3.1"
#model2_path  = " yourpath/model/ppo_models/humor_ppo_llama"
model2_path  = " yourpath/workspace/RiC/RiC/ppo/logs_ppo_assistant/train4moe_humor_princeton/epoch_2_batch_150"
weightings = [1.0, 1.0, 1.0]

device = "cuda:0"
import torch
from transformers import AutoModelForCausalLM
sft_model = AutoModelForCausalLM.from_pretrained(
    sft_model_path,
    torch_dtype=torch.bfloat16,
    device_map=device,
)
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model

peft_model0 = PeftModel.from_pretrained(sft_model, model0_path)
peft_model_dict0 = peft_model0.state_dict()
peft_model1 = PeftModel.from_pretrained(sft_model, model1_path)
peft_model_dict1 = peft_model1.state_dict()
peft_model2 = PeftModel.from_pretrained(sft_model, model2_path)
peft_model_dict2 = peft_model2.state_dict()

lora_config = LoraConfig(
    r=64, 
    lora_alpha=128, 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["down_proj","v_proj","k_proj","up_proj","o_proj","gate_proj","q_proj"],
)
fusion_peft_model  = get_peft_model(sft_model, lora_config)
fusion_peft_model_dict = fusion_peft_model.state_dict()

##############################################################

def low_rank_approximation(matrix, rank):
    U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
    #print(U, S, Vh)
    U_r = U[:, :rank]   
    S_r = S[:rank]      
    Vh_r = Vh[:rank, :]   

    A = U_r @ torch.diag(S_r.sqrt())  # A = U_r * sqrt(S_r)
    B = torch.diag(S_r.sqrt()) @ Vh_r  # B = sqrt(S_r) * Vh_r
    return A, B
    return A, B, U_r, S_r, Vh_r

##############################################################

import torch
def get_full_param_dict(peft_model_dict):
    flat_param_dict = {}
    for loraA_name, param in peft_model_dict.items():
        if 'lora_A' in loraA_name: 
            loraB_name = 'lora_B'.join(loraA_name.split('lora_A'))
            full_param = torch.einsum("ri,or->oi", peft_model_dict[loraA_name], peft_model_dict[loraB_name]).to(torch.float32)
            flat_param_dict[loraA_name] = full_param.detach().clone().to("cpu")
    return flat_param_dict

from collections import OrderedDict
import copy
def state_dict_to_vector(state_dict, weighting=1.0):
    shared_state_dict = copy.deepcopy(state_dict)
    sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
    return torch.nn.utils.parameters_to_vector(
        [value.reshape(-1)*weighting for key, value in sorted_shared_state_dict.items()]
    )
def vector_to_state_dict(vector, state_dict):
    # create a reference dict to define the order of the vector
    reference_dict = copy.deepcopy(state_dict)
    sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))
    # create a shared state dict using the refence dict
    torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())
    return sorted_reference_dict

with torch.no_grad():
    peft_dict0 = get_full_param_dict(peft_model_dict0)
    peft_dict1 = get_full_param_dict(peft_model_dict1)
    peft_dict2 = get_full_param_dict(peft_model_dict2)
    peft_vector0 = state_dict_to_vector(peft_dict0, weightings[0])
    peft_vector1 = state_dict_to_vector(peft_dict1, weightings[1])
    peft_vector2 = state_dict_to_vector(peft_dict2, weightings[2])
    print("state_dict_to_vector finish")
    flat_tvs  =  torch.stack([peft_vector0, peft_vector1, peft_vector2], dim=0)
    print(flat_tvs.shape)
    del peft_vector0, peft_vector1, peft_vector2, peft_model_dict0, peft_model_dict1, peft_model_dict2, peft_dict1, peft_dict2
    merged_tv= task_para_att_merge(flat_tvs, 0.9)
    print("task_para_att_merge finish")
    print(merged_tv.shape)

    fusion_full_param_dict = vector_to_state_dict(merged_tv.to(device), peft_dict0)
    del merged_tv
    print("vector_to_state_dict finish")
    for loraA_name, fusion_param in fusion_full_param_dict.items():
        if 'lora_A' in loraA_name: 
            loraB_name = 'lora_B'.join(loraA_name.split('lora_A'))
            
            loraA_param, loraB_param = low_rank_approximation(fusion_param, 64) #oi->or,ri
            print(loraA_param.shape, loraB_param.shape)

            reconstructed = torch.einsum("or,ri->oi", loraA_param, loraB_param)
            print(torch.mean((reconstructed - fusion_param).abs()))
            assert reconstructed.shape == fusion_param.shape

            fusion_peft_model_dict[loraB_name].copy_(loraA_param.to(device))
            fusion_peft_model_dict[loraA_name].copy_(loraB_param.to(device))

            del reconstructed, loraA_param, loraB_param
        else: 
            raise NotImplementedError

    fusion_peft_model.load_state_dict(fusion_peft_model_dict)
    fusion_peft_model.save_pretrained(" yourpath/workspace/pcbmerging/3sft_test2_helpharmhumor_pcbmerging")

    #state_dict_to_vector(full_param_dict0).shape

