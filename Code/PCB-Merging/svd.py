import torch
#device = "cuda:0"
device = "cpu"
from transformers import AutoModelForCausalLM
base_model = AutoModelForCausalLM.from_pretrained(
        "yourpath/model/llama-2-7b-chat-hf",
        device_map = device,
        #torch_dtype=torch.float16,
    )
object_model = AutoModelForCausalLM.from_pretrained(
        #"yourpath/model/metamath-7b-v1.0",
        "yourpath/model/CodeLlama-7b-hf",
        device_map = device,
        #torch_dtype=torch.float16,
    )



from peft import get_peft_model, LoraConfig
r = 1024
lora_config = LoraConfig(
    r=r, 
    lora_alpha=r*2, 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["down_proj","v_proj","k_proj","up_proj","o_proj","gate_proj","q_proj"],
)

base_model_dict = base_model.state_dict()
object_model_dict = object_model.state_dict()
peft_model  = get_peft_model(base_model, lora_config)
peft_model_dict = peft_model.state_dict()


import torch
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


def task_para_att_merge(all_checks, min_ratio=0.1, max_ratio=0.0):
    all_checks_abs = clamp(torch.abs(all_checks), min_ratio=min_ratio, max_ratio=max_ratio)
    clamped_all_checks = torch.sign(all_checks)*all_checks_abs
    return clamped_all_checks


def clamp(x, min_ratio=0, max_ratio=0):
    assert len(x.shape) == 1
    d = x.size(0)
    min, _ = torch.kthvalue(x, int(d * min_ratio), dim=0)
    max, _ = torch.kthvalue(x, int(d * (1-max_ratio)), dim=0)
    print("min", min, "max", max)
    clamped_x = torch.clamp(x, min, max)
    clamped_x[clamped_x <= min + 1e-5] = 0.0
    return clamped_x


import copy
from collections import OrderedDict
def state_dict_to_vector(state_dict):
    shared_state_dict = copy.deepcopy(state_dict)
    sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
    return torch.nn.utils.parameters_to_vector(
        [value.reshape(-1).to("cpu") for key, value in sorted_shared_state_dict.items()]
    )
def vector_to_state_dict(vector, state_dict):
    # create a reference dict to define the order of the vector
    reference_dict = copy.deepcopy(state_dict)
    sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))
    # create a shared state dict using the refence dict
    torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())
    return sorted_reference_dict

def filter(model_dict):
    keys_to_remove = [k for k in model_dict if "_proj" not in k]
    #           
    for key in keys_to_remove:
        model_dict.pop(key)
    return model_dict

    
base_model_dict = filter(base_model_dict)
object_model_dict = filter(object_model_dict)

base_model_vector = state_dict_to_vector(base_model_dict)
del base_model_dict
object_model_vector = state_dict_to_vector(object_model_dict)
delta_objective_vector = (object_model_vector - base_model_vector) / 2 
del object_model_vector, base_model_vector

delta_objective_vector = task_para_att_merge(delta_objective_vector, min_ratio=0.2, max_ratio=0.0)
delta_dict = vector_to_state_dict(delta_objective_vector, object_model_dict)
del delta_objective_vector, object_model_dict


for loraA_name, fusion_param in peft_model_dict.items():
    if 'lora_A' in loraA_name: 
        loraB_name = 'lora_B'.join(loraA_name.split('lora_A'))
        base_name = loraA_name.split("base_model.model.")[-1]
        base_name = "".join(base_name.split("lora_A.default."))
        print(base_name)
        
        loraA_param, loraB_param = low_rank_approximation(delta_dict[base_name], r) #oi->or,ri
        print(loraA_param.shape, loraB_param.shape)

        reconstructed = torch.einsum("or,ri->oi", loraA_param, loraB_param)
        print(torch.mean((reconstructed - delta_dict[base_name]).abs()))
        assert reconstructed.shape == delta_dict[base_name].shape

        peft_model_dict[loraB_name].copy_(loraA_param.to(device))
        peft_model_dict[loraA_name].copy_(loraB_param.to(device))

        del reconstructed, loraA_param, loraB_param

peft_model.load_state_dict(peft_model_dict)
peft_model.save_pretrained(f" yourpath/workspace/pcbmerging/codellama_to_llama-2-7b-chat-hf_{r}_clamp.2")


