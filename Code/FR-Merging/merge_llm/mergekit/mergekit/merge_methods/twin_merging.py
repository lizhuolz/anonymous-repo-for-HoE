# Copyright (C) 2024 Charles O. Goddard
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This software is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import copy

import torch
from pydantic import BaseModel
from typing_extensions import Literal

from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference,ModelPath
from mergekit.graph import Task
from mergekit.io.tasks import GatherTensors
from mergekit.merge_methods.base import ConfigParameterDef, MergeMethod
from mergekit.sparsify import SparsificationMethod, sparsify,magnitude_outliers


class ConsensusMethod(str, Enum):
    count = "count"
    sum = "sum"

class MergeTwin(MergeMethod, BaseModel, frozen=True):
    consensus_method: Optional[ConsensusMethod]
    sparsification_method: Optional[SparsificationMethod]
    default_normalize: bool
    lambda_type: Optional[str] = "Constant"
    task_name:str = 'lm'

    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="int8_mask", required=False, default_value=False),
            ConfigParameterDef(
                name="normalize", required=False, default_value=self.default_normalize
            ),
        ]

    def tensor_parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="weight", required=True),
            ConfigParameterDef(name="density", required=False, default_value=1.0),
            ConfigParameterDef(name="lambda", required=False, default_value=1.0),
            ConfigParameterDef(name="window_size", required=False, default_value=0),
            ConfigParameterDef(name="rescale", required=False, default_value=1),
        ]

    def make_task(
        self,
        output_weight: WeightInfo,
        tensors: GatherTensors,
        base_model: Optional[ModelReference],
        parameters: ImmutableMap[str, Any],
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
    ) -> Task:
        if self.task_name=='lm':
            task_model=ModelReference(model=ModelPath(path='WizardLMTeam/WizardLM-13B-V1.2', revision=None), lora=None)
        elif self.task_name=='math':
            task_model=ModelReference(model=ModelPath(path='vanillaOVO/WizardMath-13B-V1.0', revision=None), lora=None)
        elif self.task_name == 'code':
            task_model=ModelReference(model=ModelPath(path='layoric/llama-2-13b-code-alpaca', revision=None), lora=None)
        return TaskFourier(
            method=self,
            tensors=tensors,
            base_model=base_model,
            tensor_parameters=tensor_parameters,
            int8_mask=parameters["int8_mask"],
            normalize=parameters["normalize"],
            out_tensor_name=output_weight.name,
            task_model=task_model
        )

def get_mask(
    delta: torch.Tensor,
    method: Literal["sum", "count"] = "sum",
    mask_dtype: Optional[torch.dtype] = None,
):
    """Returns a mask determining which delta vectors should be merged
    into the final model.

    For the methodology described in the TIES paper use 'sum'. For a
    simpler naive count of signs, use 'count'."""
    if mask_dtype is None:
        mask_dtype = delta.dtype

    sign = delta.sign().to(mask_dtype)

    if method == "sum":
        sign_weight = (sign * delta.abs()).sum(dim=0)
        majority_sign = (sign_weight >= 0).to(mask_dtype) * 2 - 1
        del sign_weight
    elif method == "count":
        majority_sign = (sign.sum(dim=0) >= 0).to(mask_dtype) * 2 - 1
    else:
        raise RuntimeError(f'Unimplemented mask method "{method}"')

    return sign == majority_sign

def get_task_vectors(
    parameter_name: str,
    base_model: ModelReference,
    tensors: ImmutableMap[ModelReference, torch.Tensor],
    tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
    task_model: ModelReference,
) -> Tuple[List[Dict[str, Any]], torch.Tensor]:
    keys = list(tensors.keys())
    base = tensors[base_model]
    res = []
    for model in keys:
        if model == base_model:
            continue
        x = tensors[model].to(base.dtype)
        if x.shape != base.shape:
            if "lm_head" in parameter_name or "embed_tokens" in parameter_name:
                x = x[: base.shape[0], : base.shape[1]]
                logging.warning(f"Using submatrix of {model}:{parameter_name}")
            else:
                logging.warning(
                    f"skipping {model}:{parameter_name} due to size mismatch"
                )
                continue
        delta = x - base
        if model == task_model:
            spec = copy.deepcopy(delta)
        del x
        del tensors[model]

        d = {}
        d["model"] = model
        d["delta"] = delta
        for p in tensor_parameters[model]:
            d[p] = tensor_parameters[model][p]
        res.append(d)
    return res, base,spec


class TaskFourier(Task[torch.Tensor]):
    method: MergeTwin
    tensors: GatherTensors
    base_model: ModelReference
    out_tensor_name: str
    tensor_parameters: ImmutableMap[ModelReference, Any]
    int8_mask: bool
    normalize: bool
    task_model:ModelReference

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.tensors}

    def execute(
        self,
        tensors: Dict[ModelReference, torch.Tensor],
        **_kwargs,
    ) -> torch.Tensor:
        # collect task vectors
        tvs, base,spec = get_task_vectors(
            self.out_tensor_name,
            self.base_model,
            tensors,
            tensor_parameters=self.tensor_parameters.data,
            task_model=self.task_model
        )
        if not tvs:
            return base
            
        # sparsify
        if self.method.sparsification_method:
            for tv_info in tvs:
                kwargs = {}
                if "gamma" in tv_info:
                    kwargs["gamma"] = tv_info["gamma"]

                if "epsilon" in tv_info:
                    kwargs["epsilon"] = tv_info["epsilon"]
                tv_info["delta"] = sparsify(
                    tv_info["delta"],
                    density=tv_info["density"],
                    method=self.method.sparsification_method,
                    epsilon = tv_info["window_size"]/2,
                    rescale = tv_info["rescale"],
                    **kwargs
                )
        spec=sparsify(
                    tensor=spec,
                    density=0.01,
                    method=SparsificationMethod.svd,
                    rescale = False,
                    gamma=0.00,
                    epsilon=1
                )
        spec=spec*0.2
        deltas = torch.stack([tv["delta"] for tv in tvs], dim=0)
        device=deltas.device
        weights = torch.tensor(
            [tv["weight"] for tv in tvs], dtype=deltas.dtype, device=deltas.device
        )
        lambda_factor = tvs[0]["lambda"]
        while len(deltas.shape) > len(weights.shape):
            weights.unsqueeze_(-1)

        weighted_deltas = deltas * weights

        # get sign consensus and mix deltas
        if self.method.consensus_method:
            mask_dtype = torch.int8 if self.int8_mask else base.dtype
            mask = get_mask(
                weighted_deltas,
                method=self.method.consensus_method,
                mask_dtype=mask_dtype,
            )

            if self.method.lambda_type == "ada_row":
                
                rescal_val = weighted_deltas.ne(0).sum(dim=-1, keepdim=True)/(mask.ne(0).sum(dim=-1, keepdim=True)+1e-7)

                assert not torch.isnan(rescal_val).any()
                assert not torch.isinf(rescal_val).any()

                weighted_deltas *= rescal_val

                assert not torch.isnan(weighted_deltas).any()
                assert not torch.isinf(weighted_deltas).any()
            if self.method.lambda_type == "ada_layer":
                rescal_val = weighted_deltas.ne(0).sum(dim=-1, keepdim=True).sum(dim=1, keepdim=True)/(mask.ne(0).sum(dim=-1, keepdim=True).sum(dim=1, keepdim=True)+1e-7)

                assert not torch.isnan(rescal_val).any()
                assert not torch.isinf(rescal_val).any()
                weighted_deltas2 = weighted_deltas.clone()

                weighted_deltas *= rescal_val
                assert not torch.isnan(weighted_deltas).any()
                assert not torch.isinf(weighted_deltas).any()

            if self.method.lambda_type == "ada_row_inv":
                rescal_val = mask.ne(0).sum(dim=-1, keepdim=True)/(weighted_deltas.ne(0).sum(dim=-1, keepdim=True)+1e-7)

                assert not torch.isnan(rescal_val).any()
                assert not torch.isinf(rescal_val).any()

                weighted_deltas *= rescal_val

                assert not torch.isnan(weighted_deltas).any()
                assert not torch.isinf(weighted_deltas).any()
            mixed_delta = (weighted_deltas * mask).sum(dim=0)
            divisor = (weights * mask).sum(dim=0)
            divisor[divisor == 0] = 1
        else:
            mixed_delta = weighted_deltas.sum(dim=0)
            divisor = weights.sum(dim=0)
            divisor[divisor.abs() < 1e-8] = 1

        if self.normalize:
            mixed_delta /= divisor

        if self.method.lambda_type == "Constant":
            mixed_delta = lambda_factor*mixed_delta

        return (base + mixed_delta+spec).to(base.dtype)

def cal_low_and_high_fft_for_tensor(weight,high_ratio):
    freq_domain = torch.fft.fft2(weight)
    # freq_domain_shifted = torch.fft.fftshift(freq_domain)
    test_domain_shifted = torch.fft.ifft2(freq_domain)
    H, W = freq_domain.shape
    low_radius = min(H, W) // 10  
    high_radius = min(H, W)*high_ratio
    center = (H // 2, W // 2)
    Y, X = torch.meshgrid(torch.arange(H), torch.arange(W))
    dist_from_center = torch.sqrt((X - center[1]) ** 2 + (Y - center[0]) ** 2)
    low_freq_mask = (dist_from_center <= low_radius).float().to(weight.device)
    high_freq_mask = (dist_from_center > high_radius).float().to(weight.device)
    band_pass_mask = ((dist_from_center >= low_radius) & (dist_from_center <= high_radius)).float().to(weight.device)
    
    low_freq_component = freq_domain * low_freq_mask
    high_freq_component = freq_domain * high_freq_mask
    band_pass_component = freq_domain * band_pass_mask

    low_freq_spatial = torch.fft.ifft2(low_freq_component).real
    high_freq_spatial = torch.fft.ifft2(high_freq_component).real
    band_pass_spatial = torch.fft.ifft2(band_pass_component).real

    return low_freq_spatial,high_freq_spatial,band_pass_spatial