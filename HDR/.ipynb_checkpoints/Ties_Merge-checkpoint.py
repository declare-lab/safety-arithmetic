import sys
import os, copy
import torch
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import OrderedDict
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def state_dict_to_vector(state_dict, remove_keys=[], included_layers=None):
    # copying the dictionary
    shared_state_dict = copy.deepcopy(state_dict)

    for key in remove_keys:
        if key in shared_state_dict:
            del shared_state_dict[key]

    sorted_shared_state_dict = shared_state_dict
    start_ind = 0
    tot_len = 0
    flag = -1
    for key, value in sorted_shared_state_dict.items():
      temp_flag = -1
      for layer in included_layers:
        if str(layer) in key.split('.'):
          temp_flag = 1
      if temp_flag == 1:
        flag = 0
        tot_len += len(value.reshape(-1))
        
      else: 
        if flag==-1:
          start_ind += len(value.reshape(-1))

    return torch.nn.utils.parameters_to_vector(
        [value.reshape(-1) for key, value in sorted_shared_state_dict.items()]), start_ind, start_ind + tot_len


def vector_to_state_dict(vector, state_dict, remove_keys=[]):
    # create a reference dict to define the order of the vector
    reference_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in reference_dict:
            del reference_dict[key]

    sorted_reference_dict = reference_dict
    # create a shared state dict using the refence dict
    torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())  # convert 1D Tensor into list of 
    if "transformer.shared.weight" in sorted_reference_dict:
        for key in remove_keys:
            sorted_reference_dict[key] = sorted_reference_dict[
                "transformer.shared.weight"
            ]
    return sorted_reference_dict


def add_ptm_to_tv(tv_dict, ptm_dict):
    assert set(tv_dict.keys()) == set(
        ptm_dict.keys()
    ), "Differing parameter names in models."
    final_dict = copy.deepcopy(tv_dict)
    for k, v in ptm_dict.items():
        final_dict[k] = tv_dict[k] + v
    return final_dict


def check_parameterNamesMatch(checkpoints):
    parameter_names = set(checkpoints[0].keys())

    if len(checkpoints) >= 2:
        for checkpoint in checkpoints[1:]:
            current_parameterNames = set(checkpoint.keys())
            if current_parameterNames != parameter_names:
                raise ValueError(
                    "Differing parameter names in models. "
                    f"The different parameters are {parameter_names.symmetric_difference(current_parameterNames)}"
                )

def check_state_dicts_equal(state_dict1, state_dict2):
    if set(state_dict1.keys()) != set(state_dict2.keys()):
        return False

    for key in state_dict1.keys():
        if not torch.equal(state_dict1[key], state_dict2[key]):
            return False

    return True

# %%
"""
#### Load base model and the Finetuned Models to merge
"""

# %%
# For creating the difference between the two models : -
def create_diff_task_vector(pretrained_model, finetuned_model, included_layers=None):
    ft_checks = finetuned_model#[finetuned_model]#, model_mnli, model_sst2]
    ptm_check = pretrained_model

    remove_keys = [
      "transformer.encoder.embed_tokens.weight",
      "transformer.decoder.embed_tokens.weight",
  ]

    print(f"Flattening out Checkpoints")
    flat_ft, start_ft, end_ft = state_dict_to_vector(ft_checks, remove_keys, included_layers)
    flat_ptm, start_ptm, end_ptm = state_dict_to_vector(ptm_check, remove_keys, included_layers)

  # Creating Task vectors
    tv_flat_checks = flat_ft - flat_ptm
    return tv_flat_checks, flat_ft, ptm_check, remove_keys, start_ft, end_ft, start_ptm, end_ptm

"""
#### Merge Utils
"""


def topk_values_mask(M, K=0.7, return_mask=False):
    if K > 1:
        K /= 100

    original_shape = M.shape
    if M.dim() == 1:
        M = M.unsqueeze(0)

    n, d = M.shape

    k = int(d * K)
    k = d - k  # Keep top k elements instead of bottom k elements
    print('Finding the kth value')
    kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)
    # Create a mask tensor with True for the top k elements in each row
    mask = M.abs() >= kth_values # done to consider the top k elements

    final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask # final mask has been created

    if return_mask:
        return M * final_mask, final_mask.float().mean(dim=1), final_mask
    return M * final_mask, final_mask.float().mean(dim=1)


def resolve_zero_signs(sign_to_mult, method="majority"):
    majority_sign = torch.sign(sign_to_mult.sum()) # finding the sign after summing all the parameters

    # if there is multiple rows then this is used
    if method == "majority":
        sign_to_mult[sign_to_mult == 0] = majority_sign
    elif method == "minority":
        sign_to_mult[sign_to_mult == 0] = -1 * majority_sign
    return sign_to_mult


def resolve_sign(Tensor):

    sign_to_mult = torch.sign(Tensor.sum(dim=0)) # finding the sign of the individual parameters
    sign_to_mult = resolve_zero_signs(sign_to_mult, "majority")
    return sign_to_mult


def disjoint_merge(Tensor, merge_func, sign_to_mult):

    merge_func = merge_func.split("-")[-1]

    # If sign is provided then we select the corresponding entries and aggregate.
    if sign_to_mult is not None:
        rows_to_keep = torch.where(
            sign_to_mult.unsqueeze(0) > 0, Tensor > 0, Tensor < 0
        )
        # based on the above condition i.e. the value at that position in that tensor should not be 0 and the sign is also +ve

        selected_entries = Tensor * rows_to_keep
        # selected_entries contain the only those values filtered by rows_to_keep
    # Else we select all non-zero entries and aggregate.
    else:
        rows_to_keep = Tensor != 0
        selected_entries = Tensor * rows_to_keep

    if merge_func == "mean":
        non_zero_counts = (selected_entries != 0).sum(dim=0).float()
        disjoint_aggs = torch.sum(selected_entries, dim=0) / torch.clamp(
            non_zero_counts, min=1
        )
    elif merge_func == "sum":
        disjoint_aggs = torch.sum(selected_entries, dim=0)
    elif merge_func == "max":
        disjoint_aggs = selected_entries.abs().max(dim=0)[0]
        disjoint_aggs *= sign_to_mult
    else:
        raise ValueError(f"Merge method {merge_func} is not defined.")

    return disjoint_aggs


def ties_merging(
    flat_task_checks,
    reset_thresh=None,
    merge_func="",
):
    all_checks = flat_task_checks.clone()
    updated_checks, *_ = topk_values_mask(
        all_checks, K=reset_thresh, return_mask=False
    )
    # updated_checks contain the original parameters on which the masking has been applied keeping only the top k% of the total paramters
    print(f"RESOLVING SIGN")
    # Finding the final sign
    final_signs = resolve_sign(updated_checks)

    assert final_signs is not None

    print(f"Disjoint AGGREGATION: {merge_func}")
    merged_tv = disjoint_merge(updated_checks, merge_func, final_signs)
#     print(merged_tv)
    return merged_tv
# %%
"""
#### Loading the models and then TIES Merging
"""
# %%
def ties_merging_Model(pretrained_checkpoint=None, finetuned_checkpoint=None, included_layers=None):
    pretrained_model = pretrained_checkpoint.state_dict()
    #Loading harmful model
    finetuned_model = finetuned_checkpoint.state_dict()
    tv_flat_checks, flat_ptm, ptm_check, remove_keys, start_ft, end_ft, start_ptm, end_ptm  = create_diff_task_vector(pretrained_model, finetuned_model, included_layers)

    if len(included_layers)==0:
        start_ft = 0
        start_ptm = 0

    return tv_flat_checks, flat_ptm, ptm_check, remove_keys, start_ft, end_ft, start_ptm, end_ptm

def Perform_Ties_Merge(tv_flat_checks, flat_ptm, ptm_check, remove_keys, K, lamda,start_ft, end_ft):

    corres_layer = tv_flat_checks[start_ft:end_ft]
    corres_layer = corres_layer.unsqueeze(0)
    merge_func = "dis-mean"
    print("1 Done")
  # tv_flat_checks difference between the weights of Finetuned Model and the Original Model
    merged_tv = ties_merging(
      corres_layer,
      reset_thresh=K,
      merge_func=merge_func,
  )
  # add back the PTM to the flat merged task vector
  # Lamda is the parameter
    print("Merged_Tv has been obtained : -")

    temp = tv_flat_checks.clone()
    temp[start_ft:end_ft] = merged_tv
    merged_check = flat_ptm - lamda * temp


    merged_state_dict = vector_to_state_dict(
      merged_check, ptm_check
  )

    return merged_state_dict, temp, merged_tv