# This files implements the utils that process the gradients

from copy import deepcopy
import numpy as np
import torch
from typing import Iterable, List, Optional, Tuple

def cal_bpograd(grad_list) -> None:
    num_tasks = len(grad_list)
    grad_vec = torch.cat(
        list(
            map(lambda x: torch.nn.utils.parameters_to_vector(x).unsqueeze(0), grad_list)
        ),
        dim=0,
    )  # num_tasks * \theta dim

    regularized_grad = bpograd(grad_vec, num_tasks)
    return regularized_grad

def bpograd(grad_vec, num_tasks, bpo_c=0):
    """
    grad_vec: [num_tasks, dim]
    """
    grads = grad_vec

    GG = grads.mm(grads.t()).cpu()
    scale = (torch.diag(GG)+1e-4).sqrt().mean()
    GG = GG / scale.pow(2)
    Gg = GG.mean(1, keepdims=True)
    gg = Gg.mean(0, keepdims=True)

    w = torch.zeros(num_tasks, 1, requires_grad=True)
    w_opt = torch.optim.SGD([w], lr=25, momentum=0.5)

    c = (gg+1e-4).sqrt() * bpo_c

    w_best = None
    obj_best = np.inf
    for i in range(21):
        w_opt.zero_grad()
        ww = torch.softmax(w, 0)
        obj = ww.t().mm(Gg) + c * (ww.t().mm(GG).mm(ww) + 1e-4).sqrt()
        if obj.item() < obj_best:
            obj_best = obj.item()
            w_best = w.clone()
        if i < 20:
            obj.backward()
            w_opt.step()

    ww = torch.softmax(w_best, 0)
    gw_norm = (ww.t().mm(GG).mm(ww)+1e-4).sqrt()

    lmbda = c.view(-1) / (gw_norm+1e-4)
    g = ((1/num_tasks + ww * lmbda).view(
        -1, 1).to(grads.device) * grads).sum(0) / (1 + bpo_c**2)
    return g
    
def apply_vector_grad_to_parameters(
    parameters: Iterable[torch.Tensor], vec: torch.Tensor, accumulate: bool = False
):
    """Apply vector gradients to the parameters

    Args:
        vec (TensorType): a single vector represents the gradients of a model.
        parameters (Iterable[TensorType]): an iterator of Tensors that are the
            parameters of a model.
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError(
            "expected torch.Tensor, but got: {}".format(torch.typename(vec))
        )

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:

        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old grad of the parameter
        if accumulate:
            param.grad = (
                param.grad + vec[pointer : pointer + num_param].view_as(param).data
            )
        else:
            param.grad = vec[pointer : pointer + num_param].view_as(param).data

        # Increment the pointer
        pointer += num_param

def get_network_grad(networks):
    policy_grad = [p._grad for p in networks.parameters()]
    tem_grad = deepcopy(policy_grad)
    return tem_grad