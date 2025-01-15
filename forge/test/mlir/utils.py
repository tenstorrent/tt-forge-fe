from typing import Dict, Callable, Any
import torch
from torch import nn

def get_param_grads(named_params: Callable[[], Dict[str, nn.Parameter]]) -> Dict[str, torch.Tensor]:
    grads = {}
    for name, param in named_params():
        if param.grad is not None:
            grads[name] = param.grad.detach().clone()
    return grads


def copy_params(src: nn.Module, dst: nn.Module) -> None:
    state_dict = src.state_dict()
    for name, param in dst.named_parameters():
        if name in state_dict:
            param.data = state_dict[name].data.detach().clone()
    dst.load_state_dict(state_dict)