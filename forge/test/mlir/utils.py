# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Dict, Callable, Tuple
import torch
from torch import nn
import jax.random as random
import jax.numpy as jnp
import flax.linen as fnn


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


def prepare_jax_test(
    model: fnn.Module, test_info: Tuple[Tuple[Callable, Tuple[int, ...], jnp.dtype]]
) -> Tuple[fnn.Module, Tuple[jnp.ndarray, ...]]:
    """
    Prepare JAX test data for a given model and bind variables to the model.

    Args:
        model: The JAX model to test.
        test_info: A tuple of tuples, each containing a function, a shape, and a dtype.

    Returns:
        A tuple of the binded model and the inputs.
    """
    key = random.PRNGKey(0)
    keys = random.split(key, num=len(test_info))
    inputs = []
    for i, (fn_generate_data, shape, dtype) in enumerate(test_info):
        inputs.append(fn_generate_data(keys[i], shape=shape, dtype=dtype))

    variables = model.init(key, *inputs)
    return model.bind(variables), inputs
