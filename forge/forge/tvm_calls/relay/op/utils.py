# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import math
import numpy as np
from tvm.relay.dataflow_pattern import *

from loguru import logger

def is_unsqueeze(call):
    input_shape = call.args[0].checked_type.shape
    output_shape = call.checked_type.shape

    remove_one_from_output_shape = [x for x in output_shape if x != 1]

    joint_size = min(len(input_shape), len(remove_one_from_output_shape))
    
    superfluous_reshape = all([input_shape[i] == remove_one_from_output_shape[i] for i in range(-1, -1*joint_size - 1, -1)])

    if superfluous_reshape and len(input_shape) < len(output_shape):
        return True
        
    return False

def is_squeeze(call):
    input_shape = call.args[0].checked_type.shape
    output_shape = call.checked_type.shape

    joint_size = min(len(input_shape), len(output_shape))
    
    superfluous_reshape = all([input_shape[i] == output_shape[i] for i in range(-1, -1*joint_size - 1, -1)])

    if superfluous_reshape and len(input_shape) > len(output_shape):
        return True
        
    return False

def is_superfluous_reshape(call):
    assert call.op.name == "reshape"
    input_shape = call.args[0].checked_type.shape
    output_shape = call.checked_type.shape

    joint_size = min(len(input_shape), len(output_shape))

    superfluous_reshape = all([input_shape[i] == output_shape[i] for i in range(-1, -1*joint_size - 1, -1)])

    if len(input_shape) > len(output_shape):
        extra_dims = len(input_shape) - len(output_shape)
        if all([extra_dim == 1 for extra_dim in input_shape[:extra_dims]]):
            superfluous_reshape = superfluous_reshape and True
    elif len(output_shape) > len(input_shape):
        extra_dims = len(output_shape) - len(input_shape)
        if all([extra_dim == 1 for extra_dim in output_shape[:extra_dims]]):
            superfluous_reshape = superfluous_reshape and True
    
    return superfluous_reshape

def is_reshape_hslice(call):
    r_input_shape = call.args[0].type_args[0].shape
    r_newshape = call.args[0].checked_type.shape

    if (not (len(r_newshape) == 3 or (len(r_newshape) == 4 and r_newshape[0].value == 1)) 
    or len(r_input_shape) < 2
    or not (r_input_shape[-2].value == r_newshape[-3].value) 
    or is_superfluous_reshape(call.args[0])):
            return False

    return True

def is_transpose_hslice(call):
    t_axes = call.attrs.axes
    hslice_t_axes = (0, 2, 1, 3)
    
    if not (len(t_axes) == 4 and all([hslice_t_axes[i] == t_axes[i] for i in range(4)])):
        return False

    return True

def is_reshape_transpose_hslice(call):
    return is_reshape_hslice(call) and is_transpose_hslice(call)

def _is_transpose_reshape_hstack_helper(transpose_axes, reshape_newshape, reshape_in_shape):
    hstack_t_axes = (0, 2, 1, 3)
    
    if not (len(transpose_axes) == 4 and all([hstack_t_axes[i] == transpose_axes[i] for i in range(4)])):
        return False
    
    if (not (len(reshape_newshape) == 2 or (len(reshape_newshape) == 3) and reshape_newshape[0].value == 1)
    or not all([dim == 1 for dim in reshape_newshape[:-2]])
    or not (reshape_in_shape[-3].value == reshape_newshape[-2].value)):
            return False

    return True

def is_transpose_reshape_hstack(call):

    if is_superfluous_reshape(call):
        return False
    t_axes = call.args[0].attrs.axes

    r_newshape = call.checked_type.shape
    r_input_shape = call.type_args[0].shape
    return _is_transpose_reshape_hstack_helper(t_axes, r_newshape, r_input_shape)

def is_transpose_reshape_reshape_hstack(call):
    if is_superfluous_reshape(call) or is_superfluous_reshape(call.args[0]):
        return False

    t_axes = call.args[0].args[0].attrs.axes
    r_newshape = call.checked_type.shape
    r_input_shape = call.args[0].type_args[0].shape
    return _is_transpose_reshape_hstack_helper(t_axes, r_newshape, r_input_shape)

def is_stack_reshape_reshape_to_binary_stack(call):
    dim = len(call.checked_type.shape)
    stack_axis = call.args[0].attrs.axis.value
    if stack_axis < 0:
        stack_axis = stack_axis + dim

    input_shape = [int(dim) for dim in call.args[0].args[0][0].checked_type.shape]
    output_shape = [int(dim) for dim in call.checked_type.shape]

    # if input_shape == output_shape:
    #     return False

    works = all([(dim != stack_axis and i == o) or (dim == stack_axis and o == 2 * i) for dim, (i, o) in enumerate(zip(input_shape, output_shape))])
    return works

def is_concat_reshape_reshape_to_binary_stack(call):
    dim = len(call.checked_type.shape)
    concat_axis = call.args[0].attrs.axis - 1 # decrement by 1 because inputs are unsqueezed first 
    if concat_axis < 0:
        concat_axis = concat_axis + dim

    for operand in call.args[0].args[0]:
        if operand.op.name != "reshape":
            return False
    
    input_shape = [int(dim) for dim in call.args[0].args[0][0].args[0].checked_type.shape]
    output_shape = [int(dim) for dim in call.checked_type.shape]

    works = all([i == o or (dim == concat_axis and o == 2 * i) for dim, (i, o) in enumerate(zip(input_shape, output_shape))])
    return works

def match_einsum_pattern(pattern, query):
    query = query.replace(" ", "")
    pattern = pattern.replace(" ", "")
    if len(query) != len(pattern):
        return False

    query_dict = {}
    for char in query:
        query_dict[char] = []
    
    for i in range(len(query)):
        char = query[i]
        query_dict[char].append(i)

    pattern_dict = {}
    for char in pattern:
        pattern_dict[char] = []
    
    for i in range(len(pattern)):
        char = pattern[i]
        pattern_dict[char].append(i)

    return sorted(list(query_dict.values())) == sorted(list(pattern_dict.values()))

