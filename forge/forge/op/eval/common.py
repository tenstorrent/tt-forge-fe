# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import struct

from typing import List, Tuple
from math import prod

import torch
import tensorflow as tf
import numpy as np

from collections import defaultdict
from loguru import logger
from scipy.spatial import distance

from ...forgeglobal import TILE_DIM

from ...tensor import forge_dataformat_to_pytorch_dtype
from forge import DataFormat, MathFidelity


def to_torch_operands(*operands):
    """
    Convert input tensors into compatible torch tensors with a common promoted dtype.
    """
    # Validate
    for o in operands:
        assert isinstance(o, (int, torch.Tensor)), f"Invalid operand type: {type(o)}"

    # Extract all floating point dtypes
    float_dtypes = [o.dtype for o in operands if isinstance(o, torch.Tensor) and o.is_floating_point()]

    if not float_dtypes:
        return operands  # nothing to promote

    # Determine highest precision dtype
    promoted_dtype = float_dtypes[0]
    for dt in float_dtypes[1:]:
        promoted_dtype = torch.promote_types(promoted_dtype, dt)

    # Cast tensors as needed
    new_operands = []
    for o in operands:
        if isinstance(o, torch.Tensor) and o.is_floating_point() and o.dtype != promoted_dtype:
            new_operands.append(o.to(promoted_dtype))
        else:
            new_operands.append(o)

    return tuple(new_operands)


def cast_for_cpu_eval(t_ops, op_name=None):
    # Torch does not support int8 or float16 on CPU
    # So we cast to float32
    # Note that INT8 matmul results in INT32 output
    original_type = t_ops[0].dtype
    t_ops = list(t_ops)
    for index, op in enumerate(t_ops):
        if op.dtype == torch.float16:
            t_ops[index] = op.to(torch.float32)
        if op.dtype == torch.int8:
            t_ops[index] = op.to(torch.float32)
            if op_name == "matmul":
                original_type = torch.int32
    return t_ops, original_type


def create_constant_tensor_from_value(value: float, dims: Tuple[int, int], df: DataFormat) -> torch.Tensor:
    dim_r, dim_c = dims
    if dim_r < 0:
        dim_r = 0
    if dim_c < 0:
        dim_c = 0

    tensor_r = dim_r
    tensor_c = dim_c

    dtype = forge_dataformat_to_pytorch_dtype(df)
    if tensor_c == 0 and tensor_r == 0:
        # Unsqueezing to make this a 1d tensor.
        # Currently, the runtime expects a 1d tensor for scalar values.
        tensor = torch.unsqueeze(torch.tensor(value, dtype=dtype), dim=0)
    elif tensor_c == 0 or tensor_r == 0:
        dim = tensor_c if tensor_r == 0 else tensor_r
        tensor = torch.zeros(dim, dtype=dtype)
        tensor[0:dim] = value
    else:
        tensor = torch.zeros(tensor_r, tensor_c, dtype=dtype)
        tensor[0:tensor_r, 0:tensor_c] = value

    return tensor


def create_constant_tensor_from_tensor(
    tensor_values: List[float], tensor_shape: List[int], df: DataFormat
) -> torch.Tensor:
    assert prod(tensor_shape) == len(tensor_values)
    tensor = torch.FloatTensor(tensor_values)
    tensor = tensor.reshape(tensor_shape)
    tensor = tensor.type(forge_dataformat_to_pytorch_dtype(df))
    return tensor


def create_constant_tensor(flat_data: List[float], shape: List[int], df: DataFormat) -> torch.Tensor:
    tensor = torch.FloatTensor(flat_data)
    tensor = tensor.reshape(*shape)
    tensor = tensor.type(forge_dataformat_to_pytorch_dtype(df))
    return tensor


def dump_tensor(tensor, name, entry=0):
    fmt = "f"
    if tensor.dtype == torch.half or tensor.dtype == torch.float16 or tensor.dtype == torch.bfloat16:
        tensor = tensor.to(torch.float)
    elif tensor.dtype == torch.int or tensor.dtype == torch.int32:
        fmt = "i"
    elif tensor.dtype == torch.short or tensor.dtype == torch.int16:
        fmt = "h"
    elif tensor.dtype == torch.int8:
        fmt = "b"
    elif tensor.dtype == torch.uint8:
        fmt = "B"

    with open(f"{name}.{entry}.bin", "wb") as out_file:
        for val in tensor.ravel().tolist():
            out_file.write(struct.pack(fmt, val))
        out_file.close()


def eval_debug_print(type, inputs, output):
    mode = int(os.environ.get("EVAL_DEBUG", "0"))
    if mode > 0:
        print(
            f"{type}: inputs: {[i.shape if isinstance(i, torch.Tensor) else 1 for i in inputs]}, output: {output.shape}"
        )
        if mode > 1:
            for i, input in enumerate(inputs):
                print("input ", i)
                print(input)
            print("output:")
            print(output)


def math_fidelity_to_multiplier(fidelity: MathFidelity) -> int:
    if fidelity == MathFidelity.LoFi:
        return 1
    if fidelity == MathFidelity.HiFi2:
        return 2
    if fidelity == MathFidelity.HiFi3:
        return 3
    return 4


def data_format_to_int(df: DataFormat) -> int:
    if df == DataFormat.Float16:
        return 1
    if df == DataFormat.Float16_b:
        return 2
    if df == DataFormat.Bfp8:
        return 3
    if df == DataFormat.Bfp8_b:
        return 4
    if df == DataFormat.Bfp4:
        return 5
    if df == DataFormat.Bfp4_b:
        return 6
    if df == DataFormat.Bfp2:
        return 7
    if df == DataFormat.Bfp2_b:
        return 8
    if df == DataFormat.Float32:
        return 9
    if df == DataFormat.Lf8:
        return 11
    raise RuntimeError(f"Unknown data format {df}")


def calculate_tile_size(val):
    # We might not even care about large dim size
    # that are not divisible by 32
    if val > 32:
        return 32

    smallest_pad = 31
    current_tile_size = 32

    tile_sizes = [32, 16, 8, 4, 2, 1]

    for tile_size_ in tile_sizes:
        rem = val % tile_size_
        pad = tile_size_ - rem
        if rem == 0 and smallest_pad != 0:
            # Pick the largest tile size that divides evenly
            smallest_pad = 0
            current_tile_size = tile_size_
        elif pad <= smallest_pad:
            # pick the tile size with smallest pad
            smallest_pad = pad
            current_tile_size = tile_size_

    return current_tile_size


# Global compiler cache
g_compiler_perf_cache: defaultdict = defaultdict(dict)

# def get_compiler_cached_cycles(desc: OpModelDesc) -> int:
#     global g_compiler_perf_cache

#     if not g_compiler_perf_cache:
#         cache_file = os.environ.get("FORGE_COMPILER_CACHE", None)
#         if cache_file is not None and os.path.exists(cache_file):
#             with open(os.environ["FORGE_COMPILER_CACHE"], 'rb') as file:
#                 import pickle
#                 g_compiler_perf_cache = pickle.load(file)
#         else:
#             return None

#     cached_op_model = g_compiler_perf_cache["op_model"]

#     if desc.type in cached_op_model:
#         cache_cycles = cached_op_model[desc.type]
#         shapes = (desc.mblock_m, desc.mblock_n, desc.ublock_rt, desc.ublock_ct, desc.t)

#         if desc.type == 'matmul':  # append k dim to lookup
#             shapes = shapes + (desc.mblock_k, desc.ublock_kt)

#         if shapes in cache_cycles:
#             cycle_count = cache_cycles[shapes]
#             # print(f"Using recorded cycle count for {desc.type} of shapes {shapes} -> {cycle_count}")
#             return cycle_count

#     return None
