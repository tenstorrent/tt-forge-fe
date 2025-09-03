# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import struct
from typing import List, Tuple
from math import prod
import torch
from typing import Union
from ..tensor import Tensor
from ..parameter import Parameter
from forge._C import DataFormat
from forge._C.ops import Op, OpType
import forge
from forge.forgeglobal import get_unique_node_id, tracing
from forge.tensor import pytorch_dtype_to_forge_dataformat, forge_dataformat_to_pytorch_dtype

deprecated_name_dict = {}
deprecated_op_id = 0


class ForgeOp:
    def __init__(self, type: OpType, name: str, *operands: Union[Tensor, Parameter], **attrs):
        """
        Create an op with given parameters.
        """

        global deprecated_op_id, deprecated_name_dict
        if tracing():
            if name != "":
                self.name = name
            else:
                unique_id = get_unique_node_id()
                self.name = f"{type.name}_{unique_id}"
                if unique_id != deprecated_op_id:
                    deprecated_name_dict[f"{type.name}_{deprecated_op_id}"] = self.name
        deprecated_op_id += 1

        operands = tuple(
            forge.op.Constant("", constant=operand) if isinstance(operand, (int, float)) else operand
            for operand in operands
        )
        self.operands = operands
        self.op = Op(type, attrs)

    def get_tensor(self, out_df=None) -> Tensor:
        """
        Generate result tensor of the right shape, and if value is set, value.
        """

        # get reference output shape
        shapes = [o.shape.get_pytorch_shape() for o in self.operands]
        shape, self.operand_broadcast = self.op.shape(shapes)

        # get reference output value
        values = [o.value() if isinstance(o, (Tensor, Parameter)) else o for o in self.operands]
        ref_output = self.op.eval(values)

        if out_df is not None:  # User provided output dataformat
            data_format = out_df
        else:  # Use dataformat from the reference implementation if available (e.g. torch)
            # NOTE: This might need to be changed once we introduce config where each op can have its own dataformat
            # regardless of the reference implementation (e.g. running convolution in lower precision)
            data_format = pytorch_dtype_to_forge_dataformat(ref_output.dtype)

        result = Tensor.create_from_trace(src_op=self, shape=shape, data_format=data_format)
        result.requires_grad = any([o.requires_grad for o in self.operands])
        result.set_value(ref_output)

        return result


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
