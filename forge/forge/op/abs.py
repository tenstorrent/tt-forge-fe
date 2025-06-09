# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

import torch
import torch.nn.functional
from eval.interface import PyEltwiseUnaryOp
from loguru import logger
from eval.common import to_torch_operands
from forgeglobal import TILE_DIM
from tensor import forge_dataformat_to_pytorch_dtype
import numpy as np
from forge.op.eval.common import calculate_tile_size
from eval.lforge.abs import Abs as ForgeAbs
from forge._C import DataFormat
from forge._C.graph import OpType
from tensor import Tensor


class Abs(PyEltwiseUnaryOp):
    def __init__(self, operand: Tensor) -> Tensor:
        """
        Create an op with given parameters.
        """
        self.create("abs")

        shapes = operand.shape.get_pytorch_shape()
        shape, self.operand_broadcast = self.shape(shapes)

        has_reference_calculation = False
        ref_output = None
        
        data_format = operand.data_format

        if operand.has_value():
            value = operand.value() 
            ref_output = self.eval(value)
            data_format = pytorch_dtype_to_forge_dataformat(ref_output.dtype)
            has_reference_calculation = True

        result = Tensor.create_from_trace(src_op=self, shape=shape, data_format=data_format)
        result.requires_grad = any([o.requires_grad for o in self.operands])

        # Calculate reference if there's one
        if has_reference_calculation:
            result.set_value(ref_output)

        return result


        # Calculate reference if there's one
        if all([o.has_value() if isinstance(o, (Tensor, Parameter)) else True for o in self.operands]):
            values = [o.value() if isinstance(o, (Tensor, Parameter)) else o for o in self.operands]
            ref_output = get_f_forge_eval(self.cpp_op_type)(values)
            has_reference_calculation = True

        # TODO: pick data formats in some way when mismatched inputs are coming...
        if out_df is not None:
            data_format = out_df  # User provided output dataformat

        # we should create a map that maps input dataformat to output dataformat
        elif self.op_type in ["matmul", "conv2d"]:
            op0_df = self.operands[0].data_format
            op1_df = self.operands[1].data_format
            if op0_df == DataFormat.Int8 and op1_df == DataFormat.Int8:
                data_format = DataFormat.Int32
            else:
                data_format = self.operands[0].data_format
        elif len(self.operands) > 0:
            if self.op_type in ["where", "embedding"]:
                data_format = self.operands[1].data_format
            else:
                # Use dataformat from the reference implementation if available (e.g. torch)
                # NOTE: This might need to be changed once we introduce config where each op can have its own dataformat
                # regardless of the reference implementation (e.g. running convolution in lower precision)
                if has_reference_calculation:
                    data_format = pytorch_dtype_to_forge_dataformat(ref_output.dtype)
                else:
                    data_format = self.operands[0].data_format
        else:
            if has_reference_calculation:
                data_format = pytorch_dtype_to_forge_dataformat(ref_output.dtype)
            else:
                data_format = DataFormat.Float32

        result = Tensor.create_from_trace(src_op=self, shape=shape, data_format=data_format)
        result.requires_grad = any([o.requires_grad for o in self.operands])

        # Calculate reference if there's one
        if has_reference_calculation:
            result.set_value(ref_output)

        return result
        
    # Waiting on Vanja to merge simplifying of get_tensor

    @classmethod
    def create(cls):
        self = cls("abs")
        return self

    def eval(self, tensors):
        assert len(tensors) == 1, "Abs should have one input"
        shape = tensors[0].shape
        original_types = [o.dtype for o in tensors]
        ret = torch.abs(tensors[0])

        if ret.dtype != original_types[0]:
            ret = ret.type(original_types[0])

        return ret

    def shape(self, tensor_shapes):
        assert len(tensor_shapes) == 1, "Abs should have one input"
        shape = tensor_shapes[0]
        return shape, []

    def backward(self, ac, operand, inputs, output, grad):
        assert len(inputs) == 1, "Abs should have one input"
        assert operand == 0, "Invalid operand index"
        heaviside = ac.op("heaviside", (inputs[0], ac.constant(0.5)))
        subtract = ac.op("subtract", (heaviside, ac.constant(0.5)))
        stretched = ac.op("multiply", (subtract, ac.constant(2.0)))
        return ac.op("multiply", (stretched, grad))

    def lower(self, lc, tensors, outputs):
        assert len(tensors) == 1, "Abs should  have one input"

        if bool(int(os.environ.get("FORGE_ENABLE_TINY_TILE", "0"))):
            node_shape = list(tensors[0].shape)
            tile_height = calculate_tile_size(node_shape[-2])
            tile_width = calculate_tile_size(node_shape[-1])
            vector = "" if tile_height == TILE_DIM else "r"
        else:
            vector = None
            tile_height, tile_width = TILE_DIM, TILE_DIM

        lc.op(
            ForgeAbs.create(vector=vector),
            tensors,
            tile_height=tile_height,
            tile_width=tile_width,
        )

    def initial_flops_estimate(self, tensor_shapes):
        flops = 0
        output_shape = self.shape(tensor_shapes)[0]
        flops = np.prod(output_shape)

        return flops