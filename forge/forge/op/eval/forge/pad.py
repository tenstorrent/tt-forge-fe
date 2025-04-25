# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import ast
import torch
import math

from forge._C.graph import NodeType
from forge.forgeglobal import TILE_DIM
from forge.utils import align_up_tile, round_up_div, clamp
from forge import Tensor
from .transpose import TransposeTM
from .nop import Nop

import torch
from ..interface import PyOp, PyTM
from ..common import to_torch_operands
from ..sparse_utils import create_pad_reflect_sparse_picker, create_pad_replicate_sparse_picker


class Pad(PyOp):
    @classmethod
    def create(cls, padding, value, mode, pad_len):
        self = cls("pad")
        self.padding = padding
        self.value = value
        self.mode = mode
        self.pad_len = pad_len
        return self

    def eval(self, tensors):
        t_ops = to_torch_operands(*tensors)
        mode = self.mode
        # We know that for TVM padding (top, bottom, left, right, etc.)
        # PyTorch expects (left, right, top, bottom, etc.)
        # So, we just need to swap top/bottom and left/right for every two elements in pad_len
        mode = self.mode
        torch_padding = []
        for i in range(self.pad_len - 2, -1, -2):
            torch_padding.append(self.padding[i])
            torch_padding.append(self.padding[i + 1])

        if mode in ["replicate", "reflect"] and (self.pad_len != 4 or self.pad_len != 2):
            torch_padding = torch_padding[: int(self.pad_len / 2)]

        return torch.nn.functional.pad(t_ops[0], tuple(torch_padding), mode=mode)

    def shape(self, tensor_shapes):
        shape = list(tensor_shapes[0])
        # channel_last = self.channel_last
        mode = self.mode
        torch_padding = []
        for i in range(self.pad_len - 2, -1, -2):
            torch_padding.append(self.padding[i])
            torch_padding.append(self.padding[i + 1])

        if mode in ["replicate", "reflect"] and (self.pad_len != 4 or self.pad_len != 2):
            torch_padding = torch_padding[: int(self.pad_len / 2)]

        shape = torch.nn.functional.pad(torch.rand(shape), tuple(torch_padding), mode=mode).shape
        return tuple(shape), []


def decompose(self, dc, inputs):
    mode = self.mode
    padding = []

    # Reverses padding order from end to start in steps of 2
    # Example: if self.pad_len is 4, get pairs (2, 3), (0, 1)
    for i in range(self.pad_len - 2, -1, -2):
        padding.append(self.padding[i])
        padding.append(self.padding[i + 1])

    # Special handling for 'replicate' and 'reflect' modes.
    # If padding length isn't standard (2 or 4), only use half (remove redundant pairs)
    if mode in ["replicate", "reflect"] and (self.pad_len != 4 or self.pad_len != 2):
        padding = padding[: int(self.pad_len / 2)]

    # Early exit if all padding values are 0 (no padding needed)
    if all([x == 0 for x in padding]):
        result = dc.op(Nop.create(), [inputs[0]])
        dc.fuse(result)  # Fuse the no-op into graph and return
        return

    activations = inputs[0]
    mode = self.mode

    r = activations.shape[-2]  # Number of rows
    c = activations.shape[-1]  # Number of columns

    # Identify total padding across each axis based on 2D or 4D padding
    if len(padding) == 2:
        total_padding_c = padding[0] + padding[1]
        total_padding_r = 0
        all_around_padding = padding + [0, 0]
    elif len(padding) == 4:
        total_padding_c = padding[0] + padding[1]
        total_padding_r = padding[2] + padding[3]
        all_around_padding = padding

    # Optimization: Check if padding is only on the trailing sides
    # and doesn't cross tile boundaries, allowing us to skip adding pad ops
    if (
        ((self.pad_len == 2 and padding[0] == 0) or (self.pad_len == 4 and padding[0] == 0 and padding[2] == 0))
        and not channel_last
        and math.ceil((total_padding_r + r) / TILE_DIM) == math.ceil(r / TILE_DIM)
        and math.ceil((total_padding_c + c) / TILE_DIM) == math.ceil(c / TILE_DIM)
        and mode == "constant"
    ):
        return  # No need to decompose padding if it doesn't affect tiling layout

    # Unpack padding into named variables
    left, right, top, bottom = 0, 0, 0, 0
    if len(padding) == 2:
        left, right = padding
    elif len(padding) == 4:
        left, right, top, bottom = padding

    # Handle 'replicate' padding mode
    if mode == "replicate":
        result = activations
        orig_shape = result.shape

        # Flatten to 2D for matrix operation
        if len(orig_shape) == 2:
            shape = (1, orig_shape[-2] * orig_shape[-1])
        else:
            shape = (1, 1, orig_shape[-3], orig_shape[-2] * orig_shape[-1])

        result = dc.op_with_named_attrs("reshape", [result], {"shape": shape}, shape)
        result = dc.op(TransposeTM.create(-2, -1), [result])  # Transpose for matmul

        # Create sparse matrix that applies replicate padding
        spm = create_pad_replicate_sparse_picker(r, c, left, right, top, bottom)
        spm = dc.tensor(spm.to_dense())

        result = dc.op("matmul", [spm, result])
        result = dc.op(TransposeTM.create(-2, -1), [result])  # Transpose back

        # Reshape back to original or padded shape
        if len(orig_shape) == 2:
            shape = (orig_shape[-2] + total_padding_r, orig_shape[-1] + total_padding_c)
        else:
            shape = (1, orig_shape[-3], orig_shape[-2] + total_padding_r, orig_shape[-1] + total_padding_c)

        result = dc.op_with_named_attrs("reshape", [result], {"shape": shape}, shape)

        dc.fuse(result)  # Finalize into graph
        return

    # Handle 'reflect' padding mode (similar to replicate)
    elif mode == "reflect":
        result = activations
        orig_shape = result.shape

        # Flatten to 2D for sparse op
        if len(orig_shape) == 2:
            shape = (1, orig_shape[-2] * orig_shape[-1])
        else:
            shape = (1, 1, orig_shape[-3], orig_shape[-2] * orig_shape[-1])

        result = dc.op_with_named_attrs("reshape", [result], {"shape": shape}, shape)
        result = dc.op(TransposeTM.create(-2, -1), [result])

        # Create sparse matrix for reflect padding
        spm = create_pad_reflect_sparse_picker(r, c, left, right, top, bottom)
        spm = dc.tensor(spm.to_dense())

        result = dc.op("matmul", [spm, result])
        result = dc.op(TransposeTM.create(-2, -1), [result])

        # Reshape to padded shape
        if len(orig_shape) == 2:
            shape = (orig_shape[-2] + total_padding_r, orig_shape[-1] + total_padding_c)
        else:
            shape = (1, orig_shape[-3], orig_shape[-2] + total_padding_r, orig_shape[-1] + total_padding_c)

        result = dc.op_with_named_attrs("reshape", [result], {"shape": shape}, shape)

        dc.fuse(result)
        return

    def backward(self, ac, operand, inputs, output, grad):
        pass

    def lower(self, lc, tensors, outputs):
        pass

    def is_tm(self) -> bool:
        return False

    def is_eltwise(self) -> bool:
        return False

    def is_eltwise_binary(self) -> bool:
        return False

    def is_eltwise_unary(self) -> bool:
        return False

    def is_eltwise_nary(self) -> bool:
        return False
