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

        input_tensor = t_ops[0]
        mode = self.mode
        if mode in ["replicate", "reflect"]:
            padding = self.padding[::-1]
            padding = padding[: int(self.pad_len / 2)]  # Ensure the result is an integer

        else:
            padding = self.padding[::-1]
        breakpoint()
        return torch.nn.functional.pad(t_ops[0], tuple(padding), mode=mode)

    def shape(self, tensor_shapes):
        shape = list(tensor_shapes[0])
        # padding = self.padding[-self.pad_len:]
        # padding = padding[::-1]
        mode = self.mode
        if mode in ["replicate", "reflect"]:
            padding = self.padding[::-1]
            padding = padding[: int(self.pad_len / 2)]  # Ensure the result is an integer
        else:
            padding = self.padding[::-1]
        shape = torch.nn.functional.pad(torch.rand(shape), tuple(padding), mode=mode).shape
        breakpoint()
        return tuple(shape), []

    def decompose(self, dc, inputs):

        if self.mode in ["replicate", "reflect"]:
            padding = self.padding[::-1]
            padding = padding[: int(self.pad_len / 2)]  # Ensure the result is an integer
            if len(padding) == 2:
                total_padding_c = padding[0] + padding[1]
                total_padding_r = 0
                all_around_padding = padding + [0, 0]
            elif len(padding) == 4:
                total_padding_c = padding[0] + padding[1]
                total_padding_r = padding[2] + padding[3]
                all_around_padding = padding
            else:
                raise RuntimeError("Forge only support Pad with either 2 or 4 attributes")

        else:
            padding = self.padding[::-1]

        if all([x == 0 for x in padding]):
            # Pad size is 0
            result = dc.op(Nop.create(), [inputs[0]])
            dc.fuse(result)

        activations = inputs[0]
        mode = self.mode

        r = activations.shape[-2]
        c = activations.shape[-1]
        # Find out if padding exceeds tile boundary
        # R, C are flipped because pytorch pad starts from last axis
        if (
            ((self.pad_len == 2 and padding[0] == 0) or (self.pad_len == 4 and padding[0] == 0 and padding[2] == 0))
            and math.ceil((total_padding_r + r) / TILE_DIM) == math.ceil(r / TILE_DIM)
            and math.ceil((total_padding_c + c) / TILE_DIM) == math.ceil(c / TILE_DIM)
            and mode == "constant"  # 'constant' mode
        ):
            # Pad does not exceed tile boundary and only on the end of axis
            # Will be lowered into NOP
            return

        else:
            # Lower into concats
            if self.mode in ["reflect", "replicate"]:
                left, right, top, bottom = 0, 0, 0, 0
                if len(padding) == 2:
                    (
                        left,
                        right,
                    ) = padding

                elif len(padding) == 4:
                    (
                        left,
                        right,
                        top,
                        bottom,
                    ) = padding
                else:
                    raise RuntimeError("Forge only support Pad with either 2 or 4 pad_len")

            if mode == "replicate":  # 'replicate' mode
                result = activations
                orig_shape = result.shape
                if len(orig_shape) == 2:
                    shape = (1, orig_shape[-2] * orig_shape[-1])
                else:
                    shape = (1, 1, orig_shape[-3], orig_shape[-2] * orig_shape[-1])

                result = dc.op_with_named_attrs("reshape", [result], {"shape": shape}, shape)
                result = dc.op(TransposeTM.create(-2, -1), [result])
                spm = create_pad_replicate_sparse_picker(r, c, left, right, top, bottom)

                spm = dc.tensor(spm.to_dense())
                result = dc.op("matmul", [spm, result])
                result = dc.op(TransposeTM.create(-2, -1), [result])
                if len(orig_shape) == 2:
                    shape = (orig_shape[-2] + total_padding_r, orig_shape[-1] + total_padding_c)
                else:
                    shape = (1, orig_shape[-3], orig_shape[-2] + total_padding_r, orig_shape[-1] + total_padding_c)

                result = dc.op_with_named_attrs("reshape", [result], {"shape": shape}, shape)

                dc.fuse(result)
                return
            elif mode == "reflect":
                # Reflect mode
                result = activations
                orig_shape = result.shape
                if len(orig_shape) == 2:
                    shape = (1, orig_shape[-2] * orig_shape[-1])
                else:
                    shape = (1, 1, orig_shape[-3], orig_shape[-2] * orig_shape[-1])

                result = dc.op_with_named_attrs("reshape", [result], {"shape": shape}, shape)
                result = dc.op(TransposeTM.create(-2, -1), [result])
                spm = create_pad_reflect_sparse_picker(r, c, left, right, top, bottom)

                spm = create_pad_reflect_sparse_picker(activation, padding)

                spm = dc.tensor(spm.to_dense())
                result = dc.op("matmul", [spm, result])
                result = dc.op(TransposeTM.create(-2, -1), [result])

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
