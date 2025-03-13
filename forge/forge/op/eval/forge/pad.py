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
from ..sparse_utils import create_pad_reflect_sparse_picker


class Pad(PyOp):
    @classmethod
    def create(
        cls,
        padding,
        value,  
        mode,
        channel_last,
        pad_len
    ):
        self = cls("pad")
        self.padding = padding
        self.value = value
        self.mode = mode
        self.channel_last = int(channel_last)
        self.pad_len = pad_len
        return self

    def eval(self, tensors):
        t_ops = to_torch_operands(*tensors)

        input_tensor = t_ops[0]
        
        if self.pad_len == 2:
            padding = self.padding[-self.pad_len:]
        elif self.pad_len == 4:
            padding = self.padding[-self.pad_len:]
            padding = padding[::-1]
        else:
            raise RuntimeError("Forge only support Pad with either 2 or 4 attributes")
        mode = self.mode
        channel_last = self.channel_last
        if channel_last:
            padding = [0, 0] + padding
        return torch.nn.functional.pad(t_ops[0], tuple(padding), mode=mode)

    def shape(self, tensor_shapes):
        shape = list(tensor_shapes[0])
        channel_last = self.channel_last
       
        if self.pad_len == 2:
            padding = self.padding[-self.pad_len:]
        elif self.pad_len == 4:
            padding = self.padding[-self.pad_len:]
            padding = padding[::-1]
        else:
            raise RuntimeError("Forge only support Pad with either 2 or 4 attributes")
        if channel_last:
            shape[-2] += padding[0] + padding[1]
            if self.pad_len == 4:
                shape[-3] += padding[2] + padding[3]
        else:
            shape[-1] += padding[0] + padding[1]
            if self.pad_len == 4:
                shape[-2] += padding[2] + padding[3]
        return tuple(shape), []

    def decompose(self, dc, inputs):
        if self.pad_len == 2:
            padding = self.padding[-self.pad_len:]
        elif self.pad_len == 4:
            padding = self.padding[-self.pad_len:]
            padding = padding[::-1]
        else:
            raise RuntimeError("Forge only support Pad with either 2 or 4 attributes")
        if all([x == 0 for x in padding]):
            # Pad size is 0
            result = dc.op(Nop.create(), [inputs[0]])
            dc.fuse(result)
        
        activations = inputs[0]
        mode = self.mode
        channel_last = self.channel_last
        if channel_last:
            r = activations.shape[-3]
            c = activations.shape[-2]
        else:
            r = activations.shape[-2]
            c = activations.shape[-1]
        # Find out if padding exceeds tile boundary
        # R, C are flipped because pytorch pad starts from last axis

        if self.pad_len == 2:
            total_padding_c = padding[0] + padding[1]
            total_padding_r = 0
            all_around_padding = padding + [0, 0]
        elif self.pad_len == 4:
            total_padding_c = padding[0] + padding[1]
            total_padding_r = padding[2] + padding[3]
            all_around_padding = padding
        else:
            raise RuntimeError("Forge only support Pad with either 2 or 4 attributes")

        if (
            ((self.pad_len == 2 and padding[0] == 0) or (self.pad_len == 4 and padding[0] == 0 and padding[2] == 0))
            and not channel_last
            and math.ceil((total_padding_r + r) / TILE_DIM) == math.ceil(r / TILE_DIM)
            and math.ceil((total_padding_c + c) / TILE_DIM) == math.ceil(c / TILE_DIM)
            and mode == "constant"  # 'constant' mode
        ):
            # Pad does not exceed tile boundary and only on the end of axis
            # Will be lowered into NOP
            return

        else:
            # Lower into concats
            breakpoint()
            left, right, top, bottom = 0, 0, 0, 0
            if self.pad_len == 2:
                left, right, = padding

            elif self.pad_len == 4:
                left, right, top, bottom, = padding
            else:
                raise RuntimeError("Forge only support Pad with either 2 or 4 pad_len")

            if mode == "replicate":  # 'replicate' mode
                result = activations

                if channel_last:
                    result = dc.op(TransposeTM.create(-3, -1, result.shape[-3]), [result])

                    orig_shape = result.shape
                    result = dc.op("reshape", [result], (1, 1, orig_shape[-3], orig_shape[-2] * orig_shape[-1]))
                    result = dc.op(TransposeTM.create(-2, -1), [result])
                    spm = create_pad_replicate_sparse_picker(c, r, top, bottom, left, right)
                    spm = dc.tensor(spm)
                    result = dc.op("sparse_matmul", [spm, result])
                    result = dc.op(TransposeTM.create(-2, -1), [result])
                    result = dc.op(
                        "reshape",
                        [result],
                        (1, orig_shape[-3], orig_shape[-1] + total_padding_r, orig_shape[-2] + total_padding_c),
                    )

                    result = dc.op(TransposeTM.create(-3, -1, result.shape[-3]), [result])
                else:
                    orig_shape = result.shape
                    if len(orig_shape) == 2:
                        result = dc.op("reshape", [result], (1, orig_shape[-2] * orig_shape[-1]))
                    else:
                        result = dc.op("reshape", [result], (1, 1, orig_shape[-3], orig_shape[-2] * orig_shape[-1]))
                    result = dc.op(TransposeTM.create(-2, -1), [result])
                    spm = create_pad_replicate_sparse_picker(r, c, left, right, top, bottom)
                    spm = dc.tensor(spm)
                    result = dc.op("sparse_matmul", [spm, result])
                    result = dc.op(TransposeTM.create(-2, -1), [result])
                    if len(orig_shape) == 2:
                        result = dc.op(
                            "reshape", [result], (orig_shape[-2] + total_padding_r, orig_shape[-1] + total_padding_c)
                        )
                    else:
                        result = dc.op(
                            "reshape",
                            [result],
                            (1, orig_shape[-3], orig_shape[-2] + total_padding_r, orig_shape[-1] + total_padding_c),
                        )

                dc.fuse(result)
                return
            elif mode == "reflect":
                breakpoint()
                # Reflect mode
                result = activations

                if channel_last:
                    breakpoint()
                    result = dc.op(TransposeTM.create(-3, -1, result.shape[-3]), [result])

                    orig_shape = result.shape
                    result = dc.op_with_named_attrs(
                        "reshape",
                        [result],
                        {"shape": (1, 1, orig_shape[-3], orig_shape[-2] * orig_shape[-1])},
                        (1, 1, orig_shape[-3], orig_shape[-2] * orig_shape[-1]),
                    )
                    result = dc.op(TransposeTM.create(-2, -1), [result])
                    spm = create_pad_reflect_sparse_picker(c, r, top, bottom, left, right)
                    spm = dc.tensor(spm.to_dense())
                    result = dc.op("matmul", [spm, result])
                    result = dc.op(TransposeTM.create(-2, -1), [result])
                    result = dc.op_with_named_attrs(
                        "reshape",
                        [result],
                        {
                            "shape": (
                                1,
                                orig_shape[-3],
                                orig_shape[-1] + total_padding_r,
                                orig_shape[-2] + total_padding_c,
                            )
                        },
                        (1, orig_shape[-3], orig_shape[-1] + total_padding_r, orig_shape[-2] + total_padding_c),
                    )

                    result = dc.op(TransposeTM.create(-3, -1, result.shape[-3]), [result])
                else:
                    breakpoint()
                    orig_shape = result.shape
                    if len(orig_shape) == 2:
                        shape = (1, orig_shape[-2] * orig_shape[-1])
                    else:
                        shape = (1, 1, orig_shape[-3], orig_shape[-2] * orig_shape[-1])

                    result = dc.op_with_named_attrs("reshape", [result], {"shape": shape}, shape)
                    result = dc.op(TransposeTM.create(-2, -1), [result])
                    spm = create_pad_reflect_sparse_picker(r, c, left, right, top, bottom)
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

