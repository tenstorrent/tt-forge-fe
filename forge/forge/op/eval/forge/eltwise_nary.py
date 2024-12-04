# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from typing import List, Tuple
from math import gcd
import ast
import os
import torch
import math
import forge
from ..common import to_torch_operands
from .transpose import TransposeTM
from .nop import Nop
from .buffer import Buffer
from ..lforge.splice import Splice
from forge.forgeglobal import TILE_DIM, align_up_tile, is_tile_dim_aligned
from ..sparse_utils import (
    create_flattened_padding_removal_sparse_picker_matrix,
)
from loguru import logger


def eval(type, attr, ops):

    if type == "conv_sum":

        t_ops = to_torch_operands(*ops)

        t_ops = list(t_ops)

        # Extract attributes
        originalY = attr[0]
        originalX = attr[1]
        shifts = attr[2:]

        # Check operands
        for t_op in t_ops:
            assert (
                len(t_op.shape) == 4
            ), f"Tensor must have 4 dimensions, given {len(t_op.shape)}"

        # To forge shape
        for i in range(len(t_ops)):
            t_ops[i] = t_ops[i][:, :, : originalY * originalX, :]
            t_ops[i] = t_ops[i].transpose(2, 3)
            t_ops[i] = t_ops[i].reshape(1, t_ops[i].shape[2], originalY, originalX)

        # Shift and Add
        res = 0
        for i in range(len(t_ops)):
            res += torch.nn.functional.pad(
                t_ops[i],
                (shifts[2 * i], -shifts[2 * i], shifts[2 * i + 1], -shifts[2 * i + 1]),
            )

        # To forge shape
        res = res.reshape(1, res.shape[1], res.shape[2] * res.shape[3], 1)
        res = res.transpose(1, 3)

        return res

    elif type == "concatenate":
        assert len(attr) == 1, "Concatenate should have 1 attr"
        t_ops = to_torch_operands(*ops)
        return torch.cat(t_ops, dim=attr[0])

    elif type == "where":
        return torch.where(ops[0].type(torch.bool), ops[1], ops[2])

    elif type == "index_copy":
        t_ops = to_torch_operands(*ops)
        out = t_ops[0].index_copy(attr[0], t_ops[1], t_ops[2])
        return out

    elif type == "stack":
        assert len(attr) == 1, "Stack should have 1 attr"
        t_ops = to_torch_operands(*ops)
        return torch.stack(t_ops, dim=attr[0])

    elif type == "interleave":
        assert len(attr) == 2, "Interleave should have 2 attr"
        axis = attr[0]
        stride = attr[1]
        assert axis == -3 and stride == 1

        # Forward impl only works for Z dim interleave with stride 1
        t_ops = to_torch_operands(*ops)
        stacked = torch.stack(t_ops, dim=axis)
        target_shape = list(t_ops[0].shape)
        for op in t_ops[1:]:
            target_shape[axis] += op.shape[axis]

        return torch.reshape(stacked, target_shape)
    assert False, f"Unknown eval: {type}"


# Return shape, and list of dimensions that were broadcast on operands
def shape(type, attr, ops) -> Tuple[Tuple, List]:
    def get_eltwise_shape_and_broadcast():
        broadcast = []
        output_dims = max(len(op) for op in ops)
        for index in range(len(ops)):
            ops[index] = list(ops[index])
            if len(ops[index]) < output_dims:
                ops[index] = [1] * (output_dims - len(ops[index])) + ops[index]

        output_shape = [max(dim) for dim in zip(*ops)]
        for op_index in range(len(ops)):
            for dim_index in range(len(ops[op_index])):
                if ops[op_index][dim_index] != output_shape[dim_index]:
                    assert (
                        ops[op_index][dim_index] == 1
                    ), f"Eltwise nary ops must have same shape or operand must be 1 wide to broadcast: {ops}"
                    broadcast.append(
                        (
                            op_index,
                            dim_index - len(output_shape),
                            output_shape[dim_index],
                        )
                    )

        return tuple(output_shape), broadcast

    if type == "conv_sum":
        shapes = []
        for op in ops:
            assert (
                len(op) <= 4
            ), "Shape of an operand must be smaller than or equal to 4"
            if len(op) < 4:
                op = (4 - len(op)) * (1,) + op
            if len(shapes) > 0:
                assert shapes[-1] == op, "Shapes of all operands must be the same size"
            shapes.append(op)

        return shapes[0], []

    elif type == "concatenate":
        assert len(attr) == 1, "Concatenate should have 1 attr"
        axis = attr[0]

        output_shape = list(ops[0])
        for op in ops[1:]:
            output_shape[axis] += op[axis]

        return output_shape, []

    elif type == "where":
        return get_eltwise_shape_and_broadcast()

    elif type == "index_copy":
        return get_eltwise_shape_and_broadcast()

    elif type == "stack":
        axis = attr[0]

        output_shape = list(ops[0])
        if axis == -1:
            output_shape.append(len(ops))
        elif axis < -1:
            # axis + 1 is added because insertion at the correct position requires adjusting for negative axes to ensure proper behavior.
            output_shape.insert(axis + 1, len(ops))
        else:
            output_shape.insert(axis, len(ops))
        return output_shape, []

    elif type == "interleave":
        assert len(attr) == 2, "Interleave should have 2 attr"
        axis = attr[0]
        stride = attr[1]
        assert axis == -3 and stride == 1

        output_shape = list(ops[0])
        for op in ops[1:]:
            output_shape[axis] += op[axis]

        return output_shape, []
    assert False, f"{type} not defined in eltwise_nary"


def lower(type, attr, lc, ops, outputs):

    if type == "conv_sum":
        return lc.op(type, ops, attr, {"a": 0})

    elif type == "concatenate":
        assert len(attr) == 1, "Concatenate should have 1 attr"
        axis = attr[0]
        return lc.op(Splice.create_concatenate(axis, list(op.shape for op in ops)), ops)

    elif type == "index_copy":
        assert len(attr) == 1, "index_copy should have 1 attr"
        dim = attr[0]

        forge_attrs = {
            "axis": dim,
        }

        return lc.op("index_copy", ops, attr, forge_attrs)

    elif type == "where":
        assert False, "Where is meant to be removed by consteval"

    elif type == "interleave":
        assert len(attr) == 2, "Interleave should have 2 attr"
        axis, stride = attr
        assert axis == -3 and stride == 1
        return lc.op(
            Splice.create_interleave(axis, stride, list(op.shape for op in ops)), ops
        )

    assert False, f"{type} not defined in eltwise_nary"


def backward(op_type, attr, ac, operand, inputs, output, grad):
    if op_type == "conv_sum":
        y = attr[0]
        x = attr[1]
        shifts = attr[2:]

        return ac.op(
            "conv_sum", [grad], [y, x, -shifts[operand * 2], -shifts[operand * 2 + 1]]
        )

    elif op_type == "concatenate":
        axis = attr[0]
        dim_offset = grad.shape[axis]

        index_offset = 0
        for i, input_ in enumerate(inputs):
            if operand is not i:
                index_offset += input_.shape[axis]
                continue
            return ac.op(
                "select",
                (grad,),
                (axis, index_offset, input_.shape[axis], dim_offset),
                named_attrs={
                    "dim": axis,
                    "begin": index_offset,
                    "length": input_.shape[axis],
                    "stride": dim_offset,
                },
            )

    elif op_type == "interleave":
        axis = attr[0]
        stride = attr[1]
        assert axis == -3 and stride == 1

        num_operands = len(inputs)
        result = grad
        if grad.shape[-1] % TILE_DIM != 0:
            result = ac.op("pad_tile", (result,), (-1, grad.shape[-1]))
        if grad.shape[-2] % TILE_DIM != 0:
            result = ac.op("pad_tile", (result,), (-2, grad.shape[-2]))
        result = ac.op("hstack", (result,), (num_operands,))
        if grad.shape[-2] % TILE_DIM != 0:
            result = ac.op(
                "narrow", (result,), (-2, 0, grad.shape[-2], result.shape[-2])
            )
        result = ac.op(
            "select",
            (result,),
            (
                -1,
                operand * align_up_tile(grad.shape[-1]),
                align_up_tile(grad.shape[-1]),
                result.shape[-1],
            ),
        )
        if grad.shape[-1] % TILE_DIM != 0:
            result = ac.op(
                "narrow", (result,), (-1, 0, grad.shape[-1], result.shape[-1])
            )
        return result

    assert False, f"{op_type} not defined in eltwise_nary"


def decompose(type, attr, dc, inputs):
    if type == "stack":
        assert len(attr) == 1, "Stack should have 1 attr"
        axis = attr[0]

        new_inputs = []
        for inp in inputs:
            inp_shape = inp.shape.as_list()
            if axis == -1:
                inp_shape.append(1)
            elif axis < -1:
                # axis + 1 is added because insertion at the correct position requires adjusting for negative axes to ensure proper behavior.
                inp_shape.insert(axis + 1, 1)
            else:
                inp_shape.insert(axis, 1)
            new_inp = dc.op_with_named_attrs(
                "reshape", [inp], {"shape": (*inp_shape,)}, (*inp_shape,)
            )
            new_inputs.append(new_inp)

        output = dc.op_with_named_attrs(
            "concatenate", new_inputs, {"dim": (axis)}, (axis,)
        )
        dc.fuse(output)

    if type == "concatenate":
        if len(inputs) == 1:
            dc.fuse(dc.op(Nop.create(), [inputs[0]]))


from math import gcd
from functools import reduce


def find_gcd(list):
    x = reduce(gcd, list)
    return x


def decompose_post_optimize(type, attr, dc, inputs):
    if type == "where":

        condition = inputs[0]
        x = inputs[1]
        y = inputs[2]
        one = dc.tensor(torch.ones((1,)))
        not_condition = dc.op("subtract", [one, condition])

        t0 = dc.op("multiply", [condition, x])
        t1 = dc.op("multiply", [not_condition, y])

        add = dc.op("add", [t0, t1])
        dc.fuse(add)
