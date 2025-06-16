# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
from typing import List, Tuple
from forge.forgeglobal import TILE_DIM
from forge.tensor import Tensor
import numpy as np
import torch
from .transpose import TransposeTM
from .reciprocal import Reciprocal
from .log import Log
from .nop import Nop

from ..common import to_torch_operands
from forge.utils import align_up_tile
from forge.op.eval.common import calculate_tile_size


def eval(type, attr, ops):
    assert len(ops) == 2, "Eltwise binary should have two inputs"
    assert len(attr) == 0, "Eltwise binary should have no attributes"

    t_ops = to_torch_operands(*ops)

    f = {
        "add": lambda i: torch.add(t_ops[0], t_ops[1]),
        "divide": lambda i: torch.divide(t_ops[0], t_ops[1]),
        "subtract": lambda i: torch.subtract(t_ops[0], t_ops[1]),
        "multiply": lambda i: torch.multiply(t_ops[0], t_ops[1]),
        "maximum": lambda i: torch.maximum(t_ops[0], t_ops[1]),
        "minimum": lambda i: torch.minimum(t_ops[0], t_ops[1]),
        "heaviside": lambda i: torch.heaviside(t_ops[0], t_ops[1]),
        "power": lambda i: torch.pow(t_ops[0], t_ops[1]),
        "greater": lambda i: torch.gt(t_ops[0], t_ops[1]).to(t_ops[0].dtype),
        "greater_equal": lambda i: torch.ge(t_ops[0], t_ops[1]).to(t_ops[0].dtype),
        "less": lambda i: torch.lt(t_ops[0], t_ops[1]).to(t_ops[0].dtype),
        "less_equal": lambda i: torch.le(t_ops[0], t_ops[1]).to(t_ops[0].dtype),
        "equal": lambda i: torch.eq(t_ops[0], t_ops[1]).to(t_ops[0].dtype),
        "not_equal": lambda i: torch.ne(t_ops[0], t_ops[1]).to(t_ops[0].dtype),
        "logical_and": lambda i: torch.logical_and(t_ops[0], t_ops[1]).to(t_ops[0].dtype),
        "remainder": lambda i: torch.remainder(t_ops[0], t_ops[1]),
    }
    assert type in f, f"{type} not defined in eval map for eltwise binary ops."

    return f[type](t_ops)


# Return shape, and list of dimensions that were broadcast on operands
def shape(type, attr, ops) -> Tuple[Tuple, List]:
    assert len(ops) == 2, "Eltwise binary should have two inputs"
    assert len(attr) == 0, "Eltwise binary should have no attributes"

    broadcast = []
    output_shape = []

    ops[0] = list(ops[0])
    while len(ops[0]) < len(ops[1]):
        ops[0] = [1] + ops[0]

    ops[1] = list(ops[1])
    while len(ops[1]) < len(ops[0]):
        ops[1] = [1] + ops[1]

    for dim in range(len(ops[0])):
        if ops[0][dim] != ops[1][dim]:
            if ops[1][dim] == 1:
                broadcast.append((1, dim - len(ops[1]), ops[0][dim]))  # Convert to negative indexing
                output_shape.append(ops[0][dim])
            else:
                assert (
                    ops[0][dim] == 1
                ), f"Eltwise binary ops must have the same shape in both inputs, or one operand must be 1 wide to broadcast: {ops[0]} vs {ops[1]}"
                broadcast.append((0, dim - len(ops[0]), ops[1][dim]))  # Convert to negative indexing
                output_shape.append(ops[1][dim])
        else:
            output_shape.append(ops[0][dim])

    return tuple(output_shape), broadcast


def lower(type, attr, lc, ops, outputs):
    # TODO: Implement mlir lowering here.
    assert False


def backward(op_type, attr, ac, operand, inputs, output, grad):

    assert len(inputs) == 2, "Eltwise binary should have two inputs"
    assert len(attr) == 0, "Eltwise binary should have no attributes"
    assert operand < 2, "Invalid operand index"

    # Some operands are implicitly broadcasted, so their shapes in backward() need to be unbroadcasted for grad accumulation

    shapes = [inputs[0].shape.as_list(), inputs[1].shape.as_list()]

    # Pad to longer dims
    longer_dims = max(len(s) for s in shapes)
    shapes = [[1] * (longer_dims - len(s)) + s for s in shapes]
    # Pad gradient shape to longer dims
    grad_shape = [1] * (longer_dims - len(grad.shape.as_list())) + grad.shape.as_list()
    grad_shape_len = len(grad_shape)

    if op_type == "add":
        if inputs[operand].shape != grad.shape:
            for i in range(len(shapes[operand])):
                if shapes[operand][i] < grad_shape[i]:
                    # Negative indexing for reduce axis
                    grad = ac.op(
                        "reduce_sum",
                        (grad,),
                        (i - grad_shape_len,),
                        {"keep_dim": True, "dim_arg": [i - grad_shape_len]},
                    )
        return ac.op(Nop.create(), (grad,))  # pass gradient through

    elif op_type == "subtract":
        if inputs[operand].shape != grad.shape:
            for i in range(len(shapes[operand])):
                if shapes[operand][i] < grad.shape[i]:
                    grad = ac.op("reduce_sum", (grad,), (i,), {"keep_dim": True, "dim_arg": [i]})
        if operand == 0:
            return ac.op(Nop.create(), (grad,))
        else:
            return ac.op("multiply", (grad, ac.constant(-1)))

    elif op_type == "multiply":
        op_grad = ac.op("multiply", (grad, inputs[1 - operand]))
        if inputs[operand].shape != grad.shape:
            for i in range(len(shapes[operand])):
                if shapes[operand][i] < grad_shape[i]:
                    op_grad = ac.op(
                        "reduce_sum",
                        (op_grad,),
                        (i - grad_shape_len,),
                        {"keep_dim": True, "dim_arg": [i - grad_shape_len]},
                    )
        return op_grad

    elif op_type == "maximum":
        # TODO
        return ac.op(Nop.create(), (grad,))  # pass gradient through

    elif op_type == "power":
        if operand == 0:  # dx = y * (x^y) * recp(x)
            recip = ac.op(Reciprocal.create(), (inputs[0],))
            partial_grad = ac.op("multiply", (output, recip))
            pow_grad = ac.op("multiply", (inputs[1], partial_grad))
        if operand == 1:  # dy = (x^y) * ln(x)
            ln_x = ac.op(Log.create(), [inputs[0]])
            pow_grad = ac.op("multiply", (output, ln_x))
        return ac.op("multiply", (pow_grad, grad))

    assert False, f"{op_type} not defined in eltwise binary backward."


def decompose(op_type, attr, dc, inputs):
    if op_type == "divide":
        recip = dc.op(Reciprocal.create(), [inputs[1]])
        result = dc.op("multiply", [inputs[0], recip])
        dc.fuse(result)
        return
    # Can be used if backend don't implement maximum op in the future.
    #
    # assert len(inputs) == 2, "Eltwise binary should have two inputs"
    # if op_type == "maximum":
    #     x = inputs[0]
    #     y = inputs[1]

    #     a_ge = dc.op("greater_equal", (x, y))
    #     b_lt = dc.op("less", (x, y))
    #     a_ge_val = dc.op("multiply", (x, a_ge))
    #     b_lt_val = dc.op("multiply", (y, b_lt))
    #     res = dc.op("add", (a_ge_val, b_lt_val))

    #     dc.fuse(res)
    #     return

    ops0_dims = len(inputs[0].shape)
    ops1_dims = len(inputs[1].shape)
    if ops0_dims > ops1_dims and ops0_dims == 5:
        ops1 = dc.op("reshape", [inputs[1]], list(inputs[0].shape))
        result = dc.op(op_type, [inputs[0], ops1])
        dc.fuse(result)
    elif ops1_dims > ops0_dims and ops1_dims == 5:
        ops0 = dc.op("reshape", [inputs[0]], list(inputs[1].shape))
        result = dc.op(op_type, [ops0, inputs[1]])
        dc.fuse(result)


def decompose_post_autograd(op_type, attr, dc, inputs):
    assert len(inputs) == 2, "Eltwise binary should have two inputs"
    if op_type == "heaviside":
        x = inputs[0]
        y = inputs[1]
        shape = x.shape.as_list()
        zero = dc.tensor(torch.zeros(shape))
        x_gt = dc.op("greater", (x, zero))
        x_eq = dc.op("equal", (x, zero))
        res = dc.op("multiply", (x_eq, y))
        res = dc.op("add", (res, x_gt))
        dc.fuse(res)
        return
    elif op_type == "maximum" and os.environ.get("FORGE_ENABLE_MAXIMUM_DECOMPOSITION", "0") == "1":
        operand0, operand1 = inputs[0], inputs[1]
        orig_op0_shape = operand0.shape.as_list()
        orig_op1_shape = operand1.shape.as_list()
        vslice_op0, vslice_op1 = False, False
        slice_factor = None

        if len(orig_op1_shape) > 2 and orig_op1_shape[-3] != 1:
            slice_factor = orig_op1_shape[-3]
            vslice_op1 = True

        if len(orig_op0_shape) > 2 and orig_op0_shape[-3] != 1:
            slice_factor = orig_op0_shape[-3]
            vslice_op0 = True

        if vslice_op0 and vslice_op1:
            assert orig_op0_shape[-3] == orig_op1_shape[-3]

        op0_shape = operand0.shape.as_list()
        op1_shape = operand1.shape.as_list()

        max_operand_nd = max(len(op0_shape), len(op1_shape), 3)
        while len(operand0.shape) < max_operand_nd:
            operand0 = dc.op_with_named_attrs("unsqueeze", [operand0], {"dim": 0}, (0, len(operand0.shape)))
        while len(operand1.shape) < max_operand_nd:
            operand1 = dc.op_with_named_attrs("unsqueeze", [operand1], {"dim": 0}, (0, len(operand1.shape)))

        if slice_factor != None:
            concat_z = dc.op("interleave", [operand0, operand1], (-3, 1))
            result = dc.op("reduce_max", [concat_z], (-3, 2))
        else:
            concat_z = dc.op("concatenate", [operand0, operand1], (-3,))
            result = dc.op("reduce_max", [concat_z], (-3,))

        while len(result.shape) > max_operand_nd:
            result = dc.op("squeeze", [result], (0,))

        dc.fuse(result)
        return
    else:
        ops0_dims = len(inputs[0].shape)
        ops1_dims = len(inputs[1].shape)
        if ops0_dims > ops1_dims and ops0_dims == 5:
            ops1 = dc.op("reshape", [inputs[1]], list(inputs[0].shape))
            result = dc.op(op_type, [inputs[0], ops1])
            dc.fuse(result)
        elif ops1_dims > ops0_dims and ops1_dims == 5:
            ops0 = dc.op("reshape", [inputs[0]], list(inputs[1].shape))
            result = dc.op(op_type, [ops0, inputs[1]])
            dc.fuse(result)


def decompose_post_optimize(op_type, attr, dc, inputs):
    pass


def initial_flops_estimate(type, attr, ops):
    flops = 0
    output_shape = shape(type, attr, ops)[0]
    if type in ["add", "subtract", "power", "maximum", "minumum", "multiply"]:
        flops = np.prod(output_shape)

    return flops
