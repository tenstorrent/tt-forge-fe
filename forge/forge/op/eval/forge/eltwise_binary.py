# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple
from forge.forgeglobal import TILE_DIM
from forge.tensor import Tensor
import numpy as np
import torch

from ..common import to_torch_operands
from forge.utils import align_up_tile
from forge.op.eval.common import calculate_tile_size


def eval(type, attr, ops):
    assert len(ops) == 2, "Eltwise binary should have two inputs"
    assert len(attr) == 0, "Eltwise binary should have no attributes"

    t_ops = to_torch_operands(*ops)

    f = {
        "maximum": lambda i: torch.maximum(t_ops[0], t_ops[1]),
        "minimum": lambda i: torch.minimum(t_ops[0], t_ops[1]),
        "heaviside": lambda i: torch.heaviside(t_ops[0], t_ops[1]),
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

    if op_type == "maximum":
        # TODO
        return ac.op("nop", (grad,))  # pass gradient through

    assert False, f"{op_type} not defined in eltwise binary backward."


def decompose(op_type, attr, dc, inputs):
    pass


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


def decompose_post_optimize(op_type, attr, dc, inputs):
    pass


def initial_flops_estimate(type, attr, ops):
    flops = 0
    output_shape = shape(type, attr, ops)[0]
    if type in ["add", "maximum", "minumum"]:
        flops = np.prod(output_shape)

    return flops
