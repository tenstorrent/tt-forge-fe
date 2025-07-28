# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional
from loguru import logger
from ..common import to_torch_operands
from ....forgeglobal import TILE_DIM
from ....tensor import forge_dataformat_to_pytorch_dtype
import numpy as np
from forge.op.eval.common import calculate_tile_size
from .tanh import Tanh
from .nop import Nop


def eval(type, attr, ops):
    assert len(ops) == 1, "Eltwise unary should have one input"
    assert (
        len(attr) == 0
        or (type == "clip" and len(attr) == 2)
        or (type == "erf" and len(attr) == 0)
        or (type == "cumsum" and len(attr) == 2)
        or (type == "pow" and len(attr) == 1)
    ), "Eltwise unary should have no attributes, execpt for clip, relu, cumsum, dropout and pow"

    t_ops = to_torch_operands(*ops)

    original_types = [o.dtype for o in t_ops]

    f = {
        "erf": lambda i: torch.erf(i[0]),
        "nop": lambda i: i[0],
        "tilizer": lambda i: i[0],
        "ethernet_datacopy": lambda i: i[0],
        "clip": lambda i: torch.clip(i[0], min=attr[0], max=attr[1]),
        "abs": lambda i: torch.abs(i[0]),
        "tanh": lambda i: torch.tanh(i[0]),
        "cumsum": lambda i: torch.cumsum(i[0], dim=attr[0]),
        "pow": lambda i: torch.pow(i[0], attr[0]),
    }

    assert type in f, f"{type} not defined in eval map for eltwise unary ops."

    ret = f[type](t_ops)
    if ret.dtype != original_types[0]:
        ret = ret.type(original_types[0])

    return ret


def shape(type, attr, ops):
    assert len(ops) == 1, "Eltwise unary should have one input"
    assert (
        len(attr) == 0
        or (type == "ethernet_datacopy" and (len(attr) == 1 or len(attr) == 2))
        or (type == "clip" and len(attr) == 2)
        or (type == "cumsum" and len(attr) == 2)
        or (type == "dropout" and len(attr) == 3)
        or (type == "pow" and len(attr) == 1)
    ), "Eltwise unary should have no attributes, execpt for clip, relu, cumsum, dropout, and pow"

    return ops[0], []


def backward(type, attr, ac, operand, inputs, output, grad):

    assert len(inputs) == 1, "Eltwise unary should have one input"
    assert operand == 0, "Invalid operand index"
    assert (
        len(attr) == 0
        or (type == "clip" and len(attr) == 2)
        or (type == "cumsum" and len(attr) == 2)
        or (type == "dropout" and len(attr) == 3)
        or (type == "pow" and len(attr) == 1)
    ), "Eltwise unary should have no attributes, execpt for clip, relu, cumsum, dropout and pow"

    if type == "nop":
        return ac.op(Nop.create(), (grad,))

    if type == "tilizer":
        return ac.op(Nop.create(), (grad,))

    if type == "tanh":
        tanh_square = ac.op("multiply", (output, output))
        subtract = ac.op("subtract", (ac.constant(1), tanh_square))
        res = ac.op("multiply", (subtract, grad))
        return res

    if type == "cumsum":
        dim = attr[0]

        assert dim == 0, "Unsupported dim different then 0 for cumulative sum backward pass"

        if dim == 0:
            return ac.op(Nop.create(), (grad,))

        return res

    if type == "dropout":
        return ac.op_with_named_attrs("dropout", (grad,), {"p": attr[0], "training": attr[1], "seed": attr[2]}, attr)

    if type == "clip":
        x = inputs[0]
        shape = x.shape.as_list()
        min_value = attr[0]
        max_value = attr[1]
        min_value_tensor = ac.tensor(torch.zeros(shape) + min_value)
        max_value_tensor = ac.tensor(torch.zeros(shape) + max_value)

        ge_x = ac.op("greater_equal", (x, min_value_tensor))
        le_x = ac.op("less_equal", (x, max_value_tensor))
        mask = ac.op("multiply", (ge_x, le_x))
        res = ac.op("multiply", (mask, grad))
        return res

    elif type == "pow":
        exponent_value = attr[0]
        shape = list(inputs[0].shape.as_list())
        recip = ac.op("reciprocal", (inputs[0],))
        partial_grad = ac.op("multiply", (output, recip))
        pow_grad = ac.op("multiply", (ac.tensor(torch.zeros(shape) + exponent_value), partial_grad))
        return ac.op("multiply", (pow_grad, grad))

    elif type == "abs":

        heaviside = ac.op("heaviside", (inputs[0], ac.constant(0.5)))
        subtract = ac.op("subtract", (heaviside, ac.constant(0.5)))
        stretched = ac.op("multiply", (subtract, ac.constant(2.0)))
        return ac.op("multiply", (stretched, grad))

    assert False, f"{type} not defined in eltwise unary backward."


def decompose(type, attr, dc, inputs):
    pass


def initial_flops_estimate(type, attr, ops):
    flops = 0
    sfpu_unary_ops = [
        "sqrt",
        "abs",
        "tanh",
        "cumsum",
        "pow",
    ]
    output_shape = shape(type, attr, ops)[0]

    if type in sfpu_unary_ops:
        flops = np.prod(output_shape)

    return flops
