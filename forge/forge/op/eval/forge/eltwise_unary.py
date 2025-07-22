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
from .buffer import Buffer


def eval(type, attr, ops):
    assert len(ops) == 1, "Eltwise unary should have one input"
    assert (
        len(attr) == 0
        or (type == "clip" and len(attr) == 2)
        or (type == "erf" and len(attr) == 0)
        or (type == "relu" and len(attr) <= 2)
        or (type == "cumsum" and len(attr) == 2)
        or (type == "dropout" and len(attr) == 3)
        or (type == "pow" and len(attr) == 1)
    ), "Eltwise unary should have no attributes, execpt for clip, relu, cumsum, dropout and pow"

    t_ops = to_torch_operands(*ops)

    original_types = [o.dtype for o in t_ops]

    if type == "dropout":
        p, training, seed = attr
        rng_state = torch.get_rng_state()
        torch.manual_seed(seed)
        ret = torch.nn.functional.dropout(t_ops[0], p=p, training=bool(training))
        torch.set_rng_state(rng_state)
        return ret

    if type == "relu":

        def relu(x, threshold):
            return x * (x >= threshold).to(x.dtype)

        def inv_relu(x, threshold):
            ir = threshold * (x >= threshold).to(x.dtype) + x * (~(x >= threshold)).to(x.dtype)
            return relu(ir, 0.0)

        x = t_ops[0]
        if len(attr) > 0:
            threshold = attr[0]
        else:
            threshold = 0.0
        if len(attr) > 1:
            mode = attr[1]
        else:
            mode = "min"

        if mode == "min":
            return relu(x, threshold)
        else:
            return inv_relu(x, threshold)

    # relu_threshold = attr[0] if len(attr) > 0 else 0.0
    f = {
        "erf": lambda i: torch.erf(i[0]),
        # "relu": lambda i: i[0] * (i[0] >= relu_threshold).to(i[0].dtype),
        "nop": lambda i: i[0],
        "tilizer": lambda i: i[0],
        "ethernet_datacopy": lambda i: i[0],
        "buffer": lambda i: i[0],
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
        or (type == "relu" and len(attr) <= 2)
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
        or (type == "relu" and len(attr) <= 2)
        or (type == "cumsum" and len(attr) == 2)
        or (type == "dropout" and len(attr) == 3)
        or (type == "pow" and len(attr) == 1)
    ), "Eltwise unary should have no attributes, execpt for clip, relu, cumsum, dropout and pow"

    if type == "nop":
        return ac.op(Nop.create(), (grad,))

    if type == "tilizer":
        return ac.op(Nop.create(), (grad,))

    if type == "buffer":
        return ac.op(Buffer.create(), (grad,))

    if type == "relu":
        # set theashold
        threshold = 0.0
        shape = inputs[0].shape.as_list()
        if len(attr) > 0:
            f32_epsilon = 1.19209289551e-07
            threshold = attr[0] - f32_epsilon
        threshold_tensor = ac.tensor(torch.zeros(shape) + threshold)

        # handle different modes
        mode = "min"
        if len(attr) > 1:
            mode = attr[1]

        if mode == "min":
            relud = ac.op("greater_equal", (inputs[0], threshold_tensor))
        else:
            l_relud = ac.op("less", (inputs[0], threshold_tensor))
            g_relud = ac.op("greater_equal", (inputs[0], ac.tensor(torch.zeros(shape))))
            relud = ac.op("multiply", (l_relud, g_relud))

        return ac.op("multiply", (relud, grad))

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
        "relu",
        "abs",
        "tanh",
        "cumsum",
        "pow",
    ]
    output_shape = shape(type, attr, ops)[0]

    if type in sfpu_unary_ops:
        flops = np.prod(output_shape)

    return flops
