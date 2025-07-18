# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os

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
from .exp import Exp
from .reciprocal import Reciprocal

M_2_SQRTPI = 1.12837916709551257390  # 2/sqrt(pi)
M_SQRT2 = 1.41421356237309504880  # sqrt(2)
M_SQRT1_2 = 0.7071067811865476

# Reference implementation is at pytorch/aten/src/ATen/native/cpu/Activation.cpp
# https://github.com/pytorch/pytorch/blob/4f8b986e28736b59bc46cd0873a0f36fdaa6f5b8/aten/src/ATen/native/cpu/Activation.cpp
def gelu_derivative(x, approximate):
    if approximate == "none":
        cdf = 0.5 * (1 + torch.erf(x * M_SQRT1_2))
        pdf = 0.5 * M_SQRT1_2 * M_2_SQRTPI * torch.exp(x * x * -0.5)
        return cdf + x * pdf
    elif approximate == "tanh":
        intermediate_0 = 0.5 * (1 + torch.tanh((M_2_SQRTPI / M_SQRT2) * (x + 0.044715 * torch.pow(x, 3))))
        intermediate_1 = x * torch.exp(-0.5 * x * x) * (0.5 * M_2_SQRTPI / M_SQRT2)
        return intermediate_0 + intermediate_1
    else:
        raise RuntimeError(f"Gelu does not support {approximate} approximation mode.")


def gelu_forward(x, approximate):
    if approximate == "none":
        return torch.nn.functional.gelu(x)
    elif approximate == "tanh":
        import math

        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    else:
        raise RuntimeError(f"Gelu does not support {approximate} approximation mode.")


def tile_broadcast(attr, i):
    dim, size = attr
    while len(i.shape) <= ((-dim - 1) if dim < 0 else dim):
        i = i.unsqueeze(0)
    shape = list(i.shape)
    shape[dim] = size
    return torch.broadcast_to(i, shape)


def eval(type, attr, ops):
    assert len(ops) == 1, "Eltwise unary should have one input"
    assert (
        len(attr) == 0
        or (type == "clip" and len(attr) == 2)
        or (type == "erf" and len(attr) == 0)
        or (type == "leaky_relu" and len(attr) == 1)
        or (type == "relu" and len(attr) <= 2)
        or (type == "cumsum" and len(attr) == 2)
        or (type == "dropout" and len(attr) == 3)
        or (type == "tile_broadcast" and len(attr) == 2)
        or (type == "gelu" and len(attr) == 1)
        or (type == "gelu_derivative" and len(attr) == 1)
        or (type == "pow" and len(attr) == 1)
    ), "Eltwise unary should have no attributes, execpt for clip, leaky_relu, and cumsum"

    t_ops = to_torch_operands(*ops)

    # Some ops don't support non-fp32 in pytorch
    original_types = [o.dtype for o in t_ops]
    if original_types[0] != torch.float32:
        if type in ["gelu", "gelu_derivative"]:
            t_ops = tuple(t.type(torch.float32) for t in t_ops)

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
        "exp": lambda i: torch.exp(i[0]),
        "sqrt": lambda i: torch.sqrt(i[0]),
        # "relu": lambda i: i[0] * (i[0] >= relu_threshold).to(i[0].dtype),
        "leaky_relu": lambda i: torch.nn.functional.leaky_relu(i[0], attr[0]),
        "gelu": lambda i: gelu_forward(i[0], approximate=attr[0]),
        "gelu_derivative": lambda i: gelu_derivative(i[0], approximate=attr[0]),
        "nop": lambda i: i[0],
        "tilizer": lambda i: i[0],
        "ethernet_datacopy": lambda i: i[0],
        "buffer": lambda i: i[0],
        "reciprocal": lambda i: torch.reciprocal(i[0] + 1e-10),  # add epsilon to avoid infinity
        "log": lambda i: torch.log(i[0] + 1e-10),  # add epsilon to avoid nan
        "sigmoid": lambda i: torch.sigmoid(i[0]),
        "clip": lambda i: torch.clip(i[0], min=attr[0], max=attr[1]),
        "abs": lambda i: torch.abs(i[0]),
        "atan": lambda i: torch.atan(i[0]),
        "tile_broadcast": lambda i: tile_broadcast(attr, i[0]),
        "tanh": lambda i: torch.tanh(i[0]),
        "cumsum": lambda i: torch.cumsum(i[0], dim=attr[0]),
        "logical_not": lambda i: torch.logical_not(i[0]),
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
        or (type == "leaky_relu" and len(attr) == 1)
        or (type == "relu" and len(attr) <= 2)
        or (type == "cumsum" and len(attr) == 2)
        or (type == "dropout" and len(attr) == 3)
        or (type == "tile_broadcast" and len(attr) == 2)
        or (type == "gelu" and len(attr) == 1)
        or (type == "gelu_derivative" and len(attr) == 1)
        or (type == "pow" and len(attr) == 1)
    ), "Eltwise unary should have no attributes, execpt for clip, leaky_relu and cumsum"

    if type == "tile_broadcast":
        assert len(attr) == 2, "Tile broadcast should have two attributes - dim and size"
        dim = attr[0]
        size = attr[1]
        shape = len(ops[0].shape)
        shape[dim] = size
        return shape, []

    return ops[0], []


def backward(type, attr, ac, operand, inputs, output, grad):

    assert len(inputs) == 1, "Eltwise unary should have one input"
    assert operand == 0, "Invalid operand index"
    assert (
        len(attr) == 0
        or (type == "clip" and len(attr) == 2)
        or (type == "leaky_relu" and len(attr) == 1)
        or (type == "relu" and len(attr) <= 2)
        or (type == "cumsum" and len(attr) == 2)
        or (type == "dropout" and len(attr) == 3)
        or (type == "tile_broadcast" and len(attr) == 2)
        or (type == "gelu" and len(attr) == 1)
        or (type == "pow" and len(attr) == 1)
    ), "Eltwise unary should have no attributes, execpt for clip, leaky_relu and cumsum"

    if type == "nop":
        return ac.op(Nop.create(), (grad,))

    if type == "tilizer":
        return ac.op(Nop.create(), (grad,))

    if type == "tile_broadcast":  # the full TM broadcast will generate a reduce
        return ac.op(Nop.create(), (grad,))

    if type == "buffer":
        return ac.op(Buffer.create(), (grad,))

    if type == "exp":
        return ac.op("multiply", (output, grad))

    if type == "reciprocal":  # -1/x^2
        sq = ac.op("multiply", (output, output))
        neg = ac.op("multiply", (sq, ac.constant(-1)))
        return ac.op("multiply", (neg, grad))

    if type == "sqrt":  # 0.5 / f(x)
        rec = ac.op(Reciprocal.create(), (output,))
        mult = ac.op("multiply", (rec, ac.constant(0.5)))
        return ac.op("multiply", (mult, grad))

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

    if type == "leaky_relu":
        alpha = ac.constant(attr[0])

        relu_dx = ac.op("heaviside", (output, ac.constant(0.0)))

        l_relu_dx = ac.op("multiply", (output, ac.constant(-1.0)))
        l_relu_dx = ac.op("heaviside", (l_relu_dx, ac.constant(0.0)))
        l_relu_dx = ac.op("multiply", (l_relu_dx, alpha))
        l_relu_dx = ac.op("add", (relu_dx, l_relu_dx))

        res = ac.op("multiply", (l_relu_dx, grad))

        return res

    if type == "gelu":
        gelud = ac.op_with_named_attrs("gelu_derivative", (inputs[0],), {"approximate": attr[0]}, attr)
        return ac.op("multiply", (gelud, grad))

    if type == "log":
        recip = ac.op(Reciprocal.create(), (inputs[0],))
        return ac.op("multiply", (recip, grad))

    if type == "sigmoid":
        sigm_ = ac.op("subtract", (ac.constant(1), output))
        dsigm = ac.op("multiply", (output, sigm_))
        res = ac.op("multiply", (dsigm, grad))
        return res

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
        recip = ac.op(Reciprocal.create(), (inputs[0],))
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
    if type == "sigmoid" and bool(int(os.environ.get("FORGE_DECOMPOSE_SIGMOID", "0"))):
        inp = inputs[0]
        minus_one = dc.tensor(torch.ones([1, 1]) * -1)
        plus_one = dc.tensor(torch.ones([1, 1]))
        neg_ = dc.op("multiply", [inp, minus_one])
        exp_ = dc.op(Exp.create(), [neg_])
        result = dc.op("add", [plus_one, exp_])
        result = dc.op(Reciprocal.create(), [result])
        dc.fuse(result)

    elif type == "gelu" and bool(int(os.environ.get("FORGE_DECOMPOSE_GELU", "0"))):
        inp_node = inputs[0]
        data_type = forge_dataformat_to_pytorch_dtype(inp_node.output_df)
        one_half = dc.tensor(torch.ones((1), dtype=data_type) * 0.5)
        sqrt_2pi = dc.tensor(torch.ones((1), dtype=data_type) * 0.79788)
        one = dc.tensor(torch.ones((1), dtype=data_type))
        const = dc.tensor(torch.ones((1), dtype=data_type) * 0.044715)
        x_squared = dc.op("multiply", [inp_node, inp_node])
        x_cubed = dc.op("multiply", [inp_node, x_squared])
        x_cuber_times_const = dc.op("multiply", [x_cubed, const])
        plus_x = dc.op("add", [x_cuber_times_const, inp_node])
        times_sqrt_2pi = dc.op("multiply", [plus_x, sqrt_2pi])
        tanh = dc.op(Tanh.create(), [times_sqrt_2pi])
        plus_one = dc.op("add", [tanh, one])
        times_x = dc.op("multiply", [plus_one, inp_node])
        result = dc.op("multiply", [times_x, one_half])
        dc.fuse(result)


def initial_flops_estimate(type, attr, ops):
    flops = 0
    sfpu_unary_ops = [
        "exp",
        "sqrt",
        "relu",
        "leaky_relu",
        "gelu",
        "gelu_derivative",
        "reciprocal",
        "log",
        "sigmoid",
        "abs",
        "tanh",
        "cumsum",
        "pow",
    ]
    output_shape = shape(type, attr, ops)[0]

    if type in sfpu_unary_ops:
        flops = np.prod(output_shape)

    return flops
