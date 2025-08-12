# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


from forge._C import DataFormat
import torch

from forge.forgeglobal import TILE_DIM
from ..common import to_torch_operands, cast_for_cpu_eval


def eval(type, attr, ops):
    assert len(ops) in [2, 3], "Matrix multiply should have two or three inputs"
    assert len(attr) <= 2, "Matrix multiply should have zero to two attributes"

    accumulate = (len(attr) >= 1) and bool(attr[0])
    t_ops = to_torch_operands(*ops)
    t_ops, original_type = cast_for_cpu_eval(t_ops, type)

    if type == "matmul":
        result = torch.matmul(t_ops[0], t_ops[1])
        result = result.to(original_type)
        if len(t_ops) > 2:
            result += t_ops[2]  # bias

    if accumulate and len(result.shape) >= 3:
        result = torch.sum(result, dim=-3, keepdim=True)

    return result


def shape(type, attr, ops):
    assert len(ops) in [2, 3, 4], "Matrix multiply should have two or three inputs"
    assert len(attr) <= 2, "Matrix multiply should have zero to two attributes"

    accumulate = (len(attr) >= 1) and bool(attr[0])

    ops0_padding = 0
    ops[0] = list(ops[0])
    while len(ops[0]) < len(ops[1]):
        ops[0] = [1] + ops[0]
        ops0_padding += 1

    ops[1] = list(ops[1])
    ops1_padding = 0
    while len(ops[1]) < len(ops[0]):
        ops[1] = [1] + ops[1]
        ops1_padding += 1

    broadcast = []
    output_dim = []
    for dim in range(4, len(ops[0]) + 1):
        assert ops[0][-dim] == ops[1][-dim], f"Broadcast on dimensions beyond 3rd is not supported {ops} {dim}"
        output_dim.append(ops[0][-dim])

    # Z broadcast
    if len(ops[0]) >= 3:
        if ops[0][-3] != ops[1][-3]:
            if ops[0][-3] == 1:
                broadcast.append((0, len(ops[0]) - 3, ops[1][-3]))
                output_dim.append(ops[1][-3])
            elif ops[1][-3] == 1:
                broadcast.append((1, len(ops[0]) - 3, ops[0][-3]))
                output_dim.append(ops[0][-3])
            else:
                assert False, "If Z dimension is not the same for matmul, one of operands must have it be 1."
        else:
            output_dim.append(ops[0][-3])

    # Inner dim broadcast
    if ops[0][-1] != ops[1][-2]:
        if ops[0][-1] == 1:
            broadcast.append((0, len(ops[0]) - 1 - ops0_padding, ops[1][-2]))
        elif ops[1][-2] == 1:
            broadcast.append((1, len(ops[0]) - 2 - ops1_padding, ops[0][-1]))
        else:
            assert (
                False
            ), f"If inner dimension is not the same for matmul, one of operands must have it be 1, shapes are: {ops}"

    output_dim.extend([ops[0][-2], ops[1][-1]])
    if accumulate:
        assert len(output_dim) >= 3
        output_dim[-3] = 1

    return output_dim, broadcast


def decompose(type, attr, dc, inputs):
    pass


def backward(type, attr, ac, operand, inputs, output, grad):

    assert len(inputs) == 2, "Matrix multiply should have two inputs"
    assert operand < 2, "Invalid operand index"
    assert len(attr) <= 2, "Matrix multiply should have zero to two attributes"

    in0 = inputs[0]
    in1 = inputs[1]

    if operand == 0:
        shape_len = len(ac.get_shape(in1))
        in1t = ac.op_with_named_attrs("transpose", [in1], {"dim0": -2, "dim1": -1})
        return ac.op("matmul", (grad, in1t))

    if operand == 1:
        shape_len = len(ac.get_shape(in0))
        in0t = ac.op_with_named_attrs("transpose", [in0], {"dim0": -2, "dim1": -1})
        return ac.op("matmul", (in0t, grad))


def initial_flops_estimate(type, attr, ops):
    macc = 0
    if type == "matmul":
        output_shape = shape(type, attr, ops)[0]
        macc = output_shape[-1] * output_shape[-2]
        if len(output_shape) > 2:
            macc *= output_shape[-3]
        if len(output_shape) > 3:
            macc *= output_shape[-4]
        macc *= ops[0][-1]

    flops = macc * 2
    return flops
