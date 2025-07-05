# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


from forge._C import DataFormat
import torch

from forge.forgeglobal import TILE_DIM
from ..common import to_torch_operands, cast_for_cpu_eval
from ..sparse_utils import (
    transpose_sparse_picker_matrix,
    create_sparse_forge,
    shapeify_sparse_tiles_and_encodings,
    is_kernel_fracturing_candidate,
)
from forge.utils import round_up_div
from forge.op.eval.common import calculate_tile_size
from .transpose import TransposeTM


def eval(type, attr, ops):
    assert type == "sparse_matmul", "Only sparse_matmul supported in Python implementation"
    assert len(ops) in [2, 3], "Matrix multiply should have two or three inputs"
    assert len(attr) <= 2, "Matrix multiply should have zero to two attributes"

    accumulate = (len(attr) >= 1) and bool(attr[0])
    t_ops = to_torch_operands(*ops)
    t_ops, original_type = cast_for_cpu_eval(t_ops, type)

    a = t_ops[0]
    b = t_ops[1]

    assert len(t_ops) == 2, "Sparse matmul can't have a fused bias"
    assert a.is_sparse

    if len(a.shape) == 2:
        if len(b.shape) == 2:
            return torch.sparse.mm(a, b)
        else:
            has_w = len(b.shape) == 4
            if has_w:
                b = b.squeeze(0)

            if b.shape[-3] != 1:
                bcast_amount = b.shape[-3]
                a = torch.stack([a] * bcast_amount)
            else:
                a = a.unsqueeze(0)

            result = torch.bmm(a, b)

            if has_w:
                result = result.unsqueeze(0)
    else:
        assert len(a.shape) >= 3
        assert a.shape[-3] == 1 or b.shape[-3] == 1 or b.shape[-3] == a.shape[-3]
        has_w = len(a.shape) == 4
        while len(a.shape) < 4:
            a = a.unsqueeze(0)

        while len(b.shape) < 4:
            b = b.unsqueeze(0)

        if a.shape[-3] == 1:
            bcast_amount = b.shape[-3]
            a = torch.cat([a] * bcast_amount, dim=-3)
        elif b.shape[-3] == 1:
            broadcast_shape = list(b.shape)
            broadcast_shape[-3] = a.shape[-3]
            b = torch.broadcast_to(b, broadcast_shape)
        else:
            assert b.shape[-3] == a.shape[-3]

        if has_w:
            w = a.shape[-4]
            results = []
            for i in range(w):
                results.append(torch.bmm(a[i], b[i]))
            result = torch.stack(results)
        else:
            result = torch.bmm(a[0], b[0])

    result = result.to(original_type)

    if accumulate and len(result.shape) >= 3:
        result = torch.sum(result, dim=-3, keepdim=True)

    return result


def shape(type, attr, ops):
    assert type == "sparse_matmul", "Only sparse_matmul shape supported in Python implementation"
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
                # Sparse matmul can automatically handle broadcast in this case
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
            assert ops[0][-1] == ops[1][-2] * ops[1][-3], "Inner dimensions don't match for sparse matmul."

    output_dim.extend([ops[0][-2], ops[1][-1]])
    if accumulate:
        assert len(output_dim) >= 3
        output_dim[-3] = 1

    return output_dim, broadcast


def decompose(type, attr, dc, inputs):
    assert type == "sparse_matmul", "Only sparse_matmul decompose supported in Python implementation"
    # Special case decomp where RHS bcast over LHS Z dim i.e. in0.z > 1 and in1.z == 1
    # Sparse matmul can handle this case natively and this path enables better streaming
    in0 = inputs[0]
    in1 = inputs[1]
    picker = dc.get_pytorch_tensor(in0)
    zdim = 1 if len(picker.shape) < 3 else picker.shape[-3]

    accumulate = (len(attr) >= 1) and bool(attr[0])
    z_bcast_factor = zdim if (zdim > 1 and in1.shape[-3] == 1) else 1

    # In case of convolutions, z_bcast_factor is the volume of the conv's kernel (kernel_height * kernel_width)

    if z_bcast_factor > 1:
        picker = torch.cat([picker[0][z] for z in range(z_bcast_factor)])
        sparse = dc.tensor(picker)
        result = dc.op("sparse_matmul", [sparse, in1], (accumulate, z_bcast_factor))
        result = dc.op("vslice", [result], (z_bcast_factor,))
        dc.fuse(result)


def backward(type, attr, ac, operand, inputs, output, grad):
    assert type == "sparse_matmul", "Only sparse_matmul backward supported in Python implementation"
    assert len(inputs) == 2, "Matrix multiply should have two inputs"
    assert operand < 2, "Invalid operand index"
    assert len(attr) <= 2, "Matrix multiply should have zero to two attributes"
    assert operand == 1, "Only support gradients through operand 1"

    in0 = inputs[0]
    in1 = inputs[1]

    in0_value = ac.get_pytorch_tensor(in0)
    assert in0_value.is_sparse
    in0t_value = transpose_sparse_picker_matrix(in0_value)
    in0t = ac.tensor(in0t_value)
    return ac.op("sparse_matmul", (in0t, grad))


def initial_flops_estimate(type, attr, ops):
    # Regular matmul flops estimation moved to C++ implementation
    # Only sparse_matmul estimation remains in Python
    return 0
