# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from ..common import to_torch_operands
import torch
from loguru import logger
import forge
from forge.forgeglobal import TILE_DIM
from forge.utils import align_up_tile, round_up_div, align_up


def eval(type, attr, ops):
    assert len(ops) == 1, f"Tensor manipulation ops should have one input {len(ops)} {attr}"
    t_ops = to_torch_operands(*ops)
    dtype = ops[0].dtype

    if type == "select":
        assert len(attr) == 4, "Select should have 4 attributes"
        dim, begin, length, stride = attr
        zero_shape = list(t_ops[0].shape)
        zero_shape[dim] = 1
        zero_slice = torch.zeros(zero_shape, dtype=dtype).squeeze(dim)
        result = []
        for offset in range(0, t_ops[0].shape[dim] - begin, stride):
            for i in range(begin, begin + length):
                if offset + i < t_ops[0].shape[dim] or stride == t_ops[0].shape[dim]:
                    result.append(t_ops[0].select(dim, offset + i))
                else:
                    result.append(zero_slice)
        return torch.stack(result, dim=dim)

    assert False, f"{type} not defined in tensor manipulations"


def shape(type, attr, ops):
    assert len(ops) == 1, f"Tensor manipulation ops should have one input, has {len(ops)} input instead"

    if type == "select":
        assert len(attr) == 4, "Select should have 4 attributes"
        dim, begin, length, stride = attr
        shape = list(ops[0])
        shape[dim] = length * round_up_div(shape[dim] - begin, stride)
        return tuple(shape), []


def backward(type, attr, ac, operand, inputs, output, grad):

    assert operand == 0, "Invalid operand index"

    if type == "select":
        assert len(attr) == 4
        dim, begin, length, stride = attr
        orig_size = inputs[0].shape[dim]
        current_size = grad.shape[dim]
        # return ac.op("gather", (grad,), attributes=(dim, begin, length, stride, orig_size))

        # temporarily replace gather op (not HW supported) with select + concat
        grad_return = None
        grad_offset = 0

        for offset in range(0, orig_size, stride):
            # zero padding of front
            if begin > 0:
                zero_pre_pad_shape = inputs[0].shape.as_list()
                zero_pre_pad_shape[dim] = min(begin, orig_size - offset)
                if grad_return is None:
                    grad_return = ac.tensor(torch.zeros(zero_pre_pad_shape))
                else:
                    zero_slice = ac.tensor(torch.zeros(zero_pre_pad_shape))
                    grad_return = ac.op_with_named_attrs("concatenate", (grad_return, zero_slice), {"dim": dim})
            if offset + begin >= orig_size:
                break

            # pass the gradient for selected part
            grad_slice = ac.op(
                "select",
                (grad,),
                (dim, grad_offset, length, current_size),
                named_attrs={"dim": dim, "begin": grad_offset, "length": length, "stride": current_size},
            )
            if grad_return is None:
                grad_return = grad_slice
            else:
                grad_return = ac.op_with_named_attrs("concatenate", (grad_return, grad_slice), {"dim": dim})
            grad_offset += length
            if offset + begin + length >= orig_size:
                break

            # zero padding of back
            zero_padding_length = stride - length - begin
            if zero_padding_length > 0:
                zero_post_pad_shape = inputs[0].shape.as_list()
                zero_post_pad_shape[dim] = zero_padding_length
                zero_slice = ac.tensor(torch.zeros(zero_post_pad_shape))
                grad_return = ac.op_with_named_attrs("concatenate", (grad_return, zero_slice), {"dim": dim})
        return grad_return

    elif type == "index":
        assert len(attr) == 4
        dim, start, stop, stride = attr

        if stride != 1:
            raise NotImplementedError("Only stride == 1 is supported for index op backward")
        shape = inputs[0].shape.as_list()

        if dim >= 0:
            dim -= len(shape)

        left = start
        right = shape[dim] - stop
        value = 0.0

        # constant pad op expects padding in the following format: [dim0_low, dim0_high, dim1_low, dim1_high, ...]
        # which means we need to create padding for all dimensions zero except the one we are indexing into
        padding = [0] * (len(shape) * 2)
        padding[dim * 2] = left
        padding[dim * 2 + 1] = right

        return ac.op_with_named_attrs("constant_pad", (grad,), {"padding": padding, "value": value})

    raise NotImplementedError(f"{type}")


def unsqueeze_input_for_reshape_decomp(dc, inp):

    current_shape = inp.shape.as_list()
    while len(current_shape) < 4:
        current_shape.insert(0, 1)
        inp = dc.op_with_named_attrs("unsqueeze", (inp,), {"dim": 0})

    return inp


def squeeze_output_for_reshape_decomp(dc, output, orig_out_shape):
    current_shape_len = 4
    assert current_shape_len == output.shape.len()

    while current_shape_len > len(orig_out_shape):
        current_shape_len -= 1
        result = dc.op_with_named_attrs("squeeze", [output], {"dim": 0})

    return output


def decompose(type, attr, dc, inputs):
    pass


def decompose_post_optimize(type, attr, dc, inputs):
    pass
