# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from ..common import to_torch_operands
import numpy as np
import torch
from loguru import logger
import forge
from forge.tensor import change_rank
from forge.forgeglobal import TILE_DIM
from forge.utils import align_up_tile, round_up_div, align_up
from .pad import Pad


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

    if type == "gather":
        assert len(attr) == 5, "Gather should have 5 attributes"
        dim, begin, length, stride, orig_size = attr
        x = t_ops[0]
        result = []
        zero_shape = list(x.shape)
        if dim > 0:
            dim -= 4
        while len(zero_shape) <= abs(dim):
            zero_shape = [1] + zero_shape
            x = x.unsqueeze(0)
        zero_shape[dim] = 1
        zero_slice = torch.zeros(zero_shape, dtype=dtype).squeeze(dim)
        offset = 0
        for i in range(0, orig_size):
            range_i = (i - begin) % stride
            if i >= begin and range_i < length:
                result.append(x.select(dim, offset))
                offset += 1
            else:
                result.append(zero_slice)
        return torch.stack(result, dim=dim)

    if type == "index":
        assert len(attr) == 4, "Index should have 4 attributes"
        dim, start, stop, stride = attr
        if dim >= 0:
            dim -= len(ops[0].shape)

        if dim == -5:
            return t_ops[0][..., start:stop:stride, :, :, :, :]
        elif dim == -4:
            return t_ops[0][..., start:stop:stride, :, :, :]
        elif dim == -3:
            return t_ops[0][..., start:stop:stride, :, :]
        elif dim == -2:
            return t_ops[0][..., start:stop:stride, :]
        elif dim == -1:
            return t_ops[0][..., start:stop:stride]
        else:
            raise NotImplementedError(f"Dim={dim}")

    if type == "conv2d_depthwise_weights":
        weights = t_ops[0]
        assert len(weights.shape) == 4, "Weights should have rank 4"

        w, z, cin, cout = weights.shape

        assert cin == 1, "Depthwise weights should always have cin == 1"

        # [1, 9, 1, 65] -> [1, 9, 1, 96]
        weights = torch.nn.functional.pad(weights, (0, align_up_tile(cout) - cout))
        # [1, 9, 1, 96] -> [1, 9, 32, 96]
        weights = torch.nn.functional.pad(weights, (0, 0, 0, align_up_tile(cin) - cin))

        # Diagonally embed weights
        weights_diag = torch.zeros_like(weights, requires_grad=False)

        cnt_kernels = z
        ct = weights.shape[-1] // TILE_DIM
        for idx_kernel in range(cnt_kernels):
            for idx_ct in range(ct):
                weights_diag[:, idx_kernel, :, idx_ct * TILE_DIM : (idx_ct + 1) * TILE_DIM] = torch.diag_embed(
                    weights[:, idx_kernel, 0, idx_ct * TILE_DIM : (idx_ct + 1) * TILE_DIM]
                )

        # [1, 9, 32, 96] -> [1, 1, 9 * 32, 96]
        weights_diag = weights_diag.reshape(w, 1, -1, weights.shape[-1])

        return weights_diag

    if type == "conv2d_depthwise_weights_bw":
        assert False, "not implemented yet"

    if type == "conv2d_grouped_weights":
        weights = t_ops[0]
        w = weights.shape[0]
        z = weights.shape[1]
        cin = weights.shape[2]
        cout = weights.shape[3]
        output_group = cout // attr[0]

        weights = torch.nn.functional.pad(weights, (0, align_up_tile(cout) - cout))
        weights = weights.reshape(w, z, -1, weights.shape[-1])

        weights_sections = torch.split(weights, output_group, dim=-1)
        new_weights = torch.zeros(w, z, align_up_tile(attr[0] * cin), align_up_tile(cout))
        for i, section in enumerate(weights_sections):
            new_weights[
                :,
                :,
                i * section.shape[-2] : (i + 1) * section.shape[-2],
                i * section.shape[-1] : (i + 1) * section.shape[-1],
            ] = section

        weights = new_weights.unsqueeze(-3)

        if len(attr) == 4:
            weights = weights.transpose(2, 3)
            weights = weights.reshape(w, z, TILE_DIM, -1)
        elif len(attr) == 5:
            weights = weights.transpose(1, 2)
            weights = weights.transpose(2, 3)
            weights = weights.reshape(w, 1, align_up_tile(attr[0] * cin), -1)
        return weights

    if type == "conv2d_grouped_weights_bw":
        weights = t_ops[0]
        groups = attr[0]
        w = 1
        z = attr[1]
        cin = attr[2]
        cout = attr[3]
        output_group = cout // groups

        if len(attr) == 4:
            assert weights.shape[0] == w
            assert weights.shape[1] == z
            assert weights.shape[2] == TILE_DIM
            weights = weights.transpose(2, 3)
            weights = weights.reshape(w, z, -1, TILE_DIM, TILE_DIM)
        elif len(attr) == 5:
            weights = weights.reshape(w, 1, align_up_tile(groups * cin), -1)
            weights = weights.transpose(2, 3)
            weights = weights.transpose(1, 2)
            weights = weights.reshape(w, z, align_up_tile(groups * cin), align_up_tile(cout))

        sections = []
        for i in range(groups):
            section = weights[:, :, i * cin : (i + 1) * cin, i * output_group : (i + 1) * output_group]
            sections.append(section)

        new_weights = torch.concat(sections, dim=-1)

        weights = new_weights.reshape(w, z, cin, -1)[:, :, :, :cout]
        return weights

    if type == "conv2d_prestride_act":
        assert len(attr) == 6, "conv2d_prestride_act should have 6 attributes"
        stride_height, stride_width, kernel_height, kernel_width, original_y, original_x = attr

        act = t_ops[0]

        act = torch.nn.functional.pad(
            act,
            (0, align_up(original_x, stride_width) - original_x, 0, align_up(original_y, stride_height) - original_y),
        )

        prestrided_activations = []
        for y in range(stride_height):
            for x in range(stride_width):
                prestrided_activations.append(act[:, :, y::stride_height, x::stride_width])

        prestrided_activations = torch.cat(prestrided_activations, dim=-3)

        w, z, r, c = prestrided_activations.shape
        prestrided_activations = prestrided_activations.view(w, 1, z, r * c)
        # prestrided_activations = prestrided_activations.transpose(-1, -2)

        return prestrided_activations

    if type == "pad_tile":
        assert len(attr) == 2
        dim, original_length = attr
        act = t_ops[0]
        if dim >= 0:
            dim -= len(act.shape)
        assert dim == -2 or dim == -1
        padding = align_up_tile(act.shape[dim]) - act.shape[dim]
        if dim == -2:
            act = torch.nn.functional.pad(act, (0, 0, 0, padding))
        if dim == -1:
            act = torch.nn.functional.pad(act, (0, padding))
        return act

    if type == "pixel_shuffle":
        assert len(ops) == 1, "Pixel shuffle should have one operand."
        assert len(attr) == 1, "Pixel shuffle should have one attribute."
        return torch.nn.functional.pixel_shuffle(ops[0], attr[0])

    if type == "forge_pad":
        assert (
            len(attr) == 3
        ), "Forge pad should have three attributes. The paddings for R and C dimensions and the value to pad with."
        r_tiles, c_tiles, value = attr
        operand = t_ops[0]
        shape = operand.shape
        # Padding is always given in tiles, so we need to recompute the padding in the original dimension
        new_r_size_tile, new_c_size_tile = 0, 0
        new_r_size, new_c_size = 0, 0
        if r_tiles > 0:
            new_r_size_tile = align_up_tile(shape[-2]) - shape[-2]
            new_r_size = r_tiles * TILE_DIM
        if c_tiles > 0:
            new_c_size_tile = align_up_tile(shape[-1]) - shape[-1]
            new_c_size = c_tiles * TILE_DIM
        result = torch.nn.functional.pad(operand, [0, new_c_size_tile, 0, new_r_size_tile], value=0)
        result = torch.nn.functional.pad(result, [0, new_c_size, 0, new_r_size], value=value)
        return result

    if type == "forge_unpad":
        assert len(attr) == 4, "Forge unpad should have four attributes. The paddings and the original shape."
        r_tiles, c_tiles, orig_r, orig_c = attr
        operand = t_ops[0]
        if r_tiles > 0:
            assert operand.shape[-2] == align_up_tile(orig_r) + r_tiles * TILE_DIM
        if c_tiles > 0:
            assert operand.shape[-1] == align_up_tile(orig_c) + c_tiles * TILE_DIM
        result = torch.index_select(operand, -2, torch.arange(orig_r))
        result = torch.index_select(result, -1, torch.arange(orig_c))
        return result

    assert False, f"{type} not defined in tensor manipulations"


def shape(type, attr, ops):
    assert len(ops) == 1, f"Tensor manipulation ops should have one input, has {len(ops)} input instead"

    if type == "gather":
        assert len(attr) == 5, "Gather should have 5 attributes"
        dim, begin, length, stride, orig_size = attr
        orig_shape = list(ops[0])
        if dim > 0:
            dim -= 4
        while len(orig_shape) <= abs(dim):
            orig_shape = [1] + orig_shape
        orig_shape[dim] = orig_size
        return tuple(orig_shape), []

    if type == "index":
        assert len(attr) == 4, "Index should have 4 attributes"
        dim, start, stop, stride = attr
        shape = list(ops[0])

        if start < 0:
            start = shape[dim] + start

        shape[dim] = round_up_div(stop - start, stride)
        return tuple(shape), []

    if type == "select":
        assert len(attr) == 4, "Select should have 4 attributes"
        dim, begin, length, stride = attr
        shape = list(ops[0])
        shape[dim] = length * round_up_div(shape[dim] - begin, stride)
        return tuple(shape), []

    if type == "conv2d_depthwise_weights":
        shape = list(ops[0])

        w, k, _, cout = shape
        shape = [w, 1, k * TILE_DIM, align_up_tile(cout)]

        return tuple(shape), []

    if type == "conv2d_depthwise_weights_bw":
        assert False, "not yet implemented"

    if type == "conv2d_grouped_weights":
        shape = list(ops[0])
        if len(attr) == 4:
            shape[2] = TILE_DIM
        elif len(attr) == 5:
            _, k, cin, cout = shape
            shape[1] = 1
            shape[2] = align_up_tile(attr[0] * cin)
            shape[3] = k * align_up_tile(cout)
        return tuple(shape), []

    if type == "conv2d_grouped_weights_bw":
        shape = list(ops[0])
        if len(attr) == 4:
            assert shape[2] == TILE_DIM
            shape[2] = 1
        elif len(attr) == 5:
            w, k, cin, cout, _ = attr
            shape[1] = k
            shape[2] = cin
            shape[3] = cout
        return tuple(shape), []

    if type == "conv2d_prestride_act":
        assert len(attr) == 6, "conv2d_prestride_act should have 6 attributes"
        stride_height, stride_width, kernel_height, kernel_width, original_y, original_x = attr

        shape = list(ops[0])
        assert len(shape) == 4

        shape[-2] = (shape[-2] + stride_height - 1) // stride_height
        shape[-1] = (shape[-1] + stride_width - 1) // stride_width

        shape[-3] *= stride_height * stride_width

        # reshape (no transpose in Prestride transform in BE tilize)
        reshape_shape = [
            shape[0],
            1,
            shape[1],
            shape[2] * shape[3],
        ]

        return tuple(reshape_shape), []

    if type == "pad_tile":
        assert len(attr) == 2
        dim, original_length = attr
        if dim >= 0:
            dim -= len(ops[0])
        if not (dim == -2 or dim == -1):
            x = 2
        assert dim == -2 or dim == -1
        shape = list(ops[0])
        shape[dim] = align_up_tile(shape[dim])
        return tuple(shape), []

    if type == "pixel_shuffle":
        assert len(ops) == 1, "Pixel shuffle should have one operand."
        assert len(attr) == 1, "Pixel shuffle should have one attribute."

        orig_shape = ops[0]
        assert len(orig_shape) >= 3, "Pixel shuffle should be at least 3D."

        upscale_factor = attr[0]
        assert (
            orig_shape[-3] % (upscale_factor**2) == 0
        ), f"Op shape at dim -3 ({orig_shape[-3]}) should be divisible by upscale_factor*upscale_factor ({upscale_factor**2})."

        output_shape = (
            *orig_shape[:-3],
            orig_shape[-3] // upscale_factor**2,
            orig_shape[-2] * upscale_factor,
            orig_shape[-1] * upscale_factor,
        )
        return output_shape, []

    if type == "forge_pad":
        assert (
            len(attr) == 3
        ), "Forge pad should have three attributes. The paddings for R and C dimensions and the value to pad with."
        r_tiles, c_tiles, value = attr
        shape = list(ops[0])
        # Padding is always given in tiles, so we need to recompute the padding in the original dimension
        if r_tiles > 0:
            shape[-2] = (align_up_tile(shape[-2]) // TILE_DIM + r_tiles) * TILE_DIM
        if c_tiles > 0:
            shape[-1] = (align_up_tile(shape[-1]) // TILE_DIM + c_tiles) * TILE_DIM
        return tuple(shape), []

    if type == "forge_unpad":
        assert len(attr) == 4, "Forge unpad should have four attributes. The paddings and the original shape."
        r_tiles, c_tiles, orig_r, orig_c = attr
        if r_tiles > 0:
            assert ops[0][-2] == align_up_tile(orig_r) + r_tiles * TILE_DIM
        if c_tiles > 0:
            assert ops[0][-1] == align_up_tile(orig_c) + c_tiles * TILE_DIM
        shape = list(ops[0])
        shape[-2] = orig_r
        shape[-1] = orig_c
        return tuple(shape), []

    assert False, f"{type} not defined in tensor manipulations"


def backward(type, attr, ac, operand, inputs, output, grad):

    assert operand == 0, "Invalid operand index"

    if type == "conv2d_depthwise_weights":
        return ac.op_with_named_attrs("conv2d_depthwise_weights_bw", (grad,), {}, attr)

    elif type == "conv2d_grouped_weights":
        return ac.op_with_named_attrs("conv2d_grouped_weights_bw", (grad,), {}, attr)

    elif type == "select":
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

    elif type == "pad_tile":
        assert len(attr) == 2
        dim, original_length = attr
        return ac.op_with_named_attrs("index", (grad,), {"dim": dim, "start": 0, "stop": original_length, "stride": 1})

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

        return Pad.decompose_constant_mode(ac, grad, value, left, right, 0, 0, dim, 0)

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

    if type == "pixel_shuffle":
        result = inputs[0]  # Shape: (N, C*r*r, H, W)
        N, C_r2, H, W = result.shape
        r = attr[0]
        C = C_r2 // (r * r)

        # Step 1: Reshape to (N, C, r, r, H, W)
        reshape_dims = (N, C, r, r, H, W)
        x = dc.op_with_named_attrs("reshape", [result], {"shape": reshape_dims})

        # Step 2: Transpose sequence on x
        x = dc.op_with_named_attrs("transpose", [x], {"dim0": 2, "dim1": 4})  # [0,1,4,3,2,5]
        x = dc.op_with_named_attrs("transpose", [x], {"dim0": 3, "dim1": 4})  # [0,1,4,2,3,5]
        x = dc.op_with_named_attrs("transpose", [x], {"dim0": 4, "dim1": 5})  # [0,1,4,2,5,3]

        # Step 3: Final reshape to (N, C, H * r, W * r)
        reshape_dims = (N, C, H * r, W * r)
        x = dc.op_with_named_attrs("reshape", [x], {"shape": reshape_dims})

        dc.fuse(x)


def decompose_post_optimize(type, attr, dc, inputs):
    pass
