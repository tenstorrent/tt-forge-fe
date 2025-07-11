# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from argparse import ArgumentError
from ..common import to_torch_operands
from ..sparse_utils import (
    create_index_sparse_picker_matrix,
    create_all_around_padding_picker_matrix,
    create_padding_shift_sparse_picker_matrix,
    create_real_row_sparse_picker_matrix,
    create_reshape_flatten_sparse_picker_matrix,
    create_flattened_padding_removal_sparse_picker_matrix,
    create_sparse_interleave_picker_matrix,
    create_reshape_flatten_sparse_picker_matrix_narrower,
    create_repeat_sparse_picker_matrix,
    calculate_conv2d_prestride_weights_and_padding,
    create_pad_replicate_sparse_picker,
    create_pad_reflect_sparse_picker,
)
import numpy as np
import torch
import math
from loguru import logger
import forge
from forge.tensor import change_rank
from forge.forgeglobal import TILE_DIM
from forge.utils import align_up_tile, round_up_div, align_up
from .pad import Pad
from .nop import Nop
from .buffer import Buffer


def eval(type, attr, ops):
    assert len(ops) == 1 or (
        type == "adv_index" and len(ops) == 2
    ), f"Tensor manipulation ops should have one input {len(ops)} {attr}"
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

    if type == "adv_index":
        assert len(attr) == 1, "AdvIndex should have 1 attributes"
        assert len(t_ops[1].shape) == 1 or len(t_ops[1].shape) == 2, "indices should be 1D or 2D"
        dim = attr[0]
        indices = t_ops[1].type(torch.LongTensor)
        if len(indices.shape) == 2:
            # Indices are 2D, we need to reshape them to 1D
            indices = indices.reshape(-1)

        ret = torch.index_select(t_ops[0], dim, indices)

        return ret

    if type == "broadcast":
        assert len(attr) <= 3, "Broadcast should have two attributes - dim and size"
        explicit_bcast = len(attr) == 3 and bool(attr[2])

        tensor = t_ops[0]
        dim = attr[0]
        size = attr[1]
        while len(tensor.shape) <= ((-dim - 1) if dim < 0 else dim):
            tensor = tensor.unsqueeze(0)
        target_shape = list(tensor.shape)
        assert dim < len(target_shape), f"Trying to broadcast on dim that doesn't exist: {dim} on {target_shape}"
        target_shape[dim] = size
        return torch.broadcast_to(tensor, target_shape)

    if type == "repeat":
        sizes = attr
        return t_ops[0].repeat(*sizes)

    if type == "repeat_interleave":
        assert len(attr) == 2, "repeat_interleave should have two attributes - repeats and dim"
        repeats = attr[0]
        dim = attr[1]
        return t_ops[0].repeat_interleave(repeats=repeats, dim=dim)

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

    if type == "conv2d_prestride_weights":
        assert len(attr) == 8, "conv2d_prestride_weights should have 8 attributes"
        y, x = attr[0], attr[1]
        stride_height, stride_width = attr[2], attr[3]
        padding = [attr[4], attr[5], attr[6], attr[7]]

        weights = t_ops[0]
        assert len(weights.shape) == 4, "weights should have 4 dims"

        ps_weights, _ = calculate_conv2d_prestride_weights_and_padding(weights, y, x, stride_width, padding)
        return ps_weights

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

    if type == "narrow":
        assert len(attr) == 4
        dim, start, length, original_length = attr
        act = t_ops[0]
        return act.narrow(dim, start, length)

    if type == "unsqueeze":
        assert len(attr) == 2
        dim = attr[0]
        input_ndim = attr[1]
        act = t_ops[0]
        return torch.unsqueeze(act, dim)

    if type == "squeeze":
        assert len(attr) == 1
        dim = attr[0]
        act = t_ops[0]
        return torch.squeeze(act, dim)

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
    assert len(ops) == 1 or (
        type == "adv_index" and len(ops) == 2
    ), f"Tensor manipulation ops should have one input, has {len(ops)} input instead"

    if type == "index":
        assert len(attr) == 4, "Index should have 4 attributes"
        dim, start, stop, stride = attr
        shape = list(ops[0])

        if start < 0:
            start = shape[dim] + start

        shape[dim] = round_up_div(stop - start, stride)
        return tuple(shape), []

    if type == "adv_index":
        assert len(attr) == 1, "AdvIndex should have 1 attributes"
        assert len(ops[1]) == 1 or len(ops[1]) == 2, "indices should be 1D or 2D"
        dim = attr[0]
        shape = list(ops[0])
        shape[dim] = ops[1][-1]
        return shape, []

    if type == "select":
        assert len(attr) == 4, "Select should have 4 attributes"
        dim, begin, length, stride = attr
        shape = list(ops[0])
        shape[dim] = length * round_up_div(shape[dim] - begin, stride)
        return tuple(shape), []

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

    if type == "hslice":
        assert len(attr) == 1, "HSlice should have one attribute, the slice size"
        slice_size = attr[0]
        shape = list(ops[0])
        assert shape[-1] % slice_size == 0
        while len(shape) < 4:
            shape = [1] + shape
        shape[-1] //= slice_size
        shape[-3] *= slice_size
        return tuple(shape), []

    if type == "hstack":
        assert len(ops[0]) >= 3, "HStack should at least have 3 dims"
        assert len(attr) == 1, "hstack should have one attribute, equal to number of stacks of Z dim to create"
        slice_size = attr[0]
        assert (
            ops[0][-3] % slice_size == 0
        ), f"HStack requires Z: {ops[0][-3]} to be a multiple of slice_size: {slice_size},"
        shape = list(ops[0])
        shape[-1] *= slice_size
        shape[-3] //= slice_size
        return tuple(shape), []

    if type == "vslice":
        assert len(attr) == 1, "VSlice should have one attribute, the slice size"
        slice_size = attr[0]
        shape = list(ops[0])
        assert len(shape) >= 2, "VSlice should at least have 2 dims"
        assert shape[-2] % slice_size == 0
        while len(shape) < 3:
            shape = [1] + shape
        shape[-2] //= slice_size
        shape[-3] *= slice_size
        return tuple(shape), []

    if type == "vstack":
        assert len(ops[0]) >= 3, "VStack should at least have 3 dims"
        assert len(attr) == 1, "vstack should have one attribute, equal to number of stacks of Z dim to create"
        slice_size = attr[0]
        assert ops[0][-3] % slice_size == 0, f"VStack requires Z to be a multiple of slice_size"
        shape = list(ops[0])
        shape[-2] *= slice_size
        shape[-3] //= slice_size
        return tuple(shape), []

    if type == "broadcast":
        assert len(attr) <= 3, "Broadcast should have two attributes - dim and size"
        dim = attr[0]
        size = attr[1]
        target_shape = list(ops[0])

        if dim < 0:
            while abs(dim) > len(target_shape):
                target_shape = [1] + target_shape
        else:
            while dim >= len(target_shape):
                target_shape = [1] + target_shape

        target_shape[dim] = size
        return tuple(target_shape), []

    if type == "repeat":
        sizes = attr
        if len(ops[0]) < len(sizes):
            # Scenario: When the input is a 1D tensor and needs to be repeated in 2D,
            # `ttir.repeat` does not currently support this directly,
            # so we are calculating the new shape by expanding the dimensions
            # to match repeat attr dimensions and calculate the output shape
            shape = (1,) * (len(sizes) - len(ops[0])) + tuple(ops[0])
        else:
            shape = ops[0]
        return tuple(dim * size for dim, size in zip(list(shape), sizes)), []

    if type == "repeat_interleave":
        assert len(attr) <= 3, "repeat_interleave should have two attributes - repeats and dim"
        repeats = attr[0]
        dim = attr[1]

        if dim < 0:
            dim += len(ops[0])
        target_shape = list(ops[0])
        target_shape[dim] *= repeats
        return tuple(target_shape), []

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

    if type == "conv2d_prestride_weights":
        assert len(attr) == 8, "conv2d_prestride_weights should have 8 attributes"
        y, x = attr[0], attr[1]
        stride_height, stride_width = attr[2], attr[3]
        padding = [attr[4], attr[5], attr[6], attr[7]]

        shape = list(ops[0])
        assert len(shape) == 4
        shape, _ = calculate_conv2d_prestride_weights_and_padding(shape, y, x, stride_width, padding)
        return tuple(shape), []

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

    if type == "narrow":
        assert len(attr) == 4
        dim, start, length, original_length = attr
        shape = list(ops[0])
        shape[dim] = length
        return tuple(shape), []

    if type == "unsqueeze":
        assert len(attr) == 2
        shape = list(ops[0])
        dim = attr[0]
        input_ndim = attr[1]
        # Handle negative dimension
        if dim < 0:
            # Adjust dim to be within the correct range
            dim += input_ndim + 1
        shape.insert(dim, 1)
        return tuple(shape), []

    if type == "squeeze":
        assert len(attr) == 1
        shape = list(ops[0])
        dim = attr[0]
        del shape[dim]
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
        return ac.op("conv2d_depthwise_weights_bw", (grad,), attributes=attr)

    elif type == "conv2d_grouped_weights":
        return ac.op("conv2d_grouped_weights_bw", (grad,), attributes=attr)

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
                    grad_return = ac.op("concatenate", (grad_return, zero_slice), (dim,))
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
                grad_return = ac.op("concatenate", (grad_return, grad_slice), (dim,))
            grad_offset += length
            if offset + begin + length >= orig_size:
                break

            # zero padding of back
            zero_padding_length = stride - length - begin
            if zero_padding_length > 0:
                zero_post_pad_shape = inputs[0].shape.as_list()
                zero_post_pad_shape[dim] = zero_padding_length
                zero_slice = ac.tensor(torch.zeros(zero_post_pad_shape))
                grad_return = ac.op("concatenate", (grad_return, zero_slice), (dim,))
        return grad_return

    elif type == "pad_tile":
        assert len(attr) == 2
        dim, original_length = attr
        return ac.op(
            "narrow",
            (grad,),
            attributes=(dim, 0, inputs[0].shape[dim], original_length),
        )

    elif type == "narrow":
        assert len(attr) == 4
        dim, start, length, original_length = attr
        if dim >= 0:
            dim -= len(inputs[0].shape)
        if dim in [-1, -2] and align_up_tile(length) == align_up_tile(inputs[0].shape[dim]):
            if dim == -1:
                return ac.op("pad", (grad,), (start, original_length - length - start, 0, False))
            elif dim == -2:
                return ac.op("pad", (grad,), (0, 0, start, original_length - length - start, 0, False))
            raise ArgumentError("Only dim == 2 and dim == 3 are supported.")
        else:
            raise NotImplementedError("Unimplemented narrow in forge")

    elif type == "unsqueeze":
        assert len(attr) == 2
        if len(inputs[0].shape) == len(grad.shape):
            # Dimensionality already matches, no need to squeeze
            return grad

        dim = attr[0]
        input_ndim = attr[1]
        return ac.op("squeeze", (grad,), (dim,), {"dim": dim})

    elif type == "squeeze":
        assert len(attr) == 1
        if len(inputs[0].shape) == len(grad.shape):
            # Dimensionality already matches, no need to unsqueeze
            return grad

        dim = attr[0]
        if grad.shape.len() == 4:  # Cannot unsqueeze beyond 4D
            return ac.op(Nop.create(), (grad,))
        return ac.op("unsqueeze", (grad,), attributes=(dim, grad.shape.len()), named_attrs={"dim": dim})

    elif type == "broadcast":
        assert len(attr) == 3
        if attr[0] < 0:
            attr[0] += inputs[0].shape.len()
        delta = 4 - inputs[0].shape.len()
        attr[0] += delta
        assert attr[0] >= 0 and attr[0] <= 3, f"Invalid broadcast dim after lowering: {attr[0]}"

        if attr[0] == 2 or attr[0] == 3:
            ret = ac.op("reduce_sum", (grad,), (attr[0],), {"keep_dim": True})
        else:
            ret = ac.op_with_named_attrs(
                "transpose",
                [
                    grad,
                ],
                {"dim0": attr[0], "dim1": -2},
            )
            ret = ac.op("reduce_sum", (ret,), (-2,), {"keep_dim": True})
            ret = ac.op_with_named_attrs(
                "transpose",
                [
                    ret,
                ],
                {"dim0": attr[0], "dim1": -2},
            )
        return ret

    elif type == "repeat_interleave":
        assert len(attr) == 2, "repeat_interleave should have two attributes - repeats and dim"
        repeats = attr[0]
        dim = attr[1]
        shape = inputs[0].shape.as_list()
        if dim < 0:
            dim += len(shape)

        shape.insert(dim, repeats)

        ret = ac.op_with_named_attrs("reshape", (grad,), {"shape": shape})
        ret = ac.op("reduce_sum", (ret,), (dim, True), {"dim_arg": [dim], "keep_dim": True})
        ret = ac.op("squeeze", (ret,), (dim,), {"dim": dim})
        return ret

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
        inp = dc.op_with_named_attrs("unsqueeze", (inp,), {"dim": 0}, (0, len(inp.shape.as_list())))

    return inp


def squeeze_output_for_reshape_decomp(dc, output, orig_out_shape):
    current_shape_len = 4
    assert current_shape_len == output.shape.len()

    while current_shape_len > len(orig_out_shape):
        current_shape_len -= 1
        result = dc.op_with_named_attrs("squeeze", [output], {"dim": 0}, (0,))

    return output


def decompose(type, attr, dc, inputs):
    if type == "adv_index":
        dim = attr[0]
        in0_shape = inputs[0].shape.as_list()
        indices_shape = inputs[1].shape.as_list()

        assert len(indices_shape) == 2 or len(indices_shape) == 1, "indices tensor should be 1D or 2D"

        # Idea is to reshape the input tensor to [in0_shape[dim], -1] and then apply the embedding operation
        # The embedding operation will select the appropriate indices from the reshaped tensor
        # and then we will reshape the output back to the original shape.
        #
        # For example:
        # If the input tensor is of shape [N, C, H, W] and we want to index along dim = 2 with indices shape [X],
        # we will first reshape it: [N, C, H, W] -> [N, H, C, W] and [N, H, C, W] -> [H, N, C, W] (permuted)
        # and then reshape it to [H, N * C * W] (flattening the last 3 dimensions)
        # and then apply the embedding operation to select the appropriate indices [H, N * C * W] -> [X, N * C * W].
        # Next, we will reshape the output back to the 4D shape [X, N * C * W] -> [X, N, C, W]
        # and finally, we will transpose the output back to the original order.
        # [X, N, C, W] -> [N, X, C, W] and [N, X, C, W] -> [N, C, X, W]

        # Step 1: Move the indexed dimension to the front using a sequence of transposes
        if dim != 0:
            current = inputs[0]
            for i in range(dim, 0, -1):
                current = dc.op_with_named_attrs("transpose", [current], {"dim0": i, "dim1": i - 1})
            permuted = current
        else:
            # No need to transpose if dim is already 0
            permuted = inputs[0]

        # Step 2: Reshape to [in0_shape[dim], -1]
        # Calculate product of all dimensions except the first (after transposition)

        if len(in0_shape) != 2:
            # Calculate permuted shape, by popping the element at indexed dim and inserting it at the begging
            permuted_shape = in0_shape.copy()  # copy is needed to avoid modifying the original shape
            indexed_dim_shape = permuted_shape.pop(dim)
            permuted_shape = [indexed_dim_shape, *permuted_shape]

            rest_dims_product = math.prod(permuted_shape[1:])

            reshape_dims = [in0_shape[dim], rest_dims_product]
            reshaped = dc.op_with_named_attrs("reshape", [permuted], {"shape": reshape_dims})
        else:
            reshaped = permuted

        # Step 3: Apply embedding operation
        # embedding op expects indices tensor as first argument and embedding_table as second argument
        selected = dc.op("embedding", (inputs[1], reshaped))

        # Step 4: Reshape back to appropriate dimensions
        # The new shape replaces the indexed dimension with the indices shape
        if len(in0_shape) != 2:
            output_shape = indices_shape + permuted_shape[1:]

            reshaped_output = dc.op_with_named_attrs("reshape", [selected], {"shape": output_shape})
        else:
            reshaped_output = selected

        # Step 5: Restore original dimension order if necessary using transposes
        if dim != 0:
            # Move dimension 0 to position 'dim' using transposes
            current = reshaped_output
            for i in range(0, dim):
                current = dc.op_with_named_attrs("transpose", [current], {"dim0": i, "dim1": i + 1})
            result = current
        else:
            # No need to transpose if dim is already 0
            result = reshaped_output

        dc.fuse(result)
        return

    if type == "broadcast":
        if attr[1] == 1:
            dc.fuse(dc.op(Nop.create(), [inputs[0]]))

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

    if type == "repeat":
        input_shape = inputs[0].shape.as_list()
        target_shape = attr
        result = inputs[0]

        if len(input_shape) < len(target_shape):
            # Scenario: When the input is a 1D tensor and needs to be repeated in 2D,
            # `ttir.repeat` does not currently support this directly.
            # To handle this, we first reshape the input to ensure both the input and the repeats have the same dimensions
            new_shape = (1,) * (len(target_shape) - len(input_shape)) + tuple(input_shape)
            result = dc.op_with_named_attrs("reshape", [result], {"shape": new_shape})
            result = dc.op_with_named_attrs("repeat", [result], {"repeats": target_shape}, target_shape)
            dc.fuse(result)


def create_row_picker_matrix(col_indices, lhs_num_cols, lhs_num_channels=None, lhs_batch_size=None):
    """
    Create a sparse matrix that picks rows from a matrix.
    col_indices: indices of columns from the matrix to pick. Create picker matrix based on this.
    lhs_num_cols: number of columns in the picker matrix (which is on the left hand side of the matmul)
    lhs_num_channels: number of channels in the picker matrix
    lhs_batch_size: batch size of the picker matrix


    Example:
    col_indices = [1, 3, 5]
    lhs_num_cols = 6
    lhs_num_channels = 1
    lhs_batch_size = 1
    Picker matrix:
    [
    [
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1]
    ],
    ]
    """
    # assert that col_indices is not empty
    assert len(col_indices) > 0

    new_num_rows = len(col_indices)
    # this picker matrix has to have the same number of dimensions as the input matrix.
    # depending on the input matrix, we need to create a 2D, 3D and 4D picker matrix
    if lhs_batch_size is not None:
        # if lhs_batch_size is not none - we have a 4D input matrix
        B_left = torch.zeros((lhs_batch_size, lhs_num_channels, new_num_rows, lhs_num_cols))
        for i, index in enumerate(col_indices):
            B_left[:, :, i, index] = 1.0
    elif lhs_num_channels is not None:
        # if lhs_batch_size is none but lhs_num_channels is not none - we have a 3D input matrix
        B_left = torch.zeros((lhs_num_channels, new_num_rows, lhs_num_cols))
        for i, index in enumerate(col_indices):
            B_left[:, i, index] = 1.0
    else:
        # if lhs_batch_size and lhs_num_channels are none - we have a 2D input matrix
        B_left = torch.zeros((new_num_rows, lhs_num_cols))
        for i, index in enumerate(col_indices):
            B_left[i, index] = 1.0

    return B_left


def picker_matmul(use_sparse_mm, dc, s, result):
    if use_sparse_mm:
        lhs = dc.tensor(s)
        result = dc.op("sparse_matmul", [lhs, result])
    else:
        lhs = dc.tensor(s.to_dense())
        result = dc.op("matmul", [lhs, result])

    return result


def pad_to_tile_dim(n):
    if n % TILE_DIM == 0:
        return n
    return n + TILE_DIM - (n % TILE_DIM)


def decompose_select(attr, dc, inputs):
    orig_shape = inputs[0].shape
    dim, index, length, stride = attr
    if dim >= 0:
        dim -= len(orig_shape)

    result = inputs[0]
    if orig_shape[dim] == length:
        result = dc.op(Nop.create(), [result])
        dc.fuse(result)

    # select on z dim is supported via splice
    elif dim == -3:
        return

    # At least one of index, length, or stride is not tile dim aligned, and we are operating on either the x or y dim
    # For example selecting rows 30-35 from tensor of shape (1, 1, 64, 128)
    elif (
        not (index % TILE_DIM == length % TILE_DIM == stride % TILE_DIM == 0)
        and dim in [-2, -1]
        and stride == orig_shape[dim]
    ):
        assert len(attr) == 4, "Select should have 4 attributes"
        x = result
        x = dc.op("pad_tile", [x], (-2, orig_shape[-2]))
        x = dc.op("pad_tile", [x], (-1, orig_shape[-1]))

        cols = []
        size = len(range(index, orig_shape[dim], stride)) * len(range(index, index + length))
        for offset in range(0, orig_shape[dim], stride):
            for i in range(index, index + length):
                if offset + i < orig_shape[dim] or stride == orig_shape[dim]:
                    cols.append(offset + i)

        rows = list(range(len(cols)))
        vals = [1.0] * len(cols)

        spm = torch.sparse_coo_tensor((rows, cols), vals, (align_up_tile(size), x.shape[dim]))
        if len(result.shape) > 2 and result.shape[-3] > 1:
            spm = torch.stack([spm] * result.shape[-3], -3).unsqueeze(0)
        spm = dc.tensor(spm)

        is_x_select = dim == -1
        if is_x_select:
            x = dc.op_with_named_attrs("transpose", [x], {"dim0": -2, "dim1": -1})

        result = dc.op("sparse_matmul", [spm, x])

        if is_x_select:
            result = dc.op_with_named_attrs("transpose", [result], {"dim0": -2, "dim1": -1})

        if is_x_select:
            result = dc.op("narrow", [result], (-1, 0, size, result.shape[-1]))
            result = dc.op("narrow", [result], (-2, 0, orig_shape[-2], result.shape[-2]))
        else:

            result = dc.op("narrow", [result], (-1, 0, orig_shape[-1], result.shape[-1]))
            result = dc.op("narrow", [result], (-2, 0, size, result.shape[-2]))

        dc.fuse(result)

        return


def decompose_xy_unflatten(inputs, dc, orig_shape, attr):
    result = inputs[0]
    use_sparse_mm = True
    # Pick out Z dim values from Y and expand the X dimension result matrix by TILE_DIM times
    # Original matrix:
    #               |   0   |   1   |  ...  | LY=len(Y) - 1
    #               -------------------------------------
    #       0       | A0,0  | A0,1  |  ...  |   A0,LY   |
    #       1       | A1,0  | A1,1  |  ...  |   A1,LY   |
    #      ...      |  ...  |  ...  |  ...  |    ...    |
    #  LX=len(X)-1  | ALX,0 | ALX,1 |  ...  |   ALX,LY  |
    #
    # Picker matrix is in the following format:
    #                       |   0   |   1   |   2   |  ...  |   LX  |
    #                       ----------------------------------------|
    #        0              |   1   |   0   |   0   |  ...  |   0   |
    #        1              |   0   |   0   |   0   |   0   |   0   |
    #       ...             |   0   |   0   |   0   |   0   |   0   |
    #     TILE_DIM          |   0   |   1   |   0   |   0   |   0   |
    #       ...             |   0   |   0   |   0   |   0   |   0   |
    #    2*TILE_DIM         |   0   |   0   |   1   |   0   |   0   |
    #       ...             |   0   |   0   |   0   |   0   |   0   |
    #    LX*TILE_DIM        |   0   |   0   |   0   |   0   |   1   |
    # LX*TILE_DIM+TILE_DIM-1|   0   |   0   |   0   |   0   |   0   |

    s_pick_z = create_reshape_flatten_sparse_picker_matrix(orig_shape[-2], orig_shape[-2] * TILE_DIM)
    result = picker_matmul(use_sparse_mm, dc, s_pick_z, result)

    # Result matrix is in the following format:
    #                       |   0   |   1   |  ...  |   LY  |
    #                       ---------------------------------
    #           0           | A0,0  | A0,1  |  ...  | A0,LY |
    #           ...         |   0   |   0   |   0   |   0   |
    #       TILE_DIM        | A1,0  | A1,1  |  ...  | A1,LY |
    #           ...         |   0   |   0   |   0   |   0   |
    #       2*TILE_DIM      | A2,0  | A2,1  |  ...  | A2,LY |
    #           ...         |   0   |   0   |   0   |   0   |
    #       LX*TILE_DIM     | ALX,0 | ALX,1 |   0   | ALX,LY|
    #           ...         |   0   |   0   |   0   |   0   |
    # LX*TILE_DIM+TILE_DIM-1|   0   |   0   |   0   |   0   |

    _orig_shape = result.shape
    # Pad X dimension to TILE_DIM size
    if _orig_shape[-2] % TILE_DIM != 0:
        result = dc.op("pad_tile", [result], (-2, _orig_shape[-2]))

    # Pad Y dimension to TILE_DIM size
    if _orig_shape[-1] % TILE_DIM != 0:
        result = dc.op("pad_tile", [result], (-1, _orig_shape[-1]))

    # Transpose the result matrix
    result = dc.op(TransposeTM.create(-2, -1), [result])
    slice_factor = _orig_shape[-1] // attr[-1]

    # After matrix transpose, the result is in the following format:
    #       |   0   |  ...  |  TILE_DIM |  ...  |   2*TILE_DIM  |  ...  |  LX*TILE_DIM  |  ...  |LX*TILE_DIM+TILE_DIM-1 |
    #       --------------------------------------------------------------------------------------------------------------
    #   0   | A0,0  |   0   |   A1,0    |  ...  |   A2,0        |  ...  |   ALX,0       |  ...  |           0           |
    #   1   | A0,1  |   0   |   A1,1    |  ...  |   A2,1        |  ...  |   ALX,1       |  ...  |           0           |
    #  ...  |  ...  |   0   |   ...     |  ...  |   ...         |  ...  |   ...         |  ...  |           0           |
    #  LY   | A0,LY |   0   |   A1,LY   |  ...  |   A2,LY       |  ...  |   ALX,LY      |  ...  |           0           |

    # If new X\Y dimensions aren't divisible by TILE_DIM, we need to padd the resulting matrix
    if attr[-1] % TILE_DIM != 0 or attr[-2] % TILE_DIM != 0:
        padded_dim = math.ceil(attr[-1] / TILE_DIM) * TILE_DIM
        num_tiles = attr[-2] if attr[-1] < TILE_DIM else (math.ceil(attr[-2] / TILE_DIM) * TILE_DIM)
        new_size = num_tiles * padded_dim

        cols = torch.arange(orig_shape[-1]).tolist()
        rows = []
        for i in range(attr[-2]):
            rows.extend((torch.arange(attr[-1]) + (i * padded_dim)).tolist())

        # For example, picker matrix is in the following format, where LNY represents new Y dimension:
        #                   |   0   |   1   |  ...  |   LNY | LNY+1 | LNY+2 | ... |
        #                   -------------------------------------------------------
        #      0            |   1   |   0   |   0   |  ...  |   0   |   0   | ... |
        #      1            |   0   |   1   |   0   |  ...  |   0   |   0   | ... |
        #     ...           |   0   |   0   |   1   |  ...  |   0   |   0   | ... |
        #     LNY           |   0   |   0   |   0   |   1   |   0   |   0   | ... |
        #     ...           |   0   |   0   |   0   |   0   |   0   |   0   | ... |
        #   padded_dim      |   0   |   0   |   0   |   0   |   1   |   0   | ... |
        #  padded_dim+1     |   0   |   0   |   0   |   0   |   0   |   1   | ... |
        #     ...           |   0   |   0   |   0   |   0   |   0   |   0   | ... |
        s_pad_with_zero = torch.sparse_coo_tensor(
            [rows, cols],
            torch.ones(len(cols)),
            (new_size, result.shape[-2]),
            dtype=torch.float32,
        )
        result = picker_matmul(use_sparse_mm, dc, s_pad_with_zero, result)

    # Slice out Z dim
    result = dc.op(TransposeTM.create(-2, -1), [result])
    if orig_shape[-2] > 1:
        result = dc.op("vslice", [result], (orig_shape[-2],))
    elif len(result.shape) == 2:
        result = dc.op_with_named_attrs(
            "unsqueeze",
            [result],
            {"dim": 0},
            (
                0,
                2,
            ),
        )
    _orig_shape = result.shape
    slice_factor = attr[-2] if attr[-1] < TILE_DIM else (math.ceil(attr[-2] / TILE_DIM) * TILE_DIM)
    result = dc.op(TransposeTM.create(-2, -1), [result])

    # Slice out row size
    result = dc.op("vslice", [result], (slice_factor,))
    result = dc.op(TransposeTM.create(-2, -1), [result])
    result = dc.op("vstack", [result], (slice_factor * _orig_shape[-3],))

    # Pick out mulitple rows and pack them into tiles
    s = create_reshape_flatten_sparse_picker_matrix(slice_factor * attr[-3], result.shape[-2]).transpose(-1, -2)
    result = picker_matmul(use_sparse_mm, dc, s, result)

    if (_orig_shape[-3] > 1) and (attr[-3] > 1):
        result = dc.op("vslice", [result], (attr[-3],))

    if attr[-1] % TILE_DIM != 0:
        result = dc.op("narrow", [result], (-1, 0, attr[-1], result.shape[-1]))
    if attr[-2] % TILE_DIM != 0:
        result = dc.op("narrow", [result], (-2, 0, attr[-2], result.shape[-2]))
    return result


def decompose_non_tile_dim_aligned_vslice(inputs, dc, orig_shape, attr):
    result = unsqueeze_input_for_reshape_decomp(dc, inputs[0])
    use_sparse_mm = True

    slice_factor = attr[-3]
    result = dc.op("pad_tile", [result], (-2, orig_shape[-2]))
    result = dc.op("pad_tile", [result], (-1, orig_shape[-1]))
    if attr[-2] % TILE_DIM != 0 or orig_shape[-2] % TILE_DIM != 0:
        padded_dim = math.ceil(attr[-2] / TILE_DIM) * TILE_DIM
        num_tiles = attr[-3] if attr[-2] < TILE_DIM else (math.ceil(attr[-3] / TILE_DIM) * TILE_DIM)
        new_size = num_tiles * padded_dim

        cols = torch.arange(orig_shape[-2]).tolist()
        rows = []
        for i in range(attr[-3]):
            rows.extend((torch.arange(attr[-2]) + (i * padded_dim)).tolist())

        spm = torch.sparse_coo_tensor(
            [rows, cols],
            torch.ones(len(cols)),
            (new_size, result.shape[-2]),
            dtype=torch.float32,
        )
        if attr[-2] >= TILE_DIM:
            spm1 = create_flattened_padding_removal_sparse_picker_matrix(
                spm.shape[-2], 0, slice_factor * padded_dim, spm.shape[-2]
            )
            spm = torch.sparse.mm(spm1, spm)

        result = picker_matmul(use_sparse_mm, dc, spm, result)

    result = dc.op("vslice", [result], (slice_factor,))
    if attr[-1] % TILE_DIM != 0:
        result = dc.op("narrow", [result], (-1, 0, attr[-1], result.shape[-1]))
    if attr[-2] % TILE_DIM != 0:
        result = dc.op("narrow", [result], (-2, 0, attr[-2], result.shape[-2]))

    return result


def decompose_post_optimize(type, attr, dc, inputs):
    # TODO: remove once backend support is available
    if type == "select":
        decompose_select(attr, dc, inputs)

    elif type == "hslice":
        input_shape = inputs[0].shape.as_list()
        post_dim = input_shape[-1] // attr[0]
        result = inputs[0]
        if post_dim % TILE_DIM != 0:
            if input_shape[-2] % TILE_DIM != 0:
                result = dc.op(
                    "pad_tile",
                    [
                        result,
                    ],
                    (-2, input_shape[-2]),
                )
            cols = []
            pad_post_dim = align_up_tile(post_dim)
            pad_input_dim = pad_post_dim * attr[0]
            for i in range(attr[0]):
                cols.extend(torch.arange(i * pad_post_dim, i * pad_post_dim + post_dim).tolist())
            spm = torch.sparse_coo_tensor(
                [cols, torch.arange(input_shape[-1]).tolist()],
                torch.ones(input_shape[-1]),
                (pad_input_dim, input_shape[-1]),
                dtype=torch.float32,
            )

            while len(result.shape) < 3:
                result = dc.op_with_named_attrs(
                    "unsqueeze",
                    [
                        result,
                    ],
                    {"dim": 0},
                    (0, len(result.shape.as_list())),
                )

            spm = torch.stack([spm] * result.shape[-3], -3).unsqueeze(0)
            result = dc.op_with_named_attrs(
                "transpose",
                [
                    result,
                ],
                {"dim0": -2, "dim1": -1},
            )
            result = picker_matmul(True, dc, spm, result)
            result = dc.op_with_named_attrs("transpose", [result], {"dim0": -2, "dim1": -1})
            result = dc.op(
                "hslice",
                [
                    result,
                ],
                attr,
            )
            result = dc.op(
                "narrow",
                [
                    result,
                ],
                (-1, 0, post_dim, result.shape[-1]),
            )
            if input_shape[-2] % TILE_DIM != 0:
                result = dc.op(
                    "narrow",
                    [
                        result,
                    ],
                    (-2, 0, input_shape[-2], result.shape[-2]),
                )
            dc.fuse(result)
        elif input_shape[-2] % TILE_DIM != 0:
            result = dc.op(
                "pad_tile",
                [
                    result,
                ],
                (-2, input_shape[-2]),
            )
            result = dc.op(
                "hslice",
                [
                    result,
                ],
                attr,
            )
            result = dc.op(
                "narrow",
                [
                    result,
                ],
                (-2, 0, input_shape[-2], result.shape[-2]),
            )
            dc.fuse(result)

    elif type == "hstack":
        input_shape = inputs[0].shape.as_list()
        result = inputs[0]
        if input_shape[-1] % TILE_DIM != 0:
            if input_shape[-2] % TILE_DIM != 0:
                result = dc.op(
                    "pad_tile",
                    [
                        result,
                    ],
                    (-2, input_shape[-2]),
                )
            output_dim = input_shape[-1] * attr[0]
            pad_output_dim = align_up_tile(input_shape[-1]) * attr[0]
            result = dc.op(
                "pad_tile",
                [
                    result,
                ],
                (-1, input_shape[-1]),
            )
            result = dc.op(
                "hstack",
                [
                    result,
                ],
                attr,
            )
            rows = []
            pad_input_dim = align_up_tile(input_shape[-1])
            for i in range(attr[0]):
                rows.extend(torch.arange(i * pad_input_dim, i * pad_input_dim + input_shape[-1]).tolist())
            spm = torch.sparse_coo_tensor(
                [torch.arange(output_dim).tolist(), rows],
                torch.ones(output_dim),
                (output_dim, pad_output_dim),
                dtype=torch.float32,
            )
            spm = torch.stack([spm] * result.shape[-3], -3).unsqueeze(0)
            result = dc.op_with_named_attrs("transpose", [result], {"dim0": -2, "dim1": -1})
            result = picker_matmul(True, dc, spm, result)
            result = dc.op_with_named_attrs("transpose", [result], {"dim0": -2, "dim1": -1})
            if input_shape[-2] % TILE_DIM != 0:
                result = dc.op(
                    "narrow",
                    [
                        result,
                    ],
                    (-2, 0, input_shape[-2], result.shape[-2]),
                )
            dc.fuse(result)
        elif input_shape[-2] % TILE_DIM != 0:
            result = dc.op(
                "pad_tile",
                [
                    result,
                ],
                (-2, input_shape[-2]),
            )
            result = dc.op(
                "hstack",
                [
                    result,
                ],
                attr,
            )
            result = dc.op(
                "narrow",
                [
                    result,
                ],
                (-2, 0, input_shape[-2], result.shape[-2]),
            )
            dc.fuse(result)

    elif type == "vslice":
        input_shape = inputs[0].shape.as_list()
        post_dim = input_shape[-2] // attr[0]
        result = inputs[0]
        if post_dim % TILE_DIM != 0:
            if input_shape[-1] % TILE_DIM != 0:
                result = dc.op(
                    "pad_tile",
                    [
                        result,
                    ],
                    (-1, input_shape[-1]),
                )
            cols = []
            pad_post_dim = align_up_tile(post_dim)
            pad_input_dim = pad_post_dim * attr[0]
            for i in range(attr[0]):
                cols.extend(torch.arange(i * pad_post_dim, i * pad_post_dim + post_dim).tolist())
            spm = (
                torch.sparse_coo_tensor(
                    [cols, torch.arange(input_shape[-2]).tolist()],
                    torch.ones(input_shape[-2]),
                    (pad_input_dim, input_shape[-2]),
                    dtype=torch.float32,
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )
            spm = torch.cat([spm] * result.shape[-3], -3)
            result = picker_matmul(True, dc, spm, result)
            result = dc.op(
                "vslice",
                [
                    result,
                ],
                attr,
            )
            result = dc.op(
                "narrow",
                [
                    result,
                ],
                (-2, 0, post_dim, result.shape[-2]),
            )
            if input_shape[-1] % TILE_DIM != 0:
                result = dc.op(
                    "narrow",
                    [
                        result,
                    ],
                    (-1, 0, input_shape[-1], result.shape[-1]),
                )
            dc.fuse(result)
        elif input_shape[-1] % TILE_DIM != 0:
            result = dc.op(
                "pad_tile",
                [
                    result,
                ],
                (-1, input_shape[-1]),
            )
            result = dc.op(
                "vslice",
                [
                    result,
                ],
                attr,
            )
            result = dc.op(
                "narrow",
                [
                    result,
                ],
                (-1, 0, input_shape[-1], result.shape[-1]),
            )
            dc.fuse(result)

    elif type == "vstack":
        input_shape = inputs[0].shape.as_list()
        result = inputs[0]
        if input_shape[-2] % TILE_DIM != 0:
            if input_shape[-1] % TILE_DIM != 0:
                result = dc.op(
                    "pad_tile",
                    [
                        result,
                    ],
                    (-1, input_shape[-1]),
                )
            output_dim = input_shape[-2] * attr[0]
            pad_output_dim = align_up_tile(input_shape[-2]) * attr[0]
            result = dc.op(
                "pad_tile",
                [
                    result,
                ],
                (-2, input_shape[-2]),
            )
            result = dc.op(
                "vstack",
                [
                    result,
                ],
                attr,
            )
            rows = []
            pad_input_dim = align_up_tile(input_shape[-2])
            for i in range(attr[0]):
                rows.extend(torch.arange(i * pad_input_dim, i * pad_input_dim + input_shape[-2]).tolist())
            spm = (
                torch.sparse_coo_tensor(
                    [torch.arange(output_dim).tolist(), rows],
                    torch.ones(output_dim),
                    (output_dim, pad_output_dim),
                    dtype=torch.float32,
                )
                .coalesce()
                .unsqueeze(0)
                .unsqueeze(0)
            )
            spm = torch.cat([spm] * result.shape[-3], -3)
            result = picker_matmul(True, dc, spm, result)
            if input_shape[-1] % TILE_DIM != 0:
                result = dc.op(
                    "narrow",
                    [
                        result,
                    ],
                    (-1, 0, input_shape[-1], result.shape[-1]),
                )
            dc.fuse(result)
        elif input_shape[-1] % TILE_DIM != 0:
            result = dc.op(
                "pad_tile",
                [
                    result,
                ],
                (-1, input_shape[-1]),
            )
            result = dc.op(
                "vstack",
                [
                    result,
                ],
                attr,
            )
            result = dc.op(
                "narrow",
                [
                    result,
                ],
                (-1, 0, input_shape[-1], result.shape[-1]),
            )
            dc.fuse(result)

    return
