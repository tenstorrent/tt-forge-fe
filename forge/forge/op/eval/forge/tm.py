# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from argparse import ArgumentError
from json.encoder import py_encode_basestring
from ssl import OP_NO_RENEGOTIATION
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
import ast
import os
from loguru import logger
import forge
from forge.tensor import change_rank
from forge.forgeglobal import TILE_DIM
from forge.utils import align_up_tile, round_up_div, align_up
from .transpose import TransposeTM
from ..lforge.splice import Splice
from .nop import Nop
from ..lforge.nop import Nop as ForgeNop
from .buffer import Buffer


def eval(type, attr, ops):
    assert len(ops) == 1 or (
        type == "adv_index" and len(ops) == 2
    ), f"Tensor manipulation ops should have one input {len(ops)} {attr}"
    t_ops = to_torch_operands(*ops)
    dtype = ops[0].dtype

    if type == "transpose":
        assert len(attr) == 3, "Transpose should have 3 attributes"
        dim0, dim1, orig_size = attr
        return torch.transpose(t_ops[0], dim0, dim1)

    if type == "reshape":
        return t_ops[0].reshape(attr)

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
        dim = attr[0]
        assert dim == 0, "Currently not supported"

        if len(t_ops[1].shape) > 1:
            if len(t_ops[0].shape) > len(t_ops[1].shape) and t_ops[0].shape[0] == 1:
                # Padded
                ret = torch.unsqueeze(t_ops[0][0][t_ops[1].numpy()], 0)
            else:
                ret = torch.unsqueeze(t_ops[0][t_ops[1].numpy()], 0)
        else:
            ret = t_ops[0][t_ops[1].numpy()]
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

    if type == "pad":
        # Expect (padding_left, padding_right, mode, channel_last) or (padding_left, padding_right, padding_top, padding_bottom, mode, channel_last)
        assert len(attr) == 4 or len(attr) == 6

        mode_idx = attr[-2]
        channel_last = attr[-1]
        attr = attr[:-2]
        if channel_last:
            attr = [0, 0] + attr
        mode_options = ["constant", "replicate", "reflect"]
        return torch.nn.functional.pad(t_ops[0], tuple(attr), mode=mode_options[mode_idx])

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

    if type == "transpose":
        # Transpose has 3 attrs, [axis_0, axis_1, output Z-dim size]
        assert len(attr) == 3, f"{len(attr)}"
        dim0 = attr[0]
        dim1 = attr[1]
        shape = list(ops[0])
        a = shape[dim0]
        b = shape[dim1]
        shape[dim0] = b
        shape[dim1] = a
        return tuple(shape), []

    if type == "reshape":
        return attr, []

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
        dim = attr[0]
        assert dim == 0, "Currently not supported"
        shape = list(ops[0])
        shape[dim] = ops[1][-1]
        if len(ops[1]) > 1:
            shape.insert(dim, 1)
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

    if type == "pad":
        assert len(attr) == 4 or len(attr) == 6
        shape = list(ops[0])
        channel_last = attr[-1]

        if channel_last:
            shape[-2] += attr[0] + attr[1]
            if len(attr) == 6:
                shape[-3] += attr[2] + attr[3]
        else:
            shape[-1] += attr[0] + attr[1]
            if len(attr) == 6:
                shape[-2] += attr[2] + attr[3]
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


def lower(type, attr, lc, ops, outputs):
    assert len(ops) == 1, "Tensor manipulation ops should have one input"

    if type == "reshape":
        while len(attr) > 4:
            assert attr[0] == 1, "Cannot eliminate non-singleton dimension"
            attr = attr[1:]
        while len(attr) < 4:
            attr.insert(0, 1)

        # Pad shape to 4D before lowering
        orig_shape = []
        for i in range(ops[0].shape.len()):
            orig_shape.append(ops[0].shape[i])
        while len(orig_shape) < 4:
            orig_shape.insert(0, 1)

        assert len(attr) == 4, "Reshape should have 4 attributes"

        # Squeeze / unsqueeze ops that do not reshape a 4d tensor are nops
        if all([orig == new for orig, new in zip(orig_shape, attr)]):
            lc.op(ForgeNop.create(), ops)
        else:
            orig_w = orig_shape[-4]
            orig_z = orig_shape[-3]
            orig_r = orig_shape[-2]
            orig_c = orig_shape[-1]
            forge_attrs = {
                "orig_w": orig_w,
                "orig_z": orig_z,
                "orig_r": orig_r,
                "orig_c": orig_c,
                "w": attr[0],
                "z": attr[1],
                "r": attr[2],
                "c": attr[3],
            }
            lc.op(type, ops, (orig_w, orig_z, orig_r, orig_c, *attr), forge_attrs)

    elif type == "transpose":
        # Transpose has 3 attrs, [axis_0, axis_1, output Z-dim size]
        assert len(attr) == 3, "Transpose should have 3 attributes"
        if attr[0] < 0:
            attr[0] += ops[0].shape.len()
        if attr[1] < 0:
            attr[1] += ops[0].shape.len()

        # Adjust the broadcast dim if we're moving to more/less dimensions
        delta = 4 - ops[0].shape.len()
        attr[0] += delta
        attr[1] += delta
        assert attr[0] >= 0 and attr[0] <= 3, f"Invalid transpose dim after lowering: {attr[0]}"
        assert attr[1] >= 0 and attr[1] <= 3, f"Invalid transpose dim after lowering: {attr[0]}"

        if attr[0] == 2 and attr[1] == 3:
            lc.tm("transpose", ops[0], attr, named_attrs={"dim0": attr[0], "dim1": attr[1]})
        else:
            lc.op("transpose", ops, attr, {"dim0": attr[0], "dim1": attr[1]})

    elif type == "broadcast":
        if attr[0] < 0:
            attr[0] += ops[0].shape.len()
        # Adjust the broadcast dim if we're moving to more/less dimensions
        delta = 4 - ops[0].shape.len()
        attr[0] += delta
        assert attr[0] >= 0 and attr[0] <= 3, f"Invalid broadcast dim after lowering: {attr[0]}"

        if attr[0] == 2 or attr[0] == 3:
            # Adjust broadcast size if not divisible by tile dim
            attr[1] = int(math.ceil(attr[1] / TILE_DIM)) * TILE_DIM
            attr[1] //= TILE_DIM

        return lc.tm("broadcast", ops[0], attr)

    elif type == "repeat":
        assert False, "repeat should have been decomposed into repeat_interleave"

    elif type == "repeat_interleave":
        # Adjust the repeat interleave if we're moving to more/less dimensions
        repeats = attr[0]
        dim = attr[1]

        if dim < 0:
            dim += ops[0].shape.len()

        delta = 4 - ops[0].shape.len()
        dim += delta
        assert dim >= 0 and dim <= 3, f"Invalid repeat interleave after lowering: {dim}"

        if dim == 2:
            assert ops[0].shape[-2] % TILE_DIM == 0, "Repeat on R must be TILE_DIM aligned"
        if dim == 3:
            assert ops[0].shape[-1] % TILE_DIM == 0, "Repeat on C must be TILE_DIM aligned"
        return lc.tm("broadcast", ops[0], attr)

    elif type == "select":
        assert len(attr) == 4, "Select should have 4 attributes"
        dim, index, length, stride = attr
        return lc.op(Splice.create_select(dim, index, length, stride, ops[0].shape), ops)

    elif type == "gather":
        assert len(attr) == 5, "Gather should have 5 attributes"
        dim, index, length, stride, orig_size = attr
        if dim >= 0:
            dim += 4 - ops[0].shape.len()
        else:
            dim += 4
        return lc.op(
            "gather",
            ops,
            (dim, index, length, stride, orig_size),
            {"index": index, "length": length, "stride": stride, "size": orig_size},
        )

    elif type == "pad_tile":
        return lc.op(ForgeNop.create(), ops)

    elif type == "narrow":
        assert len(attr) == 4
        dim, start, length, original_length = attr
        if dim >= 0:
            dim -= len(ops[0].shape)
        if dim >= -2 and align_up_tile(length) == align_up_tile(ops[0].shape[dim]):
            return lc.op(ForgeNop.create(), ops)
        else:
            raise NotImplementedError("Unimplemented narrow in forge")

    elif type == "pad":
        assert (
            (len(attr) == 4 and attr[0] == 0) or (len(attr) == 6 and attr[0] == 0 and attr[2] == 0) or (attr[-2] != 0)
        ), "Nop does not support left/top padding for constant mode"
        return lc.op(ForgeNop.create(), ops)

    elif type == "unsqueeze":
        assert len(attr) == 2
        input_ndim = attr[1]
        # assert input_ndim + 1 <= 4, "Cannot unsqueeze beyond 4D"
        if input_ndim + 1 > 4:
            assert attr[0] == 0, f"Unsqueeze 4D tensors to 5D is only supported for the 1st dim: {attr[0]}"
            return lc.op(ForgeNop.create(unsqueeze="unsqueeze", unsqueeze_dim=attr[1]), ops, tag="dont_remove")

        return lc.op(ForgeNop.create(), ops)

    elif type == "squeeze":
        assert len(attr) == 1
        if len(ops[0].shape) >= 5:
            assert attr[0] == 0, f"Squeeze 5D tensors to 4D is only supported for the 1st dim: {attr[0]}"
            return lc.op(ForgeNop.create(squeeze="squeeze", squeeze_dim=attr[0]), ops, tag="dont_remove")

        return lc.op(ForgeNop.create(), ops)

    elif type == "forge_pad":
        return lc.tm("forge_pad", ops[0], attr, {"rt": attr[0], "ct": attr[1], "pad_value": attr[2]})

    elif type == "forge_unpad":
        return lc.tm("forge_unpad", ops[0], attr, {"rt": attr[0], "ct": attr[1], "orig_r": attr[2], "orig_c": attr[3]})

    else:
        lc.tm(type, ops[0], attr)  # straight 1-1 for other tms


def backward(type, attr, ac, operand, inputs, output, grad):

    assert operand == 0, "Invalid operand index"

    if type == "transpose":
        assert len(attr) == 3

        if (attr[0] == -3 and attr[1] == -4) or (attr[0] == -4 and attr[1] == -3):
            attr[-1] = -1
        elif attr[0] == -3 or attr[0] == -4:
            attr[-1] = grad.shape[attr[1]]
        elif attr[1] == -3 or attr[1] == -4:
            attr[-1] = grad.shape[attr[0]]
        else:
            attr[-1] = -1

        return ac.op("transpose", (grad,), attr)

    elif type == "reshape":
        shape = inputs[0].shape
        return ac.op(type, (grad,), attributes=(shape), named_attrs={"shape": shape})

    elif type == "conv2d_depthwise_weights":
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

    elif type == "pad":  # TODO: update it for replicate mode
        assert len(attr) == 4 or len(attr) == 6, "Not supported padding type"
        if len(attr) == 6:
            pad_left, pad_right, pad_top, pad_bottom, _, _ = attr
            original_heigth = grad.shape[-2]  # input heigth
            original_width = grad.shape[-1]  # input width
            grad = ac.op("narrow", (grad,), (-2, pad_top, original_heigth - pad_top - pad_bottom, original_heigth))
            return ac.op("narrow", (grad,), (-1, pad_left, original_width - pad_left - pad_right, original_width))
        elif len(attr) == 4:
            pad_left, pad_right, _, _ = attr
            original_width = grad.shape[-1]  # input width
            return ac.op("narrow", (grad,), (-1, pad_left, original_width - pad_left - pad_right, original_width))

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
            ret = ac.op(
                TransposeTM.create(attr[0], -2),
                [
                    grad,
                ],
            )
            ret = ac.op("reduce_sum", (ret,), (-2,), {"keep_dim": True})
            ret = ac.op(
                TransposeTM.create(attr[0], -2),
                [
                    ret,
                ],
            )
        return ret

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
    act = inputs[0]

    if type == "index":
        assert len(attr) == 4, "Index should have 4 attributes"
        dim, start, stop, stride = attr

        if start < 0:
            # If start is less than zero (Python-style indexing), convert it to positive index
            # by adding the size on that dimension
            start = act.shape[dim] + start

        if stop < 0:
            # If start is less than zero (Python-style indexing), convert it to positive index
            # by adding the size on that dimension
            stop = act.shape[dim] + stop

        is_one_dim = len(act.shape) == 1
        if is_one_dim:
            # If input is a one-dimensional tensor, reshape it to a 2D tensor with one dimension equal to 1
            # and the other equal to the length. Use unsqueeze to add a dimension to the tensor.
            act = dc.op_with_named_attrs("unsqueeze", [act], {"dim": 0}, (0, len(act.shape)))

        row_indices = list(range(start, stop, stride))

        lhs_num_cols = act.shape[-2] if dim == -2 else act.shape[dim]
        lhs_num_channels = None
        lhs_batch_size = None

        if len(act.shape) == 4:
            # If len(act.shape) == 4, we have a batch dimension
            lhs_batch_size = act.shape[-4]

        if len(act.shape) >= 3:
            # If len(act.shape) >= 3, we have a channel dimension
            # channel dimension of the left hand side of the picker matmul is act.shape[-3] unless we index on -3 (dim != -3)
            # in that case we will do transpose with axis -2 to get the channel dimension at -2 position and then index on -2.
            lhs_num_channels = act.shape[-3] if dim != -3 else act.shape[-2]

        if dim != -2:
            # We need to transpose to get the dimension we want to index by at the -2 position
            act = dc.op(TransposeTM.create(-2, dim), [act])

        orig_act_shape = None
        if len(act.shape) > 3:
            # Add reshape to handle matmul's input tensor more than 3D
            orig_act_shape = act.shape.as_list()
            new_shape = (1, math.prod(orig_act_shape[:-2]), orig_act_shape[-2], orig_act_shape[-1])
            act = dc.op_with_named_attrs("reshape", [act], {"shape": new_shape}, new_shape)

            lhs_num_cols = act.shape[-2]
            lhs_num_channels = act.shape[-3]
            lhs_batch_size = 1

        lhs = create_row_picker_matrix(row_indices, lhs_num_cols, lhs_num_channels, lhs_batch_size)
        result = picker_matmul(False, dc, lhs, act)

        if orig_act_shape is not None:
            # Reshape back to original dimensions
            orig_act_shape[-1] = result.shape[-1]
            orig_act_shape[-2] = result.shape[-2]
            result = dc.op_with_named_attrs("reshape", [result], {"shape": orig_act_shape}, orig_act_shape)

        if dim != -2:
            # We need to transpose again to return to the original order of dimensions
            result = dc.op(TransposeTM.create(-2, dim), [result])

        dc.fuse(result)
        return

    if type == "adv_index":
        dim = attr[0]
        in0_shape = inputs[0].shape
        in1_shape = inputs[1].shape
        if len(in0_shape) == 1 or in0_shape[dim] == 1:
            result = dc.op(Nop.create(), [inputs[0]])
            dc.fuse(result)
            return
        if dim == 0 and len(in1_shape) <= 2:
            # Consider the case adv_index(X,Y) where
            #    X: (A, B), Y: (1, C) or (C,) and A != 1
            if len(in0_shape) == 2:
                # embedding op expects indices tensor as first argument and weight/embedding_table as second argument
                # but the adv_index provides the reference tensor as first argument and indices tensor as second argument
                # so swaping the operands.
                result = dc.op(
                    "embedding",
                    (inputs[1], inputs[0]),
                )
                dc.fuse(result)
                return

    if type == "pad":
        if all([x == 0 for x in attr[0:-2]]):
            # Pad size is 0
            result = dc.op(Nop.create(), [inputs[0]])
            dc.fuse(result)

        activations = inputs[0]
        mode_idx = attr[-2]
        channel_last = attr[-1]
        if channel_last:
            r = activations.shape[-3]
            c = activations.shape[-2]
        else:
            r = activations.shape[-2]
            c = activations.shape[-1]

        # Find out if padding exceeds tile boundary
        # R, C are flipped because pytorch pad starts from last axis
        if len(attr) == 4:
            total_padding_c = attr[0] + attr[1]
            total_padding_r = 0
            all_around_padding = attr[:-2] + [0, 0]
        elif len(attr) == 6:
            total_padding_c = attr[0] + attr[1]
            total_padding_r = attr[2] + attr[3]
            all_around_padding = attr[:-2]
        else:
            raise RuntimeError("Forge only support Pad with either 2 or 4 attributes")

        if (
            ((len(attr) == 4 and attr[0] == 0) or (len(attr) == 6 and attr[0] == 0 and attr[2] == 0))
            and not channel_last
            and math.ceil((total_padding_r + r) / TILE_DIM) == math.ceil(r / TILE_DIM)
            and math.ceil((total_padding_c + c) / TILE_DIM) == math.ceil(c / TILE_DIM)
            and mode_idx == 0  # 'constant' mode
        ):
            # Pad does not exceed tile boundary and only on the end of axis
            # Will be lowered into NOP
            return

        else:
            # Lower into concats
            left, right, top, bottom = 0, 0, 0, 0
            if len(attr) == 4:
                left, right, _, _ = attr

            elif len(attr) == 6:
                left, right, top, bottom, _, _ = attr
            else:
                raise RuntimeError("Forge only support Pad with either 3 or 5 attributes")

            if mode_idx == 1:  # 'replicate' mode
                result = activations

                if channel_last:
                    result = dc.op(TransposeTM.create(-3, -1, result.shape[-3]), [result])

                    orig_shape = result.shape
                    result = dc.op("reshape", [result], (1, 1, orig_shape[-3], orig_shape[-2] * orig_shape[-1]))
                    result = dc.op(TransposeTM.create(-2, -1), [result])
                    spm = create_pad_replicate_sparse_picker(c, r, top, bottom, left, right)
                    spm = dc.tensor(spm)
                    result = dc.op("sparse_matmul", [spm, result])
                    result = dc.op(TransposeTM.create(-2, -1), [result])
                    result = dc.op(
                        "reshape",
                        [result],
                        (1, orig_shape[-3], orig_shape[-1] + total_padding_r, orig_shape[-2] + total_padding_c),
                    )

                    result = dc.op(TransposeTM.create(-3, -1, result.shape[-3]), [result])
                else:
                    orig_shape = result.shape
                    if len(orig_shape) == 2:
                        result = dc.op("reshape", [result], (1, orig_shape[-2] * orig_shape[-1]))
                    else:
                        result = dc.op("reshape", [result], (1, 1, orig_shape[-3], orig_shape[-2] * orig_shape[-1]))
                    result = dc.op(TransposeTM.create(-2, -1), [result])
                    spm = create_pad_replicate_sparse_picker(r, c, left, right, top, bottom)
                    spm = dc.tensor(spm)
                    result = dc.op("sparse_matmul", [spm, result])
                    result = dc.op(TransposeTM.create(-2, -1), [result])
                    if len(orig_shape) == 2:
                        result = dc.op(
                            "reshape", [result], (orig_shape[-2] + total_padding_r, orig_shape[-1] + total_padding_c)
                        )
                    else:
                        result = dc.op(
                            "reshape",
                            [result],
                            (1, orig_shape[-3], orig_shape[-2] + total_padding_r, orig_shape[-1] + total_padding_c),
                        )

                dc.fuse(result)
                return

            elif mode_idx == 0:  # 'constant' mode
                c_dim_axis = -2 if channel_last else -1
                r_dim_axis = -3 if channel_last else -2

                # On right or bottom, we can concat all the way to TILE boundary
                result = activations
                if left > 0:
                    pad_shape = result.shape.as_list().copy()
                    pad_shape[c_dim_axis] = left
                    tensor = torch.zeros(pad_shape)
                    const_tensor = dc.tensor(tensor)
                    result = dc.op("concatenate", [const_tensor, result], [c_dim_axis])

                if right > 0:
                    pad_shape = result.shape.as_list().copy()
                    pad_shape[c_dim_axis] = (
                        TILE_DIM if pad_shape[c_dim_axis] % TILE_DIM == 0 and right < TILE_DIM else right
                    )
                    tensor = torch.zeros(pad_shape)
                    const_tensor = dc.tensor(tensor)
                    result = dc.op("concatenate", [result, const_tensor], [c_dim_axis])

                if top > 0:
                    pad_shape = result.shape.as_list().copy()
                    pad_shape[r_dim_axis] = top
                    tensor = torch.zeros(pad_shape)
                    const_tensor = dc.tensor(tensor)
                    result = dc.op("concatenate", [const_tensor, result], [r_dim_axis])

                if bottom > 0:
                    pad_shape = result.shape.as_list().copy()
                    pad_shape[r_dim_axis] = (
                        TILE_DIM if pad_shape[r_dim_axis] % TILE_DIM == 0 and bottom < TILE_DIM else bottom
                    )
                    tensor = torch.zeros(pad_shape)
                    const_tensor = dc.tensor(tensor)
                    result = dc.op("concatenate", [result, const_tensor], [r_dim_axis])

                result = dc.op("narrow", [result], (c_dim_axis, 0, total_padding_c + c, result.shape[c_dim_axis]))
                if channel_last:
                    result = dc.op("select", [result], (r_dim_axis, 0, total_padding_r + r, result.shape[r_dim_axis]))
                else:
                    result = dc.op("narrow", [result], (r_dim_axis, 0, total_padding_r + r, result.shape[r_dim_axis]))

                dc.fuse(result)
                return

            elif mode_idx == 2:
                # Reflect mode
                result = activations

                if channel_last:
                    result = dc.op(TransposeTM.create(-3, -1, result.shape[-3]), [result])

                    orig_shape = result.shape
                    result = dc.op_with_named_attrs(
                        "reshape",
                        [result],
                        {"shape": (1, 1, orig_shape[-3], orig_shape[-2] * orig_shape[-1])},
                        (1, 1, orig_shape[-3], orig_shape[-2] * orig_shape[-1]),
                    )
                    result = dc.op(TransposeTM.create(-2, -1), [result])
                    spm = create_pad_reflect_sparse_picker(c, r, top, bottom, left, right)
                    spm = dc.tensor(spm.to_dense())
                    result = dc.op("matmul", [spm, result])
                    result = dc.op(TransposeTM.create(-2, -1), [result])
                    result = dc.op_with_named_attrs(
                        "reshape",
                        [result],
                        {
                            "shape": (
                                1,
                                orig_shape[-3],
                                orig_shape[-1] + total_padding_r,
                                orig_shape[-2] + total_padding_c,
                            )
                        },
                        (1, orig_shape[-3], orig_shape[-1] + total_padding_r, orig_shape[-2] + total_padding_c),
                    )

                    result = dc.op(TransposeTM.create(-3, -1, result.shape[-3]), [result])
                else:
                    orig_shape = result.shape
                    if len(orig_shape) == 2:
                        shape = (1, orig_shape[-2] * orig_shape[-1])
                    else:
                        shape = (1, 1, orig_shape[-3], orig_shape[-2] * orig_shape[-1])

                    result = dc.op_with_named_attrs("reshape", [result], {"shape": shape}, shape)
                    result = dc.op(TransposeTM.create(-2, -1), [result])
                    spm = create_pad_reflect_sparse_picker(r, c, left, right, top, bottom)
                    spm = dc.tensor(spm.to_dense())
                    result = dc.op("matmul", [spm, result])
                    result = dc.op(TransposeTM.create(-2, -1), [result])

                    if len(orig_shape) == 2:
                        shape = (orig_shape[-2] + total_padding_r, orig_shape[-1] + total_padding_c)
                    else:
                        shape = (1, orig_shape[-3], orig_shape[-2] + total_padding_r, orig_shape[-1] + total_padding_c)

                    result = dc.op_with_named_attrs("reshape", [result], {"shape": shape}, shape)

                dc.fuse(result)
                return

    if type == "broadcast":
        if attr[1] == 1:
            dc.fuse(dc.op(Nop.create(), [inputs[0]]))

    if type == "transpose":
        # canonicalize dims to use negative indexing
        dim0, dim1, orig_size = attr
        if dim0 >= 0 or dim1 >= 0:
            if dim0 >= 0:
                dim0 -= inputs[0].shape.len()
            if dim1 >= 0:
                dim1 -= inputs[0].shape.len()
            dc.fuse(dc.op(TransposeTM.create(dim0, dim1, orig_size)), inputs)

    if type == "pixel_shuffle":
        result = inputs[0]  # (1, C*r*r, H, W)
        orig_shape = result.shape
        if attr[0] != 2:
            raise NotImplementedError("Pixel shuffle decomposition only supports r=2")

        r = attr[0]
        C = orig_shape[-3] // (r * r)
        H = orig_shape[-2]
        W = orig_shape[-1]

        result = dc.op("vstack", [result], (r * r,))
        sub_slices = []
        for subsection in range(r):
            sub_slice = dc.op("select", [result], (-2, subsection * r * H, r * H, result.shape[-2]))
            sub_sub_slices = []
            for subsubsection in range(r):
                sub_sub_slices.append(dc.op("select", [sub_slice], (-2, subsubsection * H, H, sub_slice.shape[-2])))

            curr_sub_sub_slice = sub_sub_slices[0]
            for sub_sub_slice in sub_sub_slices[1:]:
                curr_sub_sub_slice = dc.op("binary_stack", [curr_sub_sub_slice, sub_sub_slice], (-1,))

            sub_slices.append(curr_sub_sub_slice)

        curr_sub_slice = dc.op(TransposeTM.create(-2, -1), [sub_slices[0]])
        for sub_slice in sub_slices[1:]:
            sub_slice = dc.op(TransposeTM.create(-2, -1), [sub_slice])
            curr_sub_slice = dc.op("binary_stack", [curr_sub_slice, sub_slice], (-1,))

        result = dc.op(TransposeTM.create(-2, -1), [curr_sub_slice])
        dc.fuse(result)

    if type == "reshape":
        assert len(inputs) == 1
        input_shape = inputs[0].shape.as_list()
        shape = list(attr)

        if shape == input_shape:
            # dc.fuse(dc.op("nop", [inputs[0]]))
            return

        rank = 0
        while len(shape) < len(input_shape):
            shape.insert(0, 1)
            rank -= 1
        while len(shape) > len(input_shape) and shape[0] == 1:
            shape = shape[1:]
            rank += 1

        is_rank_only_reshape = shape == input_shape
        if is_rank_only_reshape and rank != 0:
            result = inputs[0]
            while rank < 0:
                result = dc.op_with_named_attrs("squeeze", [result], {"dim": 0}, (0,))
                rank += 1
            while rank > 0:
                result = dc.op_with_named_attrs("unsqueeze", [result], {"dim": 0}, (0, len(result.shape.as_list())))
                rank -= 1
            dc.fuse(result)
            return
    if type == "repeat":
        input_shape = inputs[0].shape.as_list()
        target_shape = attr
        result = inputs[0]

        if len(input_shape) < len(target_shape):
            # Scenario: When the input is a 1D tensor and needs to be repeated in 2D,
            # `ttir.repeat` does not currently support this directly.
            # To handle this, we first reshape the input to ensure both the input and the repeats have the same dimensions
            new_shape = (1,) * (len(target_shape) - len(input_shape)) + tuple(input_shape)
            result = dc.op("reshape", [result], new_shape)
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
            x = dc.op(TransposeTM.create(-2, -1), [x])

        result = dc.op("sparse_matmul", [spm, x])

        if is_x_select:
            result = dc.op(TransposeTM.create(-2, -1), [result])

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
            result = dc.op(
                TransposeTM.create(-2, -1),
                [
                    result,
                ],
            )
            result = picker_matmul(True, dc, spm, result)
            result = dc.op(
                TransposeTM.create(-2, -1),
                [
                    result,
                ],
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
            result = dc.op(
                TransposeTM.create(-2, -1),
                [
                    result,
                ],
            )
            result = picker_matmul(True, dc, spm, result)
            result = dc.op(
                TransposeTM.create(-2, -1),
                [
                    result,
                ],
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


def decompose_post_autograd(type, attr, dc, inputs):
    if type == "reshape":
        assert len(inputs) == 1
        input_shape = inputs[0].shape.as_list()
        shape = list(attr)

        if shape == input_shape:
            # dc.fuse(dc.op(Nop.create(), [inputs[0]]))
            return

        rank = 0
        while len(shape) < len(input_shape):
            shape.insert(0, 1)
            rank -= 1
        while len(shape) > len(input_shape) and shape[0] == 1:
            shape = shape[1:]
            rank += 1

        is_rank_only_reshape = shape == input_shape
        if is_rank_only_reshape and rank != 0:
            result = inputs[0]
            while rank < 0:
                result = dc.op_with_named_attrs("squeeze", [result], {"dim": 0}, (0,))
                rank += 1
            while rank > 0:
                import pdb

                pdb.set_trace
                result = dc.op_with_named_attrs("unsqueeze", [result], {"dim": 0}, (0, len(result.shape.as_list())))
                rank -= 1
            dc.fuse(result)
            return
