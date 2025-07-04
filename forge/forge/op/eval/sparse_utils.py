# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import math
import numpy as np
import torch
from loguru import logger
import forge
from forge.utils import align_up_tile, align_up, round_up_div, clamp
from ...forgeglobal import TILE_DIM
from forge._C import DataFormat, compress_sparse_tensor_and_strip_info, SparseCOO, SparseFORGE, MathFidelity


def conv2d_padding_to_canonical(padding, kernel_size):
    # current implementation is without dilation

    assert isinstance(kernel_size, int) or isinstance(kernel_size, tuple), "Unsupported kernel size"
    kH, kW = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)

    if isinstance(padding, int):
        return [padding] * 4
    elif isinstance(padding, str):
        assert padding == "same", "Unsupported padding"
        padding = [kW // 2] * 2 + [kH // 2] * 2
        if kW % 2 == 0:
            padding[1] -= 1
        if kH % 2 == 0:
            padding[3] -= 1
        return padding
    elif isinstance(padding, tuple) or isinstance(padding, list):
        if len(padding) == 2:
            return [padding[1]] * 2 + [padding[0]] * 2
        elif len(padding) == 4:
            return list(padding)
        else:
            raise AssertionError("Unsupported padding")
    else:
        raise AssertionError("Unsupported padding")


def conv3d_padding_to_canonical(padding, kernel_size):
    # current implementation is without dilation

    assert isinstance(kernel_size, int) or isinstance(kernel_size, tuple), "Unsupported kernel size"
    kD, kH, kW = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)

    if isinstance(padding, int):
        return [padding] * 6
    elif isinstance(padding, str):
        assert padding == "same", "Unsupported padding"
        padding = [kW // 2] * 2 + [kH // 2] * 2 + [kD // 2] * 2
        if kW % 2 == 0:
            padding[1] -= 1
        if kH % 2 == 0:
            padding[3] -= 1
        if kD % 2 == 0:
            padding[5] -= 1
        return padding
    elif isinstance(padding, tuple) or isinstance(padding, list):
        if len(padding) == 2:
            return [padding[1]] * 3 + [padding[0]] * 3
        elif len(padding) == 6:
            return list(padding)
        else:
            raise AssertionError("Unsupported padding")
    else:
        raise AssertionError("Unsupported padding")


def calculate_conv2d_output_dimensions(
    original_y, original_x, kernel_size, stride, padding, dilation=1, ceil_mode=False
):
    if isinstance(stride, int):
        stride = [stride] * 2

    assert len(padding) == 4 and all(isinstance(x, int) for x in padding), "Padding should be list of four ints"

    # Pooling layers (max, avg)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    # Padding is [left, right, top, bottom]
    if ceil_mode:
        y = math.ceil((original_y + padding[2] + padding[3] - dilation * (kernel_size[0] - 1) - 1) / stride[0]) + 1
        x = math.ceil((original_x + padding[0] + padding[1] - dilation * (kernel_size[1] - 1) - 1) / stride[1]) + 1
    else:
        y = (original_y + padding[2] + padding[3] - dilation * (kernel_size[0] - 1) - 1) // stride[0] + 1
        x = (original_x + padding[0] + padding[1] - dilation * (kernel_size[1] - 1) - 1) // stride[1] + 1
    return y, x


def calculate_conv3d_output_dimensions(
    original_z, original_y, original_x, kernel_size, stride, padding, dilation=1, ceil_mode=False
):
    if isinstance(stride, int):
        stride = [stride] * 3

    assert len(padding) == 6 and all(isinstance(x, int) for x in padding), "Padding should be list of six ints"

    # Pooling layers (max, avg)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    # Padding is [left, right, top, bottom]
    if ceil_mode:
        z = math.ceil((original_z + padding[4] + padding[5] - dilation * (kernel_size[0] - 1) - 1) / stride[0]) + 1
        y = math.ceil((original_y + padding[2] + padding[3] - dilation * (kernel_size[1] - 1) - 1) / stride[1]) + 1
        x = math.ceil((original_x + padding[0] + padding[1] - dilation * (kernel_size[2] - 1) - 1) / stride[2]) + 1
    else:
        z = (original_z + padding[4] + padding[5] - dilation * (kernel_size[0] - 1) - 1) // stride[0] + 1
        y = (original_y + padding[2] + padding[3] - dilation * (kernel_size[1] - 1) - 1) // stride[1] + 1
        x = (original_x + padding[0] + padding[1] - dilation * (kernel_size[2] - 1) - 1) // stride[2] + 1
    return z, y, x


def calculate_conv2d_transpose_output_dimensions(
    original_y, original_x, kernel_size, stride, padding, dilation=1, output_padding=0
):
    if isinstance(padding, int):
        padding = [padding] * 4

    y = (original_y - 1) * stride - (padding[2] + padding[3]) + dilation * (kernel_size[0] - 1) + 1 + output_padding
    x = (original_x - 1) * stride - (padding[0] + padding[1]) + dilation * (kernel_size[1] - 1) + 1 + output_padding
    return y, x


def up_idx_to_orig_idx_no_align_corners(up_idx, scale_factor):
    return (up_idx + 0.5) / scale_factor - 0.5


def up_idx_to_orig_idx_align_corners(up_idx, original_size, upsample_size):
    up_idx = up_idx.tolist()
    stride = (original_size - 1) / (upsample_size - 1)
    x_ori_list = []
    # append the first coordinate
    x_ori_list.append(0)
    for i in range(1, len(up_idx) - 1):
        x_ori_list.append(0 + i * stride)
    # append the last coordinate
    x_ori_list.append(original_size - 1)
    return torch.tensor(x_ori_list)


def pad_tensor_zeros(shape):
    return torch.zeros(shape)


def pad_tensor_identity(shape, pad):
    size = shape[-2]
    shape[-2] = size + pad
    shape[-1] = size
    tensor = torch.concat((torch.eye(size), torch.zeros(pad, size)), dim=-2)
    return torch.broadcast_to(tensor, shape)


def pad_tensor_identity_transposed(size, pad):
    return torch.transpose(torch.concat((torch.eye(size), torch.zeros(pad, size)), dim=-2), -1, -2)


def pad_tensor_identity_sparse_transposed(size, pad):
    return pad_tensor_identity_transposed(size, pad).to_sparse()


def conv2d_out_shape(type, attr, ops):
    assert len(ops) <= 3, "Conv2d should have three inputs"
    # assert len(attr) == 10, f"Conv2d invalid num attributes: {len(attr)}"

    activations = ops[0]
    weights = ops[1]
    kernel_size = [weights[2], weights[3]]
    stride = [
        attr[0],
        attr[1],
    ]
    dilation = attr[2]
    groups = attr[3]
    padding = [
        attr[4],
        attr[5],
        attr[6],
        attr[7],
    ]
    is_convtranspose2d = attr[8]  # True if decomposed from convtranspose2d
    channel_last = attr[-1]

    if channel_last == 1:
        in_y = activations[1]
        in_x = activations[2]
    else:
        in_y = activations[2]
        in_x = activations[3]

    if type == "conv2d":
        y, x = calculate_conv2d_output_dimensions(in_y, in_x, kernel_size, stride, padding, dilation)

        # TODO: the existence of this `if` block is a but confusing, should be fixed once this proposal is implemented:
        # tenstorrent/forge#1761
        if is_convtranspose2d:
            # if transposed conv, the output is calculated by `calculate_conv2d_transpose_output_dimensions()`
            # however, we can't call this function on conv2d, as some attributes have been changed to fit the style of
            # conv2d (e.g. padding) - it would produce wrong numbers
            yout_transpose = attr[9]
            xout_transpose = attr[10]
            y = yout_transpose
            x = xout_transpose

        if channel_last == 1:
            return (activations[0], y, x, weights[0]), []
        else:
            return (activations[0], weights[0], y, x), []
    elif type == "conv2d_transpose":
        assert dilation == 1, "Currently only support dilation = 1"
        assert all([p == padding[0] for p in padding]), "ConvTranspose2d only supports same padding on all sides"
        assert all([s == stride[0] for s in stride]), "ConvTranspose2d only supports same strides"
        stride = stride[0]

        y, x = calculate_conv2d_transpose_output_dimensions(
            in_y, in_x, (weights[2], weights[3]), stride, padding, dilation=dilation
        )

        if channel_last == 1:
            return (activations[0], y, x, weights[1] * groups), []
        else:
            return (activations[0], weights[1] * groups, y, x), []


def conv3d_out_shape(type, attr, ops):
    assert len(ops) <= 3, "Conv3d should have three inputs"
    assert len(attr) == 12, f"Conv3d invalid num attributes: {len(attr)}"

    activations = ops[0]
    weights = ops[1]
    kernel_size = [weights[2], weights[3], weights[4]]
    stride = [attr[0], attr[1], attr[2]]
    dilation = attr[3]
    groups = attr[4]
    padding = [attr[5], attr[6], attr[7], attr[8], attr[9], attr[10]]

    if attr[-1] == 1:
        # Channel last
        in_z = activations[1]
        in_y = activations[2]
        in_x = activations[3]
    else:
        in_z = activations[2]
        in_y = activations[3]
        in_x = activations[4]

    if type == "conv3d":
        z, y, x = calculate_conv3d_output_dimensions(in_z, in_y, in_x, kernel_size, stride, padding, dilation)
        if attr[-1] == 1:
            return (activations[0], z, y, x, weights[0]), []
        else:
            return (activations[0], weights[0], z, y, x), []
