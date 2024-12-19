# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os

import torch

from forge import Tensor
from forge.forgeglobal import TILE_DIM
from forge.op.resize import INT_TO_RESIZE2d_METHOD
from forge.utils import align_up_tile, clamp, round_up_div

from ..common import to_torch_operands
from ..sparse_utils import (
    create_bilinear_upsample_picker_matrix,
    create_nearest_neighbor_downsample_picker_matrix,
    create_nearest_neighbor_upsample_picker_matrix,
)
from .nop import Nop
from .transpose import TransposeTM


def eval(type, attr, ops):

    if type == "resize2d":
        assert len(attr) == 5, "Resize2d should have 4 attrs: [size, size, method, align_corners, channel_last]"
        assert len(ops) == 1
        resize_method = INT_TO_RESIZE2d_METHOD[attr[-3]]
        acts = ops[0]
        if attr[-1]:
            # channel last
            acts = ops[0].permute((0, 3, 1, 2))

        if resize_method == "nearest":
            upsample = torch.nn.Upsample(
                size=attr[0:2],
                mode=resize_method,
            )
        else:
            upsample = torch.nn.Upsample(
                size=attr[0:2],
                mode=resize_method,
                align_corners=bool(attr[-2]),
            )

        t_ops = to_torch_operands(acts)

        result = upsample(*t_ops)

        if attr[-1]:
            result = result.permute((0, 2, 3, 1))
        return result
    elif type == "resize3d":
        assert len(attr) == 6, "Resize3d should have 6 attrs: [size, size, size, method, align_corners, channel_last]"
        assert len(ops) == 1
        resize_method = INT_TO_RESIZE2d_METHOD[attr[-3]]
        acts = ops[0]
        if attr[-1]:
            # channel last
            acts = ops[0].permute((0, 4, 1, 2, 3))

        if resize_method == "nearest":
            upsample = torch.nn.Upsample(
                size=attr[0:3],
                mode=resize_method,
            )
        else:
            upsample = torch.nn.Upsample(
                size=attr[0:3],
                mode=resize_method,
                align_corners=bool(attr[-2]),
            )

        t_ops = to_torch_operands(acts)

        result = upsample(*t_ops)

        if attr[-1]:
            result = result.permute((0, 2, 3, 4, 1))
        return result


def shape(type, attr, ops):
    if type == "resize2d":
        assert len(attr) == 5, "Resize2d should have 4 attrs: [size, size, method, align_corners]"
        shape = list(ops[0])
        channel_last = attr[-1]
        upsample = attr[0] >= shape[-3] if channel_last else attr[0] >= shape[-2]
        if channel_last:
            if upsample:
                assert attr[1] >= shape[-2], "One dim upsamples, the other dim should also upsample"
                assert (
                    attr[0] % shape[-3] == 0 and attr[1] % shape[-2] == 0
                ), "Only support upsample with integer scale factor"
            else:
                assert attr[1] < shape[-2], "One dim downsamples, the other dim should also downsample"
                assert (
                    shape[-3] % attr[0] == 0 and shape[-2] % attr[1] == 0
                ), "Only support downsample with integer scale factor"
                assert shape[-3] // attr[0] == shape[-2] // attr[1], "Only support same scale factor for H and W"
            shape[-3], shape[-2] = attr[0], attr[1]
            return shape, []
        else:
            if upsample:
                assert attr[1] >= shape[-1], "One dim upsamples, the other dim should also upsample"
                assert (
                    attr[0] % shape[-2] == 0 and attr[1] % shape[-1] == 0
                ), "Only support upsample with integer scale factor"
            else:
                assert attr[1] < shape[-1], "One dim downsamples, the other dim should also downsample"
                assert (
                    shape[-2] % attr[0] == 0 and shape[-1] % attr[1] == 0
                ), "Only support downsample with integer scale factor"
                assert shape[-2] // attr[0] == shape[-1] // attr[1], "Only support same scale factor for H and W"
            shape[-2], shape[-1] = attr[0], attr[1]
            return shape, []
    elif type == "resize3d":
        assert len(attr) == 6, "Resize3d should have 6 attrs: [size, size, size, method, align_corners]"
        shape = list(ops[0])
        channel_last = attr[-1]
        upsample = attr[0] >= shape[-4] if channel_last else attr[0] >= shape[-3]
        if channel_last:
            if upsample:
                assert attr[1] >= shape[-3], "One dim upsamples, the other dim should also upsample"
                assert attr[2] >= shape[-2], "One dim upsamples, the other dim should also upsample"
                assert (
                    attr[0] % shape[-4] == 0 and attr[1] % shape[-3] == 0 and attr[2] % shape[-2]
                ), "Only support upsample with integer scale factor"
                assert (
                    attr[0] // shape[-4] == attr[1] // shape[-3] == attr[2] // shape[-2]
                ), "Only support same scale factor for H and W"
            else:
                assert attr[1] < shape[-3], "One dim downsamples, the other dim should also downsample"
                assert attr[2] < shape[-2], "One dim downsamples, the other dim should also downsample"
                assert (
                    shape[-4] % attr[0] == 0 and shape[-3] % attr[1] == 0 and shape[-2] % attr[2]
                ), "Only support downsample with integer scale factor"
                assert (
                    shape[-4] // attr[0] == shape[-3] // attr[1] == shape[-2] // attr[2]
                ), "Only support same scale factor for H and W"
            shape[-4], shape[-3], shape[-2] = attr[0], attr[1], attr[2]
            return shape, []
        else:
            if upsample:
                assert attr[1] >= shape[-2], "One dim upsamples, the other dim should also upsample"
                assert attr[2] >= shape[-1], "One dim upsamples, the other dim should also upsample"
                assert (
                    attr[0] % shape[-3] == 0 and attr[1] % shape[-2] == 0 and attr[2] % shape[-1] == 0
                ), "Only support upsample with integer scale factor"
            else:
                assert attr[1] < shape[-2], "One dim downsamples, the other dim should also downsample"
                assert attr[2] < shape[-1], "One dim downsamples, the other dim should also downsample"
                assert (
                    shape[-3] % attr[0] == 0 and shape[-2] % attr[1] == 0 and shape[-1] % attr[2] == 0
                ), "Only support downsample with integer scale factor"
                assert (
                    shape[-3] // attr[0] == shape[-2] // attr[1] == shape[-1] // attr[2]
                ), "Only support same scale factor for H and W"
            shape[-3], shape[-2], shape[-1] = attr[0], attr[1], attr[2]
            return shape, []


def lower(type, attr, lc, ops, outputs):
    raise RuntimeError("This should never be called.")


def backward(type, attr, ac, operand, inputs, output, grad):
    raise RuntimeError("This should never be called.")


def decompose_upsample_3d(attr, dc, inputs, resize_method):
    activations = inputs[0]
    shape = inputs[0].shape
    channel_last = attr[-1]
    # if channel_last:
    #    w, y, x, cin = (shape.w, shape.z, shape.r, shape.c)
    #    activations = dc.op("reshape", [activations], (w, 1, y * x, cin))
    #    scale_factor = attr[0] // shape[-3]
    # else:
    w, cin, din, y, x = (shape.v, shape.w, shape.z, shape.r, shape.c)
    activations = dc.op(
        "reshape",
        inputs,
        (w, 1, cin * din, y * x),
    )
    activations = dc.op(TransposeTM.create(-2, -1), [activations])
    scale_factor_d = attr[0] // shape[-3]
    scale_factor_y = attr[1] // shape[-2]
    scale_factor_x = attr[2] // shape[-1]
    scale_factor = (scale_factor_x, scale_factor_y, scale_factor_d)

    if resize_method == "nearest":
        dident_yx = create_nearest_neighbor_upsample_picker_matrix(scale_factor, shape, channel_last=channel_last)
        dident_tensor_yx = dc.tensor(dident_yx)
        result = dc.op("sparse_matmul", [dident_tensor_yx, activations])
        result = dc.op(TransposeTM.create(-2, -1), [result])

        dident_din = create_nearest_neighbor_upsample_picker_matrix(
            scale_factor, shape, for_din=True, channel_last=channel_last
        )
        dident_tensor_din = dc.tensor(dident_din)
        result = dc.op("sparse_matmul", [dident_tensor_din, result])
    else:
        raise NotImplementedError("Only nearest neighbor upsampling 3D is supported")

    # if channel_last:
    #    result = dc.op("reshape", [result], (w, y * scale_factor, x * scale_factor, cin))
    #    dc.fuse(result)
    # else:
    result = dc.op(
        "reshape",
        [result],
        (
            w,
            cin,
            din * scale_factor_d,
            y * scale_factor_y,
            x * scale_factor_x,
        ),
    )

    dc.fuse(result)


def decompose_resize3d(attr, dc, inputs, resize_method):
    activations = inputs[0]
    shape = inputs[0].shape
    channel_last = attr[-1]
    upsample = attr[0] >= shape[-4] if channel_last else attr[0] >= shape[-3]
    if channel_last:
        if upsample:
            assert attr[1] >= shape[-3], "One dim upsamples, the other dim should also upsample"
            assert attr[2] >= shape[-2], "One dim upsamples, the other dim should also upsample"
            assert (
                attr[0] % shape[-4] == 0 and attr[1] % shape[-3] == 0 and attr[2] % shape[-2]
            ), "Only support upsample with integer scale factor"
            assert (
                attr[0] // shape[-4] == attr[1] // shape[-3] == attr[2] // shape[-2]
            ), "Only support same scale factor for H and W"
        else:
            assert attr[1] < shape[-3], "One dim downsamples, the other dim should also downsample"
            assert attr[2] < shape[-2], "One dim downsamples, the other dim should also downsample"
            assert (
                shape[-4] % attr[0] == 0 and shape[-3] % attr[1] == 0 and shape[-2] % attr[2]
            ), "Only support downsample with integer scale factor"
            assert (
                shape[-4] // attr[0] == shape[-3] // attr[1] == shape[-2] // attr[2]
            ), "Only support same scale factor for H and W"
    else:
        if upsample:
            assert attr[1] >= shape[-2], "One dim upsamples, the other dim should also upsample"
            assert attr[2] >= shape[-1], "One dim upsamples, the other dim should also upsample"
            assert (
                attr[0] % shape[-3] == 0 and attr[1] % shape[-2] == 0 and attr[2] % shape[-1] == 0
            ), "Only support upsample with integer scale factor"
        else:
            assert attr[1] < shape[-2], "One dim downsamples, the other dim should also downsample"
            assert attr[2] < shape[-1], "One dim downsamples, the other dim should also downsample"
            assert (
                shape[-3] % attr[0] == 0 and shape[-2] % attr[1] == 0 and shape[-1] % attr[2] == 0
            ), "Only support downsample with integer scale factor"
            assert (
                shape[-3] // attr[0] == shape[-2] // attr[1] == shape[-1] // attr[2]
            ), "Only support same scale factor for H and W"

    scale_factor_d = attr[0] // shape[-3]
    scale_factor_y = attr[1] // shape[-2]
    scale_factor_x = attr[2] // shape[-1]
    if scale_factor_x == 1 and scale_factor_y == 1 and scale_factor_d == 1:
        result = dc.op(Nop.create(), [activations])
        dc.fuse(result)
        return

    if upsample:
        decompose_upsample_3d(attr, dc, inputs, resize_method)
    else:
        raise NotImplementedError("Downsampling of resize3d is not supported yet")


def decompose(type, attr, dc, inputs):
    if type == "resize3d":
        assert len(attr) == 6, "Resize3d should have 6 attrs: [size, size, size, method, align_corners, channel_last]"
        assert len(inputs) == 1
        resize_method = INT_TO_RESIZE2d_METHOD[attr[-3]]

        decompose_resize3d(attr, dc, inputs, resize_method)
    pass
