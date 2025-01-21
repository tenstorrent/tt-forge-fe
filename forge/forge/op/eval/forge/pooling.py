# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
import ast
import os
import math
import torch.nn.functional as F
from forge.forgeglobal import TILE_DIM
from forge.utils import align_up_tile
from .transpose import TransposeTM
from .nop import Nop
from .nop import Nop
from .convolution import Conv2d
from ..interface import PyOp

from ..common import to_torch_operands
from ..sparse_utils import (
    calculate_conv2d_output_dimensions,
    calculate_conv3d_output_dimensions,
    calculate_pad_for_ceil_mode,
    create_avg_pool2d_count_include_pad_False_picker_matrix,
    create_conv2d_sparse_picker_matrix,
)


class MaxPool2d(PyOp):
    @classmethod
    def create(
        cls,
        kernel_height: int,
        kernel_width: int,
        stride_height: int,
        stride_width: int,
        dilation_height: int,
        dilation_width: int,
        ceil_mode: bool,
        padding_left: int,
        padding_right: int,
        padding_top: int,
        padding_bottom: int,
        channel_last: bool,
    ):

        self = cls("max_pool2d")
        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
        self.stride_height = stride_height
        self.stride_width = stride_width
        self.dilation_height = dilation_height
        self.dilation_width = dilation_width
        self.ceil_mode = ceil_mode
        self.padding_left = padding_left
        self.padding_right = padding_right
        self.padding_top = padding_top
        self.padding_bottom = padding_bottom
        self.channel_last = channel_last

        return self

    def eval(self, tensors):
        activations = to_torch_operands(*tensors)[0]

        if self.channel_last:
            activations = activations.permute((0, 3, 1, 2))

        padded_activations = torch.nn.functional.pad(
            activations,
            (self.padding_left, self.padding_right, self.padding_top, self.padding_bottom),
            value=float("-inf"),
        )
        result = torch.nn.functional.max_pool2d(
            padded_activations,
            kernel_size=(self.kernel_height, self.kernel_width),
            stride=(self.stride_height, self.stride_width),
            padding=0,
            dilation=(self.dilation_height, self.dilation_width),
            ceil_mode=bool(self.ceil_mode),
            return_indices=False,
        )
        if self.channel_last:
            result = result.permute((0, 2, 3, 1))

        return result

    def shape(self, input_shapes):
        act = input_shapes[0]

        batch_size = act[0]
        channels = act[-1] if self.channel_last else act[-3]

        h_in = act[-3] if self.channel_last else act[-2]
        w_in = act[-2] if self.channel_last else act[-1]

        h_numerator = (
            h_in + (self.padding_top + self.padding_bottom) - self.dilation_height * (self.kernel_height - 1) - 1
        )
        if self.ceil_mode:
            h_out = math.ceil(1 + (h_numerator / self.stride_height))
        else:
            h_out = math.floor(1 + (h_numerator / self.stride_height))

        w_numerator = (
            w_in + (self.padding_left + self.padding_right) - self.dilation_width * (self.kernel_width - 1) - 1
        )
        if self.ceil_mode:
            w_out = math.ceil(1 + (w_numerator / self.stride_width))
        else:
            w_out = math.floor(1 + (w_numerator / self.stride_width))

        out_shape = [batch_size, h_out, w_out, channels] if self.channel_last else [batch_size, channels, h_out, w_out]

        return out_shape, []

    def decompose(self, dc, inputs):
        # TTNN can only perform a channel last pooling with its maxpool2d op.
        # The TTNN  maxpool2d requires the input to be in the shape: (N, H, W, C).
        # If the forge maxpool2d op is not channel last, we must permute the input (N, C, H, W) tensor to (N, H, W, C)
        # and then transpose it back to (N, C_out, H_out, W_out) afterward.
        #     - This is done with two transposes
        #     - (N, C, H, W) --> transpose(-3, -2): (N, H, C, W) --> transpose(-2, -1): (N, H, W, C)
        # Afterward:
        #     - (N, H_out, W_out, C_out) --> transpose(-2, -1): (N, H_out, C_out, W_out) --> transpose(-3, -2): (N, C_out, H_out, W_out)
        activations = inputs[0]

        is_channel_last = self.channel_last

        if not is_channel_last:
            activations = dc.op(TransposeTM.create(dim0=-3, dim1=-2), [activations])
            activations = dc.op(TransposeTM.create(dim0=-2, dim1=-1), [activations])

            new_inputs = [activations]
            result = dc.op(
                MaxPool2d.create(
                    self.kernel_height,
                    self.kernel_width,
                    self.stride_height,
                    self.stride_width,
                    self.dilation_height,
                    self.dilation_width,
                    self.ceil_mode,
                    self.padding_left,
                    self.padding_right,
                    self.padding_top,
                    self.padding_bottom,
                    True,  # channel_last
                ),
                new_inputs,
            )
            # Since decompose should result in new set of ops that is equivalent to the decomposed one, we need to transpose result back to channel first if it was channel first on input
            result = dc.op(TransposeTM.create(dim0=-2, dim1=-1), [result])
            result = dc.op(TransposeTM.create(dim0=-3, dim1=-2), [result])

            dc.fuse(result)

    def backward(self, ac, operand, inputs, output, grad):
        pass

    def lower(self, lc, tensors, outputs):
        pass

    def is_tm(self) -> bool:
        return False

    def is_eltwise(self) -> bool:
        return False

    def is_eltwise_binary(self) -> bool:
        return False

    def is_eltwise_unary(self) -> bool:
        return False

    def is_eltwise_nary(self) -> bool:
        return False


def eval(type, attr, ops):
    assert len(ops) == 1, "Pool ops should have one input"

    t_ops = to_torch_operands(*ops)
    activations = t_ops[0]

    if type == "max_pool1d":
        assert len(attr) == 5

        kernel_size = attr[0]
        stride = attr[1]
        dilation = attr[2]
        ceil_mode = attr[3]
        padding = attr[4]

        padded_activations = torch.nn.functional.pad(
            activations,
            (padding, padding),
            value=float("-inf"),
        )

        result = torch.nn.functional.max_pool1d(
            padded_activations,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            ceil_mode=bool(ceil_mode),
            return_indices=False,
        )
    elif type == "max_pool2d":
        assert len(attr) == 13
        kernel_size = [
            attr[0],
            attr[1],
        ]
        stride = [
            attr[2],
            attr[3],
        ]
        dilation = attr[4]
        ceil_mode = attr[5]
        padding = [
            attr[6],
            attr[7],
            attr[8],
            attr[9],
        ]
        channel_last = attr[-1]
        if channel_last:
            activations = activations.permute((0, 3, 1, 2))

        padded_activations = torch.nn.functional.pad(
            activations,
            padding,
            value=float("-inf"),
        )
        result = torch.nn.functional.max_pool2d(
            padded_activations,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            ceil_mode=bool(ceil_mode),
            return_indices=False,
        )
        if channel_last:
            result = result.permute((0, 2, 3, 1))
    elif type == "max_pool3d":
        assert len(attr) == 17, f"maxpool3d attr-len = {len(attr)}"
        kernel_size = [attr[0], attr[1], attr[2]]
        stride = [attr[3], attr[4], attr[5]]
        dilation = attr[6]
        ceil_mode = attr[7]
        padding = [attr[8], attr[9], attr[10], attr[11], attr[12], attr[13]]
        channel_last = attr[-1]
        if channel_last:
            activations = activations.permute((0, 4, 1, 2, 3))

        padded_activations = torch.nn.functional.pad(
            activations,
            padding,
            value=float("-inf"),
        )
        result = torch.nn.functional.max_pool3d(
            padded_activations,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            ceil_mode=bool(ceil_mode),
            return_indices=False,
        )
        if channel_last:
            result = result.permute((0, 2, 3, 4, 1))
    elif type == "avg_pool1d":
        assert len(attr) == 7
        kernel_size = [
            attr[0],
        ]
        stride = [
            attr[1],
        ]
        dilation = attr[2]
        ceil_mode = attr[3]
        padding = [attr[4], attr[5]]
        count_include_pad = attr[6]

        assert padding[0] == padding[1]
        padding = padding[0]

        result = torch.nn.functional.avg_pool1d(
            activations,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=bool(ceil_mode),
            count_include_pad=count_include_pad,
        )
    elif type == "avg_pool2d":
        assert len(attr) == 12
        kernel_size = [
            attr[0],
            attr[1],
        ]
        stride = [
            attr[2],
            attr[3],
        ]
        dilation = attr[4]
        ceil_mode = attr[5]
        padding = [
            attr[6],
            attr[7],
            attr[8],
            attr[9],
        ]
        count_include_pad = attr[-2]
        channel_last = attr[-1]

        assert padding[0] == padding[1] and padding[2] == padding[3]
        padding = [padding[0], padding[2]]

        if channel_last:
            activations = activations.permute(0, 3, 1, 2)

        result = torch.nn.functional.avg_pool2d(
            activations,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=bool(ceil_mode),
            count_include_pad=count_include_pad,
            divisor_override=None,
        )

        if channel_last:
            result = result.permute(0, 2, 3, 1)

    elif type == "avg_pool3d":

        kernel_size = [attr[0], attr[1], attr[2]]
        stride = [attr[3], attr[4], attr[5]]
        dilation = attr[6]
        ceil_mode = attr[7]
        padding = [attr[8], attr[9], attr[10], attr[11], attr[12], attr[13]]
        count_include_pad = attr[-2]
        channel_last = attr[-1]

        assert padding[0] == padding[1] and padding[2] == padding[3] and padding[4] == padding[5], (
            f"Padding values must be symmetric. Got: "
            f"pad_front={padding[0]}, pad_back={padding[1]}, "
            f"pad_top={padding[2]}, pad_bottom={padding[3]}, "
            f"pad_left={padding[4]}, pad_right={padding[5]}"
        )

        padding = [padding[0], padding[2], padding[4]]

        if channel_last:
            activations = activations.permute(0, 4, 1, 2, 3)

        result = torch.nn.functional.avg_pool3d(
            activations,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=bool(ceil_mode),
            count_include_pad=count_include_pad,
            divisor_override=None,
        )

        if channel_last:
            result = result.permute(0, 2, 3, 4, 1)

    return result


def shape(type, attr, ops):
    assert len(ops) == 1, "Pool ops should have one input"

    if type == "max_pool1d":
        assert len(attr) == 5

        kernel_size = attr[0]
        stride = attr[1]
        dilation = attr[2]
        ceil_mode = attr[3]
        padding = attr[4]

        activations = ops[0]

        assert dilation == 1, "Currently only support dilation = 1"

        l_out = (activations[-1] + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

        return (activations[-3], activations[-2], l_out), []
    elif type == "avg_pool1d":
        assert len(attr) == 7

        kernel_size = attr[0]
        stride = attr[1]
        dilation = attr[2]
        ceil_mode = attr[3]
        padding = attr[4]

        activations = ops[0]

        assert dilation == 1, "Currently only support dilation = 1"

        if ceil_mode:
            l_out = math.ceil((activations[-1] + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1)
        else:
            l_out = math.floor((activations[-1] + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1)

        return (activations[-3], activations[-2], l_out), []
    elif type == "max_pool2d" or type == "avg_pool2d":
        assert len(attr) == 12 or (
            type == "max_pool2d" and len(attr) == 13
        ), f"Got len(attr) = {len(attr)} for type: {type}"
        kernel_size = [
            attr[0],
            attr[1],
        ]
        stride = [
            attr[2],
            attr[3],
        ]
        dilation = attr[4]
        ceil_mode = attr[5]
        padding = [
            attr[6],
            attr[7],
            attr[8],
            attr[9],
        ]

        activations = [ops[0][dim] for dim in range(len(ops[0]))]
        channel_last = attr[-1]
        if channel_last:
            activations = [activations[ii] for ii in (0, 3, 1, 2)]

        assert dilation == 1, "Currently only support dilation = 1"

        y, x = calculate_conv2d_output_dimensions(
            activations[2], activations[3], kernel_size, stride, padding, dilation=dilation, ceil_mode=ceil_mode
        )

        if channel_last:
            result = (activations[0], y, x, activations[1])
        else:
            result = (activations[0], activations[1], y, x)

        return result, []

    elif type == "max_pool3d" or "avg_pool3d":

        assert (len(attr) == 17 and type == "max_pool3d") or (
            type == "avg_pool3d" and len(attr) == 16
        ), f"Got len(attr) = {len(attr)} for type: {type}"
        kernel_size = [attr[0], attr[1], attr[2]]
        stride = [attr[3], attr[4], attr[5]]
        dilation = attr[6]
        ceil_mode = attr[7]
        padding = [attr[8], attr[9], attr[10], attr[11], attr[12], attr[13]]
        channel_last = attr[-1]

        # activations = ops[0]
        # if channel_last:
        #    activations = activations.permute((0,4,1,2,3))
        activations = [ops[0][dim] for dim in range(len(ops[0]))]
        channel_last = attr[-1]
        if channel_last:
            activations = [activations[ii] for ii in (0, 4, 1, 2, 3)]

        assert dilation == 1, "Currently only support dilation = 1"

        z, y, x = calculate_conv3d_output_dimensions(
            activations[2],
            activations[3],
            activations[4],
            kernel_size,
            stride,
            padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
        )

        if channel_last:
            result = (activations[0], z, y, x, activations[1])
        else:
            result = (activations[0], activations[1], z, y, x)

        return result, []


def lower(type, attr, lc, ops, outputs):
    assert False, "Pooling lowering is intentionally unimplemented"


def backward(type, attr, ac, operand, inputs, output, grad):
    assert False, "Pooling backward is intentionally unimplemented"


def decompose(type, attr, dc, inputs):
    if type == "avg_pool1d":
        kernel_size = attr[0]
        stride = attr[1]
        dilation = attr[2]
        ceil_mode = attr[3]
        padding = attr[4]

        activations = inputs[0]
        if kernel_size == activations.shape[-1]:
            reduce_avg = dc.op_with_named_attrs("reduce_avg", [activations], {"dim": -1, "keep_dim": True}, (-1, True))
            dc.fuse(reduce_avg)
            return
        else:
            assert False, "Only support global avg_pool1d for now"

    elif type == "avg_pool2d":
        assert len(attr) == 12
        kernel_size = [
            attr[0],
            attr[1],
        ]
        stride = [
            attr[2],
            attr[3],
        ]
        dilation = attr[4]
        ceil_mode = attr[5]
        padding = [
            attr[6],
            attr[7],
            attr[8],
            attr[9],
        ]
        channel_last = attr[-1]
        count_include_pad = attr[-2]
        assert dilation == 1, "Currently only support dilation = 1"

        activations = inputs[0]

        if channel_last:
            w, y, x, cin = (activations.shape.w, activations.shape.z, activations.shape.r, activations.shape.c)
        else:
            w, cin, y, x = (activations.shape.w, activations.shape.z, activations.shape.r, activations.shape.c)

        # Save original padding as ceil_mode may edit it
        original_padding = list(padding)

        # If ceil_mode = True, ceil function will be used to calculate output shape instead of the floor function, as
        # defined in pytorch docs:
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        if ceil_mode:
            ceil_pad_right, ceil_pad_bottom = calculate_pad_for_ceil_mode(
                original_y=y, original_x=x, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation
            )

            # If both ceil pads are 0, that is equivalent to ceil_mode=False
            if ceil_pad_right == 0 and ceil_pad_bottom == 0:
                ceil_mode = False
            else:
                padding[1] += ceil_pad_right
                padding[3] += ceil_pad_bottom

        kH, kW = kernel_size
        # If global average
        if y == kH and x == kW and ((stride[0] == kH and stride[1] == kW) or all(pad == 0 for pad in padding)):
            if channel_last:
                result = dc.op_with_named_attrs(
                    "reshape", [activations], {"shape": (w, 1, y * x, cin)}, (w, 1, y * x, cin)
                )
                result = dc.op_with_named_attrs("reduce_avg", [result], {"dim": -2, "keep_dim": True}, (-2, True))
                result = dc.op_with_named_attrs("reshape", [result], {"shape": (w, 1, 1, cin)}, (w, 1, 1, cin))
            else:
                result = dc.op_with_named_attrs(
                    "reshape", [activations], {"shape": (w, 1, cin, y * x)}, (w, 1, cin, y * x)
                )
                result = dc.op(TransposeTM.create(2, 3), [result])
                result = dc.op_with_named_attrs("reduce_avg", [result], {"dim": -2, "keep_dim": True}, (-2, True))
                result = dc.op(TransposeTM.create(2, 3), [result])
                result = dc.op_with_named_attrs("reshape", [result], {"shape": (w, cin, 1, 1)}, (w, cin, 1, 1))
            dc.fuse(result)
            return

        weight_value = 1.0 / (kH * kW)
        weight_tensor = weight_value * torch.ones((cin, 1, kH, kW))

        weight = dc.tensor(weight_tensor)
        result = dc.op_with_named_attrs(
            Conv2d.create(
                stride_height=stride[0],
                stride_width=stride[1],
                dilation_height=dilation,
                dilation_width=dilation,
                groups=cin,
                padding_left=padding[0],
                padding_right=padding[1],
                padding_top=padding[2],
                padding_bottom=padding[3],
                channel_last=channel_last,
            ),
            [activations, weight],
            {
                "stride_height": stride[0],
                "stride_width": stride[1],
                "dilation_height": dilation,
                "dilation_width": dilation,
                "groups": cin,
                "padding_left": padding[0],
                "padding_right": padding[1],
                "padding_top": padding[2],
                "padding_bottom": padding[3],
                "channel_last": channel_last,
            },
        )

        #
        # Undo math in padded regions
        #
        # Both ceil_mode=True and count_include_pad=False call for undoing math in padded regions
        # count_include_pad=False calls for excluding math in all padded regions
        #
        # ceil_mode=True calls for excluding math only in regions padded by ceil_mode:
        # https://discuss.pytorch.org/t/ceil-mode-in-avg-pool2d-seems-to-output-wrong-result/189323
        #
        # TODO: the sparse matmul below can be fused into the first sparse matmul of avgpool's decomposition graph
        # As a consequence, the in0 of the fused sparse matmul will probably have to be a higher-bit format than
        # Bfp2 since it won't just be 0s and 1s in the picker matrix.
        # For simplicity, it is initally implemented as an additional op.
        if not padding == [0, 0, 0, 0] and (ceil_mode == True or count_include_pad == False):
            if channel_last:
                _, y_out, x_out, _ = (result.shape.w, result.shape.z, result.shape.r, result.shape.c)
                result = dc.op_with_named_attrs(
                    "reshape", [result], {"shape": (w, 1, y_out * x_out, cin)}, (w, 1, y_out * x_out, cin)
                )
            else:
                _, _, y_out, x_out = (result.shape.w, result.shape.z, result.shape.r, result.shape.c)
                result = dc.op_with_named_attrs(
                    "reshape", [result], {"shape": (w, 1, cin, y_out * x_out)}, (w, 1, cin, y_out * x_out)
                )
                result = dc.op(TransposeTM.create(2, 3), [result])

            # Since count_include_pad=False undoes math in all padded regions, it takes precedence:
            #
            # if count_include_pad == False:
            #     undo_math_all_padding()
            # elif ceil_mode == True:
            #     undo_math_in_ceil_padded_areas()
            # else:
            #     nop
            undo_math_picker = create_avg_pool2d_count_include_pad_False_picker_matrix(
                y=y + (0 if count_include_pad == False else (original_padding[2] + original_padding[3])),
                x=x + (0 if count_include_pad == False else (original_padding[0] + original_padding[1])),
                k_y=kernel_size[0],
                k_x=kernel_size[1],
                stride=stride,
                padding=padding if count_include_pad == False else [0, ceil_pad_right, 0, ceil_pad_bottom],
                tile_align=False,
            )
            undo_math_picker_tensor = dc.tensor(undo_math_picker)
            result = dc.op("matmul", [undo_math_picker_tensor, result])

            if channel_last:
                result = dc.op_with_named_attrs(
                    "reshape", [result], {"shape": (w, y_out, x_out, cin)}, (w, y_out, x_out, cin)
                )
            else:
                result = dc.op(TransposeTM.create(2, 3), [result])
                result = dc.op_with_named_attrs(
                    "reshape", [result], {"shape": (w, cin, y_out, x_out)}, (w, cin, y_out, x_out)
                )

        dc.fuse(result)

    elif type == "avg_pool3d":
        # Slice the input tensor along the depth dimension.
        #     - Input shape: (B, C, D_in, H_in, W_in)
        #     - After depth slicing: (B, C, kD, H_in, W_in)
        # Then, the depth dimension is averaged across, reducing it to a single depth slice.
        #     - After averaging along depth: (B, C, 1, H_in, W_in)
        # A 2D pooling operation is applied along the spatial dimensions (H_in, W_in).
        #     - After 2D pooling: (B, C, H_out, W_out)
        # To match the final output shape, we add an extra singleton dimension to depth.
        #     - After unsqueeze: (B, C, 1, H_out, W_out)
        # Finally, the outputs from the depth slices are concatenated.
        #     - Final shape after all concatenations: (B, C, D_out, H_out, W_out)

        kernel_size = [attr[0], attr[1], attr[2]]
        stride = [attr[3], attr[4], attr[5]]
        dilation = attr[6]
        ceil_mode = attr[7]
        padding = [attr[8], attr[9], attr[10], attr[11], attr[12], attr[13]]
        count_include_pad = attr[-2]
        channel_last = attr[-1]

        activations = inputs[0]

        w, cin, din, y, x = (
            activations.shape.v,
            activations.shape.w,
            activations.shape.z,
            activations.shape.r,
            activations.shape.c,
        )

        kD, kH, kW = kernel_size
        sD, sH, sW = stride

        pad_d1, pad_h1, pad_w1, pad_d2, pad_h2, pad_w2 = padding

        out_depth = (din - kD + pad_d1 + pad_d2) // sD + 1
        out_height = (y - kH + pad_h1 + pad_h2) // sH + 1
        out_width = (x - kW + pad_w1 + pad_w2) // sW + 1

        result = dc.tensor(torch.zeros((activations.shape[0], cin, 0, out_height, out_width)))

        for i in range(out_depth):

            d_start = i * sD

            depth_slice = dc.op("index", [activations], (2, d_start, d_start + kD, activations.shape[2]))
            depth_avg = dc.op_with_named_attrs("reduce_avg", [depth_slice], {"dim": 2, "keep_dim": True}, (2, True))

            named_attrs = {
                "kernel_height": kernel_size[1],
                "kernel_width": kernel_size[2],
                "stride_height": stride[1],
                "stride_width": stride[2],
                "dilation": dilation,
                "ceil_mode": ceil_mode,
                "padding_left": padding[1],
                "padding_right": padding[2],
                "padding_top": padding[4],
                "padding_bottom": padding[5],
                "count_include_pad": count_include_pad,
                "channel_last": channel_last,
            }

            attr = (
                kernel_size[1],
                kernel_size[2],
                stride[1],
                stride[2],
                dilation,
                ceil_mode,
                padding[1],
                padding[2],
                padding[4],
                padding[5],
                count_include_pad,
                channel_last,
            )

            depth_avg_pooled = dc.op_with_named_attrs("avg_pool2d", [depth_avg], named_attrs, attr)

            depth_avg_pooled = dc.op_with_named_attrs(
                "unsqueeze", [depth_avg_pooled], {"dim": 2}, (0, len(depth_avg_pooled.shape))
            )
            result = dc.op_with_named_attrs("concatenate", [result, depth_avg_pooled], {"dim": (2)}, (2,))

        dc.fuse(result)


def initial_flops_estimate(type, attr, ops):
    # TODO: Add global pool
    flops = 0
    if type == "avg_pool2d" or type == "max_pool2d":
        output_shape = shape(type, attr, ops)[0]
        flops = output_shape[-1] * output_shape[-2] * attr[0] * attr[1]
        if len(output_shape) > 2:
            flops *= output_shape[-3]
        if len(output_shape) > 3:
            flops *= output_shape[-4]

    return flops
