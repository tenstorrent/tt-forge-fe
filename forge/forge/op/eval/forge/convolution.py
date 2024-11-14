# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import ast
import torch
import math

from forge._C.graph import NodeType
from forge.forgeglobal import TILE_DIM
from forge.utils import align_up_tile, round_up_div, clamp
from forge import Tensor
from forge.config import _get_global_compiler_config
from .transpose import TransposeTM


import torch
from ..interface import PyOp, PyTM
from ..common import to_torch_operands


class Conv2d(PyOp):
    @classmethod
    def create(
        cls,
        stride_height,
        stride_width,
        dilation_height,
        dilation_width,
        groups,
        padding_left,
        padding_right,
        padding_top,
        padding_bottom,
        channel_last,
    ):
        self = cls("conv2d")
        self.stride_height = stride_height
        self.stride_width = stride_width
        self.dilation_height = dilation_height
        self.dilation_width = dilation_width
        self.groups = groups
        self.padding_left = padding_left
        self.padding_right = padding_right
        self.padding_top = padding_top
        self.padding_bottom = padding_bottom
        self.channel_last = int(channel_last)
        return self

    def eval(self, tensors):
        assert len(tensors) <= 3, "Conv ops should have up to three inputs (input, weight, bias)"
        assert len(tensors) >= 2, "Conv ops should have at least two inputs (input, weight)"
        t_ops = to_torch_operands(*tensors)

        activations = t_ops[0]
        weights = t_ops[1]
        bias = t_ops[2] if len(t_ops) == 3 else None

        stride = [self.stride_height, self.stride_width]
        dilation = [self.dilation_height, self.dilation_width]
        groups = self.groups
        padding = [
            self.padding_left,
            self.padding_right,
            self.padding_top,
            self.padding_bottom,
        ]

        channel_last = self.channel_last
        if channel_last:
            activations = activations.permute((0, 3, 1, 2))

        padded_activations = torch.nn.functional.pad(
            activations,
            padding,
        )
        if t_ops[1].dtype == torch.int8:
            target_dtype = torch.int32
            padded_activations, weights = padded_activations.float(), weights.float()
            if bias is not None:
                bias = bias.float()
        else:
            target_dtype = torch.float32

        result = torch.nn.functional.conv2d(
            padded_activations,
            weights,
            bias=bias,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
        )

        if channel_last:
            result = result.permute((0, 2, 3, 1))

        result = result.to(target_dtype)
        return result

    def shape(self, tensor_shapes):
        act, weight = tensor_shapes[:2]
        batch_size = act[0]
        cout = weight[0]

        h_in = act[-3] if self.channel_last else act[-2]
        w_in = act[-2] if self.channel_last else act[-1]

        h_numerator = h_in + (self.padding_top + self.padding_bottom) - self.dilation_height * (weight[-2] - 1) - 1
        h_out = math.floor(1 + (h_numerator / self.stride_height))

        w_numerator = w_in + (self.padding_left + self.padding_right) - self.dilation_width * (weight[-1] - 1) - 1
        w_out = math.floor(1 + (w_numerator / self.stride_width))

        out_shape = [batch_size, h_out, w_out, cout] if self.channel_last else [batch_size, cout, h_out, w_out]

        return out_shape, []

    def decompose(self, dc, inputs):
        # TTNN can only perform a channel last convolution with its conv2d op.
        # The TTNN conv2d requires the input to be in the shape: (N, H, W, C) or (1, 1, N*H*W, C).
        # It requires the weight to be in the shape: (C_out, C_in, kernel_height, kernel_width).
        # It requires the bias to be in the shape: (1, 1, 1, C_out).
        #
        # If the forge conv2d op is channel-first, we must permute the input (N, C, H, W) tensor to (N, H, W, C)
        # and then transpose it back to (N, C_out, H_out, W_out) afterward.
        #     - This is done with two transposes
        #     - (N, C, H, W) --> transpose(-3, -2): (N, H, C, W) --> transpose(-2, -1): (N, H, W, C)
        # Afterward:
        #     - (N, H_out, W_out, C_out) --> transpose(-2, -1): (N, H_out, C_out, W_out) --> transpose(-3, -2): (N, C_out, H_out, W_out)
        activations = inputs[0]
        weight = inputs[1]
        bias = inputs[2] if len(inputs) == 3 else None

        is_channel_last = self.channel_last

        if bias is not None and len(bias.shape) < len(activations.shape):
            while len(bias.shape) < len(activations.shape):
                bias = dc.op("unsqueeze", [bias], (0, len(bias.shape)))

        is_bias_unchanged = bias is None or bias == inputs[2]

        if not is_channel_last:
            activations = dc.op(TransposeTM.create(dim0=-3, dim1=-2), [activations])
            activations = dc.op(TransposeTM.create(dim0=-2, dim1=-1), [activations])

        # Only want to re-create the Conv2d op if something has changed. Otherwise it the compiler will infinitely
        # decompose the same Conv2d over and over.
        if not is_bias_unchanged or not is_channel_last:
            new_inputs = [activations, weight] if bias is None else [activations, weight, bias]
            result = dc.op(
                Conv2d.create(
                    self.stride_height,
                    self.stride_width,
                    self.dilation_height,
                    self.dilation_width,
                    self.groups,
                    self.padding_left,
                    self.padding_right,
                    self.padding_top,
                    self.padding_bottom,
                    True,  # If the original Conv2d was channel-last, that will not change.
                    # If it was channel-first, it the input will have been permuted by this point.
                    # So, the Conv2d op being created here is certainly channel-last.
                ),
                new_inputs,
            )

            if not is_channel_last:
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


class Conv2dTranspose(PyOp):
    @classmethod
    def create(
        cls,
        stride_height,
        stride_width,
        dilation_height,
        dilation_width,
        groups,
        padding_left,
        padding_right,
        padding_top,
        padding_bottom,
        channel_last,
    ):
        self = cls("conv2d_transpose")
        self.stride_height = stride_height
        self.stride_width = stride_width
        self.dilation_height = dilation_height
        self.dilation_width = dilation_width
        self.groups = groups
        self.padding_left = padding_left
        self.padding_right = padding_right
        self.padding_top = padding_top
        self.padding_bottom = padding_bottom
        self.channel_last = int(channel_last)
        return self

    def eval(self, tensors):
        assert len(tensors) <= 3, "ConvTranspose ops should have up to three inputs (input, weight, bias)"
        assert len(tensors) >= 2, "ConvTranspose ops should have at least two inputs (input, weight)"
        t_ops = to_torch_operands(*tensors)
        activations = t_ops[0]
        weights = t_ops[1]
        bias = t_ops[2] if len(t_ops) == 3 else None

        stride = [self.stride_height, self.stride_width]
        dilation = [self.dilation_height, self.dilation_width]
        groups = self.groups
        # TODO: Add support for asymmetric padding cases in convtranspose2d
        if self.padding_left == self.padding_right and self.padding_top == self.padding_bottom:
            padding = (self.padding_top, self.padding_left)
        else:
            assert (
                False
            ), "Currently, different left and right padding or different top and bottom padding isn't supported. Please check out this issue (https://github.com/tenstorrent/tt-forge-fe/issues/665) for more details."

        channel_last = self.channel_last
        if channel_last:
            activations = activations.permute((0, 3, 1, 2))

        if t_ops[1].dtype == torch.int8:
            target_dtype = torch.int32
            activations, weights = activations.float(), weights.float()
            if bias is not None:
                bias = bias.float()
        else:
            target_dtype = torch.float32

        result = torch.nn.functional.conv_transpose2d(
            activations,
            weights,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

        if channel_last:
            result = result.permute((0, 2, 3, 1))
        result = result.to(target_dtype)
        return result

    def shape(self, tensor_shapes):
        act, weight = tensor_shapes[:2]
        batch_size = act[0]
        cout = weight[1] * self.groups

        h_in = act[-3] if self.channel_last else act[-2]
        w_in = act[-2] if self.channel_last else act[-1]

        output_padding_height = 0
        output_padding_width = 0

        h_out = (
            (h_in - 1) * self.stride_height
            - (self.padding_top + self.padding_bottom)
            + self.dilation_height * (weight[-2] - 1)
            + output_padding_height
            + 1
        )
        w_out = (
            (w_in - 1) * self.stride_width
            - (self.padding_left + self.padding_right)
            + self.dilation_width * (weight[-1] - 1)
            + output_padding_width
            + 1
        )
        out_shape = [batch_size, h_out, w_out, cout] if self.channel_last else [batch_size, cout, h_out, w_out]
        return out_shape, []

    def decompose(self, dc, inputs):
        # TTNN can only perform a channel last convolution with its conv_transpose2d op.
        # The TTNN conv_transpose2d requires the input to be in the shape: (N, H, W, C) or (1, 1, N*H*W, C).
        # It requires the weight to be in the shape: (C_out, C_in, kernel_height, kernel_width).
        # It requires the bias to be in the shape: (1, 1, 1, C_out).
        #
        # If the forge conv_transpose2d op is channel-first, we must permute the input (N, C, H, W) tensor to (N, H, W, C)
        # and then transpose it back to (N, C_out, H_out, W_out) afterward.
        #     - This is done with two transposes
        #     - (N, C, H, W) --> transpose(-3, -2): (N, H, C, W) --> transpose(-2, -1): (N, H, W, C)
        # Afterward:
        #     - (N, H_out, W_out, C_out) --> transpose(-2, -1): (N, H_out, C_out, W_out) --> transpose(-3, -2): (N, C_out, H_out, W_out)
        activations = inputs[0]
        weight = inputs[1]
        bias = inputs[2] if len(inputs) == 3 else None

        is_channel_last = self.channel_last

        if bias is not None and len(bias.shape) < len(activations.shape):
            while len(bias.shape) < len(activations.shape):
                bias = dc.op("unsqueeze", [bias], (0, len(bias.shape)))

        is_bias_unchanged = bias is None or bias == inputs[2]

        if not is_channel_last:
            activations = dc.op(TransposeTM.create(dim0=-3, dim1=-2), [activations])
            activations = dc.op(TransposeTM.create(dim0=-2, dim1=-1), [activations])

        # Only want to re-create the Conv2dTranspose op if something has changed. Otherwise it the compiler will infinitely
        # decompose the same Conv2dTranspose over and over.
        if not is_bias_unchanged or not is_channel_last:
            new_inputs = [activations, weight] if bias is None else [activations, weight, bias]
            result = dc.op(
                Conv2dTranspose.create(
                    self.stride_height,
                    self.stride_width,
                    self.dilation_height,
                    self.dilation_width,
                    self.groups,
                    self.padding_left,
                    self.padding_right,
                    self.padding_top,
                    self.padding_bottom,
                    True,  # If the original Conv2dTranspose was channel-last, that will not change.
                    # If it was channel-first, it the input will have been permuted by this point.
                    # So, the Conv2dTranspose op being created here is certainly channel-last.
                ),
                new_inputs,
            )

            if not is_channel_last:
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


class Conv3d(PyOp):
    @classmethod
    def create(
        cls,
        stride_depth,
        stride_height,
        stride_width,
        dilation_depth,
        dilation_height,
        dilation_width,
        groups,
        padding_front,
        padding_back,
        padding_left,
        padding_right,
        padding_top,
        padding_bottom,
        channel_last,
    ):
        self = cls("conv3d")
        self.stride_depth = stride_depth
        self.stride_height = stride_height
        self.stride_width = stride_width
        self.dilation_depth = dilation_depth
        self.dilation_height = dilation_height
        self.dilation_width = dilation_width
        self.groups = groups
        self.padding_front = padding_front
        self.padding_back = padding_back
        self.padding_left = padding_left
        self.padding_right = padding_right
        self.padding_top = padding_top
        self.padding_bottom = padding_bottom
        self.channel_last = int(channel_last)
        return self

    def eval(self, tensors):
        assert len(tensors) <= 3, "Conv ops should have up to three inputs (input, weight, bias)"
        assert len(tensors) >= 2, "Conv ops should have at least two inputs (input, weight)"
        t_ops = to_torch_operands(*tensors)

        activations = t_ops[0]
        weights = t_ops[1]
        bias = t_ops[2] if len(t_ops) == 3 else None

        stride = [self.stride_depth, self.stride_height, self.stride_width]
        dilation = [self.dilation_depth, self.dilation_height, self.dilation_width]
        groups = self.groups
        padding = [
            self.padding_front,
            self.padding_back,
            self.padding_top,
            self.padding_bottom,
            self.padding_left,
            self.padding_right,
        ]

        if self.channel_last:
            activations = activations.permute((0, 4, 1, 2, 3))

        padded_activations = torch.nn.functional.pad(
            activations,
            padding,
        )

        result = torch.nn.functional.conv3d(
            padded_activations,
            weights,
            bias=bias,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
        )

        if self.channel_last:
            result = result.permute((0, 2, 3, 4, 1))

        result = result.to(activations.dtype)
        return result

    def shape(self, tensor_shapes):

        act, weight = tensor_shapes[:2]
        batch_size = act[0]
        cout = weight[0]

        d_in = act[-4] if self.channel_last else act[-3]
        h_in = act[-3] if self.channel_last else act[-2]
        w_in = act[-2] if self.channel_last else act[-1]

        d_numerator = d_in + (self.padding_front + self.padding_back) - self.dilation_depth * (weight[-3] - 1) - 1
        d_out = math.floor(1 + (d_numerator / self.stride_depth))

        h_numerator = h_in + (self.padding_top + self.padding_bottom) - self.dilation_height * (weight[-2] - 1) - 1
        h_out = math.floor(1 + (h_numerator / self.stride_height))

        w_numerator = w_in + (self.padding_left + self.padding_right) - self.dilation_width * (weight[-1] - 1) - 1
        w_out = math.floor(1 + (w_numerator / self.stride_width))

        out_shape = (
            [batch_size, d_out, h_out, w_out, cout] if self.channel_last else [batch_size, cout, d_out, h_out, w_out]
        )

        return out_shape, []

    def decompose(self, dc, inputs):
        # conv3d op is not yet supported in TTNN, based on refrence from conv2d following transformations are done
        # TTNN can only perform a channel-last convolution with its conv3d op.
        # The TTNN conv3d requires the input to be in the shape: (N, D, H, W, C) or (1, 1, N*D*H*W, C).
        # It requires the weight to be in the shape: (C_out, C_in, kernel_depth, kernel_height, kernel_width).
        # It requires the bias to be in the shape: (1, 1, 1, 1, C_out).
        #
        # If the forge conv3d op is channel-first, we must permute the input (N, C, D, H, W) tensor to (N, D, H, W, C)
        # and then transpose it back to (N, C_out, D_out, H_out, W_out) afterward.
        #     - This is done with three transposes
        #     - (N, C, D, H, W) --> transpose(-4, -3): (N, D, C, H, W) --> transpose(-3, -2): (N, D, H, C, W)
        #     --> transpose(-2, -1): (N, D, H, W, C)
        # Afterward:
        #     - (N, D_out, H_out, W_out, C_out) --> transpose(-3, -2): (N, D_out, H_out, C_out, W_out)
        #     --> transpose(-4, -3): (N, C_out, D_out, H_out, W_out)

        activations = inputs[0]
        weight = inputs[1]
        bias = inputs[2] if len(inputs) == 3 else None

        is_channel_last = self.channel_last

        if bias is not None and len(bias.shape) < len(activations.shape):
            while len(bias.shape) < len(activations.shape):
                bias = dc.op("unsqueeze", [bias], (0, len(bias.shape)))
        is_bias_unchanged = bias is None or bias == inputs[2]

        if not is_channel_last:
            activations = dc.op(TransposeTM.create(dim0=-4, dim1=-3), [activations])
            activations = dc.op(TransposeTM.create(dim0=-3, dim1=-2), [activations])
            activations = dc.op(TransposeTM.create(dim0=-2, dim1=-1), [activations])

        # Only want to re-create the Conv3d op if something has changed. Otherwise it the compiler will infinitely
        # decompose the same Conv3d over and over.
        if not is_bias_unchanged or not is_channel_last:

            new_inputs = [activations, weight] if bias is None else [activations, weight, bias]
            result = dc.op(
                Conv3d.create(
                    self.stride_depth,
                    self.stride_height,
                    self.stride_width,
                    self.dilation_depth,
                    self.dilation_height,
                    self.dilation_width,
                    self.groups,
                    self.padding_front,
                    self.padding_back,
                    self.padding_left,
                    self.padding_right,
                    self.padding_top,
                    self.padding_bottom,
                    True,
                ),
                new_inputs,
            )

            if not is_channel_last:
                result = dc.op(TransposeTM.create(dim0=-1, dim1=-2), [result])
                result = dc.op(TransposeTM.create(dim0=-2, dim1=-3), [result])
                result = dc.op(TransposeTM.create(dim0=-3, dim1=-4), [result])
            dc.fuse(result)

    def backward(self, ac, operand, inputs, output, grad):
        raise NotImplementedError("Backward operation is not yet supported for this op")

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
