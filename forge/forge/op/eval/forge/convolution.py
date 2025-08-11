# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch

from forge._C.graph import NodeType
from forge.forgeglobal import TILE_DIM
from forge.utils import align_up_tile, round_up_div, clamp
from forge import Tensor


import torch
from ..interface import PyOp, PyTM
from ..common import to_torch_operands


class Conv2dTranspose(PyOp):
    @classmethod
    def create(
        cls,
        stride,  # Input format: [sH, sW]
        dilation,  # Input format: [dH, dW]
        groups,
        padding,  # Input format: [pT, pL, pB, pR]
        output_padding,  # Input format: [opH, opW]
        channel_last,
    ):
        self = cls("conv2d_transpose")
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding = padding
        self.output_padding = output_padding
        self.channel_last = int(channel_last)
        return self

    def eval(self, tensors):
        assert len(tensors) <= 3, "ConvTranspose ops should have up to three inputs (input, weight, bias)"
        assert len(tensors) >= 2, "ConvTranspose ops should have at least two inputs (input, weight)"
        t_ops = to_torch_operands(*tensors)
        activations = t_ops[0]
        weights = t_ops[1]
        bias = t_ops[2] if len(t_ops) == 3 else None

        stride = self.stride
        dilation = self.dilation
        groups = self.groups
        padding = (
            self.padding[0],
            self.padding[1],
        )

        channel_last = self.channel_last
        if channel_last:
            activations = activations.permute((0, 3, 1, 2))

        if t_ops[1].dtype == torch.int8:
            activations, weights = activations.float(), weights.float()
            if bias is not None:
                bias = bias.float()
            cast_result_to_int32 = True
        else:
            cast_result_to_int32 = False

        result = torch.nn.functional.conv_transpose2d(
            activations,
            weights,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            output_padding=self.output_padding,
        )

        if channel_last:
            result = result.permute((0, 2, 3, 1))
        if cast_result_to_int32:
            result = result.to(torch.int32)
        return result

    def shape(self, tensor_shapes):
        stride_height = self.stride[0]
        stride_width = self.stride[1]
        dilation_height = self.dilation[0]
        dilation_width = self.dilation[1]
        padding_top = self.padding[0]
        padding_left = self.padding[1]
        padding_bottom = self.padding[2]
        padding_right = self.padding[3]

        act, weight = tensor_shapes[:2]
        batch_size = act[0]
        cout = weight[1] * self.groups

        h_in = act[-3] if self.channel_last else act[-2]
        w_in = act[-2] if self.channel_last else act[-1]

        output_padding = self.output_padding
        output_padding_height, output_padding_width = output_padding if output_padding else (0, 0)

        h_out = (
            (h_in - 1) * stride_height
            - (padding_top + padding_bottom)
            + dilation_height * (weight[-2] - 1)
            + output_padding_height
            + 1
        )
        w_out = (
            (w_in - 1) * stride_width
            - (padding_left + padding_right)
            + dilation_width * (weight[-1] - 1)
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
                bias = dc.op_with_named_attrs("unsqueeze", [bias], {"dim": 0}, (0,))

        is_bias_unchanged = bias is None or bias == inputs[2]

        if not is_channel_last:
            activations = dc.op_with_named_attrs("transpose", [activations], {"dim0": -3, "dim1": -2})
            activations = dc.op_with_named_attrs("transpose", [activations], {"dim0": -2, "dim1": -1})

        # The below condition checks whether the channel is last and the required output channel position (1, 1, 1, C_out) in the bias, as well as the bias type.
        if (bias is not None) and (bias.shape[-1] != weight.shape[-4]) and (not is_channel_last):
            bias = dc.op_with_named_attrs("transpose", [bias], {"dim0": -3, "dim1": -2})
            bias = dc.op_with_named_attrs("transpose", [bias], {"dim0": -2, "dim1": -1})

        # Only want to re-create the Conv2dTranspose op if something has changed. Otherwise it the compiler will infinitely
        # decompose the same Conv2dTranspose over and over.
        if not is_bias_unchanged or not is_channel_last:
            new_inputs = [activations, weight] if bias is None else [activations, weight, bias]
            result = dc.op(
                Conv2dTranspose.create(
                    stride=self.stride,
                    dilation=self.dilation,
                    groups=self.groups,
                    padding=self.padding,
                    output_padding=self.output_padding,
                    channel_last=True,  # If the original Conv2dTranspose was channel-last, that will not change.
                    # If it was channel-first, it the input will have been permuted by this point.
                    # So, the Conv2dTranspose op being created here is certainly channel-last.
                ),
                new_inputs,
            )

            if not is_channel_last:
                result = dc.op_with_named_attrs("transpose", [result], {"dim0": -2, "dim1": -1})
                result = dc.op_with_named_attrs("transpose", [result], {"dim0": -3, "dim1": -2})

            dc.fuse(result)

    def backward(self, ac, operand, inputs, output, grad):
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
