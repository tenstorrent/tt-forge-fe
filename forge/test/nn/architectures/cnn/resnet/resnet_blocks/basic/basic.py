# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Test for Resnet Basic Block
#
# from turtle import forward

import forge
import forge.op

from forge import ForgeModule
from forge.op.nn import Conv2dModule



class TestBasicBlock(ForgeModule):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        dilation,
        depthwise,
        bias,
    ):
        super().__init__("ResNet Basic Block")

        assert in_channels == out_channels, "Number of input and output channels must be the same"

        # Two convolutional layers with the same arguments
        self.conv1, self.conv2 = [
            Conv2dModule(
                f"conv2d_{i}",
                in_channels,
                out_channels if not depthwise else in_channels,
                kernel_size,
                stride=stride,
                padding=(kernel_size // 2),  # only padding we support today
                dilation=dilation,
                groups=in_channels if depthwise else 1,
                bias=bias,
            )
            for i in range(1, 3)
        ]

    def forward(self, activations):

        # Left side of the block, two convolutional layers with relu
        conv1 = self.conv1(activations)
        relu1 = forge.op.Relu("relu1", conv1)
        conv2 = self.conv2(relu1)

        # Right side of the block, just identity tensor, activations 
        # added to convolution from the left side
        add1 = forge.op.Add("out", conv2, activations)

        # Sum of outputs from the left and right sides with applied relu
        relu2 = forge.op.Relu("relu2", add1)

        return relu2