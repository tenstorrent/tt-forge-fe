# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 3
#   Clip operators defined by Forge API
#   These kinds of tests test only single specific operator through different Forge architectures
#


import torch
from torch.distributions import Normal

import forge
import forge.op
import forge.op.nn as nn
from forge import ForgeModule, Tensor


class ForgeClipTest(ForgeModule):
    """
    Forge Test 3

    """

    INPUTS_RANGE_MIN = -1.0
    INPUTS_RANGE_MAX = 1.0
    INPUTS_DISTRIBUTION = Normal

    WEIGHTS_RANGE_MIN = -1.0
    WEIGHTS_RANGE_MAX = 1.0
    WEIGHTS_DISTRIBUTION = Normal

    def __init__(self, shape, min_value, max_value):
        super().__init__("Forge Test 3")

        self.testname = "Operator Clip, Test 3"
        self.shape = shape
        self.min_value = min_value
        self.max_value = max_value

        self.train_param1 = forge.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = forge.Parameter(*self.shape, requires_grad=True)

        self.inputs = []
        for i in range(2):
            input = ForgeClipTest.INPUTS_DISTRIBUTION(
                ForgeClipTest.INPUTS_RANGE_MIN, ForgeClipTest.INPUTS_RANGE_MAX
            ).sample(self.shape)
            self.inputs.append(Tensor.create_from_torch(input))

        for i in range(1, 3):
            weights = ForgeClipTest.WEIGHTS_DISTRIBUTION(
                ForgeClipTest.WEIGHTS_RANGE_MIN, ForgeClipTest.WEIGHTS_RANGE_MAX
            ).sample(self.shape)
            weights.requires_grad = True
            self.set_parameter("train_param" + str(i), weights)

    def forward(self, x1, x2):

        # Layer 2
        clip1 = forge.op.Clip("clip1", x1, min=self.min_value, max=self.max_value)
        clip2 = forge.op.Clip("clip2", self.train_param1, min=self.min_value, max=self.max_value)
        clip3 = forge.op.Clip("clip3", x2, min=self.min_value, max=self.max_value)
        clip4 = forge.op.Clip("clip4", self.train_param2, min=self.min_value, max=self.max_value)

        # Layer 3
        mul1 = forge.op.Multiply("mul1", clip1, clip2)
        mul2 = forge.op.Multiply("mul2", clip2, clip3)
        mul3 = forge.op.Multiply("mul3", clip3, clip4)

        # Layer 4
        clip5 = forge.op.Clip("clip5", mul1, min=self.min_value, max=self.max_value)
        clip6 = forge.op.Clip("clip6", mul2, min=self.min_value, max=self.max_value)
        clip7 = forge.op.Clip("clip7", mul3, min=self.min_value, max=self.max_value)

        # Layer 5
        mul4 = forge.op.Multiply("mul4", clip5, self.train_param1)
        mul5 = forge.op.Multiply("mul5", clip6, x2)
        mul6 = forge.op.Multiply("mul6", clip7, clip4)

        # Layer 6
        clip8 = forge.op.Clip("clip8", mul4, min=self.min_value, max=self.max_value)
        clip9 = forge.op.Clip("clip9", mul5, min=self.min_value, max=self.max_value)
        clip10 = forge.op.Clip("clip10", mul6, min=self.min_value, max=self.max_value)

        # Layer 7
        add1 = forge.op.Add("add1", clip8, mul2)
        add2 = forge.op.Add("add2", clip4, clip10)
        mul7 = forge.op.Multiply("mul7", clip9, mul3)

        # Layer 8
        clip11 = forge.op.Clip("clip11", add1, min=self.min_value, max=self.max_value)
        clip12 = forge.op.Clip("clip12", mul7, min=self.min_value, max=self.max_value)
        clip13 = forge.op.Clip("clip13", add2, min=self.min_value, max=self.max_value)

        return clip11, clip12, clip13
