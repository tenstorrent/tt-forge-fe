# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 2
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
    Forge Test 2

    """

    INPUTS_RANGE_MIN = -1.0
    INPUTS_RANGE_MAX = 1.0
    INPUTS_DISTRIBUTION = Normal

    WEIGHTS_RANGE_MIN = -1.0
    WEIGHTS_RANGE_MAX = 1.0
    WEIGHTS_DISTRIBUTION = Normal

    def __init__(self, shape, min_value, max_value):
        super().__init__("Forge Test 2")

        self.testname = "Operator Clip, Test 2"
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
        mul1 = forge.op.Multiply("mul1", x1, self.train_param1)
        mul2 = forge.op.Multiply("mul2", x2, self.train_param1)
        mul3 = forge.op.Multiply("mul3", x2, self.train_param2)

        # Layer 3
        clip1 = forge.op.Clip("clip1", mul1, min=self.min_value, max=self.max_value)
        clip2 = forge.op.Clip("clip2", mul2, min=self.min_value, max=self.max_value)
        clip3 = forge.op.Clip("clip3", mul3, min=self.min_value, max=self.max_value)

        # Layer 4
        mul4 = forge.op.Multiply("mul4", clip1, clip2)
        mul5 = forge.op.Multiply("mul5", clip2, clip3)

        # Layer 5
        clip4 = forge.op.Clip("clip4", mul4, min=self.min_value, max=self.max_value)
        clip5 = forge.op.Clip("clip5", mul5, min=self.min_value, max=self.max_value)

        # Layer 6
        mul6 = forge.op.Multiply("mul6", clip4, clip5)

        return mul6
