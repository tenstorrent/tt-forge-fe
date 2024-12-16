# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 4
#   LeakyRelu operators defined by Forge API
#   These kinds of tests test only single specific operator through different Forge architectures
#


import torch
from torch.distributions import Normal

import forge
import forge.op
import forge.op.nn as nn
from forge import ForgeModule, Tensor


class ForgeLeakyReluTest(ForgeModule):
    """
    Forge Test 4

    """

    INPUTS_RANGE_MIN = -1.0
    INPUTS_RANGE_MAX = 1.0
    INPUTS_DISTRIBUTION = Normal

    WEIGHTS_RANGE_MIN = -1.0
    WEIGHTS_RANGE_MAX = 1.0
    WEIGHTS_DISTRIBUTION = Normal

    def __init__(self, shape, alpha):
        super().__init__("Forge Test 4")

        self.testname = "Operator LeakyRelu, Test 4"
        self.shape = shape
        self.alpha = alpha

        self.train_param1 = forge.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = forge.Parameter(*self.shape, requires_grad=True)
        self.train_param3 = forge.Parameter(*self.shape, requires_grad=True)

        self.inputs = []
        for i in range(3):
            input = ForgeLeakyReluTest.INPUTS_DISTRIBUTION(
                ForgeLeakyReluTest.INPUTS_RANGE_MIN, ForgeLeakyReluTest.INPUTS_RANGE_MAX
            ).sample(self.shape)
            self.inputs.append(Tensor.create_from_torch(input))

        for i in range(1, 4):
            weights = ForgeLeakyReluTest.WEIGHTS_DISTRIBUTION(
                ForgeLeakyReluTest.WEIGHTS_RANGE_MIN, ForgeLeakyReluTest.WEIGHTS_RANGE_MAX
            ).sample(self.shape)
            weights.requires_grad = True
            self.set_parameter("train_param" + str(i), weights)

    def forward(self, x1, x2, x3):

        # Layer 2
        mul1 = forge.op.Multiply("mul1", x1, self.train_param1)
        mul2 = forge.op.Multiply("mul2", x2, self.train_param2)
        mul3 = forge.op.Multiply("mul3", x3, self.train_param3)

        # Layer 3
        lrelu1 = forge.op.LeakyRelu("lrelu1", mul1, alpha=self.alpha)
        lrelu2 = forge.op.LeakyRelu("lrelu2", mul2, alpha=self.alpha)
        lrelu3 = forge.op.LeakyRelu("lrelu3", mul3, alpha=self.alpha)

        # Layer 4
        mul4 = forge.op.Multiply("mul4", lrelu1, x2)
        mul5 = forge.op.Multiply("mul5", lrelu2, x3)
        mul6 = forge.op.Multiply("mul6", self.train_param2, lrelu3)

        # Layer 5
        lrelu4 = forge.op.LeakyRelu("lrelu4", mul4, alpha=self.alpha)
        lrelu5 = forge.op.LeakyRelu("lrelu5", mul5, alpha=self.alpha)
        lrelu6 = forge.op.LeakyRelu("lrelu6", mul6, alpha=self.alpha)

        # Layer 6
        mul7 = forge.op.Multiply("mul7", lrelu4, mul2)
        mul8 = forge.op.Multiply("mul8", lrelu5, mul3)
        mul9 = forge.op.Multiply("mul9", lrelu6, mul1)
        mul10 = forge.op.Multiply("mul10", lrelu4, lrelu5)

        # Layer 7
        lrelu7 = forge.op.LeakyRelu("lrelu7", mul10, alpha=self.alpha)
        lrelu8 = forge.op.LeakyRelu("lrelu8", mul8, alpha=self.alpha)
        lrelu9 = forge.op.LeakyRelu("lrelu9", mul9, alpha=self.alpha)

        # Layer 8
        mul11 = forge.op.Multiply("mul11", mul7, lrelu7)
        mul12 = forge.op.Multiply("mul12", lrelu8, mul6)
        mul13 = forge.op.Multiply("mul13", mul5, lrelu9)

        # Layer 9
        lrelu10 = forge.op.LeakyRelu("lrelu10", mul11, alpha=self.alpha)
        lrelu11 = forge.op.LeakyRelu("lrelu11", mul12, alpha=self.alpha)
        lrelu12 = forge.op.LeakyRelu("lrelu12", mul13, alpha=self.alpha)

        # Layer 10
        mul14 = forge.op.Multiply("mul14", lrelu10, mul8)
        mul15 = forge.op.Multiply("mul15", lrelu11, mul9)
        mul16 = forge.op.Multiply("mul16", lrelu12, lrelu6)

        # Layer 11
        mul17 = forge.op.Multiply("mul17", mul14, lrelu8)
        mul18 = forge.op.Multiply("mul18", mul15, mul16)

        # Layer 12
        lrelu13 = forge.op.LeakyRelu("lrelu13", mul18, alpha=self.alpha)

        return mul17, lrelu13
