# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 4
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
    Forge Test 4

    """

    INPUTS_RANGE_MIN = -1.0
    INPUTS_RANGE_MAX = 1.0
    INPUTS_DISTRIBUTION = Normal

    WEIGHTS_RANGE_MIN = -1.0
    WEIGHTS_RANGE_MAX = 1.0
    WEIGHTS_DISTRIBUTION = Normal

    def __init__(self, shape, min_value, max_value):
        super().__init__("Forge Test 4")

        self.testname = "Operator Clip, Test 4"
        self.shape = shape
        self.min_value = min_value
        self.max_value = max_value

        self.train_param1 = forge.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = forge.Parameter(*self.shape, requires_grad=True)
        self.train_param3 = forge.Parameter(*self.shape, requires_grad=True)

        self.inputs = []
        for i in range(3):
            input = ForgeClipTest.INPUTS_DISTRIBUTION(
                ForgeClipTest.INPUTS_RANGE_MIN, ForgeClipTest.INPUTS_RANGE_MAX
            ).sample(self.shape)
            self.inputs.append(Tensor.create_from_torch(input))

        for i in range(1, 4):
            weights = ForgeClipTest.WEIGHTS_DISTRIBUTION(
                ForgeClipTest.WEIGHTS_RANGE_MIN, ForgeClipTest.WEIGHTS_RANGE_MAX
            ).sample(self.shape)
            weights.requires_grad = True
            self.set_parameter("train_param" + str(i), weights)

    def forward(self, x1, x2, x3):

        # Layer 2
        add1 = forge.op.Add("add1", x1, self.train_param1)
        add2 = forge.op.Add("add2", x1, x2)
        add3 = forge.op.Add("add3", x2, self.train_param3)
        add4 = forge.op.Add("add4", x3, self.train_param2)

        # Layer 3
        clip1 = forge.op.Clip("clip1", add1, min=self.min_value, max=self.max_value)
        clip2 = forge.op.Clip("clip2", add2, min=self.min_value, max=self.max_value)
        clip3 = forge.op.Clip("clip3", add3, min=self.min_value, max=self.max_value)
        clip4 = forge.op.Clip("clip4", add4, min=self.min_value, max=self.max_value)

        # Layer 4
        clip5 = forge.op.Clip("clip5", self.train_param1, min=self.min_value, max=self.max_value)
        clip6 = forge.op.Clip("clip6", self.train_param2, min=self.min_value, max=self.max_value)
        clip7 = forge.op.Clip("clip7", self.train_param3, min=self.min_value, max=self.max_value)

        # Layer 5
        mul1 = forge.op.Multiply("mul1", clip1, clip5)
        mul2 = forge.op.Multiply("mul2", clip2, clip3)
        mul3 = forge.op.Multiply("mul3", clip5, clip4)
        mul4 = forge.op.Multiply("mul4", clip6, clip7)

        # Layer 6
        clip8 = forge.op.Clip("clip8", mul1, min=self.min_value, max=self.max_value)
        clip9 = forge.op.Clip("clip9", mul2, min=self.min_value, max=self.max_value)
        clip10 = forge.op.Clip("clip10", mul3, min=self.min_value, max=self.max_value)
        clip11 = forge.op.Clip("clip11", mul4, min=self.min_value, max=self.max_value)

        # Layer 7
        add5 = forge.op.Add("add5", clip8, clip5)
        add6 = forge.op.Add("add6", clip9, clip6)
        add7 = forge.op.Add("add7", clip10, clip7)
        add8 = forge.op.Add("add8", clip4, clip11)

        # Layer 8
        clip12 = forge.op.Clip("clip12", add5, min=self.min_value, max=self.max_value)
        clip13 = forge.op.Clip("clip13", add6, min=self.min_value, max=self.max_value)
        clip14 = forge.op.Clip("clip14", add7, min=self.min_value, max=self.max_value)
        clip15 = forge.op.Clip("clip15", add8, min=self.min_value, max=self.max_value)

        # Layer 9
        mul5 = forge.op.Multiply("mul5", clip1, clip12)
        mul6 = forge.op.Multiply("mul6", mul2, clip13)
        mul7 = forge.op.Multiply("mul7", clip6, clip14)
        mul8 = forge.op.Multiply("mul8", clip15, clip7)

        # Layer 10
        clip16 = forge.op.Clip("clip16", mul5, min=self.min_value, max=self.max_value)
        clip17 = forge.op.Clip("clip17", mul6, min=self.min_value, max=self.max_value)
        clip18 = forge.op.Clip("clip18", mul7, min=self.min_value, max=self.max_value)
        clip19 = forge.op.Clip("clip19", mul8, min=self.min_value, max=self.max_value)

        # Layer 11
        mul9 = forge.op.Multiply("mul9", clip16, clip17)
        mul10 = forge.op.Multiply("mul10", clip17, clip18)
        mul11 = forge.op.Multiply("mul11", clip18, clip19)

        # Layer 12
        mul12 = forge.op.Multiply("mul12", mul9, clip9)
        mul13 = forge.op.Multiply("mul13", mul10, mul11)

        return mul12, mul13
