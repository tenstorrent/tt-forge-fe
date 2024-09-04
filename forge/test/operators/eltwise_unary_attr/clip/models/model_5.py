# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 5
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
        Forge Test 5

    """

    INPUTS_RANGE_MIN = -1.0
    INPUTS_RANGE_MAX = 1.0
    INPUTS_DISTRIBUTION = Normal

    WEIGHTS_RANGE_MIN = -1.0
    WEIGHTS_RANGE_MAX = 1.0
    WEIGHTS_DISTRIBUTION = Normal

    def __init__(
        self,
        shape,
        min_value,
        max_value
    ):
        super().__init__("Forge Test 5")

        self.testname = "Operator Clip, Test 5"
        self.shape = shape
        self.min_value = min_value
        self.max_value = max_value
        
        self.train_param1 = forge.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = forge.Parameter(*self.shape, requires_grad=True)
        self.train_param3 = forge.Parameter(*self.shape, requires_grad=True)

        self.inputs = []
        for i in range(3):
            input = ForgeClipTest.INPUTS_DISTRIBUTION(
                ForgeClipTest.INPUTS_RANGE_MIN, 
                ForgeClipTest.INPUTS_RANGE_MAX).sample(self.shape)
            self.inputs.append(Tensor.create_from_torch(input))

        for i in range(1, 4):
            weights = ForgeClipTest.WEIGHTS_DISTRIBUTION(
                ForgeClipTest.WEIGHTS_RANGE_MIN, 
                ForgeClipTest.WEIGHTS_RANGE_MAX).sample(self.shape)
            weights.requires_grad = True
            self.set_parameter("train_param" + str(i), weights)



    def forward(self, x1, x2, x3):

        # Layer 2
        mul1 = forge.op.Multiply("mul1", x1, self.train_param1)
        mul2 = forge.op.Multiply("mul2", x2, self.train_param2)
        mul3 = forge.op.Multiply("mul3", x3, self.train_param3)

        # Layer 3
        mul4 = forge.op.Multiply("mul4", x2, self.train_param1)
        mul5 = forge.op.Multiply("mul5", x3, self.train_param2)
        clip1 = forge.op.Clip("clip1", mul1, min=self.min_value, max=self.max_value)
        clip2 = forge.op.Clip("clip2", mul2, min=self.min_value, max=self.max_value)
        clip3 = forge.op.Clip("clip3", mul3, min=self.min_value, max=self.max_value)

        # Layer 4
        clip4 = forge.op.Clip("clip4", mul4, min=self.min_value, max=self.max_value)
        clip5 = forge.op.Clip("clip5", mul5, min=self.min_value, max=self.max_value)

        # Layer 5
        add1 = forge.op.Add("add1", clip1, self.train_param1)
        add2 = forge.op.Add("add2", clip4, x2)
        add3 = forge.op.Add("add3", clip2, self.train_param2)
        add4 = forge.op.Add("add4", clip5, x3)
        add5 = forge.op.Add("add5", clip3, self.train_param3)

        # Layer 6
        clip6 = forge.op.Clip("clip6", add1, min=self.min_value, max=self.max_value)
        clip7 = forge.op.Clip("clip7", add2, min=self.min_value, max=self.max_value)
        clip8 = forge.op.Clip("clip8", add3, min=self.min_value, max=self.max_value)
        clip9 = forge.op.Clip("clip9", add4, min=self.min_value, max=self.max_value)
        clip10 = forge.op.Clip("clip10", add5, min=self.min_value, max=self.max_value)

        # Layer 7
        mul6 = forge.op.Multiply("mul6", clip6, clip4)
        mul7 = forge.op.Multiply("mul7", mul1, clip7)
        mul8 = forge.op.Multiply("mul8", mul2, clip8)
        mul9 = forge.op.Multiply("mul9", clip3, clip9)
        mul10 = forge.op.Multiply("mul10", add3, clip10)

        # Layer 8
        clip11 = forge.op.Clip("clip11", mul6, min=self.min_value, max=self.max_value)
        clip12 = forge.op.Clip("clip12", mul7, min=self.min_value, max=self.max_value)
        clip13 = forge.op.Clip("clip13", mul8, min=self.min_value, max=self.max_value)
        clip14 = forge.op.Clip("clip14", mul9, min=self.min_value, max=self.max_value)
        clip15 = forge.op.Clip("clip15", mul10, min=self.min_value, max=self.max_value)

        # Layer 9
        mul11 = forge.op.Multiply("mul11", clip11, clip8)
        mul12 = forge.op.Multiply("mul12", clip12, clip5)
        mul13 = forge.op.Multiply("mul13", clip13, clip7)
        mul14 = forge.op.Multiply("mul14", clip14, add5)
        mul15 = forge.op.Multiply("mul15", clip13, mul5)

        # Layer 10
        clip16 = forge.op.Clip("clip16", mul11, min=self.min_value, max=self.max_value)
        clip17 = forge.op.Clip("clip17", mul12, min=self.min_value, max=self.max_value)
        clip18 = forge.op.Clip("clip18", mul13, min=self.min_value, max=self.max_value)
        clip19 = forge.op.Clip("clip19", mul14, min=self.min_value, max=self.max_value)
        clip20 = forge.op.Clip("clip20", mul15, min=self.min_value, max=self.max_value)

        # Layer 11
        mul16 = forge.op.Multiply("mul16", clip16, clip12)
        mul17 = forge.op.Multiply("mul17", clip17, clip13)
        mul18 = forge.op.Multiply("mul18", clip18, clip19)
        mul19 = forge.op.Multiply("mul19", clip13, clip20)

        # Layer 12
        clip21 = forge.op.Clip("clip21", mul16, min=self.min_value, max=self.max_value)
        clip22 = forge.op.Clip("clip22", mul17, min=self.min_value, max=self.max_value)
        clip23 = forge.op.Clip("clip23", mul18, min=self.min_value, max=self.max_value)
        clip24 = forge.op.Clip("clip24", mul19, min=self.min_value, max=self.max_value)

        # Layer 13
        mul20 = forge.op.Multiply("mul20", clip21, mul12)
        mul21 = forge.op.Multiply("mul21", clip22, clip18)
        mul22 = forge.op.Multiply("mul22", clip23, clip24)

        return mul20, mul21, mul22
