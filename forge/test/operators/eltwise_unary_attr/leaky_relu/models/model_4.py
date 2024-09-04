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

    def __init__(
        self,
        shape,
        alpha
    ):
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
                ForgeLeakyReluTest.INPUTS_RANGE_MIN, 
                ForgeLeakyReluTest.INPUTS_RANGE_MAX).sample(self.shape)
            self.inputs.append(Tensor.create_from_torch(input))

        for i in range(1, 4):
            weights = ForgeLeakyReluTest.WEIGHTS_DISTRIBUTION(
                ForgeLeakyReluTest.WEIGHTS_RANGE_MIN, 
                ForgeLeakyReluTest.WEIGHTS_RANGE_MAX).sample(self.shape)
            weights.requires_grad = True
            self.set_parameter("train_param" + str(i), weights)



    def forward(self, x1, x2, x3):

        # Layer 2
        add1 = forge.op.Add("add1", x1, self.train_param1)
        add2 = forge.op.Add("add2", x2, self.train_param1)
        add3 = forge.op.Add("add3", x3, self.train_param2)
        mul1 = forge.op.Multiply("mul1", x2, self.train_param2)
        mul2 = forge.op.Multiply("mul2", x3, self.train_param3)

        # Layer 3
        lrelu1 = forge.op.LeakyRelu("lrelu1", add1, alpha=self.alpha)
        lrelu2 = forge.op.LeakyRelu("lrelu2", add2, alpha=self.alpha)
        lrelu3 = forge.op.LeakyRelu("lrelu3", mul1, alpha=self.alpha)
        lrelu4 = forge.op.LeakyRelu("lrelu4", add3, alpha=self.alpha)
        lrelu5 = forge.op.LeakyRelu("lrelu5", mul2, alpha=self.alpha)

        # Layer 4
        mul3 = forge.op.Multiply("mul3", lrelu1, self.train_param1)
        mul4 = forge.op.Multiply("mul4", lrelu2, x2)
        mul5 = forge.op.Multiply("mul5", lrelu3, self.train_param2)
        mul6 = forge.op.Multiply("mul6", lrelu4, x3)
        add4 = forge.op.Add("add4", lrelu5, self.train_param3)

        # Layer 5
        lrelu6 = forge.op.LeakyRelu("lrelu6", mul3, alpha=self.alpha)
        lrelu7 = forge.op.LeakyRelu("lrelu7", mul4, alpha=self.alpha)
        lrelu8 = forge.op.LeakyRelu("lrelu8", mul5, alpha=self.alpha)
        lrelu9 = forge.op.LeakyRelu("lrelu9", mul6, alpha=self.alpha)
        lrelu10 = forge.op.LeakyRelu("lrelu10", add4, alpha=self.alpha)

        # Layer 6
        mul7 = forge.op.Multiply("mul7", lrelu6, add2)
        mul8 = forge.op.Multiply("mul8", lrelu8, lrelu4)
        mul9 = forge.op.Multiply("mul9", lrelu9, lrelu5)
        mul10 = forge.op.Multiply("mul10", lrelu10, self.train_param3)
        add5 = forge.op.Add("add5", lrelu7, lrelu3)

        # Layer 7
        lrelu11 = forge.op.LeakyRelu("lrelu11", mul7, alpha=self.alpha)
        lrelu12 = forge.op.LeakyRelu("lrelu12", add5, alpha=self.alpha)
        lrelu13 = forge.op.LeakyRelu("lrelu13", mul8, alpha=self.alpha)
        lrelu14 = forge.op.LeakyRelu("lrelu14", mul9, alpha=self.alpha)
        lrelu15 = forge.op.LeakyRelu("lrelu15", mul10, alpha=self.alpha)

        # Layer 8
        add6 = forge.op.Add("add6", lrelu11, mul3)
        add7 = forge.op.Add("add7", lrelu12, mul8)
        mul11 = forge.op.Multiply("mul11", lrelu13, mul5)
        mul12 = forge.op.Multiply("mul12", lrelu14, add4)
        mul13 = forge.op.Multiply("mul13", mul5, lrelu15)

        # Layer 9
        lrelu16 = forge.op.LeakyRelu("lrelu16", add6, alpha=self.alpha)
        lrelu17 = forge.op.LeakyRelu("lrelu17", add7, alpha=self.alpha)
        lrelu18 = forge.op.LeakyRelu("lrelu18", mul11, alpha=self.alpha)
        lrelu19 = forge.op.LeakyRelu("lrelu19", mul12, alpha=self.alpha)
        lrelu20 = forge.op.LeakyRelu("lrelu20", mul13, alpha=self.alpha)

        # Layer 10
        mul14 = forge.op.Multiply("mul14", lrelu16, mul7)
        mul15 = forge.op.Multiply("mul15", lrelu17, mul8)
        mul16 = forge.op.Multiply("mul16", lrelu18, lrelu19)
        mul17 = forge.op.Multiply("mul17", add5, lrelu20)

        # Layer 11
        lrelu21 = forge.op.LeakyRelu("lrelu21", mul14, alpha=self.alpha)
        lrelu22 = forge.op.LeakyRelu("lrelu22", mul15, alpha=self.alpha)
        lrelu23 = forge.op.LeakyRelu("lrelu23", mul16, alpha=self.alpha)
        lrelu24 = forge.op.LeakyRelu("lrelu24", mul17, alpha=self.alpha)

        # Layer 12
        add8 = forge.op.Add("add8", lrelu21, lrelu23)
        add9 = forge.op.Add("add9", lrelu22, lrelu24)

        return add8, add9
