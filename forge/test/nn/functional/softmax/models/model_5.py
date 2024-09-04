# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 5
#   Softmax operators defined by Forge API
#   These kinds of tests test only single specific operator through different Forge architectures
# 


import torch
from torch.distributions import Uniform, Normal

import forge
import forge.op
import forge.op.nn as nn

from forge import ForgeModule, Tensor


class ForgeSoftmaxTest(ForgeModule):
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
        dim,
        stable):
        super().__init__("Forge Test 5")

        assert hasattr(shape, '__iter__'), "Shape must be iterable"
        assert dim < len(shape), "Dimension out of the shape"
        assert dim >= 0, "Dimension cant' be negative"

        self.testname = "Operator softmax Test 5"
        self.shape = shape
        self.dim = dim
        self.stable=stable
        
        self.train_param1 = forge.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = forge.Parameter(*self.shape, requires_grad=True)
        self.train_param3 = forge.Parameter(*self.shape, requires_grad=True)

        self.inputs = []
        for i in range(3):
            input = ForgeSoftmaxTest.INPUTS_DISTRIBUTION(
                ForgeSoftmaxTest.INPUTS_RANGE_MIN, 
                ForgeSoftmaxTest.INPUTS_RANGE_MAX).sample(self.shape)
            self.inputs.append(Tensor.create_from_torch(input))

        for i in range(1, 4):
            weights = ForgeSoftmaxTest.WEIGHTS_DISTRIBUTION(
                ForgeSoftmaxTest.WEIGHTS_RANGE_MIN, 
                ForgeSoftmaxTest.WEIGHTS_RANGE_MAX).sample(self.shape)
            weights.requires_grad = True
            self.set_parameter("train_param" + str(i), weights)

    def forward(self, x1, x2, x3):

        # Layer 2
        mul1 = forge.op.Multiply("mul1", x1, self.train_param2)
        mul2 = forge.op.Multiply("mul2", x2, self.train_param1)
        mul3 = forge.op.Multiply("mul3", x3, self.train_param3)

        sm1 = nn.Softmax("sm1", mul2, dim=self.dim, stable=self.stable)
        sm2 = nn.Softmax("sm2", mul1, dim=self.dim, stable=self.stable)
        sm3 = nn.Softmax("sm3", mul3, dim=self.dim, stable=self.stable)
        sm4 = nn.Softmax("sm4", mul2, dim=self.dim, stable=self.stable)

        # Layer 3
        add1 = forge.op.Add("add1", self.train_param1, sm2)
        add2 = forge.op.Add("add2", sm4, sm1)
        add3 = forge.op.Add("add3", sm3, x3)

        sm5 = nn.Softmax("sm5", add2, dim=self.dim, stable=self.stable)
        sm6 = nn.Softmax("sm6", add1, dim=self.dim, stable=self.stable)
        sm7 = nn.Softmax("sm7", add3, dim=self.dim, stable=self.stable)
        sm8 = nn.Softmax("sm8", add2, dim=self.dim, stable=self.stable)

        # Layer 4
        mul4 = forge.op.Multiply("mul4", x2, sm7)
        mul5 = forge.op.Multiply("mul5", sm6, sm8)
        mul6 = forge.op.Multiply("mul6", sm5, self.train_param2)

        sm9 = nn.Softmax("sm9", mul5, dim=self.dim, stable=self.stable)
        sm10 = nn.Softmax("sm10", mul4, dim=self.dim, stable=self.stable)
        sm11 = nn.Softmax("sm11", mul6, dim=self.dim, stable=self.stable)
        sm12 = nn.Softmax("sm12", mul5, dim=self.dim, stable=self.stable)

        # Layer 5
        mul7 = forge.op.Multiply("mul7", sm9, sm11)
        mul8 = forge.op.Multiply("mul8", sm10, sm12)

        return mul7, mul8

    def values(self):
        return [item.value() for item in self.inputs]   