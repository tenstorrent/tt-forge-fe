# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 2
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
        Forge Test 2

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
        super().__init__("Forge Test 2")

        assert hasattr(shape, '__iter__'), "Shape must be iterable"
        assert dim < len(shape), "Dimension out of the shape"
        assert dim >= 0, "Dimension cant' be negative"

        self.testname = "Operator softmax Test 2"
        self.shape = shape
        self.dim = dim
        self.stable = stable
        
        self.train_param1 = forge.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = forge.Parameter(*self.shape, requires_grad=True)

        self.inputs = []
        for i in range(2):
            input = ForgeSoftmaxTest.INPUTS_DISTRIBUTION(
                ForgeSoftmaxTest.INPUTS_RANGE_MIN, 
                ForgeSoftmaxTest.INPUTS_RANGE_MAX).sample(self.shape)
            self.inputs.append(Tensor.create_from_torch(input))

        for i in range(1, 3):
            weights = ForgeSoftmaxTest.WEIGHTS_DISTRIBUTION(
                ForgeSoftmaxTest.WEIGHTS_RANGE_MIN, 
                ForgeSoftmaxTest.WEIGHTS_RANGE_MAX).sample(self.shape)
            weights.requires_grad = True
            self.set_parameter("train_param" + str(i), weights)

    def forward(self, x1, x2):

        # Layer 2
        mul1 = forge.op.Multiply("mul1", x1, self.train_param1)
        add1 = forge.op.Add("add1", x2, self.train_param2)

        # Layer 3
        sm1 = nn.Softmax("sm1", self.train_param1, dim=self.dim, stable=self.stable)
        mul2 = forge.op.Multiply("mul2", add1, sm1)
        sm2 = nn.Softmax("sm2", self.train_param2, dim=self.dim, stable=self.stable)

        # Layer 4
        add2 = forge.op.Add("add2", x1, mul1)
        mul3 = forge.op.Multiply("mul3", sm1, add2)
        mul4 = forge.op.Multiply("mul4", mul2, sm2)

        # Layer 5
        sm3 = nn.Softmax("sm3", add2, dim=self.dim, stable=self.stable)
        sm4 = nn.Softmax("sm4", mul3, dim=self.dim, stable=self.stable)
        sm5 = nn.Softmax("sm5", mul2, dim=self.dim, stable=self.stable)
        mul5 = forge.op.Multiply("mul5", sm5, self.train_param2)

        # Layer 6
        mul6 = forge.op.Multiply("mul6", sm3, sm4)
        add3 = forge.op.Add("add3", mul5, sm2)
        mul7 = forge.op.Multiply("mul7", sm5, add3)

        # Layer 7
        sm6 = nn.Softmax("sm6", mul6, dim=self.dim, stable=self.stable)
        sm7 = nn.Softmax("sm7", mul7, dim=self.dim, stable=self.stable)

        # Layer 8
        add4 = forge.op.Add("add4", sm6, sm7)
        add5 = forge.op.Add("add5", mul4, add4)

        return add5

    def values(self):
        return [item.value() for item in self.inputs]   