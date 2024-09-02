# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 3
#   Softmax operators defined by Forge API
#   These kinds of tests test only single specific operator through different Forge architectures
# 


import torch
from torch.distributions import Uniform, Normal

import forge
import forge.op
import forge.op.nn as nn

from forge import ForgeModule, Tensor


class BudaSoftmaxTest(ForgeModule):
    """
        Buda Test 3

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
        super().__init__("Buda Test 3")

        assert hasattr(shape, '__iter__'), "Shape must be iterable"
        assert dim < len(shape), "Dimension out of the shape"
        assert dim >= 0, "Dimension cant' be negative"

        self.testname = "Operator softmax Test 3"
        self.shape = shape
        self.dim = dim
        self.stable = stable
        
        self.train_param1 = forge.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = forge.Parameter(*self.shape, requires_grad=True)

        self.inputs = []
        for i in range(2):
            input = BudaSoftmaxTest.INPUTS_DISTRIBUTION(
                BudaSoftmaxTest.INPUTS_RANGE_MIN, 
                BudaSoftmaxTest.INPUTS_RANGE_MAX).sample(self.shape)
            self.inputs.append(Tensor.create_from_torch(input))

        for i in range(1, 3):
            weights = BudaSoftmaxTest.WEIGHTS_DISTRIBUTION(
                BudaSoftmaxTest.WEIGHTS_RANGE_MIN, 
                BudaSoftmaxTest.WEIGHTS_RANGE_MAX).sample(self.shape)
            weights.requires_grad = True
            self.set_parameter("train_param" + str(i), weights)

    def forward(self, x1, x2):

        # Layer 2
        mul1 = forge.op.Multiply("mul1", x1, self.train_param1)
        mul2 = forge.op.Multiply("mul2", x2, self.train_param2)

        # Layer 3
        sm1 = nn.Softmax("sm1", mul1, dim=self.dim, stable=self.stable)
        sm2 = nn.Softmax("sm2", self.train_param1, dim=self.dim, stable=self.stable)
        sm3 = nn.Softmax("sm3", mul2, dim=self.dim, stable=self.stable)
        sm4 = nn.Softmax("sm4", self.train_param2, dim=self.dim, stable=self.stable)

        # Layer 4
        sm5 = nn.Softmax("sm5", sm1, dim=self.dim, stable=self.stable)
        add1 = forge.op.Add("add1", sm2, sm3)
        sm6 = nn.Softmax("sm6", sm4, dim=self.dim, stable=self.stable)

        # Layer 5
        mul3 = forge.op.Multiply("mul3", sm5, add1)
        add2 = forge.op.Add("add2", self.train_param2, sm6)

        # Layer 6
        add3 = forge.op.Add("add3", mul3, add1)
        mul4 = forge.op.Multiply("mul4", add2, sm4)

        # Layer 7
        sm7 = nn.Softmax("sm7", add3, dim=self.dim, stable=self.stable)
        sm8 = nn.Softmax("sm8", mul4, dim=self.dim, stable=self.stable)

        # Layer 8
        mul5 = forge.op.Multiply("mul5", sm7, sm8)

        return mul5

    def values(self):
        return [item.value() for item in self.inputs]   