# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 0
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
        Buda Test 0

    """

    INPUTS_RANGE_MIN = 0.3
    INPUTS_RANGE_MAX = 0.5
    INPUTS_DISTRIBUTION = Uniform

    WEIGHTS_RANGE_MIN = 0.4
    WEIGHTS_RANGE_MAX = 0.5
    WEIGHTS_DISTRIBUTION = Uniform

    def __init__(
        self,
        shape,
        dim,
        stable):
        super().__init__("Buda Test 0")

        assert hasattr(shape, '__iter__'), "Shape must be iterable"
        assert dim < len(shape), "Dimension out of the shape"
        assert dim >= 0, "Dimension cant' be negative"

        self.testname = "Operator softmax Test 0"
        self.shape = shape
        self.dim = dim
        self.stable = stable

        # print(f"shape: {self.shape}, dim: {self.dim}")
        
        self.train_param = forge.Parameter(*self.shape, requires_grad=True)

        input = BudaSoftmaxTest.INPUTS_DISTRIBUTION(
            BudaSoftmaxTest.INPUTS_RANGE_MIN, 
            BudaSoftmaxTest.INPUTS_RANGE_MAX).sample(self.shape)
        self.inputs = [Tensor.create_from_torch(input)]

        weights = BudaSoftmaxTest.WEIGHTS_DISTRIBUTION(
            BudaSoftmaxTest.WEIGHTS_RANGE_MIN, 
            BudaSoftmaxTest.WEIGHTS_RANGE_MAX).sample(self.shape)
        weights.requires_grad = True
        self.set_parameter("train_param", weights)

    def forward(self, x):

        # Layer 2
        mul = forge.op.Multiply("mul", x, self.train_param)

        # Layer 3
        sm1 = nn.Softmax("sm1", mul, dim=self.dim, stable=self.stable)
        sm2 = nn.Softmax("sm2", self.train_param, dim=self.dim, stable=self.stable)

        return sm1, sm2

    def values(self):
        return [item.value() for item in self.inputs]
