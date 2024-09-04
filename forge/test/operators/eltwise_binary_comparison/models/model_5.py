# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 5
#   Cimparison operators defined by Forge API
#   These kinds of tests test only single specific operator through different Forge architectures
# 


import torch

import forge
import forge.op
import forge.op.nn as nn

from forge import ForgeModule, Tensor


class ForgeComparisonTest(ForgeModule):
    """
        Forge Test 5

    """

    def __init__(
        self,
        shape,
        opname,
        operator,
        mask,
        rng_min,
        rng_max
        ):
        super().__init__("Forge Test 5")

        self.testname = "Comparison Operator, Test 5"
        self.shape = shape
        self.opname = opname
        self.operator = operator
        self.mask = mask
        self.rng_min = rng_min
        self.rng_max = rng_max
        
        self.train_param1 = forge.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = forge.Parameter(*self.shape, requires_grad=True)

        self.inputs = []
        for _ in range(2):
            input_ = torch.rand(*self.shape) * (self.rng_max - self.rng_min) + self.rng_min
            if self.mask:
                input_ *= (1.0 * torch.randint(0, 2, self.shape))
            self.inputs.append(Tensor.create_from_torch(input_))
        for i in range(1, 3):
            self.set_parameter("train_param" + str(i), torch.rand(*self.shape, requires_grad=True))

    def forward(self, x1, x2):

        # Layer 2
        comp1 = self.operator(self.opname + "1", x1, self.train_param1)
        comp2 = self.operator(self.opname + "2", self.train_param2, x2)
        comp3 = self.operator(self.opname + "3", self.train_param1, self.train_param2)

        # Layer 3
        mul1 = forge.op.Multiply("mul1", comp1, self.train_param1)
        mul2 = forge.op.Multiply("mul2", comp2, x2)
        mul3 = forge.op.Multiply("mul3", comp3, self.train_param2)

        # Layer 4
        mul4 = forge.op.Multiply("mul4", x1, mul2)
        mul5 = forge.op.Multiply("mul5", x2, mul3)

        # Layer 5
        mul6 = forge.op.Multiply("mul6", mul1, mul2)
        mul7 = forge.op.Multiply("mul7", mul4, mul3)

        # Layer 6
        comp4 = self.operator(self.opname + "4", mul6, mul4)
        comp5 = self.operator(self.opname + "5", mul7, mul5)

        # Layer 7
        mul8 = forge.op.Multiply("mul8", mul1, comp4)
        mul9 = forge.op.Multiply("mul9", mul7, comp5)
        comp6 = self.operator(self.opname + "6", mul4, mul2)
        comp7 = self.operator(self.opname + "7", mul5, mul3)

        # Layer 8
        mul10 = forge.op.Multiply("mul10", comp6, mul7)
        mul11 = forge.op.Multiply("mul11", comp7, mul5)

        # Layer 9
        mul12 = forge.op.Multiply("mul12", mul8, mul10)
        mul13 = forge.op.Multiply("mul13", mul10, mul9)
        mul14 = forge.op.Multiply("mul14", mul9, mul11)

        # Layer 10
        comp8 = self.operator(self.opname + "8", mul12, mul13)
        comp9 = self.operator(self.opname + "9", mul13, mul11)

        # Layer 11
        mul15 = forge.op.Multiply("mul15", comp8, mul13)
        mul16 = forge.op.Multiply("mul16", comp9, mul14)

        return mul15, mul16

    def values(self):
        return [item.value() for item in self.inputs]   