# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 4
#   Cimparison operators defined by Forge API
#   These kinds of tests test only single specific operator through different Forge architectures
# 


import torch

import forge
import forge.op
import forge.op.nn as nn

from forge import ForgeModule, Tensor


class BudaComparisonTest(ForgeModule):
    """
        Buda Test 4

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
        super().__init__("Buda Test 4")

        self.testname = "Comparison Operator, Test 4"
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
        mul1 = forge.op.Multiply("mul1", x1, self.train_param1)
        mul2 = forge.op.Multiply("mul2", x2, self.train_param1)
        mul3 = forge.op.Multiply("mul3", x2, self.train_param2)

        # Layer 3
        comp1 = self.operator(self.opname + "1", mul1, self.train_param1)
        comp2 = self.operator(self.opname + "2", mul2, x2)
        comp3 = self.operator(self.opname + "3", mul3, self.train_param2)

        # Layer 4
        mul4 = forge.op.Multiply("mul4", comp1, mul2)
        mul5 = forge.op.Multiply("mul5", comp2, mul3)
        mul6 = forge.op.Multiply("mul6", comp3, mul1)

        # Layer 5
        comp4 = self.operator(self.opname + "4", mul4, mul2)
        comp5 = self.operator(self.opname + "5", mul5, mul1)
        comp6 = self.operator(self.opname + "6", mul3, mul6)

        # Layer 6
        mul7 = forge.op.Multiply("mul7", comp4, mul5)
        mul8 = forge.op.Multiply("mul8", comp5, self.train_param2)
        mul9 = forge.op.Multiply("mul9", mul5, comp6)

        # Layer 7
        mul10 = forge.op.Multiply("mul10", mul7, mul8)
        mul11 = forge.op.Multiply("mul11", mul8, mul9)

        # Layer 8
        comp7 = self.operator(self.opname + "7", mul10, mul2)
        comp8 = self.operator(self.opname + "8", mul11, mul8)
        comp9 = self.operator(self.opname + "9", mul9, comp6)

        # Layer 9
        comp10 = self.operator(self.opname + "10", mul10, mul11)
        comp11 = self.operator(self.opname + "11", mul8, mul11)
        comp12 = self.operator(self.opname + "12", mul8, mul9)

        # Layer 10
        mul12 = forge.op.Multiply("mul12", comp10, self.train_param1)
        mul13 = forge.op.Multiply("mul13", comp11, mul9)
        mul14 = forge.op.Multiply("mul14", comp12, mul6)

        # Layer 11
        mul15 = forge.op.Multiply("mul15", mul12, mul13)
        mul16 = forge.op.Multiply("mul16", mul13, mul14)

        # Layer 12
        mul17 = forge.op.Multiply("mul17", mul13, mul16)
        mul18 = forge.op.Multiply("mul18", mul15, mul17)
        mul19 = forge.op.Multiply("mul19", mul16, mul17)

        return mul18, mul19

    def values(self):
        return [item.value() for item in self.inputs]   