# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 7
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
        Forge Test 7

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
        super().__init__("Forge Test 7")

        self.testname = "Comparison Operator, Test 7"
        self.shape = shape
        self.opname = opname
        self.operator = operator
        self.mask = mask
        self.rng_min = rng_min
        self.rng_max = rng_max
        
        self.train_param1 = forge.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = forge.Parameter(*self.shape, requires_grad=True)
        self.train_param3 = forge.Parameter(*self.shape, requires_grad=True)

        self.inputs = []
        for _ in range(3):
            input_ = torch.rand(*self.shape) * (self.rng_max - self.rng_min) + self.rng_min
            if self.mask:
                input_ *= (1.0 * torch.randint(0, 2, self.shape))
            self.inputs.append(Tensor.create_from_torch(input_))
        for i in range(1, 4):
            self.set_parameter("train_param" + str(i), torch.rand(*self.shape, requires_grad=True))

    def forward(self, x1, x2, x3):

        # Layer 2
        comp1 = self.operator(self.opname + "1", x1, self.train_param1)
        comp2 = self.operator(self.opname + "2", self.train_param1, x2)
        comp3 = self.operator(self.opname + "3", x2, self.train_param2)
        comp4 = self.operator(self.opname + "4", self.train_param2, x3)
        comp5 = self.operator(self.opname + "5", self.train_param2, self.train_param3)

        # Layer 3
        mul1 = forge.op.Multiply("mul1", x1, comp2)
        mul2 = forge.op.Multiply("mul2", self.train_param1, comp4)
        mul3 = forge.op.Multiply("mul3", x2, comp5)
        mul4 = forge.op.Multiply("mul4", comp3, x3)
        mul5 = forge.op.Multiply("mul5", comp4, comp5)
        mul6 = forge.op.Multiply("mul6", comp5, self.train_param3)

        # Layer 4
        comp6 = self.operator(self.opname + "6", mul1, mul2)
        comp7 = self.operator(self.opname + "7", comp1, mul2)
        comp8 = self.operator(self.opname + "8", mul3, comp4)
        comp9 = self.operator(self.opname + "9", mul4, mul5)
        comp10 = self.operator(self.opname + "10", comp5, mul6)

        # Layer 5
        mul7 = forge.op.Multiply("mul7", comp6, mul2)
        mul8 = forge.op.Multiply("mul8", mul1, comp9)
        mul9 = forge.op.Multiply("mul9", comp7, mul4)
        mul10 = forge.op.Multiply("mul10", comp8, mul5)
        mul11 = forge.op.Multiply("mul11", mul4, comp10)

        # Layer 6
        mul12 = forge.op.Multiply("mul12", mul7, mul3)
        mul13 = forge.op.Multiply("mul13", mul8, mul2)
        mul14 = forge.op.Multiply("mul14", mul9, mul5)
        mul15 = forge.op.Multiply("mul15", mul10, mul11)

        # Layer 7
        mul16 = forge.op.Multiply("mul16", mul12, mul13)
        mul17 = forge.op.Multiply("mul17", mul13, mul14)
        mul18 = forge.op.Multiply("mul18", mul14, mul15)

        return mul16, mul17, mul18

    def values(self):
        return [item.value() for item in self.inputs]   