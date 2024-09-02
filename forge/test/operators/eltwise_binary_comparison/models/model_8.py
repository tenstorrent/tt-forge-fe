# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 8
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
        Buda Test 8

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
        super().__init__("Buda Test 8")

        self.testname = "Comparison Operator, Test 8"
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
        mul1 = forge.op.Multiply("mul1", x1, self.train_param2)
        mul2 = forge.op.Multiply("mul2", self.train_param1, x3)
        mul3 = forge.op.Multiply("mul3", x2, self.train_param3)

        # Layer 3
        mul4 = forge.op.Multiply("mul4", x1, self.train_param1)
        mul5 = forge.op.Multiply("mul5", x2, self.train_param2)
        mul6 = forge.op.Multiply("mul6", x3, self.train_param3)

        # Layer 4
        comp1 = self.operator(self.opname + "1", x1, self.train_param1)
        comp2 = self.operator(self.opname + "2", mul4, mul2)
        comp3 = self.operator(self.opname + "3", mul2, mul5)
        comp4 = self.operator(self.opname + "4", mul3, mul6)
        comp5 = self.operator(self.opname + "5", mul6, x3)

        # Layer 5
        mul7 = forge.op.Multiply("mul7", comp1, mul4)
        mul8 = forge.op.Multiply("mul8", comp2, mul1)
        mul9 = forge.op.Multiply("mul9", comp3, mul5)
        mul10 = forge.op.Multiply("mul10", comp4, mul6)
        mul11 = forge.op.Multiply("mul11", comp5, self.train_param3)

        # Layer 6
        mul12 = forge.op.Multiply("mul12", mul7, comp3)
        mul13 = forge.op.Multiply("mul13", comp1, mul9)
        mul14 = forge.op.Multiply("mul14", mul8, comp4)
        mul15 = forge.op.Multiply("mul15", mul10, comp5)
        mul16 = forge.op.Multiply("mul16", comp2, mul11)

        # Layer 7
        comp6 = self.operator(self.opname + "6", mul12, mul4)
        comp7 = self.operator(self.opname + "7", mul8, mul13)
        comp8 = self.operator(self.opname + "8", mul14, mul10)
        comp9 = self.operator(self.opname + "9", mul15, mul16)

        # Layer 8
        mul17 = forge.op.Multiply("mul17", comp6, mul8)
        mul18 = forge.op.Multiply("mul18", comp7, mul9)
        mul19 = forge.op.Multiply("mul19", comp8, mul15)
        mul20 = forge.op.Multiply("mul20", comp9, mul14)

        # Layer 9
        comp10 = self.operator(self.opname + "10", mul17, mul18)
        comp11 = self.operator(self.opname + "11", mul18, mul19)
        comp12 = self.operator(self.opname + "12", mul19, mul20)
        comp13 = self.operator(self.opname + "13", mul20, mul16)

        # Layer 10
        mul21 = forge.op.Multiply("mul21", comp10, mul18)
        mul22 = forge.op.Multiply("mul22", comp11, mul19)
        mul23 = forge.op.Multiply("mul23", comp12, mul20)
        mul24 = forge.op.Multiply("mul24", comp13, mul15)

        # Layer 11
        mul25 = forge.op.Multiply("mul25", mul17, mul22)
        mul26 = forge.op.Multiply("mul26", mul21, mul23)
        mul27 = forge.op.Multiply("mul27", mul19, mul24)

        return mul25, mul26, mul27

    def values(self):
        return [item.value() for item in self.inputs]   