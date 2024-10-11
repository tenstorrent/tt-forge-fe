# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 3
#   HStack, HSlice operators defined by Forge API
#   These kinds of tests test only single specific operator through different Forge architectures
#


import torch

import forge
import forge.op
import forge.op.nn as nn

from forge import ForgeModule, Tensor


class ForgeHStackHSliceTest(ForgeModule):
    """
    Forge Test 3

    """

    def __init__(self, shape, slice):
        super().__init__("Forge Test 3")

        assert hasattr(shape, "__iter__"), "Shape must be iterable"
        assert len(shape) == 4, "Shape must be 4"
        assert shape[1] > 1, "Z dimension must be bigger than 1"
        assert shape[-1] % slice == 0, "The last dimension must be divisible by slice"

        self.testname = "Operator HStack, HSLice, Test 3"
        self.shape = shape
        self.slice = slice

        if type(self.shape) == tuple:
            self.shape = list(self.shape)
        self.shape[1] *= self.slice
        self.shape[-1] *= self.slice

        self.train_param1 = forge.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = forge.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape)) for i in range(2)]
        for i in range(1, 3):
            self.set_parameter("train_param" + str(i), torch.rand(*self.shape, requires_grad=True))

    def forward(self, x1, x2):

        # Layer 2
        hst1 = forge.op.HStack("hst1", x1, self.slice)
        # +1
        hst2 = forge.op.HStack("hst2", self.train_param1, self.slice)
        # +1
        hst3 = forge.op.HStack("hst3", x2, self.slice)
        # +1
        hst4 = forge.op.HStack("hst4", self.train_param2, self.slice)
        # +1

        # Layer 3
        mul1 = forge.op.Multiply("mul1", hst1, hst2)
        # +1
        mul2 = forge.op.Multiply("mul2", hst3, hst4)
        # +1

        # Layer 4
        hsl1 = forge.op.HSlice("hsl1", mul1, self.slice)
        # 0
        mul3 = forge.op.Multiply("mul3", hst2, mul2)
        # +1

        # Layer 5
        mul4 = forge.op.Multiply("mul4", hsl1, x2)
        # 0

        # Layer 6
        hsl2 = forge.op.HSlice("hsl2", mul4, self.slice)
        # -1
        hsl3 = forge.op.HSlice("hsl3", mul3, self.slice)
        # 0
        hst5 = forge.op.HStack("hst5", self.train_param1, self.slice)
        # +1
        hst6 = forge.op.HStack("hst6", self.train_param2, self.slice)
        # +1

        # Layer 7
        # hst7 = forge.op.HStack("hst7", hst6, self.slice)

        # Layer 8
        add1 = forge.op.Add("add1", hsl2, forge.op.HSlice("hsl5", hsl3, self.slice))
        # -1
        mul5 = forge.op.Multiply("mul5", hst5, hst6)
        # +1

        # Layer 9
        hst8 = forge.op.HStack("hst8", add1, self.slice)
        # 0
        hst9 = forge.op.HStack("hst9", hst8, self.slice)
        # +1
        # hst10 = forge.op.HStack("hsl10", mul5, self.slice)

        # Layer 10
        sub1 = forge.op.Subtract("sub1", hst9, mul5)
        # +1

        # Layer 11
        hst10 = forge.op.HStack("hst10", sub1, self.slice)

        return hst10

    def values(self):
        return [item.value() for item in self.inputs]
