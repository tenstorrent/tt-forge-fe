# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 4
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
    Forge Test 4

    """

    def __init__(self, shape, slice):
        super().__init__("Forge Test 4")

        assert hasattr(shape, "__iter__"), "Shape must be iterable"
        assert len(shape) == 4, "Shape must be 4"
        assert shape[1] > 1, "Z dimension must be bigger than 1"
        assert shape[-1] % slice == 0, "The last dimension must be divisible by slice"

        self.testname = "Operator HStack, HSLice, Test 4"
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
        hsl1 = forge.op.HSlice("hsl1", x1, self.slice)
        mul1 = forge.op.Multiply("mul1", x1, self.train_param1)
        mul2 = forge.op.Multiply("mul2", x2, self.train_param2)

        # Layer 3
        hsl2 = forge.op.HSlice("hsl2", mul1, self.slice)
        hsl3 = forge.op.HSlice("hsl3", mul2, self.slice)
        hsl4 = forge.op.HSlice("hsl4", self.train_param2, self.slice)

        # Layer 4
        add1 = forge.op.Add("add1", hsl1, hsl2)
        sub1 = forge.op.Subtract("sub1", hsl3, hsl4)

        # Layer 5
        hsl5 = forge.op.HSlice("hsl5", add1, self.slice)
        hsl6 = forge.op.HSlice("hsl6", sub1, self.slice)

        # Layer 6
        sub2 = forge.op.Subtract("sub2", self.train_param1, self.train_param2)
        add2 = forge.op.Add("add2", hsl5, hsl6)

        # Layer 7
        hsl7 = forge.op.HSlice("hsl7", sub2, self.slice)
        hst1 = forge.op.HStack("hst1", add2, self.slice)

        # Layer 8
        add3 = forge.op.Add("add3", hsl7, hst1)

        # Layer 9
        hsl8 = forge.op.HSlice("hsl8", add3, self.slice)

        return hsl8

    def values(self):
        return [item.value() for item in self.inputs]
