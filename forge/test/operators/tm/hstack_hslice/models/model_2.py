# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 2
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
    Forge Test 2

    """

    def __init__(self, shape, slice):
        super().__init__("Forge Test 2")

        assert hasattr(shape, "__iter__"), "Shape must be iterable"
        assert len(shape) == 4, "Shape must be 4"
        assert shape[1] > 1, "Z dimension must be bigger than 1"
        assert shape[-1] % slice == 0, "The last dimension must be divisible by slice"

        self.testname = "Operator HStack, HSLice, Test 2"
        self.shape = shape
        self.slice = slice

        self.train_param1 = forge.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = forge.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape)) for i in range(2)]
        for i in range(1, 3):
            self.set_parameter("train_param" + str(i), torch.rand(*self.shape, requires_grad=True))

    def forward(self, x1, x2):

        # Layer 2
        mul1 = forge.op.Multiply("mul1", x1, self.train_param1)
        mul2 = forge.op.Multiply("mul2", x2, self.train_param2)

        # Layer 3
        hsl1 = forge.op.HSlice("hsl1", mul1, self.slice)
        hsl2 = forge.op.HSlice("hsl2", mul2, self.slice)

        # Layer 4
        mul3 = forge.op.Multiply("mul3", hsl1, hsl2)
        mul4 = forge.op.Multiply("mul4", self.train_param1, self.train_param2)

        # Layer 5
        hst1 = forge.op.HStack("hst1", mul3, self.slice)

        # Layer 6
        add1 = forge.op.Add("add1", hst1, mul4)

        # Layer 7
        hst2 = forge.op.HStack("hst2", add1, self.slice)

        return hst2

    def values(self):
        return [item.value() for item in self.inputs]
