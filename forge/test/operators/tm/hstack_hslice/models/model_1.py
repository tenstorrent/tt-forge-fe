# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 1 
#   HStack, HSlice operators defined by Forge API
#   These kinds of tests test only single specific operator through different Forge architectures
# 


import torch

import forge
import forge.op
import forge.op.nn as nn

from forge import ForgeModule, Tensor


class BudaHStackHSliceTest(ForgeModule):
    """
        Buda Test 1

    """

    def __init__(
        self,
        shape,
        slice):
        super().__init__("Buda Test 1")

        assert hasattr(shape, '__iter__'), "Shape must be iterable"
        assert len(shape) == 4, "Shape must be 4"
        assert shape[1] > 1, "Z dimension must be bigger than 1"
        assert shape[-1] % slice == 0, "The last dimension must be divisible by slice"

        self.testname = "Operator HStack, HSLice, Test 1"
        self.shape = shape
        self.slice = slice
        
        self.train_param = forge.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape))]
        self.set_parameter("train_param", torch.rand(*self.shape, requires_grad=True))

    def forward(self, x):

        # Layer 2
        hst1 = forge.op.HStack("hst1", x, self.slice)
        hst2 = forge.op.HStack("hst2", self.train_param, self.slice)
        mul1 = forge.op.Multiply("mul1", x, self.train_param)

        # Layer 3
        mul2 = forge.op.Multiply("mul2", hst1, hst2)
        hst3 = forge.op.HStack("hst3", mul1, self.slice)

        # Layer 4
        mul3 = forge.op.Multiply("mul3", mul2, hst3)

        # Layer 5
        hsl1 = forge.op.HSlice("hsl1", mul3, self.slice)

        # Layer 6
        mul4 = forge.op.Multiply("mul4", hsl1, self.train_param)

        return mul4

    def values(self):
        return [item.value() for item in self.inputs]   