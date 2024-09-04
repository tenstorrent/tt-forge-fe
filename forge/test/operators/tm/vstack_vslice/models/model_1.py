# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 1 
#   VStack, VSlice operators defined by Forge API
#   These kinds of tests test only single specific operator through different Forge architectures/graphs
# 


import torch

import forge
import forge.op
import forge.op.nn as nn

from forge import ForgeModule, Tensor


class ForgeVStackVSliceTest(ForgeModule):
    """
        Forge Test 1
    """

    def __init__(
        self,
        shape,
        slice):
        super().__init__("Forge Test 1")

        assert hasattr(shape, '__iter__'), "Shape must be iterable"
        assert len(shape) == 4, "Shape must be 4"
        assert shape[1] > 1, "Z dimension must be bigger than 1"
        assert shape[-2] % slice == 0, "The last dimension must be divisible by slice"

        self.testname = "Operator VStack, VSLice, Test 1"
        self.shape = shape
        self.slice = slice

        print(f"SHAPE: {self.shape}")
        print(f"SLICE: {self.slice}")
        
        self.train_param = forge.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape))]
        self.set_parameter("train_param", torch.rand(*self.shape, requires_grad=True))

    def forward(self, x):

        # Layer 2
        vst1 = forge.op.VStack("vst1", x, self.slice)
        vst2 = forge.op.VStack("vst2", self.train_param, self.slice)
        mul1 = forge.op.Multiply("mul1", x, self.train_param)

        # Layer 3
        mul2 = forge.op.Multiply("mul2", vst1, vst2)
        vst3 = forge.op.VStack("vst3", mul1, self.slice)

        # Layer 4
        mul3 = forge.op.Multiply("mul3", mul2, vst3)

        # Layer 5
        vsl1 = forge.op.VSlice("vsl1", mul3, self.slice)

        # Layer 6
        mul4 = forge.op.Multiply("mul4", vsl1, self.train_param)

        return mul4  