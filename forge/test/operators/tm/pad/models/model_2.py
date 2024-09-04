# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 2
#   Pad operators defined by Forge API
#   These kinds of tests test only single specific operator through different Forge architectures
# 


import torch

import forge
import forge.op
import forge.op.nn as nn

from forge import ForgeModule, Tensor


class ForgePadTest(ForgeModule):
    """
        Forge Test 2

    """

    def __init__(
        self,
        shape,
        pad
    ):
        super().__init__("Forge Test 2")


        self.testname = "Operator Pad, Test 2"
        self.shape = shape
        self.pad = pad
        
        self.train_param1 = forge.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = forge.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape)) for _ in range(2)]

        self.set_parameter("train_param1", torch.rand(*self.shape, requires_grad=True))
        self.set_parameter("train_param2", torch.rand(*self.shape, requires_grad=True))

    def forward(self, x1, x2):

        # Layer 2
        mul1 = forge.op.Multiply("mul1", x1, self.train_param1)
        mul2 = forge.op.Multiply("mul2", x2, self.train_param2)

        # Layer 3
        pad1 = forge.op.Pad("pad1", mul1, self.pad)
        pad2 = forge.op.Pad("pad2", mul2, self.pad)

        # Layer 4
        mul3 = forge.op.Multiply("mul3", pad1, pad2)
        pad3 = forge.op.Pad("pad3", x1, self.pad)
        pad4 = forge.op.Pad("pad4", self.train_param2, self.pad)

        # Layer 5
        mul4 = forge.op.Multiply("mul4", pad3, mul3)
        mul5 = forge.op.Multiply("mul5", mul3, pad4)

        # Layer 6
        pad5 = forge.op.Pad("pad5", mul4, self.pad)
        pad6 = forge.op.Pad("pad6", mul5, self.pad)

        # Layer 7
        mul6 = forge.op.Multiply("mul6", pad5, pad6)

        return mul6