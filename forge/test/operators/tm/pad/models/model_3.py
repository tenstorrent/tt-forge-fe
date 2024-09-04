# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 3
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
        Forge Test 3

    """

    def __init__(
        self,
        shape,
        pad
    ):
        super().__init__("Forge Test 3")


        self.testname = "Operator Pad, Test 3"
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
        mul2 = forge.op.Multiply("mul2", self.train_param1, x2)
        mul3 = forge.op.Multiply("mul3", x2, self.train_param2)

        # Layer 3
        pad1 = forge.op.Pad("pad1", mul1, self.pad)
        pad2 = forge.op.Pad("pad2", mul2, self.pad)
        pad3 = forge.op.Pad("pad3", mul3, self.pad)

        # Layer 4
        mul4 = forge.op.Multiply("mul4", self.train_param1, x2)
        add1 = forge.op.Add("add1", x2, self.train_param2)

        # Layer 5
        pad4 = forge.op.Pad("pad4", mul4, self.pad)
        pad5 = forge.op.Pad("pad5", add1, self.pad)

        # Layer 6
        mul5 = forge.op.Multiply("mul5", pad1, pad4)
        mul6 = forge.op.Multiply("mul6", pad2, pad3)
        add2 = forge.op.Add("add2", pad3, pad5)

        # Layer 7
        pad6 = forge.op.Pad("pad6", mul5, self.pad)
        pad7 = forge.op.Pad("pad7", mul6, self.pad)
        pad8 = forge.op.Pad("pad8", add2, self.pad)

        # Layer 8
        add4 = forge.op.Add("add4", pad6, pad7)
        add5 = forge.op.Add("add5", pad6, pad8)
        add6 = forge.op.Add("add6", pad7, pad8)

        # Layer 9
        pad9 = forge.op.Pad("pad9", add4, self.pad)
        pad10 = forge.op.Pad("pad10", add5, self.pad)
        pad11 = forge.op.Pad("pad11", add6, self.pad)

        return pad9, pad10, pad11