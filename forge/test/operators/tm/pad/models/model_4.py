# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 4
#   Pad operators defined by Forge API
#   These kinds of tests test only single specific operator through different Forge architectures
# 


import torch

import forge
import forge.op
import forge.op.nn as nn

from forge import ForgeModule, Tensor


class BudaPadTest(ForgeModule):
    """
        Buda Test 4

    """

    def __init__(
        self,
        shape,
        pad
    ):
        super().__init__("Buda Test 4")


        self.testname = "Operator Pad, Test 4"
        self.shape = shape
        self.pad = pad
        
        self.train_param1 = forge.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = forge.Parameter(*self.shape, requires_grad=True)
        self.train_param3 = forge.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape)) for _ in range(3)]
        
        self.set_parameter("train_param1", torch.rand(*self.shape, requires_grad=True))
        self.set_parameter("train_param2", torch.rand(*self.shape, requires_grad=True))
        self.set_parameter("train_param3", torch.rand(*self.shape, requires_grad=True))

    def forward(self, x1, x2, x3):

        # Layer 2
        pad1 = forge.op.Pad("pad1", x1, self.pad)
        pad2 = forge.op.Pad("pad2", self.train_param1, self.pad)
        pad3 = forge.op.Pad("pad3", x2, self.pad)
        pad4 = forge.op.Pad("pad4", self.train_param2, self.pad)
        pad5 = forge.op.Pad("pad5", x3, self.pad)
        pad6 = forge.op.Pad("pad6", self.train_param3, self.pad)

        # Layer 3
        mul1 = forge.op.Multiply("mul1", x1, self.train_param1)
        mul2 = forge.op.Multiply("mul2", x2, self.train_param2)
        mul3 = forge.op.Multiply("mul3", x3, self.train_param3)

        # Layer 4
        pad7 = forge.op.Pad("pad7", mul1, self.pad)
        pad8 = forge.op.Pad("pad8", mul2, self.pad)
        pad9 = forge.op.Pad("pad9", mul3, self.pad)

        # Layer 5
        mul4 = forge.op.Multiply("mul4", pad7, pad1)
        mul5 = forge.op.Multiply("mul5", pad2, pad8)
        mul6 = forge.op.Multiply("mul6", pad8, pad4)
        mul7 = forge.op.Multiply("mul7", pad3, pad9)
        mul8 = forge.op.Multiply("mul8", pad5, pad6)

        # Layer 6
        pad10 = forge.op.Pad("pad10", pad7, self.pad)
        pad11 = forge.op.Pad("pad11", mul4, self.pad)
        pad12 = forge.op.Pad("pad12", mul5, self.pad)
        pad13 = forge.op.Pad("pad13", mul6, self.pad)
        pad14 = forge.op.Pad("pad14", mul7, self.pad)
        pad15 = forge.op.Pad("pad15", mul8, self.pad)
        pad16 = forge.op.Pad("pad16", pad6, self.pad)

        # Layer 7
        mul9 = forge.op.Multiply("mul9", pad10, pad12)
        mul10 = forge.op.Multiply("mul10", pad11, pad14)
        mul11 = forge.op.Multiply("mul11", pad13, pad15)
        mul12 = forge.op.Multiply("mul12", pad15, pad16)

        # Layer 8
        pad17 = forge.op.Pad("pad17", mul9, self.pad)
        pad18 = forge.op.Pad("pad18", mul10, self.pad)
        pad19 = forge.op.Pad("pad19", mul11, self.pad)
        pad20 = forge.op.Pad("pad20", mul12, self.pad)

        # Layer 9
        mul13 = forge.op.Multiply("mul13", pad17, pad18)
        mul14 = forge.op.Multiply("mul14", pad18, pad19)
        mul15 = forge.op.Multiply("mul15", pad19, pad20)

        return mul13, mul14, mul15