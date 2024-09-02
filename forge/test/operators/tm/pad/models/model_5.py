# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 5
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
        Buda Test 5

    """

    def __init__(
        self,
        shape,
        pad
    ):
        super().__init__("Buda Test 5")


        self.testname = "Operator Pad, Test 5"
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
        mul1 = forge.op.Multiply("mul1", x1, self.train_param1)
        mul2 = forge.op.Multiply("mul2", x2, self.train_param2)
        mul3 = forge.op.Multiply("mul3", x3, self.train_param3)

        # Layer 3
        pad1 = forge.op.Pad("pad1", x1, self.pad)
        pad2 = forge.op.Pad("pad2", mul1, self.pad)
        pad3 = forge.op.Pad("pad3", self.train_param1, self.pad)
        pad4 = forge.op.Pad("pad4", x2, self.pad)
        pad5 = forge.op.Pad("pad5", mul2, self.pad)
        pad6 = forge.op.Pad("pad6", self.train_param2, self.pad)
        pad7 = forge.op.Pad("pad7", x3, self.pad)
        pad8 = forge.op.Pad("pad8", mul3, self.pad)
        pad9 = forge.op.Pad("pad9", self.train_param3, self.pad)

        # Layer 4
        pad10 = forge.op.Pad("pad10", x1, self.pad)
        mul4 = forge.op.Multiply("mul4", pad1, pad2)
        mul5 = forge.op.Multiply("mul5", pad2, pad3)
        mul6 = forge.op.Multiply("mul6", pad4, pad5)
        mul7 = forge.op.Multiply("mul7", pad5, pad6)
        mul8 = forge.op.Multiply("mul8", pad7, pad8)
        mul9 = forge.op.Multiply("mul9", pad8, pad9)

        # Layer 5
        mul10 = forge.op.Multiply("mul10", pad10, mul4)
        pad11 = forge.op.Pad("pad11", x2, self.pad)
        mul11 = forge.op.Multiply("mul11", mul5, pad11)
        pad12 = forge.op.Pad("pad12", x3, self.pad)
        mul12 = forge.op.Multiply("mul12", mul7, pad12)
        pad13 = forge.op.Pad("pad13", self.train_param3, self.pad)
        mul13 = forge.op.Multiply("mul13", mul9, pad13)

        # Layer 6
        pad14 = forge.op.Pad("pad14", mul10, self.pad)
        pad15 = forge.op.Pad("pad15", mul11, self.pad)
        pad16 = forge.op.Pad("pad16", mul6, self.pad)
        pad17 = forge.op.Pad("pad17", mul12, self.pad)
        pad18 = forge.op.Pad("pad18", mul8, self.pad)
        pad19 = forge.op.Pad("pad19", mul13, self.pad)

        # Layer 7
        mul14 = forge.op.Multiply("mul14", pad14, pad15)
        mul15 = forge.op.Multiply("mul15", pad16, pad17)
        mul16 = forge.op.Multiply("mul16", pad18, pad19)

        # Layer 8
        pad20 = forge.op.Pad("pad20", pad14, self.pad)
        pad21 = forge.op.Pad("pad21", mul14, self.pad)
        pad22 = forge.op.Pad("pad22", pad16, self.pad)
        pad23 = forge.op.Pad("pad23", mul15, self.pad)
        pad24 = forge.op.Pad("pad24", pad19, self.pad)
        pad25 = forge.op.Pad("pad25", mul16, self.pad)

        # Layer 9
        mul17 = forge.op.Multiply("mul17", pad20, pad23)
        mul18 = forge.op.Multiply("mul18", pad22, pad25)
        mul19 = forge.op.Multiply("mul19", pad21, pad24)

        # Layer 10
        pad26 = forge.op.Pad("pad26", mul17, self.pad)
        pad27 = forge.op.Pad("pad27", mul18, self.pad)
        pad28 = forge.op.Pad("pad28", mul19, self.pad)

        # Layer 11
        add1 = forge.op.Add("add1", pad26, pad27)
        add2 = forge.op.Add("add2", pad27, pad28)

        # Layer 12
        pad29 = forge.op.Pad("pad29", add1, self.pad)
        pad30 = forge.op.Pad("pad30", add2, self.pad)

        return pad29, pad30