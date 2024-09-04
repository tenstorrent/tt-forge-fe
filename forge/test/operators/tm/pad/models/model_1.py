# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 1 
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
        Forge Test 1

    """

    def __init__(
        self,
        shape,
        pad
    ):
        super().__init__("Forge Test 1")


        self.testname = "Operator Pad, Test 1"
        self.shape = shape
        self.pad = pad
        
        self.train_param = forge.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape))]
        self.set_parameter("train_param", torch.rand(*self.shape, requires_grad=True))

    def forward(self, x):

        # Layer 2
        mul = forge.op.Multiply("mul", x, self.train_param)

        # Layer 3
        pad = forge.op.Pad("pad", mul, self.pad)

        return pad