# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 5
#   Reshape operators defined by Forge API
#   These kinds of tests test only single specific operator through different Forge architectures
#


import numpy as np
import torch

import forge
import forge.op
import forge.op.nn as nn
from forge import ForgeModule, Tensor


class ForgeReshapeTest(ForgeModule):
    """
    Forge Test 5

    """

    def __init__(self, old_shape, new_shape):
        super().__init__("Forge Test 5")

        assert np.prod(old_shape) == np.prod(new_shape), "Size of a tensor should stay the same"

        self.testname = "Operator reshape Test 5"
        self.old_shape = old_shape
        self.new_shape = new_shape

        self.train_param1 = forge.Parameter(*self.old_shape, requires_grad=True)
        self.train_param2 = forge.Parameter(*self.old_shape, requires_grad=True)
        self.train_param3 = forge.Parameter(*self.old_shape, requires_grad=True)
        self.train_param4 = forge.Parameter(*self.old_shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.old_shape)) for i in range(2)]
        for i in range(1, 5):
            self.set_parameter("train_param" + str(i), torch.rand(*self.old_shape, requires_grad=True))

    def forward(self, x1, x2):

        # Layer 2
        add1 = forge.op.Add("add1", x1, self.train_param1)
        add2 = forge.op.Add("add2", x2, self.train_param2)

        # Layer 3
        mul1 = forge.op.Multiply("mul1", add1, add2)

        # Layer 4
        rsh1 = forge.op.Reshape("rsh1", add1, self.new_shape)
        rsh2 = forge.op.Reshape("rsh2", add2, self.new_shape)

        # Layer 5
        mul2 = forge.op.Multiply("mul2", rsh1, rsh2)

        # Layer 6
        mul3 = forge.op.Multiply("mul3", mul1, self.train_param3)
        rsh3 = forge.op.Reshape("rsh3", self.train_param4, self.new_shape)
        mul4 = forge.op.Multiply("mul4", mul2, rsh3)

        return mul3, mul4

    def values(self):
        return [item.value() for item in self.inputs]
