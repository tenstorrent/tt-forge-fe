# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 2
#   Reshape operators defined by Forge API
#   These kinds of tests test only single specific operator through different Forge architectures
#


import torch
import numpy as np

import forge
import forge.op
import forge.op.nn as nn

from forge import ForgeModule, Tensor


class ForgeReshapeTest(ForgeModule):
    """
    Forge Test 2

    """

    def __init__(self, old_shape, new_shape):
        super().__init__("Forge Test 2")

        assert np.prod(old_shape) == np.prod(new_shape), "Size of a tensor should stay the same"

        self.testname = "Operator reshape Test 2"
        self.old_shape = old_shape
        self.new_shape = new_shape

        self.train_param1 = forge.Parameter(*self.old_shape, requires_grad=True)
        self.train_param2 = forge.Parameter(*self.old_shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.old_shape)) for i in range(2)]
        for i in range(1, 3):
            self.set_parameter("train_param" + str(i), torch.rand(*self.old_shape, requires_grad=True))

    def forward(self, x1, x2):

        # Layer 2
        mul1 = forge.op.Multiply("mul1", x1, self.train_param1)
        mul2 = forge.op.Multiply("mul2", x2, self.train_param2)

        # Layer 3
        rsh1 = forge.op.Reshape("rsh1", mul1, self.new_shape)
        rsh2 = forge.op.Reshape("rsh2", mul2, self.new_shape)

        # Layer 4
        mul3 = forge.op.Multiply("mul3", rsh1, rsh2)
        mul4 = forge.op.Multiply("mul4", self.train_param1, x2)

        # Layer 5
        rsh3 = forge.op.Reshape("rsh3", mul4, self.new_shape)

        # Layer 6
        mul5 = forge.op.Multiply("mul5", mul3, rsh3)

        return mul5

    def values(self):
        return [item.value() for item in self.inputs]
