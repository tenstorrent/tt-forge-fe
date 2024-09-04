# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 3
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
        Forge Test 3

    """

    def __init__(
        self,
        old_shape,
        new_shape):
        super().__init__("Forge Test 3")

        assert np.prod(old_shape) == np.prod(new_shape), "Size of a tensor should stay the same"

        self.testname = "Operator reshape Test 3"
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
        mul2 = forge.op.Multiply("mul2", self.train_param1, x2)
        mul3 = forge.op.Multiply("mul3", x2, self.train_param2)

        # Layer 3
        rsh1 = forge.op.Reshape("rsh1", mul1, self.new_shape)
        rsh2 = forge.op.Reshape("rsh2", mul2, self.new_shape)
        rsh3 = forge.op.Reshape("rsh3", mul3, self.new_shape)

        # Layer 4
        mul4 = forge.op.Multiply("mul4", rsh1, rsh2)
        mul5 = forge.op.Multiply("mul5", rsh2, rsh3)

        # Layer 5
        rsh4 = forge.op.Reshape("rsh4", mul4, self.old_shape)
        rsh5 = forge.op.Reshape("rsh5", mul5, self.old_shape)

        # Layer 6
        mul6 = forge.op.Multiply("mul6", rsh4, self.train_param1)
        mul7 = forge.op.Multiply("mul7", rsh5, self.train_param2)

        return mul6, mul7

    def values(self):
        return [item.value() for item in self.inputs]   