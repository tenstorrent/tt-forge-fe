# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 1 
#   Matmul operator defined by Forge API
#   These kinds of tests test only single specific operator through different Forge architectures
# 


import torch

import forge

from forge import ForgeModule, Tensor


class ForgeMatmulTest(ForgeModule):
    """
        Forge Matmul Test 1

        In this test we have only one operator with two operands.
        One operand represents input and the other one is trainable paramater.
    """

    def __init__(self, shape):
        super().__init__("Forge Matmul Test 1")
        self.testname = "Operator Matmul Test 1"
        self.shape = shape
        self.train_param = forge.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape))]
        self.set_parameter("train_param", torch.rand(*self.shape, requires_grad=True))

    def forward(self, x):
        tr = forge.op.Transpose("tr", self.train_param, -1, -2)
        return forge.op.Matmul("mm", x, tr)

    def values(self):
        return [item.value() for item in self.inputs]