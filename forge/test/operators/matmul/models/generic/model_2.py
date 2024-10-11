# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 2
#   Matmul operator defined by Forge API
#   These kinds of tests test only single specific operator through different Forge architectures
#


import torch

import forge

from forge import ForgeModule, Tensor


class ForgeMatmulTest(ForgeModule):
    """
    Forge Test 2

    In this test we have 5 operations, and three input tensors and three trainable variables.
    One operand represents input and the other one is trainable paramater.
    """

    def __init__(self, shape):
        super().__init__("Forge Test 2")
        self.testname = "Operator Matmul Test 2"
        self.shape = shape
        self.train_param1 = forge.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = forge.Parameter(*self.shape, requires_grad=True)
        self.train_param3 = forge.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape)) for i in range(3)]
        for i in range(3):
            self.set_parameter("train_param" + str(i + 1), torch.rand(*self.shape, requires_grad=True))

    def forward(self, x1, x2, x3):

        # Layer 2
        tr1 = forge.op.Transpose("tr1", self.train_param1, -1, -2)
        # (W, Z, R, C) --> (W, Z, C, R)
        mm1 = forge.op.Matmul("mm1", x1, tr1)
        # (W, Z, R, C) x (W, Z, C, R) --> (W, Z, R, R)
        tr2 = forge.op.Transpose("tr2", self.train_param2, -1, -2)
        # (W, Z, R, C) --> (W, Z, C, R)
        mm2 = forge.op.Matmul("mm2", x2, tr2)
        # (W, Z, R, C) x (W, Z, C, R) --> (W, Z, R, R)
        tr3 = forge.op.Transpose("tr3", self.train_param3, -1, -2)
        # (W, Z, R, C) --> (W, Z, C, R)
        mm3 = forge.op.Matmul("mm3", x3, tr3)
        # (W, Z, R, C) x (W, Z, C, R) --> (W, Z, R, R)

        # Layer 3
        mm4 = forge.op.Matmul("mm4", mm1, mm2)
        # (W, Z, R, R) x (W, Z, R, R) --> (W, Z, R, R)

        # Layer 4
        mm5 = forge.op.Matmul("mm5", mm4, mm3)
        # (W, Z, R, R) x (W, Z, R, R) --> (W, Z, R, R)

        return mm5

    def values(self):
        return [item.value() for item in self.inputs]
