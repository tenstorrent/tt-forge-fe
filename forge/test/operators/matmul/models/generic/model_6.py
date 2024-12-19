# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 6
#   Matmul operator defined by Forge API
#   These kinds of tests test only single specific operator through different Forge architectures
#


import torch

import forge
from forge import ForgeModule, Tensor


class ForgeMatmulTest(ForgeModule):
    """
    Forge Test 6

    In this test we have 13 operations, and 4 input tensors and 4 trainable variables.
    One operand represents input and the other one is trainable paramater.
    """

    def __init__(self, shape):
        super().__init__("Forge Test 6")
        self.testname = "Operator Matmul Test 6"
        self.shape = shape
        self.train_param1 = forge.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = forge.Parameter(*self.shape, requires_grad=True)
        self.train_param3 = forge.Parameter(*self.shape, requires_grad=True)
        self.train_param4 = forge.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape)) for i in range(4)]
        for i in range(4):
            self.set_parameter("train_param" + str(i + 1), torch.rand(*self.shape, requires_grad=True))

    def forward(self, x1, x2, x3, x4):

        # Layer 2
        tr1 = forge.op.Transpose("tr1", self.train_param1, -1, -2)
        mm1 = forge.op.Matmul("mm1", x1, tr1)

        tr2 = forge.op.Transpose("tr2", self.train_param2, -1, -2)
        mm2 = forge.op.Matmul("mm2", x2, tr2)

        tr3 = forge.op.Transpose("tr3", x3, -1, -2)
        mm3 = forge.op.Matmul("mm3", tr3, self.train_param3)

        tr4 = forge.op.Transpose("tr4", x4, -1, -2)
        mm4 = forge.op.Matmul("mm4", tr4, self.train_param4)

        # Layer 3
        mm5 = forge.op.Matmul("mm5", mm1, mm2)
        mm6 = forge.op.Matmul("mm6", x3, mm3)
        mm7 = forge.op.Matmul("mm7", mm3, mm4)

        # Layer 4
        mm8 = forge.op.Matmul("mm8", mm5, mm6)
        mm9 = forge.op.Matmul("mm9", mm3, mm7)
        mm10 = forge.op.Matmul("mm10", mm6, mm7)

        # Layer 5
        mm11 = forge.op.Matmul("mm11", mm8, mm9)
        tr5 = forge.op.Transpose("tr5", mm10, -1, -2)
        mm12 = forge.op.Matmul("mm12", mm9, tr5)

        # Layer 6
        mm13 = forge.op.Matmul("mm13", mm11, mm12)

        return mm13

    def values(self):
        return [item.value() for item in self.inputs]
