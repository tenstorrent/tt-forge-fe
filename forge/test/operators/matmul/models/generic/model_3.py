# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 3
#   Matmul operator defined by Forge API
#   These kinds of tests test only single specific operator through different Forge architectures
# 


import torch

import forge

from forge import ForgeModule, Tensor


class ForgeMatmulTest(ForgeModule):
    """
        Forge Test 3

        In this test we have 10 operations, and three input tensors and three trainable variables.
        One operand represents input and the other one is trainable paramater.
    """

    def __init__(self, shape):
        super().__init__("Forge Test 3")
        self.testname = "Operator Matmul Test 3"
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
        mm1 = forge.op.Matmul("mm1", x1, tr1)
        tr2 = forge.op.Transpose("tr2", self.train_param2, -1, -2)
        mm2 = forge.op.Matmul("mm2", x2, tr2)
        tr3 = forge.op.Transpose("tr3", x3, -1, -2)
        mm3 = forge.op.Matmul("mm3", tr3, self.train_param3)

        # Layer 3
        mm4 = forge.op.Matmul("mm4", mm1, x2)
        mm5 = forge.op.Matmul("mm5", self.train_param2, mm3)
        mm6 = forge.op.Matmul("mm6", mm3, tr3)
        
        # Layer 4
        mm7 = forge.op.Matmul("mm7", mm2, mm5)
        mm8 = forge.op.Matmul("mm8", mm6, x3)
        
        # Layer 5
        mm9 = forge.op.Matmul("mm9", mm7, mm8)

        # Layer 6
        tr4 = forge.op.Transpose("tr4", mm4, -1, -2)
        mm10 = forge.op.Matmul("mm10", tr4, mm9)

        return mm10

    def values(self):
        return [item.value() for item in self.inputs]