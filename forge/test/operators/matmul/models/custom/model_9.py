# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 9
#   Matmul operator defined by Forge API
#   These kinds of tests test only single specific operator through different Forge architectures
#


import torch

import forge

from forge import ForgeModule, Tensor


class ForgeMatmulTest(ForgeModule):
    """
    Forge Test 9

    In this test we have 22 operations, and 3 input tensors and 13 trainable variables.
    One operand represents input and the other one is trainable paramater.
    """

    def __init__(self):
        super().__init__("Forge Test 9")
        self.testname = "Operator Matmul Test 9"

        # Input shapes
        self.shape_input1 = (1, 16, 64, 210)
        self.shape_input2 = (1, 16, 70, 64)
        self.shape_input3 = (1, 16, 240, 512)

        # Trainable parameters shapes
        self.shape_train1 = (1, 16, 210, 78)
        self.shape_train2 = (1, 16, 64, 36)
        self.shape_train3 = (1, 16, 512, 64)
        self.shape_train4 = (1, 16, 512, 64)
        self.shape_train5 = (1, 16, 64, 256)
        self.shape_train6 = (1, 16, 512, 50)
        self.shape_train7 = (1, 16, 64, 240)
        self.shape_train8 = (1, 16, 36, 240)
        self.shape_train9 = (1, 16, 70, 64)
        self.shape_train10 = (1, 16, 210, 240)
        self.shape_train11 = (1, 16, 50, 256)
        self.shape_train12 = (1, 16, 210, 70)
        self.shape_train13 = (1, 16, 50, 512)

        self.train_param1 = forge.Parameter(*self.shape_train1, requires_grad=True)
        self.train_param2 = forge.Parameter(*self.shape_train2, requires_grad=True)
        self.train_param3 = forge.Parameter(*self.shape_train3, requires_grad=True)
        self.train_param4 = forge.Parameter(*self.shape_train4, requires_grad=True)
        self.train_param5 = forge.Parameter(*self.shape_train5, requires_grad=True)
        self.train_param6 = forge.Parameter(*self.shape_train6, requires_grad=True)
        self.train_param7 = forge.Parameter(*self.shape_train7, requires_grad=True)
        self.train_param8 = forge.Parameter(*self.shape_train8, requires_grad=True)
        self.train_param9 = forge.Parameter(*self.shape_train9, requires_grad=True)
        self.train_param10 = forge.Parameter(*self.shape_train10, requires_grad=True)
        self.train_param11 = forge.Parameter(*self.shape_train11, requires_grad=True)
        self.train_param12 = forge.Parameter(*self.shape_train12, requires_grad=True)
        self.train_param13 = forge.Parameter(*self.shape_train13, requires_grad=True)

        def my_rand(*shape, requires_grad=False):
            return (torch.rand(*shape, requires_grad=requires_grad) - 0.5).detach()

        self.inputs = [
            Tensor.create_from_torch(my_rand(*self.shape_input1)),
            Tensor.create_from_torch(my_rand(*self.shape_input2)),
            Tensor.create_from_torch(my_rand(*self.shape_input3)),
        ]

        self.set_parameter("train_param1", my_rand(*self.shape_train1, requires_grad=True))
        self.set_parameter("train_param2", my_rand(*self.shape_train2, requires_grad=True))
        self.set_parameter("train_param3", my_rand(*self.shape_train3, requires_grad=True))
        self.set_parameter("train_param4", my_rand(*self.shape_train4, requires_grad=True))
        self.set_parameter("train_param5", my_rand(*self.shape_train5, requires_grad=True))
        self.set_parameter("train_param6", my_rand(*self.shape_train6, requires_grad=True))
        self.set_parameter("train_param7", my_rand(*self.shape_train7, requires_grad=True))
        self.set_parameter("train_param8", my_rand(*self.shape_train8, requires_grad=True))
        self.set_parameter("train_param9", my_rand(*self.shape_train9, requires_grad=True))
        self.set_parameter("train_param10", my_rand(*self.shape_train10, requires_grad=True))
        self.set_parameter("train_param11", my_rand(*self.shape_train11, requires_grad=True))
        self.set_parameter("train_param12", my_rand(*self.shape_train12, requires_grad=True))
        self.set_parameter("train_param13", my_rand(*self.shape_train13, requires_grad=True))

    def forward(self, x1, x2, x3):

        # Layer 2
        mm1 = forge.op.Matmul("mm1", x1, self.train_param1)
        # (1, 16, 64, 210) x (1, 16, 210, 78) -> (1, 16, 64, 78)
        mm2 = forge.op.Matmul("mm2", x2, self.train_param2)
        # (1, 16, 70, 64) x (1, 16, 64, 36) -> (1, 16, 70, 36)
        mm3 = forge.op.Matmul("mm3", x3, self.train_param3)
        # (1, 16, 240, 512) x (1, 16, 512, 64) -> (1, 16, 240, 64)

        # Layer 3
        tr1 = forge.op.Transpose("tr1", self.train_param1, 3, 2)
        # (1, 16, 210, 78) -> (1, 16, 78, 210)
        mm4 = forge.op.Matmul("mm4", mm1, tr1)
        # (1, 16, 64, 78) x (1, 16, 78, 210) -> (1, 16, 64, 210)
        inter1 = forge.op.Matmul("iter1", x2, self.train_param7)
        # (1, 16, 70, 64) x (1, 16, 64, 240) -> (1, 16, 70, 240)
        mm5 = forge.op.Matmul("mm5", inter1, x3)
        # (1, 16, 70, 240) x (1, 16, 240, 512) -> (1, 16, 70, 512)
        inter2 = forge.op.Matmul("inter2", mm2, self.train_param8)
        # (1, 16, 70, 36) x (1, 16, 36, 240) -> (1, 16, 70, 240)
        mm6 = forge.op.Matmul("mm6", inter2, mm3)
        # (1, 16, 70, 240) x (1, 16, 240, 64) -> (1, 16, 70, 64)

        # Layer 4
        tr2 = forge.op.Transpose("tr2", mm4, 3, 2)
        # (1, 16, 64, 210) -> (1, 16, 210, 64)
        tr3 = forge.op.Transpose("tr3", x2, 3, 2)
        # (1, 16, 70, 64) -> (1, 16, 64, 70)
        mm7 = forge.op.Matmul("mm7", tr2, tr3)
        # (1, 16, 210, 64) x (1, 16, 64, 70) -> (1, 16, 210, 70)
        inter6 = forge.op.Matmul("inter6", mm4, self.train_param12)
        # (1, 16, 64, 210) x (1, 16, 210, 70) -> (1, 16, 64, 70)
        mm8 = forge.op.Matmul("mm8", inter6, mm5)
        # (1, 16, 64, 70) x (1, 16, 70, 512) -> (1, 16, 64, 512)
        mm9 = forge.op.Matmul("mm9", mm5, self.train_param4)
        # (1, 16, 70, 512) x (1, 16, 512, 64) -> (1, 16, 70, 64)
        tr4 = forge.op.Transpose("tr4", self.train_param3, 3, 2)
        # (1, 16, 512, 64) -> (1, 16, 64, 512)
        mm10 = forge.op.Matmul("mm10", mm6, tr4)
        # (1, 16, 70, 64) x (1, 16, 64, 512) -> (1, 16, 70, 512)

        # Layer 5
        inter3 = forge.op.Matmul("inter3", mm7, self.train_param9)
        # (1, 16, 210, 70) x (1, 16, 70, 64) -> (1, 16, 210, 64)
        mm11 = forge.op.Matmul("mm11", inter3, mm8)
        # (1, 16, 210, 64) x (1, 16, 64, 512) -> (1, 16, 210, 512)
        inter4 = forge.op.Matmul("inter4", x1, self.train_param10)
        # (1, 16, 64, 210) x (1, 16, 210, 240) -> (1, 16, 64, 240)
        mm12 = forge.op.Matmul("mm12", inter4, x3)
        # (1, 16, 64, 240) x (1, 16, 240, 512) -> (1, 16, 64, 512)
        tr5 = forge.op.Transpose("tr5", mm6, 3, 2)
        # (1, 16, 70, 64) -> (1, 16, 64, 70)
        mm13 = forge.op.Matmul("mm13", tr5, mm10)
        # (1, 16, 64, 70) x (1, 16, 70, 512) -> (1, 16, 64, 512)

        # Layer 6
        mm14 = forge.op.Matmul("mm14", x1, mm11)
        # (1, 16, 64, 210) x (1, 16, 210, 512) -> (1, 16, 64, 512)
        tr6 = forge.op.Transpose("tr6", mm12, 3, 2)
        # (1, 16, 64, 512) -> (1, 16, 512, 64)
        mm15 = forge.op.Matmul("mm15", tr6, mm13)
        # (1, 16, 512, 64) x (1, 16, 64, 512) -> (1, 16, 512, 512)
        mm16 = forge.op.Matmul("mm16", mm9, self.train_param5)
        # (1, 16, 70, 64) x (1, 16, 64, 256) -> (1, 16, 70, 256)

        # Layer 7
        mm17 = forge.op.Matmul("mm17", mm14, mm15)
        # (1, 16, 64, 512) x (1, 16, 512, 512) -> (1, 16, 64, 512)
        mm18 = forge.op.Matmul("mm18", mm15, self.train_param6)
        # (1, 16, 512, 512) x (1, 16, 512, 50) -> (1, 16, 512, 50)
        tr7 = forge.op.Transpose("tr7", mm16, 3, 2)
        # (1, 16, 70, 256) -> (1, 16, 256, 70)
        mm19 = forge.op.Matmul("mm19", tr7, mm10)
        # (1, 16, 256, 70) x (1, 16, 70, 512) -> (1, 16, 256, 512)

        # Layer 8
        mm20 = forge.op.Matmul("mm20", mm17, mm18)
        # (1, 16, 64, 512) x (1, 16, 512, 50) -> (1, 16, 64, 50)

        inter5 = forge.op.Matmul("inter5", mm18, self.train_param11)
        # (1, 16, 512, 50) x (1, 16, 50, 256) -> (1, 16, 512, 256)
        mm21 = forge.op.Matmul("mm21", inter5, mm19)
        # (1, 16, 512, 256) x (1, 16, 256, 512) -> (1, 16, 512, 512)

        # Layer 9
        inter7 = forge.op.Matmul("inter7", mm20, self.train_param13)
        # (1, 16, 64, 50) x (1, 16, 50, 512) -> (1, 16, 64, 512)
        mm22 = forge.op.Matmul("mm22", inter7, mm21)
        # (1, 16, 64, 512) x (1, 16, 512, 512) -> (1, 16, 64, 512)

        return mm22

    def values(self):
        return [item.value() for item in self.inputs]
