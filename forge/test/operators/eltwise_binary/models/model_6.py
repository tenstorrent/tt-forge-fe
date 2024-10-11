# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 6
#   Binary element-wise operators defined by Forge API
#   These kinds of tests test only single specific operator through different Forge architectures
#


import torch

import forge

from forge import ForgeModule, Tensor


class ForgeElementWiseBinaryTest(ForgeModule):
    """
        Forge Test 6

        In this test we have 13 operators with 4 input operands and 4 trainable operands.

    Args:
        operator (function): Forge binary element-wise operator.
        opname (str): Operation name (e.g. add, mul, sub, ...).
                      This name test use to generate names of operation nodes in a graph/model.
    """

    def __init__(self, operator, opname, shape):
        super().__init__("Forge Test 6")
        self.operator = operator
        self.opname = opname
        self.testname = "Operator " + opname + " Test 6"
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
        op1 = self.operator(self.opname + "1", x1, self.train_param1)
        op2 = self.operator(self.opname + "2", x2, self.train_param2)
        op3 = self.operator(self.opname + "3", x3, self.train_param3)
        op4 = self.operator(self.opname + "4", x4, self.train_param4)

        # Layer 3
        op5 = self.operator(self.opname + "5", op1, op2)
        op6 = self.operator(self.opname + "6", op2, op3)
        op7 = self.operator(self.opname + "7", op2, op4)

        # Layer 4
        op8 = self.operator(self.opname + "8", op5, op6)
        op9 = self.operator(self.opname + "9", op5, op7)
        op10 = self.operator(self.opname + "10", op6, op7)

        # Layer 5
        op11 = self.operator(self.opname + "11", op8, op9)
        op12 = self.operator(self.opname + "12", op9, op10)

        # Layer 6
        op13 = self.operator(self.opname + "13", op11, op12)

        return op13

    def values(self):
        return [item.value() for item in self.inputs]
