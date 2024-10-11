# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 5
#   Unary element-wise operators defined by Forge API
#   These kinds of tests test only single specific operator through different Forge architectures
#


import torch

import forge

from forge import ForgeModule, Tensor


class ForgeElementWiseUnaryTest(ForgeModule):
    """
        Forge Test 5

        In this test we have 23 unary operations, and three input tensors and three trainable variables.

    Args:
        operator (function): Forge unary element-wise operator.
        opname (str): Operation name (e.g. exp, sqrt, gelu, ...).
                      This name test use to generate names of operation nodes in a graph/model.
        shape (tuple, list): Shape of the input tensors.
    """

    def __init__(self, operator, opname, shape, **kwargs):
        super().__init__("Forge Test 5")
        self.operator = operator
        self.opname = opname
        self.testname = "Operator " + opname + " Test 5"
        self.shape = shape
        self.kwargs = kwargs

        self.train_param1 = forge.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = forge.Parameter(*self.shape, requires_grad=True)
        self.train_param3 = forge.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape)) for i in range(3)]

        for i in range(1, 4):
            self.set_parameter("train_param{}".format(i), torch.rand(*self.shape, requires_grad=True))

    def forward(self, x1, x2, x3):

        # Layer 2
        un1 = self.operator(self.opname + "1", x1, **self.kwargs)
        un2 = self.operator(self.opname + "2", self.train_param1, **self.kwargs)
        un3 = self.operator(self.opname + "3", x2, **self.kwargs)
        un4 = self.operator(self.opname + "4", self.train_param2, **self.kwargs)
        un5 = self.operator(self.opname + "5", x3, **self.kwargs)
        un6 = self.operator(self.opname + "6", self.train_param3, **self.kwargs)

        # Layer 3
        add1 = forge.op.Add("add1", un4, un5)
        sub1 = forge.op.Subtract("sub1", un1, un3)
        add2 = forge.op.Add("add2", un2, un6)
        sub2 = forge.op.Subtract("sub2", un1, un4)
        add3 = forge.op.Add("add3", un3, un6)
        sub3 = forge.op.Subtract("sub3", un2, un5)

        # Layer 4
        un7 = self.operator(self.opname + "7", add1, **self.kwargs)
        un8 = self.operator(self.opname + "8", sub1, **self.kwargs)
        un9 = self.operator(self.opname + "9", add2, **self.kwargs)
        un10 = self.operator(self.opname + "10", sub2, **self.kwargs)
        un11 = self.operator(self.opname + "11", add3, **self.kwargs)
        un12 = self.operator(self.opname + "12", sub3, **self.kwargs)

        # Layer 5
        add4 = forge.op.Add("add4", un7, self.train_param1)
        mul1 = forge.op.Multiply("mul1", un8, un3)
        mul2 = forge.op.Multiply("mul2", un9, self.train_param2)
        mul3 = forge.op.Multiply("mul3", un10, x3)
        add5 = forge.op.Add("add5", un11, un6)
        sub4 = forge.op.Subtract("sub4", un12, self.train_param3)

        # Layer 6
        un13 = self.operator(self.opname + "13", add4, **self.kwargs)
        un14 = self.operator(self.opname + "14", mul1, **self.kwargs)
        un15 = self.operator(self.opname + "15", mul2, **self.kwargs)
        un16 = self.operator(self.opname + "16", mul3, **self.kwargs)
        un17 = self.operator(self.opname + "17", add5, **self.kwargs)
        un18 = self.operator(self.opname + "18", sub4, **self.kwargs)

        # Layer 7
        add6 = forge.op.Add("add6", un13, un14)
        add7 = forge.op.Add("add7", un14, un9)
        mul4 = forge.op.Multiply("mul4", un15, un10)
        add8 = forge.op.Add("add8", un16, un17)
        sub5 = forge.op.Subtract("sub5", un11, un18)

        # Layer 8
        add9 = forge.op.Add("add9", add6, add7)
        mul5 = forge.op.Multiply("mul5", un15, mul4)
        mul6 = forge.op.Multiply("mul6", add8, sub5)

        # Layer 9
        un19 = self.operator(self.opname + "19", add9, **self.kwargs)
        un20 = self.operator(self.opname + "20", mul6, **self.kwargs)

        # Layer 10
        mul7 = forge.op.Multiply("mul7", un19, mul5)
        mul8 = forge.op.Multiply("mul8", mul5, un20)

        # Layer 11
        un21 = self.operator(self.opname + "21", mul7, **self.kwargs)
        un22 = self.operator(self.opname + "22", mul8, **self.kwargs)

        # Layer 12
        add10 = forge.op.Add("add10", un21, un22)

        # Layer 13
        un23 = self.operator(self.opname + "23", add10, **self.kwargs)

        return un23

    def values(self):
        return [item.value() for item in self.inputs]
