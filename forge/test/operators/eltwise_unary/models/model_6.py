# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 6
#   Unary element-wise operators defined by Forge API
#   These kinds of tests test only single specific operator through different Forge architectures
# 


import torch

import forge

from forge import ForgeModule, Tensor


class BudaElementWiseUnaryTest(ForgeModule):
    """
        Buda Test 6

        In this test we have 15 unary operations, and 3 input tensors and 6 trainable variables.

    Args:
        operator (function): Forge unary element-wise operator.
        opname (str): Operation name (e.g. exp, sqrt, gelu, ...).
                      This name test use to generate names of operation nodes in a graph/model.
        shape (tuple, list): Shape of the input tensors.
    """

    def __init__(self, operator, opname, shape, **kwargs):
        super().__init__("Buda Test 6")
        self.operator = operator
        self.opname = opname
        self.testname = "Operator " + opname + " Test 6"
        self.shape = shape
        self.kwargs = kwargs

        self.train_param1 = forge.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = forge.Parameter(*self.shape, requires_grad=True)
        self.train_param3 = forge.Parameter(*self.shape, requires_grad=True)

        self.train_param4 = forge.Parameter(*self.shape, requires_grad=True)
        self.train_param5 = forge.Parameter(*self.shape, requires_grad=True)
        self.train_param6 = forge.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape)) for i in range(3)]

        for i in range(1, 7):
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
        add1 = forge.op.Add("add1", un1, un2)
        mul1 = forge.op.Multiply("mul1", un3, un4)
        add2 = forge.op.Add("add2", un5, un6)

        # Layer 4
        mul2 = forge.op.Multiply("mul2", add1, self.train_param6)
        mul3 = forge.op.Multiply("mul3", mul1, self.train_param5)
        mul4 = forge.op.Multiply("mul4", add2, self.train_param4)

        # Layer 5
        un7 = self.operator(self.opname + "7", mul2, **self.kwargs)
        un8 = self.operator(self.opname + "8", mul3, **self.kwargs)
        un9 = self.operator(self.opname + "9", mul4, **self.kwargs)

        # Layer 6
        mul5 = forge.op.Multiply("mul5", un7, self.train_param4)
        mul6 = forge.op.Multiply("mul6", un8, self.train_param5)
        mul7 = forge.op.Multiply("mul7", un9, self.train_param6)

        # Layer 7
        un10 = self.operator(self.opname + "10", mul5, **self.kwargs)
        un11 = self.operator(self.opname + "11", mul6, **self.kwargs)
        un12 = self.operator(self.opname + "12", mul7, **self.kwargs)

        # Layer 8
        mul8 = forge.op.Multiply("mul8", un10, self.train_param1)
        mul9 = forge.op.Multiply("mul9", un11, self.train_param2)
        mul10 = forge.op.Multiply("mul10", un12, self.train_param3)

        # Layer 9
        un14 = self.operator(self.opname + "14", mul10, **self.kwargs)
        mul11 = forge.op.Multiply("mul11", mul8, mul9)

        # Layer 10
        un13 = self.operator(self.opname + "13", mul11, **self.kwargs)

        # Layer 11
        mul12 = forge.op.Multiply("mul12", un13, un14)

        # Layer 12
        un15 = self.operator(self.opname + "15", mul12, **self.kwargs)

        return un15

    def values(self):
        return [item.value() for item in self.inputs]