# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 4
#   Unary element-wise operators defined by Forge API
#   These kinds of tests test only single specific operator through different Forge architectures
#


import torch

import forge

from forge import ForgeModule, Tensor


class ForgeElementWiseUnaryTest(ForgeModule):
    """
        Forge Test 4

        In this test we have 11 unary operations, and 2 input tensors and 2 trainable variables.

    Args:
        operator (function): Forge unary element-wise operator.
        opname (str): Operation name (e.g. exp, sqrt, gelu, ...).
                      This name test use to generate names of operation nodes in a graph/model.
        shape (tuple, list): Shape of the input tensors.
    """

    def __init__(self, operator, opname, shape, **kwargs):
        super().__init__("Forge Test 4")
        self.operator = operator
        self.opname = opname
        self.testname = "Operator " + opname + " Test 4"
        self.shape = shape
        self.kwargs = kwargs

        self.train_param1 = forge.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = forge.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape)) for i in range(2)]

        for i in range(1, 3):
            self.set_parameter("train_param{}".format(i), torch.rand(*self.shape, requires_grad=True))

    def forward(self, x1, x2):

        # Layer 2
        add1 = forge.op.Add("add1", x1, self.train_param1)
        mul1 = forge.op.Multiply("mul1", x2, self.train_param2)

        # Layer 3
        un1 = self.operator(self.opname + "1", add1, **self.kwargs)
        un2 = self.operator(self.opname + "2", self.train_param1, **self.kwargs)
        un3 = self.operator(self.opname + "3", mul1, **self.kwargs)
        un4 = self.operator(self.opname + "4", x2, **self.kwargs)

        # Layer 4
        sub1 = forge.op.Subtract("sub1", un1, un2)
        add2 = forge.op.Add("add2", un3, un4)

        # Layer 5
        un5 = self.operator(self.opname + "5", self.train_param2, **self.kwargs)
        un6 = self.operator(self.opname + "6", sub1, **self.kwargs)
        un7 = self.operator(self.opname + "7", x1, **self.kwargs)
        un8 = self.operator(self.opname + "8", add2, **self.kwargs)

        # Layer 6
        mul2 = forge.op.Multiply("mul2", un5, un6)
        add3 = forge.op.Add("add3", un7, un8)

        # Layer 7
        un9 = self.operator(self.opname + "9", mul2, **self.kwargs)
        un10 = self.operator(self.opname + "10", add3, **self.kwargs)
        add4 = forge.op.Add("add4", un9, un10)

        # Layer 8
        un11 = self.operator(self.opname + "11", add4, **self.kwargs)

        return un11

    def values(self):
        return [item.value() for item in self.inputs]
