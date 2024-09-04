# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 7
#   Unary element-wise operators defined by Forge API
#   These kinds of tests test only single specific operator through different Forge architectures
# 


import torch

import forge

from forge import ForgeModule, Tensor


class ForgeElementWiseUnaryTest(ForgeModule):
    """
        Forge Test 7

        In this test we have 6 unary operations, and 3 input tensors and 6 trainable variables.

    Args:
        operator (function): Forge unary element-wise operator.
        opname (str): Operation name (e.g. exp, sqrt, gelu, ...).
                      This name test use to generate names of operation nodes in a graph/model.
        shape (tuple, list): Shape of the input tensors.
    """

    def __init__(self, operator, opname, shape, **kwargs):
        super().__init__("Forge Test 3")
        self.operator = operator
        self.opname = opname
        self.testname = "Operator " + opname + " Test 3"
        self.shape = shape
        self.kwargs = kwargs

        self.train_param1 = forge.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = forge.Parameter(*self.shape, requires_grad=True)
        self.train_param3 = forge.Parameter(*self.shape, requires_grad=True)

        self.train_param4 = forge.Parameter(*self.shape, requires_grad=True)
        self.train_param5 = forge.Parameter(*self.shape, requires_grad=True)
        self.train_param6 = forge.Parameter(*self.shape, requires_grad=True)

        self.train_param7 = forge.Parameter(*self.shape, requires_grad=True)
        self.train_param8 = forge.Parameter(*self.shape, requires_grad=True)
        self.train_param9 = forge.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape)) for i in range(3)]

        for i in range(1, 10):
            self.set_parameter("train_param{}".format(i), torch.rand(*self.shape, requires_grad=True))

    def forward(self, x1, x2, x3):

        # Layer 2
        mul1 = forge.op.Multiply("mul1", x1, self.train_param1)
        mul2 = forge.op.Multiply("mul2", x2, self.train_param2)
        mul3 = forge.op.Multiply("mul3", x3, self.train_param3)

        # Layer 3
        un1 = self.operator(self.opname + "1", mul1, **self.kwargs)
        un2 = self.operator(self.opname + "2", mul2, **self.kwargs)
        un3 = self.operator(self.opname + "3", mul3, **self.kwargs)

        # Layer 4
        mul4 = forge.op.Multiply("mul4", un1, self.train_param4)
        mul5 = forge.op.Multiply("mul5", un2, self.train_param5)
        mul6 = forge.op.Multiply("mul6", un3, self.train_param6)

        # Layer 5
        un4 = self.operator(self.opname + "4", mul4, **self.kwargs)
        un5 = self.operator(self.opname + "5", mul5, **self.kwargs)
        un6 = self.operator(self.opname + "6", mul6, **self.kwargs)

        # Layer 6
        mul7 = forge.op.Multiply("mul7", un4, self.train_param7)
        mul8 = forge.op.Multiply("mul8", un5, self.train_param8)
        mul9 = forge.op.Multiply("mul9", un6, self.train_param9)

        return un4, un5, un6, mul9, mul8, mul7

    def values(self):
        return [item.value() for item in self.inputs]
