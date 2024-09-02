# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 3
#   Reduce operators defined by Forge API
#   These kinds of tests test only single specific operator through different Forge architectures
# 


import random
import torch

import forge

from forge import ForgeModule, Tensor



class BudaReduceTest(ForgeModule):
    """
        Buda Test 3

    Args:
        operator (function): Forge reduce operator.
        opname (str): Operation name (e.g. reduce_sum, reduce_avg, ...).
                      This name test uses to generate names of operation nodes in a graph/model.
    """

    def __init__(
        self, 
        operator, 
        opname,
        shape,
        dim,
        keepdim):
        super().__init__("Buda Test 3")

        assert hasattr(shape, '__iter__'), "Shape must be iterable"
        assert dim < len(shape), "Dimension out of the shape"
        assert dim >= 0, "Dimension cant' be negative"

        self.operator = operator
        self.opname = opname
        self.testname = "Operator " + opname + " Test 3"
        self.shape = shape
        self.dim = dim
        self.keepdim = keepdim
        
        self.train_param1 = forge.Parameter(*self.shape, requires_grad=True)
        self.train_param2 = forge.Parameter(*self.shape, requires_grad=True)
        self.train_param3 = forge.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape)) for i in range(3)]
        for i in range(1, 4):
            self.set_parameter("train_param" + str(i), torch.rand(*self.shape, requires_grad=True))

    def forward(self, x1, x2, x3):

        # Layer 2
        mul1 = forge.op.Multiply("mul1", x1, self.train_param1)
        mul2 = forge.op.Multiply("mul2", x2, x3)
        mul3 = forge.op.Multiply("mul3", self.train_param2, self.train_param3)

        # Layer 3
        operands = [mul1, self.train_param1, mul2, mul3, x3, self.train_param3]
        reds = []
        for i in range(len(operands)):
            reds.append(self.operator(self.opname + str(i + 1), operands[i], self.dim, self.keepdim))
        self.shape = reds[0].shape

        # Layer 4
        mul4 = forge.op.Multiply("mul4", reds[0], reds[1])
        mul5 = forge.op.Multiply("mul5", reds[2], reds[4])
        mul6 = forge.op.Multiply("mul6", reds[3], reds[5])

        if self.keepdim or len(self.shape) > 0:
            self.dim = random.randint(0, len(self.shape) - 1)
            # Layer 5
            lenop = len(operands)
            operands = [mul4, reds[2], mul5, mul6]
            preds = []
            for i in range(len(operands)):
                preds.append(self.operator(self.opname + str(i + 1 + lenop), operands[i], self.dim, self.keepdim))
            # Layer 6
            mul7 = forge.op.Multiply("mul7", preds[0], preds[1])
            mul8 = forge.op.Multiply("mul8", preds[2], preds[3])
        else:
            # Layer 6
            mul7 = forge.op.Multiply("mul7", mul4, reds[2])
            mul8 = forge.op.Multiply("mul8", mul5, mul6)

        # Layer 7
        mul9 = forge.op.Multiply("mul9", mul7, mul8)

        return mul9

    def values(self):
        return [item.value() for item in self.inputs] 