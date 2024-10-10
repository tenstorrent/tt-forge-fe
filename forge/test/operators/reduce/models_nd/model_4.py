# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 4
#   Reduce operators defined by Forge API
#   These kinds of tests test only single specific operator through different Forge architectures
#


from pickletools import pyunicode
import random
import torch

import forge

from forge import ForgeModule, Tensor


class ForgeReduceTest(ForgeModule):
    """
        Forge Test 4

    Args:
        operator (function): Forge reduce operator.
        opname (str): Operation name (e.g. reduce_sum, reduce_avg, ...).
                      This name test uses to generate names of operation nodes in a graph/model.
    """

    def __init__(self, operator, opname, shape, dim, keepdim):
        super().__init__("Forge Test 4")

        assert hasattr(shape, "__iter__"), "Shape must be iterable"
        assert dim < len(shape), "Dimension out of the shape"
        assert dim >= 0, "Dimension cant' be negative"

        self.operator = operator
        self.opname = opname
        self.testname = "Operator " + opname + " Test 4"
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
        pass

        # Layer 2
        mul1 = forge.op.Multiply("mul1", x1, x2)
        mul2 = forge.op.Multiply("mul2", self.train_param1, self.train_param2)
        mul3 = forge.op.Multiply("mul3", self.train_param2, self.train_param3)
        mul4 = forge.op.Multiply("mul4", x2, x3)

        # Layer 3
        operands = [x1, self.train_param1, mul1, mul2, x2, self.train_param2, mul3, mul4, x3, self.train_param3]
        reds = []
        for i in range(len(operands)):
            reds.append(self.operator(self.opname + str(i + 1), operands[i], self.dim, self.keepdim))
        self.shape = reds[0].shape

        # Layer 4
        mul5 = forge.op.Multiply("mul5", reds[0], reds[1])
        mul6 = forge.op.Multiply("mul6", reds[1], reds[2])
        mul7 = forge.op.Multiply("mul7", reds[4], reds[5])
        mul8 = forge.op.Multiply("mul8", reds[7], reds[8])

        # Layer 5
        hvs1 = forge.op.Heaviside("hvs1", mul6, reds[3])
        hvs2 = forge.op.Heaviside("hvs2", mul7, reds[6])
        hvs3 = forge.op.Heaviside("hvs3", mul8, reds[9])

        # Layer 6
        max1 = forge.op.Max("max1", mul5, hvs1)
        max2 = forge.op.Multiply("max2", reds[4], hvs2)
        max3 = forge.op.Multiply("max3", reds[7], hvs3)

        # Layer 7
        add1 = forge.op.Add("add1", reds[1], max1)
        add2 = forge.op.Add("add2", reds[3], max2)
        add3 = forge.op.Add("add3", reds[6], max3)

        if self.keepdim or len(self.shape) > 0:
            self.dim = random.randint(0, len(self.shape) - 1)
            # Layer 8
            lenop = len(operands)
            operands = [add1, reds[3], add2, reds[6], add3, reds[9]]
            preds = []
            for i in range(len(operands)):
                preds.append(self.operator(self.opname + str(i + 1 + lenop), operands[i], self.dim, self.keepdim))
            # Layer 9
            mul9 = forge.op.Multiply("mul9", preds[0], preds[1])
            mul10 = forge.op.Multiply("mul10", preds[2], preds[3])
            mul11 = forge.op.Multiply("mul11", preds[4], preds[5])
        else:
            # Layer 9
            mul9 = forge.op.Multiply("mul9", add1, reds[3])
            mul10 = forge.op.Multiply("mul10", add2, reds[6])
            mul11 = forge.op.Multiply("mul11", add3, reds[9])

        # Layer 10
        mul12 = forge.op.Multiply("mul12", mul9, mul10)

        # Layer 11
        add4 = forge.op.Add("add4", mul12, mul11)

        return add4

    def values(self):
        return [item.value() for item in self.inputs]
