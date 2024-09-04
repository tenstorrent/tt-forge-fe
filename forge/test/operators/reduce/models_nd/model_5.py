# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 5
#   Reduce operators defined by Forge API
#   These kinds of tests test only single specific operator through different Forge architectures
# 


from audioop import add
import random
import torch

import forge

from forge import ForgeModule, Tensor



class ForgeReduceTest(ForgeModule):
    """
        Forge Test 5

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
        super().__init__("Forge Test 5")

        assert hasattr(shape, '__iter__'), "Shape must be iterable"
        assert dim < len(shape), "Dimension out of the shape"
        assert dim >= 0, "Dimension cant' be negative"

        self.operator = operator
        self.opname = opname
        self.testname = "Operator " + opname + " Test 5"
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
        mul2 = forge.op.Multiply("mul2", x2, self.train_param2)
        mul3 = forge.op.Multiply("mul3", x3, self.train_param3)

        # Layer 3
        operands = [mul1, self.train_param1, mul2, self.train_param2, mul3, self.train_param3]
        reds = []
        for i in range(len(operands)):
            reds.append(self.operator(self.opname + str(i + 1), self.dim, self.keepdim))
        self.shape = reds[0].shape

        # Layer 4
        add1 = forge.op.Add("add1", reds[0], reds[1])
        mul4 = forge.op.Multiply("mul4", reds[1], reds[2])
        add2 = forge.op.Add("add2", reds[2], reds[3])
        mul5 = forge.op.Multiply("mul5", reds[3], reds[4])
        add3 = forge.op.Add("add3", reds[4], reds[5])

        if self.keepdim or len(self.shape) > 0:
            self.dim = random.randint(0, len(self.shape) - 1)
            # Layer 5
            lenop = len(operands)
            operands = [add1, reds[1], mul4, add2, reds[3], mul5, add3, reds[5]]
            preds = []
            for i in range(len(operands)):
                preds.append(self.operator(self.opname + str(i + 1 + lenop), self.dim, self.keepdim))
            self.shape = preds[0].shape
            # Layer 6
            add4 = forge.op.Add("add4", preds[0], preds[1])
            sub1 = forge.op.Subtract("sub1", preds[2], preds[3])
            max1 = forge.op.Max("max1", preds[4], preds[5])
            sub2 = forge.op.Subtract("sub2", preds[6], preds[7])
        else:
            # Layer 6
            add4 = forge.op.Add("add4", add1, reds[1])
            sub1 = forge.op.Subtract("sub1", mul4, add2)
            max1 = forge.op.Max("max1", reds[3], mul5)
            sub2 = forge.op.Subtract("sub2", add3, reds[5])
        
        if self.keepdim or len(self.shape) > 0:
            self.dim = random.randint(0, len(self.shape) - 1)
            # Layer 7
            lenop += len(operands)
            operands = [reds[0], add4, sub1, max1, reds[4], sub2]
            preds = []
            for i in range(len(operands)):
                preds.append(self.operator(self.opname + str(i + 1 + lenop), self.dim, self.keepdim))
            # Layer 8
            mul6 = forge.op.Multiply("mul6", preds[0], preds[1])
            mul7 = forge.op.Multiply("mul7", preds[2], preds[3])
            mul8 = forge.op.Multiply("mul8", preds[4], preds[5])
        else:
            # Layer 8
            mul6 = forge.op.Multiply("mul6", reds[0], add4)
            mul7 = forge.op.Multiply("mul7", sub1, max1)
            mul8 = forge.op.Multiply("mul8", reds[4], sub2)

        return mul6, mul7, mul8

    def values(self):
        return [item.value() for item in self.inputs] 