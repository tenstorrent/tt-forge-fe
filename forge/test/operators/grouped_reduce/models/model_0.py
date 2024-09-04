# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 0
#   Grouped reduce operators defined by Forge API
#   These kinds of tests test only single specific operator through different Forge architectures
# 


import torch

import forge

from forge import ForgeModule, Tensor


class ForgeReduceTest(ForgeModule):
    """
        Forge Test 0

    Args:
        operator (function): Forge reduce operator.
        opname (str): Operation name (e.g. reduce_sum, reduce_avg, ...).
                      This name test uses to generate names of operation nodes in a graph/model.
    """

    def __init__(self, operator, opname, shape, dim, groups, keep_dims):
        super().__init__("Forge Test 0")
        self.operator = operator
        self.opname = opname
        self.testname = "Operator " + opname + " Test 0"
        self.shape = shape
        self.dim = dim 
        self.groups = groups
        self.keep_dims = keep_dims
        self.train_param = forge.Parameter(torch.randn(*self.shape), requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape))]

    def forward(self, x):
        mul = forge.op.Multiply("mul", x, self.train_param)
        red = self.operator(self.opname, mul, self.dim, self.groups, self.keep_dims)

        return red

    def values(self):
        return [item.value() for item in self.inputs]