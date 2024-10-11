# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 0
#   Reduce operators defined by Forge API
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

    def __init__(self, operator, opname, shape):
        super().__init__("Forge Test 0")
        self.operator = operator
        self.opname = opname
        self.testname = "Operator " + opname + " Test 0"
        self.shape = shape
        self.train_param = forge.Parameter(*self.shape, requires_grad=True)

        self.inputs = [Tensor.create_from_torch(torch.rand(*self.shape))]
        self.set_parameter("train_param", torch.rand(*self.shape, requires_grad=True))

    def forward(self, x):
        mul = forge.op.Multiply("mul", x, self.train_param)
        #         # (W, Z, R, C) * (W, Z, R, C) --> (W, Z, R, C)
        red = self.operator(self.opname, mul, 2)
        # (W, Z, R, C) --> (W, Z, 1, C)
        # red2 = self.operator(self.opname + "2", mul, 3)
        #         # (W, Z, R, C) --> (W, Z, R, 1)
        # mm = forge.op.Matmul("mm", red2, red1)
        #         # (W, Z, R, 1) x (W, Z, 1, C) --> (W, Z, R, C)
        return red

    def values(self):
        return [item.value() for item in self.inputs]
