# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 1
#   LeakyRelu operators defined by Forge API
#   These kinds of tests test only single specific operator through different Forge architectures
#


import torch
from torch.distributions import Normal

import forge
import forge.op
import forge.op.nn as nn
from forge import ForgeModule, Tensor


class ForgeLeakyReluTest(ForgeModule):
    """
    Forge Test 1

    """

    INPUTS_RANGE_MIN = -1.0
    INPUTS_RANGE_MAX = 1.0
    INPUTS_DISTRIBUTION = Normal

    WEIGHTS_RANGE_MIN = -1.0
    WEIGHTS_RANGE_MAX = 1.0
    WEIGHTS_DISTRIBUTION = Normal

    def __init__(self, shape, alpha):
        super().__init__("Forge Test 1")

        self.testname = "Operator LeakyRelu, Test 1"
        self.shape = shape
        self.alpha = alpha

        self.train_param = forge.Parameter(*self.shape, requires_grad=True)

        input = ForgeLeakyReluTest.INPUTS_DISTRIBUTION(
            ForgeLeakyReluTest.INPUTS_RANGE_MIN, ForgeLeakyReluTest.INPUTS_RANGE_MAX
        ).sample(self.shape)
        self.inputs = [Tensor.create_from_torch(input)]

        weights = ForgeLeakyReluTest.WEIGHTS_DISTRIBUTION(
            ForgeLeakyReluTest.WEIGHTS_RANGE_MIN, ForgeLeakyReluTest.WEIGHTS_RANGE_MAX
        ).sample(self.shape)
        weights.requires_grad = True
        self.set_parameter("train_param", weights)

    def forward(self, x):

        # Layer 2
        mul = forge.op.Multiply("mul", x, self.train_param)

        # Layer 3
        lrelu = forge.op.LeakyRelu("lrelu", mul, alpha=self.alpha)

        return lrelu
