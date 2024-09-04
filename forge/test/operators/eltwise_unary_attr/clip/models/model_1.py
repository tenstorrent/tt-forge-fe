# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
#   Test 1 
#   Clip operators defined by Forge API
#   These kinds of tests test only single specific operator through different Forge architectures
# 


import torch
from torch.distributions import Normal

import forge
import forge.op
import forge.op.nn as nn

from forge import ForgeModule, Tensor


class ForgeClipTest(ForgeModule):
    """
        Forge Test 1

    """

    INPUTS_RANGE_MIN = -1.0
    INPUTS_RANGE_MAX = 1.0
    INPUTS_DISTRIBUTION = Normal

    WEIGHTS_RANGE_MIN = -1.0
    WEIGHTS_RANGE_MAX = 1.0
    WEIGHTS_DISTRIBUTION = Normal

    def __init__(
        self,
        shape,
        min_value,
        max_value
    ):
        super().__init__("Forge Test 1")

        self.testname = "Operator Clip, Test 1"
        self.shape = shape
        self.min_value = min_value
        self.max_value = max_value
        
        self.train_param = forge.Parameter(*self.shape, requires_grad=True)

        input = ForgeClipTest.INPUTS_DISTRIBUTION(
            ForgeClipTest.INPUTS_RANGE_MIN, 
            ForgeClipTest.INPUTS_RANGE_MAX).sample(self.shape)
        self.inputs = [Tensor.create_from_torch(input)]

        weights = ForgeClipTest.WEIGHTS_DISTRIBUTION(
            ForgeClipTest.WEIGHTS_RANGE_MIN, 
            ForgeClipTest.WEIGHTS_RANGE_MAX).sample(self.shape)
        weights.requires_grad = True
        self.set_parameter("train_param", weights)

    def forward(self, x):

        # Layer 2
        mul = forge.op.Multiply("mul", x, self.train_param)

        # Layer 3
        clip = forge.op.Clip("clip", mul, min=self.min_value, max=self.max_value)

        return clip
