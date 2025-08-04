# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from loguru import logger
from torch import nn

import forge
from forge.verify.verify import verify


def test_sig_1():
    class sig_1(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            return torch.sigmoid(a)

    a = torch.randn(1, 1, 256, 256)

    logger.info("a.shape={}", a.shape)
    logger.info("a.dtype={}", a.dtype)
    logger.info("a={}", a)

    has_inf = torch.isinf(a).any()
    logger.info("ip Contains inf:={}", has_inf.item())

    inputs = [a]

    framework_model = sig_1()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)
