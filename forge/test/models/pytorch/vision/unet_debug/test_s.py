# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from loguru import logger
from torch import nn

import forge
from forge.verify.verify import verify


def test_sig():
    class sig(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            return torch.sigmoid(a)

    a = torch.load("sig_ip.pt")

    logger.info("a.shape={}", a.shape)
    logger.info("a.dtype={}", a.dtype)
    logger.info("a={}", a)

    has_inf = torch.isinf(a).any()
    logger.info("ip Contains inf:={}", has_inf.item())

    inputs = [a]

    framework_model = sig()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)
