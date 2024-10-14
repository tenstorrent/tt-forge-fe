# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import numpy as np

import torch
from torch import nn

import forge
from forge.op.eval.common import compare_with_golden_pcc

import os


@pytest.mark.parametrize("dims", [(1, 784), (4, 784), (16, 784), (32, 784), (64, 784)])
def test_matmul_bias_fusion(dims):
    class Linear(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(784, 10, bias=True)

        def forward(self, a):
            return self.l1(a)

    inputs = [torch.rand(dims)]

    framework_model = Linear()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out
    assert all([compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])
