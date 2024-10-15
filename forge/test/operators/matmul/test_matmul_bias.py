# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import numpy as np
import torch
from torch import nn

import forge
from forge.op.eval.common import compare_with_golden_pcc

INPUT_FEATURES = 784
OUTPUT_DIM = 10

# if batch is >16 fuse_matmul is applied
@pytest.mark.parametrize("batch_size", [1, 4, 16, 32, 64])
def test_matmul_bias(batch_size):
    class Linear(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(INPUT_FEATURES, OUTPUT_DIM, bias=True)

        def forward(self, a):
            return self.l1(a)

    # global param for fuse_bias
    os.environ["FORGE_FUSE_MATMUL_BIAS"] = "1"

    inputs = [torch.rand(batch_size, INPUT_FEATURES)]

    framework_model = Linear()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out
    assert all([compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])
