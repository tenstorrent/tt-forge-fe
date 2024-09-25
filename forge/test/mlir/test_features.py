# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import pytest

import pytest
import torch
from torch import nn

import forge
from forge.op.eval.common import compare_with_golden_pcc

def test_multiple_inputs():
    class MultipleInputs(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b, c):
            return a + b + c
        
    inputs = [torch.rand(1, 32, 32), torch.rand(1, 32, 32), torch.rand(1, 32, 32)]
    
    framework_model = MultipleInputs()
    fw_out = framework_model(*inputs)
    
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)
    
    co_out = [co.to("cpu") for co in co_out]
    assert [compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)]


@pytest.mark.parametrize("a_shape, b_shape, c_shape", [
    ((1, 1, 32, 64), (1, 1, 64, 128), (1, 1, 128, 32)), 
])
def test_input_order(a_shape, b_shape, c_shape):
    class InputOrderWithConstants(nn.Module):
        def __init__(self):
            super().__init__()
            self.const1 = torch.rand(1, 1, 32, 32)
            self.const2 = torch.rand(1, 1, 32, 32)

        def forward(self, a, b, c):
            x = torch.matmul(a, b)
            x = torch.matmul(x, c)
            x = x + self.const1
            x = x * self.const2
            return x

    a = torch.rand(*a_shape)
    b = torch.rand(*b_shape)
    c = torch.rand(*c_shape)

    framework_model = InputOrderWithConstants()
    fw_out = framework_model(a, b, c)

    compiled_model = forge.compile(framework_model, sample_inputs=[a, b, c])
    co_out = compiled_model(a, b, c)

    assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0][0], pcc=0.99)


def test_differently_ranked_matmul():

    class DifferentRankMatmul(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return a @ b

    a = torch.randn(1, 1, 64, 128)
    b = torch.randn(128, 256)

    framework_model = DifferentRankMatmul()
    fw_out = framework_model(a, b)

    compiled_model = forge.compile(framework_model, sample_inputs=[a, b])
    co_out = compiled_model(a, b)[0]
    
    assert co_out.shape == fw_out.shape
    assert compare_with_golden_pcc(golden=fw_out, calculated=co_out, pcc=0.99)



