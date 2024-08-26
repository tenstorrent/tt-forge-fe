# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import pytest

import pytest
import torch
from torch import nn

import pybuda
from pybuda.op.eval.common import compare_with_golden_pcc

def test_add():
    class Add(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return a + b
        
    inputs = [torch.rand(1, 32, 32), torch.rand(1, 32, 32)]
    
    framework_model = Add()
    fw_out = framework_model(*inputs)
    
    compiled_model = pybuda.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)
    
    co_out = [co.to("cpu") for co in co_out]
    assert [compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)]


def test_subtract():
    class Subtract(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return a - b
        
    inputs = [torch.rand(1, 32, 32), torch.rand(1, 32, 32)]
    
    framework_model = Subtract()
    fw_out = framework_model(*inputs)
    
    compiled_model = pybuda.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)
    
    co_out = [co.to("cpu") for co in co_out]
    assert [compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)]


def test_multiply():
    class Multiply(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return a * b
        
    inputs = [torch.rand(1, 32, 32), torch.rand(1, 32, 32)]
    
    framework_model = Multiply()
    fw_out = framework_model(*inputs)
    
    compiled_model = pybuda.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)
    
    co_out = [co.to("cpu") for co in co_out]
    assert [compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)]


def test_relu():
    class ReLU(nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()

        def forward(self, a):
            return self.relu(a)
        
    inputs = [torch.rand(1, 32)]
    
    framework_model = ReLU()
    fw_out = framework_model(*inputs)
    
    compiled_model = pybuda.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)
    
    co_out = [co.to("cpu") for co in co_out]
    assert [compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)]

@pytest.mark.skip(reason="This is not ready yet")
def test_linear():
    class Linear(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(20, 30, bias=True)

        def forward(self, a):
            return self.l1(a)
        
    inputs = [torch.rand(1, 128, 20)]
    
    framework_model = Linear()
    fw_out = framework_model(*inputs)
    
    compiled_model = pybuda.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)
    
    co_out = [co.to("cpu") for co in co_out]
    assert [compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)]


def test_softmax():
    class Softmax(nn.Module):
        def __init__(self):
            super().__init__()
            self.softmax = nn.Softmax(dim=1)

        def forward(self, a):
            return self.softmax(a)
        
    inputs = [torch.rand(1, 128)]
    
    framework_model = Softmax()
    fw_out = framework_model(*inputs)
    
    compiled_model = pybuda.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)
    
    co_out = [co.to("cpu") for co in co_out]
    assert [compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)]

@pytest.mark.parametrize("input_shape", [(1,32,32), (1,64,64), (1,128,128,128)], ids=["32","64","128"])
@pytest.mark.parametrize("dim", [-1,-2], ids=["-1","-2"])
def test_reduce_sum(input_shape, dim):
    class ReduceSum(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            # reduce is supported on tt-metal only with keepdim=True
            return torch.sum(a, dim=dim, keepdim=True)
        
    inputs = [torch.rand(input_shape)]
    
    framework_model = ReduceSum()
    fw_out = framework_model(*inputs)
    
    compiled_model = pybuda.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)
    
    co_out = [co.to("cpu") for co in co_out]
    assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)

@pytest.mark.parametrize("input_shape", [(1,32,32), (1,64,64), (1,128,128,128)], ids=["32","64","128"])
@pytest.mark.parametrize("dim", [-1,-2], ids=["-1","-2"])
def test_reduce_mean(input_shape, dim):
    class ReduceMean(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            # reduce is supported on tt-metal only with keepdim=True
            return torch.mean(a, dim=1, keepdim=True)
        
    inputs = [torch.rand(1, 32, 32)]
    
    framework_model = ReduceMean()
    fw_out = framework_model(*inputs)
    
    compiled_model = pybuda.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)
    
    co_out = [co.to("cpu") for co in co_out]
    assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)


@pytest.mark.parametrize("batch_size", [1, 7, 32])
@pytest.mark.parametrize("outer_dim_x", [7, 32, 41, 64])
@pytest.mark.parametrize("outer_dim_y", [7, 32, 41, 64])
@pytest.mark.parametrize("inner_dim", [1, 7, 32, 41, 64])
def test_matmul(batch_size, outer_dim_x, outer_dim_y, inner_dim):
    class Matmul(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.matmul(x, y)
        
    inputs = [
        torch.rand(batch_size, outer_dim_x, inner_dim),
        torch.rand(batch_size, inner_dim, outer_dim_y),
    ]

    framework_model = Matmul()
    fw_out = framework_model(*inputs)
    
    compiled_model = pybuda.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)
    
    co_out = [co.to("cpu") for co in co_out]
    assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)


@pytest.mark.parametrize("x_shape", [7, 32, 41])
@pytest.mark.parametrize("y_shape", [7, 32, 41])
@pytest.mark.parametrize("dim", [1, 2])
def test_mean(x_shape, y_shape, dim):
    if dim == 1:
        pytest.skip("FFE: Unsupported squeeze operation")
    if dim == 2:
        # Note: Some tests are passing when run in group, while failing when running individually
        pytest.skip("TTNN: Tensor layout bugs")
    
    class Mean(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.mean(x, dim=dim)
        
    inputs = [
        torch.rand(1, x_shape, y_shape),
    ]

    framework_model = Mean()
    fw_out = framework_model(*inputs)
    
    compiled_model = pybuda.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)
    
    co_out = [co.to("cpu") for co in co_out]
    assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)


@pytest.mark.parametrize("x_shape", [7, 32, 41])
@pytest.mark.parametrize("y_shape", [7, 32, 41])
def test_sqrt(x_shape, y_shape):    
    pytest.skip("FFE: Requires MLIR uplift")
    class Sqrt(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.sqrt(x)
        
    inputs = [
        torch.rand(1, x_shape, y_shape),
    ]

    framework_model = Sqrt()
    fw_out = framework_model(*inputs)
    
    compiled_model = pybuda.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)
    
    co_out = [co.to("cpu") for co in co_out]
    assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)
