# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import pytest

import pytest
import torch
from torch import nn

import forge
from forge.op.eval.common import compare_with_golden_pcc, compare_with_golden


def test_add():
    class Add(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return a + b
        
    inputs = [torch.rand(2, 32, 32), torch.rand(2, 32, 32)]
    
    framework_model = Add()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out
    assert [compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)]


@pytest.mark.parametrize("params", [
    ((1, 32, 64), (-1, -2)),
    ((1, 64, 32), (1, 2)),
    ((1, 32, 64, 128), (3, 2)),
    ((32, 128), (0, 1)),
    ((18, 65), (1, 0)),
    ((6, 33, 34), (-1, 1))
])
def test_transpose(params):
    class Transpose(nn.Module):
        def __init__(self, dims):
            super().__init__()
            self.dims = dims

        def forward(self, a):
            return torch.transpose(a, *self.dims)

    input_shape, dims = params
    inputs = [torch.rand(input_shape)]

    framework_model = Transpose(dims)
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out
    assert [compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)]

@pytest.mark.parametrize("source_and_target_shape",
    [((8, 32, 256), (2, 4, 32, 256)),
     ((8, 32, 32), (1, 2, 4, 32, 32)),
     ((8192, 128), (1, 256, 32, 128))
     ],
    ids=["1", "2", "3"])
def test_reshape(source_and_target_shape):
    source_shape, target_shape = source_and_target_shape
    class Reshape(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            return torch.reshape(a, target_shape)

    inputs = [torch.rand(source_shape)]
    framework_model = Reshape()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)

@pytest.mark.parametrize("dims", [
    (1, 32, 64), (6, 33), (4, 16, 17)
])
def test_greater_equal(dims):
    class GreaterEqual(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return torch.greater_equal(a, b)

    inputs = [torch.rand(dims), torch.rand(dims)]

    framework_model = GreaterEqual()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    output = co_out[0].to(torch.bool)
    assert compare_with_golden(golden=fw_out, calculated=output)

def test_subtract():
    class Subtract(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return a - b
        
    inputs = [torch.rand(1, 32, 32), torch.rand(1, 32, 32)]
    
    framework_model = Subtract()
    fw_out = framework_model(*inputs)
    
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)
    
    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out
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
    
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)
    
    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out
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
    
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)
    
    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out
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
    
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)
    
    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out
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
    
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)
    
    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out
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
    
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
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
    
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
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
    
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
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
    
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)
    
    co_out = [co.to("cpu") for co in co_out]
    assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)


@pytest.mark.parametrize("x_shape", [7, 32, 41])
@pytest.mark.parametrize("y_shape", [7, 32, 41])
def test_sqrt(x_shape, y_shape):    
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
    
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)
    
    co_out = [co.to("cpu") for co in co_out]
    assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)


# @pytest.mark.parametrize("vocab_size", [2048, 16384, 32000])
# @pytest.mark.parametrize("token_num", [1, 7, 32])
# @pytest.mark.parametrize("embedding_dim", [128, 512, 3200])
@pytest.mark.xfail(reason="L1 allocation issue on Metal")
@pytest.mark.parametrize("vocab_size", [32000])
@pytest.mark.parametrize("token_num", [12])
@pytest.mark.parametrize("embedding_dim", [3200])
def test_embedding(vocab_size, token_num, embedding_dim):
    compiler_cfg = pyforge.config._get_global_compiler_config()
    compiler_cfg.enable_tvm_cpu_fallback = False

    class Embedding(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        def forward(self, x):
            return self.embedding(x)

    inputs = [
        torch.randint(0, vocab_size, (1, token_num)),
    ]

    framework_model = Embedding()
    fw_out = framework_model(*inputs)

    compiled_model = pyforge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)


@pytest.mark.xfail(reason="Program expects 2 inputs, found 1 in input tensors vector")
@pytest.mark.parametrize("shape", [
    (7,),           # 1D tensor
    (32,),          # 1D tensor
    (7, 32),        # 2D tensor
    (32, 41),       # 2D tensor
    (1, 7, 32),     # 3D tensor
    (1, 32, 41),    # 3D tensor
    (1, 7, 32, 41), # 4D tensor
    (2, 7, 32, 41)  # 4D tensor
])
def test_reciprocal(shape):
    class Reciprocal(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.reciprocal(x)
        
    inputs = [
        torch.rand(*shape) + 0.1,  # Adding 0.1 to avoid division by zero
    ]

    framework_model = Reciprocal()
    fw_out = framework_model(*inputs)
    
    compiled_model = pyforge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)
    
    co_out = [co.to("cpu") for co in co_out]
    assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)


@pytest.mark.parametrize("shape", [
    (7,),           # 1D tensor
    (32,),          # 1D tensor
    (7, 32),        # 2D tensor
    (32, 41),       # 2D tensor
    (1, 7, 32),     # 3D tensor
    (1, 32, 41),    # 3D tensor
    (1, 7, 32, 41), # 4D tensor
    (2, 7, 32, 41)  # 4D tensor
])
def test_sigmoid(shape):
    class Sigmoid(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x):
            return torch.sigmoid(x)
        
    inputs = [
        torch.rand(*shape),
    ]
    framework_model = Sigmoid()
    fw_out = framework_model(*inputs)
    
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)
    
    co_out = [co.to("cpu") for co in co_out]
    assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)