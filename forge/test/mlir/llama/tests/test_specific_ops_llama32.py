# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import pytest

import torch
from torch import nn

import forge
from forge.op.eval.common import compare_with_golden_pcc, compare_with_golden


@pytest.mark.parametrize(
    "shapes",
    [
        [(1, 11, 1), (1,)],
        [(1, 32, 11, 64), (1, 32, 11, 64)],
        [(1, 8, 11, 64), (1, 8, 11, 64)],
        [(1, 32, 11, 11), (1, 1, 11, 11)],
        [(1, 11, 2048), (1, 11, 2048)],
    ],
)
@pytest.mark.push
def test_add(shapes):
    if shapes[0] != shapes[1]:
        pytest.xfail("eltwise_add broadcast not supported")

    class Add(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return a + b

    inputs = [torch.rand(shapes[0]), torch.rand(shapes[1])]

    framework_model = Add()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)


@pytest.mark.parametrize(
    "inputs_and_dim",
    [
        ((1, 11, 32), (1, 11, 32), -1),
        ((1, 32, 11, 32), (1, 32, 11, 32), -1),
        ((1, 8, 11, 32), (1, 8, 11, 32), -1),
    ],
)
@pytest.mark.push
def test_concat(inputs_and_dim):
    in_shape1, in_shape2, dim = inputs_and_dim

    class Concat(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return torch.cat((a, b), dim)

    inputs = [torch.rand(in_shape1), torch.rand(in_shape2)]

    framework_model = Concat()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)


@pytest.mark.parametrize("shapes", [(1, 11, 64)])
@pytest.mark.push
def test_cosine(shapes):
    class Cosine(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            return torch.cos(a)

    inputs = [torch.rand(shapes)]

    framework_model = Cosine()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)


@pytest.mark.parametrize("shapes", [(1, 11, 64)])
@pytest.mark.push
def test_sine(shapes):
    class Sine(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            return torch.sin(a)

    inputs = [torch.rand(shapes)]

    framework_model = Sine()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)


@pytest.mark.parametrize(
    "shapes",
    [
        ((1, 11), 128256, 2048),
    ],
)
@pytest.mark.xfail(reason="TTNN Layout::ROW_MAJOR error")
@pytest.mark.push
def test_embedding(shapes):
    input_size, vocab_size, embedding_dim = shapes

    class Embedding(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        def forward(self, x):
            return self.embedding(x)

    inputs = [torch.randint(0, vocab_size, input_size)]

    framework_model = Embedding()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)


@pytest.mark.parametrize(
    "shapes",
    [
        ((11, 2048), (2048, 2048)),
        ((1, 32, 1), (1, 1, 11)),
        ((11, 2048), (2048, 512)),
        ((32, 11, 64), (32, 64, 11)),
        ((32, 11, 11), (32, 11, 64)),
        ((11, 2048), (2048, 8192)),
        ((1, 11, 8192), (8192, 2048)),
        ((1, 11, 2048), (2048, 128256)),
    ],
)
@pytest.mark.push
def test_matmul(shapes):
    if shapes == ((1, 11, 8192), (8192, 2048)):
        pytest.xfail("pcc < 0.95")

    shape1, shape2 = shapes

    class Matmul(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.matmul(x, y)

    inputs = [
        torch.rand(shape1),
        torch.rand(shape2),
    ]

    framework_model = Matmul()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.95)


@pytest.mark.parametrize(
    "shapes",
    [
        ((1, 11, 2048), (1, 11, 2048)),
        ((1, 11, 2048), (1, 11, 1)),
        ((2048,), (1, 11, 2048)),
        ((1, 32, 11, 64), (1, 1, 11, 64)),
        ((1, 32, 11, 32), (1,)),
        ((1, 8, 11, 64), (1, 1, 11, 64)),
        ((1, 8, 11, 32), (1,)),
        ((1, 32, 11, 11), (1,)),
        ((1, 11, 8192), (1, 11, 8192)),
    ],
)
@pytest.mark.push
def test_multiply(shapes):
    if shapes == ((1, 32, 11, 64), (1, 1, 11, 64)) or shapes == ((1, 8, 11, 64), (1, 1, 11, 64)):
        pytest.xfail("eltwise multiply broadcast not supported")

    shape1, shape2 = shapes

    class Multiply(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return x * y

    inputs = [
        torch.rand(shape1),
        torch.rand(shape2),
    ]

    framework_model = Multiply()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)[0].to("cpu")

    assert compare_with_golden_pcc(fw_out, co_out, pcc=0.99)


@pytest.mark.parametrize(
    "shapes",
    [
        (1, 11, 2048),
    ],
)
@pytest.mark.xfail(reason="pcc < 0.75")
@pytest.mark.push
def test_reduce_avg(shapes):
    class ReduceAvg(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.mean(x, dim=-1, keepdim=True)

    inputs = [torch.rand(shapes)]
    framework_model = ReduceAvg()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)[0].to("cpu")

    assert compare_with_golden_pcc(fw_out, co_out, pcc=0.75)


@pytest.mark.parametrize(
    "shapes",
    [
        (1, 11, 8192),
    ],
)
@pytest.mark.push
def test_sigmoid(shapes):
    class Sigmoid(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.sigmoid(x)

    inputs = [torch.rand(shapes)]
    framework_model = Sigmoid()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)[0].to("cpu")

    assert compare_with_golden_pcc(fw_out, co_out, pcc=0.99)


@pytest.mark.parametrize(
    "shapes",
    [
        ((1, 11, 1),),
    ],
)
@pytest.mark.push
def test_reciprocal(shapes):
    class Reciprocal(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.reciprocal(x)

    inputs = [torch.rand(shapes[0]) + 0.1]
    framework_model = Reciprocal()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)[0].to("cpu")

    assert compare_with_golden_pcc(fw_out, co_out, pcc=0.99)


@pytest.mark.parametrize(
    "source_and_target_shape",
    [
        ((1, 11, 2048), (11, 2048)),
        ((1, 11, 2048), (1, 11, 32, 64)),
        ((1, 11, 2048), (1, 11, 2048)),
        ((1, 32, 11, 64), (32, 11, 64)),
        ((1, 32, 11, 64), (1, 32, 11, 64)),
        ((11, 512), (1, 11, 8, 64)),
        ((1, 8, 4, 11, 64), (32, 11, 64)),
        ((1, 8, 4, 11, 64), (1, 32, 11, 64)),
        ((32, 11, 11), (1, 32, 11, 11)),
        ((32, 11, 11), (32, 11, 11)),
        ((1, 32, 64, 11), (32, 64, 11)),
        ((1, 11, 32, 64), (11, 2048)),
        ((11, 8192), (1, 11, 8192)),
    ],
)
@pytest.mark.push
def test_reshape(source_and_target_shape):
    source_shape, target_shape = source_and_target_shape

    if len(source_shape) > 4 or len(target_shape) > 4:
        pytest.xfail("Only 2D, 3D, and 4D tensors are supported")

    if (
        source_and_target_shape == ((32, 11, 11), (1, 32, 11, 11))
        or source_and_target_shape == ((32, 11, 11), (32, 11, 11))
        or source_and_target_shape == ((1, 32, 64, 11), (32, 64, 11))
    ):
        pytest.xfail("pcc < 0.99")

    class Reshape(nn.Module):
        def __init__(self, target_shape):
            super().__init__()
            self.target_shape = target_shape

        def forward(self, a):
            return torch.reshape(a, self.target_shape)

    inputs = [torch.rand(source_shape, dtype=torch.bfloat16)]
    framework_model = Reshape(target_shape)
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)


@pytest.mark.parametrize(
    "shapes",
    [
        ((1, 32, 11, 11), -1),
    ],
)
@pytest.mark.push
def test_softmax(shapes):
    shape, dim = shapes

    class Softmax(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, x):
            return torch.softmax(x, dim=dim)

    inputs = [torch.rand(shape)]
    framework_model = Softmax(dim)
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)[0].to("cpu")

    assert compare_with_golden_pcc(fw_out, co_out, pcc=0.99)


@pytest.mark.parametrize(
    "shapes",
    [
        (1, 11, 1),
    ],
)
@pytest.mark.push
def test_sqrt(shapes):
    class Sqrt(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.sqrt(x)

    inputs = [torch.rand(shapes)]
    framework_model = Sqrt()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)[0].to("cpu")

    assert compare_with_golden_pcc(fw_out, co_out, pcc=0.99)


@pytest.mark.parametrize(
    "input_shape_and_dim",
    [
        ((32), 0),
        ((1, 11, 64), 1),
    ],
)
@pytest.mark.push
def test_unsqueeze(input_shape_and_dim):
    input_shape, dim = input_shape_and_dim

    class Unsqueeze(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, a):
            return torch.unsqueeze(a, self.dim)

    inputs = [torch.rand(input_shape, dtype=torch.bfloat16)]

    framework_model = Unsqueeze(dim)
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    assert co_out[0].shape == fw_out.shape
    assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)


@pytest.mark.parametrize(
    "params",
    [
        ((2048, 2048), (-2, -1)),
        ((1, 11, 32, 64), (-3, -2)),
        ((1, 32, 11), (-2, -1)),
        ((512, 2048), (-2, -1)),
        ((1, 11, 8, 64), (-3, -2)),
        ((32, 11, 64), (-2, -1)),
        ((32, 11, 64), (-3, -2)),
        ((32, 64, 11), (-2, -1)),
        ((8192, 2048), (-2, -1)),
        ((2048, 8192), (-2, -1)),
        ((128256, 2048), (-2, -1)),
    ],
)
@pytest.mark.push
def test_transpose(params):
    shapes, dims = params

    class Transpose(nn.Module):
        def __init__(self, dims):
            super().__init__()
            self.dims = dims

        def forward(self, a):
            return torch.transpose(a, *self.dims)

    inputs = [torch.rand(shapes)]
    framework_model = Transpose(dims)
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)[0].to("cpu")

    assert compare_with_golden_pcc(fw_out, co_out, pcc=0.99)
