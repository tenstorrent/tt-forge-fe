# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import pytest

import pytest
import torch
from torch import nn

import forge
from forge.op.eval.common import compare_with_golden_pcc, compare_with_golden
from forge.tensor import to_forge_tensors, to_pt_tensors


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
    assert False, "This test is supposed to fail"
    assert all([compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


@pytest.mark.parametrize(
    "params",
    [
        ((1, 32, 64), (-1, -2)),
        ((1, 64, 32), (1, 2)),
        ((1, 32, 64, 128), (3, 2)),
        ((32, 128), (0, 1)),
        ((18, 65), (1, 0)),
        ((6, 33, 34), (-1, 1)),
    ],
)
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
    assert all([compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


@pytest.mark.parametrize(
    "source_and_target_shape",
    [((8, 32, 256), (2, 4, 32, 256)), ((8, 32, 32), (1, 2, 4, 32, 32)), ((8192, 128), (1, 256, 32, 128))],
    ids=["1", "2", "3"],
)
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


@pytest.mark.parametrize(
    "input_shape_and_dim",
    [
        ((1, 8, 16, 32, 32), 0),
        ((8, 1, 16, 32, 32), 1),
        ((8, 16, 1, 32, 32), 2),
        ((1, 8, 16, 32, 32), -5),
        ((8, 1, 16, 32, 32), -4),
        ((8, 16, 1, 32, 32), -3),
        ([1, 12, 3200], 0),
    ],
)
def test_squeeze(input_shape_and_dim):
    input_shape, dim = input_shape_and_dim

    if input_shape == [1, 12, 3200]:
        pytest.xfail("TTNN: Tensor layout issues with non tile dim aligned shapes")

    class Squeeze(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            return torch.squeeze(a, dim)

    inputs = [torch.rand(*input_shape)]

    framework_model = Squeeze()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    assert co_out[0].shape == fw_out.shape
    assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)


@pytest.mark.parametrize(
    "input_shape_and_dim",
    [
        ((8, 16, 32, 32), 0),
        ((8, 16, 32, 32), 1),
        ((8, 16, 32, 32), 2),
        ((8, 16, 32, 32), -3),
        ((8, 16, 32, 32), -4),
        ([12, 8640], 0),
    ],
)
def test_unsqueeze(input_shape_and_dim):
    input_shape, dim = input_shape_and_dim

    if input_shape == [12, 8640]:
        pytest.xfail("TTNN: Tensor layout issues with non tile dim aligned shapes")

    class Unsqueeze(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            return torch.unsqueeze(a, dim)

    inputs = [torch.rand(*input_shape)]

    framework_model = Unsqueeze()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    assert co_out[0].shape == fw_out.shape
    assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)


@pytest.mark.parametrize(
    "inputs_and_dim",
    [
        ((2, 2, 32, 32), (2, 2, 32, 32), 0),
        ((2, 2, 32, 32), (2, 2, 32, 32), 1),
        ((2, 2, 32, 32), (2, 2, 32, 32), 2),
        ((2, 2, 32, 32), (2, 2, 32, 32), 3),
        ((2, 2, 32, 32), (2, 2, 32, 32), -1),
        ((2, 2, 32, 32), (2, 2, 32, 32), -2),
        ((2, 2, 32, 32), (2, 2, 32, 32), -3),
        ((2, 2, 32, 32), (2, 2, 32, 32), -4),
    ],
    ids=["0", "1", "2", "3", "-1", "-2", "-3", "-4"],
)
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


@pytest.mark.parametrize("dims", [(1, 32, 64), (6, 33), (4, 16, 17)])
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
    assert all([compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


@pytest.mark.parametrize(
    "shape",
    [
        (1, 32, 32),
        (12, 8640),
    ],
)
def test_multiply(shape):
    class Multiply(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return a * b

    inputs = [torch.rand(shape), torch.rand(shape)]

    framework_model = Multiply()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out
    assert all([compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


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
    assert all([compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


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
    assert all([compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


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
    assert all([compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


@pytest.mark.parametrize("input_shape", [(1, 32, 32), (1, 64, 64), (1, 128, 128, 128)], ids=["32", "64", "128"])
@pytest.mark.parametrize("dim", [-1, -2], ids=["-1", "-2"])
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


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 32, 12),
        (1, 12, 32),
        (1, 12, 3200),
        (1, 32, 32),
        (1, 64, 64),
        (1, 128, 128, 128),
    ],
)
@pytest.mark.parametrize(
    "dim",
    [
        -1,
        -2,
    ],
)
def test_reduce_mean(input_shape, dim):

    if input_shape == (1, 12, 3200) and dim == -1:
        # Tensor mismatch(PCC: 0.72) - https://github.com/tenstorrent/tt-mlir/issues/869
        pytest.xfail("Tensor mismatch between PyTorch and TTNN (PCC: 0.72)")

    class ReduceMean(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            # reduce is supported on tt-metal only with keepdim=True
            return torch.mean(a, dim=dim, keepdim=True)

    inputs = [torch.rand(input_shape)]

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


@pytest.mark.xfail(
    reason="Unable to reshape a tensor in TILE_LAYOUT to non-tile height and width! Please convert the tensor to ROW_MAJOR_LAYOUT first"
)
@pytest.mark.parametrize("x_shape", [7, 32, 41])
@pytest.mark.parametrize("y_shape", [7, 32, 41])
@pytest.mark.parametrize("dim", [1, 2])
def test_mean(x_shape, y_shape, dim):
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
@pytest.mark.xfail(reason="ttnn.embedding op fails while reshaping the input_tensor in TILE_LAYOUT")
@pytest.mark.parametrize("vocab_size", [32000])
@pytest.mark.parametrize("token_num", [12])
@pytest.mark.parametrize("embedding_dim", [3200])
def test_embedding(vocab_size, token_num, embedding_dim):
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.enable_tvm_cpu_fallback = False

    class Embedding(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        def forward(self, x):
            return self.embedding(x)

    inputs = [
        torch.randint(0, vocab_size, (1, token_num)).to(torch.int32),
    ]

    framework_model = Embedding()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)


@pytest.mark.parametrize(
    "shape",
    [
        (7,),  # 1D tensor
        (32,),  # 1D tensor
        (7, 32),  # 2D tensor
        (32, 41),  # 2D tensor
        (1, 7, 32),  # 3D tensor
        (1, 32, 41),  # 3D tensor
        (1, 7, 32, 41),  # 4D tensor
        (2, 7, 32, 41),  # 4D tensor
    ],
)
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

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)


@pytest.mark.parametrize(
    "shape",
    [
        (7,),  # 1D tensor
        (32,),  # 1D tensor
        (7, 32),  # 2D tensor
        (32, 41),  # 2D tensor
        (1, 7, 32),  # 3D tensor
        (1, 32, 41),  # 3D tensor
        (1, 7, 32, 41),  # 4D tensor
        (2, 7, 32, 41),  # 4D tensor
    ],
)
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


@pytest.mark.parametrize("dim", [-1, -2, -3], ids=["-1", "-2", "-3"])
@pytest.mark.parametrize("start", [0], ids=["0"])
@pytest.mark.parametrize("stop", [2, 32, 64], ids=["2", "32", "64"])
@pytest.mark.parametrize("stride", [1, 2, 4, 8], ids=["1", "2", "4", "8"])
@pytest.mark.parametrize("shape", [(1, 32, 64, 64), (32, 64, 64), (64, 64)])
def test_indexing(dim, start, stop, stride, shape):
    if len(shape) == 2 and dim == -3:
        pytest.skip("Skipping since indexing on dim=-3, 2D tensor doesn't make sense")
    if stop > shape[dim]:
        pytest.skip("Skipping since stop > shape[dim]")

    class ForgeIndexing(forge.ForgeModule):
        def __init__(self, dim, start, stop, stride):
            super().__init__("ForgeTest")

        def forward(self, x):
            return forge.op.Index("indexing_op_1", x, dim, start, stop, stride)

    inputs = to_forge_tensors([torch.rand(*shape)])
    model = ForgeIndexing(dim, start, stop, stride)
    golden_out = model(*inputs)

    compiled_model = forge.compile(model, sample_inputs=inputs)

    inputs = to_pt_tensors(inputs)
    compiled_output = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in compiled_output]
    assert compare_with_golden_pcc(golden=golden_out.value(), calculated=co_out[0], pcc=0.99)


@pytest.mark.xfail(reason="ttnn.embedding op fails while reshaping the input_tensor in TILE_LAYOUT")
@pytest.mark.parametrize(
    "indices_shape",
    [
        (12,),
        (32,),
        (1, 7),
        (1, 28),
    ],
)
@pytest.mark.parametrize(
    "input_tensor_shape",
    [
        (12, 100),
        (3200, 512),
        (2048, 128),
        (4127, 256),
    ],
)
def test_adv_index_embedding_decompostion(indices_shape, input_tensor_shape):
    class ForgeAdvIndex(forge.ForgeModule):
        def __init__(self, name):
            super().__init__(name)

        def forward(self, input_tensor, indices):
            return forge.op.AdvIndex("adv_index_op_1", input_tensor, indices)

    model = ForgeAdvIndex("ForgeAdvIndex")

    # Sample Inputs
    pt_input_tensor = torch.rand(input_tensor_shape).to(torch.float32)
    pt_indices = torch.randint(input_tensor_shape[0], indices_shape).to(torch.int32)
    inputs = to_forge_tensors([pt_input_tensor, pt_indices])

    # Sanity run
    golden_out = model(*inputs)

    # Compile the model
    compiled_model = forge.compile(model, sample_inputs=inputs)

    # Run on TT device
    inputs = to_pt_tensors(inputs)
    compiled_output = compiled_model(*inputs)
    co_out = [co.to("cpu") for co in compiled_output]

    # Validate results
    assert compare_with_golden_pcc(golden=golden_out.value(), calculated=co_out[0], pcc=0.99)


@pytest.mark.parametrize(
    "input_shape",
    [
        (2, 32, 64, 64),
        (3, 22, 37, 41),
        (2, 32, 64),
        (3, 22, 37),
    ],
)
@pytest.mark.parametrize(
    "dim",
    [
        0,
        1,
        2,
        3,
        -1,
        -2,
        -3,
        -4,
    ],
)
def test_reduce_max(input_shape, dim):

    reduce_max_dim = dim
    if reduce_max_dim < 0:
        reduce_max_dim = reduce_max_dim + len(input_shape)
    if (reduce_max_dim < 0) or (reduce_max_dim >= len(input_shape)):
        pytest.skip()

    if (input_shape in [(2, 32, 64, 64), (3, 22, 37, 41)] and dim in [0, -4, 1, -3]) or (
        input_shape in [(2, 32, 64), (3, 22, 37)] and dim in [0, -3]
    ):
        pytest.xfail("TTNN Issue: Unsupported dim")

    # TTNN Max issues:
    #   Unsupported dim - https://github.com/tenstorrent/tt-metal/issues/13186
    #   Shape mismatch along the H and W dimension - https://github.com/tenstorrent/tt-metal/issues/13189
    #   Tensor rank is not 4 - https://github.com/tenstorrent/tt-metal/issues/13190

    class ReduceMax(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            return torch.max(a, dim=dim, keepdim=True)[0]

    inputs = [torch.rand(input_shape)]

    framework_model = ReduceMax()
    framework_model.eval()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)
