# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

import forge
from forge.verify.verify import verify


@pytest.mark.parametrize("batch_size", [1, 7, 32])
@pytest.mark.parametrize("outer_dim_x", [7, 32, 41, 64])
@pytest.mark.parametrize("outer_dim_y", [7, 32, 41, 64])
@pytest.mark.parametrize("inner_dim", [1, 7, 32, 41, 64])
@pytest.mark.push
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
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "x_shape, y_shape",
    [
        ((1, 1, 3, 5), (1, 15, 5, 2)),
        ((1, 1, 1, 4, 5), (1, 41, 14, 5, 2)),
        ((1, 1, 1, 4, 5), (1, 9, 12, 5, 2)),
        ((1, 1, 1, 1, 3, 3), (1, 8, 160, 160, 3, 1)),
    ],
)
@pytest.mark.xfail(
    reason="""[TTNN] RuntimeError: TT_FATAL @ third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/matmul_op.cpp:1545: a_shape[i] == b_shape[i]
    info: bmm (non-bcast matmul) expects input tensors of shapes BCMK*BCKN=BCMN or equivalent """
)
def test_Matmul_ND(x_shape, y_shape):
    class Matmul_ND(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.matmul(x, y)

    inputs = [
        torch.rand(*x_shape),
        torch.rand(*y_shape),
    ]

    framework_model = Matmul_ND()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "x_shape, y_shape, expected_shape",
    [
        # Basic 2D matrix multiplication
        ((3, 4), (4, 5), (3, 5)),
        # 1D x 2D -> 1D (vector-matrix multiplication)
        ((4,), (4, 3), (3,)),
        # 2D x 1D -> 1D (matrix-vector multiplication)
        ((3, 4), (4,), (3,)),
        # Batch dimension broadcasting - same batch size
        ((2, 3, 4), (2, 4, 5), (2, 3, 5)),
        # Batch dimension broadcasting - one operand has batch size 1
        pytest.param(
            (1, 3, 4),
            (2, 4, 5),
            (2, 3, 5),
            marks=pytest.mark.xfail(
                reason="TTNN backend limitation: bmm expects matching batch dimensions, doesn't support broadcasting"
            ),
        ),
        ((2, 3, 4), (1, 4, 5), (2, 3, 5)),
        # Multiple batch dimensions - all same
        ((2, 3, 4, 5), (2, 3, 5, 6), (2, 3, 4, 6)),
        # Multiple batch dimensions - broadcast first dimension
        pytest.param(
            (1, 3, 4, 5),
            (2, 3, 5, 6),
            (2, 3, 4, 6),
            marks=pytest.mark.xfail(
                reason="TVM frontend limitation: BatchMatmul doesn't support complex broadcasting patterns"
            ),
        ),
        # Multiple batch dimensions - broadcast second dimension
        pytest.param(
            (2, 1, 4, 5),
            (2, 3, 5, 6),
            (2, 3, 4, 6),
            marks=pytest.mark.xfail(
                reason="TVM frontend limitation: BatchMatmul doesn't support complex broadcasting patterns"
            ),
        ),
        # Different number of batch dimensions
        ((4, 5), (2, 3, 5, 6), (2, 3, 4, 6)),
        ((2, 3, 4, 5), (5, 6), (2, 3, 4, 6)),
        # Complex broadcasting scenario with proper dimension matching
        pytest.param(
            (1, 2, 1, 4, 5),
            (3, 2, 1, 5, 6),
            (3, 2, 1, 4, 6),
            marks=pytest.mark.xfail(
                reason="TVM frontend limitation: BatchMatmul doesn't support complex broadcasting patterns"
            ),
        ),
        # 1D tensor with batch dimensions
        ((2, 3, 4), (4,), (2, 3)),
        pytest.param(
            (4,),
            (2, 3, 4, 5),
            (2, 3, 5),
            marks=pytest.mark.xfail(
                reason="TTNN backend limitation: bmm expects matching batch dimensions, doesn't support broadcasting"
            ),
        ),
    ],
)
@pytest.mark.push
def test_matmul_broadcasting(x_shape, y_shape, expected_shape):
    """Test matmul broadcasting behavior matches torch.matmul exactly"""

    class MatmulBroadcast(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.matmul(x, y)

    # Create input tensors
    x = torch.rand(*x_shape)
    y = torch.rand(*y_shape)

    # Verify expected shape with PyTorch
    torch_result = torch.matmul(x, y)
    assert torch_result.shape == expected_shape, f"Expected shape {expected_shape}, got {torch_result.shape}"

    inputs = [x, y]
    framework_model = MatmulBroadcast()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize("size", [3, 5, 7])
@pytest.mark.xfail(reason="Scalar output from 1D x 1D matmul may need special handling")
def test_matmul_dot_product(size):
    """Test 1D x 1D matmul (dot product) that produces scalar output"""

    class DotProduct(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.matmul(x, y)

    # Create 1D input tensors
    x = torch.rand(size)
    y = torch.rand(size)

    # Verify expected shape with PyTorch (should be scalar)
    torch_result = torch.matmul(x, y)
    assert torch_result.shape == (), f"Expected scalar (), got {torch_result.shape}"

    inputs = [x, y]
    framework_model = DotProduct()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)
