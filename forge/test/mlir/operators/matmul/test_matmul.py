# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

import forge
from forge.verify.verify import verify, VerifyConfig


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
