# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

import forge


@pytest.mark.parametrize(
    "input_shape, dim, keepdim",
    [
        ((64,), 0, True),
        ((64,), -1, True),
        ((4, 64), 0, True),
        ((32, 32), -2, True),
        ((2, 32, 32), 0, True),
        ((1, 64, 32), 2, True),
        ((4, 32, 64), -2, True),
        ((4, 128, 128, 128), 0, True),
        ((1, 128, 128, 128), 2, True),
        ((1, 128, 128, 128), -3, True),
        ((4, 128, 128, 128), -4, True),
        pytest.param(
            (64,),
            0,
            False,
            marks=pytest.mark.xfail(reason="'ttir.squeeze' op Output tensor must have at least one dimension."),
        ),
        pytest.param(
            (64,),
            -1,
            False,
            marks=pytest.mark.xfail(reason="'ttir.squeeze' op Output tensor must have at least one dimension."),
        ),
        pytest.param(
            (4, 64),
            0,
            False,
        ),
        pytest.param(
            (32, 32),
            -2,
            False,
        ),
        ((2, 32, 32), 0, False),
        ((1, 64, 32), 2, False),
        ((4, 32, 64), -2, False),
        ((4, 128, 128, 128), 0, False),
        (
            (1, 128, 128, 128),
            2,
            False,
        ),
        ((1, 128, 128, 128), -3, False),
        ((4, 128, 128, 128), -4, False),
    ],
)
@pytest.mark.push
def test_reduce_sum(input_shape, dim, keepdim):
    class ReduceSum(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            return torch.sum(a, dim=dim, keepdim=keepdim)

    inputs = [torch.rand(input_shape)]

    framework_model = ReduceSum()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    # Skipping PCC check due to inconsistencies between Framework and Compiled model
    #
    # co_out = [co.to("cpu") for co in co_out]
    # assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)


@pytest.mark.parametrize(
    "input_shape, dim, keepdim",
    [
        ((64,), 0, True),
        ((64,), -1, True),
        ((4, 64), 0, True),
        ((32, 32), -2, True),
        ((2, 32, 32), 0, True),
        ((1, 64, 32), 2, True),
        ((4, 32, 64), -2, True),
        ((4, 128, 128, 128), 0, True),
        ((1, 128, 128, 128), 2, True),
        ((1, 128, 128, 128), -3, True),
        ((4, 128, 128, 128), -4, True),
        pytest.param(
            (64,),
            0,
            False,
            marks=pytest.mark.xfail(reason="'ttir.squeeze' op Output tensor must have at least one dimension."),
        ),
        pytest.param(
            (64,),
            -1,
            False,
            marks=pytest.mark.xfail(reason="'ttir.squeeze' op Output tensor must have at least one dimension."),
        ),
        pytest.param(
            (4, 64),
            0,
            False,
        ),
        pytest.param(
            (32, 32),
            -2,
            False,
        ),
        ((2, 32, 32), 0, False),
        ((1, 64, 32), 2, False),
        ((4, 32, 64), -2, False),
        ((4, 128, 128, 128), 0, False),
        (
            (1, 128, 128, 128),
            2,
            False,
        ),
        ((1, 128, 128, 128), -3, False),
        ((4, 128, 128, 128), -4, False),
    ],
)
@pytest.mark.push
def test_reduce_mean(input_shape, dim, keepdim):
    class ReduceMean(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            return torch.mean(a, dim=dim, keepdim=keepdim)

    inputs = [torch.rand(input_shape)]

    framework_model = ReduceMean()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    # Skipping PCC check due to inconsistencies between Framework and Compiled model
    #
    # co_out = [co.to("cpu") for co in co_out]
    # assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)


@pytest.mark.parametrize("x_shape", [7, 32, 41])
@pytest.mark.parametrize("y_shape", [7, 32, 41])
@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.push
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

    # Skipping PCC check due to inconsistencies between Framework and Compiled model
    #
    # co_out = [co.to("cpu") for co in co_out]
    # assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)


@pytest.mark.parametrize(
    "input_shape, dim, keepdim",
    [
        ((64,), 0, True),
        ((64,), -1, True),
        ((4, 64), 0, True),
        ((32, 32), -2, True),
        ((2, 32, 32), 0, True),
        ((1, 64, 32), 2, True),
        ((4, 32, 64), -2, True),
        ((4, 128, 128, 128), 0, True),
        ((1, 128, 128, 128), 2, True),
        ((1, 128, 128, 128), -3, True),
        ((4, 128, 128, 128), -4, True),
        pytest.param(
            (64,),
            0,
            False,
            marks=pytest.mark.xfail(reason="[mlir::AffineMap collapsedLinearAffineMap] Assertion `end > 0' failed."),
        ),
        pytest.param(
            (64,),
            -1,
            False,
            marks=pytest.mark.xfail(reason="[mlir::AffineMap collapsedLinearAffineMap] Assertion `end > 0' failed."),
        ),
        ((4, 64), 0, False),
        ((32, 32), -2, False),
        ((2, 32, 32), 0, False),
        ((1, 64, 32), 2, False),
        ((4, 32, 64), -2, False),
        ((4, 128, 128, 128), 0, False),
        ((1, 128, 128, 128), 2, False),
        ((1, 128, 128, 128), -3, False),
        ((4, 128, 128, 128), -4, False),
    ],
)
@pytest.mark.push
def test_reduce_max(input_shape, dim, keepdim):
    input = (input_shape, dim, keepdim)
    if input in [((64,), 0, False), ((64,), -1, False)]:
        pytest.xfail(reason="[mlir::AffineMap collapsedLinearAffineMap] Assertion `end > 0' failed.")

    # TTNN Max issues:
    #   Unsupported dim - https://github.com/tenstorrent/tt-metal/issues/13186
    #   Shape mismatch along the H and W dimension - https://github.com/tenstorrent/tt-metal/issues/13189
    #   Tensor rank is not 4 - https://github.com/tenstorrent/tt-metal/issues/13190

    class ReduceMax(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            return torch.max(a, dim=dim, keepdim=keepdim)[0]

    inputs = [torch.rand(input_shape)]

    framework_model = ReduceMax()
    framework_model.eval()
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    co_out = compiled_model(*inputs)

    co_out = [co.to("cpu") for co in co_out]

    # Skipping PCC check due to inconsistencies between Framework and Compiled model
    #
    # co_out = [co.to("cpu") for co in co_out]
    # assert compare_with_golden_pcc(golden=fw_out, calculated=co_out[0], pcc=0.99)
