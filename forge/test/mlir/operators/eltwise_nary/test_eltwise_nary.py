# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch import nn

import forge
from forge.verify.verify import verify


@pytest.mark.xfail(
    reason="[MLIR] error: type of return operand 0 doesn't match function result type in function @forward"
)
@pytest.mark.parametrize(
    "shapes",
    [
        [(41,), (30,)],
        [(30,), (40,), (50,)],
        [(21,), (31,), (49,), (50,)],
        [(62,), (22,), (36,), (14,), (15,)],
        [(9,), (19,), (29,), (39,), (49,), (59,)],
    ],
)
@pytest.mark.push
def test_meshgrid(shapes):
    class Meshgrid(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, *inputs):
            return torch.meshgrid(*inputs)

    inputs = [torch.arange(i * 10 + 1, i * 10 + 1 + shape[0], dtype=torch.float32) for i, shape in enumerate(shapes)]

    framework_model = Meshgrid()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "condition, input, other",
    [
        (
            [[1, 0], [0, 1]],
            [[1, 2], [3, 4]],
            [[10, 20], [30, 40]],
        ),
    ],
)
@pytest.mark.xfail(reason="Unsupported data format during lowering from TTForge to TTIR: Bfp2_b")
@pytest.mark.push
def test_where(condition, input, other):
    class Where(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, condition, input1, input2):
            return torch.where(condition, input1, input2)

    condition = torch.tensor(condition, dtype=torch.bool)
    input = torch.tensor(input)
    other = torch.tensor(other)

    inputs = [condition, input, other]

    framework_model = Where()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)


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
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)
