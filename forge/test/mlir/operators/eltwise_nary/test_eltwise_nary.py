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
        # [(9,), (19,), (29,), (39,), (49,), (59,)], # Flaky test, using close to 32GB on host
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
@pytest.mark.push
@pytest.mark.xfail(
    reason="[MLIR] error: where op casts inputs to float32 and predicate to float32 so we get float32 output"
)
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


@pytest.mark.parametrize(
    "input_shapes",
    [
        [(1, 78), (52,)],
        [(18,), (26,), (13,)],
        [(1, 11), (2, 2), (31, 3), (1, 5)],
        [(31,), (12, 3), (66,), (13, 12), (20,)],
        [(2, 2), (1, 3), (11,), (62,), (11,), (31,)],
        [(384, 384)] * 7,
        [(96, 96)] * 8,
    ],
)
@pytest.mark.push
def test_block_diag(input_shapes):

    inputs = [torch.randn(*shape) for shape in input_shapes]

    class block_diag(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, *inputs):
            return torch.block_diag(*inputs)

    framework_model = block_diag()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    verify(inputs, framework_model, compiled_model)
