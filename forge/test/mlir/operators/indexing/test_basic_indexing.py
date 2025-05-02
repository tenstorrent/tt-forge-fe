# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Literal
import pytest
import torch
from torch import nn

import forge
from forge.verify.verify import verify


@pytest.mark.xfail(
    reason="RuntimeError: TT_ASSERT @ /proj_sw/user_dev/vkovinic/tt-forge-fe/forge/csrc/graph_lib/shape.cpp:135: (i >= 0) && (i < (int)dims_.size()) "
    + "info: "
    + "Trying to access element outside of dimensions: -1"
)
@pytest.mark.parametrize(
    "index_shape",
    [
        (0, (10,)),
        (2, (10,)),
        (-1, (10,)),
    ],
)
@pytest.mark.push
def test_python_indexing(forge_property_recorder, index_shape: Literal[0] | Literal[2] | Literal[-1]):

    index, shape = index_shape

    class IndexingModule(nn.Module):
        def __init__(self, index):
            super().__init__()
            self.index = index

        def forward(self, x):
            return x[self.index]

    # create random vector of floats size 10

    inputs = [torch.randn(shape)]

    framework_model = IndexingModule(index)
    compiled_model = forge.compile(framework_model, inputs, forge_property_handler=forge_property_recorder)

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize(
    "index_shape",
    [
        ([0, 2, 4], (10,)),  # vector
        pytest.param(
            ([[0, 1], [2, 3]], (5, 5)),  # 2D matrix indexing
            marks=pytest.mark.xfail(
                reason='AssertionError: Dim to drop needs to be singleton (Failed on "DecomposeMultiIndexAdvIndex" TVM callback)'
            ),
        ),
        pytest.param(
            ([[0, 1, -1], [2, 3, -1]], (5, 5, 5)),  # 3D matrix indexing
            marks=pytest.mark.xfail(
                reason='AssertionError: Dim to drop needs to be singleton (Failed on "DecomposeMultiIndexAdvIndex" TVM callback)'
            ),
        ),
    ],
)
@pytest.mark.push
def test_python_indexing_with_lists(forge_property_recorder, index_shape: list[int] | list[list[int]]):
    indices, shape = index_shape

    class ListIndexingModule(nn.Module):
        def __init__(self, indices):
            super().__init__()
            self.indices = indices

        def forward(self, x):
            return x[self.indices]

    # Create random tensor
    inputs = [torch.randn(shape)]

    framework_model = ListIndexingModule(indices)
    compiled_model = forge.compile(framework_model, inputs, forge_property_handler=forge_property_recorder)

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize(
    "index_shape",
    [
        ([0, 2, 4], (10,)),  # vector
        pytest.param(
            ([[0, 1], [2, 3]], (5, 5)),  # 2D matrix indexing
            marks=pytest.mark.xfail(
                reason="AssertionError: Setting a tensor value of incorrect shape: (1, 2, 5) vs torch.Size([1, 2])"
            ),
        ),
        pytest.param(
            ([[0, 1, -1], [2, 3, -1]], (5, 5, 5)),  # 3D matrix indexing
            marks=pytest.mark.xfail(
                reason="AssertionError: Setting a tensor value of incorrect shape: (1, 3, 5, 5) vs torch.Size([1, 3, 5])"
            ),
        ),
    ],
)
@pytest.mark.push
def test_python_indexing_with_tensors(forge_property_recorder, index_shape):
    indices, shape = index_shape

    class TensorIndexingModule(nn.Module):
        def __init__(self, indices):
            super().__init__()
            self.indices = indices

        def forward(self, x):
            return x[self.indices]

    # Create random tensor
    inputs = [torch.randn(shape)]

    framework_model = TensorIndexingModule(torch.tensor(indices))
    compiled_model = forge.compile(framework_model, inputs, forge_property_handler=forge_property_recorder)

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
