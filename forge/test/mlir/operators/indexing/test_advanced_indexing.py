# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch import nn

import forge
from forge.verify.verify import verify


@pytest.mark.parametrize(
    "tensor_and_indices",
    [
        pytest.param(
            (torch.arange(10, dtype=torch.float32).reshape(2, 5), torch.tensor([0, 2, 8, -1])),
            marks=pytest.mark.xfail(reason="IndexError: index 2 is out of bounds for dimension 0 with size 2"),
        ),
        pytest.param(
            (torch.arange(27, dtype=torch.float32).reshape(3, 3, 3), torch.tensor([0, 13, -1])),
            marks=pytest.mark.xfail(reason="IndexError: index 13 is out of bounds for dimension 0 with size 3"),
        ),
    ],
)
@pytest.mark.push
def test_take(forge_property_recorder, tensor_and_indices):
    tensor, indices = tensor_and_indices

    class TorchTakeIndexingModule(nn.Module):
        def __init__(self, indices):
            super().__init__()
            self.indices = indices

        def forward(self, x):
            return torch.take(x, self.indices)

    # Inputs for the test
    inputs = [tensor]

    framework_model = TorchTakeIndexingModule(indices)
    compiled_model = forge.compile(framework_model, inputs, forge_property_handler=forge_property_recorder)

    # Run verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


# NOTE: from pytorch docs:
# This operation may behave nondeterministically when
# given tensors on a CUDA device. See Reproducibility for more information.
@pytest.mark.parametrize(
    "input_dim_index_source",
    [
        pytest.param(
            (torch.zeros(5, dtype=torch.float32), 0, torch.tensor([0, 2, 4]), torch.tensor([1.0, 2.0, 3.0])),
            marks=pytest.mark.xfail(
                reason="NotImplementedError: The following operators are not implemented: ['aten::index_add']"
            ),
        ),
        pytest.param(
            (
                torch.zeros(3, 3, dtype=torch.float32),
                1,
                torch.tensor([0, 2]),
                torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            ),
            marks=pytest.mark.xfail(
                reason="NotImplementedError: The following operators are not implemented: ['aten::index_add']"
            ),
        ),
    ],
)
@pytest.mark.push
def test_index_add(forge_property_recorder, input_dim_index_source):
    input_tensor, dim, index, source = input_dim_index_source

    class IndexAddModule(nn.Module):
        def __init__(self, dim, index, source):
            super().__init__()
            self.dim = dim
            self.index = index
            self.source = source

        def forward(self, x):
            # add values from `source` into `x` at indices `index`
            return torch.index_add(x, self.dim, self.index, self.source)

    # Inputs for the test
    inputs = [input_tensor]

    framework_model = IndexAddModule(dim, index, source)
    compiled_model = forge.compile(framework_model, inputs, forge_property_handler=forge_property_recorder)

    # Run verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize(
    "input_dim_index_value",
    [
        pytest.param(
            (torch.zeros(3, 5, dtype=torch.float32), 1, torch.tensor([1, 3]), 10),
            marks=pytest.mark.xfail(
                reason="NotImplementedError: The following operators are not implemented: ['aten::index_fill']"
            ),
        ),
        pytest.param(
            (torch.zeros(3, 5, 2, dtype=torch.float32), 2, torch.tensor([0, 1]), 5),
            marks=pytest.mark.xfail(
                reason="NotImplementedError: The following operators are not implemented: ['aten::index_fill']"
            ),
        ),
    ],
)
@pytest.mark.push
def test_index_fill(forge_property_recorder, input_dim_index_value):
    input_tensor, dim, index, value = input_dim_index_value

    class IndexFillModule(nn.Module):
        def __init__(self, dim, index, value):
            super().__init__()
            self.dim = dim
            self.index = index
            self.value = value

        def forward(self, x):
            # Fill elements in `x` at `index` along `dim` with `value`
            return x.index_fill(self.dim, self.index, self.value)

    # Inputs for the test
    inputs = [input_tensor]

    framework_model = IndexFillModule(dim, index, value)
    compiled_model = forge.compile(framework_model, inputs, forge_property_handler=forge_property_recorder)

    # Run verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


# NOTE: from pytorch docs:
# If index contains duplicate entries, multiple elements from
# tensor will be copied to the same index of self. The result
# is nondeterministic since it depends on which copy occurs last.
@pytest.mark.parametrize(
    "input_dim_index_source",
    [
        pytest.param(
            (
                torch.zeros(3, 5, dtype=torch.float32),
                0,
                torch.tensor([0, 2]),
                torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0, 2.0]]),
            ),
            marks=pytest.mark.xfail(
                reason="AttributeError: module 'tvm.relay.op.transform' has no attribute 'scatter'"
            ),
        ),
        pytest.param(
            (torch.zeros(4, 4, 4, dtype=torch.float32), 1, torch.tensor([1, 3]), torch.ones(4, 2, 4)),
            marks=pytest.mark.xfail(
                reason="AttributeError: module 'tvm.relay.op.transform' has no attribute 'scatter'"
            ),
        ),
    ],
)
@pytest.mark.push
def test_index_copy(forge_property_recorder, input_dim_index_source):
    input_tensor, dim, index, source = input_dim_index_source

    class IndexCopyModule(nn.Module):
        def __init__(self, dim, index, source):
            super().__init__()
            self.dim = dim
            self.index = index
            self.source = source

        def forward(self, x):
            # Copy values from `source` into `x` at indices `index` along `dim`
            return torch.index_copy(x, self.dim, self.index, self.source)

    # Inputs for the test
    inputs = [input_tensor]

    framework_model = IndexCopyModule(dim, index, source)
    compiled_model = forge.compile(framework_model, inputs, forge_property_handler=forge_property_recorder)

    # Run verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


# NOTE: from pytorch docs:
# The returned tensor does not use the same storage as the original tensor.
# If out has a different shape than expected, we silently change it to the correct shape,
# reallocating the underlying storage if necessary.
@pytest.mark.parametrize(
    "input_dim_index",
    [
        (torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]), 0, torch.tensor([0, 2])),  # Select rows
        (torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]), 1, torch.tensor([1, 2])),  # Select columns
        (torch.arange(24, dtype=torch.float32).reshape(4, 3, 2), 2, torch.tensor([0, 1])),  # 3D tensor case
    ],
)
@pytest.mark.push
def test_index_select(forge_property_recorder, input_dim_index):
    input_tensor, dim, index = input_dim_index

    class IndexSelectModule(nn.Module):
        def __init__(self, dim, index):
            super().__init__()
            self.dim = dim
            self.index = index

        def forward(self, x):
            # Select elements from `x` along dimension `dim` using `index`
            return torch.index_select(x, self.dim, self.index)

    # Inputs for the test
    inputs = [input_tensor]

    framework_model = IndexSelectModule(dim, index)
    compiled_model = forge.compile(framework_model, inputs, forge_property_handler=forge_property_recorder)

    # Run verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize(
    "input_indices_values_accumulate",
    [
        pytest.param(
            (
                torch.zeros(5, dtype=torch.float32),  # Input tensor
                [torch.tensor([0, 2, 4])],  # Indices
                torch.tensor([1.1, 2.2, 3.3], dtype=torch.float32),  # Values
                False,  # Accumulate flag
            ),
            marks=pytest.mark.xfail(
                reason="tvm.error.InternalError: Check failed: size_t(mdim->value) <= ndim (3 vs. 1) : ScatterND: Given updates with shape (Y_0, ..., Y_{K-1}, X_M, ..., X_{N-1}), and indices with shape (M, Y_0, ..., Y_{K-1}), M must be less than or equal to N."
            ),
        ),
        pytest.param(
            (
                torch.ones(3, 3, dtype=torch.float32),  # 2D Input tensor
                [torch.tensor([0, 2]), torch.tensor([1, 0])],  # Indices
                torch.tensor([1.1, 2.2], dtype=torch.float32),  # Values
                True,  # Accumulate flag
            ),
            marks=pytest.mark.xfail(reason="Encountered unsupported op node type: scatter_nd, on device: tt"),
        ),
    ],
)
@pytest.mark.push
def test_index_put(forge_property_recorder, input_indices_values_accumulate):
    input_tensor, indices, values, accumulate = input_indices_values_accumulate

    class IndexPutModule(nn.Module):
        def __init__(self, indices, values, accumulate):
            super().__init__()
            self.indices = indices
            self.values = values
            self.accumulate = accumulate

        def forward(self, x):
            # Put `values` into `x` at `indices`, optionally accumulating
            return torch.index_put(x, self.indices, self.values, accumulate=self.accumulate)

    # Inputs for the test
    inputs = [input_tensor]

    framework_model = IndexPutModule(indices, values, accumulate)
    compiled_model = forge.compile(framework_model, inputs, forge_property_handler=forge_property_recorder)

    # Run verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
