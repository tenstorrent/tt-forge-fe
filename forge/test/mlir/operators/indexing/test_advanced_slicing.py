# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch import nn

import forge
from forge.verify.verify import verify


@pytest.mark.parametrize(
    "input_dim_index",
    [
        (torch.arange(1.0, 10.0).reshape(3, 3), 0, 1),  # Basic 3x3 tensor, select second row
        (torch.arange(1.0, 13.0).reshape(3, 2, 2), 0, 2),  # 3D tensor, select last along first dim
        (torch.arange(1.0, 13.0).reshape(3, 2, 2), 1, 0),  # 3D tensor, select first along middle dim
        (torch.arange(1.0, 13.0).reshape(3, 2, 2), 2, 1),  # 3D tensor, select second along last dim
        (torch.arange(1.0, 5.0).reshape(4, 1), 0, 0),  # Tall matrix, select first row
        (torch.arange(1.0, 5.0).reshape(1, 4), 1, 2),  # Wide matrix, select third column
    ],
)
@pytest.mark.push
def test_select(input_dim_index):
    input_tensor, dim, index = input_dim_index

    class SelectModule(nn.Module):
        def __init__(self, dim, index):
            super().__init__()
            self.dim = dim
            self.index = index

        def forward(self, x):
            return torch.select(x, self.dim, self.index)

    # Inputs for the test
    inputs = [input_tensor]

    framework_model = SelectModule(dim, index)
    compiled_model = forge.compile(framework_model, inputs)

    # Run verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "input_tensor_sizes_dim",
    [
        pytest.param(
            (torch.arange(10.0), [3, 3, 4], 0),  # 1D tensor  # Sizes to split  # Dimension to split along
        ),
        pytest.param(
            (torch.arange(20.0).reshape(4, 5), 2, 0),  # 2D tensor  # Number of parts  # Dimension to split along
        ),
        pytest.param(
            (
                torch.arange(20.0).reshape(4, 5),  # 2D tensor
                3,  # Number of parts (uneven split)
                1,  # Dimension to split along
            ),
        ),
    ],
)
@pytest.mark.push
def test_split(input_tensor_sizes_dim):
    input_tensor, sizes_or_parts, dim = input_tensor_sizes_dim

    class SplitModule(nn.Module):
        def __init__(self, sizes_or_parts, dim):
            super().__init__()
            self.sizes_or_parts = sizes_or_parts
            self.dim = dim

        def forward(self, x):
            return torch.split(x, self.sizes_or_parts, self.dim)

    # Inputs for the test
    inputs = [input_tensor]

    framework_model = SplitModule(sizes_or_parts, dim)
    compiled_model = forge.compile(framework_model, inputs)

    # Run verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "input_tensor_chunks_dim",
    [
        pytest.param(
            (torch.arange(10.0), 3, 0),  # 1D tensor  # Number of chunks  # Dimension to chunk along
            marks=pytest.mark.xfail(reason="KeyError: 'prim::ListUnpack_0_2'"),
        ),
        pytest.param(
            (torch.arange(20.0).reshape(4, 5), 2, 0),  # 2D tensor  # Number of chunks  # Dimension to chunk along
        ),
        pytest.param(
            (torch.arange(20.0).reshape(4, 5), 3, 1),  # 2D tensor  # Number of chunks  # Dimension to chunk along
        ),
    ],
)
@pytest.mark.push
def test_chunk(input_tensor_chunks_dim):
    input_tensor, chunks, dim = input_tensor_chunks_dim

    class ChunkModule(nn.Module):
        def __init__(self, chunks, dim):
            super().__init__()
            self.chunks = chunks
            self.dim = dim

        def forward(self, x):
            # Chunk the tensor into `chunks` along dimension `dim`
            return torch.chunk(x, self.chunks, self.dim)

    # Inputs for the test
    inputs = [input_tensor]

    framework_model = ChunkModule(chunks, dim)
    compiled_model = forge.compile(framework_model, inputs)

    # Run verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "input_tensor_indices",
    [
        pytest.param(
            (
                torch.arange(1.0, 10.0).reshape(3, 3),  # 3x3 input tensor with floats
                torch.tensor([0, 4, 8]),  # Indices to take
            ),
            id="standard_indices",
            marks=pytest.mark.xfail(reason="IndexError: index 4 is out of bounds for dimension 0 with size 3"),
        ),
        pytest.param(
            (
                torch.arange(1.0, 10.0).reshape(3, 3),  # 3x3 input tensor with floats
                torch.tensor([0, 4, -1]),  # Indices, including a negative index
            ),
            id="negative_index",
            marks=pytest.mark.xfail(reason="IndexError: index 4 is out of bounds for dimension 0 with size 3"),
        ),
        pytest.param(
            (
                torch.arange(1.0, 10.0).reshape(3, 3),
                torch.tensor([0, 1, 2]),  # Sequential indices
            ),
            id="sequential_indices",
            marks=pytest.mark.xfail(reason="tvm.error.InternalError:"),
        ),
        pytest.param(
            (
                torch.arange(1.0, 10.0).reshape(3, 3),
                torch.tensor([0, 0, 0]),  # Repeated indices
            ),
            id="repeated_indices",
            marks=pytest.mark.xfail(
                reason="AssertionError: Data mismatch on output 0 between framework and Forge codegen"
            ),
        ),
        pytest.param(
            (
                torch.arange(1.0, 10.0).reshape(3, 3),
                torch.tensor([-1, -2, -3]),  # All negative indices
            ),
            id="negative_indices",
            marks=pytest.mark.xfail(
                reason="AssertionError: Data mismatch on output 0 between framework and Forge codegen"
            ),
        ),
        pytest.param(
            (
                torch.arange(1.0, 13.0).reshape(3, 2, 2),  # 3D tensor
                torch.tensor([0, 5, 10]),  # Valid indices for flattened tensor
            ),
            id="3d_tensor",
            marks=pytest.mark.xfail(reason="IndexError: index 5 is out of bounds for dimension 0 with size 3"),
        ),
    ],
)
@pytest.mark.push
def test_take(input_tensor_indices):
    input_tensor, indices = input_tensor_indices

    class TakeModule(nn.Module):
        def __init__(self, indices):
            super().__init__()
            self.indices = indices

        def forward(self, x):
            return torch.take(x, self.indices)

    # Inputs for the test
    inputs = [input_tensor]

    framework_model = TakeModule(indices)
    compiled_model = forge.compile(framework_model, inputs)

    # Run verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "input_tensor_as_tuple",
    [
        pytest.param(
            (torch.tensor([0.0, 1.1, 0.0, 2.2, 0.0, 3.3], dtype=torch.float32), True),
            marks=pytest.mark.xfail(
                reason=" AttributeError: <class 'tvm.relay.expr.Call'> has no attribute type_annotation"
            ),
        ),
        pytest.param(
            (torch.tensor([[0.0, 1.1], [2.2, 0.0], [0.0, 3.3]], dtype=torch.float32), True),
            marks=pytest.mark.xfail(
                reason=" AttributeError: <class 'tvm.relay.expr.Call'> has no attribute type_annotation"
            ),
        ),
        pytest.param(
            (torch.tensor([[[0.0], [1.1]], [[2.2], [0.0]], [[0.0], [3.3]]], dtype=torch.float32), True),
            marks=pytest.mark.xfail(
                reason=" AttributeError: <class 'tvm.relay.expr.Call'> has no attribute type_annotation"
            ),
        ),
        pytest.param(
            (torch.tensor([0.0, 1.1, 0.0, 2.2, 0.0, 3.3], dtype=torch.float32), False),
            marks=pytest.mark.xfail(
                reason=" InternalError: Check failed: (pval != nullptr) is false: Cannot allocate memory symbolic tensor shape [T.Any(), 1]"
            ),
        ),
        pytest.param(
            (torch.tensor([[0.0, 1.1], [2.2, 0.0], [0.0, 3.3]], dtype=torch.float32), False),
            marks=pytest.mark.xfail(
                reason=" InternalError: Check failed: (pval != nullptr) is false: Cannot allocate memory symbolic tensor shape [T.Any(), 2]"
            ),
        ),
        pytest.param(
            (torch.tensor([[[0.0], [1.1]], [[2.2], [0.0]], [[0.0], [3.3]]], dtype=torch.float32), False),
            marks=pytest.mark.xfail(
                reason=" InternalError: Check failed: (pval != nullptr) is false: Cannot allocate memory symbolic tensor shape [T.Any(), 3]"
            ),
        ),
    ],
)
@pytest.mark.push
def test_nonzero(input_tensor_as_tuple):
    input_tensor, as_tuple = input_tensor_as_tuple

    class NonZeroModule(nn.Module):
        def __init__(self, as_tuple):
            super().__init__()
            self.as_tuple = as_tuple

        def forward(self, x):
            # Returns the indices of non-zero elements
            return torch.nonzero(x, as_tuple=self.as_tuple)

    # Inputs for the test
    inputs = [input_tensor]

    framework_model = NonZeroModule(as_tuple=as_tuple)
    compiled_model = forge.compile(framework_model, inputs)

    # Run verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "input_dim_start_length",
    [
        pytest.param(
            (
                torch.arange(12).reshape(3, 4).float(),
                0,
                1,
                2,
            ),  # Narrow along dimension 0, starting at index 1, length 2
            id="narrow_dim_0",
        ),
        pytest.param(
            (
                torch.arange(12).reshape(3, 4).float(),
                1,
                2,
                2,
            ),  # Narrow along dimension 1, starting at index 2, length 2
            id="narrow_dim_1",
        ),
        pytest.param(
            (
                torch.arange(24).reshape(3, 4, 2).float(),
                2,
                0,
                1,
            ),  # Narrow along dimension 2, starting at index 0, length 1
            id="narrow_dim_2",
        ),
        pytest.param(
            (
                torch.arange(20).reshape(5, 4).float(),
                0,
                3,
                1,
            ),  # Narrow along dimension 0, starting at index 3, length 1
            id="narrow_single_element_dim_0",
        ),
        pytest.param(
            (torch.arange(12).reshape(3, 4).float(), 1, -1, 1),  # Handle negative indexing
            id="negative_index_narrow_dim_1",
        ),
    ],
)
@pytest.mark.push
def test_narrow(input_dim_start_length):
    input_tensor, dim, start, length = input_dim_start_length

    class NarrowModule(torch.nn.Module):
        def __init__(self, dim, start, length):
            super().__init__()
            self.dim = dim
            self.start = start
            self.length = length

        def forward(self, x):
            return torch.narrow(x, self.dim, self.start, self.length)

    # Inputs for the test
    inputs = [input_tensor]

    framework_model = NarrowModule(dim, start, length)
    compiled_model = forge.compile(framework_model, inputs)

    # Run verification
    verify(inputs, framework_model, compiled_model)
