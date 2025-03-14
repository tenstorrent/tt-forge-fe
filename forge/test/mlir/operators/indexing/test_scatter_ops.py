# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch import nn

import forge
from forge.verify.verify import verify


@pytest.mark.parametrize(
    "input_tensor, mask, source",
    [
        pytest.param(
            torch.zeros(10, dtype=torch.float32),  # 1D input tensor
            torch.tensor([True, False, True, False, True, False, True, False, True, False]),  # Mask
            torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),  # Source tensor
            id="test_masked_scatter_1",
            marks=pytest.mark.xfail(reason="RuntimeError: users.size() > 0"),
        ),
        # Less Number of elements in source
        pytest.param(
            torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32),  # input_tensor shape = (5,)
            torch.tensor([True, False, True, False, True], dtype=torch.bool),  # mask shape = (5,)
            torch.tensor([10, 20, 30], dtype=torch.float32),  # source shape = (3,)
            id="test_masked_scatter_2",
            marks=pytest.mark.xfail(reason="RuntimeError: users.size() > 0"),
        ),
        # Broadcasting: 1D input and mask with 2D source tensor
        pytest.param(
            torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32),  # input_tensor shape = (5,)
            torch.tensor([True, False, True, False, True], dtype=torch.bool),  # mask shape = (5,)
            torch.tensor([[10], [20], [30]], dtype=torch.float32),  # source shape = (3, 1)
            id="test_masked_scatter_3",
            marks=pytest.mark.xfail(reason="RuntimeError: users.size() > 0"),
        ),
        # 2D tensors where mask has a different shape from input tensor but can be broadcasted
        pytest.param(
            torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32),  # input_tensor shape = (3, 2)
            torch.tensor([[True, False], [False, True], [True, True]], dtype=torch.bool),  # mask shape = (3, 2)
            torch.tensor([10, 20, 30, 40], dtype=torch.float32),  # source shape = (4,)
            id="test_masked_scatter_4",
            marks=pytest.mark.xfail(reason="RuntimeError: users.size() > 0"),
        ),
        # Test with a mask of all False (nothing should be replaced)
        pytest.param(
            torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32),  # input_tensor shape = (5,)
            torch.tensor([False, False, False, False, False], dtype=torch.bool),  # mask shape = (5,)
            torch.tensor([10, 20, 30], dtype=torch.float32),  # source shape = (3,)
            id="test_masked_scatter_5",
            marks=pytest.mark.xfail(reason="RuntimeError: users.size() > 0"),
        ),
        # Test with a mask of all True (everything should be replaced)
        pytest.param(
            torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32),  # input_tensor shape = (5,)
            torch.tensor([True, True, True, True, True], dtype=torch.bool),  # mask shape = (5,)
            torch.tensor([10, 20, 30, 40, 50], dtype=torch.float32),  # source shape = (5,)
            id="test_masked_scatter_6",
            marks=pytest.mark.xfail(reason="RuntimeError: users.size() > 0"),
        ),
        pytest.param(
            torch.zeros((4, 4), dtype=torch.float32),  # 2D input tensor
            torch.tensor(
                [
                    [True, False, True, False],  # Mask
                    [False, True, False, True],
                    [True, True, False, False],
                    [False, False, True, True],
                ]
            ),
            torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),  # Source tensor
            id="test_masked_scatter_7",
            marks=pytest.mark.xfail(reason="RuntimeError: users.size() > 0"),
        ),
    ],
)
@pytest.mark.push
def test_masked_scatter(forge_property_recorder, input_tensor, mask, source):
    class MaskedScatterModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, mask, source):
            # Apply masked_scatter
            return x.masked_scatter(mask, source)

    # Inputs for the test
    inputs = [input_tensor, mask, source]

    framework_model = MaskedScatterModule()
    compiled_model = forge.compile(framework_model, inputs, forge_property_handler=forge_property_recorder)

    # Verify outputs
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize(
    "input_tensor, dim, index, source",
    [
        pytest.param(
            torch.zeros(3, 5, dtype=torch.float32),  # Input tensor
            1,  # Dimension along which to scatter
            torch.tensor([[0, 1, 2, 3, 4], [3, 4, 0, 1, 2], [2, 3, 1, 0, 4]]),  # Indices
            torch.tensor(
                [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0], [11.0, 12.0, 13.0, 14.0, 15.0]]  # Source values
            ),
            id="scatter_basic",
        ),
        pytest.param(
            torch.zeros(2, 4, dtype=torch.float32),  # Small tensor
            0,  # Scatter along first dimension
            torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]]),  # Alternating indices
            torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
            id="scatter_alternating",
        ),
        pytest.param(
            torch.zeros(3, 3, dtype=torch.float32),  # Square tensor
            1,  # Scatter along second dimension
            torch.tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]]),  # Same index per row
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
            id="scatter_same_indices",
        ),
        pytest.param(
            torch.zeros(2, 2, 2, dtype=torch.float32),  # 3D tensor
            2,  # Scatter along last dimension
            torch.tensor([[[0, 1], [1, 0]], [[1, 0], [0, 1]]]),
            torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            id="scatter_3d",
        ),
        pytest.param(
            torch.zeros(4, 1, dtype=torch.float32),  # Column vector
            0,
            torch.tensor([[2], [1], [3], [0]]),  # Permutation indices
            torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
            id="scatter_column",
        ),
        pytest.param(
            torch.zeros(2, 3, dtype=torch.float32),
            0,
            torch.tensor([[0, 1, 0], [1, 0, 1]]),  # Overlapping indices
            torch.ones((2, 3), dtype=torch.float32),  # All ones source
            id="scatter_overlapping",
        ),
    ],
)
@pytest.mark.xfail(reason="AssertionError: Encountered unsupported op types. Check error logs for more details.")
@pytest.mark.push
def test_scatter(forge_property_recorder, input_tensor, dim, index, source):
    class ScatterModule(torch.nn.Module):
        def __init__(self, dim, index, source):
            super().__init__()
            self.dim = dim
            self.index = index
            self.source = source

        def forward(self, x):
            return torch.scatter(x, self.dim, self.index, self.source)

    # Inputs for the test
    inputs = [input_tensor]

    framework_model = ScatterModule(dim, index, source)
    compiled_model = forge.compile(framework_model, inputs, forge_property_handler=forge_property_recorder)

    # Verify outputs
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


# NOTES: from pytorch docs:
# WARNING:
# When indices are not unique, the behavior is non-deterministic
# (one of the values from src will be picked arbitrarily) and
# the gradient will be incorrect (it will be propagated to all locations
# in the source that correspond to the same index)!
@pytest.mark.parametrize(
    "input_tensor, dim, index, source, reduce_mode",
    [
        pytest.param(
            torch.zeros(3, 5, dtype=torch.float32),  # Input tensor
            1,  # Dimension along which to scatter
            torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]),  # Unique Indices
            torch.tensor(
                [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0], [11.0, 12.0, 13.0, 14.0, 15.0]]  # Source values
            ),
            "sum",  # Reduction mode (add values)
            id="scatter_reduce_sum",
        ),
        pytest.param(
            torch.ones(3, 4, dtype=torch.float32),
            0,
            torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]]),
            torch.tensor([[2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0], [10.0, 11.0, 12.0, 13.0]]),
            "prod",  # Multiply overlapping values
            id="scatter_reduce_prod",
        ),
        pytest.param(
            torch.full((2, 3, 2), 5.0, dtype=torch.float32),  # 3D tensor filled with 5.0
            1,
            torch.tensor([[[0, 1], [1, 2], [0, 1]], [[1, 0], [0, 2], [1, 0]]]),
            torch.ones((2, 3, 2), dtype=torch.float32),
            "mean",  # Average overlapping values
            id="scatter_reduce_mean_3d",
        ),
        pytest.param(
            torch.zeros(4, 2, dtype=torch.float32),
            0,
            torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]]),
            torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
            "amax",  # Maximum of overlapping values
            id="scatter_reduce_amax",
        ),
        pytest.param(
            torch.full((2, 4), 10.0, dtype=torch.float32),
            1,
            torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]]),
            torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
            "amin",  # Minimum of overlapping values
            id="scatter_reduce_amin",
        ),
        pytest.param(
            torch.zeros(3, 3, dtype=torch.float32),
            0,
            torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2]]),
            torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]),
            "sum",  # Sum with regular pattern
            id="scatter_reduce_sum_regular",
        ),
        pytest.param(
            torch.zeros(2, 2, 2, dtype=torch.float32),
            2,
            torch.tensor([[[0, 0], [0, 0]], [[0, 0], [0, 0]]]),
            torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
            "mean",  # Mean with all-same indices
            id="scatter_reduce_mean_same_indices",
        ),
    ],
)
@pytest.mark.xfail(reason="Encountered unsupported op node type: scatter_elements, on device: tt")
@pytest.mark.push
def test_scatter_reduce(forge_property_recorder, input_tensor, dim, index, source, reduce_mode):
    class ScatterReduceModule(torch.nn.Module):
        def __init__(self, dim, index, source, reduce_mode):
            super().__init__()
            self.dim = dim
            self.index = index
            self.source = source
            self.reduce_mode = reduce_mode

        def forward(self, x):
            return torch.scatter_reduce(x, self.dim, self.index, self.source, reduce=self.reduce_mode)

    # Inputs for the test
    inputs = [input_tensor]

    framework_model = ScatterReduceModule(dim, index, source, reduce_mode)
    compiled_model = forge.compile(framework_model, inputs, forge_property_handler=forge_property_recorder)

    # Verify outputs
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


# NOTES: from pytorch docs:
# This operation may behave nondeterministically when given tensors on a CUDA device.
# See Reproducibility for more information.
@pytest.mark.parametrize(
    "input_tensor, dim, index, source",
    [
        pytest.param(
            torch.zeros(3, 5, dtype=torch.float32),
            1,
            torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]),
            torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0], [11.0, 12.0, 13.0, 14.0, 15.0]]),
            id="scatter_add_valid",
        ),
        pytest.param(
            torch.zeros(2, 4, dtype=torch.float32),
            0,
            torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]]),
            torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
            id="scatter_add_dim0",
        ),
        pytest.param(
            torch.zeros(3, 3, dtype=torch.float32),
            1,
            torch.tensor([[0, 1, 2], [1, 2, 0], [2, 0, 1]]),
            torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]),
            id="scatter_add_repeated_indices",
        ),
        pytest.param(
            torch.zeros(2, 2, 2, dtype=torch.float32),
            2,
            torch.tensor([[[0, 1], [1, 0]], [[1, 0], [0, 1]]]),
            torch.ones(2, 2, 2, dtype=torch.float32),
            id="scatter_add_3d",
        ),
        pytest.param(
            torch.zeros(4, dtype=torch.float32),
            0,
            torch.tensor([0, 1, 1, 2]),
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            id="scatter_add_1d",
        ),
    ],
)
@pytest.mark.xfail(reason="Encountered unsupported op node type: scatter_elements, on device: tt")
@pytest.mark.push
def test_scatter_add(forge_property_recorder, input_tensor, dim, index, source):
    class ScatterAddModule(torch.nn.Module):
        def __init__(self, dim, index, source):
            super().__init__()
            self.dim = dim
            self.index = index
            self.source = source

        def forward(self, x):
            # Apply scatter_add
            return torch.scatter_add(x, self.dim, self.index, self.source)

    # Inputs for the test
    inputs = [input_tensor]

    framework_model = ScatterAddModule(dim, index, source)
    compiled_model = forge.compile(framework_model, inputs, forge_property_handler=forge_property_recorder)

    # Verify outputs
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize(
    "input_tensor, source, offset, dim1, dim2",
    [
        pytest.param(
            torch.zeros(3, 3, dtype=torch.float32),  # Input tensor
            torch.tensor([1.0, 2.0, 3.0]),  # Source values for diagonal
            0,  # Offset: Main diagonal
            0,  # First dimension of diagonal
            1,  # Second dimension of diagonal
            id="diagonal_scatter_main_diagonal",
        ),
        pytest.param(
            torch.zeros(4, 4, dtype=torch.float32),  # Input tensor
            torch.tensor([10.0, 20.0, 30.0]),  # Source values for diagonal
            -1,  # Offset: Below main diagonal
            0,  # First dimension of diagonal
            1,  # Second dimension of diagonal
            id="diagonal_scatter_lower_diagonal",
        ),
        pytest.param(
            torch.zeros(5, 5, dtype=torch.float32),  # Input tensor
            torch.tensor([100.0, 200.0, 300.0, 400.0]),  # Source values for diagonal
            1,  # Offset: Above main diagonal
            0,  # First dimension of diagonal
            1,  # Second dimension of diagonal
            id="diagonal_scatter_upper_diagonal",
        ),
    ],
)
@pytest.mark.xfail(
    reason="NotImplementedError: The following operators are not implemented: ['aten::diagonal_scatter']"
)
@pytest.mark.push
def test_diagonal_scatter(forge_property_recorder, input_tensor, source, offset, dim1, dim2):
    class DiagonalScatterModule(torch.nn.Module):
        def __init__(self, source, offset, dim1, dim2):
            super().__init__()
            self.source = source
            self.offset = offset
            self.dim1 = dim1
            self.dim2 = dim2

        def forward(self, x):
            # Apply diagonal_scatter
            return x.diagonal_scatter(self.source, offset=self.offset, dim1=self.dim1, dim2=self.dim2)

    # Inputs for the test
    inputs = [input_tensor]

    framework_model = DiagonalScatterModule(source, offset, dim1, dim2)
    compiled_model = forge.compile(framework_model, inputs, forge_property_handler=forge_property_recorder)

    # Verify outputs
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize(
    "input_tensor, source, dim, index",
    [
        pytest.param(
            torch.zeros(3, 5, dtype=torch.float32),  # Input tensor
            torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0]),  # Source values
            0,  # Dimension to scatter
            1,  # Index to replace
            id="select_scatter_dim0",
        ),
        pytest.param(
            torch.ones(3, 4, dtype=torch.float32),  # Input tensor
            torch.tensor([100.0, 200.0, 300.0]),  # Source values
            1,  # Dimension to scatter
            2,  # Index to replace
            id="select_scatter_dim1",
        ),
    ],
)
@pytest.mark.xfail(reason="NotImplementedError: The following operators are not implemented: ['aten::select_scatter']")
@pytest.mark.push
def test_select_scatter(forge_property_recorder, input_tensor, source, dim, index):
    class SelectScatterModule(torch.nn.Module):
        def __init__(self, source, dim, index):
            super().__init__()
            self.source = source
            self.dim = dim
            self.index = index

        def forward(self, x):
            # Apply select_scatter
            return torch.select_scatter(x, self.source, self.dim, self.index)

    # Inputs for the test
    inputs = [input_tensor]

    framework_model = SelectScatterModule(source, dim, index)
    compiled_model = forge.compile(framework_model, inputs, forge_property_handler=forge_property_recorder)

    # Verify outputs
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.parametrize(
    "input_tensor, src, dim, start, end, step",
    [
        pytest.param(
            torch.zeros(4, 8, dtype=torch.float32),  # Input tensor
            torch.tensor(
                [
                    [1.0, 2.0, 3.0, 4.0],  # Source tensor
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0],
                ]
            ),
            1,  # Scatter along dimension 1
            2,  # Start index
            6,  # End index
            1,  # Step size
            id="step_1",
        ),
        pytest.param(
            torch.zeros(4, 8, dtype=torch.float32),  # Input tensor
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]),  # Source tensor
            1,  # Scatter along dimension 1
            1,  # Start index
            7,  # End index
            2,  # Step size
            id="step_2",
        ),
    ],
)
@pytest.mark.xfail(reason="NotImplementedError: The following operators are not implemented: ['aten::slice_scatter']")
@pytest.mark.push
def test_slice_scatter(forge_property_recorder, input_tensor, src, dim, start, end, step):
    class SliceScatterModule(torch.nn.Module):
        def __init__(self, source, dim, start, end, step):
            super().__init__()
            self.source = source
            self.dim = dim
            self.start = start
            self.end = end
            self.step = step

        def forward(self, x):
            # Perform slice_scatter operation
            return torch.slice_scatter(x, self.source, dim=dim, start=start, end=end, step=step)

    # Inputs for the test
    inputs = [input_tensor]

    framework_model = SliceScatterModule(src, dim, start, end, step)
    compiled_model = forge.compile(framework_model, inputs, forge_property_handler=forge_property_recorder)

    # Verify outputs
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
