# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch import nn

import forge
from forge.verify.verify import verify


@pytest.mark.parametrize(
    "input_tensor, offset, dim1, dim2",
    [
        pytest.param(
            torch.arange(1, 10, dtype=torch.float32).reshape(3, 3),  # 2D square matrix
            0,  # Main diagonal
            0,  # First dimension
            1,  # Second dimension
            id="2D_main_diagonal",
        ),
        pytest.param(
            torch.arange(1, 10, dtype=torch.float32).reshape(3, 3),  # 2D square matrix
            1,  # Diagonal above the main diagonal
            0,  # First dimension
            1,  # Second dimension
            id="2D_upper_diagonal",
        ),
        pytest.param(
            torch.arange(1, 10, dtype=torch.float32).reshape(3, 3),  # 2D square matrix
            -1,  # Diagonal below the main diagonal
            0,  # First dimension
            1,  # Second dimension
            id="2D_lower_diagonal",
        ),
        pytest.param(
            torch.arange(27, dtype=torch.float32).reshape(3, 3, 3),  # 3D tensor
            0,  # Main diagonal along dim1 and dim2
            1,  # Second dimension
            2,  # Third dimension
            id="3D_tensor_diagonal_dim1_dim2",
        ),
        pytest.param(
            torch.arange(27, dtype=torch.float32).reshape(3, 3, 3),  # 3D tensor
            0,  # Main diagonal along dim0 and dim2
            0,  # First dimension
            2,  # Third dimension
            id="3D_tensor_diagonal_dim0_dim2",
        ),
    ],
)
@pytest.mark.xfail(reason="NotImplementedError: The following operators are not implemented: ['aten::diagonal']")
def test_diagonal(input_tensor, offset, dim1, dim2):
    class DiagonalModule(nn.Module):
        def __init__(self, offset, dim1, dim2):
            super().__init__()
            self.offset = offset
            self.dim1 = dim1
            self.dim2 = dim2

        def forward(self, x):
            return torch.diagonal(x, offset=self.offset, dim1=self.dim1, dim2=self.dim2)

    # Inputs for the test
    inputs = [input_tensor]

    framework_model = DiagonalModule(offset, dim1, dim2)
    compiled_model = forge.compile(framework_model, inputs)

    # Run verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "input_tensor, diagonal",
    [
        pytest.param(
            torch.arange(1, 10, dtype=torch.float32).reshape(3, 3),  # Square matrix
            0,  # Main diagonal
            id="2D_main_diagonal",
        ),
        pytest.param(
            torch.arange(1, 10, dtype=torch.float32).reshape(3, 3),
            1,  # Above main diagonal
            id="2D_above_diagonal",
        ),
        pytest.param(
            torch.arange(1, 10, dtype=torch.float32).reshape(3, 3),
            -1,  # Below main diagonal
            id="2D_below_diagonal",
        ),
        pytest.param(
            torch.arange(1, 9, dtype=torch.float32).reshape(2, 4),  # Rectangular matrix
            0,
            id="2D_rectangular",
        ),
        pytest.param(
            torch.arange(1, 6, dtype=torch.float32),  # 1D input
            0,
            id="1D_to_2D",
        ),
    ],
)
@pytest.mark.xfail(reason="NotImplementedError: The following operators are not implemented: ['aten::diag']")
def test_diag(input_tensor, diagonal):
    class DiagModule(nn.Module):
        def __init__(self, diagonal):
            super().__init__()
            self.diagonal = diagonal

        def forward(self, x):
            return torch.diag(x, diagonal=self.diagonal)

    inputs = [input_tensor]

    framework_model = DiagModule(diagonal)
    compiled_model = forge.compile(framework_model, inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "input_tensor, offset, dim1, dim2",
    [
        pytest.param(
            torch.arange(1, 4, dtype=torch.float32),  # 1D vector
            0,  # Main diagonal
            -2,  # Second-to-last dim
            -1,  # Last dim
            id="1D_default_dims",
        ),
        pytest.param(
            torch.arange(1, 4, dtype=torch.float32),
            1,  # Above main diagonal
            0,  # First dim
            1,  # Second dim
            id="1D_custom_dims_above",
        ),
        pytest.param(
            torch.arange(1, 4, dtype=torch.float32),
            -1,  # Below main diagonal
            0,  # First dim
            1,  # Second dim
            id="1D_custom_dims_below",
        ),
        pytest.param(
            torch.arange(1, 10, dtype=torch.float32).reshape(3, 3),  # 2D input
            0,
            1,  # Second dim
            2,  # Third dim
            id="2D_input_3D_output",
        ),
        pytest.param(
            torch.arange(27, dtype=torch.float32).reshape(3, 3, 3),  # 3D input
            0,
            2,  # Third dim
            3,  # Fourth dim
            id="3D_input_4D_output",
        ),
    ],
)
@pytest.mark.xfail(reason="NotImplementedError: The following operators are not implemented: ['aten::diag_embed']")
def test_diag_embed(input_tensor, offset, dim1, dim2):
    class DiagEmbedModule(nn.Module):
        def __init__(self, offset, dim1, dim2):
            super().__init__()
            self.offset = offset
            self.dim1 = dim1
            self.dim2 = dim2

        def forward(self, x):
            return torch.diag_embed(x, offset=self.offset, dim1=self.dim1, dim2=self.dim2)

    inputs = [input_tensor]

    framework_model = DiagEmbedModule(offset, dim1, dim2)
    compiled_model = forge.compile(framework_model, inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "input_tensor, diagonal",
    [
        pytest.param(
            torch.arange(1, 10, dtype=torch.float32).reshape(3, 3),  # 2D square matrix
            0,  # Main diagonal
            id="2D_main_diagonal",
        ),
        pytest.param(
            torch.arange(1, 10, dtype=torch.float32).reshape(3, 3),  # 2D square matrix
            1,  # One above main diagonal
            id="2D_above_main_diagonal",
        ),
        pytest.param(
            torch.arange(1, 10, dtype=torch.float32).reshape(3, 3),  # 2D square matrix
            -1,  # One below main diagonal
            id="2D_below_main_diagonal",
        ),
        pytest.param(
            torch.arange(1, 9, dtype=torch.float32).reshape(2, 4),  # 2D rectangular matrix
            0,  # Main diagonal
            id="2D_rectangular",
        ),
        pytest.param(
            torch.arange(27, dtype=torch.float32).reshape(3, 3, 3),  # 3D tensor
            0,  # Main diagonal
            id="3D_tensor",
        ),
    ],
)
@pytest.mark.xfail(reason="AssertionError: Data mismatch on output 0 between framework and Forge codegen")
def test_triu(input_tensor, diagonal):
    class TriuModule(nn.Module):
        def __init__(self, diagonal):
            super().__init__()
            self.diagonal = diagonal

        def forward(self, x):
            return torch.triu(x, diagonal=self.diagonal)

    # Inputs for the test
    inputs = [input_tensor]

    framework_model = TriuModule(diagonal)
    compiled_model = forge.compile(framework_model, inputs)

    # Run verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "input_tensor, diagonal",
    [
        pytest.param(
            torch.arange(1, 10, dtype=torch.float32).reshape(3, 3),  # 2D square matrix
            0,  # Main diagonal
            id="2D_main_diagonal",
        ),
        pytest.param(
            torch.arange(1, 10, dtype=torch.float32).reshape(3, 3),  # 2D square matrix
            1,  # One above main diagonal
            id="2D_above_main_diagonal",
        ),
        pytest.param(
            torch.arange(1, 10, dtype=torch.float32).reshape(3, 3),  # 2D square matrix
            -1,  # One below main diagonal
            id="2D_below_main_diagonal",
        ),
        pytest.param(
            torch.arange(1, 9, dtype=torch.float32).reshape(2, 4),  # 2D rectangular matrix
            0,  # Main diagonal
            id="2D_rectangular",
        ),
        pytest.param(
            torch.arange(27, dtype=torch.float32).reshape(3, 3, 3),  # 3D tensor
            0,  # Main diagonal
            id="3D_tensor",
        ),
    ],
)
def test_tril(input_tensor, diagonal):
    class TrilModule(nn.Module):
        def __init__(self, diagonal):
            super().__init__()
            self.diagonal = diagonal

        def forward(self, x):
            return torch.tril(x, diagonal=self.diagonal)

    inputs = [input_tensor]

    framework_model = TrilModule(diagonal)
    compiled_model = forge.compile(framework_model, inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "input_tensor, indices, dim",
    [
        pytest.param(
            torch.arange(1, 10, dtype=torch.float32).reshape(3, 3),  # 2D: [3,3]
            torch.tensor([[0, 1], [1, 2], [0, 2]], dtype=torch.int64),  # Must be [3,2]
            1,
            id="2D_last_dim",
        ),
        pytest.param(
            torch.arange(1, 10, dtype=torch.float32).reshape(3, 3),  # [3,3]
            torch.tensor([[0], [1], [2]], dtype=torch.int64),  # [3,1] - correct
            0,
            id="2D_first_dim",
        ),
        pytest.param(
            torch.arange(27, dtype=torch.float32).reshape(3, 3, 3),  # [3,3,3]
            torch.tensor(
                [[[0, 1], [1, 2], [0, 2]], [[1, 2], [0, 1], [1, 0]], [[2, 0], [2, 1], [0, 1]]], dtype=torch.int64
            ),  # Must be [3,3,2]
            2,
            id="3D_last_dim",
        ),
        pytest.param(
            torch.arange(27, dtype=torch.float32).reshape(3, 3, 3),  # [3,3,3]
            torch.tensor([[[0], [1], [2]], [[1], [2], [0]], [[2], [0], [1]]], dtype=torch.int64),  # [3,3,1]
            1,
            id="3D_middle_dim",
        ),
        pytest.param(
            torch.arange(27, dtype=torch.float32).reshape(3, 3, 3),  # [3,3,3]
            torch.tensor([[[0]], [[1]], [[2]]], dtype=torch.int64),  # [3,1,1] - correct
            0,
            id="3D_first_dim",
        ),
    ],
)
@pytest.mark.xfail(reason="NotImplementedError: The following operators are not implemented: ['aten::take_along_dim']")
def test_take_along_dim(input_tensor, indices, dim):
    class TakeAlongDimModule(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, x, indices):
            return torch.take_along_dim(x, indices, dim=self.dim)

    inputs = [input_tensor, indices]

    framework_model = TakeAlongDimModule(dim)
    compiled_model = forge.compile(framework_model, inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "input_tensor, index, dim, sparse_grad",
    [
        pytest.param(
            torch.arange(1, 10, dtype=torch.float32).reshape(3, 3),
            torch.tensor([[0, 1], [1, 2], [0, 2]], dtype=torch.int64),
            1,
            False,
            id="2D_gather_dim1_dense",
            marks=pytest.mark.xfail(
                reason='Failed on "DecomposeRoll" TVM callback: index 2 is out of bounds for axis 1 with size 2'
            ),
        ),
        pytest.param(
            torch.arange(1, 10, dtype=torch.float32).reshape(3, 3),
            torch.tensor([[0, 1], [1, 2], [0, 2]], dtype=torch.int64),
            1,
            True,
            id="2D_gather_dim1_sparse",
            marks=pytest.mark.xfail(
                reason='Failed on "DecomposeRoll" TVM callback: index 2 is out of bounds for axis 1 with size 2'
            ),
        ),
        pytest.param(
            torch.arange(27, dtype=torch.float32).reshape(3, 3, 3),
            torch.tensor(
                [[[0, 1], [1, 2], [0, 2]], [[1, 2], [0, 1], [2, 0]], [[2, 0], [1, 0], [1, 2]]], dtype=torch.int64
            ),
            2,
            False,
            id="3D_gather_dim2_dense",
            marks=pytest.mark.xfail(
                reason='Failed on "DecomposeRoll" TVM callback: index 2 is out of bounds for axis 2 with size 2'
            ),
        ),
        pytest.param(
            torch.arange(27, dtype=torch.float32).reshape(3, 3, 3),
            torch.tensor(
                [[[0, 1], [1, 2], [0, 2]], [[1, 2], [0, 1], [2, 0]], [[2, 0], [1, 0], [1, 2]]], dtype=torch.int64
            ),
            2,
            True,
            id="3D_gather_dim2_sparse",
            marks=pytest.mark.xfail(
                reason='Failed on "DecomposeRoll" TVM callback: index 2 is out of bounds for axis 2 with size 2'
            ),
        ),
    ],
)
def test_gather(input_tensor, index, dim, sparse_grad):
    class GatherModule(nn.Module):
        def __init__(self, dim, sparse_grad, index):
            super().__init__()
            self.dim = dim
            self.sparse_grad = sparse_grad
            self.index = index

        def forward(self, x):
            return torch.gather(x, self.dim, self.index, sparse_grad=self.sparse_grad)

    inputs = [input_tensor]

    framework_model = GatherModule(dim, sparse_grad, index)
    compiled_model = forge.compile(framework_model, inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "indices, shape",
    [
        pytest.param(
            torch.tensor([4, 5], dtype=torch.int64),
            (3, 3),
            id="2D_shape",
        ),
        pytest.param(
            torch.tensor([0, 8], dtype=torch.int64),
            (3, 3),
            id="2D_shape_corners",
        ),
        pytest.param(
            torch.tensor([13, 14], dtype=torch.int64),
            (3, 3, 3),
            id="3D_shape",
        ),
        pytest.param(
            torch.tensor([0, 26], dtype=torch.int64),
            (3, 3, 3),
            id="3D_shape_corners",
        ),
        pytest.param(
            torch.tensor([40, 41], dtype=torch.int64),
            (3, 4, 2, 2),
            id="4D_shape",
        ),
    ],
)
@pytest.mark.xfail(reason="Not supported in our version of pytorch")
def test_unravel_index(indices, shape):
    class UnravelIndexModule(nn.Module):
        def __init__(self, shape):
            super().__init__()
            self.shape = shape

        def forward(self, indices):
            return torch.unravel_index(indices, self.shape)

    inputs = [indices]

    framework_model = UnravelIndexModule(shape)
    compiled_model = forge.compile(framework_model, inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "input_tensor, indices, values",
    [
        pytest.param(
            torch.arange(1, 10, dtype=torch.float32).reshape(3, 3),
            torch.tensor([0, 4, 8], dtype=torch.int64),
            torch.tensor([99, 88, 77], dtype=torch.float32),
            id="2D_tensor",
        ),
        pytest.param(
            torch.arange(27, dtype=torch.float32).reshape(3, 3, 3),
            torch.tensor([0, 13, 26], dtype=torch.int64),
            torch.tensor([99, 88, 77], dtype=torch.float32),
            id="3D_tensor",
        ),
        pytest.param(
            torch.arange(1, 10, dtype=torch.float32),
            torch.tensor([0, 4, 8], dtype=torch.int64),
            torch.tensor([99, 88, 77], dtype=torch.float32),
            id="1D_tensor",
        ),
        pytest.param(
            torch.arange(16, dtype=torch.float32).reshape(2, 2, 2, 2),
            torch.tensor([0, 7, 15], dtype=torch.int64),
            torch.tensor([99, 88, 77], dtype=torch.float32),
            id="4D_tensor",
        ),
    ],
)
@pytest.mark.xfail(reason="NotImplementedError: The following operators are not implemented: ['aten::put']")
def test_put(input_tensor, indices, values):
    class PutModule(nn.Module):
        def __init__(self, indices, values):
            super().__init__()
            self.indices = indices
            self.values = values

        def forward(self, x):
            return torch.put(x, self.indices, self.values)

    inputs = [input_tensor]

    framework_model = PutModule(indices, values)
    compiled_model = forge.compile(framework_model, inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "input_tensor, sorted, return_inverse, return_counts, dim",
    [
        pytest.param(
            torch.tensor([1, 3, 2, 3, 1, 2], dtype=torch.float32),
            True,
            False,
            False,
            None,
            id="1D_sorted",
            marks=pytest.mark.xfail(reason="tvm.error.InternalError: Traceback (most recent call last):"),
        ),
        pytest.param(
            torch.tensor([1, 3, 2, 3, 1, 2], dtype=torch.float32),
            False,
            True,
            True,
            None,
            id="1D_with_inverse_counts",
            marks=pytest.mark.xfail(reason="AssertionError: Dynamic shapes not supported"),
        ),
        pytest.param(
            torch.arange(1, 10, dtype=torch.float32).reshape(3, 3).repeat(1, 2),
            True,
            False,
            True,
            1,
            id="2D_dim1",
            marks=pytest.mark.xfail(
                reason="NotImplementedError: The following operators are not implemented: ['aten::unique_dim']"
            ),
        ),
        pytest.param(
            torch.arange(27, dtype=torch.float32).reshape(3, 3, 3).repeat(1, 1, 2),
            True,
            True,
            False,
            2,
            id="3D_dim2",
            marks=pytest.mark.xfail(
                reason="NotImplementedError: The following operators are not implemented: ['aten::unique_dim']"
            ),
        ),
    ],
)
def test_unique(input_tensor, sorted, return_inverse, return_counts, dim):
    class UniqueModule(nn.Module):
        def __init__(self, sorted, return_inverse, return_counts, dim):
            super().__init__()
            self.sorted = sorted
            self.return_inverse = return_inverse
            self.return_counts = return_counts
            self.dim = dim

        def forward(self, x):
            return torch.unique(
                x,
                sorted=self.sorted,
                return_inverse=self.return_inverse,
                return_counts=self.return_counts,
                dim=self.dim,
            )

    inputs = [input_tensor]

    framework_model = UniqueModule(sorted, return_inverse, return_counts, dim)
    compiled_model = forge.compile(framework_model, inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "input_tensor, return_inverse, return_counts, dim",
    [
        pytest.param(
            torch.tensor([1, 1, 2, 2, 3, 1, 1, 2], dtype=torch.float32),
            False,
            False,
            None,
            id="1D_basic",
        ),
        pytest.param(
            torch.tensor([1, 1, 2, 2, 3, 1, 1, 2], dtype=torch.float32),
            True,
            True,
            None,
            id="1D_with_inverse_counts",
        ),
        pytest.param(
            torch.tensor([[1, 1, 2], [2, 2, 2], [3, 3, 3]], dtype=torch.float32),
            False,
            True,
            0,
            id="2D_dim0",
        ),
        pytest.param(
            torch.arange(27, dtype=torch.float32).reshape(3, 3, 3).repeat(1, 1, 2),
            True,
            False,
            2,
            id="3D_dim2",
        ),
    ],
)
@pytest.mark.xfail(
    reason="NotImplementedError: The following operators are not implemented: ['aten::unique_consecutive']"
)
def test_unique_consecutive(input_tensor, return_inverse, return_counts, dim):
    class UniqueConsecutiveModule(nn.Module):
        def __init__(self, return_inverse, return_counts, dim):
            super().__init__()
            self.return_inverse = return_inverse
            self.return_counts = return_counts
            self.dim = dim

        def forward(self, x):
            return torch.unique_consecutive(
                x, return_inverse=self.return_inverse, return_counts=self.return_counts, dim=self.dim
            )

    inputs = [input_tensor]

    framework_model = UniqueConsecutiveModule(return_inverse, return_counts, dim)
    compiled_model = forge.compile(framework_model, inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "input_tensor1, input_tensor2",
    [
        pytest.param(
            torch.arange(-4, 5, dtype=torch.float32).reshape(3, 3),
            torch.full((3, 3), 99.0),
            id="2D_basic",
        ),
        pytest.param(
            torch.arange(-4, 5, dtype=torch.float32),
            torch.full((9,), 99.0),
            id="1D_tensor",
        ),
        pytest.param(
            torch.arange(-13, 14, dtype=torch.float32).reshape(3, 3, 3),
            torch.full((3, 3, 3), 99.0),
            id="3D_tensor",
        ),
    ],
)
@pytest.mark.xfail(reason="BinaryOpType cannot be mapped to BcastOpMath")
def test_where(input_tensor1, input_tensor2):
    class WhereModule(nn.Module):
        def __init__(self, input2):
            super().__init__()
            self.input2 = input2

        def forward(self, x):
            return torch.where(x > 0, x, self.input2)

    inputs = [input_tensor1]

    framework_model = WhereModule(input_tensor2)
    compiled_model = forge.compile(framework_model, inputs)

    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "input_tensor",
    [
        pytest.param(
            torch.arange(-4, 5, dtype=torch.float32).reshape(3, 3),
            id="2D_basic",
        ),
        pytest.param(
            torch.arange(-4, 5, dtype=torch.float32),
            id="1D_tensor",
        ),
        pytest.param(
            torch.arange(-13, 14, dtype=torch.float32).reshape(3, 3, 3),
            id="3D_tensor",
        ),
    ],
)
# @pytest.mark.xfail(reason="NotImplementedError: The following operators are not implemented: ['aten::argwhere']")
def test_argwhere(input_tensor):
    class ArgwhereModule(nn.Module):
        def forward(self, x):
            return torch.argwhere(x > 0)

    inputs = [input_tensor]

    framework_model = ArgwhereModule()
    compiled_model = forge.compile(framework_model, inputs)

    verify(inputs, framework_model, compiled_model)
