# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch import nn

import forge
from forge.verify.verify import verify

import math
from forge.config import CompilerConfig
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AllCloseValueChecker

from test.operators.utils.compat import create_torch_inputs, verify_module_for_inputs
from test.operators.utils.datatypes import ValueRanges
from test.operators.utils.utils import TensorUtils


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
def test_take(tensor_and_indices):
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
    compiled_model = forge.compile(framework_model, inputs)

    # Run verification
    verify(inputs, framework_model, compiled_model)


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
def test_index_add(input_dim_index_source):
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
    compiled_model = forge.compile(framework_model, inputs)

    # Run verification
    verify(inputs, framework_model, compiled_model)


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
def test_index_fill(input_dim_index_value):
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
    compiled_model = forge.compile(framework_model, inputs)

    # Run verification
    verify(inputs, framework_model, compiled_model)


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
            marks=pytest.mark.xfail(reason="IndexCopy - unsupported op in lowering to TTIR"),
        ),
        pytest.param(
            (torch.zeros(4, 4, 4, dtype=torch.float32), 1, torch.tensor([1, 3]), torch.ones(4, 2, 4)),
            marks=pytest.mark.xfail(reason="IndexCopy - unsupported op in lowering to TTIR"),
        ),
    ],
)
@pytest.mark.push
def test_index_copy(input_dim_index_source):
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
    compiled_model = forge.compile(framework_model, inputs)

    # Run verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize(
    "input_dim_index_source",
    [
        # fill cache equivalent. Dims of first operand (cache) are: (batch_size, num_heads, max_cache_len, head_dim)
        pytest.param(
            (torch.zeros(1, 8, 32, 100, dtype=torch.float32), 2, torch.tensor([0, 1, 2, 3]), torch.ones(1, 8, 4, 100)),
        ),
        # update cache equivalent
        pytest.param(
            (torch.zeros(1, 8, 32, 100, dtype=torch.float32), 2, torch.tensor([14]), torch.ones(1, 8, 1, 100)),
        ),
    ],
)
@pytest.mark.push
def test_index_copy_kv_cache(input_dim_index_source):
    """
    This test simulates the behavior of copying values into a key-value cache using index_copy.
    """
    cache, dim, index, update_cache = input_dim_index_source

    class IndexCopyModule(nn.Module):
        def __init__(self, dim, index, cache):
            super().__init__()
            self.dim = dim
            self.index = index
            self.cache = cache

        def forward(self, update_cache):
            # Copy values from `source` into `x` at indices `index` along `dim`
            return torch.index_copy(self.cache, self.dim, self.index, update_cache)

    # Inputs for the test
    inputs = [update_cache]

    framework_model = IndexCopyModule(dim, index, cache)
    compiled_model = forge.compile(framework_model, inputs)

    # Run verification
    verify(inputs, framework_model, compiled_model)


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
def test_index_select(input_dim_index):
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
    compiled_model = forge.compile(framework_model, inputs)

    # Run verification
    verify(inputs, framework_model, compiled_model)


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
def test_index_put(input_indices_values_accumulate):
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
    compiled_model = forge.compile(framework_model, inputs)

    # Run verification
    verify(inputs, framework_model, compiled_model)


# INDEX_COPY - FILL CACHE TESTS THAT ARE FAILING:

# error message:
## E       RuntimeError: TT_FATAL @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/kv_cache/device/update_cache_op.cpp:56: (num_blocks_of_work <= compute_with_storage_grid_size.x * compute_with_storage_grid_size.y)
# test ids with this error:
# test_fill_cache_error[{'index': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 'dim': 2}-(1, 100, 100, 100)-ModelFromAnotherOp]
# test_fill_cache_error[{'index': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 'dim': 2}-(1, 100, 100, 100)-ModelDirect]
# test_fill_cache_error[{'index': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 'dim': 2}-(1, 100, 100, 100)-ModelConstEvalPass]

# error message:
# E       RuntimeError: TT_FATAL @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/kv_cache/device/update_cache_op.cpp:38: input_tensor.padded_shape()[0] == 1
# test ids with this error:
# test_fill_cache_error[{'index': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 'dim': 2}-(3, 11, 45, 17)-ModelFromAnotherOp]
# test_fill_cache_error[{'index': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 'dim': 2}-(3, 11, 45, 17)-ModelDirect]
# test_fill_cache_error[{'index': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 'dim': 2}-(3, 11, 45, 17)-ModelConstEvalPass]

### run all: pytest -svv -rap 'forge/test/mlir/operators/indexing/test_advanced_indexing.py::test_fill_cache_error'
### run single: pytest -svv -rap "forge/test/mlir/operators/indexing/test_advanced_indexing.py::test_fill_cache_error[{'index': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 'dim': 2}-(1, 100, 100, 100)-ModelFromAnotherOp]"


def make_source_shape(input_shape, dim, index):
    m = len(index)
    source_shape = list(input_shape)
    source_shape[dim] = m
    return tuple(source_shape)


@pytest.mark.parametrize("model_type", ["ModelFromAnotherOp", "ModelDirect", "ModelConstEvalPass"])
@pytest.mark.parametrize(
    "input, dim, index",
    [
        pytest.param(
            (1, 100, 100, 100),
            2,
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
            id="{'index': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 'dim': 2}-(1, 100, 100, 100)",
        ),
        pytest.param(
            (3, 11, 45, 17),
            2,
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
            id="{'index': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 'dim': 2}-(3, 11, 45, 17)",
        ),
    ],
)
def test_fill_cache_error(model_type, input, dim, index):
    class ModelFromAnotherOp(torch.nn.Module):
        def __init__(self, dim, index):
            super().__init__()
            self.dim = dim
            self.index = index

        def forward(self, x, y):
            xx = torch.add(x, x)
            yy = torch.add(y, y)
            return torch.Tensor.index_copy(xx, self.dim, self.index, yy)

    class ModelDirect(torch.nn.Module):
        def __init__(self, dim, index):
            super().__init__()
            self.dim = dim
            self.index = index

        def forward(self, x, y):
            return torch.Tensor.index_copy(x, self.dim, self.index, y)

    class ModelConstEvalPass(torch.nn.Module):
        def __init__(self, self_shape, source_shape, dim, index, dtype, value_range):
            super(ModelConstEvalPass, self).__init__()
            self.dim = dim
            self.index = index
            c1 = TensorUtils.create_torch_constant(
                input_shape=self_shape,
                dev_data_format=dtype,
                value_range=value_range,
                random_seed=math.prod(self_shape),
            )
            c2 = TensorUtils.create_torch_constant(
                input_shape=source_shape,
                dev_data_format=dtype,
                value_range=value_range,
                random_seed=math.prod(source_shape),
            )
            self.register_buffer("c1", c1)
            self.register_buffer("c2", c2)

        def forward(self, x: torch.Tensor, y: torch.Tensor):
            v1 = torch.Tensor.index_copy(self.c1, self.dim, self.index, self.c2)
            # v2 = torch.add(x, x)
            v2 = torch.Tensor.index_copy(x, self.dim, self.index, y)
            # add consume inputs
            add = torch.add(v1, v2)
            return add

    input_shape, dim, index = input, dim, index
    value_range = ValueRanges.SMALL  # [-1, 1]

    index = torch.tensor(index, dtype=torch.long)
    self_shape = input_shape
    source_shape = make_source_shape(input_shape, dim, index)
    input_shapes = [self_shape, source_shape]
    # prepare model

    if model_type == "ModelConstEvalPass":
        framework_model = eval(model_type)(
            self_shape=self_shape,
            source_shape=source_shape,
            dim=dim,
            index=index,
            dtype=None,  # torch.float32
            value_range=value_range,
        )
    else:
        framework_model = eval(model_type)(
            dim=dim,
            index=index,
        )

    # prepare inputs
    inputs = create_torch_inputs(
        input_shapes=input_shapes,
        dev_data_format=None,  # torch.float32
        value_range=value_range,  # [-1, 1]
    )

    verify_config = VerifyConfig(value_checker=AllCloseValueChecker(rtol=1e-2, atol=1e-2))

    # this 2 lines will be executed in method `verify_module_for_inputs`, need to be like this because of pcc error level handling logic that is writthen in `verify_module_for_inputs`
    ## compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    ## verify(inputs, framework_model, compiled_model, verify_cfg=verify_cfg)
    verify_module_for_inputs(
        model=framework_model,
        inputs=inputs,
        compiler_cfg=CompilerConfig(),
        verify_config=verify_config,
        convert_to_forge=False,
    )
