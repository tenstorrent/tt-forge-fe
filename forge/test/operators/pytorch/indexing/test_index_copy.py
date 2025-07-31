# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from enum import Enum
import torch
import random
import math

from typing import List, Dict
from loguru import logger

from forge._C import DataFormat

from forge.config import CompilerConfig
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AllCloseValueChecker

from test.operators.utils import VerifyUtils
from test.operators.utils import InputSource
from test.operators.utils import TestVector
from test.operators.utils import TestPlan
from test.operators.utils.compat import TestDevice
from test.operators.utils import TestCollection
from test.operators.utils import TestCollectionCommon
from test.operators.utils import ValueRanges
from test.operators.utils.utils import PytorchUtils, TensorUtils
from test.operators.pytorch.ids.loader import TestIdsDataLoader
from test.operators.utils.test_data import TestCollectionTorch


class ModelFromAnotherOp(torch.nn.Module):
    def __init__(self, operator, kwargs):
        super().__init__()
        self.testname = "index_copy_operator_test_op_src_from_another_op"
        self.operator = operator
        self.dim = kwargs.get("dim")
        self.index = kwargs.get("index")

    def forward(self, x, y):
        xx = torch.add(x, x)
        yy = torch.add(y, y)
        return self.operator(xx, self.dim, self.index, yy)


class ModelDirect(torch.nn.Module):
    def __init__(self, operator, kwargs):
        super().__init__()
        self.testname = "index_copy_operator_test_op_src_direct"
        self.operator = operator
        self.dim = kwargs.get("dim")
        self.index = kwargs.get("index")

    def forward(self, x, y):
        return self.operator(x, self.dim, self.index, y)
    

class ModelConstEvalPass(torch.nn.Module):
    def __init__(self, operator, self_shape, source_shape, kwargs, dtype, value_range):
        super(ModelConstEvalPass, self).__init__()
        self.testname = "index_copy_operator_test_op_src_const_eval_pass"
        self.operator = operator
        self.dim = kwargs.get("dim")
        self.index = kwargs.get("index")
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
        v1 = self.operator(self.c1, self.dim, self.index, self.c2)
        # v2 = torch.add(x, x)
        v2 = self.operator(x, self.dim, self.index, y)
        # add consume inputs
        add = torch.add(v1, v2)
        return add
    

class TestVerification:

    MODEL_TYPES = {
        InputSource.FROM_ANOTHER_OP: ModelFromAnotherOp,
        InputSource.FROM_HOST: ModelDirect,
        InputSource.CONST_EVAL_PASS: ModelConstEvalPass,
    }

    @classmethod
    def make_source_shape(cls, input_shape, dim, index):
        m = len(index)
        source_shape = list(input_shape)
        source_shape[dim] = m
        return tuple(source_shape)

    @classmethod
    def verify(
        cls,
        test_device: TestDevice,
        test_vector: TestVector,
        input_params: List[Dict] = [],
        warm_reset: bool = False,
    ):
        operator = PytorchUtils.get_op_class_by_name(test_vector.operator)
        
        value_range = ValueRanges.SMALL
        kwargs = test_vector.kwargs if test_vector.kwargs else {}
        kwargs["index"] = torch.tensor(kwargs["index"], dtype=torch.long)
        
        self_shape = test_vector.input_shape
        source_shape = cls.make_source_shape(test_vector.input_shape, kwargs["dim"], kwargs["index"])
        input_shapes = [self_shape, source_shape]
        logger.trace(f"***input_shapes: {input_shapes}")

        model_type = cls.MODEL_TYPES[test_vector.input_source]
        if test_vector.input_source == InputSource.CONST_EVAL_PASS:
            pytorch_model = model_type(
                operator=operator,
                self_shape=self_shape,
                source_shape=source_shape,
                kwargs=kwargs,
                dtype=test_vector.dev_data_format,
                value_range=value_range,
            )
        else:
            pytorch_model = model_type(
                operator=operator,
                kwargs=kwargs,
            )

        verify_config = VerifyConfig(value_checker=AllCloseValueChecker(rtol=1e-2, atol=1e-2))
        VerifyUtils.verify(
            model=pytorch_model,
            test_device=test_device,
            input_shapes=input_shapes,
            input_params=input_params,
            dev_data_format=test_vector.dev_data_format,
            math_fidelity=test_vector.math_fidelity,
            warm_reset=warm_reset,
            value_range=value_range,
            verify_config=verify_config,
        )


class TestParamsData:

    __test__ = False

    test_plan: TestPlan = None
    
    MAX_INDEX_SIZE = 20

    class TestType(Enum):
        INDEX_VALUES_SINGLE = 1
        INDEX_VALUES_IN_ORDER = 2
        INDEX_VALUES_REVERSE_ORDER = 3
        INDEX_VALUES_OUT_OF_ORDER = 4
        INDEX_VALUES_FULL_COVERAGE = 5

    @classmethod
    def generate_kwargs_test_type(cls, test_vector: TestVector, test_type: TestType, dim: int = None):
        rng = random.Random(sum(test_vector.input_shape))
        max_dim = len(test_vector.input_shape)
        for d in range(max_dim):
            N = test_vector.input_shape[d]
            index = None
            if N <= 1:
                continue  # no valid index for this dimension
            if test_type == cls.TestType.INDEX_VALUES_SINGLE:
                m = rng.randint(0, (N - 1))
                index = [m]  # single value index
            elif test_type == cls.TestType.INDEX_VALUES_IN_ORDER:
                m = rng.randint(1, min(N - 1, cls.MAX_INDEX_SIZE))
                if m == 1:
                    continue  # single value index is handled in separate test
                index = list(range(m))
            elif test_type == cls.TestType.INDEX_VALUES_REVERSE_ORDER:
                m = rng.randint(1, min(N - 1, cls.MAX_INDEX_SIZE))
                if m == 1:
                    continue  # single value index is handled in separate test
                index = list(range(m - 1, -1, -1))
            elif test_type == cls.TestType.INDEX_VALUES_OUT_OF_ORDER:
                m = rng.randint(1, min(N - 1, cls.MAX_INDEX_SIZE))
                index = rng.sample(range(N), m)
                if m == 1:
                    continue  # single value index is handled in separate test
                is_in_order = all(index[i] == index[i + 1] - 1 for i in range(len(index) - 1))
                if is_in_order:
                    continue  # in order index values test case is handled in separate test
            elif test_type == cls.TestType.INDEX_VALUES_FULL_COVERAGE:
                m = N - 1
                if m == 1:
                    continue  # single value index is handled in separate test
                index = list(range(m))
            if index is not None:
                if dim is not None:
                    if isinstance(dim, int):
                        if d == dim or (d - max_dim) == dim:
                            yield {
                                "index": index,
                                "dim": d,
                            }
                    elif isinstance(dim, list):
                        if d in dim or (d - max_dim) in dim:
                            yield {
                                "index": index,
                                "dim": d,
                            }
                elif dim is None:
                    yield {
                        "index": index,
                        "dim": d,
                    }

    

class TestCollectionData:

    __test__ = False

    test_collection: TestCollection = None

    operator_index_copy = ["index_copy"]
    operator_update_cache = ["update_cache"]
    operator_fill_cache = ["fill_cache"]


TestParamsData.test_plan = TestPlan(
    verify=lambda test_device, test_vector: TestVerification.verify(
        test_device,
        test_vector,
    ),
    collections=[
        # TESTS ONLY CURRENTLY SUPPORTED CASES (update_cache and fill_cache) - only 4D shapes, dim=-2, single and in order index tensor values
        # Test update_cache
        TestCollection(
            operators=TestCollectionData.operator_update_cache,
            input_sources=TestCollectionCommon.all.input_sources,
            input_shapes=[
                shape for shape in TestCollectionCommon.all.input_shapes if len(shape) == 4 and shape[0] == 1
            ],
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_test_type(
                test_vector, TestParamsData.TestType.INDEX_VALUES_SINGLE, dim=-2
            ),
        ),
        # Test fill_cache - shape[0] == 1
        TestCollection(
            operators=TestCollectionData.operator_fill_cache,
            input_sources=TestCollectionCommon.all.input_sources,
            input_shapes=[
                shape for shape in TestCollectionCommon.all.input_shapes if len(shape) == 4 and shape[0] == 1
            ],
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_test_type(
                test_vector, TestParamsData.TestType.INDEX_VALUES_IN_ORDER, dim=-2
            ),
        ),
        # Test fill_cache - shape[0] != 1
        TestCollection(
            operators=TestCollectionData.operator_fill_cache,
            input_sources=TestCollectionCommon.all.input_sources,
            input_shapes=[
                shape for shape in TestCollectionCommon.all.input_shapes if len(shape) == 4 and shape[0] != 1
            ],
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_test_type(
                test_vector, TestParamsData.TestType.INDEX_VALUES_IN_ORDER, dim=-2
            ),
        ),
        # TESTS ALL CASES - not yet supported so expected to fail
        # Test only supported test cases
        TestCollection(
            operators=TestCollectionData.operator_index_copy,
            input_sources=TestCollectionCommon.all.input_sources,
            input_shapes=TestCollectionCommon.quick.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_test_type(
                test_vector, TestParamsData.TestType.INDEX_VALUES_SINGLE
            ),
        ),
        TestCollection(
            operators=TestCollectionData.operator_index_copy,
            input_sources=TestCollectionCommon.all.input_sources,
            input_shapes=TestCollectionCommon.all.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_test_type(
                test_vector, TestParamsData.TestType.INDEX_VALUES_IN_ORDER
            ),
        ),
        # Test operators with reverse order index tensor
        TestCollection(
            operators=TestCollectionData.operator_index_copy,
            input_sources=TestCollectionCommon.all.input_sources,
            input_shapes=TestCollectionCommon.quick.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_test_type(
                test_vector, TestParamsData.TestType.INDEX_VALUES_REVERSE_ORDER
            ),
        ),
        # Test operators with shuffled values in index tensor
        TestCollection(
            operators=TestCollectionData.operator_index_copy,
            input_sources=TestCollectionCommon.all.input_sources,
            input_shapes=TestCollectionCommon.quick.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_test_type(
                test_vector, TestParamsData.TestType.INDEX_VALUES_OUT_OF_ORDER
            ),
        ),
        # Test operators with full coverage values in index tensor
        TestCollection(
            operators=TestCollectionData.operator_index_copy,
            input_sources=TestCollectionCommon.all.input_sources,
            input_shapes=[
                (6, 10, 1000, 100),
            ],
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_test_type(
                test_vector, TestParamsData.TestType.INDEX_VALUES_FULL_COVERAGE
            ),
        ),
        # Test Data formats collection:
        TestCollection(
            operators=TestCollectionData.operator_index_copy,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_test_type(
                test_vector, TestParamsData.TestType.INDEX_VALUES_IN_ORDER
            ),
            dev_data_formats=[
                item
                for item in TestCollectionTorch.all.dev_data_formats
                if item not in TestCollectionTorch.single.dev_data_formats
            ],
            math_fidelities=TestCollectionCommon.single.math_fidelities,
        ),
        # Test math fidelity collection:
        TestCollection(
            operators=TestCollectionData.operator_index_copy,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_test_type(
                test_vector, TestParamsData.TestType.INDEX_VALUES_IN_ORDER
            ),
            dev_data_formats=TestCollectionTorch.single.dev_data_formats,
            math_fidelities=TestCollectionCommon.all.math_fidelities,
        ),
    ],
    failing_rules=[
        *TestIdsDataLoader.build_failing_rules(operators=TestCollectionData.operator_index_copy),
    ],
)


def get_test_plans() -> List[TestPlan]:
    return [TestParamsData.test_plan]