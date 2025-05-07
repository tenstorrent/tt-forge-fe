# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import torch

from typing import List, Dict
from loguru import logger

from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AllCloseValueChecker, AutomaticValueChecker

from test.operators.utils import VerifyUtils
from test.operators.utils import FailingReasons
from test.operators.utils import ValueRanges
from test.operators.utils import InputSource
from test.operators.utils import ShapeUtils
from test.operators.utils import TestVector
from test.operators.utils import TestPlan
from test.operators.utils import TestPlanUtils
from test.operators.utils import TestCollection
from test.operators.utils import TestCollectionCommon
from test.operators.utils import TestCollectionTorch
from test.operators.utils.utils import PytorchUtils, TestDevice


class ModelFromAnotherOp(torch.nn.Module):
    def __init__(self, operator):
        super().__init__()
        self.testname = "Matmul_operator_test_op_src_from_another_op"
        self.operator = operator

    def forward(self, x, y):
        xx = torch.add(x, x)
        yy = torch.add(y, y)
        return self.operator(xx, yy)


class ModelFromHost(torch.nn.Module):
    def __init__(self, operator):
        super().__init__()
        self.testname = "Matmul_operator_test_op_src_from_host"
        self.operator = operator

    def forward(self, x, y):
        return self.operator(x, y)


class ModelConstEvalPass(torch.nn.Module):
    def __init__(self, operator, shape_1, shape_2):
        super().__init__()
        self.testname = "Matmul_operator_test_op_src_const_eval_pass"
        self.operator = operator
        self.c1 = (torch.rand(shape_1, requires_grad=False) - 0.5).detach()
        self.c2 = (torch.rand(shape_2, requires_grad=False) - 0.5).detach()

    def forward(self, x, y):
        mm1 = self.operator(self.c1, self.c2)
        mm2 = self.operator(x, y)
        aa = torch.add(mm1, mm2)
        return aa


class TestVerification:

    MODEL_TYPES = {
        InputSource.FROM_ANOTHER_OP: ModelFromAnotherOp,
        InputSource.FROM_HOST: ModelFromHost,
        InputSource.CONST_EVAL_PASS: ModelConstEvalPass,
    }

    @classmethod
    def verify(
        cls,
        test_device: TestDevice,
        test_vector: TestVector,
        input_params: List[Dict] = [],
        warm_reset: bool = False,
    ):
        model_type = cls.MODEL_TYPES[test_vector.input_source]
        operator = PytorchUtils.get_op_class_by_name(test_vector.operator)

        if isinstance(test_vector.input_shape[0], int):
            # input_shape is tuple of ints e.g. (a, b)
            input_shapes = [test_vector.input_shape, ShapeUtils.switch_last_two(test_vector.input_shape)]
        else:
            # input_shape is tuple of tuples e.g. ((a, b), (c, d))
            input_shapes = [test_vector.input_shape[0], test_vector.input_shape[1]]

        if test_vector.input_source == InputSource.CONST_EVAL_PASS:
            pytorch_model = model_type(operator, *input_shapes)
        else:
            pytorch_model = model_type(operator)

        logger.trace(f"***input_shapes: {input_shapes}")

        # Using AllCloseValueChecker in all cases except for integer data formats:
        verify_config = VerifyConfig(value_checker=AllCloseValueChecker(rtol=1e-2, atol=1e-2))
        if test_vector.dev_data_format in TestCollectionTorch.int.dev_data_formats:
            verify_config = VerifyConfig(value_checker=AutomaticValueChecker())

        VerifyUtils.verify(
            model=pytorch_model,
            test_device=test_device,
            input_shapes=input_shapes,
            input_params=input_params,
            dev_data_format=test_vector.dev_data_format,
            math_fidelity=test_vector.math_fidelity,
            warm_reset=warm_reset,
            value_range=ValueRanges.SMALL,
            deprecated_verification=False,
            verify_config=verify_config,
        )


class TestIdsData:

    __test__ = False  # Avoid collecting TestIdsData as a pytest test

    failed_allclose_value_checker = TestPlanUtils.load_test_ids_from_file(
        f"{os.path.dirname(__file__)}/test_matmul_ids_failed_allclose_value_checker.txt"
    )


class TestParamsData:

    __test__ = False

    test_plan: TestPlan = None

    operators = ["matmul"]


TestParamsData.test_plan = TestPlan(
    verify=lambda test_device, test_vector: TestVerification.verify(
        test_device,
        test_vector,
    ),
    collections=[
        # Test all shapes and input sources collection:
        TestCollection(
            operators=TestParamsData.operators,
            input_sources=TestCollectionCommon.all.input_sources,
            input_shapes=TestCollectionCommon.all.input_shapes,
        ),
        # Test special cases collection:
        TestCollection(
            operators=TestParamsData.operators,
            input_sources=TestCollectionCommon.all.input_sources,
            input_shapes=[
                ((3,), (3,)),
                ((7, 3), (3, 7)),
                ((3,), (3, 7)),
                ((32, 64), (64,)),
                ((64,), (3, 1, 64, 32)),
            ],
        ),
        # Test Data formats collection:
        TestCollection(
            operators=TestParamsData.operators,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            dev_data_formats=[
                item
                for item in TestCollectionTorch.all.dev_data_formats
                if item not in TestCollectionTorch.single.dev_data_formats
            ],
            math_fidelities=TestCollectionCommon.single.math_fidelities,
        ),
        # Test math fidelity collection:
        TestCollection(
            operators=TestParamsData.operators,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            dev_data_formats=TestCollectionTorch.single.dev_data_formats,
            math_fidelities=TestCollectionCommon.all.math_fidelities,
        ),
    ],
    failing_rules=[
        # Memory issue (too large input shapes):
        TestCollection(
            operators=TestParamsData.operators,
            input_shapes=[(10, 10, 10000, 1)],
            skip_reason=FailingReasons.ALLOCATION_FAILED,
            failing_reason=FailingReasons.ALLOCATION_FAILED,
        ),
        # All-close value checker failed (rtol=1e-2, atol=1e-2):
        #    >> ValueError: Data mismatch -> AllCloseValueChecker (all_close)
        TestCollection(
            criteria=lambda test_vector: test_vector.get_id() in TestIdsData.failed_allclose_value_checker,
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        # Unsupported data format:
        TestCollection(
            operators=TestParamsData.operators,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            dev_data_formats=[
                torch.int8,
                torch.int32,
                torch.int64,
                torch.float16,
            ],
            failing_reason=FailingReasons.UNSUPPORTED_DATA_FORMAT,
        ),
        # Matmul op when two input tensors are vectors is not supported:
        #    >> tvm.error.InternalError
        TestCollection(
            operators=TestParamsData.operators,
            input_shapes=[((3,), (3,))],
            failing_reason=FailingReasons.COMPILATION_FAILED,
        ),
        # Matmul op when one of the arguments is 1-dimensional and the other one is N-dimensional:
        #    >> AssertionError: Data mismatch on output 0 between framework and Forge codegen
        TestCollection(
            operators=TestParamsData.operators,
            input_shapes=[((64,), (3, 1, 64, 32))],
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
    ],
)


def get_test_plans() -> List[TestPlan]:
    return [TestParamsData.test_plan]
