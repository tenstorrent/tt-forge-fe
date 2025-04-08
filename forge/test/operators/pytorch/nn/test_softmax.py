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
from test.operators.utils import TestVector
from test.operators.utils import TestPlan
from test.operators.utils import TestPlanUtils
from test.operators.utils import TestCollection
from test.operators.utils import TestCollectionCommon
from test.operators.utils import TestCollectionTorch
from test.operators.utils import PytorchUtils
from test.operators.utils.utils import TestDevice

from test.operators.pytorch.eltwise_unary import ModelFromAnotherOp, ModelDirect, ModelConstEvalPass


class TestVerification:

    MODEL_TYPES = {
        InputSource.FROM_ANOTHER_OP: ModelFromAnotherOp,
        InputSource.FROM_HOST: ModelDirect,
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
        module = PytorchUtils.get_pytorch_module(test_vector.operator)
        operator = getattr(module, test_vector.operator)
        kwargs = test_vector.kwargs if test_vector.kwargs else {}
        model_type = cls.MODEL_TYPES[test_vector.input_source]

        pytorch_model = (
            model_type(operator, test_vector.input_shape, kwargs)
            if test_vector.input_source in (InputSource.CONST_EVAL_PASS,)
            else model_type(operator, kwargs)
        )

        input_shapes = tuple([test_vector.input_shape])

        logger.trace(f"***input_shapes: {input_shapes}")

        # We use AllCloseValueChecker in all cases except for integer data formats:
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
        f"{os.path.dirname(__file__)}/test_softmax_ids_failed_allclose_value_checker.txt"
    )


class TestParamsData:

    __test__ = False

    test_plan: TestPlan = None

    operators = ["softmax"]

    @classmethod
    def generate_kwargs_pos(cls, test_vector: TestVector):
        for dim in range(len(test_vector.input_shape)):
            yield {"dim": dim}

    @classmethod
    def generate_kwargs_neg(cls, test_vector: TestVector):
        for dim in range(-1, -len(test_vector.input_shape) - 1, -1):
            yield {"dim": dim}


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
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_pos(test_vector),
        ),
        # Test some shapes and input sources with negative dim values:
        TestCollection(
            operators=TestParamsData.operators,
            input_sources=TestCollectionCommon.all.input_sources,
            input_shapes=TestCollectionCommon.quick.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_neg(test_vector),
        ),
        # Test Data formats collection:
        TestCollection(
            operators=TestParamsData.operators,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            kwargs=[{"dim": -1}],
            # dev_data_formats=TestCollectionTorch.all.dev_data_formats,
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
            kwargs=[{"dim": -1}],
            # dev_data_formats=TestCollectionTorch.single.dev_data_formats,  # Can't use it because it's unsupported data format
            math_fidelities=TestCollectionCommon.all.math_fidelities,
        ),
    ],
    failing_rules=[
        # All dim values are not supported except for the last one:
        TestCollection(
            operators=TestParamsData.operators,
            criteria=lambda test_vector: test_vector.kwargs["dim"] != len(test_vector.input_shape) - 1
            and test_vector.kwargs["dim"] != -1,
            skip_reason=FailingReasons.UNSUPORTED_AXIS,
            failing_reason=FailingReasons.UNSUPORTED_AXIS,
            subcollections=[
                # One test case as check-flag to indicate that the dim value is not supported:
                TestCollection(
                    operators=TestParamsData.operators,
                    input_sources=TestCollectionCommon.single.input_sources,
                    input_shapes=TestCollectionCommon.single.input_shapes,
                    kwargs=[{"dim": 0}],
                    # skip_reason=None,  # No need to explicit 'clear' the skip_reason
                    failing_reason=FailingReasons.UNSUPORTED_AXIS,
                ),
            ],
        ),
        # All-close value checker failed (rtol=1e-2, atol=1e-2):
        TestCollection(
            criteria=lambda test_vector: test_vector.get_id() in TestIdsData.failed_allclose_value_checker,
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        # Softmax lastdim kernel not implemented for some data formats:
        TestCollection(
            operators=TestParamsData.operators,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            dev_data_formats=[
                torch.int8,
                torch.int32,
                torch.int64,
            ],
            failing_reason=FailingReasons.UNSUPPORTED_DATA_FORMAT,
        ),
    ],
)


def get_test_plans() -> List[TestPlan]:
    return [TestParamsData.test_plan]
