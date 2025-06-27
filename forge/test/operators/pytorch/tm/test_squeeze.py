# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import forge
import torch
import pytest

from typing import List, Dict
from loguru import logger

from forge.verify.config import VerifyConfig

from forge.verify.value_checkers import AllCloseValueChecker, AutomaticValueChecker

from test.operators.utils import VerifyUtils
from test.operators.utils import InputSource
from test.operators.utils import TestVector
from test.operators.utils import TestPlan
from test.operators.utils.compat import TestDevice
from test.operators.utils import TestCollection
from test.operators.utils import TestCollectionCommon
from test.operators.utils import ValueRanges
from test.operators.utils.utils import PytorchUtils
from test.operators.pytorch.ids.loader import TestIdsDataLoader
from test.operators.utils.test_data import TestCollectionTorch

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

        operator = PytorchUtils.get_op_class_by_name(test_vector.operator)

        value_range = ValueRanges.LARGE
        kwargs = test_vector.kwargs if test_vector.kwargs else {}

        model_type = cls.MODEL_TYPES[test_vector.input_source]
        if test_vector.input_source == InputSource.CONST_EVAL_PASS:
            pytorch_model = model_type(
                operator=operator,
                shape=test_vector.input_shape,
                kwargs=kwargs,
                dtype=test_vector.dev_data_format,
                value_range=value_range,
            )
        else:
            pytorch_model = model_type(
                operator=operator,
                kwargs=kwargs,
            )

        input_shapes = tuple([test_vector.input_shape])

        logger.trace(f"***input_shapes: {input_shapes}")

        # We use AllCloseValueChecker in all cases except for integer data formats:
        verify_config = VerifyConfig(value_checker=AllCloseValueChecker(atol=1e-2, rtol=1e-8))
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
            value_range=value_range,
            deprecated_verification=False,
            verify_config=verify_config,
        )


class TestParamsData:

    __test__ = False

    test_plan: TestPlan = None

    operators = ["squeeze"]

    # fmt: off
    specific_squeezes = {
        # input_shape: (squeeze_dims)
        (2, 1, 2, 1, 2):      (None, 0, 1),
        (100, 1, 100, 1):     (None, 1, 2, 3),
        (84, 25, 100, 1, 41): (None, 1, 2, 3),
        (5, 5, 5, 5, 5):      (None, 0,),
    }
    # fmt: on

    @classmethod
    def generate_kwargs(cls, test_vector: TestVector):

        input_shape = test_vector.input_shape

        for item in cls.specific_squeezes[input_shape]:
            # in pytorch docs version 2.1 None value for dim should be supported but it's not working, but optional dim is supported
            if item is None:
                # optional dim
                yield None
            else:
                # given dim
                yield {"dim": item}


TestParamsData.test_plan = TestPlan(
    verify=lambda test_device, test_vector: TestVerification.verify(
        test_device,
        test_vector,
    ),
    collections=[
        # Test operators with all shapes and input sources collection:
        TestCollection(
            operators=TestParamsData.operators,
            input_sources=TestCollectionCommon.all.input_sources,
            input_shapes=TestCollectionCommon.all.input_shapes,
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
        # Test Math fidelities collection:
        TestCollection(
            operators=TestParamsData.operators,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            dev_data_formats=TestCollectionTorch.single.dev_data_formats,
            math_fidelities=TestCollectionCommon.all.math_fidelities,
        ),
        # Test specific squeezes collection:
        TestCollection(
            operators=TestParamsData.operators,
            input_sources=TestCollectionCommon.all.input_sources,
            input_shapes=TestParamsData.specific_squeezes.keys(),
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
        ),
    ],
    failing_rules=[
        *TestIdsDataLoader.build_failing_rules(operators=TestParamsData.operators),
        # TestCollection(
        #     operators=TestParamsData.operators,
        #     input_shapes=[(1, 1)],
        #     failing_reason=FailingReasons.INFERENCE_FAILED,
        # ),
        # TestCollection(
        #     operators=TestParamsData.operators,
        #     input_shapes=[
        #         (2, 1, 2, 1, 2),
        #         (100, 1, 100, 1),
        #         (84, 25, 100, 1, 41),
        #         (5, 5, 5, 5, 5),
        #     ],
        #     kwargs=[{"dim": None}],
        #     failing_reason=FailingReasons.INFERENCE_FAILED,
        # ),
        # # tvm.error.InternalError: ...
        # TestCollection(
        #     operators=TestParamsData.operators,
        #     # fmt: off
        #     criteria=lambda test_vector: test_vector.input_shape == (2, 1, 2, 1, 2)  and test_vector.kwargs == {"dim": 0} or
        #                                  test_vector.input_shape == (100, 1, 100, 1) and test_vector.kwargs == {"dim": 2} or
        #                                  test_vector.input_shape == (84, 25, 100, 1, 41) and test_vector.kwargs in [{"dim": 1}, {"dim": 2}] or
        #                                  test_vector.input_shape == (5, 5, 5, 5, 5)  and test_vector.kwargs == {"dim": 0},
        #     # fmt: on
        #     failing_reason=FailingReasons.COMPILATION_FAILED,
        # ),
    ],
)


def get_test_plans() -> List[TestPlan]:
    return [TestParamsData.test_plan]
