# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math
import torch
import random

from typing import List, Dict
from loguru import logger

from forge.verify.config import VerifyConfig

from forge.verify.value_checkers import AllCloseValueChecker, AutomaticValueChecker

from test.operators.utils import VerifyUtils
from test.operators.utils import InputSource
from test.operators.utils import TestVector
from test.operators.utils import TestPlan
from test.operators.utils.compat import TestDevice, TestTensorsUtils
from test.operators.utils import TestCollection
from test.operators.utils import TestCollectionCommon
from test.operators.utils import ValueRanges
from test.operators.utils.utils import PytorchUtils
from test.operators.utils.test_data import TestCollectionTorch

from test.operators.pytorch.ids.loader import TestIdsDataLoader
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
                dtype=TestTensorsUtils.get_dtype_for_df(test_vector.dev_data_format),
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

    operators = ["unsqueeze"]

    @classmethod
    def generate_kwargs(cls, test_vector: TestVector):

        rng = random.Random(math.prod(test_vector.input_shape))
        dim = len(test_vector.input_shape)

        yield {"dim": rng.randint(-dim - 1, dim)}


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
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
        ),
        # Test Data formats collection:
        TestCollection(
            operators=TestParamsData.operators,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
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
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
            dev_data_formats=TestCollectionTorch.single.dev_data_formats,
            math_fidelities=TestCollectionCommon.all.math_fidelities,
        ),
    ],
    failing_rules=[
        *TestIdsDataLoader.build_failing_rules(operators=TestParamsData.operators),
    ],
)


def get_test_plans() -> List[TestPlan]:
    return [TestParamsData.test_plan]
