# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math
import torch
import random
import os
import forge

from typing import List, Dict
from loguru import logger

from forge.verify.config import VerifyConfig

from forge.verify.value_checkers import AllCloseValueChecker, AutomaticValueChecker

from test.operators.utils import InputSourceFlags, VerifyUtils
from test.operators.utils import InputSource
from test.operators.utils import TestVector
from test.operators.utils import TestPlan
from test.operators.utils import TestPlanUtils
from test.operators.utils import FailingReasons
from test.operators.utils.compat import TestDevice
from test.operators.utils import TestCollection
from test.operators.utils import TestCollectionCommon
from test.operators.utils import ValueRanges
from test.operators.utils.utils import PytorchUtils

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
        verify_config = VerifyConfig(value_checker=AllCloseValueChecker(atol=1e-2))
        if test_vector.dev_data_format in TestCollectionCommon.int.dev_data_formats:
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


class TestParamsData:

    __test__ = False

    test_plan: TestPlan = None

    operator = ["repeat_interleave"]

    specific_cases = {
        # input_shape: [(repeats, dim)...]
        (1, 1, 1, 58): [(1, 0), (1, 1), (58, 2)],  # used in llama model
    }

    @classmethod
    def generate_kwargs(cls, test_vector: TestVector):

        rng = random.Random(math.prod(test_vector.input_shape))

        yield {
            # repeats is only int values, as tensor is not supported yet
            "repeats": rng.randint(1, 10),
            "dim": rng.choice([None] + list(range(len(test_vector.input_shape)))),
        }

    @classmethod
    def generate_specific_kwargs(cls, test_vector: TestVector):

        for repeats, dim in cls.specific_cases[test_vector.input_shape]:
            yield {
                "repeats": repeats,
                "dim": dim,
            }


TestParamsData.test_plan = TestPlan(
    verify=lambda test_device, test_vector: TestVerification.verify(
        test_device,
        test_vector,
    ),
    collections=[
        # Test operators with all shapes and input sources collection:
        TestCollection(
            operators=TestParamsData.operator,
            input_sources=TestCollectionCommon.all.input_sources,
            input_shapes=TestCollectionCommon.all.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
        ),
        # Test Data formats collection:
        TestCollection(
            operators=TestParamsData.operator,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
            dev_data_formats=[
                item
                for item in TestCollectionCommon.all.dev_data_formats
                if item not in TestCollectionCommon.single.dev_data_formats
            ],
        ),
        # Test math fidelity collection:
        TestCollection(
            operators=TestParamsData.operator,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
            dev_data_formats=TestCollectionCommon.single.dev_data_formats,
            math_fidelities=TestCollectionCommon.all.math_fidelities,
        ),
        # Test specific cases collection:
        TestCollection(
            operators=TestParamsData.operator,
            input_sources=TestCollectionCommon.all.input_sources,
            input_shapes=TestParamsData.specific_cases.keys(),
            kwargs=lambda test_vector: TestParamsData.generate_specific_kwargs(test_vector),
        ),
    ],
    failing_rules=[
        # Failed automatic value checker:
        TestCollection(
            input_sources=[InputSource.FROM_HOST],
            kwargs=[
                {"repeats": 7, "dim": 3},
            ],
            dev_data_formats=[
                forge.DataFormat.Int8,
                forge.DataFormat.Int32,
            ],
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        # Unsupported special cases when dim = 0:
        TestCollection(
            criteria=lambda test_vector: test_vector.kwargs is not None
            and "dim" in test_vector.kwargs
            and test_vector.kwargs["dim"] is None,
            failing_reason=FailingReasons.UNSUPPORTED_SPECIAL_CASE,
        ),
        # To large repeats inference failed:
        TestCollection(
            kwargs=[
                {"repeats": 58, "dim": 2},
            ],
            failing_reason=FailingReasons.INFERENCE_FAILED,
        ),
    ],
)


def get_test_plans() -> List[TestPlan]:
    return [TestParamsData.test_plan]
