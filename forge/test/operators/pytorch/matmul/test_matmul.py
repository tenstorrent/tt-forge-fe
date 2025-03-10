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

from test.operators.pytorch.eltwise_binary import ModelFromAnotherOp, ModelDirect, ModelConstEvalPass


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
        number_of_operands: int = 2

        module = PytorchUtils.get_pytorch_module(test_vector.operator)
        operator = getattr(module, test_vector.operator)
        kwargs = test_vector.kwargs if test_vector.kwargs else {}
        model_type = cls.MODEL_TYPES[test_vector.input_source]

        pytorch_model = model_type(operator, "matmul", test_vector.input_shape, kwargs)

        # input_shapes = tuple([test_vector.input_shape])
        input_shapes = tuple([test_vector.input_shape for _ in range(number_of_operands)])
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

    # failed_allclose_value_checker = TestPlanUtils.load_test_ids_from_file(
    #     f"{os.path.dirname(__file__)}/test_matmul_ids_failed_allclose_value_checker.txt"
    # )


class TestParamsData:

    __test__ = False

    test_plan: TestPlan = None

    operators = ["matmul"]

    # @classmethod
    # def generate_kwargs(cls, test_vector: TestVector):
    #     return


TestParamsData.test_plan = TestPlan(
    verify=lambda test_device, test_vector: TestVerification.verify(
        test_device,
        test_vector,
    ),
    collections=[
        # Test all shapes and input sources collection:
        TestCollection(
            operators=TestParamsData.operators,
            input_sources=[  # TODO: use TestCollectionCommon.all.input_sources when becames available
                InputSource.FROM_ANOTHER_OP,
                InputSource.FROM_HOST,
                InputSource.CONST_EVAL_PASS,
            ],
            input_shapes=TestCollectionCommon.all.input_shapes,
        ),
        # Test Data formats collection:
        TestCollection(
            operators=TestParamsData.operators,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
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
            # dev_data_formats=TestCollectionTorch.single.dev_data_formats,  # Can't use it because it's unsupported data format
            math_fidelities=TestCollectionCommon.all.math_fidelities,
        ),
    ],
    failing_rules=[],
)


def get_test_plans() -> List[TestPlan]:
    return [TestParamsData.test_plan]