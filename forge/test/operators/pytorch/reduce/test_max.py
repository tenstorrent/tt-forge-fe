# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import random
import math

from torch import nn

from typing import List, Dict
from loguru import logger

from forge.op_repo import TensorShape
from forge.verify.config import VerifyConfig

from forge.verify.value_checkers import AllCloseValueChecker

from test.operators.utils import FailingReasons
from test.operators.utils import InputSourceFlags, VerifyUtils
from test.operators.utils import InputSource
from test.operators.utils import TestVector
from test.operators.utils import TestPlan
from test.operators.utils.compat import TestDevice
from test.operators.utils import TestCollection
from test.operators.utils import TestCollectionCommon
from test.operators.utils import ValueRanges

from test.operators.pytorch.eltwise_unary import ModelFromAnotherOp, ModelDirect, ModelConstEvalPass


class ModelFromAnotherOpMax(nn.Module):
    def __init__(self, operator, kwargs):
        super().__init__()
        self.testname = "Element_wise_unary_operators_test_op_src_from_another_op"
        self.operator = operator
        self.kwargs = kwargs

    def forward(self, x):
        xx = torch.add(x, x)
        return self.operator(xx, **self.kwargs)[0]


class ModelDirectMax(nn.Module):
    def __init__(self, operator, kwargs):
        super().__init__()
        self.testname = "Element_wise_unary_operators_test_op_src_from_host"
        self.operator = operator
        self.kwargs = kwargs

    def forward(self, x):
        return self.operator(x, **self.kwargs)[0]


class ModelConstEvalPassMax(nn.Module):
    def __init__(self, operator, shape: TensorShape, kwargs):
        super().__init__()
        self.testname = "Element_wise_unary_operators_test_op_src_const_eval_pass"
        self.operator = operator
        self.kwargs = kwargs
        self.c = (torch.rand(shape, requires_grad=False) - 0.5).detach()

    def forward(self, x):
        cc = self.operator(self.c, **self.kwargs)[0]
        xx = self.operator(x, **self.kwargs)[0]
        return torch.add(xx, cc)


class TestVerification:

    MODEL_TYPES = {
        InputSource.FROM_ANOTHER_OP: ModelFromAnotherOp,
        InputSource.FROM_HOST: ModelDirect,
        InputSource.CONST_EVAL_PASS: ModelConstEvalPass,
    }

    MODEL_TYPES_MAX_SPECIFIC = {
        InputSource.FROM_ANOTHER_OP: ModelFromAnotherOpMax,
        InputSource.FROM_HOST: ModelDirectMax,
        InputSource.CONST_EVAL_PASS: ModelConstEvalPassMax,
    }

    @classmethod
    def verify(
        cls,
        test_device: TestDevice,
        test_vector: TestVector,
        input_params: List[Dict] = [],
        warm_reset: bool = False,
    ):

        operator = getattr(torch, test_vector.operator)
        kwargs = test_vector.kwargs if test_vector.kwargs else {}

        if not kwargs:
            # if kwargs is empty, max operator returns a tensor
            model_type = cls.MODEL_TYPES[test_vector.input_source]
        else:
            # if kwargs is not empty, max operator returns a tuple of tensors so we need a different model
            model_type = cls.MODEL_TYPES_MAX_SPECIFIC[test_vector.input_source]

        pytorch_model = (
            model_type(operator, test_vector.input_shape, kwargs)
            if test_vector.input_source in (InputSource.CONST_EVAL_PASS,)
            else model_type(operator, kwargs)
        )

        input_shapes = tuple([test_vector.input_shape])

        logger.trace(f"***input_shapes: {input_shapes}")

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
            verify_config=VerifyConfig(value_checker=AllCloseValueChecker()),
        )


class TestParamsData:

    __test__ = False

    test_plan: TestPlan = None

    operator = ["max"]

    @classmethod
    def generate_kwargs(cls, test_vector: TestVector):

        rng = random.Random(math.prod(test_vector.input_shape))

        dim = rng.choice(range(0, len(test_vector.input_shape)))

        for ch in [True, False]:
            yield {"dim": dim, "keepdim": ch}


TestParamsData.test_plan = TestPlan(
    verify=lambda test_device, test_vector: TestVerification.verify(
        test_device,
        test_vector,
    ),
    collections=[
        # torch.max(input)
        TestCollection(
            operators=TestParamsData.operator,
            input_sources=TestCollectionCommon.all.input_sources,
            input_shapes=TestCollectionCommon.all.input_shapes,
        ),
        # torch.max(input, dim=..., keepdim=...)
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
            kwargs=[{"dim": 1, "keepdim": False}],
            dev_data_formats=[
                item
                for item in TestCollectionCommon.all.dev_data_formats
                if item not in TestCollectionCommon.single.dev_data_formats
            ],
            math_fidelities=TestCollectionCommon.single.math_fidelities,
        ),
        # Test Math fidelities collection:
        TestCollection(
            operators=TestParamsData.operator,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            kwargs=[{"dim": 1, "keepdim": False}],
            dev_data_formats=TestCollectionCommon.single.dev_data_formats,
            math_fidelities=TestCollectionCommon.all.math_fidelities,
        ),
    ],
    failing_rules=[
        TestCollection(
            operators=TestParamsData.operator,
            input_sources=TestCollectionCommon.all.input_sources,
            input_shapes=TestCollectionCommon.all.input_shapes,
            criteria=lambda test_vector: test_vector.kwargs is None,
            failing_reason=FailingReasons.COMPILATION_FAILED,
            skip_reason="This test is expected to fail because the max operator is not supported for 'torch.max(input)' way of usage",
        )
    ],
)


def get_test_plans() -> List[TestPlan]:
    return [TestParamsData.test_plan]
