# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AllCloseValueChecker, AutomaticValueChecker
import pytest

from typing import List, Dict, Type, Optional, Any
from loguru import logger

import random
from test.operators.utils.test_data import TestCollectionTorch
import torch
import forge
import forge.op
import os

from forge.op_repo import TensorShape

from test.operators.utils import InputSourceFlags, VerifyUtils
from test.operators.utils import ShapeUtils
from test.operators.utils import InputSource
from test.operators.utils import TestVector
from test.operators.utils import TestPlan
from test.operators.utils import FailingReasons
from test.operators.utils.compat import TestDevice
from test.operators.utils import TestCollection
from test.operators.utils import TestPlanUtils
from test.operators.utils import TestCollectionCommon
from test.operators.utils import ValueRanges
from test.operators.utils.utils import PytorchUtils, TensorUtils
from test.operators.pytorch.ids.loader import TestIdsDataLoader


class ModelFromAnotherOp(torch.nn.Module):

    model_name = "model_op_src_from_another_op"

    def __init__(self, operator, shape, kwargs):
        super(ModelFromAnotherOp, self).__init__()
        self.testname = "Reduce_pytorch_operator_" + operator + "_test_op_src_from_another_op"
        self.operator = operator
        self.shape = shape
        self.kwargs = kwargs

    def forward(self, x: torch.Tensor):
        # we use Add operator to create one operands which is input for the reduce operator
        add1 = torch.add(x, x)
        output = self.operator(add1, **self.kwargs)
        return output


class ModelDirect(torch.nn.Module):

    model_name = "model_op_src_from_host"

    def __init__(self, operator, shape, kwargs):
        super(ModelDirect, self).__init__()
        self.testname = "Reduce_pytorch_operator_" + operator + "_test_op_src_from_host"
        self.operator = operator
        self.shape = shape
        self.kwargs = kwargs

    def forward(self, x: torch.Tensor):
        output = self.operator(x, **self.kwargs)
        return output


class ModelConstEvalPass(torch.nn.Module):

    model_name = "model_op_src_const_eval_pass"

    def __init__(self, operator, shape, kwargs, dtype, value_range):
        super(ModelConstEvalPass, self).__init__()
        self.testname = "Reduce_pytorch_operator_" + operator + "_test_op_src_const_eval_pass"
        self.operator = operator
        self.shape = shape
        # self.constant_shape = ShapeUtils.reduce_microbatch_size(shape)
        self.kwargs = kwargs

        # self.c1 = torch.rand(*self.constant_shape) - 0.5
        self.c1 = TensorUtils.create_torch_constant(
            input_shape=shape,
            dev_data_format=dtype,
            value_range=value_range,
        )

    def forward(self, x):
        v1 = self.operator(self.c1, **self.kwargs)
        v2 = self.operator(x, **self.kwargs)
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
    def verify(
        cls,
        test_device: TestDevice,
        test_vector: TestVector,
        input_params: List[Dict] = [],
        warm_reset: bool = False,
    ):
        """Common verification function for all tests"""

        operator = PytorchUtils.get_op_class_by_name(test_vector.operator)
        kwargs = test_vector.kwargs if test_vector.kwargs else {}
        # Old behavior when dev_data_format was not set
        value_range = (None if test_vector.dev_data_format is not None else ValueRanges.SMALL_POSITIVE,)

        model_type = cls.MODEL_TYPES[test_vector.input_source]
        pytorch_model = (
            model_type(
                operator=operator,
                shape=test_vector.input_shape,
                kwargs=kwargs,
                dtype=test_vector.dev_data_format,
                value_range=value_range,
            )
            if test_vector.input_source in (InputSource.CONST_EVAL_PASS,)
            else model_type(operator, kwargs)
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

    __test__ = False  # Avoid collecting TestParamsData as a pytest test

    test_plan: TestPlan = None

    @classmethod
    def generate_kwargs(cls, test_vector: TestVector):
        shape_with_kwargs = cls.extend_shape_with_dims_and_keepdims(test_vector.input_shape)
        kwarg_list = []
        for item in shape_with_kwargs:
            kwargs = {}
            kwargs["dim"] = item[1]
            kwargs["keepdim"] = item[2]
            kwarg_list.append(kwargs)
        return kwarg_list

    @classmethod
    def extend_shape_with_dims_and_keepdims(cls, shape):
        shape_with_dims_and_keepdims = list()
        for dim in list(range(0, len(shape), 1)):
            shape_with_dims_and_keepdims.append((shape, dim, True))
            shape_with_dims_and_keepdims.append((shape, dim, False))
        return shape_with_dims_and_keepdims


class TestCollectionData:

    __test__ = False  # Avoid collecting TestCollectionData as a pytest test

    all = TestCollection(
        operators=[
            "sum",  # 00
            "mean",  # 01
        ],
        input_sources=TestCollectionCommon.all.input_sources,
        input_shapes=TestCollectionCommon.all.input_shapes,
        dev_data_formats=[
            item
            for item in TestCollectionTorch.all.dev_data_formats
            if item not in TestCollectionTorch.single.dev_data_formats
        ],
        math_fidelities=TestCollectionCommon.all.math_fidelities,
    )

    single = TestCollection(
        input_sources=TestCollectionCommon.single.input_sources,
        input_shapes=TestCollectionCommon.single.input_shapes,
        dev_data_formats=TestCollectionTorch.single.dev_data_formats,
        math_fidelities=TestCollectionCommon.single.math_fidelities,
    )

    sum = TestCollection(
        operators=[
            "sum",  # 00
        ],
    )

    mean = TestCollection(
        operators=[
            "mean",  # 01
        ],
        dev_data_formats=[
            item
            for item in TestCollectionTorch.float.dev_data_formats
            if item not in TestCollectionTorch.single.dev_data_formats
        ],
    )


class TestIdsData:

    __test__ = False  # Avoid collecting TestIdsData as a pytest test


TestParamsData.test_plan = TestPlan(
    verify=lambda test_device, test_vector: TestVerification.verify(
        test_device,
        test_vector,
    ),
    collections=[
        # Test plan:
        # 2. Operand source(s):
        # 3. Operand shapes type(s):
        # 4. Operand / output size of dimensions
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.all.input_sources,
            input_shapes=TestCollectionData.all.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
        ),
        # Test plan:
        # 5. Data format - sum operator
        TestCollection(
            operators=TestCollectionData.sum.operators,
            input_sources=TestCollectionData.single.input_sources,
            input_shapes=TestCollectionData.single.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
            dev_data_formats=TestCollectionData.all.dev_data_formats,
            math_fidelities=TestCollectionData.single.math_fidelities,
        ),
        # 5. Data format - mean operator
        TestCollection(
            operators=TestCollectionData.mean.operators,
            input_sources=TestCollectionData.single.input_sources,
            input_shapes=TestCollectionData.single.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
            dev_data_formats=TestCollectionData.mean.dev_data_formats,
            math_fidelities=TestCollectionData.single.math_fidelities,
        ),
        # Test plan:
        # 6. Math fidelity
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.single.input_sources,
            input_shapes=TestCollectionData.single.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
            dev_data_formats=TestCollectionData.single.dev_data_formats,
            math_fidelities=TestCollectionData.all.math_fidelities,
        ),
    ],
    failing_rules=[
        # Skip all tests with input shapes with 2 dimensions
        TestCollection(
            criteria=lambda test_vector: len(test_vector.input_shape) == 2,
            skip_reason=FailingReasons.NOT_IMPLEMENTED,
        ),
        *TestIdsDataLoader.build_failing_rules(operators=TestCollectionData.all.operators),
    ],
)


def get_test_plans() -> List[TestPlan]:
    return [
        TestParamsData.test_plan,
    ]
