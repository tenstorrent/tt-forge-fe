# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Tests for testing of linear operators
#
# In this test we test pytorch linear operator


import random
import math

from typing import List, Dict
from loguru import logger

import torch
import forge
import forge.op

from forge._C import DataFormat

from forge.config import CompilerConfig
from forge.verify.config import VerifyConfig
from forge.config import CompilerConfig
from forge.verify.value_checkers import AllCloseValueChecker

from forge._C import DataFormat

from test.operators.utils import (
    VerifyUtils,
    ValueRanges,
    InputSource,
    TestVector,
    TensorUtils,
    FailingReasons,
    TestPlan,
    TestCollection,
    TestCollectionCommon,
    TestCollectionTorch,
)
from test.operators.utils.compat import TestDevice
from test.operators.utils.utils import PytorchUtils
from test.operators.pytorch.ids.loader import TestIdsDataLoader


class ModelFromAnotherOp(torch.nn.Module):

    model_name = "model_op_src_from_another_op"

    def __init__(self, operator, shape, kwargs):
        super(ModelFromAnotherOp, self).__init__()
        self.testname = "Linear_pytorch_operator_test_op_src_from_another_op"
        self.operator = operator
        self.shape = shape
        self.kwargs = {
            "in_features": kwargs["in_features"],
            "out_features": kwargs["out_features"],
            "bias": kwargs["bias"],
            "dtype": kwargs["dtype"],
        }

        self.l1 = self.operator(**self.kwargs)

    def forward(self, x: torch.Tensor):
        # we use Add operator to create one operands which is input for the linear operator
        add = torch.add(x, x)
        output = self.l1(add)
        return output


class ModelDirect(torch.nn.Module):

    model_name = "model_op_src_from_host"

    def __init__(self, operator, shape, kwargs):
        super(ModelDirect, self).__init__()
        self.testname = "Linear_pytorch_operator_test_op_src_from_host"
        self.operator = operator
        self.shape = shape
        self.kwargs = {
            "in_features": kwargs["in_features"],
            "out_features": kwargs["out_features"],
            "bias": kwargs["bias"],
            "dtype": kwargs["dtype"],
        }

        self.l1 = self.operator(**self.kwargs)

    def forward(self, x: torch.Tensor):
        output = self.l1(x)
        return output


class ModelConstEvalPass(torch.nn.Module):

    model_name = "model_op_src_const_eval_pass"

    def __init__(self, operator, shape, kwargs, value_range):
        super(ModelConstEvalPass, self).__init__()
        self.testname = "Linear_pytorch_operator_test_op_src_const_eval_pass"
        self.operator = operator
        self.shape = shape
        self.kwargs = {
            "in_features": kwargs["in_features"],
            "out_features": kwargs["out_features"],
            "bias": kwargs["bias"],
            "dtype": kwargs["dtype"],
        }

        self.const = TensorUtils.create_torch_constant(
            input_shape=shape,
            dev_data_format=self.kwargs.get("dtype"),
            value_range=value_range,
            random_seed=math.prod(shape),
        )
        self.register_buffer("constant", self.const)

        self.l1 = self.operator(**self.kwargs)

    def forward(self, x: torch.Tensor):
        v1 = self.l1(self.const)
        # v2 = torch.add(x, x)
        v2 = self.l1(x)
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
        number_of_operands: int = 1,
        warm_reset: bool = False,
    ):
        """Common verification function for all tests"""

        operator = PytorchUtils.get_op_class_by_name(test_vector.operator)

        value_range = ValueRanges.SMALL
        kwargs = test_vector.kwargs if test_vector.kwargs else {}

        model_type = cls.MODEL_TYPES[test_vector.input_source]
        if test_vector.input_source == InputSource.CONST_EVAL_PASS:
            pytorch_model = model_type(
                operator=operator,
                shape=test_vector.input_shape,
                kwargs=kwargs,
                value_range=value_range,
            )
        else:
            pytorch_model = model_type(
                operator=operator,
                shape=test_vector.input_shape,
                kwargs=kwargs,
            )

        dtype = kwargs.get("dtype")
        compiler_cfg = CompilerConfig()

        if dtype is torch.bfloat16:
            pytorch_model.to(dtype)
            compiler_cfg.default_df_override = DataFormat.Float16_b

        input_shapes = tuple([test_vector.input_shape for _ in range(number_of_operands)])
        logger.trace(f"***input_shapes: {input_shapes}")

        # We don't test int data type as there is no sense for linear operator
        # Using AllCloseValueChecker
        verify_config = VerifyConfig(value_checker=AllCloseValueChecker(rtol=1e-2, atol=1e-2))

        VerifyUtils.verify(
            model=pytorch_model,
            test_device=test_device,
            input_shapes=input_shapes,
            input_params=input_params,
            compiler_cfg=compiler_cfg,
            dev_data_format=test_vector.dev_data_format,
            math_fidelity=test_vector.math_fidelity,
            pcc=test_vector.pcc,
            warm_reset=warm_reset,
            deprecated_verification=False,
            verify_config=verify_config,
            value_range=value_range,
        )


class TestParamsData:

    __test__ = False  # Avoid collecting TestParamsData as a pytest test

    test_plan: TestPlan = None

    @classmethod
    def get_out_features(cls, input_shape: List[int]):
        treshold = 10000
        out_features = []
        rng = random.Random(sum(input_shape))
        for _ in range(2):
            out_features.append(rng.randint(1, 1000))
        out_features.append(sum(input_shape) % treshold)
        return out_features

    @classmethod
    def generate_kwargs(cls, test_vector: TestVector):
        kwarg_list = []
        in_features = test_vector.input_shape[-1]
        out_features_list = TestParamsData.get_out_features(test_vector.input_shape)
        bias_list = [True, False]
        for out_features in out_features_list:
            for bias in bias_list:
                kwarg_list.append(
                    {
                        "in_features": in_features,
                        "out_features": out_features,
                        "bias": bias,
                        "dtype": test_vector.dev_data_format,
                    }
                )
        return kwarg_list


class TestCollectionData:

    __test__ = False  # Avoid collecting TestCollectionData as a pytest test

    all = TestCollection(
        operators=[
            "linear",  # 00
        ],
        input_sources=TestCollectionCommon.all.input_sources,
        input_shapes=TestCollectionCommon.all.input_shapes,
        dev_data_formats=TestCollectionTorch.all.dev_data_formats,
        math_fidelities=TestCollectionCommon.all.math_fidelities,
    )

    single = TestCollection(
        input_sources=TestCollectionCommon.single.input_sources,
        input_shapes=TestCollectionCommon.single.input_shapes,
        dev_data_formats=TestCollectionTorch.single.dev_data_formats,
        math_fidelities=TestCollectionCommon.single.math_fidelities,
    )


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
        # 5. Data format
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.single.input_sources,
            input_shapes=TestCollectionData.single.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
            dev_data_formats=[
                item
                for item in TestCollectionTorch.float.dev_data_formats
                if item not in TestCollectionTorch.single.dev_data_formats
            ],
            math_fidelities=TestCollectionData.single.math_fidelities,
        ),
        # Test plan:
        # 6. Math fidelity
        TestCollection(
            operators=TestCollectionData.all.operators,
            input_sources=TestCollectionData.single.input_sources,
            input_shapes=TestCollectionData.single.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
            dev_data_formats=TestCollectionTorch.single.dev_data_formats,
            math_fidelities=TestCollectionData.all.math_fidelities,
        ),
    ],
    failing_rules=[
        *TestIdsDataLoader.build_failing_rules(operators=TestCollectionData.all.operators),
        TestCollection(
            dev_data_formats=[torch.float16],
            skip_reason=FailingReasons.FATAL_ERROR,
        ),
    ],
)


def get_test_plans() -> List[TestPlan]:
    return [
        TestParamsData.test_plan,
    ]
