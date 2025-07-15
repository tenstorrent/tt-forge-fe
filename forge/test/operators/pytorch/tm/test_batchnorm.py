# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import forge
import torch
import pytest

from typing import List, Dict
from loguru import logger

from forge._C import DataFormat

from forge.config import CompilerConfig
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AllCloseValueChecker, AutomaticValueChecker
from forge.verify.verify import verify as forge_verify

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
        self.testname = "BatchNorm2d_operator_test_op_src_from_another_op"
        self.operator = operator
        self.batchnorm2d = self.operator(**kwargs)

    def forward(self, x):
        xx = torch.add(x, x)
        return self.batchnorm2d(xx)


class ModelDirect(torch.nn.Module):
    def __init__(self, operator, kwargs):
        super().__init__()
        self.testname = "BatchNorm2d_operator_test_op_src_direct"
        self.operator = operator
        self.batchnorm2d = self.operator(**kwargs)

    def forward(self, x):
        return self.batchnorm2d(x)


class ModelConstEvalPass(torch.nn.Module):
    def __init__(self, operator, input_shape, kwargs, value_range):
        super().__init__()
        self.testname = "BatchNorm2d_operator_test_op_src_const_eval_pass"
        self.operator = operator
        self.batchnorm2d = self.operator(**kwargs)
        self.dtype = kwargs.get("dtype")
        self.const = TensorUtils.create_torch_constant(
            input_shape=input_shape,
            dev_data_format=self.dtype,
            value_range=value_range,
        )
        self.register_buffer("constant", self.const)

    def forward(self, x):
        cc = self.batchnorm2d(self.const)
        xx = self.batchnorm2d(x)
        return torch.add(cc, xx)


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

        value_range = ValueRanges.SMALL
        kwargs = test_vector.kwargs if test_vector.kwargs else {}

        model_type = cls.MODEL_TYPES[test_vector.input_source]
        if test_vector.input_source == InputSource.CONST_EVAL_PASS:
            pytorch_model = model_type(
                operator=operator,
                input_shape=test_vector.input_shape,
                kwargs=kwargs,
                value_range=value_range,
            )
        else:
            pytorch_model = model_type(
                operator=operator,
                kwargs=kwargs,
            )

        dtype = kwargs.get("dtype")
        compiler_cfg = CompilerConfig()
        if dtype is torch.bfloat16:
            pytorch_model.to(dtype)
            compiler_cfg.default_df_override = DataFormat.Float16_b

        input_shapes = tuple([test_vector.input_shape])
        logger.trace(f"***input_shapes: {input_shapes}")

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
            warm_reset=warm_reset,
            value_range=value_range,
            verify_config=verify_config,
        )


class TestParamsData:

    __test__ = False

    test_plan: TestPlan = None

    @classmethod
    def generate_kwargs(cls, test_vector: TestVector):
        kwargs = {
            "num_features": test_vector.input_shape[1],
            "dtype": test_vector.dev_data_format,
        }
        yield kwargs

    @classmethod
    def generate_kwargs_eps_momentum(cls, test_vector: TestVector):
        for eps in [1e-1, 1e-3, 1e-5, 1e-12]:
            for momentum in [0.0, 0.1, 0.9, 1.0, None]:
                yield {
                    "num_features": test_vector.input_shape[1],
                    "eps": eps,
                    "momentum": momentum,
                }


class TestCollectionData:

    __test__ = False

    operators = ["batch_norm_2d"]

    input_shapes = [shape for shape in TestCollectionCommon.all.input_shapes if len(shape) == 4]


TestParamsData.test_plan = TestPlan(
    verify=lambda test_device, test_vector: TestVerification.verify(
        test_device,
        test_vector,
    ),
    collections=[
        # Test operators with all shapes and input sources collection
        TestCollection(
            operators=TestCollectionData.operators,
            input_sources=TestCollectionCommon.all.input_sources,
            input_shapes=TestCollectionData.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
        ),
        # Test Data formats collection
        TestCollection(
            operators=TestCollectionData.operators,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            dev_data_formats=[
                item
                for item in TestCollectionTorch.float.dev_data_formats
                if item not in TestCollectionTorch.single.dev_data_formats
            ],
            math_fidelities=TestCollectionCommon.single.math_fidelities,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
        ),
        # Test Math fidelities collection
        TestCollection(
            operators=TestCollectionData.operators,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            dev_data_formats=TestCollectionTorch.single.dev_data_formats,
            math_fidelities=TestCollectionCommon.all.math_fidelities,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
        ),
        # Test eps and momentum collection
        TestCollection(
            operators=TestCollectionData.operators,
            input_sources=TestCollectionCommon.all.input_sources,
            input_shapes=[(10, 32, 64, 64)],
            kwargs=lambda test_vector: TestParamsData.generate_kwargs_eps_momentum(test_vector),
        ),
        # Test affine = False collection
        TestCollection(
            operators=TestCollectionData.operators,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            kwargs=lambda test_vector: (
                kwargs
                for kwargs in [
                    {
                        "num_features": test_vector.input_shape[1],
                        "affine": False,
                    }
                ]
            ),
        ),
        # Test track_running_stats = False collection
        TestCollection(
            operators=TestCollectionData.operators,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            kwargs=lambda test_vector: (
                kwargs
                for kwargs in [
                    {
                        "num_features": test_vector.input_shape[1],
                        "track_running_stats": False,
                    }
                ]
            ),
        ),
    ],
    failing_rules=[
        *TestIdsDataLoader.build_failing_rules(operators=TestCollectionData.operators),
    ],
)


def get_test_plans() -> List[TestPlan]:
    return [
        TestParamsData.test_plan,
    ]
