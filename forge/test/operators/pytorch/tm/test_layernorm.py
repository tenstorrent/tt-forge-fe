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
        self.testname = "Layernorm_operator_test_op_src_from_another_op"
        self.operator = operator
        self.lnorm = self.operator(**kwargs)

    def forward(self, x):
        xx = torch.add(x, x)
        return self.lnorm(xx)


class ModelDirect(torch.nn.Module):
    def __init__(self, operator, kwargs):
        super().__init__()
        self.testname = "Layernorm_operator_test_op_src_direct"
        self.operator = operator
        self.lnorm = self.operator(**kwargs)

    def forward(self, x):
        return self.lnorm(x)


class ModelConstEvalPass(torch.nn.Module):
    def __init__(self, operator, input_shape, kwargs, dtype, value_range):
        super().__init__()
        self.testname = "Layernorm_operator_test_op_src_const_eval_pass"
        self.operator = operator
        self.lnorm = self.operator(**kwargs)
        self.const = TensorUtils.create_torch_constant(
            input_shape=input_shape,
            dev_data_format=dtype,
            value_range=value_range,
        )
        self.register_buffer("constant", self.const)

    def forward(self, x):
        cc = self.lnorm(self.const)
        xx = self.lnorm(x)
        return torch.add(xx, cc)


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
                dtype=test_vector.dev_data_format,
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

        # We don't test int data type as there is no sense for layernorm operator
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
            deprecated_verification=False,
            verify_config=verify_config,
        )


class TestParamsData:

    __test__ = False

    test_plan: TestPlan = None

    operator = ["layer_norm"]

    @classmethod
    def generate_kwargs(cls, test_vector: TestVector):
        kwargs = (
            {
                # only normalization over the last dimension is currently supported
                "normalized_shape": (test_vector.input_shape[-1],),
                # "eps": 1e-05,  # default
                # "elementwise_affine": True,  # default
                # "bias": True,  # default
                "dtype": test_vector.dev_data_format,
            }
            if test_vector.dev_data_format is not None
            else {
                # only normalization over the last dimension is currently supported
                "normalized_shape": (test_vector.input_shape[-1],),
                # "eps": 1e-05,  # default
                # "elementwise_affine": True,  # default
                # "bias": True,  # default
            }
        )
        yield kwargs


TestParamsData.test_plan = TestPlan(
    verify=lambda test_device, test_vector: TestVerification.verify(
        test_device,
        test_vector,
    ),
    collections=[
        # Test operators with all shapes and input sources collection
        TestCollection(
            operators=TestParamsData.operator,
            input_sources=TestCollectionCommon.all.input_sources,
            input_shapes=TestCollectionCommon.all.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
        ),
        # Test Data formats collection
        TestCollection(
            operators=TestParamsData.operator,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            dev_data_formats=[
                item
                for item in TestCollectionTorch.float.dev_data_formats  # can't use int data formats
                if item not in TestCollectionTorch.single.dev_data_formats
            ],
            math_fidelities=TestCollectionCommon.single.math_fidelities,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
        ),
        # Test Math fidelities collection
        TestCollection(
            operators=TestParamsData.operator,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            dev_data_formats=TestCollectionTorch.single.dev_data_formats,
            math_fidelities=TestCollectionCommon.all.math_fidelities,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
        ),
        # Test elementwise_affine=False collection
        TestCollection(
            operators=TestParamsData.operator,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            kwargs=lambda test_vector: (
                kwargs
                for kwargs in [
                    {
                        "normalized_shape": (test_vector.input_shape[-1],),
                        "elementwise_affine": False,
                    }
                ]
            ),
        ),
        # Test bias=False collection
        TestCollection(
            operators=TestParamsData.operator,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            kwargs=lambda test_vector: (
                kwargs
                for kwargs in [
                    {
                        "normalized_shape": (test_vector.input_shape[-1],),
                        "bias": False,
                    }
                ]
            ),
        ),
        # Test normalize over the last two dimensions collection
        TestCollection(
            operators=TestParamsData.operator,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            kwargs=lambda test_vector: (
                kwargs
                for kwargs in [
                    {
                        "normalized_shape": (test_vector.input_shape[-2:]),
                    }
                ]
            ),
        ),
    ],
    failing_rules=[
        *TestIdsDataLoader.build_failing_rules(operators=TestParamsData.operator),
    ],
)


def get_test_plans() -> List[TestPlan]:
    return [TestParamsData.test_plan]
