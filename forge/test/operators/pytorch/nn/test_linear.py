# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Tests for testing of embedding operators
#
# In this test we test pytorch embedding operator

# GENERAL OP SUPPORT TEST PLAN:
# 1. Operand type - any supported type
# 2. Operand source(s):
# (+)  2.1 From another op
#       - Operator -> input
# (+)  2.2 From DRAM queue - Removed from test plan
#       - Operator is first node in network
#       - Input_queue flag = false
# (+)  2.3 Const Inputs (const eval pass)
#       - Operator where all inputs are constants.
# (+)  2.4 From host
#       - Input tensor as input of network
#       - Operator is first node in network
#       - Input_queue flag = true
# 3 Operand shapes type(s):
# (+)  3.1 Full tensor (i.e. full expected shape)
#       - 3-4 by default P1 (high prioriy)
#       - 2, 5, ++ include P2 (lower prioriy)
# (+)  3.2 Tensor reduce on one or more dims to 1
#       - Vector
#       - Only one dim is not equal to 1
# (+)  3.3 Scalar P2
#       - Create tensor of dimension equal to 0 (tensor from scalar) or just to use scalar as simple value
# 4. Operand / output size of dimensions (few examples of each, 10 values total)
# (+)  4.1 Divisible by 32
# (+)  4.2 Prime numbers
# (+)  4.3 Very large (thousands, 10s of thousands)
#       - 100x100, 100x1000
#       - maybe nightly only
# (+)  4.4 Extreme ratios between height/width
#      4.5 ...probably many more interesting combinations here
# 5. Data format - all supported formats
# (/)  5.1 Output DF
# (/)  5.2 Intermediate DF
# (/)  5.3 Accumulation DF
# (+)  5.4 Operand DFs
#       - Fix HiFi4 for math fidelity value
# (+) 6. Math fidelity - LoFi, HiFi2a, Hifi2b, Hifi3, Hifi4
#       - Fix fp16b (default) for data format value
# (/) 7. Special attributes - if applicable.. like approx_mode for Exp, for example
# (/) 8. Special cases - if applicable
# 9. Variable number of operands - if applicable
# (/) Few representative values
# (/) Reuse inputs for selected operators


from functools import reduce
import random
import pytest

from typing import List, Dict, Type, Optional, Any
from loguru import logger

import torch
import forge
import forge.op

from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AllCloseValueChecker

from test.operators.utils import InputSourceFlags, VerifyUtils, ValueRanges
from test.operators.utils import InputSource
from test.operators.utils import TestVector
from test.operators.utils import TestPlan
from test.operators.utils import FailingReasons
from test.operators.utils.compat import TestDevice
from test.operators.utils.compat import TestTensorsUtils
from test.operators.utils import TestCollection
from test.operators.utils import TestCollectionCommon


class ModelFromAnotherOp(torch.nn.Module):

    model_name = "model_op_src_from_another_op"

    def __init__(self, operator, opname, shape, kwargs):
        super(ModelFromAnotherOp, self).__init__()
        self.testname = "Embedding_pytorch_operator_" + opname + "_test_op_src_from_another_op"
        self.operator = operator
        self.opname = opname
        self.shape = shape
        self.kwargs = {
            "in_features": kwargs["in_features"],
            "out_features": kwargs["out_features"],
        }

        self.l1 = self.operator(**self.kwargs)

    def forward(self, x: torch.Tensor):
        # we use Add operator to create one operands which is input for the embedding operator
        add = torch.add(x, x)
        output = self.l1(add)
        return output


class ModelDirect(torch.nn.Module):

    model_name = "model_op_src_from_host"

    def __init__(self, operator, opname, shape, kwargs):
        super(ModelDirect, self).__init__()
        self.testname = "Embedding_pytorch_operator_" + opname + "_test_op_src_from_host"
        self.operator = operator
        self.opname = opname
        self.shape = shape
        self.kwargs = {
            "in_features": kwargs["in_features"],
            "out_features": kwargs["out_features"],
        }

        self.l1 = self.operator(**self.kwargs)

    def forward(self, x: torch.Tensor):
        output = self.l1(x)
        return output


class ModelConstEvalPass(torch.nn.Module):

    model_name = "model_op_src_const_eval_pass"

    def __init__(self, operator, opname, shape, kwargs, dtype):
        super(ModelConstEvalPass, self).__init__()
        self.testname = "Embedding_pytorch_operator_" + opname + "_test_op_src_const_eval_pass"
        self.operator = operator
        self.opname = opname
        self.shape = shape
        self.kwargs = {
            "in_features": kwargs["in_features"],
            "out_features": kwargs["out_features"],
        }

        self.constant = torch.rand(self.shape, dtype=dtype)
        self.l1 = self.operator(**self.kwargs)

    def forward(self, x: torch.Tensor):
        v1 = self.l1(self.constant)
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

        operator = getattr(torch.nn, test_vector.operator)

        kwargs = test_vector.kwargs if test_vector.kwargs else {}

        model_type = cls.MODEL_TYPES[test_vector.input_source]
        if test_vector.input_source == InputSource.CONST_EVAL_PASS:
            pytorch_model = model_type(
                operator=operator,
                opname=test_vector.operator,
                shape=test_vector.input_shape,
                kwargs=kwargs,
                dtype=TestTensorsUtils.get_dtype_for_df(test_vector.dev_data_format),
            )
        else:
            pytorch_model = model_type(
                operator=operator,
                opname=test_vector.operator,
                shape=test_vector.input_shape,
                kwargs=kwargs,
            )

        input_shapes = tuple([test_vector.input_shape for _ in range(number_of_operands)])
        logger.trace(f"***input_shapes: {input_shapes}")

        VerifyUtils.verify(
            model=pytorch_model,
            test_device=test_device,
            input_shapes=input_shapes,
            input_params=input_params,
            dev_data_format=test_vector.dev_data_format,
            math_fidelity=test_vector.math_fidelity,
            pcc=test_vector.pcc,
            warm_reset=warm_reset,
            deprecated_verification=False,
            verify_config=VerifyConfig(value_checker=AllCloseValueChecker(rtol=1e-2, atol=1e-2)),
            value_range=ValueRanges.SMALL,
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
                    }
                )
        return kwarg_list


class TestCollectionData:

    __test__ = False  # Avoid collecting TestCollectionData as a pytest test

    all = TestCollection(
        operators=[
            "Linear",  # 00
        ],
        input_sources=TestCollectionCommon.all.input_sources,
        input_shapes=TestCollectionCommon.all.input_shapes,
        dev_data_formats=TestCollectionCommon.all.dev_data_formats,
        math_fidelities=TestCollectionCommon.all.math_fidelities,
    )

    single = TestCollection(
        input_sources=TestCollectionCommon.single.input_sources,
        input_shapes=TestCollectionCommon.single.input_shapes,
        dev_data_formats=TestCollectionCommon.single.dev_data_formats,
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
            dev_data_formats=TestCollectionCommon.float.dev_data_formats,
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
        # E   RuntimeError: The expanded size of the tensor (x) must match the existing size (y) at non-singleton dimension 0.  Target sizes: [x].  Tensor sizes: [y]
        TestCollection(
            input_sources=TestCollectionData.all.input_sources,
            criteria=lambda test_vector: len(test_vector.input_shape) == 4 and test_vector.input_shape[0] > 1,
            failing_reason=FailingReasons.MICROBATCHING_UNSUPPORTED,
        ),
        # E   ValueError: Data mismatch -> AllCloseValueChecker (all_close):
        TestCollection(
            input_shapes=[
                (1, 10000),
            ],
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        # # THIS ERROR OCCURES WHEN USING DEPRICATED VERIFICATION METHOD (NOT ALLCLOSE VALUE CHECKER)
        # # E   AssertionError: PCC check failed
        # # this also happens for other 2 dim ipnut shapes where microbatch size is 1 and out_features is 1 - not all cases are failing
        # TestCollection(
        #     input_shapes=[
        #         (1, 10000),
        #     ],
        #     kwargs=[
        #         {
        #             "out_features": 1,
        #         },
        #     ],
        #     failing_reason=FailingReasons.DATA_MISMATCH,
        # ),
    ],
)


def get_test_plans() -> List[TestPlan]:
    return [
        TestParamsData.test_plan,
    ]
