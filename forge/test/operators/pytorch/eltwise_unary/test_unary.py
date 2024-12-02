# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


# GENERAL OP SUPPORT TEST PLAN:
# 1. Operand type - any supported type (e.g. add, matmul, conv2d, etc.)
# 2. Operand source(s):
#    (+)  2.1 From another op
#           - Operator -> input
#    (+)  2.2 From DRAM queue
#           - Operator is first node in network
#           - Input_queue flag = false
#    (+)  2.3 Const Inputs (const eval pass)
#           - Operator where all inputs are constants.
#    (+)  2.4 From host
#           - Input tensor as input of network
#           - Operator is first node in network
#           - Input_queue flag = true
# 3. Tensor ranks:
#    (+)  3.1 Full tensor (i.e. full expected shape)
#           - 3-4 by default P1 (high prioriy)
#           - 2, 5, ++ include P2 (lower prioriy)
#    (+)  3.2 Tensor reduce on one or more dims to 1
#           - Vector
#           - Only one dim is not equal to 1
#    (-)  3.3 Scalar P2
#           - Create tensor of dimension equal to 0 (tensor from scalar) or just to use scalar as simple value
# 4. Operand / output size of dimensions (few examples of each, 10 values total)
#    (+)  4.1 Divisible by 32
#    (+)  4.2 Prime numbers
#    (+)  4.3 Very large (thousands, 10s of thousands)
#           - 100x100, 100x1000
#           - maybe nightly only
#    (+)  4.4 Extreme ratios between height/width
#    (/)  4.5 ...probably many more interesting combinations here
# 5. Data format - all supported formats
#    (/)  5.1 Output DF
#    (/)  5.2 Intermediate DF
#    (/)  5.3 Accumulation DF
#    (+)  5.4 Operand DFs
#           - Fix HiFi4 for math fidelity value
#    (+) 6. Math fidelity - LoFi, HiFi2a, Hifi2b, Hifi3, Hifi4
#           - Fix fp16b (default) for data format value
#    (/) 7. Special attributes - if applicable.. like approx_mode for Exp, for example
#    (/) 8. Special cases - if applicable
# 9. Variable number of operands - if applicable
#    (/) Few representative values
#    (/) Reuse inputs for selected operators


import pytest
import torch
import torch.nn as nn
import forge
from forge.op_repo import TensorShape

from typing import List, Dict
from loguru import logger
from forge import MathFidelity, DataFormat

from test.operators.utils import InputSourceFlags, VerifyUtils
from test.operators.utils import InputSource
from test.operators.utils import TestVector
from test.operators.utils import TestPlan
from test.operators.utils import FailingReasons
from test.operators.utils.compat import TestDevice
from test.operators.utils import TestCollection
from test.operators.utils import TestCollectionCommon
from test.operators.utils import ValueRanges


class ModelFromAnotherOp(nn.Module):
    def __init__(self, operator):
        super().__init__()
        self.testname = "Element_wise_unary_operators_test_op_src_from_another_op"
        self.operator = operator

    def forward(self, x):
        xx = torch.add(x, x)
        return self.operator(xx)


class ModelDirect(nn.Module):
    def __init__(self, operator):
        super().__init__()
        self.testname = "Element_wise_unary_operators_test_op_src_from_host"
        self.operator = operator

    def forward(self, x):
        return self.operator(x)


class ModelConstEvalPass(nn.Module):
    def __init__(self, operator, shape: TensorShape):
        super().__init__()
        self.testname = "Element_wise_unary_operators_test_op_src_const_eval_pass"
        self.operator = operator
        self.c = (torch.rand(shape, requires_grad=False) - 0.5).detach()

    def forward(self, x):
        cc = self.operator(self.c)
        xx = self.operator(x)
        return torch.add(xx, cc)


class TestVerification:

    MODEL_TYPES = {
        InputSource.FROM_ANOTHER_OP: ModelFromAnotherOp,
        InputSource.FROM_HOST: ModelDirect,
        InputSource.FROM_DRAM_QUEUE: ModelDirect,
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

        input_source_flag: InputSourceFlags = None
        if test_vector.input_source in (InputSource.FROM_DRAM_QUEUE,):
            input_source_flag = InputSourceFlags.FROM_DRAM

        operator = getattr(torch, test_vector.operator)

        model_type = cls.MODEL_TYPES[test_vector.input_source]
        pytorch_model = (
            model_type(operator, test_vector.input_shape)
            if test_vector.input_source in (InputSource.CONST_EVAL_PASS,)
            else model_type(operator)
        )

        input_shapes = tuple([test_vector.input_shape])

        logger.trace(f"***input_shapes: {input_shapes}")

        VerifyUtils.verify(
            model=pytorch_model,
            test_device=test_device,
            input_shapes=input_shapes,
            input_params=input_params,
            input_source_flag=input_source_flag,
            dev_data_format=test_vector.dev_data_format,
            math_fidelity=test_vector.math_fidelity,
            pcc=test_vector.pcc,
            warm_reset=warm_reset,
            value_range=ValueRanges.SMALL,
        )


class TestParamsData:

    __test__ = False

    test_plan_implemented: TestPlan = None
    test_plan_not_implemented: TestPlan = None


class TestCollectionData:

    __test__ = False

    implemented = TestCollection(
        operators=[
            "relu",
            "sqrt",
            "reciprocal",
            "sigmoid",
            "abs",
            # "absolute",     # alias for abs
            "cos",
            "exp",
            "neg",
            # "negative",     # alias for neg
            "rsqrt",
            "sin",
            "square",
        ],
    )
    not_implemented = TestCollection(
        operators=[
            "acos",
            "arccos",
            "acosh",
            "arccosh",
            "angle",
            "asin",
            "arcsin",
            "asinh",
            "arcsinh",
            "atan",
            "arctan",
            "atanh",
            "arctanh",
            "bitwise_not",
            "ceil",
            "conj_physical",
            "cosh",
            "deg2rad",
            "digamma",
            "erf",
            "erfc",
            "erfinv",
            "exp2",
            "expm1",
            "fix",
            "floor",
            "frac",
            "lgamma",
            "log",
            "log10",
            "log1p",
            "log2",
            "logit",
            "i0",
            "isnan",
            "nan_to_num",
            "positive",
            "rad2deg",
            "round",
            "sign",
            "sgn",
            "signbit",
            "sinc",
            "sinh",
            "tan",
            "tanh",
            "trunc",
        ],
    )


TestParamsData.test_plan_implemented = TestPlan(
    verify=lambda test_device, test_vector: TestVerification.verify(
        test_device,
        test_vector,
    ),
    collections=[
        # Test operators with all shapes and input sources collection:
        TestCollection(
            operators=TestCollectionData.implemented.operators,
            input_sources=TestCollectionCommon.all.input_sources,
            input_shapes=TestCollectionCommon.all.input_shapes,
        ),
        # Test Data formats collection:
        TestCollection(
            operators=TestCollectionData.implemented.operators,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            dev_data_formats=[
                item
                for item in TestCollectionCommon.all.dev_data_formats
                if item not in TestCollectionCommon.single.dev_data_formats
            ],
            math_fidelities=TestCollectionCommon.single.math_fidelities,
        ),
        # Test Math fidelities collection:
        TestCollection(
            operators=TestCollectionData.implemented.operators,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            dev_data_formats=TestCollectionCommon.single.dev_data_formats,
            math_fidelities=TestCollectionCommon.all.math_fidelities,
        ),
    ],
    failing_rules=[
        # Skip 2D shapes as we don't test them:
        TestCollection(
            criteria=lambda test_vector: len(test_vector.input_shape) in (2,),
            skip_reason=FailingReasons.NOT_IMPLEMENTED,
        ),
        TestCollection(
            operators=["reciprocal"],
            input_sources=[InputSource.FROM_HOST],
            input_shapes=[(1, 2, 3, 4)],
            dev_data_formats=[
                DataFormat.Int8,
                DataFormat.Int32,
            ],
            math_fidelities=[MathFidelity.HiFi4],
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        TestCollection(
            operators=["sigmoid"],
            input_sources=[InputSource.FROM_HOST],
            input_shapes=[(1, 2, 3, 4)],
            dev_data_formats=[
                DataFormat.RawUInt8,
                DataFormat.RawUInt16,
                DataFormat.RawUInt32,
                DataFormat.Int8,
                DataFormat.UInt16,
                DataFormat.Int32,
            ],
            math_fidelities=[MathFidelity.HiFi4],
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        TestCollection(
            operators=["abs", "cos", "neg", "sin"],
            input_sources=[InputSource.FROM_HOST],
            input_shapes=[(1, 2, 3, 4)],
            dev_data_formats=[
                DataFormat.Int8,
                DataFormat.Int32,
            ],
            math_fidelities=[MathFidelity.HiFi4],
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        TestCollection(
            operators=["rsqrt"],
            input_sources=[InputSource.FROM_HOST],
            input_shapes=[(1, 2, 3, 4)],
            dev_data_formats=[
                DataFormat.Bfp2,
                DataFormat.Bfp2_b,
                DataFormat.Bfp4,
                DataFormat.Bfp4_b,
                DataFormat.Bfp8,
                DataFormat.Bfp8_b,
                DataFormat.Float16,
                DataFormat.Float32,
                DataFormat.Lf8,
            ],
            math_fidelities=[MathFidelity.HiFi4],
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        TestCollection(
            operators=["rsqrt"],
            input_sources=[InputSource.FROM_HOST],
            input_shapes=[(1, 2, 3, 4)],
            dev_data_formats=[DataFormat.Float16_b],
            math_fidelities=[
                MathFidelity.LoFi,
                MathFidelity.HiFi2,
                MathFidelity.HiFi3,
                MathFidelity.HiFi4,
            ],
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        TestCollection(
            operators=["square"],
            input_sources=[InputSource.FROM_HOST],
            input_shapes=[(1, 2, 3, 4)],
            dev_data_formats=[
                DataFormat.RawUInt8,
                DataFormat.RawUInt16,
                DataFormat.RawUInt32,
                DataFormat.Int8,
                DataFormat.UInt16,
                DataFormat.Int32,
            ],
            math_fidelities=[MathFidelity.HiFi4],
            failing_reason=FailingReasons.ATTRIBUTE_ERROR,
        ),
    ],
)


TestParamsData.test_plan_not_implemented = TestPlan(
    verify=lambda test_device, test_vector: TestVerification.verify(
        test_device,
        test_vector,
    ),
    collections=[
        TestCollection(
            operators=TestCollectionData.not_implemented.operators,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
        )
    ],
    failing_rules=[
        TestCollection(
            operators=TestCollectionData.not_implemented.operators,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            failing_reason=FailingReasons.NOT_IMPLEMENTED,
        ),
    ],
)


def get_test_plans() -> List[TestPlan]:
    return [
        TestParamsData.test_plan_implemented,
        TestParamsData.test_plan_not_implemented,
    ]
