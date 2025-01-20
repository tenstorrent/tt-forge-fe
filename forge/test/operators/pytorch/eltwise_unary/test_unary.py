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


import torch

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

from .models import ModelFromAnotherOp, ModelDirect, ModelConstEvalPass


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

        kwargs = test_vector.kwargs if test_vector.kwargs else {}

        model_type = cls.MODEL_TYPES[test_vector.input_source]
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

    no_kwargs = [
        None,
    ]

    kwargs_clamp = [
        {"min": 0.0, "max": 0.5},
        {"min": 0.5, "max": 0.0},
        {"min": 0.2},
        {"max": 0.2},
    ]

    kwargs_pow = [
        {"exponent": 0.5},
        {"exponent": 2.0},
        {"exponent": 10.0},
    ]

    @classmethod
    def generate_kwargs(cls, test_vector: TestVector):
        if test_vector.operator in ("clamp",):
            return cls.kwargs_clamp
        if test_vector.operator in ("pow",):
            return cls.kwargs_pow
        return cls.no_kwargs


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
            "pow",
            "clamp",
            # "clip",         # alias for clamp
            "log",
            "log1p",
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
            "log10",
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
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
        ),
        # Test Data formats collection:
        TestCollection(
            operators=TestCollectionData.implemented.operators,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
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
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
            dev_data_formats=TestCollectionCommon.single.dev_data_formats,
            math_fidelities=TestCollectionCommon.all.math_fidelities,
        ),
        # Test Special cases collection for "pow" operator:
        TestCollection(
            operators=["pow"],
            input_sources=TestCollectionCommon.all.input_sources,
            input_shapes=TestCollectionCommon.specific.input_shapes,
            kwargs=[
                {"exponent": 0.0},
                {"exponent": 0.5},
                {"exponent": 2.0},
                {"exponent": 10.0},
                {"exponent": -2.0},
                {"exponent": -1.26},
                {"exponent": 1.52},
            ],
            dev_data_formats=TestCollectionCommon.all.dev_data_formats,
            math_fidelities=TestCollectionCommon.single.math_fidelities,
        ),
    ],
    failing_rules=[
        # Skip 2D shapes as we don't test them:
        TestCollection(
            criteria=lambda test_vector: len(test_vector.input_shape) in (2,),
            skip_reason=FailingReasons.NOT_IMPLEMENTED,
        ),
        # reciprocal: Data mismatch for specific data formats:
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
        # sigmoid: Data mismatch for specific data formats:
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
        # abs, cos, neg, sin: Data mismatch for specific data formats:
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
        # rsqrt: Data mismatch for specific data formats:
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
        # rsqrt: Data mismatch for specific math fidelities:
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
        # square: Attribute error for specific data formats:
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
        # pow: Exponent 0.0 data mismatch:
        TestCollection(
            operators=["pow"],
            kwargs=[{"exponent": 0.0}],
            input_sources=[
                InputSource.FROM_HOST,
                InputSource.FROM_ANOTHER_OP,
                InputSource.FROM_DRAM_QUEUE,
            ],
            input_shapes=[
                (1, 2, 3, 4),
                (1, 45, 17),
                (1, 100, 100),
                (1, 10000, 1),
                (1, 17, 41),
                (11, 1, 23),
                (1, 11, 1, 23),
                (1, 1, 10, 1000),
                (14, 13, 89, 3),
            ],
            dev_data_formats=[
                DataFormat.RawUInt8,
                DataFormat.RawUInt16,
                DataFormat.RawUInt32,
                DataFormat.UInt16,
            ],
            math_fidelities=[MathFidelity.HiFi4],
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        # pow: Exponent 0.5 data mismatch:
        TestCollection(
            operators=["pow"],
            kwargs=[{"exponent": 0.5}],
            input_sources=[InputSource.CONST_EVAL_PASS],
            input_shapes=[
                (1, 45, 17),
                (1, 100, 100),
                (1, 10000, 1),
                (1, 17, 41),
                (11, 1, 23),
                (1, 11, 1, 23),
                (1, 1, 10, 1000),
                (14, 13, 89, 3),
            ],
            dev_data_formats=[
                DataFormat.Bfp2,
                DataFormat.Bfp2_b,
                DataFormat.Bfp4,
                DataFormat.Bfp4_b,
                DataFormat.Bfp8,
                DataFormat.Bfp8_b,
                DataFormat.Float16,
                DataFormat.Float16_b,
                DataFormat.Float32,
                DataFormat.Lf8,
                DataFormat.Int8,
                DataFormat.Int32,
            ],
            math_fidelities=[MathFidelity.HiFi4],
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        # pow: Exponent 2.0 data mismatch:
        TestCollection(
            operators=["pow"],
            kwargs=[{"exponent": 2.0}],
            input_sources=[
                InputSource.FROM_ANOTHER_OP,
                InputSource.CONST_EVAL_PASS,
            ],
            input_shapes=[
                (1, 45, 17),
                (1, 100, 100),
                (1, 17, 41),
                (11, 1, 23),
                (1, 11, 1, 23),
                (1, 1, 10, 1000),
                (14, 13, 89, 3),
            ],
            dev_data_formats=[
                DataFormat.Int8,
                DataFormat.Int32,
            ],
            math_fidelities=[MathFidelity.HiFi4],
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        # pow: Exponent 2.0 data mismatch:
        TestCollection(
            operators=["pow"],
            kwargs=[{"exponent": 2.0}],
            input_sources=[InputSource.CONST_EVAL_PASS],
            input_shapes=[(1, 10000, 1)],
            dev_data_formats=[
                DataFormat.Int8,
                DataFormat.Int32,
            ],
            math_fidelities=[MathFidelity.HiFi4],
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        # pow: Exponent 10.0 data mismatch:
        TestCollection(
            operators=["pow"],
            kwargs=[{"exponent": 10.0}],
            input_sources=[InputSource.FROM_HOST],
            input_shapes=[(1, 2, 3, 4)],
            dev_data_formats=[
                DataFormat.Int8,
                DataFormat.Int32,
            ],
            math_fidelities=[MathFidelity.HiFi4],
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        # pow: Exponent 10.0 data mismatch:
        TestCollection(
            operators=["pow"],
            kwargs=[{"exponent": 10.0}],
            input_sources=[
                InputSource.FROM_HOST,
                InputSource.FROM_DRAM_QUEUE,
                InputSource.CONST_EVAL_PASS,
            ],
            input_shapes=[
                (1, 45, 17),
                (1, 100, 100),
                (1, 10000, 1),
                (1, 17, 41),
                (11, 1, 23),
                (1, 11, 1, 23),
                (1, 1, 10, 1000),
                (14, 13, 89, 3),
            ],
            dev_data_formats=[
                DataFormat.Int8,
                DataFormat.Int32,
            ],
            math_fidelities=[MathFidelity.HiFi4],
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        # pow: Exponent -2.0 data mismatch:
        TestCollection(
            operators=["pow"],
            kwargs=[{"exponent": -2.0}],
            input_shapes=[
                (1, 1000, 100),
                (100, 100, 100),
                (10, 1000, 100),
                (10, 10000, 1),
                (32, 32, 64),
                (64, 160, 96),
                (1, 100, 100, 100),
                (1, 10, 1000, 100),
                (1, 10, 10000, 1),
                (1, 32, 32, 64),
                (1, 64, 160, 96),
                (6, 100, 100, 100),
                (7, 10, 1000, 100),
                (8, 1, 10, 1000),
                (9, 1, 9920, 1),
                (10, 10, 10000, 1),
                (11, 32, 32, 64),
                (12, 64, 160, 96),
                (13, 11, 17, 41),
            ],
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        # pow: Exponent -2.0, -1.26, 1.52 data mismatch:
        TestCollection(
            operators=["pow"],
            kwargs=[
                {"exponent": -2.0},
                {"exponent": -1.26},
                {"exponent": 1.52},
            ],
            input_shapes=[
                (1, 45, 17),
                (1, 100, 100),
                (1, 10000, 1),
                (1, 17, 41),
                (11, 1, 23),
                (1, 11, 1, 23),
                (1, 1, 10, 1000),
                (14, 13, 89, 3),
            ],
            dev_data_formats=[
                DataFormat.Int8,
                DataFormat.Int32,
            ],
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        # pow: Exponent -2.0 data mismatch:
        TestCollection(
            operators=["pow"],
            kwargs=[{"exponent": -2.0}],
            input_shapes=[(14, 13, 89, 3)],
            input_sources=[
                InputSource.FROM_HOST,
                InputSource.FROM_DRAM_QUEUE,
                InputSource.CONST_EVAL_PASS,
            ],
            dev_data_formats=[
                DataFormat.Bfp2,
                DataFormat.Bfp2_b,
                DataFormat.Bfp4,
                DataFormat.Bfp4_b,
                DataFormat.Bfp8,
                DataFormat.Bfp8_b,
                DataFormat.Float16,
                DataFormat.Float16_b,
                DataFormat.Float32,
                DataFormat.Lf8,
            ],
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        # pow: Cases float exponent (values -1.26 and 1.52) are not supported:
        TestCollection(
            operators=["pow"],
            kwargs=[
                {"exponent": -1.26},
                {"exponent": 1.52},
            ],
            failing_reason=FailingReasons.UNSUPPORTED_SPECIAL_CASE,
        ),
        # clamp: min=0.5, max=0.0 data mismatch:
        TestCollection(
            operators=["clamp"],
            input_sources=[InputSource.CONST_EVAL_PASS],
            kwargs=[
                {"min": 0.5},
                {"max": 0.0},
            ],
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        # clamp: min=0.2 data mismatch:
        TestCollection(
            operators=["clamp"],
            input_sources=[InputSource.FROM_HOST],
            input_shapes=[(1, 2, 3, 4)],
            dev_data_formats=[
                DataFormat.RawUInt8,
                DataFormat.RawUInt16,
                DataFormat.RawUInt32,
                DataFormat.UInt16,
                DataFormat.Int8,
                DataFormat.Int32,
            ],
            kwargs=[
                {"min": 0.2},
            ],
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        TestCollection(
            operators=["clamp"],
            input_sources=[InputSource.FROM_HOST],
            input_shapes=[(1, 2, 3, 4)],
            dev_data_formats=[
                DataFormat.Int8,
                DataFormat.Int32,
            ],
            kwargs=[
                {"max": 0.2},
            ],
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        # log: Data mismatch for specific data formats:
        TestCollection(
            operators=["log"],
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
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        # log: Data mismatch for specific math fidelities:
        TestCollection(
            operators=["log"],
            input_sources=[InputSource.FROM_HOST],
            input_shapes=[(1, 2, 3, 4)],
            dev_data_formats=[
                DataFormat.Float16_b,
            ],
            math_fidelities=[
                MathFidelity.LoFi,
                MathFidelity.HiFi2,
                MathFidelity.HiFi3,
                MathFidelity.HiFi4,
            ],
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        # log1p: Data mismatch for specific data formats:
        TestCollection(
            operators=["log1p"],
            input_sources=[InputSource.FROM_HOST],
            input_shapes=[(1, 2, 3, 4)],
            dev_data_formats=[
                DataFormat.RawUInt8,
                DataFormat.RawUInt16,
                DataFormat.RawUInt32,
                DataFormat.UInt16,
            ],
            failing_reason=FailingReasons.DATA_MISMATCH,
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
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
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
