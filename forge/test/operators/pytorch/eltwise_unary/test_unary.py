# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


# GENERAL OP SUPPORT TEST PLAN:
# 1. Operand type - any supported type (e.g. add, matmul, conv2d, etc.)
# 2. Operand source(s):
#    (+)  2.1 From another op
#           - Operator -> input
#    (+)  2.2 From DRAM queue - Removed from test plan
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

import os
import torch

from typing import List, Dict
from loguru import logger
from forge import MathFidelity, DataFormat

from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AllCloseValueChecker, AutomaticValueChecker

from test.operators.utils import InputSourceFlags, VerifyUtils
from test.operators.utils import InputSource
from test.operators.utils import PytorchUtils
from test.operators.utils import TestVector
from test.operators.utils import TestPlan
from test.operators.utils import TestPlanUtils
from test.operators.utils import FailingReasons
from test.operators.utils.compat import TestDevice
from test.operators.utils import TestCollection
from test.operators.utils import TestCollectionCommon
from test.operators.utils import TestCollectionTorch
from test.operators.utils import ValueRanges

from .models import ModelFromAnotherOp, ModelDirect, ModelConstEvalPass


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

        # Using AllCloseValueChecker in all cases except for integer data formats
        verify_config: VerifyConfig
        if test_vector.dev_data_format in TestCollectionTorch.int.dev_data_formats:
            verify_config = VerifyConfig(value_checker=AutomaticValueChecker())
        else:
            verify_config = VerifyConfig(value_checker=AllCloseValueChecker(rtol=1e-2, atol=1e-2))

        VerifyUtils.verify(
            model=pytorch_model,
            test_device=test_device,
            input_shapes=input_shapes,
            input_params=input_params,
            dev_data_format=test_vector.dev_data_format,
            math_fidelity=test_vector.math_fidelity,
            value_range=ValueRanges.SMALL,
            warm_reset=warm_reset,
            deprecated_verification=False,
            verify_config=verify_config,
        )


class TestParamsData:

    __test__ = False

    test_plan_implemented: TestPlan = None
    test_plan_implemented_float: TestPlan = None
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

    kwargs_gelu = [
        {"approximate": "tanh"},
        {},
    ]

    kwargs_leaky_relu = [
        {"negative_slope": 0.01, "inplace": True},
        {"negative_slope": 0.1, "inplace": False},
        {},
    ]

    @classmethod
    def generate_kwargs(cls, test_vector: TestVector):
        if test_vector.operator in ("clamp",):
            return cls.kwargs_clamp
        if test_vector.operator in ("pow",):
            return cls.kwargs_pow
        if test_vector.operator in ("gelu",):
            return cls.kwargs_gelu
        if test_vector.operator in ("leaky_relu",):
            return cls.kwargs_leaky_relu
        if test_vector.operator in ("cumsum",):
            return [{"dim": d} for d in range(0, len(test_vector.input_shape))]
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
            "cumsum",
            "isnan",
            "tanh",
        ],
    )
    implemented_float = TestCollection(
        operators=[
            "gelu",
            "leaky_relu",
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
            "trunc",
        ],
    )

    # torch.float16 is not supported well - python crashes
    common_to_skip = TestCollection(
        dev_data_formats=[torch.float16],
        failing_reason=FailingReasons.UNSUPPORTED_DATA_FORMAT,
        skip_reason=FailingReasons.UNSUPPORTED_DATA_FORMAT,
    )


class TestIdsData:

    __test__ = False  # Avoid collecting TestIdsData as a pytest test

    sqrt_runtime_error_failed = TestPlanUtils.load_test_ids_from_file(
        f"{os.path.dirname(__file__)}/failed_tests_op_ids/sqrt_operator/sqrt_runtime_error_failed.txt"
    )

    exp_all_close_value_checker = TestPlanUtils.load_test_ids_from_file(
        f"{os.path.dirname(__file__)}/failed_tests_op_ids/exp_operator/exp_all_close_value_checker.txt"
    )

    reciprocal_all_close_value_checker = TestPlanUtils.load_test_ids_from_file(
        f"{os.path.dirname(__file__)}/failed_tests_op_ids/reciprocal_operator/reciprocal_all_close_value_checker.txt"
    )

    rsqrt_runtime_error_failed = TestPlanUtils.load_test_ids_from_file(
        f"{os.path.dirname(__file__)}/failed_tests_op_ids/rsqrt_operator/rsqrt_runtime_error_failed.txt"
    )

    clamp_all_close_value_checker = TestPlanUtils.load_test_ids_from_file(
        f"{os.path.dirname(__file__)}/failed_tests_op_ids/clamp_operator/clamp_all_close_value_checker.txt"
    )

    log_runtime_error_failed = TestPlanUtils.load_test_ids_from_file(
        f"{os.path.dirname(__file__)}/failed_tests_op_ids/log_operator/log_runtime_error_failed.txt"
    )

    log1p_runtime_error_failed = TestPlanUtils.load_test_ids_from_file(
        f"{os.path.dirname(__file__)}/failed_tests_op_ids/log1p_operator/log1p_runtime_error_failed.txt"
    )

    log1p_all_close_value_checker = TestPlanUtils.load_test_ids_from_file(
        f"{os.path.dirname(__file__)}/failed_tests_op_ids/log1p_operator/log1p_all_close_value_checker.txt"
    )

    cumsum_all_close_value_checker = TestPlanUtils.load_test_ids_from_file(
        f"{os.path.dirname(__file__)}/failed_tests_op_ids/cumsum_operator/cumsum_all_close_value_checker.txt"
    )

    isnan_dtype_mismatch_failed = TestPlanUtils.load_test_ids_from_file(
        f"{os.path.dirname(__file__)}/failed_tests_op_ids/isnan_operator/isnan_dtype_mismatch_failed.txt"
    )

    tanh_all_close_value_checker = TestPlanUtils.load_test_ids_from_file(
        f"{os.path.dirname(__file__)}/failed_tests_op_ids/tanh_operator/tanh_all_close_value_checker.txt"
    )

    pow_all_close_value_checker = TestPlanUtils.load_test_ids_from_file(
        f"{os.path.dirname(__file__)}/failed_tests_op_ids/pow_operator/pow_all_close_value_checker.txt"
    )

    pow_assertion_error_exponent_value = TestPlanUtils.load_test_ids_from_file(
        f"{os.path.dirname(__file__)}/failed_tests_op_ids/pow_operator/pow_assertion_error_exponent_value.txt"
    )

    pow_assertion_error_pcc_nan = TestPlanUtils.load_test_ids_from_file(
        f"{os.path.dirname(__file__)}/failed_tests_op_ids/pow_operator/pow_assertion_error_pcc_nan.txt"
    )

    pow_automatic_value_checker = TestPlanUtils.load_test_ids_from_file(
        f"{os.path.dirname(__file__)}/failed_tests_op_ids/pow_operator/pow_automatic_value_checker.txt"
    )

    pow_runtime_error_failed = TestPlanUtils.load_test_ids_from_file(
        f"{os.path.dirname(__file__)}/failed_tests_op_ids/pow_operator/pow_runtime_error_failed.txt"
    )

    pow_value_error_dtype_mismatch = TestPlanUtils.load_test_ids_from_file(
        f"{os.path.dirname(__file__)}/failed_tests_op_ids/pow_operator/pow_value_error_dtype_mismatch.txt"
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
                for item in TestCollectionTorch.all.dev_data_formats
                if item not in TestCollectionTorch.single.dev_data_formats
            ],
            math_fidelities=TestCollectionCommon.single.math_fidelities,
        ),
        # Test Math fidelities collection:
        TestCollection(
            operators=TestCollectionData.implemented.operators,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
            dev_data_formats=TestCollectionTorch.single.dev_data_formats,
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
            dev_data_formats=TestCollectionTorch.all.dev_data_formats,
            math_fidelities=TestCollectionCommon.single.math_fidelities,
        ),
    ],
    failing_rules=[
        # ValueError: Dtype mismatch: framework_model.dtype=torch.float32, compiled_model.dtype=torch.int32
        TestCollection(
            operators=["sqrt", "exp", "reciprocal", "rsqrt", "log", "log1p", "sigmoid", "cos", "sin", "tanh"],
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            dev_data_formats=[
                torch.int8,
                torch.int32,
                torch.int64,
            ],
            math_fidelities=TestCollectionCommon.single.math_fidelities,
            failing_reason=FailingReasons.DTYPE_MISMATCH,
        ),
        # *********** sqrt failing rules ***********
        # RuntimeError: ... !has_special_values(a)
        TestCollection(
            criteria=lambda test_vector: test_vector.get_id() in TestIdsData.sqrt_runtime_error_failed,
            failing_reason=FailingReasons.UNSUPPORTED_SPECIAL_CASE,
        ),
        # *********** exp failing rules ***********
        # AllClose ValueChecker
        TestCollection(
            criteria=lambda test_vector: test_vector.get_id() in TestIdsData.exp_all_close_value_checker,
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        # *********** reciprocal failing rules ***********
        # AllClose ValueChecker
        TestCollection(
            criteria=lambda test_vector: test_vector.get_id() in TestIdsData.reciprocal_all_close_value_checker,
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        # *********** rsqrt failing rules ***********
        # RuntimeError: ... !has_special_values(a)
        TestCollection(
            criteria=lambda test_vector: test_vector.get_id() in TestIdsData.rsqrt_runtime_error_failed,
            failing_reason=FailingReasons.UNSUPPORTED_SPECIAL_CASE,
        ),
        # *********** clamp failing rules ***********
        # RuntimeError: value cannot be converted to type at::BFloat16 without overflow
        TestCollection(
            operators=["clamp"],
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            kwargs=[
                {"min": 0.2},
                {"max": 0.2},
            ],
            dev_data_formats=TestCollectionTorch.single.dev_data_formats,
            failing_reason=FailingReasons.UNSUPPORTED_SPECIAL_CASE,
        ),
        # ValueError: Dtype mismatch: framework_model.dtype=torch.float32, compiled_model.dtype=torch.int32
        TestCollection(
            operators=["clamp"],
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            kwargs=[
                {"min": 0.0, "max": 0.5},
                {"min": 0.2},
                {"max": 0.2},
            ],
            dev_data_formats=[
                torch.int8,
                torch.int32,
                torch.int64,
            ],
            math_fidelities=TestCollectionCommon.single.math_fidelities,
            failing_reason=FailingReasons.DTYPE_MISMATCH,
        ),
        # Unsupported DataType!
        TestCollection(
            operators=["clamp"],
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            kwargs=[
                {"min": 0.5, "max": 0.0},
            ],
            dev_data_formats=[
                torch.int8,
                torch.int32,
                torch.int64,
            ],
            math_fidelities=TestCollectionCommon.single.math_fidelities,
            failing_reason=FailingReasons.UNSUPPORTED_DATA_FORMAT,
        ),
        # AllClose ValueChecker
        TestCollection(
            criteria=lambda test_vector: test_vector.get_id() in TestIdsData.clamp_all_close_value_checker,
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        # *********** log failing rules ***********
        # RuntimeError: ... !has_special_values(a)
        TestCollection(
            criteria=lambda test_vector: test_vector.get_id() in TestIdsData.log_runtime_error_failed,
            failing_reason=FailingReasons.UNSUPPORTED_SPECIAL_CASE,
        ),
        # ************ log1p failing rules ***********
        # RuntimeError: ... !has_special_values(a)
        TestCollection(
            criteria=lambda test_vector: test_vector.get_id() in TestIdsData.log1p_runtime_error_failed,
            failing_reason=FailingReasons.UNSUPPORTED_SPECIAL_CASE,
        ),
        # AllClose ValueChecker
        TestCollection(
            criteria=lambda test_vector: test_vector.get_id() in TestIdsData.log1p_all_close_value_checker,
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        # ************ cumsum failing rules ***********
        # RuntimeError: ... !has_special_values(a)
        TestCollection(
            operators=["cumsum"],
            input_sources=[InputSource.FROM_ANOTHER_OP],
            input_shapes=[(10, 1000, 100)],
            kwargs=[{"dim": 2}],
            failing_reason=FailingReasons.UNSUPPORTED_SPECIAL_CASE,
        ),
        # AllClose ValueChecker
        TestCollection(
            criteria=lambda test_vector: test_vector.get_id() in TestIdsData.cumsum_all_close_value_checker,
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        # ************ square failing rules ***********
        TestCollection(
            operators=["square"],
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            dev_data_formats=[
                torch.int8,
                torch.int32,
                torch.int64,
            ],
            math_fidelities=TestCollectionCommon.single.math_fidelities,
            failing_reason=FailingReasons.ATTRIBUTE_ERROR,
        ),
        # ************* isnan failing rules ***********
        # ValueError: Dtype mismatch: framework_model.dtype=torch.uint8, compiled_model.dtype=torch.float32
        TestCollection(
            criteria=lambda test_vector: test_vector.get_id() in TestIdsData.isnan_dtype_mismatch_failed,
            failing_reason=FailingReasons.DTYPE_MISMATCH,
        ),
        # ************** tanh failing rules ***********
        # AllClose ValueChecker
        TestCollection(
            criteria=lambda test_vector: test_vector.get_id() in TestIdsData.tanh_all_close_value_checker,
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        # ************** pow failing rules ***********
        TestCollection(
            criteria=lambda test_vector: test_vector.get_id() in TestIdsData.pow_all_close_value_checker,
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        TestCollection(
            criteria=lambda test_vector: test_vector.get_id() in TestIdsData.pow_assertion_error_exponent_value,
            failing_reason=FailingReasons.UNSUPPORTED_SPECIAL_CASE,
        ),
        TestCollection(
            criteria=lambda test_vector: test_vector.get_id() in TestIdsData.pow_assertion_error_pcc_nan,
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        TestCollection(
            criteria=lambda test_vector: test_vector.get_id() in TestIdsData.pow_automatic_value_checker,
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        TestCollection(
            criteria=lambda test_vector: test_vector.get_id() in TestIdsData.pow_runtime_error_failed,
            failing_reason=FailingReasons.UNSUPPORTED_SPECIAL_CASE,
        ),
        TestCollection(
            criteria=lambda test_vector: test_vector.get_id() in TestIdsData.pow_value_error_dtype_mismatch,
            failing_reason=FailingReasons.DTYPE_MISMATCH,
        ),
        TestCollectionData.common_to_skip,
    ],
)


TestParamsData.test_plan_implemented_float = TestPlan(
    verify=lambda test_device, test_vector: TestVerification.verify(
        test_device,
        test_vector,
    ),
    collections=[
        # Test gelu, leaky_relu operators collection:
        TestCollection(
            operators=TestCollectionData.implemented_float.operators,
            input_sources=TestCollectionCommon.all.input_sources,
            input_shapes=TestCollectionCommon.all.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
        ),
        # Test gelu, leaky_relu data formats collection:
        TestCollection(
            operators=TestCollectionData.implemented_float.operators,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
            dev_data_formats=[
                item
                for item in TestCollectionTorch.float.dev_data_formats
                if item not in TestCollectionTorch.single.dev_data_formats
            ],
            math_fidelities=TestCollectionCommon.single.math_fidelities,
        ),
        # Test gelu, leaky_relu math fidelities collection:
        TestCollection(
            operators=TestCollectionData.implemented_float.operators,
            input_sources=TestCollectionCommon.single.input_sources,
            input_shapes=TestCollectionCommon.single.input_shapes,
            kwargs=lambda test_vector: TestParamsData.generate_kwargs(test_vector),
            dev_data_formats=TestCollectionTorch.single.dev_data_formats,
            math_fidelities=TestCollectionCommon.all.math_fidelities,
        ),
    ],
    failing_rules=[
        TestCollectionData.common_to_skip,
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
        TestCollectionData.common_to_skip,
    ],
)


def get_test_plans() -> List[TestPlan]:
    return [
        TestParamsData.test_plan_implemented,
        TestParamsData.test_plan_implemented_float,
        TestParamsData.test_plan_not_implemented,
    ]
