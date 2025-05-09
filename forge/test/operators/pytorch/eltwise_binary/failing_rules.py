# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Failing rules for element-wise binary operators

import os

import forge
import torch

from test.operators.utils import InputSource
from test.operators.utils import TestCollection
from test.operators.utils import TestPlanUtils
from test.operators.utils import TestResultFailing
from test.operators.utils import FailingReasons
from test.operators.utils import TestCollectionCommon
from test.operators.utils import FailingRulesConverter


class TestIdsData:

    __test__ = False  # Avoid collecting TestIdsData as a pytest test

    add_failed_allclose_value_checker = TestPlanUtils.load_test_ids_from_file(
        f"{os.path.dirname(__file__)}/failed_tests_op_ids/add_operator/add_allclose_value_checker.txt"
    )

    add_failed_assertion_error = TestPlanUtils.load_test_ids_from_file(
        f"{os.path.dirname(__file__)}/failed_tests_op_ids/add_operator/add_assertion_error.txt"
    )

    sub_failed_allclose_value_checker = TestPlanUtils.load_test_ids_from_file(
        f"{os.path.dirname(__file__)}/failed_tests_op_ids/sub_operator/sub_allclose_value_checker.txt"
    )

    sub_failed_assertion_error = TestPlanUtils.load_test_ids_from_file(
        f"{os.path.dirname(__file__)}/failed_tests_op_ids/sub_operator/sub_assertion_error.txt"
    )

    div_failed_allclose_value_checker = TestPlanUtils.load_test_ids_from_file(
        f"{os.path.dirname(__file__)}/failed_tests_op_ids/div_operator/div_allclose_value_checker.txt"
    )

    div_failed_assertion_error = TestPlanUtils.load_test_ids_from_file(
        f"{os.path.dirname(__file__)}/failed_tests_op_ids/div_operator/div_assertion_error.txt"
    )

    ge_failed_dtype_mismatch = TestPlanUtils.load_test_ids_from_file(
        f"{os.path.dirname(__file__)}/failed_tests_op_ids/ge_operator/ge_dtype_mismatch.txt"
    )

    ne_failed_dtype_mismatch = TestPlanUtils.load_test_ids_from_file(
        f"{os.path.dirname(__file__)}/failed_tests_op_ids/ne_operator/ne_dtype_mismatch.txt"
    )

    gt_failed_dtype_mismatch = TestPlanUtils.load_test_ids_from_file(
        f"{os.path.dirname(__file__)}/failed_tests_op_ids/gt_operator/gt_dtype_mismatch.txt"
    )

    lt_failed_dtype_mismatch = TestPlanUtils.load_test_ids_from_file(
        f"{os.path.dirname(__file__)}/failed_tests_op_ids/lt_operator/lt_dtype_mismatch.txt"
    )


class FailingRulesData:

    # Fatal Python error: Aborted
    common = TestCollection(
        input_sources=TestCollectionCommon.single.input_sources,
        input_shapes=TestCollectionCommon.single.input_shapes,
        dev_data_formats=[torch.float16],
        math_fidelities=TestCollectionCommon.single.math_fidelities,
        failing_reason=FailingReasons.UNSUPPORTED_DATA_FORMAT,
        skip_reason=FailingReasons.UNSUPPORTED_DATA_FORMAT,
    )

    add = [
        # ValueError: Data mismatch -> AllCloseValueChecker (all_close):
        TestCollection(
            criteria=lambda test_vector: test_vector.get_id() in TestIdsData.add_failed_allclose_value_checker,
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        # AssertionError: Data mismatch on output 0 between framework and Forge codegen:
        TestCollection(
            criteria=lambda test_vector: test_vector.get_id() in TestIdsData.add_failed_assertion_error,
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        # RuntimeError: TT_ASSERT @ /home/vobojevic/src/ttforge/tt-forge-fe/forge/csrc/verif/verif_ops.cpp
        # info:
        # Unsupported data type for tensor a: {}
        TestCollection(
            input_sources=[InputSource.FROM_HOST],
            input_shapes=[(1, 2, 3, 4)],
            dev_data_formats=[torch.int8],
            failing_reason=FailingReasons.UNSUPPORTED_DATA_FORMAT,
        ),
        common,
    ]

    sub = [
        # ValueError: Data mismatch -> AllCloseValueChecker (all_close):
        TestCollection(
            criteria=lambda test_vector: test_vector.get_id() in TestIdsData.sub_failed_allclose_value_checker,
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        # AssertionError: Data mismatch on output 0 between framework and Forge codegen:
        TestCollection(
            criteria=lambda test_vector: test_vector.get_id() in TestIdsData.sub_failed_assertion_error,
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        # RuntimeError: TT_ASSERT @ /home/vobojevic/src/ttforge/tt-forge-fe/forge/csrc/verif/verif_ops.cpp
        # info:
        # Unsupported data type for tensor a: {}
        TestCollection(
            input_sources=[InputSource.FROM_HOST],
            input_shapes=[(1, 2, 3, 4)],
            dev_data_formats=[torch.int8],
            failing_reason=FailingReasons.UNSUPPORTED_DATA_FORMAT,
        ),
        common,
    ]

    mul = [
        # RuntimeError: TT_ASSERT @ /home/vobojevic/src/ttforge/tt-forge-fe/forge/csrc/verif/verif_ops.cpp
        # info:
        # Unsupported data type for tensor a: {}
        TestCollection(
            input_sources=[InputSource.FROM_HOST],
            input_shapes=[(1, 2, 3, 4)],
            dev_data_formats=[torch.int8],
            failing_reason=FailingReasons.UNSUPPORTED_DATA_FORMAT,
        ),
        # ValueError: Data mismatch -> AutomaticValueChecker (compare_with_golden)
        TestCollection(
            input_sources=[InputSource.FROM_HOST],
            input_shapes=[(1, 2, 3, 4)],
            dev_data_formats=[
                torch.int32,
                torch.int64,
            ],
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        common,
    ]

    div = [
        # ValueError: Data mismatch -> AllCloseValueChecker (all_close):
        TestCollection(
            criteria=lambda test_vector: test_vector.get_id() in TestIdsData.div_failed_allclose_value_checker,
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        # AssertionError: Data mismatch on output 0 between framework and Forge codegen:
        TestCollection(
            criteria=lambda test_vector: test_vector.get_id() in TestIdsData.div_failed_assertion_error,
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        # RuntimeError: TT_ASSERT @ /home/vobojevic/src/ttforge/tt-forge-fe/forge/csrc/verif/verif_ops.cpp
        # info:
        # Tensor a contains NaN/Inf values
        TestCollection(
            input_sources=[InputSource.FROM_ANOTHER_OP],
            input_shapes=[(7, 10, 1000, 100)],
            failing_reason=FailingReasons.UNSUPPORTED_TYPE_FOR_VALIDATION,  # TODO: check if this is correct
        ),
        # ValueError: Data mismatch -> AutomaticValueChecker (compare_with_golden)
        TestCollection(
            input_sources=[InputSource.FROM_HOST],
            input_shapes=[(1, 2, 3, 4)],
            dev_data_formats=[torch.int64],
            kwargs=[
                {"rounding_mode": None},
                {},
            ],
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        # RuntimeError: TT_ASSERT @ /home/vobojevic/src/ttforge/tt-forge-fe/forge/csrc/verif/verif_ops.cpp
        # info:
        # Input tensors must have the same data type, but got {} and {}
        TestCollection(
            input_sources=[InputSource.FROM_HOST],
            input_shapes=[(1, 2, 3, 4)],
            dev_data_formats=[
                torch.int8,
                torch.int32,
                torch.int64,
            ],
            kwargs=[
                {"rounding_mode": "trunc"},
                {"rounding_mode": "floor"},
            ],
            failing_reason=FailingReasons.UNSUPPORTED_DATA_FORMAT,
        ),
        common,
    ]

    ge = [
        # ValueError: Dtype mismatch: framework_model.dtype=torch.xxx, compiled_model.dtype=torch.xxx
        TestCollection(
            criteria=lambda test_vector: test_vector.get_id() in TestIdsData.ge_failed_dtype_mismatch,
            failing_reason=FailingReasons.DTYPE_MISMATCH,
        ),
        common,
    ]

    ne = [
        # ValueError: Dtype mismatch: framework_model.dtype=torch.xxx, compiled_model.dtype=torch.xxx
        TestCollection(
            criteria=lambda test_vector: test_vector.get_id() in TestIdsData.ne_failed_dtype_mismatch,
            failing_reason=FailingReasons.DTYPE_MISMATCH,
        ),
        common,
    ]

    gt = [
        # ValueError: Dtype mismatch: framework_model.dtype=torch.xxx, compiled_model.dtype=torch.xxx
        TestCollection(
            criteria=lambda test_vector: test_vector.get_id() in TestIdsData.gt_failed_dtype_mismatch,
            failing_reason=FailingReasons.DTYPE_MISMATCH,
        ),
        common,
    ]

    lt = [
        # ValueError: Dtype mismatch: framework_model.dtype=torch.xxx, compiled_model.dtype=torch.xxx
        TestCollection(
            criteria=lambda test_vector: test_vector.get_id() in TestIdsData.lt_failed_dtype_mismatch,
            failing_reason=FailingReasons.DTYPE_MISMATCH,
        ),
        common,
    ]

    maximum = [
        # RuntimeError: TT_ASSERT @ /home/vobojevic/src/ttforge/tt-forge-fe/forge/csrc/verif/verif_ops.cpp
        # info:
        # Unsupported data type for tensor a: {}
        TestCollection(
            input_sources=[InputSource.FROM_HOST],
            input_shapes=[(1, 2, 3, 4)],
            dev_data_formats=[torch.int8],
            math_fidelities=[forge.MathFidelity.HiFi4],
            failing_reason=FailingReasons.UNSUPPORTED_DATA_FORMAT,
        ),
        # ValueError: Data mismatch -> AutomaticValueChecker (compare_with_golden)
        TestCollection(
            input_sources=[InputSource.FROM_HOST],
            input_shapes=[(1, 2, 3, 4)],
            dev_data_formats=[torch.int64],
            math_fidelities=[forge.MathFidelity.HiFi4],
            failing_reason=FailingReasons.DATA_MISMATCH,
        ),
        common,
    ]

    minimum = [
        # RuntimeError: TT_ASSERT @ /home/vobojevic/src/ttforge/tt-forge-fe/forge/csrc/verif/verif_ops.cpp
        # info:
        # Unsupported data type for tensor a: {}
        TestCollection(
            input_sources=[InputSource.FROM_HOST],
            input_shapes=[(1, 2, 3, 4)],
            dev_data_formats=[torch.int8],
            math_fidelities=[forge.MathFidelity.HiFi4],
            failing_reason=FailingReasons.UNSUPPORTED_DATA_FORMAT,
        ),
        # RuntimeError: Tensor 2 - data type mismatch:
        TestCollection(
            input_sources=[InputSource.FROM_HOST],
            input_shapes=[(1, 2, 3, 4)],
            dev_data_formats=[
                torch.int32,
                torch.int64,
                torch.bfloat16,
            ],
            math_fidelities=[
                forge.MathFidelity.LoFi,
                forge.MathFidelity.HiFi2,
                forge.MathFidelity.HiFi3,
                forge.MathFidelity.HiFi4,
            ],
            failing_reason=FailingReasons.DTYPE_MISMATCH,
        ),
        common,
    ]
