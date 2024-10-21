# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Tests for testing all pytorch operators


# Examples
# TEST_ID='no_device-ge-FROM_HOST-None-(1, 2, 3, 4)-Float16_b-HiFi4' pytest -svv forge/test/operators/pytorch/test_all.py::test_single

# pytest -svv forge/test/operators/pytorch/test_all.py::test_plan
# pytest -svv forge/test/operators/pytorch/test_all.py::test_skipped
# pytest -svv forge/test/operators/pytorch/test_all.py::test_failed
# pytest -svv forge/test/operators/pytorch/test_all.py::test_not_implemented
# pytest -svv forge/test/operators/pytorch/test_all.py::test_data_mismatch
# pytest -svv forge/test/operators/pytorch/test_all.py::test_unsupported_df
# pytest -svv forge/test/operators/pytorch/test_all.py::test_unsupported_df_fatal


import os
import pytest
import forge

from loguru import logger

from test.operators.utils import DeviceUtils
from test.operators.utils import TestVector
from test.operators.utils import TestCollection
from test.operators.utils import TestPlanScanner
from test.operators.utils import FailingReasons


class TestVerification:

    DRY_RUN = False
    # DRY_RUN = True

    @classmethod
    def verify(cls, test_vector: TestVector, test_device):
        if cls.DRY_RUN:
            # pytest.skip("Dry run")
            return
        test_vector.verify(test_device)


class TestParamsData:

    __test__ = False  # Avoid collecting TestParamsData as a pytest test

    test_suite = TestPlanScanner.build_test_suite(current_directory=os.path.dirname(__file__))

    @classmethod
    def get_single_list(cls):
        test_id_single = os.getenv("TEST_ID", None)
        return [test_id_single] if test_id_single else []


class TestCollectionData:

    __test__ = False  # Avoid collecting TestCollectionData as a pytest test

    all = TestCollection(
        operators=None,  # All operators
    )


class VectorLambdas:

    ALL_OPERATORS = lambda test_vector: test_vector in TestCollectionData.all
    NONE = lambda test_vector: False

    FAILING = (
        lambda test_vector: test_vector.failing_result is not None
        and test_vector.failing_result.failing_reason is not None
    )
    SKIPED = (
        lambda test_vector: test_vector.failing_result is not None
        and test_vector.failing_result.skip_reason is not None
    )
    NOT_FAILING = (
        lambda test_vector: test_vector.failing_result is None or test_vector.failing_result.failing_reason is None
    )
    NOT_SKIPED = (
        lambda test_vector: test_vector.failing_result is None or test_vector.failing_result.skip_reason is None
    )

    HAS_DATA_FORMAT = lambda test_vector: test_vector.dev_data_format is not None
    NO_DATA_FORMAT = lambda test_vector: test_vector.dev_data_format is None

    NOT_IMPLEMENTED = (
        lambda test_vector: test_vector.failing_result is not None
        and test_vector.failing_result.failing_reason == FailingReasons.NOT_IMPLEMENTED
    )
    DATA_MISMATCH = (
        lambda test_vector: test_vector.failing_result is not None
        and test_vector.failing_result.failing_reason == FailingReasons.DATA_MISMATCH
    )
    UNSUPPORTED_DATA_FORMAT = (
        lambda test_vector: test_vector.failing_result is not None
        and test_vector.failing_result.failing_reason == FailingReasons.UNSUPPORTED_DATA_FORMAT
    )


test_suite = TestParamsData.test_suite


@pytest.mark.parametrize(
    "test_vector",
    test_suite.query_all().filter(VectorLambdas.ALL_OPERATORS)
    # .filter(VectorLambdas.HAS_DATA_FORMAT)
    # .filter(VectorLambdas.NO_DATA_FORMAT)
    # .filter(lambda test_vector: test_vector in TestCollection(
    #     operators=["add", ],
    #     # input_shapes=[
    #     #     (1, 1)
    #     # ],
    #     failing_reason=FailingReasons.DATA_MISMATCH,
    #     # operators=["sub", ],
    # ))
    # .log()
    # .filter(lambda test_vector: test_vector.dev_data_format in [forge.DataFormat.Bfp2])
    # .log()
    # .skip(lambda test_vector: test_vector.kwargs is not None and "rounding_mode" in test_vector.kwargs and test_vector.kwargs["rounding_mode"] in ["trunc", "floor"])
    # .log()
    # .range(0, 10)
    # .log()
    # .range_skip(2, 5)
    # .log()
    # .index(3, 5)
    # .log()
    # .range(0, 10)
    # .log()
    # .filter(VectorLambdas.NONE)
    .to_params(),
)
def test_plan(test_vector: TestVector, test_device):
    TestVerification.verify(test_vector, test_device)


@pytest.mark.parametrize(
    "test_vector",
    test_suite.query_all()
    .filter(VectorLambdas.ALL_OPERATORS)
    .filter(VectorLambdas.FAILING)
    .filter(VectorLambdas.NOT_SKIPED)
    .to_params(),
)
def test_failed(test_vector: TestVector, test_device):
    TestVerification.verify(test_vector, test_device)


@pytest.mark.parametrize(
    "test_vector",
    test_suite.query_all().filter(VectorLambdas.ALL_OPERATORS).filter(VectorLambdas.SKIPED).to_params(),
)
def test_skipped(test_vector: TestVector, test_device):
    TestVerification.verify(test_vector, test_device)


@pytest.mark.parametrize(
    "test_vector",
    test_suite.query_all()
    .filter(VectorLambdas.ALL_OPERATORS)
    .filter(VectorLambdas.FAILING)
    .filter(VectorLambdas.UNSUPPORTED_DATA_FORMAT)
    .filter(lambda test_vector: test_vector.failing_result.skip_reason == FailingReasons.FATAL_ERROR)
    .to_params(),
)
def test_unsupported_df_fatal(test_vector: TestVector, test_device):
    TestVerification.verify(test_vector, test_device)


@pytest.mark.parametrize(
    "test_vector",
    test_suite.query_all()
    .filter(VectorLambdas.ALL_OPERATORS)
    .filter(VectorLambdas.FAILING)
    .filter(VectorLambdas.NOT_SKIPED)
    .filter(VectorLambdas.UNSUPPORTED_DATA_FORMAT)
    .to_params(),
)
def test_unsupported_df(test_vector: TestVector, test_device):
    TestVerification.verify(test_vector, test_device)


@pytest.mark.parametrize(
    "test_vector",
    test_suite.query_all()
    .filter(VectorLambdas.ALL_OPERATORS)
    .filter(VectorLambdas.FAILING)
    .filter(VectorLambdas.NOT_SKIPED)
    .filter(VectorLambdas.NOT_IMPLEMENTED)
    .to_params(),
)
def test_not_implemented(test_vector: TestVector, test_device):
    TestVerification.verify(test_vector, test_device)


@pytest.mark.parametrize(
    "test_vector",
    test_suite.query_all()
    .filter(VectorLambdas.ALL_OPERATORS)
    .filter(VectorLambdas.FAILING)
    .filter(VectorLambdas.NOT_SKIPED)
    .filter(VectorLambdas.DATA_MISMATCH)
    .to_params(),
)
def test_data_mismatch(test_vector: TestVector, test_device):
    TestVerification.verify(test_vector, test_device)


@pytest.mark.parametrize(
    "test_vector",
    test_suite.query_from_id_list(TestParamsData.get_single_list()).to_params(),
)
def test_single(test_vector: TestVector, test_device):
    TestVerification.verify(test_vector, test_device)
