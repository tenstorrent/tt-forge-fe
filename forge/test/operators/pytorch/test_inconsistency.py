# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Tests for testing all pytorch operators


# Examples
# pytest -svv forge/test/operators/pytorch/test_inconsistency.py::test_binary_order1
# pytest -svv forge/test/operators/pytorch/test_inconsistency.py::test_binary_order2
# pytest -svv forge/test/operators/pytorch/test_inconsistency.py::test_binary_with_reset


import os
import pytest

from loguru import logger

from test.operators.utils import DeviceUtils
from test.operators.utils import TestVector
from test.operators.utils import TestCollection
from test.operators.utils import TestPlanScanner


class TestParamsData:

    __test__ = False  # Avoid collecting TestParamsData as a pytest test

    test_suite = TestPlanScanner.build_test_suite(scan_file=__file__, scan_package=__package__)


test_suite = TestParamsData.test_suite


class TestInconsistency:

    __test__ = False  # Avoid collecting TestInconsistency as a pytest test

    test_ids = [
        "div-CONST_EVAL_PASS-{}-(13, 89, 3)-None-None",
        "mul-CONST_EVAL_PASS-{}-(2, 3, 4)-None-None",
        "mul-CONST_EVAL_PASS-{}-(11, 45, 17)-None-None",
    ]

    # The collection defines test vectors that triggers a warm reset of the device before verification
    # It allows specifying individual test vectors that should trigger a warm reset and skip warn reset for others
    warm_reset_collection = TestCollection(
        criteria=lambda test_vector: test_vector.get_id()
        in [
            "div-CONST_EVAL_PASS-{}-(13, 89, 3)-None-None",  # Resetting before this test does not cause halt on step 'Running model forward on device...'
            "mul-CONST_EVAL_PASS-{}-(2, 3, 4)-None-None",  # Uncomment to cause halt on step 'Running model forward on device...'
            # "mul-CONST_EVAL_PASS-{}-(11, 45, 17)-None-None",  # Uncomment to cause halt on step 'Running model forward on device...'
        ],
    )


# The fixture is used to setup warm reset before the test, based on the warm_reset_collection_inconsistency collection
@pytest.fixture
def warm_reset_inconsistency(test_vector: TestVector):
    if test_vector in TestInconsistency.warm_reset_collection:
        logger.warning(f"Test vector {test_vector.get_id()} requires warm reset")
        DeviceUtils.warm_reset()
    yield


# The fixture is used to setup warm reset before each test
@pytest.fixture
def warm_reset_all():
    DeviceUtils.warm_reset()
    yield


@pytest.mark.parametrize(
    "test_vector",
    test_suite.query_from_id_list(TestInconsistency.test_ids).to_params(),
)
def test_binary_order1(test_vector: TestVector, test_device):
    test_vector.verify(test_device)


@pytest.mark.parametrize(
    "test_vector",
    test_suite.query_from_id_list(TestInconsistency.test_ids).reverse().to_params(),
)
def test_binary_order2(test_vector: TestVector, test_device):
    test_vector.verify(test_device)


@pytest.mark.parametrize(
    "test_vector",
    test_suite.query_from_id_list(TestInconsistency.test_ids).to_params(),
)
# @pytest.mark.usefixtures("warm_reset_inconsistency")
def test_binary_with_reset(test_vector: TestVector, test_device, warm_reset_inconsistency):
    test_vector.verify(test_device)
