# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Tests for testing all pytorch operators


# Examples
# TEST_ID='no_device-ge-FROM_HOST-None-(1, 2, 3, 4)-Float16_b-HiFi4' pytest -svv forge/test/operators/pytorch/test_all.py::test_single

# pytest -svv forge/test/operators/pytorch/test_all.py::test_plan
# pytest -svv forge/test/operators/pytorch/test_all.py::test_failed
# pytest -svv forge/test/operators/pytorch/test_all.py::test_skipped
# pytest -svv forge/test/operators/pytorch/test_all.py::test_fatal
# pytest -svv forge/test/operators/pytorch/test_all.py::test_not_implemented
# pytest -svv forge/test/operators/pytorch/test_all.py::test_data_mismatch
# pytest -svv forge/test/operators/pytorch/test_all.py::test_unsupported_df
# pytest -svv forge/test/operators/pytorch/test_all.py::test_custom


import os
import pytest
import forge

from loguru import logger

from test.operators.utils import DeviceUtils
from test.operators.utils import TestVector
from test.operators.utils import TestCollection
from test.operators.utils import TestCollectionCommon
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

    quick = TestCollection(
        input_shapes=[]
        + [shape for shape in TestCollectionCommon.all.input_shapes if len(shape) in (2,) and shape[0] == 1][:2]
        + [shape for shape in TestCollectionCommon.all.input_shapes if len(shape) in (2,) and shape[0] != 1][:2]
        + [shape for shape in TestCollectionCommon.all.input_shapes if len(shape) in (3,) and shape[0] == 1][:2]
        + [shape for shape in TestCollectionCommon.all.input_shapes if len(shape) in (3,) and shape[0] != 1][:2]
        + [shape for shape in TestCollectionCommon.all.input_shapes if len(shape) in (4,) and shape[0] == 1][:2]
        + [shape for shape in TestCollectionCommon.all.input_shapes if len(shape) in (4,) and shape[0] != 1][:2],
        dev_data_formats=[
            None,
            forge.DataFormat.Float16_b,
            forge.DataFormat.Int8,
        ],
    )


class VectorLambdas:

    ALL_OPERATORS = lambda test_vector: test_vector in TestCollectionData.all
    NONE = lambda test_vector: False
    QUICK = lambda test_vector: test_vector in TestCollectionData.quick
    SINGLE_SHAPE = lambda test_vector: test_vector.input_shape in TestCollectionCommon.single.input_shapes

    FAILING = (
        lambda test_vector: test_vector.failing_result is not None
        and test_vector.failing_result.failing_reason is not None
    )
    SKIPED = (
        lambda test_vector: test_vector.failing_result is not None
        and test_vector.failing_result.skip_reason is not None
    )
    SKIPED_FATAL = (
        lambda test_vector: test_vector.failing_result is not None
        and test_vector.failing_result.skip_reason == FailingReasons.FATAL_ERROR
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
    test_suite.query_all().filter(
        VectorLambdas.ALL_OPERATORS,
        VectorLambdas.QUICK,
        # VectorLambdas.SINGLE_SHAPE,
        # VectorLambdas.HAS_DATA_FORMAT,
        # VectorLambdas.NO_DATA_FORMAT,
    )
    # .filter(lambda test_vector: test_vector in TestCollection(
    #     operators=["add", ],
    #     input_shapes=[
    #         (1, 1)
    #     ],
    #     failing_reason=FailingReasons.DATA_MISMATCH,
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
def test_custom(test_vector: TestVector, test_device):
    TestVerification.verify(test_vector, test_device)


@pytest.mark.parametrize("test_vector", test_suite.query_all().filter(VectorLambdas.ALL_OPERATORS).to_params())
def test_plan(test_vector: TestVector, test_device):
    TestVerification.verify(test_vector, test_device)


# 1480 passed, 20 xfailed, 2 warnings in 529.46s (0:08:49)
# 4 failed, 1352 passed, 212 xfailed, 115 xpassed, 2 warnings in 590.56s (0:09:50)
# 1 failed, 4041 passed, 20 skipped, 321 xfailed, 2 warnings in 1510.10s (0:25:10)
# 3894 passed, 108 skipped, 444 xfailed, 252 xpassed, 2 warnings in 1719.04s (0:28:39)
# 3834 passed, 60 skipped, 372 xfailed, 252 xpassed, 2 warnings in 1511.94s (0:25:11)
# 10 failed, 3442 passed, 59 skipped, 1030 xfailed, 1 xpassed, 2 warnings in 1787.61s (0:29:47)
# 12 failed, 3443 passed, 59 skipped, 1028 xfailed, 2 warnings in 1716.62s (0:28:36
# 10 failed, 3443 passed, 59 skipped, 1027 xfailed, 2 warnings in 1819.59s (0:30:19)
# 5 failed, 3443 passed, 59 skipped, 1032 xfailed, 2 warnings in 1715.26s (0:28:35)
# 3443 passed, 59 skipped, 1037 xfailed, 2 warnings in 1726.30s (0:28:46)
# 8 failed, 3432 passed, 59 skipped, 1028 xfailed, 8 xpassed in 1591.84s (0:26:31)
# 3440 passed, 59 skipped, 1035 xfailed in 1587.97s (0:26:27)
# 3500 passed, 1056 xfailed in 1668.66s (0:27:48)


@pytest.mark.parametrize(
    "test_vector",
    test_suite.query_all()
    .filter(VectorLambdas.ALL_OPERATORS, VectorLambdas.FAILING, VectorLambdas.NOT_SKIPED)
    .to_params(),
)
def test_failed(test_vector: TestVector, test_device):
    TestVerification.verify(test_vector, test_device)


@pytest.mark.parametrize(
    "test_vector",
    test_suite.query_all().filter(VectorLambdas.ALL_OPERATORS, VectorLambdas.SKIPED).to_params(),
)
def test_skipped(test_vector: TestVector, test_device):
    TestVerification.verify(test_vector, test_device)


@pytest.mark.parametrize(
    "test_vector",
    test_suite.query_all()
    .filter(VectorLambdas.ALL_OPERATORS, VectorLambdas.FAILING, VectorLambdas.SKIPED_FATAL)
    .to_params(),
)
def test_fatal(test_vector: TestVector, test_device):
    TestVerification.verify(test_vector, test_device)


@pytest.mark.parametrize(
    "test_vector",
    test_suite.query_all()
    .filter(VectorLambdas.ALL_OPERATORS, VectorLambdas.FAILING, VectorLambdas.NOT_SKIPED)
    .filter(VectorLambdas.UNSUPPORTED_DATA_FORMAT)
    .to_params(),
)
def test_unsupported_df(test_vector: TestVector, test_device):
    TestVerification.verify(test_vector, test_device)


@pytest.mark.parametrize(
    "test_vector",
    test_suite.query_all()
    .filter(VectorLambdas.ALL_OPERATORS, VectorLambdas.FAILING, VectorLambdas.NOT_SKIPED)
    .filter(VectorLambdas.NOT_IMPLEMENTED)
    .to_params(),
)
def test_not_implemented(test_vector: TestVector, test_device):
    TestVerification.verify(test_vector, test_device)


@pytest.mark.parametrize(
    "test_vector",
    test_suite.query_all()
    .filter(VectorLambdas.ALL_OPERATORS, VectorLambdas.FAILING, VectorLambdas.NOT_SKIPED)
    .filter(VectorLambdas.DATA_MISMATCH)
    .to_params(),
)
def test_data_mismatch(test_vector: TestVector, test_device):
    TestVerification.verify(test_vector, test_device)


@pytest.mark.parametrize("test_vector", test_suite.query_from_id_list(TestParamsData.get_single_list()).to_params())
def test_single(test_vector: TestVector, test_device):
    TestVerification.verify(test_vector, test_device)
