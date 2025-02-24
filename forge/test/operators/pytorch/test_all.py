# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Tests for testing all pytorch operators


# Examples
# pytest -svv forge/test/operators/pytorch/test_all.py::test_plan
# pytest -svv forge/test/operators/pytorch/test_all.py::test_custom
# pytest -svv forge/test/operators/pytorch/test_all.py::test_query
# pytest -svv forge/test/operators/pytorch/test_all.py::test_unique
# pytest -svv forge/test/operators/pytorch/test_all.py::test_single


import os
import pytest
import forge
import textwrap

from loguru import logger
from tabulate import tabulate
from typing import List

from test.operators.utils import DeviceUtils
from test.operators.utils import InputSource
from test.operators.utils import TestVector
from test.operators.utils import TestCollection
from test.operators.utils import TestCollectionCommon
from test.operators.utils import TestSuite
from test.operators.utils import TestPlanScanner
from test.operators.utils import TestPlanUtils
from test.operators.utils import FailingReasons


class TestVerification:
    """Helper class for performing test verification. It allows running tests in dry-run mode."""

    DRY_RUN = False
    # DRY_RUN = True

    @classmethod
    def verify(cls, test_vector: TestVector, test_device):
        if cls.DRY_RUN:
            # pytest.skip("Dry run")
            return
        test_vector.verify(test_device)


class TestParamsData:
    """
    Helper class for providing data for test parameters.
    This is a parameter manager that collects quering criterias from environment variables to determine the tests that should run. It can helper filtering test collection and filtering lambdas.
    """

    __test__ = False  # Avoid collecting TestParamsData as a pytest test

    @classmethod
    def get_single_list(cls) -> list[str]:
        """Provide a list of test ids to run for test_single method"""
        test_id_single = os.getenv("TEST_ID", None)
        return [test_id_single] if test_id_single else []

    @classmethod
    def get_ids_from_file(cls) -> list[str]:
        """Provide a list of test ids from a file to run for test_ids method"""
        id_file = os.getenv("ID_FILE", None)
        if id_file is not None:
            test_ids = TestPlanUtils.load_test_ids_from_file(id_file)
        else:
            test_ids = []
        return test_ids

    @classmethod
    def build_filtered_collection(cls) -> TestCollection:
        """
        Builds a filtering test collection based on environment variables
        Query criterias are defined by the following environment variables:
        - OPERATORS: List of operators to filter
        - INPUT_SOURCES: List of input sources to filter
        - INPUT_SHAPES: List of input shapes to filter
        - DEV_DATA_FORMATS: List of data formats to filter
        - MATH_FIDELITIES: List of math fidelities to filter
        - KWARGS: List of kwargs dictionaries to filter
        """
        operators = os.getenv("OPERATORS", None)
        if operators:
            operators = operators.split(",")
        else:
            operators = None

        input_sources = os.getenv("INPUT_SOURCES", None)
        if input_sources:
            input_sources = input_sources.split(",")
            input_sources = [getattr(InputSource, input_source) for input_source in input_sources]

        input_shapes = os.getenv("INPUT_SHAPES", None)
        if input_shapes:
            input_shapes = eval(input_shapes)

        dev_data_formats = os.getenv("DEV_DATA_FORMATS", None)
        if dev_data_formats:
            dev_data_formats = dev_data_formats.split(",")
            dev_data_formats = [getattr(forge.DataFormat, dev_data_format) for dev_data_format in dev_data_formats]

        math_fidelities = os.getenv("MATH_FIDELITIES", None)
        if math_fidelities:
            math_fidelities = math_fidelities.split(",")
            math_fidelities = [getattr(forge.MathFidelity, math_fidelity) for math_fidelity in math_fidelities]

        kwargs = os.getenv("KWARGS", None)
        if kwargs:
            kwargs = eval(kwargs)

        filtered_collection = TestCollection(
            operators=operators,
            input_sources=input_sources,
            input_shapes=input_shapes,
            dev_data_formats=dev_data_formats,
            math_fidelities=math_fidelities,
            kwargs=kwargs,
        )

        return filtered_collection

    def build_filter_lambdas():
        """
        Builds a list of lambdas for filtering test vectors based on environment variables.
        The lambdas are built based on the following environment variables:
        - FILTERS: List of lambdas defined in VectorLambdas to filter
        - FAILING_REASONS: List of failing reasons to filter
        - SKIP_REASONS: List of skip reasons to filter
        """
        lambdas = []

        # Include selected filters from VectorLambdas
        filters = os.getenv("FILTERS", None)
        if filters:
            filters = filters.split(",")
            filters = [getattr(VectorLambdas, filter) for filter in filters]
            lambdas = lambdas + filters

        # TODO: Extend TestCollection with list of failing reasons and skip reasons and move this logic to build_filtered_collection
        failing_reasons = os.getenv("FAILING_REASONS", None)
        if failing_reasons:
            failing_reasons = failing_reasons.split(",")
            failing_reasons = [getattr(FailingReasons, failing_reason) for failing_reason in failing_reasons]

        skip_reasons = os.getenv("SKIP_REASONS", None)
        if skip_reasons:
            skip_reasons = skip_reasons.split(",")
            skip_reasons = [getattr(FailingReasons, skip_reason) for skip_reason in skip_reasons]

        if failing_reasons:
            lambdas.append(
                lambda test_vector: test_vector.failing_result is not None
                and test_vector.failing_result.failing_reason in failing_reasons
            )

        if skip_reasons:
            lambdas.append(
                lambda test_vector: test_vector.failing_result is not None
                and test_vector.failing_result.skip_reason in skip_reasons
            )

        return lambdas

    @classmethod
    def get_random_seed(cls) -> int:
        """Provide a random seed based on environment variables"""
        random_seed = os.getenv("RANDOM_SEED", None)
        return int(random_seed) if random_seed else 0

    @classmethod
    def get_filter_sample(cls) -> float:
        """Provide a sample of test vectors to run based on environment variables"""

        sample = os.getenv("SAMPLE", None)

        return float(sample) if sample else 100

    @classmethod
    def get_filter_range(cls) -> tuple[int, int]:
        """Provide a range of test vectors to run based on environment variables"""

        range = os.getenv("RANGE", None)
        if range:
            range = range.split(",")
            if len(range) == 1:
                return 0, int(range[0])
            else:
                return int(range[0]), int(range[1])

        return 0, 100000

    @classmethod
    def filter_suite_by_operators(cls, test_suite: TestSuite, operators: List[str]) -> TestSuite:
        """Filter test plans based on operator list to speed up test filtering"""
        if operators is None:
            return test_suite
        else:
            test_plans = [
                test_plan
                for test_plan in test_suite.test_plans
                if len(list(set(test_plan.collections[0].operators) & set(operators))) > 0
            ]
            return TestSuite(test_plans)


class TestCollectionData:
    """Helper test collections"""

    __test__ = False  # Avoid collecting TestCollectionData as a pytest test

    # Test collections for query criterias from environment variables
    filtered = TestParamsData.build_filtered_collection()

    # All available test vectors
    all = TestCollection(
        # operators=None,  # All available operators
        operators=filtered.operators,  # Operators selected by filter
    )

    # Quick test collection for faster testing consisting of a subset of input shapes and data formats
    quick = TestCollection(
        # 2 examples for each dimension and microbatch size
        input_shapes=[]
        + [shape for shape in TestCollectionCommon.all.input_shapes if len(shape) in (2,) and shape[0] == 1][:2]
        + [shape for shape in TestCollectionCommon.all.input_shapes if len(shape) in (2,) and shape[0] != 1][:2]
        + [shape for shape in TestCollectionCommon.all.input_shapes if len(shape) in (3,) and shape[0] == 1][:2]
        + [shape for shape in TestCollectionCommon.all.input_shapes if len(shape) in (3,) and shape[0] != 1][:2]
        + [shape for shape in TestCollectionCommon.all.input_shapes if len(shape) in (4,) and shape[0] == 1][:2]
        + [shape for shape in TestCollectionCommon.all.input_shapes if len(shape) in (4,) and shape[0] != 1][:2],
        # one example for float and int data formats
        dev_data_formats=[
            None,
            forge.DataFormat.Float16_b,
            forge.DataFormat.Int8,
        ],
    )


class TestSuiteData:

    __test__ = False  # Avoid collecting TestSuiteData as a pytest test

    all = TestPlanScanner.build_test_suite(scan_file=__file__, scan_package=__package__)

    filtered = TestParamsData.filter_suite_by_operators(all, TestCollectionData.all.operators)


class VectorLambdas:
    """Helper lambdas for filtering test vectors"""

    ALL_OPERATORS = lambda test_vector: test_vector in TestCollectionData.all
    NONE = lambda test_vector: False

    QUICK = lambda test_vector: test_vector in TestCollectionData.quick
    FILTERED = lambda test_vector: test_vector in TestCollectionData.filtered

    SINGLE_SHAPE = lambda test_vector: test_vector.input_shape in TestCollectionCommon.single.input_shapes

    SHAPES_2D = lambda test_vector: len(test_vector.input_shape) == 2
    SHAPES_3D = lambda test_vector: len(test_vector.input_shape) == 3
    SHAPES_4D = lambda test_vector: len(test_vector.input_shape) == 4

    MICROBATCH_SIZE_ONE = lambda test_vector: test_vector.input_shape[0] == 1
    MICROBATCH_SIZE_MULTI = lambda test_vector: test_vector.input_shape[0] > 1

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


@pytest.mark.parametrize(
    "test_vector",
    TestSuiteData.filtered.query_all().filter(
        VectorLambdas.ALL_OPERATORS,
        VectorLambdas.QUICK,
        # VectorLambdas.FILTERED,
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


@pytest.mark.parametrize(
    "test_vector",
    TestSuiteData.filtered.query_all()
    .filter(VectorLambdas.FILTERED)
    .filter(*TestParamsData.build_filter_lambdas())
    .sample(TestParamsData.get_filter_sample(), TestParamsData.get_random_seed())
    .range(*TestParamsData.get_filter_range())
    .to_params(),
)
def test_query(test_vector: TestVector, test_device):
    TestVerification.verify(test_vector, test_device)


@pytest.mark.parametrize(
    "test_vector",
    TestSuiteData.filtered.query_all()
    .filter(VectorLambdas.ALL_OPERATORS, VectorLambdas.SINGLE_SHAPE)
    .filter(
        lambda test_vector: test_vector.input_source in [InputSource.FROM_HOST]
        if (TestCollectionData.all.operators is None or len(TestCollectionData.all.operators) > 5)
        else True
    )
    .group_limit(["operator", "input_source", "kwargs"], 1)
    .to_params(),
)
def test_unique(test_vector: TestVector, test_device):
    TestVerification.verify(test_vector, test_device)


@pytest.mark.parametrize(
    "test_vector", TestSuiteData.all.query_from_id_list(TestParamsData.get_single_list()).to_params()
)
def test_single(test_vector: TestVector, test_device):
    TestVerification.verify(test_vector, test_device)


@pytest.mark.parametrize(
    "test_vector", TestSuiteData.all.query_from_id_list(TestParamsData.get_ids_from_file()).to_params()
)
def test_ids(test_vector: TestVector, test_device):
    test_vector.verify(test_device)


@pytest.mark.nightly_sweeps
@pytest.mark.parametrize(
    "test_vector", TestSuiteData.filtered.query_all().filter(VectorLambdas.ALL_OPERATORS).to_params()
)
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
# 4401 passed, 1423 xfailed in 2185.56s (0:36:25)
# 4395 passed, 1429 xfailed in 2577.15s (0:42:57)


# Below are examples of custom test functions that utilize filtering lambdas to run specific tests


class InfoUtils:
    @classmethod
    def print_query_params(cls, max_width=80):
        print("Query parameters:")
        cls.print_query_values(max_width)
        print("Query examples:")
        cls.print_query_examples(max_width)

    @classmethod
    def print_query_values(cls, max_width=80):

        operators = [key for key in TestSuiteData.all.indices]
        operators = sorted(operators)
        operators = ", ".join(operators)

        filters = [key for key, value in VectorLambdas.__dict__.items() if not key.startswith("__")]
        filters = [filter for filter in filters if filter not in ["ALL_OPERATORS", "NONE", "FILTERED"]]
        filters = ", ".join(filters)

        input_sources = [f"{input_source.name}" for input_source in TestCollectionCommon.all.input_sources]
        input_sources = ", ".join(input_sources)

        input_shapes = [f"{input_shape}" for input_shape in TestCollectionCommon.all.input_shapes]
        input_shapes = ", ".join(input_shapes)

        dev_data_formats = [f"{dev_data_format.name}" for dev_data_format in TestCollectionCommon.all.dev_data_formats]
        dev_data_formats = ", ".join(dev_data_formats)

        math_fidelities = [f"{math_fidelity.name}" for math_fidelity in TestCollectionCommon.all.math_fidelities]
        math_fidelities = ", ".join(math_fidelities)

        failing_reasons = [
            key for key, value in FailingReasons.__dict__.items() if not callable(value) and not key.startswith("__")
        ]
        failing_reasons = ", ".join(failing_reasons)

        parameters = [
            {"name": "OPERATORS", "description": f"List of operators. Supported values: {operators}"},
            {"name": "FILTERS", "description": f"List of lambda filters. Supported values: {filters}"},
            {"name": "INPUT_SOURCES", "description": f"List of input sources. Supported values: {input_sources}"},
            {"name": "INPUT_SHAPES", "description": f"List of input shapes. Supported values: {input_shapes}"},
            {
                "name": "DEV_DATA_FORMATS",
                "description": f"List of dev data formats. Supported values: {dev_data_formats}",
            },
            {"name": "MATH_FIDELITIES", "description": f"List of math fidelities. Supported values: {math_fidelities}"},
            {
                "name": "KWARGS",
                "description": "List of kwargs dictionaries. Kwarg is a mandatory or optional attribute of an operator. See operator documentation for each operator or use `test_unique` to find examples.",
            },
            {"name": "FAILING_REASONS", "description": f"List of failing reasons. Supported values: {failing_reasons}"},
            {"name": "SKIP_REASONS", "description": "Same as FAILING_REASONS"},
            {"name": "RANDOM_SEED", "description": "Seed for random number generator"},
            {"name": "SAMPLE", "description": "Percentage of results to sample"},
            {"name": "RANGE", "description": "Limit number of results"},
            {"name": "TEST_ID", "description": "Id of a test containing test parameters"},
            {"name": "ID_FILE", "description": "Path to a file containing test ids"},
        ]

        cls.print_formatted_parameters(parameters, max_width, headers=["Parameter", "Supported values"])

    @classmethod
    def print_query_examples(cls, max_width=80):

        parameters = [
            {"name": "OPERATORS", "description": "export OPERATORS=add"},
            {"name": "OPERATORS", "description": "export OPERATORS=add,div"},
            {"name": "FILTERS", "description": "export FILTERS=HAS_DATA_FORMAT,QUICK"},
            {"name": "INPUT_SOURCES", "description": "export INPUT_SOURCES=FROM_HOST,FROM_DRAM_QUEUE"},
            {"name": "INPUT_SHAPES", "description": 'export INPUT_SHAPES="[(3, 4), (45, 17)]"'},
            {"name": "DEV_DATA_FORMATS", "description": "export DEV_DATA_FORMATS=Float16_b,Int8"},
            {"name": "MATH_FIDELITIES", "description": "export MATH_FIDELITIES=HiFi4,HiFi3"},
            {
                "name": "KWARGS",
                "description": "export KWARGS=\"[{'rounding_mode': 'trunc'},{'rounding_mode': 'floor'}]\"",
            },
            {"name": "FAILING_REASONS", "description": "export FAILING_REASONS=DATA_MISMATCH,UNSUPPORTED_DATA_FORMAT"},
            {"name": "FAILING_REASONS", "description": "export FAILING_REASONS=NOT_IMPLEMENTED"},
            {"name": "SKIP_REASONS", "description": "export SKIP_REASONS=FATAL_ERROR"},
            {"name": "RANGE", "description": "export RANGE=5"},
            {"name": "RANDOM_SEED", "description": "export RANDOM_SEED=42"},
            {"name": "SAMPLE", "description": "export SAMPLE=20"},
            {"name": "RANGE", "description": "export RANGE=10,20"},
            {"name": "TEST_ID", "description": "export TEST_ID='ge-FROM_HOST-None-(1, 2, 3, 4)-Float16_b-HiFi4'"},
            {"name": "ID_FILE", "description": "export ID_FILE='/path/to/test_ids.log'"},
        ]

        cls.print_formatted_parameters(parameters, max_width, headers=["Parameter", "Examples"])

    @classmethod
    def print_formatted_parameters(cls, parameters, max_width=80, headers=["Parameter", "Description"]):
        for param in parameters:
            param["description"] = "\n".join(textwrap.wrap(param["description"], width=max_width))

        table_data = [[param["name"], param["description"]] for param in parameters]

        print(tabulate(table_data, headers, tablefmt="grid"))
