# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Tests for testing all pytorch operators


# Examples
# pytest -svv forge/test/operators/pytorch/test_all.py::test_query


import os
import pytest
import forge
import textwrap

from loguru import logger
from tabulate import tabulate
from typing import List, Optional

from test.operators.utils import DeviceUtils
from test.operators.utils import InputSource
from test.operators.utils import TestVector
from test.operators.utils import TestCollection
from test.operators.utils import TestCollectionCommon
from test.operators.utils import TestQuery
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
    def get_test_ids_filenames(cls) -> list[str]:
        """Provide a list of files with test ids to test"""
        id_files = os.getenv("ID_FILES", None)
        if id_files is not None:
            id_files = id_files.split(",")
            id_files = [id_file for id_file in id_files if id_file]
        else:
            id_files = []
        return id_files

    @classmethod
    def get_ignore_test_ids_filenames(cls, id_files_ignore_default: Optional[List[str]] = None) -> list[str]:
        """Provide a list of files with test ids to ignore"""
        id_files_ignore = os.getenv("ID_FILES_IGNORE", None)
        if id_files_ignore is not None:
            id_files_ignore = id_files_ignore.split(",")
            id_files_ignore = [id_file for id_file in id_files_ignore if id_file]
        else:
            id_files_ignore = id_files_ignore_default
        return id_files_ignore

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
            dev_data_formats = [
                TestPlanUtils.dev_data_format_from_str(dev_data_format) for dev_data_format in dev_data_formats
            ]

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
                if len(list(set(test_plan.operators) & set(operators))) > 0
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


class TestIdsData:

    __test__ = False  # Avoid collecting TestIdsData as a pytest test

    _base_dir = os.path.dirname(os.path.abspath(__file__))

    _default_ignore_files = ["ids/blacklist.txt"]
    _ignore_files = TestParamsData.get_ignore_test_ids_filenames(_default_ignore_files)

    test_ids_list = (
        TestPlanUtils.load_test_ids_from_files(_base_dir, TestParamsData.get_test_ids_filenames())
        if os.getenv("ID_FILES", None)
        else None
    )

    single_ids_list = TestParamsData.get_single_list() if os.getenv("TEST_ID", None) else None

    ignore_list = list(TestPlanUtils.load_test_ids_from_files(_base_dir, _ignore_files))


class TestSuiteData:

    __test__ = False  # Avoid collecting TestSuiteData as a pytest test

    all = TestPlanScanner.build_test_suite(scan_file=__file__, scan_package=__package__)

    filtered = TestParamsData.filter_suite_by_operators(all, TestCollectionData.all.operators)


class VectorLambdas:
    """Helper lambdas for filtering test vectors"""

    ALL_OPERATORS = lambda test_vector: test_vector in TestCollectionData.all
    ALL = lambda test_vector: True
    NONE = lambda test_vector: False

    QUICK = lambda test_vector: test_vector in TestCollectionData.quick
    FILTERED = lambda test_vector: test_vector in TestCollectionData.filtered

    SKIP_IGNORE_LIST = (
        lambda test_vector: TestIdsData.ignore_list is None or test_vector.get_id() not in TestIdsData.ignore_list
    )

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


class TestQueries:

    __test__ = False  # Avoid collecting TestQueries as a pytest test

    @classmethod
    def _filter_tests_ids_by_operators(cls, test_ids: List[str]) -> List[str]:
        """Keep only test ids that contain any of the operators from filtered collection"""
        operators = TestCollectionData.filtered.operators
        if operators is None:
            return test_ids
        operators = [f"{operator}-" for operator in operators]
        test_ids = [test_id for test_id in test_ids if any([operator in test_id for operator in operators])]
        return test_ids

    @classmethod
    def query_source(cls) -> TestQuery:
        if os.getenv("TEST_ID", None) and os.getenv("ID_FILES", None):
            raise ValueError("TEST_ID and ID_FILES cannot be used together")

        test_suite = TestSuiteData.filtered

        if TestIdsData.test_ids_list:
            logger.info("Using test ids from file")
            test_ids = cls._filter_tests_ids_by_operators(TestIdsData.test_ids_list)
            query = test_suite.query_from_id_list(test_ids)
        elif TestIdsData.single_ids_list:
            logger.info("Using single test id")
            test_ids = cls._filter_tests_ids_by_operators(TestIdsData.single_ids_list)
            query = test_suite.query_from_id_list(test_ids)
        else:
            logger.info("Using all test vectors")
            query = test_suite.query_all()

        return query

    @classmethod
    def query_filter(cls, query: TestQuery) -> TestQuery:

        query = (
            query.filter(VectorLambdas.FILTERED)
            .filter(*TestParamsData.build_filter_lambdas())
            .sample(TestParamsData.get_filter_sample(), TestParamsData.get_random_seed())
            # if TEST_ID is set, ignore the ignore list
            .filter(VectorLambdas.SKIP_IGNORE_LIST if not os.getenv("TEST_ID", None) else VectorLambdas.ALL)
            .range(*TestParamsData.get_filter_range())
            # .log()
        )

        if os.getenv("UNIQUE_KWARGS", "false").lower() == "true":
            query = query.group_limit(["operator", "kwargs"], 1)

        return query


@pytest.mark.nightly_sweeps
@pytest.mark.parametrize("test_vector", TestQueries.query_filter(TestQueries.query_source()).to_params())
def test_query(test_vector: TestVector, test_device):
    TestVerification.verify(test_vector, test_device)


class InfoUtils:
    @classmethod
    def print_query_params(cls, max_width=80):
        print("Query parameters:")
        cls.print_query_values(max_width)
        print("Query examples:")
        cls.print_query_examples(max_width)
        print("Configuration parameters:")
        cls.print_configuration_params(max_width)
        print("Configuration examples:")
        cls.print_configuration_examples(max_width)

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
                "description": "List of kwargs dictionaries. Kwarg is a mandatory or optional attribute of an operator. See operator documentation for each operator or use parameter `UNIQUE_KWARGS` to find examples.",
            },
            {"name": "FAILING_REASONS", "description": f"List of failing reasons. Supported values: {failing_reasons}"},
            {"name": "SKIP_REASONS", "description": "Same as FAILING_REASONS"},
            {"name": "RANDOM_SEED", "description": "Seed for random number generator"},
            {"name": "SAMPLE", "description": "Percentage of results to sample"},
            {"name": "UNIQUE_KWARGS", "description": "Only representative tests with unique kwargs values"},
            {"name": "RANGE", "description": "Limit number of results"},
            {"name": "TEST_ID", "description": "Id of a test containing test parameters"},
            {"name": "ID_FILES", "description": "Paths to files containing test ids instead of tests from test plan"},
            {"name": "ID_FILES_IGNORE", "description": "Paths to files containing test ids to be ignored"},
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
            {"name": "UNIQUE_KWARGS", "description": "export UNIQUE_KWARGS=true"},
            {"name": "RANGE", "description": "export RANGE=10,20"},
            {"name": "TEST_ID", "description": "export TEST_ID='ge-FROM_HOST-None-(1, 2, 3, 4)-Float16_b-HiFi4'"},
            {"name": "ID_FILES", "description": "export ID_FILES='/path/to/test_ids.log'"},
            {"name": "ID_FILES_IGNORE", "description": "export ID_FILES_IGNORE='ids/blacklist.txt,ids/ignore.txt'"},
        ]

        cls.print_formatted_parameters(parameters, max_width, headers=["Parameter", "Examples"])

    @classmethod
    def print_configuration_params(cls, max_width=80):

        parameters = [
            {
                "name": "SKIP_FORGE_VERIFICATION",
                "description": f"Skip Forge model verification including model compiling and inference",
                "default": "false",
            },
        ]

        cls.print_formatted_parameters(parameters, max_width, headers=["Parameter", "Description", "Default"])

    @classmethod
    def print_configuration_examples(cls, max_width=80):

        parameters = [
            {"name": "SKIP_FORGE_VERIFICATION", "description": "export SKIP_FORGE_VERIFICATION=true"},
        ]

        cls.print_formatted_parameters(parameters, max_width, headers=["Parameter", "Examples"])

    @classmethod
    def print_formatted_parameters(cls, parameters, max_width=80, headers=["Parameter", "Description"]):
        for param in parameters:
            param["description"] = "\n".join(textwrap.wrap(param["description"], width=max_width))

        table_data = [[param["name"], param["description"]] for param in parameters]

        print(tabulate(table_data, headers, tablefmt="grid"))
