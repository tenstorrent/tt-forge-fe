# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Test plan management utilities

import types
import pytest
import forge
import re

import os
import importlib
import inspect
from types import ModuleType
from itertools import chain

from _pytest.mark import Mark
from _pytest.mark import ParameterSet

from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
from typing import Callable, Generator, Optional, List, Dict, Union, Tuple, TypeAlias

from forge import MathFidelity, DataFormat
from forge.op_repo import TensorShape

from .datatypes import OperatorParameterTypes
from .pytest import PytestParamsUtils
from .compat import TestDevice


class InputSource(Enum):
    FROM_ANOTHER_OP = 1
    FROM_HOST = 2
    FROM_DRAM_QUEUE = 3
    CONST_EVAL_PASS = 4


@dataclass
class TestResultFailing:
    """
    Dataclass for defining failing test result

    Args:
        failing_reason: Failing reason
        skip_reason: Skip reason
    """

    __test__ = False  # Avoid collecting TestResultFailing as a pytest test

    failing_reason: Optional[str] = None
    skip_reason: Optional[str] = None

    def get_marks(self) -> List[Mark]:
        marks = []
        if self.failing_reason is not None:
            marks.append(pytest.mark.xfail(reason=self.failing_reason))
        if self.skip_reason is not None:
            marks.append(pytest.mark.skip(reason=self.skip_reason))
        return marks


@dataclass
class TestVector:
    """
    Dataclass for defining single test vector

    Args:
        operator: Operator name
        input_source: Input source
        input_shape: Input shape
        dev_data_format: Data format
        math_fidelity: Math fidelity
        kwargs: Operator parameters
        pcc: PCC value
        failing_result: Failing result
    """

    __test__ = False  # Avoid collecting TestVector as a pytest test

    operator: Optional[str]
    input_source: InputSource
    input_shape: TensorShape  # TODO - Support multiple input shapes
    number_of_operands: Optional[int] = None
    dev_data_format: Optional[DataFormat] = None
    math_fidelity: Optional[MathFidelity] = None
    kwargs: Optional[OperatorParameterTypes.Kwargs] = None
    pcc: Optional[float] = None
    failing_result: Optional[TestResultFailing] = None
    test_plan: Optional["TestPlan"] = None  # Needed for verification

    def get_id(self, fields: Optional[List[str]] = None) -> str:
        """Get test vector id"""
        if fields is None:
            return f"{self.operator}-{self.input_source.name}-{self.kwargs}-{self.input_shape}{'-' + str(self.number_of_operands) + '-' if self.number_of_operands else '-'}{self.dev_data_format.name if self.dev_data_format else None}-{self.math_fidelity.name if self.math_fidelity else None}"
        else:
            attr = [
                (getattr(self, field).name if getattr(self, field) is not None else None)
                if field in ("input_source", "dev_data_format", "math_fidelity")
                else getattr(self, field)
                for field in fields
            ]
            attr = [str(a) for a in attr]
            return "-".join(attr)

    def get_marks(self) -> List[Mark]:
        """Get marks for the test vector"""
        marks = self.failing_result.get_marks() if self.failing_result is not None else []
        return marks

    def to_param(self) -> ParameterSet:
        """Convert test vector to pytest parameter set"""
        return pytest.param(self, marks=self.get_marks(), id=self.get_id())

    def verify(self, test_device: "TestDevice"):
        """Verify the test vector"""
        self.test_plan.verify(test_device=test_device, test_vector=self)


@dataclass
class TestCollection:
    """
    The test collection defines rules for generating test vectors.

    Args:
        operators: List of operators
        input_sources: List of input sources
        input_shapes: List of input shapes
        dev_data_formats: List of data formats
        math_fidelities: List of math fidelities
        kwargs: List of operator parameters
        pcc: PCC value
        failing_reason: Failing reason
        skip_reason: Skip reason
    """

    __test__ = False  # Avoid collecting TestCollection as a pytest test

    operators: Optional[List[str]] = None
    input_sources: Optional[List[InputSource]] = None
    input_shapes: Optional[List[TensorShape]] = None  # TODO - Support multiple input shapes
    numbers_of_operands: Optional[List[int]] = None
    dev_data_formats: Optional[List[DataFormat]] = None
    math_fidelities: Optional[List[MathFidelity]] = None
    kwargs: Optional[
        Union[List[OperatorParameterTypes.Kwargs], Callable[["TestVector"], List[OperatorParameterTypes.Kwargs]]]
    ] = None
    pcc: Optional[float] = None
    criteria: Optional[Callable[["TestVector"], bool]] = None

    failing_reason: Optional[str] = None
    skip_reason: Optional[str] = None

    def __post_init__(self):
        if self.operators is not None:
            self.operators = PytestParamsUtils.strip_param_sets(self.operators)
        if self.input_sources is not None:
            self.input_sources = PytestParamsUtils.strip_param_sets(self.input_sources)
        if self.input_shapes is not None:
            self.input_shapes = PytestParamsUtils.strip_param_sets(self.input_shapes)
        if self.dev_data_formats is not None:
            self.dev_data_formats = PytestParamsUtils.strip_param_sets(self.dev_data_formats)
        if self.math_fidelities is not None:
            self.math_fidelities = PytestParamsUtils.strip_param_sets(self.math_fidelities)
        if self.kwargs is not None and not isinstance(self.kwargs, types.FunctionType):
            self.kwargs = PytestParamsUtils.strip_param_sets(self.kwargs)

    def __contains__(self, item):
        if isinstance(item, TestVector):
            return TestPlanUtils.test_vector_in_collection(item, self)
        raise ValueError(f"Unsupported type: {type(item)} while checking if object is in TestCollection")


@dataclass
class TestQuery:
    """
    Dataclass for specifying test vectors queries

    Args:
        test_vectors: Test vectors
    """

    test_vectors: Generator[TestVector, None, None]

    def _filter_allowed(self, *filters: Callable[[TestVector], bool]) -> Generator[TestVector, None, None]:
        for test_vector in self.test_vectors:
            if all([filter(test_vector) for filter in filters]):
                yield test_vector

    def _filter_skiped(self, *filters: Callable[[TestVector], bool]) -> Generator[TestVector, None, None]:
        for test_vector in self.test_vectors:
            if any([not filter(test_vector) for filter in filters]):
                yield test_vector

    def _filter_indices(
        self, indices: Union[int, Tuple[int, int], List[int]] = None, allow_or_skip=True
    ) -> Generator[TestVector, None, None]:
        index = 0
        for test_vector in self.test_vectors:
            found = False
            if isinstance(indices, tuple):
                # logger.info(f"Tuple type indices: {indices}")
                range_min, range_max = indices
                if range_min <= index < range_max:
                    found = True
            elif isinstance(indices, list):
                # logger.info(f"List type indices: {indices}")
                if index in indices:
                    found = True
            else:
                logger.error(f"Invalid indices: {indices}")

            index += 1
            if allow_or_skip == found:
                yield test_vector

    def _filter_group_limit(self, groups: List[str], limit: int) -> Generator[TestVector, None, None]:
        groups_count = {}
        for test_vector in self.test_vectors:
            test_vector_group = test_vector.get_id(fields=groups)
            if test_vector_group not in groups_count:
                groups_count[test_vector_group] = 0
            groups_count[test_vector_group] += 1
            if groups_count[test_vector_group] <= limit:
                yield test_vector

    def _calculate_failing_result(self) -> Generator[TestVector, None, None]:
        for test_vector in self.test_vectors:
            test_vector.failing_result = test_vector.test_plan.check_test_failing(test_vector)
            yield test_vector

    def _reverse(self) -> Generator[TestVector, None, None]:
        test_vectors = list(self.test_vectors)
        test_vectors = test_vectors[::-1]
        for test_vector in test_vectors:
            yield test_vector

    def _log(self) -> Generator[TestVector, None, None]:
        test_vectors = list(self.test_vectors)
        print("\nParameters:")
        for test_vector in test_vectors:
            print(f"{test_vector.get_id()}")
            yield test_vector
        print(f"Count: {len(test_vectors)}\n")

    def filter(self, *filters: Callable[[TestVector], bool]) -> "TestQuery":
        """Filter test vectors based on the filter functions"""
        return TestQuery(self._filter_allowed(*filters))

    def skip(self, filters: Callable[[TestVector], bool]) -> "TestQuery":
        """Skip test vectors based on the filter functions"""
        return TestQuery(self._filter_skiped(*filters))

    def index(self, *args: int) -> "TestQuery":
        """Filter test vectors based on the indices"""
        indices = list(args)
        return TestQuery(self._filter_indices(indices, allow_or_skip=True))

    def range(self, start_index: int, end_index: int) -> "TestQuery":
        """Filter test vectors based on the range of indices"""
        return TestQuery(self._filter_indices((start_index, end_index), allow_or_skip=True))

    def index_skip(self, *args: int) -> "TestQuery":
        """Skip test vectors based on the indices"""
        indices = list(args)
        return TestQuery(self._filter_indices(indices, allow_or_skip=False))

    def group_limit(self, groups: List[str], limit: int) -> "TestQuery":
        """Limit the number of test vectors per group"""
        return TestQuery(self._filter_group_limit(groups, limit))

    def range_skip(self, start_index: int, end_index: int) -> "TestQuery":
        """Skip test vectors based on the range of indices"""
        return TestQuery(self._filter_indices((start_index, end_index), allow_or_skip=False))

    def calculate_failing_result(self) -> "TestQuery":
        """Calculate and set the failing result based on the test plan"""
        return TestQuery(self._calculate_failing_result())

    def reverse(self) -> "TestQuery":
        """Reverse the order of test vectors"""
        return TestQuery(self._reverse())

    def log(self) -> "TestQuery":
        """Log the test vectors"""
        return TestQuery(self._log())

    def to_params(self) -> Generator[ParameterSet, None, None]:
        """Convert test vectors to pytest parameter sets"""
        test_vectors = self.test_vectors
        for test_vector in test_vectors:
            yield test_vector.to_param()
        logger.trace("To params done")

    @classmethod
    def all(cls, test_plan: Union["TestPlan", "TestSuite"]) -> "TestQuery":
        test_vectors = test_plan.generate()
        query = TestQuery(test_vectors)
        return query.calculate_failing_result()

    @classmethod
    def query_from_id_file(cls, test_plan: Union["TestPlan", "TestSuite"], test_ids_file: str) -> "TestQuery":
        test_vectors = test_plan.load_test_vectors_from_id_file(test_ids_file)
        query = TestQuery(test_vectors)
        return query.calculate_failing_result()

    @classmethod
    def query_from_id_list(cls, test_plan: Union["TestPlan", "TestSuite"], test_ids: List[str]) -> "TestQuery":
        test_vectors = test_plan.load_test_vectors_from_id_list(test_ids)
        query = TestQuery(test_vectors)
        return query.calculate_failing_result()


@dataclass
class TestPlan:
    """
    Define test plan for the operator testing. Define failing rules for the tests.

    Args:
        collections: List of test collections
        failing_rules: List of failing rules
    """

    __test__ = False  # Avoid collecting TestPlan as a pytest test

    collections: Optional[List[TestCollection]] = None
    failing_rules: Optional[List[TestCollection]] = None
    verify: Optional[Callable[[TestVector, TestDevice], None]] = None

    def check_test_failing(
        self,
        test_vector: TestVector,
    ) -> Optional[TestResultFailing]:
        """Check if the test is failing based on the test plan

        Args:
            test_vector: Test vector with all the parameters
        """

        failing_result = None

        for failing_rule in self.failing_rules:
            if test_vector in failing_rule:
                if failing_rule.failing_reason is not None or failing_rule.skip_reason is not None:
                    failing_result = TestResultFailing(failing_rule.failing_reason, failing_rule.skip_reason)
                else:
                    # logger.debug(f"Test should pass: {test_vector.get_id()}")
                    failing_result = None

        return failing_result

    def generate(self) -> Generator[TestVector, None, None]:
        """Generate test vectors based on the test plan"""

        for test_collection in self.collections:

            dev_data_formats = test_collection.dev_data_formats
            if dev_data_formats is None:
                dev_data_formats = [None]

            math_fidelities = test_collection.math_fidelities
            if math_fidelities is None:
                math_fidelities = [None]

            kwargs_list = test_collection.kwargs
            if kwargs_list is None:
                kwargs_list = [None]

            numbers_of_operands = test_collection.numbers_of_operands
            if numbers_of_operands is None:
                numbers_of_operands = [None]

            for input_operator in test_collection.operators:
                for input_source in test_collection.input_sources:
                    for input_shape in test_collection.input_shapes:
                        for number_of_operands in numbers_of_operands:
                            for dev_data_format in dev_data_formats:
                                for math_fidelity in math_fidelities:

                                    test_vector_no_kwargs = TestVector(
                                        test_plan=self,  # Set the test plan to support verification
                                        operator=input_operator,
                                        input_source=input_source,
                                        input_shape=input_shape,
                                        number_of_operands=number_of_operands,
                                        dev_data_format=dev_data_format,
                                        math_fidelity=math_fidelity,
                                        pcc=test_collection.pcc,
                                    )

                                    # filter collection based on criteria
                                    if test_collection.criteria is None or test_collection.criteria(
                                        test_vector_no_kwargs
                                    ):

                                        if isinstance(test_collection.kwargs, types.FunctionType):
                                            kwargs_list = test_collection.kwargs(test_vector_no_kwargs)

                                        for kwargs in kwargs_list:
                                            # instantiate a new test vector to avoid mutating the original test_vector_no_kwargs
                                            test_vector = TestVector(
                                                test_plan=self,  # Set the test plan to support verification
                                                operator=input_operator,
                                                input_source=input_source,
                                                input_shape=input_shape,
                                                number_of_operands=number_of_operands,
                                                dev_data_format=dev_data_format,
                                                math_fidelity=math_fidelity,
                                                pcc=test_collection.pcc,
                                                kwargs=kwargs,
                                            )

                                            yield test_vector

    def load_test_vectors_from_id_file(self, test_ids_file: str) -> List[TestVector]:
        test_ids = TestPlanUtils.load_test_ids_from_file(test_ids_file)

        return self.load_test_vectors_from_id_list(test_ids)

    def load_test_vectors_from_id_list(self, test_ids: List[str]) -> List[TestVector]:
        test_vectors = TestPlanUtils.test_ids_to_test_vectors(test_ids)

        for test_vector in test_vectors:
            if test_vector.operator not in self.collections[0].operators:
                raise ValueError(f"Operator {test_vector.operator} not found in test plan")
            test_vector.test_plan = self

        return test_vectors

    def query_all(self) -> TestQuery:
        return TestQuery.all(self)

    def query_from_id_file(self, test_ids_file: str) -> TestQuery:
        return TestQuery.query_from_id_file(self, test_ids_file)

    def query_from_id_list(self, test_ids: List[str]) -> TestQuery:
        return TestQuery.query_from_id_list(self, test_ids)


@dataclass
class TestSuite:

    __test__ = False  # Avoid collecting TestSuite as a pytest test

    test_plans: List[TestPlan] = None
    indices: Optional[Dict[str, TestPlan]] = None  # TODO remove optional

    @staticmethod
    def get_test_plan_index(test_plans: List[TestPlan]) -> Dict[str, TestPlan]:
        indices = {}
        for test_plan in test_plans:
            for operator in test_plan.collections[0].operators:
                if operator not in indices:
                    indices[operator] = test_plan
        return indices

    def __post_init__(self):
        self.indices = self.get_test_plan_index(self.test_plans)
        logger.trace(f"Test suite indices: {self.indices.keys()} test_plans: {len(self.test_plans)}")

    def generate(self) -> Generator[TestVector, None, None]:
        """Generate test vectors based on the test plan"""
        generators = [test_plan.generate() for test_plan in self.test_plans]
        return chain(*generators)

    def load_test_vectors_from_id_file(self, test_ids_file: str) -> List[TestVector]:
        test_ids = TestPlanUtils.load_test_ids_from_file(test_ids_file)

        return self.load_test_vectors_from_id_list(test_ids)

    def load_test_vectors_from_id_list(self, test_ids: List[str]) -> List[TestVector]:
        test_vectors = TestPlanUtils.test_ids_to_test_vectors(test_ids)

        for test_vector in test_vectors:
            if test_vector.operator not in self.indices:
                raise ValueError(f"Operator {test_vector.operator} not found in test suite")
            test_vector.test_plan = self.indices[test_vector.operator]

        return test_vectors

    def query_all(self) -> TestQuery:
        logger.trace("Query all test vectors")
        return TestQuery.all(self)

    def query_from_id_file(self, test_ids_file: str) -> TestQuery:
        return TestQuery.query_from_id_file(self, test_ids_file)

    def query_from_id_list(self, test_ids: List[str]) -> TestQuery:
        return TestQuery.query_from_id_list(self, test_ids)


class TestPlanUtils:
    """
    Utility functions for test vectors
    """

    @classmethod
    def _match(cls, rule_collection: Optional[List], vector_value):
        """
        Check if the vector value is in the rule collection

        Args:
            rule_collection: Collection of acceptable values
            vector_value: Value to check
        """
        return rule_collection is None or vector_value in rule_collection

    @classmethod
    def _match_kwargs_list(
        cls,
        kwargs_def_list: List[OperatorParameterTypes.Kwargs],
        kwargs: OperatorParameterTypes.Kwargs,
    ):

        if kwargs_def_list is None:
            # No rules to check
            return True

        for kwargs_def in kwargs_def_list:
            if cls._match_kwargs_single(kwargs_def, kwargs):
                # One of the rules matched so collection is valid
                return True

        # No rule matched
        return False

    @classmethod
    def _match_kwargs_single(
        cls,
        kwargs_def: OperatorParameterTypes.Kwargs,
        kwargs: OperatorParameterTypes.Kwargs,
    ):
        for (kwarg_name, kwarg_def_value) in kwargs_def.items():
            if kwarg_name not in kwargs:
                # logger.warning(f"Rule defined for kwarg {kwarg_name} not present in test vector")
                return False
            kwarg_val = kwargs[kwarg_name]
            if isinstance(kwarg_def_value, tuple):
                # Checking if kwarg_val is in range
                if not (kwarg_def_value[0] <= kwarg_val <= kwarg_def_value[1]):
                    # logger.warning(f"Kwarg value out of range: {kwarg_name}={kwarg_val} not in {kwarg_def_value}")
                    return False
            else:
                # Checking if kwarg_val as primitive type is equal to kwarg_def_value
                # In case of complex argument type this needs to be extended
                if kwarg_def_value != kwarg_val and not (kwarg_def_value is None and kwarg_val is None):
                    # logger.warning(f"Kwarg value mismatch: {kwarg_name}={kwarg_val} != {kwarg_def_value}")
                    return False

        # All kwargs checked and no mismatch found
        return True

    @classmethod
    def _match_criteria(cls, rule_criteria: Callable[[TestVector], bool], test_vector: TestVector):
        return rule_criteria is None or rule_criteria(test_vector)

    @classmethod
    def _match_failing_reason(
        cls, rule_failing_reason: str, vector_failing_result: TestResultFailing, check_failing_reason: bool = False
    ):
        return not check_failing_reason or (
            rule_failing_reason is None
            or (vector_failing_result is not None and rule_failing_reason == vector_failing_result.failing_reason)
        )

    @classmethod
    def _match_skip_reason(
        cls, rule_skip_reason: str, vector_failing_result: TestResultFailing, check_failing_reason: bool = False
    ):
        return not check_failing_reason or (
            rule_skip_reason is None
            or (vector_failing_result is not None and rule_skip_reason == vector_failing_result.skip_reason)
        )

    @classmethod
    def test_vector_in_collection(
        cls, test_vector: TestVector, test_collection: TestCollection, check_failing_reason: bool = False
    ) -> bool:
        return (
            cls._match(test_collection.operators, test_vector.operator)
            and cls._match(test_collection.input_sources, test_vector.input_source)
            and cls._match(test_collection.input_shapes, test_vector.input_shape)
            and cls._match(test_collection.numbers_of_operands, test_vector.number_of_operands)
            and cls._match(test_collection.dev_data_formats, test_vector.dev_data_format)
            and cls._match(test_collection.math_fidelities, test_vector.math_fidelity)
            and cls._match_kwargs_list(test_collection.kwargs, test_vector.kwargs)
            and cls._match_criteria(test_collection.criteria, test_vector)
            and cls._match_failing_reason(
                test_collection.failing_reason, test_vector.failing_result, check_failing_reason
            )
            and cls._match_skip_reason(test_collection.skip_reason, test_vector.failing_result, check_failing_reason)
        )

    @classmethod
    def load_test_ids_from_file(cls, test_ids_file: str) -> List[str]:
        """Load test ids from a file to a list of strings"""
        with open(test_ids_file, "r") as file:
            test_ids = file.readlines()

            test_ids = [line.strip() for line in test_ids]

            return test_ids

    @classmethod
    def test_id_to_test_vector(cls, test_id: str) -> TestVector:

        test_id = test_id.replace("no_device-", "")

        # Split by '-' but not by ' -'
        parts = re.split(r"(?<! )-", test_id)
        assert len(parts) == 6 or len(parts) == 7, f"Invalid test id: {test_id} / {parts}"
        if len(parts) == 6:
            dev_data_format_index = 4
            math_fidelity_index = 5
        else:
            dev_data_format_index = 5
            math_fidelity_index = 6

        input_operator = parts[0]
        input_source = InputSource[parts[1]]
        kwargs = eval(parts[2])
        input_shape = eval(parts[3])

        dev_data_format_part = parts[dev_data_format_index]
        if dev_data_format_part == "None":
            dev_data_format_part = None
        dev_data_format = eval(f"forge._C.{dev_data_format_part}") if dev_data_format_part is not None else None

        math_fidelity_part = parts[math_fidelity_index]
        if math_fidelity_part == "None":
            math_fidelity_part = None
        # TODO remove hardcoded values here
        if math_fidelity_part in (
            "HiFi40",
            "HiFi41",
        ):
            math_fidelity_part = "HiFi4"
        math_fidelity = eval(f"forge._C.{math_fidelity_part}") if math_fidelity_part is not None else None

        return TestVector(
            operator=input_operator,
            input_source=input_source,
            input_shape=input_shape,
            kwargs=kwargs,
            dev_data_format=dev_data_format,
            math_fidelity=math_fidelity,
        )

    @classmethod
    def test_ids_to_test_vectors(cls, test_ids: List[str]) -> List[TestVector]:
        return [cls.test_id_to_test_vector(test_id) for test_id in test_ids]


class FailingRulesConverter:
    """Helper class for building failing rules for test plans"""

    @classmethod
    def build_rules(
        cls,
        rules: List[
            Union[
                Tuple[
                    Union[Optional[InputSource], List[InputSource]],
                    Union[Optional[TensorShape], List[TensorShape]],
                    Union[Optional[OperatorParameterTypes.Kwargs], List[OperatorParameterTypes.Kwargs]],
                    Union[Optional[forge.DataFormat], List[forge.DataFormat]],
                    Union[Optional[forge.MathFidelity], List[forge.MathFidelity]],
                    Optional[TestResultFailing],
                ],
                TestCollection,
            ]
        ],
        params: List[str] = [
            "input_source",
            "input_shape",
            "kwargs",
            "dev_data_format",
            "math_fidelity",
            "result_failing",
        ],
    ) -> List[TestCollection]:
        """Convert failing rules to TestCollection(s)"""
        input_soource_index = params.index("input_source") if "input_source" in params else None
        input_shape_index = params.index("input_shape") if "input_shape" in params else None
        kwargs_index = params.index("kwargs") if "kwargs" in params else None
        dev_data_format_index = params.index("dev_data_format") if "dev_data_format" in params else None
        math_fidelity_index = params.index("math_fidelity") if "math_fidelity" in params else None
        result_failing_index = params.index("result_failing") if "result_failing" in params else None
        test_collections = [
            cls.build_rule(
                input_source=rule[input_soource_index] if input_soource_index is not None else None,
                input_shape=rule[input_shape_index] if input_shape_index is not None else None,
                kwargs=rule[kwargs_index] if kwargs_index is not None else None,
                dev_data_format=rule[dev_data_format_index] if dev_data_format_index is not None else None,
                math_fidelity=rule[math_fidelity_index] if math_fidelity_index is not None else None,
                result_failing=rule[result_failing_index] if result_failing_index is not None else None,
            )
            if isinstance(rule, tuple)
            else rule  # if rule is already TestCollection there is no need to convert it
            for rule in rules
        ]

        return test_collections

    @classmethod
    def build_rule(
        cls,
        input_source: Optional[Union[InputSource, List[InputSource]]],
        input_shape: Optional[Union[TensorShape, List[TensorShape]]],
        kwargs: Optional[Union[OperatorParameterTypes.Kwargs, List[OperatorParameterTypes.Kwargs]]],
        dev_data_format: Optional[Union[forge.DataFormat, List[forge.DataFormat]]],
        math_fidelity: Optional[Union[forge.MathFidelity, List[forge.MathFidelity]]],
        result_failing: Optional[TestResultFailing],
    ) -> TestCollection:
        """Convert failing rule tuple to TestCollection"""

        if input_source is not None and not isinstance(input_source, list):
            input_source = [input_source]
        if input_shape is not None and not isinstance(input_shape, list):
            input_shape = [input_shape]
        if kwargs is not None and not isinstance(kwargs, list):
            kwargs = [kwargs]
        if dev_data_format is not None and not isinstance(dev_data_format, list):
            dev_data_format = [dev_data_format]
        if math_fidelity is not None and not isinstance(math_fidelity, list):
            math_fidelity = [math_fidelity]

        test_collection = TestCollection(
            input_sources=input_source,
            input_shapes=input_shape,
            dev_data_formats=dev_data_format,
            math_fidelities=math_fidelity,
            kwargs=kwargs,
            failing_reason=result_failing.failing_reason if result_failing is not None else None,
            skip_reason=result_failing.skip_reason if result_failing is not None else None,
        )

        return test_collection


class TestPlanScanner:

    METHOD_COLLECT_TEST_PLANS = "get_test_plans"

    @classmethod
    def find_modules_in_directory(cls, directory: str) -> List[str]:
        """Search for all modules in the directory and subdirectories."""
        modules = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".py") and not file.startswith("__init__"):
                    # Convert file path to Python module path
                    module_path = os.path.relpath(os.path.join(root, file), directory)
                    module_name = module_path[:-3].replace(os.sep, ".")
                    modules.append(module_name)
        return modules

    @classmethod
    def find_and_call_method(cls, module: ModuleType, method_name: str) -> Generator:
        """Find and call all method functions."""
        for name, func in inspect.getmembers(module, inspect.isfunction):
            if name == method_name and not inspect.isclass(func.__qualname__.split(".")[0]):
                logger.trace(f"Calling {method_name} from function: {name} in module: {module.__name__}")
                try:
                    results: List[Union[TestPlan, TestSuite], None, None] = func()  # Call the function
                    for result in results:
                        yield result
                except Exception as e:
                    logger.error(f"Error calling {name} in {module.__name__}: {e}")
                    raise e
        # return functions_called

    @classmethod
    def scan_and_invoke(cls, scan_file: str, scan_package: str, method_name: str) -> Generator:
        """
        Scan the directory and invoke all method functions.
        Scan modules relative path to the current directory.
        """

        directory = os.path.relpath(os.path.dirname(scan_file))
        logger.trace(f"Scan directory: {directory} for base package: {scan_package}")

        modules = cls.find_modules_in_directory(directory)

        for module_name in modules:
            try:
                module_name = f"{scan_package}.{module_name}"
                logger.trace(f"Loading module: {module_name}")
                # Dynamic module loading
                module = importlib.import_module(module_name)
                results = cls.find_and_call_method(module, method_name)
                for result in results:
                    yield result
            except Exception as e:
                logger.error(f"Problem loading module {module_name}: {e}")
                raise e

    @classmethod
    def collect_test_plans(cls, result: Union[TestPlan, TestSuite]) -> Generator[TestPlan, None, None]:
        if isinstance(result, TestSuite):
            test_suite = result
            for test_plan in test_suite.test_plans:
                yield test_plan
        elif isinstance(result, TestPlan):
            test_plan = result
            yield test_plan
        else:
            raise ValueError(f"Unsupported suite/plan type: {type(result)}")

    @classmethod
    def get_all_test_plans(cls, scan_file: str, scan_package: str) -> Generator[TestPlan, None, None]:
        """Get all test suites from the current directory."""
        results = cls.scan_and_invoke(scan_file, scan_package, cls.METHOD_COLLECT_TEST_PLANS)
        for result in results:
            for test_plan in cls.collect_test_plans(result):
                yield test_plan
        return results

    @classmethod
    def build_test_suite(cls, scan_file: str, scan_package: str) -> TestSuite:
        """Build test suite from scaned test plans."""
        logger.trace(f"Building test suite from file: {scan_file} and package: {scan_package}")
        test_plans = cls.get_all_test_plans(scan_file, scan_package)
        test_plans = list(test_plans)
        logger.trace(f"Found test plans: {len(test_plans)}")
        return TestSuite(test_plans=test_plans)
