# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Test plan management utilities

import types
import pytest
import forge
import types
import re

from _pytest.mark import Mark
from _pytest.mark import ParameterSet

from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
from typing import Callable, Generator, Optional, List, Dict, Union, Tuple, TypeAlias

from forge import MathFidelity, DataFormat
from forge.op_repo import TensorShape

from .pytest import PytestParamsUtils


class InputSource(Enum):
    FROM_ANOTHER_OP = 1
    FROM_HOST = 2
    FROM_DRAM_QUEUE = 3
    CONST_EVAL_PASS = 4


OperatorParameterSingleType: TypeAlias = Union[int, float]

OperatorParameterRangeType: TypeAlias = Tuple[OperatorParameterSingleType, OperatorParameterSingleType]

OperatorParameterType: TypeAlias = Union[OperatorParameterSingleType, OperatorParameterRangeType]

KwargsType: TypeAlias = Dict[str, OperatorParameterType]


@dataclass
class TestResultFailing:
    '''
    Dataclass for defining failing test result

    Args:
        failing_reason: Failing reason
        skip_reason: Skip reason
    '''

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
    '''
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
    '''

    __test__ = False  # Avoid collecting TestVector as a pytest test

    operator: Optional[str]
    input_source: InputSource
    input_shape: TensorShape  # TODO - Support multiple input shapes
    dev_data_format: Optional[DataFormat] = None
    math_fidelity: Optional[MathFidelity] = None
    kwargs: Optional[KwargsType] = None
    pcc: Optional[float] = None
    failing_result: Optional[TestResultFailing] = None

    def get_id(self) -> str:
        '''Get test vector id'''
        return f"{self.operator}-{self.input_source.name}-{self.kwargs}-{self.input_shape}-{self.dev_data_format.name if self.dev_data_format else None}-{self.math_fidelity.name if self.math_fidelity else None}"

    def get_marks(self) -> List[Mark]:
        '''Get marks for the test vector'''
        marks = self.failing_result.get_marks() if self.failing_result is not None else []
        return marks

    def to_param(self) -> ParameterSet:
        '''Convert test vector to pytest parameter set'''
        return pytest.param(self, marks=self.get_marks(), id=self.get_id())


@dataclass
class TestCollection:
    '''
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
    '''

    __test__ = False  # Avoid collecting TestCollection as a pytest test

    operators: Optional[List[str]] = None
    input_sources: Optional[List[InputSource]] = None
    input_shapes: Optional[List[TensorShape]] = None  # TODO - Support multiple input shapes
    dev_data_formats: Optional[List[DataFormat]] = None
    math_fidelities: Optional[List[MathFidelity]] = None
    kwargs: Optional[Union[List[KwargsType], Callable[['TestVector'], List[KwargsType]]]] = None
    pcc: Optional[float] = None
    criteria: Optional[Callable[['TestVector'], bool]] = None

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


@dataclass
class TestPlan:
    '''
    Define test plan for the operator testing. Define failing rules for the tests.
    
    Args:
        collections: List of test collections
        failing_rules: List of failing rules
    '''

    __test__ = False  # Avoid collecting TestPlan as a pytest test

    collections: Optional[List[TestCollection]] = None
    failing_rules: Optional[List[TestCollection]] = None

    def _check_kwargs_matching (
        self,
        kwargs_def_list: List[Dict[str, OperatorParameterType]],
        kwargs: Dict,
    ):
        for kwargs_def in kwargs_def_list:
            for(kwarg_name, kwarg_def) in kwargs_def.items():
                if kwarg_name not in kwargs:
                    # logger.warning(f"Missing kwarg: {kwarg_name}")
                    continue
                kwarg_val = kwargs[kwarg_name]
                if isinstance(kwarg_def, tuple):
                    if not (kwarg_def[0] <= kwarg_val <= kwarg_def[1]):
                        # logger.warning(f"Kwarg value out of range: {kwarg_name}={kwarg_val} not in {kwarg_def}")
                        return False
                else:
                    if kwarg_def != kwarg_val:
                        # logger.warning(f"Kwarg value mismatch: {kwarg_name}={kwarg_val} != {kwarg_def}")
                        return False
        return True

    def _check_test_failing(
        self,
        test_vector: TestVector,
    ) -> Optional[TestResultFailing]:
        '''Check if the test is failing based on the test plan
        
        Args:
            test_vector: Test vector with all the parameters
        '''

        failing_result = None

        for failing_rule in self.failing_rules:
            if (
                (failing_rule.operators is None or test_vector.operator in failing_rule.operators) \
                and (failing_rule.input_sources is None or test_vector.input_source in failing_rule.input_sources) \
                and (failing_rule.input_shapes is None or test_vector.input_shape in failing_rule.input_shapes) \
                and (failing_rule.dev_data_formats is None or test_vector.dev_data_format in failing_rule.dev_data_formats) \
                and (failing_rule.math_fidelities is None or test_vector.math_fidelity in failing_rule.math_fidelities) \
                and (failing_rule.kwargs is None or self._check_kwargs_matching(failing_rule.kwargs, test_vector.kwargs)) \
                and (failing_rule.criteria is None or failing_rule.criteria(test_vector)) \
            ):
                if failing_rule.failing_reason is not None or failing_rule.skip_reason is not None:
                    failing_result = TestResultFailing(failing_rule.failing_reason, failing_rule.skip_reason)
                else:
                    # logger.debug(f"Test should pass: {test_vector.get_id()}")
                    failing_result = None

        return failing_result

    def generate(self) -> Generator[TestVector, None, None]:
        '''Generate test vectors based on the test plan'''

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

            for input_operator in test_collection.operators:
                for input_source in test_collection.input_sources:
                    for input_shape in test_collection.input_shapes:
                        if isinstance(input_shape, ParameterSet):
                            input_shape = input_shape[0][0]
                        for dev_data_format in dev_data_formats:
                            if isinstance(dev_data_format, ParameterSet):
                                dev_data_format = dev_data_format[0][0]
                            for math_fidelity in math_fidelities:
                                if isinstance(math_fidelity, ParameterSet):
                                    math_fidelity = math_fidelity[0][0]

                                test_vector = TestVector(
                                    operator=input_operator,
                                    input_source=input_source,
                                    input_shape=input_shape,
                                    dev_data_format=dev_data_format,
                                    math_fidelity=math_fidelity,
                                    pcc=test_collection.pcc,
                                )

                                if isinstance(test_collection.kwargs, types.FunctionType):
                                    kwargs_list = test_collection.kwargs(test_vector)

                                for kwargs in kwargs_list:
                                    test_vector.kwargs = kwargs

                                    test_vector.failing_result = self._check_test_failing(test_vector)
                                    yield test_vector


@dataclass
class TestParamsFilter:
    '''
    Dataclass for specifying test parameters filter

    Args:
        allow: Allow function
        indices: Indices to filter
        reversed: Reverse the order
        log: Log the parameters
    '''
    allow: Optional[Callable[[TestVector], bool]] = lambda test_vector: True
    indices: Optional[Union[int, Tuple[int, int], List[int]]] = None
    reversed: bool = False
    log: bool = False


class TestPlanUtils:
    '''
    Utility functions for test vectors
    '''

    @classmethod
    def load_test_ids_from_file(cls, test_ids_file: str) -> List[str]:
        with open(test_ids_file, 'r') as file:
            # Učitaj svaki red u listu stringova
            test_ids = file.readlines()

            test_ids = [line.strip() for line in test_ids]

            return test_ids

    @classmethod
    def test_id_to_test_vector(cls, test_id: str) -> TestCollection:
        # Split by '-' but not by ' -'
        parts = re.split(r'(?<! )-', test_id)
        assert len(parts) == 7, f"Invalid test id: {test_id} / {parts}"

        input_operator = parts[1]
        input_source = InputSource[parts[2]]
        kwargs = eval(parts[3])
        input_shape = eval(parts[4])

        dev_data_format_part = parts[5]
        if dev_data_format_part == "None":
            dev_data_format_part = None
        # print(f"dev_data_format_part: {dev_data_format_part}")
        dev_data_format = eval(f"forge._C.{dev_data_format_part}") if dev_data_format_part is not None else None

        math_fidelity_part = parts[6]
        if math_fidelity_part == "None":
            math_fidelity_part = None
        # TODO remove hardcoded values here
        if math_fidelity_part in ("HiFi40", "HiFi41", ):
            math_fidelity_part = "HiFi4"
        # print(f"math_fidelity_part: {math_fidelity_part} {math_fidelity_part is not None}")
        math_fidelity = eval(f"forge._C.{math_fidelity_part}") if math_fidelity_part is not None else None

        return TestCollection(
            operators=[input_operator],
            input_sources=[input_source],
            input_shapes=[input_shape],
            kwargs=[kwargs],
            dev_data_formats=[dev_data_format],
            math_fidelities=[math_fidelity],
        )

    @classmethod
    def build_test_plan_from_id_list(cls, test_ids: List[str], test_plan_failing: Optional[TestPlan] = None) -> TestPlan:
        test_plan = TestPlan(
            collections = [cls.test_id_to_test_vector(test_id) for test_id in test_ids],
            failing_rules = test_plan_failing.failing_rules if test_plan_failing is not None else [],
        )

        return test_plan

    @classmethod
    def build_test_plan_from_id_file(cls, test_ids_file: str, test_plan_failing: TestPlan) -> TestPlan:
        test_ids = cls.load_test_ids_from_file(test_ids_file)

        test_plan = cls.build_test_plan_from_id_list(test_ids, test_plan_failing)

        return test_plan

    @classmethod
    def generate_params(cls, test_plan: TestPlan, filter: Optional[TestParamsFilter] = None) -> Generator[ParameterSet, None, None]:
        test_vectors = test_plan.generate()

        test_vectors = cls.process_filter(test_vectors, filter)

        for test_vector in test_vectors:
            yield test_vector.to_param()

    @classmethod
    def yield_test_vectors(cls, test_vector: Union[TestVector, List[TestVector], Generator[TestVector, None, None]]) -> Generator[TestVector, None, None]:
        if test_vector is None:
            pass
        elif isinstance(test_vector, TestVector):
            yield test_vector
        elif isinstance(test_vector, types.GeneratorType):
            return test_vector
        elif isinstance(test_vector, list):
            for item in test_vector:
                yield item

    @classmethod
    def filter_allowed(cls, test_params: Generator[TestVector, None, None], filter: TestParamsFilter) -> Generator[TestVector, None, None]:
        index = 0
        for p in test_params:
            allowed = False
            if filter.allow is None:
                allowed = True
            elif filter.allow(p):
                if filter.indices is None:
                    allowed = True
                else:
                    if isinstance(filter.indices, int):
                        # logger.info(f"Int type filter.indices: {filter.indices}")
                        if filter.indices == index:
                            allowed = True
                    elif isinstance(filter.indices, tuple):
                        # logger.info(f"Tuple type filter.indices: {filter.indices}")
                        range_min, range_max = filter.indices
                        if range_min <= index <= range_max:
                            allowed = True
                    elif isinstance(filter.indices, list):
                        # logger.info(f"List type filter.indices: {filter.indices}")
                        if index in filter.indices:
                            allowed = True
                    else:
                        logger.error(f"Invalid filter.indices: {filter.indices}")
            
            index += 1
            if allowed:
                yield p

    @classmethod
    def process_filter(cls, test_params: Generator[TestVector, None, None], filter: Optional[TestParamsFilter] = None) -> Generator[TestVector, None, None]:
        if filter is not None:
            test_params = cls.filter_allowed(test_params, filter)

            if filter.reversed == True:

                if not isinstance(test_params, list):
                    test_params = list(test_params)

                test_params = test_params[::-1]

        if filter is not None and filter.log == True:
            logger.info("Parameters:")
        for p in test_params:
            if filter is not None and filter.log == True:
                logger.info(f"{p.get_id()}")
            yield p
