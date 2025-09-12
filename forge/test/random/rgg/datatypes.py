# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Generic test model randomizer


from typing import Dict, List, Union, Type, Optional, Final, Any, Callable
from dataclasses import dataclass, field
import random
import torch

from forge import ForgeModule

from forge.op_repo import TensorShape
from forge.op_repo import OperatorDefinition
from forge.op_repo import OperatorRepository
from forge.op_repo import ShapeCalculationContext

from test.operators.utils.compat import TestDevice

from loguru import logger


@dataclass
class RandomizerInputNode:
    constant: Final[bool] = field(default=False, init=False)
    out_value: str
    input_shape: TensorShape


@dataclass
class RandomizerConstantNode:
    constant: Final[bool] = field(default=True, init=False)
    out_value: str
    input_shape: TensorShape


@dataclass
class RandomizerNode:
    constant: Final[bool] = field(default=False, init=False)
    index: Optional[int] = None
    out_value: Optional[str] = None
    operator: Optional[OperatorDefinition] = None
    input_num: int = field(init=False)
    inputs: List["RandomizerNode"] = field(init=False)
    constructor_kwargs: Dict[str, object] = field(default_factory=dict)
    forward_kwargs: Dict[str, object] = field(default_factory=dict)
    input_shapes: List[TensorShape] = field(default_factory=list)
    output_shape: TensorShape = None

    def __post_init__(self):
        # List of input nodes is initialized with None values for each input
        # Inputs will be set later during graph construction
        self.input_num = self.operator.input_num_range.operands_min
        self.init_inputs()

    def init_inputs(self):
        self.inputs = [None for _ in range(self.input_num)]

    @property
    def operator_name(self):
        return f"op{self.index}"

    @property
    def layer_name(self):
        return f"l{self.index}"

    @property
    def node_name(self):
        return self.operator_name if self.operator.is_operator else self.layer_name

    @property
    def name(self):
        return self.operator.name

    @property
    def node_info(self):
        return f"{self.node_name} {self.name}"


class NodeShapeCalculationContext(ShapeCalculationContext):
    def __init__(self, node: RandomizerNode, test_context: "RandomizerTestContext"):
        self.node = node
        self.test_context = test_context

    @property
    def operator(self) -> OperatorDefinition:
        return self.node.operator

    @property
    def input_num(self) -> int:
        return self.node.input_num

    @property
    def constructor_kwargs(self) -> Dict[str, object]:
        return self.node.constructor_kwargs

    @property
    def forward_kwargs(self) -> Dict[str, object]:
        return self.node.forward_kwargs

    @property
    def output_shape(self) -> TensorShape:
        return self.node.output_shape

    @property
    def rng_shape(self) -> random.Random:
        return self.test_context.rng_shape


@dataclass
class ExecutionContext:
    values: Dict
    last_value: torch.Tensor
    node: Optional[RandomizerNode] = None
    inputs: Optional[List[torch.Tensor]] = None


@dataclass
class OperatorList:
    framework_name: str
    operators: List[str]

    def __add__(self, other: Union["OperatorList", List[str]]) -> "OperatorList":
        if isinstance(other, OperatorList):
            other_operators = other.operators
        else:
            other_operators = other
        return OperatorList(self.framework_name, self.operators + other_operators)

    def __sub__(self, other: Union["OperatorList", List[str]]) -> "OperatorList":
        operators = self.operators
        if isinstance(other, OperatorList):
            skip_operators = other.operators
        else:
            skip_operators = other

        initial_operator_count = len(operators)
        if len(skip_operators) == 0:
            # Nothing to skip, avoid logging
            return self
        logger.trace(
            f"Skipping {len(skip_operators)} operators for framework {self.framework_name}: {[op for op in skip_operators]}"
        )
        for skip_op in skip_operators:
            if skip_op not in operators:
                logger.warning(f"Operator {skip_op} not found in framework {self.framework_name}, can't skip it")
                for op in operators:
                    logger.warning(f"Available operator: {op}")
        operators = [op for op in operators if op not in skip_operators]
        logger.debug(
            f"Skipped num of operators for framework {self.framework_name}: {initial_operator_count} -> {len(operators)}"
        )
        assert (
            len(operators) + len(skip_operators) == initial_operator_count
        ), f"Operators count should match after skipping operators {len(operators)} + {len(skip_operators)} == {initial_operator_count}"
        return OperatorList(self.framework_name, operators)

    def __mul__(self, other: Union["OperatorList", List[str]]) -> "OperatorList":
        operators = self.operators
        if isinstance(other, OperatorList):
            allow_operators = other.operators
        else:
            allow_operators = other

        initial_operator_count = len(operators)
        logger.trace(
            f"Allowing {len(allow_operators)} operators for framework {self.framework_name}: {[op for op in allow_operators]}"
        )
        skip_operators = [op for op in operators if op not in allow_operators]
        operators = (self - skip_operators).operators
        logger.debug(
            f"Allowed num of operators for framework {self.framework_name}: {initial_operator_count} -> {len(operators)}"
        )
        assert len(allow_operators) == len(
            operators
        ), f"Operators count should match after allowing operators {len(operators)} == {len(allow_operators)}"
        return OperatorList(self.framework_name, operators)


@dataclass
class Framework:

    template_name: str
    framework_name: str
    ModelBuilderType: Type["ModelBuilder"]
    operator_repository: OperatorRepository

    @property
    def operator_list(self) -> OperatorList:
        return OperatorList(self.framework_name, [op.name for op in self.operator_repository.operators])


@dataclass
class Algorithm:

    name: str
    GraphBuilderType: Type["GraphBuilder"]


@dataclass
class RandomizerParameters:
    test_index: int
    random_seed: int
    test_device: TestDevice
    framework: Framework
    graph_builder_name: str


# TODO load from file
@dataclass
class RandomizerGraph:
    # parameters: RandomizerParameters
    nodes: List[RandomizerNode] = field(default_factory=list)
    input_nodes: List[RandomizerInputNode] = field(default_factory=list)
    constant_nodes: List[RandomizerConstantNode] = field(default_factory=list)
    # graph_builder: Optional[str] = None


@dataclass
class RandomizerConfig:
    print_graph: bool = True
    print_code: bool = False
    run_test: bool = True
    test_dir: str = "forge/test/random_tests"
    save_tests: bool = False
    save_failing_tests: bool = False
    # build_model_from_code: bool = False  # TODO remove obsoleted
    debug_shapes: bool = (False,)
    verify_shapes: bool = (False,)
    verification_timeout: int = 60
    dim_min: int = 3
    dim_max: int = 4
    op_size_per_dim_min: int = 16
    op_size_per_dim_max: int = 512
    op_size_quantization: int = 1
    microbatch_size_min: int = 1
    microbatch_size_max: int = 8
    num_of_nodes_min: int = 5
    num_of_nodes_max: int = 10
    num_fork_joins_max: int = 50
    constant_input_rate: int = 20
    same_inputs_percent_limit: int = 10


@dataclass
class RandomizerTestContext:
    randomizer_config: RandomizerConfig
    parameters: RandomizerParameters
    # framework: Framework
    # graph_builder: GraphBuilder
    graph: Optional[RandomizerGraph]  # graph will be constructed later during test processing
    test_name: str = "Default"

    # random number generators for graph building
    rng_graph: Optional[random.Random] = None
    # random number generators for shape generation
    rng_shape: Optional[random.Random] = None
    # random number generators for parameters
    rng_params: Optional[random.Random] = None
    record_property: Optional[Callable[[str, Any], None]] = None


class ModelBuilder:
    """
    ModelBuilder is an interface that each framework should implement for instantiated model instances from a previously generated test model class.
    """

    def build_model(self, graph: RandomizerGraph, GeneratedTestModel: Type) -> ForgeModule:
        raise Exception("Method build_model() not implemented")


class InvalidShape(Exception):
    def __init__(self, message):
        super().__init__(message)
