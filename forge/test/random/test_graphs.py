# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Test random graph configurations by utilizing Random Graph Generator Algorithm and targeting Forge and PyTorch frameworks

from enum import Enum
import pytest

from copy import copy
from dataclasses import dataclass
from typing import Generator, Optional, Tuple, List, Union, Type, Callable, Any
from loguru import logger

from forge.op_repo import OperatorParamNumber

from test.random.rgg import Framework
from test.random.rgg import Frameworks
from test.random.rgg import FrameworkTestUtils
from test.random.rgg import Algorithm
from test.random.rgg import Algorithms
from test.random.rgg import RandomGraphAlgorithm
from test.random.rgg import RandomizerConfig
from test.random.rgg import process_test

import os
import random
import textwrap

from tabulate import tabulate

from .rgg import get_randomizer_config_default


@dataclass
class RGGRunParams:
    """Run parameters for Random Graph Generator"""

    random_test_count: int
    random_test_selected: Optional[List[int]]
    random_test_seed: int
    frameworks: List[str]
    configs: List[str]
    algorithms: List[str]

    @classmethod
    def _get_env_list(cls, name: str, default: str = "") -> Optional[List[str]]:
        values = os.getenv(name, default).strip().split(",")
        values = [value for value in values if value]
        if len(values) == 0:
            values = None
        return values

    @classmethod
    def from_env(cls):
        random_test_count = int(os.environ.get("RANDOM_TEST_COUNT", 5))
        random_test_selected = cls._get_env_list("RANDOM_TESTS_SELECTED")
        if random_test_selected:
            random_test_selected = [int(i) for i in random_test_selected]
        random_test_seed = int(os.environ.get("RANDOM_TEST_SEED", 0))
        frameworks = cls._get_env_list("FRAMEWORKS")
        configs = cls._get_env_list("CONFIGS", "DEFAULT")
        algorithms = cls._get_env_list("ALGORITHMS", "RANDOM")
        return cls(
            random_test_count=random_test_count,
            random_test_selected=random_test_selected,
            random_test_seed=random_test_seed,
            frameworks=frameworks,
            configs=configs,
            algorithms=algorithms,
        )


class OperatorLists:

    NOT_IMPLEMENTED_FORGE = (
        # Unary operators
        "atan",
        "buffer",
        "pow",
        "logical_not",  # bug
        "logical_and",  # bug
        "dropout",  # pcc?
        # Binary operators
        "less_equal",
        "power",  # occasionally fails
        # Nary operators
        "interleave",
        # Other
    )

    UNSTABLE_FORGE = (
        # Unary operators
        "exp",  # pcc?
        "sqrt",  # skip because it's failing for negative values
        "cumsum",  # bug
        "argmax",  # shape calc is wrong
        "tilizer",  # bug
        # Binary operators
        "divide",  # bug
        # Nary operators
        "where",  # pcc?
        # Other
    )

    NOT_IMPLEMENTED_PYTORCH = (
        # Unary operators
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
        # "bitwise_not",
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
        "isnan",
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
        # "tanh",
        "trunc",
        # Binary operators
        # Nary operators
    )

    UNSTABLE_PYTORCH = (
        # Unary operators
        # "relu",
        "sqrt",  # skip because it's failing for negative values
        # "reciprocal",
        # "sigmoid",
        # "abs",
        # "cos",
        # "exp",
        # "neg",
        # "rsqrt",
        # "sin",
        # "square",
        "pow",
        "clamp",
        # "log",
        # "log1p",
        # "gelu",
        # "leaky_relu",
        "bitwise_not",
        # Binary operators
        # Nary operators
        # Other
        # "linear",
        "conv2d",  # skip until calc_input_shapes is properly implemented
    )


class FrameworkBuilder:
    """Adjust repositories to test healthy operators"""

    @classmethod
    def healty_forge(cls):
        SKIP_OPERATORS = ()
        SKIP_OPERATORS += OperatorLists.NOT_IMPLEMENTED_FORGE
        SKIP_OPERATORS += OperatorLists.UNSTABLE_FORGE

        framework = FrameworkTestUtils.copy_framework(Frameworks.FORGE.value, "Healthy Forge", SKIP_OPERATORS)

        pow_operator = FrameworkTestUtils.copy_operator(framework, "pow")
        if pow_operator:
            pow_operator.forward_params = [
                # float exponent is currently not supported due to issue #2592
                # OperatorParamNumber("exponent", float, 0, 100),
                # OperatorParamNumber("exponent", int, 0, 100),
                OperatorParamNumber("exponent", int, 0, 4),  # pcc for higher numbers fails
            ]

        return framework

    @classmethod
    def unstable_forge(cls):
        framework = FrameworkTestUtils.copy_framework(Frameworks.FORGE.value, "Unstable Forge")
        FrameworkTestUtils.allow_operators(framework, OperatorLists.UNSTABLE_FORGE)
        return framework

    @classmethod
    def not_implemented_forge(cls):
        framework = FrameworkTestUtils.copy_framework(Frameworks.FORGE.value, "Not implemented Forge")
        FrameworkTestUtils.allow_operators(framework, OperatorLists.NOT_IMPLEMENTED_FORGE)
        return framework

    @classmethod
    def healty_pytorch(cls):
        SKIP_OPERATORS = ()
        SKIP_OPERATORS += OperatorLists.NOT_IMPLEMENTED_PYTORCH
        SKIP_OPERATORS += OperatorLists.UNSTABLE_PYTORCH

        framework = FrameworkTestUtils.copy_framework(Frameworks.PYTORCH.value, "Healthy PyTorch", SKIP_OPERATORS)

        return framework

    @classmethod
    def unstable_pytorch(cls):
        framework = FrameworkTestUtils.copy_framework(Frameworks.PYTORCH.value, "Unstable PyTorch")
        FrameworkTestUtils.allow_operators(framework, OperatorLists.UNSTABLE_PYTORCH)
        return framework

    @classmethod
    def not_implemented_pytorch(cls):
        framework = FrameworkTestUtils.copy_framework(Frameworks.PYTORCH.value, "Not implemented PyTorch")
        FrameworkTestUtils.allow_operators(framework, OperatorLists.NOT_IMPLEMENTED_PYTORCH)
        return framework

    @staticmethod
    def forge_fork_joins():
        SKIP_OPERATORS = ()

        framework = FrameworkTestUtils.copy_framework(Frameworks.FORGE.value, "Forge fork joins", SKIP_OPERATORS)

        ALLOW_OPERATORS = (
            "relu",
            "tanh",
            "add",
            "matmul",
        )

        FrameworkTestUtils.allow_operators(framework, ALLOW_OPERATORS)

        return framework

    @staticmethod
    def forge_nary():
        SKIP_OPERATORS = ()

        framework = FrameworkTestUtils.copy_framework(Frameworks.FORGE.value, "Forge nary", SKIP_OPERATORS)

        ALLOW_OPERATORS = (
            # "relu",
            "tanh",
            "add",
            "matmul",  # Skip matmul to increase chance for stack operator
            "interleave",
            # "where",  # pcc?
            "concatenate",
            "stack",
        )

        FrameworkTestUtils.allow_operators(framework, ALLOW_OPERATORS)

        return framework


class RandomizerConfigBuilder:

    DEFAULT = get_randomizer_config_default()

    @classmethod
    def default(cls):
        # adjust randomizer_config
        randomizer_config = copy(cls.DEFAULT)
        # randomizer_config.debug_shapes = True
        # randomizer_config.verify_shapes = True

        # Uncomment the following randomizer_config values to override the default values
        # randomizer_config.dim_min = 3
        # randomizer_config.dim_max = 4
        # randomizer_config.op_size_per_dim_min = 4
        # # randomizer_config.op_size_per_dim_min = 16
        # randomizer_config.op_size_per_dim_max = 8
        # # randomizer_config.op_size_per_dim_max = 64
        # # randomizer_config.op_size_per_dim_max = 256
        # randomizer_config.microbatch_size_min = 1
        # randomizer_config.microbatch_size_max = 8
        # randomizer_config.num_of_nodes_min = 5
        # randomizer_config.num_of_nodes_max = 10
        # randomizer_config.num_fork_joins_max = 5

        return randomizer_config

    @classmethod
    def forge_fork_joins(cls):
        # adjust randomizer_config
        randomizer_config = copy(cls.DEFAULT)
        # randomizer_config.debug_shapes = True
        # randomizer_config.verify_shapes = True
        randomizer_config.dim_min = 3
        randomizer_config.dim_max = 4
        randomizer_config.op_size_per_dim_min = 4
        # randomizer_config.op_size_per_dim_min = 16
        randomizer_config.op_size_per_dim_max = 8
        # randomizer_config.op_size_per_dim_max = 64
        # randomizer_config.op_size_per_dim_max = 256
        randomizer_config.microbatch_size_min = 1
        randomizer_config.microbatch_size_max = 8
        randomizer_config.num_of_nodes_min = 10
        randomizer_config.num_of_nodes_max = 15
        randomizer_config.num_fork_joins_max = 10

        return randomizer_config

    @classmethod
    def forge_nary(cls):
        # adjust randomizer_config
        randomizer_config = copy(cls.DEFAULT)
        # randomizer_config.debug_shapes = True
        # randomizer_config.verify_shapes = True
        randomizer_config.dim_min = 3
        randomizer_config.dim_max = 4
        randomizer_config.op_size_per_dim_min = 2  # avoid failing tests with smaller dimensions?
        # randomizer_config.op_size_per_dim_min = 4
        # randomizer_config.op_size_per_dim_min = 16
        randomizer_config.op_size_per_dim_max = 8
        # randomizer_config.op_size_per_dim_max = 64
        # randomizer_config.op_size_per_dim_max = 256
        randomizer_config.op_size_quantization = 2
        randomizer_config.microbatch_size_min = 1
        randomizer_config.microbatch_size_max = 8
        randomizer_config.num_of_nodes_min = 10
        randomizer_config.num_of_nodes_max = 15
        randomizer_config.num_fork_joins_max = 10

        return randomizer_config


@dataclass
class RGGTestConfiguration:
    framework: Framework
    algorithm: Algorithm
    config_name: str
    randomizer_config: RandomizerConfig

    def get_id(self):
        return f"{self.framework.template_name}_{self.algorithm.name}_{self.config_name}"


class RGGConfiguraionProvider:
    def __init__(self):
        self.run_params = RGGRunParams.from_env()
        logger.debug(f"run_params = {self.run_params}")

        self.all_tests = [
            RGGTestConfiguration(
                framework=FrameworkBuilder.healty_forge(),
                algorithm=Algorithms.RANDOM.value,
                config_name="Default",
                randomizer_config=RandomizerConfigBuilder.default(),
            ),
            RGGTestConfiguration(
                framework=FrameworkBuilder.unstable_forge(),
                algorithm=Algorithms.RANDOM.value,
                config_name="Unstable",
                randomizer_config=RandomizerConfigBuilder.default(),
            ),
            RGGTestConfiguration(
                framework=FrameworkBuilder.not_implemented_forge(),
                algorithm=Algorithms.RANDOM.value,
                config_name="Not implemented",
                randomizer_config=RandomizerConfigBuilder.default(),
            ),
            RGGTestConfiguration(
                framework=FrameworkBuilder.healty_pytorch(),
                algorithm=Algorithms.RANDOM.value,
                config_name="Default",
                randomizer_config=RandomizerConfigBuilder.default(),
            ),
            RGGTestConfiguration(
                framework=FrameworkBuilder.unstable_pytorch(),
                algorithm=Algorithms.RANDOM.value,
                config_name="Unstable",
                randomizer_config=RandomizerConfigBuilder.default(),
            ),
            RGGTestConfiguration(
                framework=FrameworkBuilder.not_implemented_pytorch(),
                algorithm=Algorithms.RANDOM.value,
                config_name="Not implemented",
                randomizer_config=RandomizerConfigBuilder.default(),
            ),
            RGGTestConfiguration(
                framework=FrameworkBuilder.forge_fork_joins(),
                algorithm=Algorithms.RANDOM.value,
                config_name="Fork Joins",
                randomizer_config=RandomizerConfigBuilder.forge_fork_joins(),
            ),
            RGGTestConfiguration(
                framework=FrameworkBuilder.forge_nary(),
                algorithm=Algorithms.RANDOM.value,
                config_name="Nary",
                randomizer_config=RandomizerConfigBuilder.forge_nary(),
            ),
        ]

    def get_tests(self) -> Generator[Tuple[Framework, str, RandomizerConfig], None, None]:
        if self.run_params.frameworks:
            frameworks = [framwork.lower() for framwork in self.run_params.frameworks]
        else:
            frameworks = []
        if self.run_params.algorithms:
            algorithms = [algorithm.lower() for algorithm in self.run_params.algorithms]
        else:
            algorithms = []
        if self.run_params.configs:
            config_names = [config_name.lower() for config_name in self.run_params.configs]
        else:
            config_names = []

        for test in self.all_tests:
            if not (frameworks is None or test.framework.template_name.lower() in frameworks):
                continue
            if not (algorithms is None or test.algorithm.name.lower() in algorithms):
                continue
            if not (config_names is None or test.config_name.lower() in config_names):
                continue
            yield pytest.param(
                test.framework, test.algorithm, test.config_name, test.randomizer_config, id=test.get_id()
            )

    def get_random_seeds(self) -> List[Tuple[int, int]]:
        test_count = self.run_params.random_test_count
        test_selected = self.run_params.random_test_selected
        if test_selected:
            last_test_selected = max(test_selected)
            if test_count < last_test_selected + 1:
                test_count = last_test_selected + 1
        else:
            test_selected = range(test_count)

        test_rg = random.Random()
        test_rg.seed(self.run_params.random_test_seed)

        seeds = []
        # generate a new random seed for each test.
        for _ in range(test_count):
            seeds.append(test_rg.randint(0, 1000000))

        selected_seeds = [(index, seeds[index]) for index in test_selected]

        return selected_seeds


configuration_provider = RGGConfiguraionProvider()


@pytest.mark.parametrize("test_index, random_seed", configuration_provider.get_random_seeds())
@pytest.mark.parametrize("framework, algorithm, config_name, randomizer_config", configuration_provider.get_tests())
def test_graphs(
    test_index: int,
    random_seed: int,
    test_device: "TestDevice",
    framework: Framework,
    algorithm: Algorithm,
    config_name: str,
    randomizer_config: RandomizerConfig,
    record_property: Callable[[str, Any], None],
):
    process_test(
        test_name=config_name,
        test_index=test_index,
        random_seed=random_seed,
        test_device=test_device,
        randomizer_config=randomizer_config,
        framework=framework,
        algorithm=algorithm,
        record_property=record_property,
    )


@dataclass
class InfoColumn:
    name: str
    header: str
    width: Union[int, float]


class InfoUtils:
    @classmethod
    def print_query_params(cls, max_width=80):
        print("Query parameters:")
        cls.print_query_values(max_width)
        print("Query examples:")
        cls.print_query_examples(max_width)

    @classmethod
    def print_query_values(cls, max_width=80):

        all_tests = configuration_provider.all_tests

        frameworks = [test.framework.template_name.upper() for test in all_tests]
        frameworks = set(frameworks)
        frameworks = ", ".join(frameworks)

        algorithms = [test.algorithm.name.upper() for test in all_tests]
        algorithms = set(algorithms)
        algorithms = ", ".join(algorithms)

        config_names = [test.config_name.upper() for test in all_tests]
        config_names = set(config_names)
        config_names = ", ".join(config_names)

        parameters = [
            {
                "name": "FRAMEWORKS",
                "description": "List of frameworks.",
                "supported_values": f"{frameworks}",
                "default": "",
            },
            {
                "name": "ALGORITHMS",
                "description": "List of algorithms.",
                "supported_values": f"{algorithms}",
                "default": "RANDOM",
            },
            {
                "name": "CONFIGS",
                "description": "List of config names.",
                "supported_values": f"{config_names}",
                "default": "DEFAULT",
            },
            {
                "name": "RANDOM_TEST_SEED",
                "description": "Initial seed for RGG.",
                "supported_values": "",
                "default": "0",
            },
            {
                "name": "RANDOM_TEST_COUNT",
                "description": "Number of random tests to be generated and executed.",
                "supported_values": "",
                "default": "5",
            },
            {
                "name": "RANDOM_TESTS_SELECTED",
                "description": "Limiting random tests to only selected subset defined as comma separated list of test indexes.",
                "supported_values": "",
                "default": "no limitation if not specified or empty",
            },
            {
                "name": "VERIFICATION_TIMEOUT",
                "description": "Limit time for inference verification in seconds.",
                "supported_values": "",
                "default": "60",
            },
            {
                "name": "MIN_DIM",
                "description": "Minimal number of dimensions of input tensors.",
                "supported_values": "",
                "default": "3",
            },
            {
                "name": "MAX_DIM",
                "description": "Maximum number of dimensions of input tensors.",
                "supported_values": "",
                "default": "4",
            },
            {
                "name": "MIN_OP_SIZE_PER_DIM",
                "description": "Minimal size of an operand dimension.",
                "supported_values": "",
                "default": "16",
            },
            {
                "name": "MAX_OP_SIZE_PER_DIM",
                "description": "Maximum size of an operand dimension. Smaller operand size results in fewer failed tests.",
                "supported_values": "",
                "default": "512",
            },
            {
                "name": "OP_SIZE_QUANTIZATION",
                "description": "Quantization factor for operand size.",
                "supported_values": "",
                "default": "1",
            },
            {
                "name": "MIN_MICROBATCH_SIZE",
                "description": "Minimal size of microbatch of an input tensor.",
                "supported_values": "",
                "default": "1",
            },
            {
                "name": "MAX_MICROBATCH_SIZE",
                "description": "Maximum size of microbatch of an input tensor.",
                "supported_values": "",
                "default": "8",
            },
            {
                "name": "NUM_OF_NODES_MIN",
                "description": "Minimal number of nodes to be generated by RGG.",
                "supported_values": "",
                "default": "5",
            },
            {
                "name": "NUM_OF_NODES_MAX",
                "description": "Maximum number of nodes to be generated by RGG.",
                "supported_values": "",
                "default": "10",
            },
            {
                "name": "NUM_OF_FORK_JOINS_MAX",
                "description": "Maximum number of fork joins to be generated by random graph algorithm in RGG.",
                "supported_values": "",
                "default": "50",
            },
            {
                "name": "CONSTANT_INPUT_RATE",
                "description": "Rate of constant inputs in RGG in percents.",
                "supported_values": "",
                "default": "50",
            },
            {
                "name": "SAME_INPUTS_PERCENT_LIMIT",
                "description": "Percent limit of nodes which have same value on multiple inputes.",
                "supported_values": "",
                "default": "10",
            },
        ]

        cls.print_formatted_parameters(
            parameters,
            max_width,
            columns=[
                InfoColumn("name", "Parameter", 25),
                InfoColumn("description", "Description", 0.6),
                InfoColumn("supported_values", "Supported values", 0.4),
                InfoColumn("default", "Default", 20),
            ],
        )

    @classmethod
    def print_query_examples(cls, max_width=80):

        parameters = [
            {"name": "FRAMEWORKS", "example": "export FRAMEWORKS=FORGE"},
            {"name": "ALGORITHMS", "example": "export ALGORITHMS=RANDOM"},
            {"name": "CONFIGS", "example": "export CONFIGS=DEFAULT"},
            {"name": "RANDOM_TEST_COUNT", "example": "export RANDOM_TEST_COUNT='3,4,6'"},
        ]

        cls.print_formatted_parameters(
            parameters,
            max_width,
            columns=[
                InfoColumn("name", "Parameter", 25),
                InfoColumn("example", "Examples", 0.8),
            ],
        )

    @classmethod
    def print_formatted_parameters(cls, parameters, max_width: int, columns: List[InfoColumn]):

        fixed_width = sum([col.width for col in columns if isinstance(col.width, int)])
        for col in columns:
            if isinstance(col.width, float):
                col.width = int((max_width - fixed_width) * col.width)

        for param in parameters:
            for col in columns:
                param[col.name] = "\n".join(textwrap.wrap(param[col.name], width=col.width))

        table_data = [[param[column.name] for column in columns] for param in parameters]

        headers = [column.header for column in columns]
        print(tabulate(table_data, headers, tablefmt="grid"))
