# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Test random graph configurations by utilizing Random Graph Generator Algorithm and targeting Forge and PyTorch frameworks

from enum import Enum
import pytest

from copy import copy
from dataclasses import dataclass
from typing import Generator, Tuple, List, Union

from forge.op_repo import OperatorParamNumber

from test.random.rgg import Framework
from test.random.rgg import Frameworks
from test.random.rgg import FrameworkTestUtils
from test.random.rgg import RandomGraphAlgorithm
from test.random.rgg import RandomizerConfig
from test.random.rgg import process_test

import os
import random
import textwrap

from tabulate import tabulate

from .rgg import get_randomizer_config_default


def get_random_seeds():
    if "RANDOM_TEST_COUNT" in os.environ:
        test_count = int(os.environ["RANDOM_TEST_COUNT"])
    else:
        test_count = 5
    tests_selected_indecies = []
    if "RANDOM_TESTS_SELECTED" in os.environ:
        tests_selected = os.environ["RANDOM_TESTS_SELECTED"]
        tests_selected = tests_selected.strip()
        if len(tests_selected) > 0:
            tests_selected_indecies = tests_selected.split(",")
            tests_selected_indecies = [int(i) for i in tests_selected_indecies]
    if len(tests_selected_indecies) > 0:
        # metafunc.parametrize("test_index", tests_selected_indecies)
        last_test_selected = max(tests_selected_indecies)
        if test_count < last_test_selected + 1:
            test_count = last_test_selected + 1
    else:
        tests_selected_indecies = range(test_count)

    test_rg = random.Random()
    if "RANDOM_TEST_SEED" in os.environ:
        test_rg.seed(int(os.environ["RANDOM_TEST_SEED"]))
    else:
        test_rg.seed(0)

    seeds = []
    # generate a new random seed for each test. Do it upfront so that
    # we can run any index in isolation
    for _ in range(test_count):
        seeds.append(test_rg.randint(0, 1000000))

    selected_tests = [(index, seeds[index]) for index in tests_selected_indecies]

    return selected_tests


class FrameworkBuilder:
    """Adjust repositories to test healthy operators"""

    NOT_IMPLEMENTED_FORGE = (
        # Unary operators
        "buffer",
        "pow",
        # Binary operators
        "less_equal",
        # Nary operators
        "interleave",
    )

    UNSTABLE_FORGE = (
        # Unary operators
        "exp",  # pcc?
        "sqrt",  # skip because it's failing for negative values
        "cumsum",  # bug
        "argmax",  # shape calc is wrong
        "logical_not",  # bug
        "dropout",  # pcc?
        "tilizer",  # bug
        # Binary operators
        "divide",  # bug
        "binary_stack",  # bug
        "power",  # occasionally fails
        "logical_and",  # bug
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
        "bitwise_not",
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
        # Binary operators
        # Nary operators
        # Other
        # "linear",
        "conv2d",  # skip until calc_input_shapes is properly implemented
    )

    @classmethod
    def healty_forge(cls):
        SKIP_OPERATORS = ()
        SKIP_OPERATORS += cls.NOT_IMPLEMENTED_FORGE
        SKIP_OPERATORS += cls.UNSTABLE_FORGE

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
        FrameworkTestUtils.allow_operators(framework, cls.UNSTABLE_FORGE)
        return framework

    @classmethod
    def not_implemented_forge(cls):
        framework = FrameworkTestUtils.copy_framework(Frameworks.FORGE.value, "Not implemented Forge")
        FrameworkTestUtils.allow_operators(framework, cls.NOT_IMPLEMENTED_FORGE)
        return framework

    @classmethod
    def healty_pytorch(cls):
        SKIP_OPERATORS = ()
        SKIP_OPERATORS += cls.NOT_IMPLEMENTED_PYTORCH
        SKIP_OPERATORS += cls.UNSTABLE_PYTORCH

        framework = FrameworkTestUtils.copy_framework(Frameworks.PYTORCH.value, "Healthy PyTorch", SKIP_OPERATORS)

        return framework

    @classmethod
    def unstable_pytorch(cls):
        framework = FrameworkTestUtils.copy_framework(Frameworks.PYTORCH.value, "Unstable PyTorch")
        FrameworkTestUtils.allow_operators(framework, cls.UNSTABLE_PYTORCH)
        return framework

    @classmethod
    def not_implemented_pytorch(cls):
        framework = FrameworkTestUtils.copy_framework(Frameworks.PYTORCH.value, "Not implemented PyTorch")
        FrameworkTestUtils.allow_operators(framework, cls.NOT_IMPLEMENTED_PYTORCH)
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
    config_name: str
    randomizer_config: RandomizerConfig

    def get_id(self):
        return f"{self.framework.template_name}_{self.config_name}"


class RGGConfiguraionProvider:

    ALL_TESTS = [
        RGGTestConfiguration(FrameworkBuilder.healty_forge(), "Default", RandomizerConfigBuilder.default()),
        RGGTestConfiguration(FrameworkBuilder.unstable_forge(), "Unstable", RandomizerConfigBuilder.default()),
        RGGTestConfiguration(
            FrameworkBuilder.not_implemented_forge(), "Not implemented", RandomizerConfigBuilder.default()
        ),
        RGGTestConfiguration(FrameworkBuilder.healty_pytorch(), "Default", RandomizerConfigBuilder.default()),
        RGGTestConfiguration(FrameworkBuilder.unstable_pytorch(), "Unstable", RandomizerConfigBuilder.default()),
        RGGTestConfiguration(
            FrameworkBuilder.not_implemented_pytorch(), "Not implemented", RandomizerConfigBuilder.default()
        ),
        RGGTestConfiguration(
            FrameworkBuilder.forge_fork_joins(), "Fork Joins", RandomizerConfigBuilder.forge_fork_joins()
        ),
        RGGTestConfiguration(FrameworkBuilder.forge_nary(), "Nary", RandomizerConfigBuilder.forge_nary()),
    ]

    @classmethod
    def get_tests(cls) -> Generator[Tuple[Framework, str, RandomizerConfig], None, None]:
        if "FRAMEWORKS" in os.environ:
            frameworks = os.environ["FRAMEWORKS"].lower().split(",")
        else:
            frameworks = None

        if "CONFIGS" in os.environ:
            config_names = os.environ["CONFIGS"].split(",")
        else:
            config_names = ["DEFAULT"]
        config_names = [config_name.lower() for config_name in config_names]

        for test in cls.ALL_TESTS:
            if not (frameworks is None or test.framework.template_name.lower() in frameworks):
                continue
            if not (config_names is None or test.config_name.lower() in config_names):
                continue
            yield pytest.param(test.framework, test.config_name, test.randomizer_config, id=test.get_id())


@pytest.mark.parametrize("test_index, random_seed", get_random_seeds())
@pytest.mark.parametrize("framework, config_name, randomizer_config", RGGConfiguraionProvider.get_tests())
def test_random_graph_algorithm(
    test_index, random_seed, test_device, framework, config_name: str, randomizer_config: RandomizerConfig
):
    process_test(
        config_name,
        test_index,
        random_seed,
        test_device,
        randomizer_config,
        graph_builder_type=RandomGraphAlgorithm,
        framework=framework,
    )


@dataclass
class InfoColumn:
    name: str
    header: str
    width: Union[int, float]


class InfoUtils:

    ALL_TESTS = RGGConfiguraionProvider.ALL_TESTS

    @classmethod
    def print_query_params(cls, max_width=80):
        print("Query parameters:")
        cls.print_query_values(max_width)
        print("Query examples:")
        cls.print_query_examples(max_width)

    @classmethod
    def print_query_values(cls, max_width=80):

        frameworks = [test.framework.template_name.upper() for test in cls.ALL_TESTS]
        frameworks = set(frameworks)
        frameworks = ", ".join(frameworks)

        config_names = [test.config_name.upper() for test in cls.ALL_TESTS]
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
