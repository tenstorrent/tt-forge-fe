# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Test random graph configurations by utilizing Random Graph Generator Algorithm and targeting PyBuda and PyTorch frameworks

from enum import Enum
import pytest

from copy import copy

from forge.op_repo import OperatorParamNumber

from test.random.rgg import Frameworks
from test.random.rgg import FrameworkTestUtils
from test.random.rgg import RandomGraphAlgorithm
from test.random.rgg import RandomizerConfig
from test.random.rgg import process_test

import os
import random

from .rgg import get_randomizer_config_default


@pytest.fixture
def randomizer_config():
    return get_randomizer_config_default()


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


class FrameworksHealthy(Enum):
    """Adjust repositories to test healthy operators"""

    @staticmethod
    def healty_pybuda():
        SKIP_OPERATORS = (
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
        )

        framework = FrameworkTestUtils.copy_framework(Frameworks.PYBUDA.value, "Healthy PyBuda", SKIP_OPERATORS)

        pow_operator = FrameworkTestUtils.copy_operator(framework, "pow")
        if pow_operator:
            pow_operator.forward_params = [
                # float exponent is currently not supported due to issue #2592
                # OperatorParamNumber("exponent", float, 0, 100),
                # OperatorParamNumber("exponent", int, 0, 100),
                OperatorParamNumber("exponent", int, 0, 4),  # pcc for higher numbers fails
            ]

        return framework

    @staticmethod
    def healty_pytorch():
        SKIP_OPERATORS = (
            "sqrt",  # skip because it's failing for negative values
            # "linear",
            "conv2d",  # skip until calc_input_shapes is properly implemented
        )

        framework = FrameworkTestUtils.copy_framework(Frameworks.PYTORCH.value, "Healthy PyTorch", SKIP_OPERATORS)

        return framework

    PYBUDA = healty_pybuda()
    PYTORCH = healty_pytorch()


class FrameworksCustom(Enum):
    """Adjust repositories to prepare custom framework configurations"""

    @staticmethod
    def pybuda_fork_joins():
        SKIP_OPERATORS = ()

        framework = FrameworkTestUtils.copy_framework(Frameworks.PYBUDA.value, "PyBuda fork joins", SKIP_OPERATORS)

        ALLOW_OPERATORS = (
            "relu",
            "tanh",
            "add",
            "matmul",
        )

        FrameworkTestUtils.allow_operators(framework, ALLOW_OPERATORS)

        return framework

    @staticmethod
    def pybuda_nary():
        SKIP_OPERATORS = ()

        framework = FrameworkTestUtils.copy_framework(Frameworks.PYBUDA.value, "PyBuda nary", SKIP_OPERATORS)

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

    PYBUDA_FORK_JOINS = pybuda_fork_joins()
    PYBUDA_NARY = pybuda_nary()


@pytest.mark.parametrize("test_index, random_seed", get_random_seeds())
@pytest.mark.parametrize(
    "framework",
    [
        FrameworksHealthy.PYBUDA.value,
    ],
)
def test_random_graph_algorithm_pybuda(
    test_index, random_seed, test_device, randomizer_config: RandomizerConfig, framework
):
    # adjust randomizer_config
    randomizer_config = copy(randomizer_config)
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

    process_test(
        "Default",
        test_index,
        random_seed,
        test_device,
        randomizer_config,
        graph_builder_type=RandomGraphAlgorithm,
        framework=framework,
    )


@pytest.mark.parametrize("test_index, random_seed", get_random_seeds())
@pytest.mark.parametrize(
    "framework",
    [
        FrameworksHealthy.PYTORCH.value,
    ],
)
def test_random_graph_algorithm_pytorch(
    test_index, random_seed, test_device, randomizer_config: RandomizerConfig, framework
):
    # adjust randomizer_config
    randomizer_config = copy(randomizer_config)
    # randomizer_config.debug_shapes = True
    # randomizer_config.verify_shapes = True

    # Uncomment the following randomizer_config values to override the default values
    # randomizer_config.dim_min = 4
    # randomizer_config.dim_max = 4
    # randomizer_config.op_size_per_dim_min = 4
    # # randomizer_config.op_size_per_dim_min = 16
    # randomizer_config.op_size_per_dim_max = 8
    # # randomizer_config.op_size_per_dim_max = 64
    # # randomizer_config.op_size_per_dim_max = 256
    # randomizer_config.microbatch_size_min = 1
    # randomizer_config.microbatch_size_max = 8
    # randomizer_config.num_of_nodes_min = 3
    # randomizer_config.num_of_nodes_max = 5
    # randomizer_config.num_fork_joins_max = 5

    process_test(
        "Default",
        test_index,
        random_seed,
        test_device,
        randomizer_config,
        graph_builder_type=RandomGraphAlgorithm,
        framework=framework,
    )


@pytest.mark.parametrize("test_index, random_seed", get_random_seeds())
@pytest.mark.parametrize(
    "framework",
    [
        FrameworksCustom.PYBUDA_FORK_JOINS.value,
    ],
)
def ttest_random_graph_algorithm_pybuda_fork_joins(
    test_index, random_seed, test_device, randomizer_config: RandomizerConfig, framework
):
    # adjust randomizer_config
    randomizer_config = copy(randomizer_config)
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

    process_test(
        "Fork Joins",
        test_index,
        random_seed,
        test_device,
        randomizer_config,
        graph_builder_type=RandomGraphAlgorithm,
        framework=framework,
    )


# @pytest.mark.xfail(reason="Nary operators are buggy")
@pytest.mark.parametrize("test_index, random_seed", get_random_seeds())
@pytest.mark.parametrize(
    "framework",
    [
        FrameworksCustom.PYBUDA_NARY.value,
    ],
)
def ttest_random_graph_algorithm_pybuda_nary(
    test_index, random_seed, test_device, randomizer_config: RandomizerConfig, framework
):
    # adjust randomizer_config
    randomizer_config = copy(randomizer_config)
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

    process_test(
        "Nary",
        test_index,
        random_seed,
        test_device,
        randomizer_config,
        graph_builder_type=RandomGraphAlgorithm,
        framework=framework,
    )
