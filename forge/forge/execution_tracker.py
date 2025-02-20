# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import json
from enum import Enum
from loguru import logger
from typing import Union


class ExecutionPhase(Enum):
    """
    High-level phases of the execution process.
    """

    NOT_STARTED = "NOT_STARTED"  # didn't manage to pass TVM
    COMPILE_TVM = "COMPILE_TVM"  # generated Forge module
    COMPILE_FFE = "COMPILE_FFE"  # compiled upto split_graph
    COMPILE_MLIR = "COMPILE_MLIR"  # generated binary (flat buffer)
    EXECUTED_TTNN = "EXECUTED_TTNN"  # executed using TTNN (binary)
    PASSED = "PASSED"  # fully passed with PCC and related checks

    def __str__(self):
        return self.value


class ExecutionStage(Enum):
    """
    Detailed stages within each high-level execution phase.

    Each stage is linked to a high-level phase (ExecutionPhase) and describes
    a specific step in the execution process.
    """

    # Stage under COMPILE_TVM

    # 1. Generated Relay IRModule using tvm.relay.frontend.from_pytorch (compile_pytorch_for_forge)
    TVM_GENERATE_RELAY_IRMODULE = (
        ExecutionPhase.COMPILE_TVM,
        "TVM_GENERATE_RELAY_IRMODULE",
    )

    # 2. Flattened input and output using tvm.relay.op.contrib.flatten_IO (compile_pytorch_for_forge).
    TVM_FLATTEN_IO = (
        ExecutionPhase.COMPILE_TVM,
        "TVM_FLATTEN_IO",
    )

    # 3. Transformed the Relay IR in the run_relay_compile_passes function.
    TVM_RELAY_IR_TRANSFORM = (
        ExecutionPhase.COMPILE_TVM,
        "TVM_RELAY_IR_TRANSFORM",
    )

    # 4. Applied pattern callbacks in the run_forge_compile_passes function.
    TVM_PATTERN_CALLBACKS = (
        ExecutionPhase.COMPILE_TVM,
        "TVM_PATTERN_CALLBACKS",
    )

    # 5. Partitioned the Relay IR graph using the partition_for_forge function.
    TVM_GRAPH_PARTITIONING = (
        ExecutionPhase.COMPILE_TVM,
        "TVM_GRAPH_PARTITIONING",
    )

    # 6. Generated Forge Module  in generate_initial_graph function
    TVM_GENERATE_FORGE_MODULE = (
        ExecutionPhase.COMPILE_TVM,
        "TVM_GENERATE_FORGE_MODULE",
    )

    # Stage under COMPILE_FFE

    # 1. Generated the initial Forge graph (generate_initial_graph).
    FORGE_GENERATE_INITIAL_GRAPH = (
        ExecutionPhase.COMPILE_FFE,
        "FORGE_GENERATE_INITIAL_GRAPH",
    )

    # 2. Stage after the run_post_initial_graph_pass function.
    FORGE_POST_INIT = (ExecutionPhase.COMPILE_FFE, "FORGE_POST_INIT")

    # 3. Stage after the run_consteval_pass function.
    FORGE_CONSTEVAL = (ExecutionPhase.COMPILE_FFE, "FORGE_CONSTEVAL")

    # 4. Stage after the run_optimization_graph_passes function.
    FORGE_OPTIMIZE = (ExecutionPhase.COMPILE_FFE, "FORGE_OPTIMIZE")

    # 5. Stage after the run_post_optimize_decompose_graph_passes function.
    FORGE_POST_OPTIMIZE_DECOMP = (
        ExecutionPhase.COMPILE_FFE,
        "FORGE_POST_OPTIMIZE_DECOMP",
    )

    # 6. Stage after the run_autograd_pass function.
    FORGE_AUTOGRAD = (ExecutionPhase.COMPILE_FFE, "FORGE_AUTOGRAD")

    # 7. Stage after the run_post_autograd_pass function.
    FORGE_GRAD_DECOMP = (ExecutionPhase.COMPILE_FFE, "FORGE_GRAD_DECOMP")

    # 8. Stage after the run_pre_lowering_passes function.
    FORGE_PRE_LOWERING = (ExecutionPhase.COMPILE_FFE, "FORGE_PRE_LOWERING")

    # 9. Stage after the split_graph function.
    FORGE_GRAPH_SPLIT = (ExecutionPhase.COMPILE_FFE, "FORGE_GRAPH_SPLIT")

    # Stage under PASSED

    # 1. Compared the golden and framework output (compare_with_golden in forge/forge/verify/compare.py).
    COMPARE_WITH_GOLDEN = (
        ExecutionPhase.PASSED,
        "COMPARE_WITH_GOLDEN",
    )

    # 2. Performed verification (verify function in forge/forge/verify/verify.py).
    VERIFICATON = (ExecutionPhase.PASSED, "VERIFICATON")

    def __init__(self, phase, stage_name):
        self._phase = phase
        self._stage_name = stage_name

    @property
    def phase(self):
        """Returns the high-level phase associated with this stage."""
        return self._phase

    def __str__(self):
        return self._stage_name


def record_execution_phase_and_stage(execution_phase_or_stage: Union[ExecutionPhase, ExecutionStage]):
    """
    Record the execution phase and stage for the current test.

    This function updates a test report file with the execution phase and stage for the current test.
    The execution information is determined based on the provided enum member, which can be either
    a high-level phase (ExecutionPhase) or a detailed stage (ExecutionStage).

    The function follows these steps:
      1. Validate that the provided argument is a member of either ExecutionPhase or ExecutionStage.
      2. Determine the execution phase and stage values based on the type of enum.
         - If a detailed stage is provided, extract its associated high-level phase.
         - If a phase is provided, use it for both phase and stage.
      3. Retrieve the test report file path and current test identifier from environment variables.
      4. Read the existing test report (if available) and update or add the current test's entry.
      5. Write the updated test report back to the file.

    Parameters:
        execution_phase_or_stage (Union[ExecutionPhase, ExecutionStage]):
            An enum member representing either the execution phase or a more granular stage.
    """
    # Validate that the provided argument is an instance of the expected enum types.
    if not isinstance(execution_phase_or_stage, (ExecutionPhase, ExecutionStage)):
        logger.warning(
            f"Provided {execution_phase_or_stage} is not a member of the ExecutionPhase or ExecutionStage enum class"
        )
        return

    # Determine execution phase and stage based on the enum type.
    if isinstance(execution_phase_or_stage, ExecutionStage):
        # For a detailed stage, retrieve the associated high-level phase and the stage string.
        execution_phase = str(execution_phase_or_stage.phase)
        execution_stage = str(execution_phase_or_stage)
    else:
        # For a high-level phase, use the same value for both phase and stage.
        execution_phase = str(execution_phase_or_stage)
        execution_stage = str(execution_phase_or_stage)

    # Retrieve the test report file path from the environment, defaulting to "test_report.json" if not set.
    test_report_file_path = os.environ.get("PYTEST_REPORT_FILE_PATH", "test_report.json")

    # Retrieve the current test identifier from the environment.
    current_test = os.environ.get("CURRENT_TEST", None)
    if current_test is None:
        logger.warning("Unable to find current test")
        return

    # Initialize or update the test report.
    if os.path.exists(test_report_file_path):
        # Read the existing test report.
        with open(test_report_file_path, "r") as file:
            test_report = json.load(file)

        # Update the current test entry if it already exists.
        if current_test in test_report and isinstance(test_report[current_test], dict):
            test_report[current_test]["ExecutionPhase"] = execution_phase
            test_report[current_test]["ExecutionStage"] = execution_stage
        else:
            # Add a new entry for the current test.
            test_report[current_test] = {"ExecutionPhase": execution_phase, "ExecutionStage": execution_stage}
    else:
        # If the test report file does not exist, create a new report.
        test_report = {current_test: {"ExecutionPhase": execution_phase, "ExecutionStage": execution_stage}}

    # Write the updated test report back to the file with pretty-printed JSON.
    test_report_object = json.dumps(test_report, indent=4)
    with open(test_report_file_path, "w") as file:
        file.write(test_report_object)


def fetch_execution_phase_and_stage():
    """
    Retrieve the execution phase and stage for the current test from the test report file.

    The function looks up the test report file (default "test_report.json" or as specified by the
    environment variable "PYTEST_REPORT_FILE_PATH") and returns the test execution phase and stage for the test
    identified by the "CURRENT_TEST" environment variable.

    Returns:
        Tuple(str) or None: Return ExecutionPhase and  ExecutionStage if found, otherwise None.
    """

    # Get the test report file path from the environment variable or use default
    test_report_file_path = os.environ.get("PYTEST_REPORT_FILE_PATH", "test_report.json")

    # Get the current test identifier from the environment
    current_test = os.environ.get("CURRENT_TEST", None)
    if current_test is None:
        logger.warning("Unable to find current test")
        return None, None

    # Check if the test report file exists
    if os.path.exists(test_report_file_path):
        with open(test_report_file_path, "r") as file:
            test_report = json.load(file)

            # Check if the current test exists and has a recorded ExecutionPhase and ExecutionStage
            if (
                current_test in test_report.keys()
                and isinstance(test_report[current_test], dict)
                and "ExecutionPhase" in test_report[current_test].keys()
                and "ExecutionStage" in test_report[current_test].keys()
            ):
                return test_report[current_test]["ExecutionPhase"], test_report[current_test]["ExecutionStage"]
            logger.warning("Not able to fetch ExecutionPhase and ExecutionStage!")
            return None, None
    else:
        logger.warning(f"The test reports file path {test_report_file_path} doesn't exist!")
        return None, None
