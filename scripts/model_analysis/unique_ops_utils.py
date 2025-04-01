# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import json
from loguru import logger
import subprocess
import pandas as pd
from typing import List
import ast

import torch

from forge.tvm_unique_op_generation import Operation, NodeType, UniqueOperations
from utils import dump_logs, collect_all_model_analysis_test, run_pip_freeze


def generate_and_export_unique_ops_tests(
    test_directory_or_file_path: str,
    unique_ops_output_directory_path: str,
    extract_tvm_unique_ops_config: bool = False,
    timeout: int = 1200,
):
    """
    Collect all the tests that doesn't contain skip_model_analysis marker in the test_directory_or_file_path specified by the user
    and then generate unique op test for all the collected test and return the list of directory path
    containing exported models unique op configuration as xlsx file
    """

    # Collect all the test that doesn't contain skip_model_analysis marker inside the test_directory_or_file_path specified by the user
    test_list = collect_all_model_analysis_test(test_directory_or_file_path, unique_ops_output_directory_path)

    assert test_list != [], f"No tests found in the {test_directory_or_file_path} path"

    # Create a dictonary contains model_name as key and model tests(i.e include variant, task) as values
    model_name_to_tests = {}
    for test in test_list:
        model_name = test.split("::")[0].split("/")[-1].replace(".py", "").replace("test_", "")
        if model_name not in model_name_to_tests.keys():
            model_name_to_tests[model_name] = [test]
        else:
            model_name_to_tests[model_name].append(test)

    # Generate unique op test for the all collected test and save the models unique ops test information in the unique_ops_output_directory_path
    model_output_dir_paths = []
    for model_name, tests in model_name_to_tests.items():
        model_output_dir_path = os.path.join(unique_ops_output_directory_path, model_name)
        os.makedirs(model_output_dir_path, exist_ok=True)
        model_output_dir_paths.append(model_output_dir_path)
        for test in tests:
            logger.info(f"Running the tests : {test}")
            test_case_name = test.split("::")[-1]
            os.makedirs(os.path.join(model_output_dir_path, test_case_name), exist_ok=True)
            test_log_file_path = os.path.join(model_output_dir_path, test_case_name, "test.log")
            before_pip_freeze_file_path = os.path.join(model_output_dir_path, test_case_name, "before.log")
            after_pip_freeze_file_path = os.path.join(model_output_dir_path, test_case_name, "after.log")
            try:
                run_pip_freeze(before_pip_freeze_file_path)
                result = subprocess.run(
                    ["pytest", test, "-vss", "--no-skips"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=timeout,
                    env=dict(
                        os.environ,
                        FORGE_TVM_GENERATE_UNIQUE_OPS_TESTS="1" if not extract_tvm_unique_ops_config else "0",
                        FORGE_EXTRACT_TVM_UNIQUE_OPS_CONFIG="1" if extract_tvm_unique_ops_config else "0",
                        FORGE_DISABLE_REPORTIFY_DUMP="1",
                        FORGE_EXPORT_TVM_UNIQUE_OPS_CONFIG_DETAILS="1",
                        FORGE_EXPORT_TVM_UNIQUE_OPS_CONFIG_DETAILS_DIR_PATH=model_output_dir_path,
                    ),
                )
                run_pip_freeze(after_pip_freeze_file_path)
                message = ""
                if result.stderr:
                    message += "STDERR: \n\n"
                    message += result.stderr
                if result.stdout:
                    message += "STDOUT: \n\n"
                    message += result.stdout
                if result.returncode != 0:
                    error_message = f"Error while running the pytest:"
                    error_message+=message
                    logger.info(error_message)
                    dump_logs(test_log_file_path, error_message)
                else:
                    dump_logs(test_log_file_path, message)
                    logger.info(f"Successfully generated and exported unique ops test")

            except subprocess.CalledProcessError as e:
                error_message = f"Error while running the pytest:\n {e.output}"
                logger.error(error_message)
                dump_logs(test_log_file_path, error_message)
                run_pip_freeze(after_pip_freeze_file_path)

            except subprocess.TimeoutExpired as e:
                error_message = f"Test timed out after {timeout} seconds"
                logger.error(error_message)
                dump_logs(test_log_file_path, error_message)
                run_pip_freeze(after_pip_freeze_file_path)

            except Exception as e:
                error_message = f"An unexpected error occurred while running {test}: {e}"
                logger.error(error_message)
                dump_logs(test_log_file_path, error_message)
                run_pip_freeze(after_pip_freeze_file_path)

    return model_output_dir_paths


def extract_unique_op_tests_from_models(
    model_output_dir_paths: List[str], unique_ops_config_file_path: str, use_constant_value: bool = True
):
    """
    Extract unique op configuration across all the models which will avoid running the redudant
    op configuration again by using the exported unique op configuration test details and models metadata
    """

    # Dictionary to store all the operations found in model variants
    models_operations = {}
    unique_op_count = 0

    # Dictionary to store constants (name and tensor) used in the model variants
    models_constants = {}

    # Iterate through all provided model directories
    for model_output_dir_path in model_output_dir_paths:

        # Extract the model name from the directory path
        model_name = model_output_dir_path.split("/")[-1]

        # List all model variants in the directory
        model_variants = os.listdir(model_output_dir_path)

        # Process each model variant
        for model_variant in model_variants:

            model_variant_dir_path = os.path.join(model_output_dir_path, model_variant)

            # Look for `.xlsx` and `.json` file containing unique operation details and metadata
            model_variant_tvm_generated_unique_op_xslx_file_path = None
            model_variant_tvm_generated_unique_op_metadata_file_path = None
            for f in os.listdir(model_variant_dir_path):
                if f.endswith(".xlsx"):
                    model_variant_tvm_generated_unique_op_xslx_file_path = os.path.join(model_variant_dir_path, f)
                elif f.endswith(".json"):
                    model_variant_tvm_generated_unique_op_metadata_file_path = os.path.join(model_variant_dir_path, f)

            # Skip if either `.xlsx` or `.json` file is missing
            if (
                model_variant_tvm_generated_unique_op_xslx_file_path is None
                or model_variant_tvm_generated_unique_op_metadata_file_path is None
            ):
                continue

            # Read the `.xlsx` file contains model variant unique op configuration details
            model_variant_df = pd.read_excel(
                model_variant_tvm_generated_unique_op_xslx_file_path,
                header=0,
                usecols=[
                    "Op",
                    "Operand_Names",
                    "Operand_Shapes",
                    "Operand_Types",
                    "Operand_Dtypes",
                    "Args",
                    "Testfile",
                ],
            )

            # Read the `.json` file contains model variant metadata information
            with open(model_variant_tvm_generated_unique_op_metadata_file_path, "r") as json_file:
                model_variant_metadata = json.load(json_file)

            if use_constant_value:
                # Load model variants parameters and buffers as tensors from specified files
                named_parameters = torch.load(model_variant_metadata["named_params_file_name"])
                if model_variant_metadata["param_file_name"] is not None:
                    serialized_params = torch.load(model_variant_metadata["param_file_name"])
                    named_parameters.update(serialized_params)
                named_buffers = torch.load(model_variant_metadata["named_buffers_file_name"])
                named_parameters.update(named_buffers)

            # Process each row in the `.xlsx` file to extract operation configurations
            for index, row in model_variant_df.iterrows():
                row = row.to_dict()
                unique_op_count += 1

                operand_names = ast.literal_eval(row["Operand_Names"])
                operand_types = ast.literal_eval(row["Operand_Types"])
                operand_types = [NodeType.from_json(operand_type) for operand_type in operand_types]
                operand_shapes = ast.literal_eval(row["Operand_Shapes"])

                # Prepare metadata associated with the operation
                metadata = {}
                metadata["model_variant_info"] = {}
                metadata["model_variant_info"]["model_name"] = model_name
                metadata["model_variant_info"]["variant_name"] = model_variant_metadata["module_name"]
                metadata["model_variant_info"]["framework"] = model_variant_metadata["framework"]
                if not pd.isna(row["Testfile"]):
                    metadata["model_variant_info"]["Testfile"] = row["Testfile"]

                # Create an Operation object with op name, shape, nodetype, dtype, arguments and operation metadata
                models_operations[unique_op_count] = Operation(
                    function_name=row["Op"],
                    input_names=operand_names,
                    args=ast.literal_eval(row["Args"]),
                    input_shapes=operand_shapes,
                    input_dtypes=ast.literal_eval(row["Operand_Dtypes"]),
                    input_node_types=operand_types,
                    metadata=metadata,
                )

                if use_constant_value:
                    # Store tensor which has constant nodetype as operands
                    for operand_type, operand_name in zip(operand_types, operand_names):
                        if operand_type == NodeType.Constant:
                            models_constants[operand_name] = named_parameters[operand_name]

    # Extract unique operation configuration configuration across all the model variants
    unique_operations = UniqueOperations.create_unique_operations(
        models_operations, models_constants, use_constant_value=use_constant_value
    )

    # Dump the extracted unique op configuration across all the model varaiants into log file.
    dump_logs(unique_ops_config_file_path, str(unique_operations))

    return unique_operations
