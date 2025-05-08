# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import json
from loguru import logger
import subprocess
import pandas as pd
from typing import List, Optional, Dict, Any
import ast

import torch

from forge.tvm_unique_op_generation import Operation, NodeType, UniqueOperations
from forge.tensor import forge_dataformat_to_pytorch_dtype
from forge.python_codegen import forge_df_from_str, pytorch_df_from_str
from forge._C import DataFormat
from utils import (
    dump_logs,
    collect_all_model_analysis_test,
    extract_framework_from_test_file_path,
    extract_test_file_path_and_test_case_func,
    filter_tests,
    extract_models_ops_test_params,
    check_path,
)


def generate_and_export_unique_ops_tests(
    test_directory_or_file_path: str,
    unique_ops_output_directory_path: str,
    extract_tvm_unique_ops_config: bool = False,
    timeout: int = 1200,
    tests_to_filter: Optional[List[str]] = None,
):
    """
    Collect all the tests that doesn't contain skip_model_analysis marker in the test_directory_or_file_path specified by the user
    and then generate unique op test for all the collected test and return the list of directory path
    containing exported models unique op configuration as xlsx file
    """

    # Collect all the test that doesn't contain skip_model_analysis marker inside the test_directory_or_file_path specified by the user
    test_list = collect_all_model_analysis_test(test_directory_or_file_path, unique_ops_output_directory_path)

    assert test_list != [], f"No tests found in the {test_directory_or_file_path} path"

    test_list = filter_tests(test_list, tests_to_filter)

    # Create a dictonary contains framework as key and value as another dictonary containing model_name as key and list of test command as values
    framework_and_model_name_to_tests = {}
    for test in test_list:
        test_file_path, _ = extract_test_file_path_and_test_case_func(test)
        if test_file_path is None:
            logger.warning(f"Unable to extract test_file_path from the test {test}")
            continue
        model_name = test_file_path.split("/")[-1].replace(".py", "").replace("test_", "")
        framework_name = extract_framework_from_test_file_path(test_file_path)
        if framework_name in framework_and_model_name_to_tests.keys():
            if model_name in framework_and_model_name_to_tests[framework_name].keys():
                framework_and_model_name_to_tests[framework_name][model_name].append(test)
            else:
                framework_and_model_name_to_tests[framework_name][model_name] = [test]
        else:
            framework_and_model_name_to_tests[framework_name] = {model_name: [test]}

    # Generate unique op test for the all collected test and save the models unique ops test information in the unique_ops_output_directory_path
    model_output_dir_paths = []
    for framework_name, model_name_to_tests in framework_and_model_name_to_tests.items():
        for model_name, tests in model_name_to_tests.items():
            model_output_dir_path = os.path.join(unique_ops_output_directory_path, framework_name, model_name)
            os.makedirs(model_output_dir_path, exist_ok=True)
            model_output_dir_paths.append(model_output_dir_path)
            for test in tests:
                logger.info(f"Running the tests : {test}")
                test_file_path, test_case_name = extract_test_file_path_and_test_case_func(test)
                if test_file_path is None or test_case_name is None:
                    logger.warning(f"Unable to extract test_file_path and test_case_name from the test {test}")
                    continue
                framework = extract_framework_from_test_file_path(test_file_path)
                model_script_output_logs_dir_path = os.path.join(model_output_dir_path, "model_script_logs")
                test_case_name = test_case_name.replace("/", "-")
                test_log_file_path = os.path.join(model_script_output_logs_dir_path, test_case_name + ".log")
                try:
                    result = subprocess.run(
                        ["pytest", test, "-vss", "--no-skips", "--runxfail"],
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

                    message = ""
                    if result.stderr:
                        message += "STDERR: \n\n"
                        message += result.stderr
                    if result.stdout:
                        message += "STDOUT: \n\n"
                        message += result.stdout
                    if result.returncode != 0:
                        error_message = f"Error while running the pytest:"
                        error_message += message
                        logger.info(error_message)
                        dump_logs(test_log_file_path, error_message)
                    else:
                        dump_logs(test_log_file_path, message)
                        logger.info(f"Successfully generated and exported unique ops test")

                except subprocess.CalledProcessError as e:
                    error_message = f"Error while running the pytest:\n {e.output}"
                    logger.error(error_message)
                    dump_logs(test_log_file_path, error_message)

                except subprocess.TimeoutExpired as e:
                    error_message = f"Test timed out after {timeout} seconds"
                    logger.error(error_message)
                    dump_logs(test_log_file_path, error_message)

                except Exception as e:
                    error_message = f"An unexpected error occurred while running {test}: {e}"
                    logger.error(error_message)
                    dump_logs(test_log_file_path, error_message)

    return model_output_dir_paths


def extract_unique_op_tests_from_models(
    model_output_dir_paths: List[str],
    unique_ops_config_file_path: str,
    use_constant_value: bool = True,
    convert_param_and_const_to_activation: bool = False,
    existing_unique_ops_config: Optional[UniqueOperations] = None,
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

            if model_variant == "model_script_logs":
                continue

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
        models_operations,
        models_constants,
        use_constant_value=use_constant_value,
        convert_param_and_const_to_activation=convert_param_and_const_to_activation,
        existing_unique_operations=existing_unique_ops_config,
    )

    # Dump the extracted unique op configuration across all the model varaiants into log file.
    dump_logs(unique_ops_config_file_path, str(unique_operations))

    return unique_operations


def extract_existing_unique_ops_config(
    models_ops_tests_directory_path: str,
    existing_unique_ops_config_file_path: str,
    ops_to_filter: Optional[List[str]] = None,
):

    assert check_path(
        models_ops_tests_directory_path
    ), f"Provided models ops tests directory path {models_ops_tests_directory_path} doesn't exists!"

    models_ops_pytest_file_paths = [
        os.path.join(models_ops_tests_directory_path, f)
        for f in os.listdir(models_ops_tests_directory_path)
        if f.endswith(".py") and f.startswith("test_")
    ]

    op_count = 0
    models_operations = {}

    if ops_to_filter is not None and ops_to_filter:
        ops_to_filter = list([op_name.lower() for op_name in ops_to_filter])

    for models_ops_pytest_file_path in models_ops_pytest_file_paths:
        existing_op_name = models_ops_pytest_file_path.split("/")[-1].replace("test_", "").replace(".py", "")
        if ops_to_filter is not None and ops_to_filter and existing_op_name not in ops_to_filter:
            continue
        if not check_path(models_ops_pytest_file_path):
            logger.warning(f"The models ops tests file(i.e {models_ops_pytest_file_path}) doesn't exist!")
            continue
        unique_ops_configs = extract_models_ops_test_params(models_ops_pytest_file_path)
        for unique_ops_config in unique_ops_configs:
            model_variants = unique_ops_config["model_names"]
            for model_variant in model_variants:
                op_count = op_count + 1
                operand_types = unique_ops_config["operand_types"]
                operand_types = [NodeType.from_json(operand_type) for operand_type in operand_types]
                operand_dtypes = unique_ops_config["operand_dtypes"]
                new_operand_dtypes = []
                for operand_dtype in operand_dtypes:
                    if operand_dtype.startswith("forge.DataFormat."):
                        forge_df = DataFormat.from_json(operand_dtype.split(".")[-1])
                        new_operand_dtypes.append(forge_df_from_str(forge_df, ""))
                    elif operand_dtype.startswith("torch."):
                        new_operand_dtypes.append(pytorch_df_from_str(getattr(torch, operand_dtype.split(".")[-1]), ""))
                    else:
                        raise ValueError("Unhandled dataformat!")

                metadata = {}
                metadata["pcc"] = unique_ops_config["pcc"]
                metadata["forge_module_name"] = unique_ops_config["module_name"]

                max_int = unique_ops_config["max_int"]
                markers = unique_ops_config["markers"]

                if max_int is not None:
                    metadata["max_int"] = max_int
                if markers:
                    metadata["markers"] = markers

                metadata["model_variant_info"] = {"variant_name": model_variant}

                models_operations[op_count] = Operation(
                    function_name=unique_ops_config["forge_op_name"],
                    input_names=unique_ops_config["operand_names"],
                    args=unique_ops_config["op_args"],
                    input_shapes=unique_ops_config["operand_shapes"],
                    input_dtypes=new_operand_dtypes,
                    input_node_types=operand_types,
                    metadata=metadata,
                )

    # Extract unique operation configuration configuration across all the model variants
    existing_unique_operations = UniqueOperations.create_unique_operations(
        models_operations, {}, use_constant_value=False
    )

    dump_logs(existing_unique_ops_config_file_path, str(existing_unique_operations))

    return existing_unique_operations
