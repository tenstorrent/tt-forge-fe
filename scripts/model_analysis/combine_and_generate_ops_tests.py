# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
from loguru import logger
import argparse
from utils import create_python_package, run_precommit, remove_directory, filter_unique_operations, check_path
from unique_ops_utils import (
    extract_unique_op_tests_from_models,
    extract_existing_unique_ops_config,
)
from forge.tvm_unique_op_generation import generate_models_ops_test


def main():
    parser = argparse.ArgumentParser(
        description="""Combine the unique ops configuration extracted across all the models and
        generate models ops test inside the models_ops_test_output_directory_path specified by the user"""
    )

    parser.add_argument(
        "--extracted_unique_ops_config_directory_path",
        type=str,
        required=True,
        help="Specify the directory containing extracted models unique ops configuration",
    )
    parser.add_argument(
        "--models_ops_test_output_directory_path",
        default=os.path.join(os.getcwd(), "forge/test"),
        required=False,
        help="Specify the directory path for saving generated models ops test",
    )
    parser.add_argument(
        "--models_ops_test_package_name",
        default="models_ops",
        required=False,
        help="Specify the python package name for saving generated models ops test",
    )

    args = parser.parse_args()

    models_ops_tests_directory_path = os.path.join(
        args.models_ops_test_output_directory_path, args.models_ops_test_package_name
    )

    unique_ops_config_file_path = os.path.join(
        args.extracted_unique_ops_config_directory_path, "extracted_unique_ops_config_across_all_models_ops_tests.log"
    )
    unique_operations_across_all_models_ops_test = extract_unique_op_tests_from_models(
        models_unique_ops_config_output_dir_path=args.extracted_unique_ops_config_directory_path,
        unique_ops_config_file_path=unique_ops_config_file_path,
        use_constant_value=False,
        convert_param_and_const_to_activation=True,
    )
    remove_directory(directory_path=models_ops_tests_directory_path)
    create_python_package(
        package_path=args.models_ops_test_output_directory_path, package_name=args.models_ops_test_package_name
    )
    generate_models_ops_test(
        unique_operations_across_all_models_ops_test,
        models_ops_tests_directory_path,
    )
    run_precommit(directory_path=models_ops_tests_directory_path)


if __name__ == "__main__":
    main()
