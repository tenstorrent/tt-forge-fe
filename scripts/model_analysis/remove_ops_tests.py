# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import argparse
from utils import create_python_package, run_precommit, remove_directory
from unique_ops_utils import extract_existing_unique_ops_config
from forge.tvm_unique_op_generation import generate_models_ops_test


def main():
    parser = argparse.ArgumentParser(description="Remove the generated unique ops tests for specific models")
    parser.add_argument(
        "--unique_ops_output_directory_path",
        default=os.path.join(os.getcwd(), "models_unique_ops_output"),
        required=False,
        help="Specify the output directory path for saving existing unique ops configs outputs",
    )
    parser.add_argument(
        "--models_ops_test_output_directory_path",
        default=os.path.join(os.getcwd(), "forge/test"),
        required=False,
        help="Specify the directory path of the generated models ops tests",
    )
    parser.add_argument(
        "--models_ops_test_package_name",
        default="models_ops",
        required=False,
        help="Specify the python package name of the generated models ops tests",
    )
    parser.add_argument(
        "--model_names",
        nargs="+",
        type=str,
        required=False,
        help="Specify the list of model names to which the generate models ops tests need to be removed",
    )

    args = parser.parse_args()

    models_ops_tests_directory_path = os.path.join(
        args.models_ops_test_output_directory_path, args.models_ops_test_package_name
    )

    existing_unique_ops_config_file_path = os.path.join(
        args.unique_ops_output_directory_path, "existing_unique_ops_config_across_all_models_ops_tests.log"
    )
    existing_unique_ops_config = extract_existing_unique_ops_config(
        models_ops_tests_directory_path=models_ops_tests_directory_path,
        existing_unique_ops_config_file_path=existing_unique_ops_config_file_path,
        remove_model_names=args.model_names,
    )
    remove_directory(directory_path=models_ops_tests_directory_path)
    create_python_package(
        package_path=args.models_ops_test_output_directory_path, package_name=args.models_ops_test_package_name
    )
    generate_models_ops_test(
        existing_unique_ops_config,
        models_ops_tests_directory_path,
    )
    run_precommit(directory_path=models_ops_tests_directory_path)


if __name__ == "__main__":
    main()
