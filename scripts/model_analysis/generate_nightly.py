# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
from loguru import logger
import argparse
from utils import create_python_package, run_precommit
from unique_ops_utils import generate_and_export_unique_ops_tests, extract_unique_op_tests_from_models
from forge.tvm_unique_op_generation import generate_nightly_tests


def main():
    parser = argparse.ArgumentParser(
        description="""Extract the unique ops configuration for the models present in the test_directory_or_file_path
        specified by the user and then extract the unique ops configuration across all the models and
        generate nightly tests inside the nightly_tests_directory_path specified by the user"""
    )

    parser.add_argument(
        "--test_directory_or_file_path",
        type=str,
        default=os.path.join(os.getcwd(), "forge/test/models/pytorch"),
        help="Specify the directory or file path containing models test with model_analysis pytest marker",
    )
    parser.add_argument(
        "--unique_ops_output_directory_path",
        default=os.path.join(os.getcwd(), "models_unique_ops_output"),
        required=False,
        help="Specify the output directory path for saving models unique op tests outputs(i.e xlsx and json files)",
    )
    parser.add_argument(
        "--nightly_tests_directory_path",
        default=os.path.join(os.getcwd(), "forge/test"),
        required=False,
        help="Specify the directory path for saving generated nightly tests",
    )
    parser.add_argument(
        "--nightly_test_package_name",
        default="nightly",
        required=False,
        help="Specify the python package name for saving generated nightly tests",
    )

    args = parser.parse_args()

    model_output_dir_paths = generate_and_export_unique_ops_tests(
        test_directory_or_file_path=args.test_directory_or_file_path,
        unique_ops_output_directory_path=args.unique_ops_output_directory_path,
        extract_tvm_unique_ops_config=True,
    )

    unique_ops_config_across_all_models_nightly_file_path = os.path.join(
        args.unique_ops_output_directory_path, "extracted_unique_op_config_across_all_models_nightly.log"
    )
    unique_operations_across_all_models_nightly = extract_unique_op_tests_from_models(
        model_output_dir_paths=model_output_dir_paths,
        unique_ops_config_file_path=unique_ops_config_across_all_models_nightly_file_path,
        use_constant_value=False,
    )
    create_python_package(package_path=args.nightly_tests_directory_path, package_name=args.nightly_test_package_name)
    generate_nightly_tests(
        unique_operations_across_all_models_nightly,
        os.path.join(args.nightly_tests_directory_path, args.nightly_test_package_name),
    )
    run_precommit(directory_path=os.path.join(args.nightly_tests_directory_path, args.nightly_test_package_name))


if __name__ == "__main__":
    main()
