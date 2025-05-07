# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
from loguru import logger
import argparse
from utils import create_python_package, run_precommit, remove_directory, filter_unique_operations
from unique_ops_utils import generate_and_export_unique_ops_tests, extract_unique_op_tests_from_models
from forge.tvm_unique_op_generation import generate_models_ops_test


def main():
    parser = argparse.ArgumentParser(
        description="""Extract the unique ops configuration for the models present in the test_directory_or_file_path
        specified by the user and then extract the unique ops configuration across all the models and
        generate models ops test inside the models_ops_test_output_directory_path specified by the user"""
    )

    parser.add_argument(
        "--test_directory_or_file_path",
        type=str,
        default=os.path.join(os.getcwd(), "forge/test/models"),
        help="Specify the directory or file path containing models test with model_analysis pytest marker",
    )
    parser.add_argument(
        "--unique_ops_output_directory_path",
        default=os.path.join(os.getcwd(), "models_unique_ops_output"),
        required=False,
        help="Specify the output directory path for saving models unique ops outputs(i.e xlsx and json files)",
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
    parser.add_argument(
        "--tests_to_filter",
        nargs="+",
        type=str,
        required=False,
        help=(
            "Only extract unique ops configurations and generate ops tests for the specified tests that match with collected model analysis tests "
            "(e.g., `forge/test/models/pytorch/vision/fpn/test_fpn.py`, "
            "`forge/test/models/pytorch/text/albert/test_albert.py`). "
            "By default, all collected model analysis tests are processed."
        ),
    )
    parser.add_argument(
        "--ops_to_filter",
        nargs="+",
        type=str,
        required=False,
        help=(
            "Only generate ops test for the list of provided forge ops names (e.g. `Conv2d`, `Add`)"
            "By default, every operator extracted across all the models will have tests generated."
        ),
    )

    args = parser.parse_args()

    model_output_dir_paths = generate_and_export_unique_ops_tests(
        test_directory_or_file_path=args.test_directory_or_file_path,
        unique_ops_output_directory_path=args.unique_ops_output_directory_path,
        extract_tvm_unique_ops_config=True,
        tests_to_filter=args.tests_to_filter,
    )

    unique_ops_config_across_all_models_ops_test_file_path = os.path.join(
        args.unique_ops_output_directory_path, "extracted_unique_ops_config_across_all_models_ops_test.log"
    )
    unique_operations_across_all_models_ops_test = extract_unique_op_tests_from_models(
        model_output_dir_paths=model_output_dir_paths,
        unique_ops_config_file_path=unique_ops_config_across_all_models_ops_test_file_path,
        use_constant_value=False,
    )
    unique_operations_across_all_models_ops_test = filter_unique_operations(
        unique_operations=unique_operations_across_all_models_ops_test, ops_to_filter=args.ops_to_filter
    )
    remove_directory(
        directory_path=os.path.join(args.models_ops_test_output_directory_path, args.models_ops_test_package_name)
    )
    create_python_package(
        package_path=args.models_ops_test_output_directory_path, package_name=args.models_ops_test_package_name
    )
    generate_models_ops_test(
        unique_operations_across_all_models_ops_test,
        os.path.join(args.models_ops_test_output_directory_path, args.models_ops_test_package_name),
    )
    run_precommit(
        directory_path=os.path.join(args.models_ops_test_output_directory_path, args.models_ops_test_package_name)
    )


if __name__ == "__main__":
    main()
