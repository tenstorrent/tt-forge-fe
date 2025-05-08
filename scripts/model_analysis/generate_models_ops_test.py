# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
from loguru import logger
import argparse
from utils import create_python_package, run_precommit, remove_directory, filter_unique_operations, check_path
from unique_ops_utils import (
    generate_and_export_unique_ops_tests,
    extract_unique_op_tests_from_models,
    extract_existing_unique_ops_config,
)
from forge.tvm_unique_op_generation import generate_models_ops_test

# python scripts/model_analysis/generate_models_ops_test.py --tests_to_filter forge/test/models/pytorch/vision/hrnet/test_hrnet.py::test_hrnet_timm_pytorch --override_existing_ops &> generate_models_ops_test.log
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
    parser.add_argument(
        "--override_existing_ops",
        action="store_true",
        help=(
            "If set, it will extract unique ops configurations from existing generated models ops tests directory path(i.e forge/test/models_ops)"
            "and then combine it with unique ops configuration extracted for the list of models tests specified in the tests_to_filter and then generate models ops tests"
        ),
    )

    args = parser.parse_args()

    if args.override_existing_ops:
        assert (
            args.tests_to_filter is not None and args.tests_to_filter
        ), "List of models tests must be provided in tests_to_filter when override_existing_ops is enabled"

    models_ops_tests_directory_path = os.path.join(
        args.models_ops_test_output_directory_path, args.models_ops_test_package_name
    )

    model_output_dir_paths = generate_and_export_unique_ops_tests(
        test_directory_or_file_path=args.test_directory_or_file_path,
        unique_ops_output_directory_path=args.unique_ops_output_directory_path,
        extract_tvm_unique_ops_config=True,
        tests_to_filter=args.tests_to_filter,
    )

    existing_unique_ops_config = None
    if args.override_existing_ops:
        existing_unique_ops_config_file_path = os.path.join(
            args.unique_ops_output_directory_path, "existing_unique_ops_config_across_all_models_ops_tests.log"
        )
        existing_unique_ops_config = extract_existing_unique_ops_config(
            models_ops_tests_directory_path=models_ops_tests_directory_path,
            existing_unique_ops_config_file_path=existing_unique_ops_config_file_path,
        )

    unique_ops_config_file_path = os.path.join(
        args.unique_ops_output_directory_path, "extracted_unique_ops_config_across_all_models_ops_tests.log"
    )
    unique_operations_across_all_models_ops_test = extract_unique_op_tests_from_models(
        model_output_dir_paths=model_output_dir_paths,
        unique_ops_config_file_path=unique_ops_config_file_path,
        use_constant_value=False,
        convert_param_and_const_to_activation=True,
        existing_unique_ops_config=existing_unique_ops_config,
    )
    unique_operations_across_all_models_ops_test = filter_unique_operations(
        unique_operations=unique_operations_across_all_models_ops_test, ops_to_filter=args.ops_to_filter
    )
    if (check_path(models_ops_tests_directory_path) and not args.override_existing_ops) or (
        not check_path(models_ops_tests_directory_path)
    ):
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
