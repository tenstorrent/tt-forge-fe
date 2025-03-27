# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import argparse
from loguru import logger
from utils import check_path
from models_ops_test_failure_update import extract_models_ops_test_params, update_params, read_file


def unformat_pytest_param_list(file_path: str):
    # Extract test parameters and their associated marker information from the priovided file.
    models_ops_test_params, marker_with_reason_and_params = extract_models_ops_test_params(
        file_path=file_path, return_marker_with_reason=True
    )

    if len(models_ops_test_params) == 0:
        logger.warning(f"There is no models ops tests params found in the {file_path} file")
        return

    # Update test parameters with respective marker and reason.
    new_models_ops_test_params = update_params(
        models_ops_test_params=models_ops_test_params,
        marker_with_test_config={},
        marker_with_reason_and_params=marker_with_reason_and_params,
    )

    # Read the entire content of the test file
    lines = read_file(file_path)

    new_lines = []  # Will store the updated file content
    is_pytest_params = False  # Flag to indicate if we are inside the old test parameter block

    # Process each line of the file
    for line in lines:
        # When encountering the marker "@pytest.mark.nightly_models_ops", insert the updated test parameters before it
        if "@pytest.mark.nightly_models_ops" in line:
            new_lines.append("forge_modules_and_shapes_dtypes_list = [\n")
            # Append each updated test parameter (indented for formatting)
            for test_param in new_models_ops_test_params:
                new_lines.append(f"\t{test_param},\n")
            new_lines.append("]\n")
            new_lines.append("\n")
            new_lines.append("\n")
            # Append the marker line itself
            new_lines.append(line)
            is_pytest_params = False  # End the parameter block insertion
        # Skip lines that are part of the old test parameter block
        elif "forge_modules_and_shapes_dtypes_list = [" in line or is_pytest_params:
            is_pytest_params = True
        else:
            # Retain lines that are not part of the test parameter block
            new_lines.append(line)

    # Write the updated file content back to the test file
    with open(file_path, "w") as file:
        file.writelines(new_lines)


def main():
    """
    This script is used to restore the original formatting of the pytest parameter list
    (i.e., `forge_modules_and_shapes_dtypes_list`) in models ops test files that may have been altered by Black.
    It accepts one or more file paths as command-line arguments, checks if each file exists using `check_path`,
    and then processes each file to update its parameter list formatting.

    Command-line Arguments:
        --file_paths: A list of paths to the models ops test files to process.
    """
    parser = argparse.ArgumentParser(
        description="Revert the black formating applied to tests parameters list(i.e forge_modules_and_shapes_dtypes_list) present in the models ops tests"
    )
    parser.add_argument("--file_paths", nargs="+", type=str, required=True, help="List of models ops tests file path")
    args = parser.parse_args()
    file_paths = args.file_paths
    for file_path in file_paths:
        if not check_path(file_path):
            continue
        unformat_pytest_param_list(file_path=file_path)


if __name__ == "__main__":
    main()
