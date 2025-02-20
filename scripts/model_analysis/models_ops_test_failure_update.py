# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import ast
import re
from loguru import logger
import argparse
from utils import check_path, run_precommit
from typing import List, Dict


class ErrorMessageUpdater:
    """
    Class to handle error message updates by mapping known error messages
    to more descriptive ones or extracting relevant fatal error messages.
    """

    def __init__(self):
        self.error_message_updater_map = {
            "ValueError: Data mismatch -> AutomaticValueChecker (compare_with_golden)": "Data mismatch between framework output and compiled model output",
            "Fatal error": self.get_fatal_error_message,
        }

    def get_fatal_error_message(self, lines: List[str]):
        """
        Extracts and returns the first fatal error message found in the log lines.

        Args:
            lines (List[str]): List of log lines to search for fatal errors.

        Returns:
            str: The first fatal error message found, or an empty string if none is found.
        """
        fatal_error_message_list = [
            "Unsupported data type",
        ]
        for line in lines:
            if "fatal" in line.lower():
                for fatal_error in fatal_error_message_list:
                    if fatal_error in line:
                        return fatal_error
                return line
        return ""

    def update(self, error_message: str, lines: List[str]):
        """
        Updates the given error message based on predefined mappings.
        If the error message matches a key in the mapping, it is replaced
        with the corresponding value. If the mapping function is callable,
        it processes the log lines accordingly.

        Args:
            error_message (str): The original error message.
            lines (List[str]): Log file lines to provide additional context.

        Returns:
            str: The updated error message.
        """
        updated_error_message = ""
        for match_error_message in self.error_message_updater_map.keys():
            if match_error_message in error_message:
                if callable(self.error_message_updater_map[match_error_message]):
                    updated_error_message = self.error_message_updater_map[match_error_message](lines)
                else:
                    updated_error_message = self.error_message_updater_map[match_error_message]
                break
        if len(updated_error_message) == 0:
            updated_error_message = error_message
        return updated_error_message.replace("E  ", "").strip("\n").strip()


def read_file(file_path: str):
    """
    Reads a file and returns its content as a list of lines.

    Args:
        file_path (str): Path to the file.

    Returns:
        List[str]: List of lines in the file.
    """
    with open(file_path, "r") as file:
        lines = file.readlines()
    return lines


def extract_failed_models_ops_tests_with_failure_reason(log_files: List[str], models_ops_test_dir_path: str):
    """
    Extracts failed test cases and their corresponding failure reasons from pytest log files.

    The function processes the provided log files to identify failed tests that belong to a specified
    directory (models_ops_test_dir_path). It first scans the short test summary for any failed tests,
    and then it parses the detailed failure section to extract the error messages associated with those tests.

    Args:
        log_files (List[str]): A list of file paths to pytest log files.
        models_ops_test_dir_path (str): The directory path that contains model ops tests;
            used to filter which failures to consider.

    Returns:
        Dict[str, str]: A dictionary where the keys correponds to the failed test cases and
            the values are the corresponding failure reasons (error messages).
    """
    # Instantiate a helper object to update/refine error messages with additional context.
    error_message_updater = ErrorMessageUpdater()

    # Regular expression pattern to match the test function name
    test_func_pattern = r"^_+\s+(test_[a-zA-Z0-9_]+\[.*\])\s+_+$"

    # Dictionary to store the mapping from failed test cases to their failure messages.
    failed_models_ops_tests = {}

    # Define the maximum number of consecutive error lines to consider when assembling the error message.
    maximum_error_lines = 3

    for log_file in log_files:

        if check_path(log_file):

            lines = read_file(log_file)

            # Flag to indicate that the "short test summary info" section has started.
            collect_failed_models_ops_tests = False
            for line in lines:
                # Detect the start of the short test summary section.
                if "==== short test summary info ====" in line:
                    collect_failed_models_ops_tests = True

                # Once in the summary section, collect lines that indicate a test failure.
                elif collect_failed_models_ops_tests and "FAILED" in line:
                    # Remove the "FAILED" tag and trim whitespace/newlines to get the test identifier.
                    failed_test = line.replace("FAILED", "").strip("\n").strip()

                    # Only consider tests from the specified directory and ensure no duplicates.
                    if models_ops_test_dir_path in failed_test and failed_test not in failed_models_ops_tests:
                        failed_models_ops_tests[failed_test] = ""

            # If no failed tests were found in the current log file, log a warning and skip further processing.
            if len(failed_models_ops_tests) == 0:
                logger.warning(f"There is no failure in the {log_file}")
                continue

            # Sort the failed models ops test dict
            failed_models_ops_tests = dict(sorted(failed_models_ops_tests.items(), key=lambda kv: (kv[1], kv[0])))

            # Variables to track the current test case function and to mark the start of the detailed failure section.
            test_case_func = ""
            collect_failure_reason = False

            # Iterate over each line of the log file to extract detailed failure messages.
            for current_line, line in enumerate(lines):
                # Detect the beginning of the detailed failures section.
                if "==== FAILURES ====" in line:
                    collect_failure_reason = True

                elif collect_failure_reason:
                    # If we haven't yet identified the current test function, look for its name.
                    if len(test_case_func) == 0:
                        match = re.search(test_func_pattern, line)
                        if match:
                            # Capture the test function name (including any parameterized details).
                            test_case_func = match.group(1)
                    else:
                        # Once a test function has been identified, look for lines that start with "E  ",
                        # which indicate error message lines.
                        if line.startswith("E  "):
                            # Check if the next few lines (up to maximum_error_lines) all start with "E  ".
                            if all(
                                error_line.startswith("E  ")
                                for error_line in lines[current_line : current_line + maximum_error_lines]
                            ):
                                # Extract and clean up each of the consecutive error lines.
                                error_message = [
                                    error_line.replace("E  ", "").strip("\n").strip()
                                    for error_line in lines[current_line : current_line + maximum_error_lines]
                                ]
                                # Combine the multiple lines into a single error message string.
                                error_message = " ".join(error_message)
                            else:
                                # If not all the subsequent lines are error lines, use the current line as the error message.
                                error_message = line.replace("E  ", "").strip("\n").strip()

                            # Enhance the error message with additional context by using the updater helper.
                            error_message = error_message_updater.update(
                                error_message, lines[current_line + 1 : current_line + 11]
                            )

                            # For each recorded failed test, check if the current test function name is part of its identifier.
                            for failed_test in failed_models_ops_tests.keys():
                                if test_case_func in failed_test:
                                    failed_models_ops_tests[failed_test] = error_message

                            # Reset the test function tracker to allow extraction of the next failure.
                            test_case_func = ""
        else:
            logger.warning(f"Provided {log_file} path doesn't exists!!")

    return failed_models_ops_tests


def extract_failed_models_ops_tests_config(failed_models_ops_tests: Dict[str, str]):
    """
    Extracts and organizes configuration details from failed model operation tests.

    This function processes a dictionary mapping failed test identifiers to their error messages.
    It extracts configuration details such as the module name and operand shapes and dtypes from the test case
    identifier (using a regular expression) and organizes this information under the corresponding test file path.

    Args:
        failed_models_ops_tests (Dict[str, str]): A dictionary where each key is a string in the format
            "<test_file_path>::<test_case_identifier>[module_name-[shapes_and_dtypes]]" and each value is
            the associated error message.

    Returns:
        Dict[str, List[Dict[str, str]]]: A dictionary mapping test file paths to a list of configuration
            dictionaries. Each configuration dictionary contains the extracted error message, module name,
            and operand shapes and dtypes.
    """

    # Dictionary to store organized configuration information by test file path
    failed_models_ops_tests_info = {}

    # Regular expression to capture module name and operand shapes and dtypes from the test case identifier.
    regex = r"\[([^\[]+)-\[(.*)\]\]"

    for idx, (failed_test, error_message) in enumerate(failed_models_ops_tests.items()):

        failed_test_path, failed_test_cases = failed_test.split("::")

        # Attempt to extract module name and operand shapes/dtypes using the regular expression
        match = re.search(regex, failed_test_cases)

        # Initialize a configuration dictionary for the current failed test and save the error message in the configuration dictionary
        failed_test_config = {}
        failed_test_config["error_message"] = error_message

        # If the regex matches, extract the module name and operand shapes/dtypes
        if match:
            module_name = match.group(1)
            shapes_and_dtypes = match.group(2)
            failed_test_config["module_name"] = module_name
            failed_test_config["shapes_and_dtypes"] = f"[{shapes_and_dtypes}]"

        # Organize the configuration details under the appropriate test file path in the result dictionary
        if failed_test_path in failed_models_ops_tests_info.keys():
            failed_models_ops_tests_info[failed_test_path].append(failed_test_config)
        else:
            failed_models_ops_tests_info[failed_test_path] = [failed_test_config]

    return failed_models_ops_tests_info


def extract_models_ops_test_params(file_path: str):
    """
    Extracts model operation test parameters from a Python file.

    This function reads the content of the specified file and parses it into an Abstract Syntax Tree (AST).
    It then locates the assignment to the variable 'forge_modules_and_shapes_dtypes_list' and extracts
    test parameters from the assigned list. Test parameters can be specified either directly as tuples or as
    calls to pytest.param() where the first argument is a tuple. Each test parameter tuple is converted
    back to its source code representation using ast.unparse, and the resulting tuple elements are stored
    as a list of strings.

    Args:
        file_path (str): The path to the Python file containing the model operation test parameters.

    Returns:
        List[List[str]]: A list of test parameter tuples, with each tuple represented as a list of strings.
    """

    with open(file_path, "r") as file:
        content = file.read()

    # Parse the file content into an AST (Abstract Syntax Tree).
    tree = ast.parse(content)

    # List to hold the extracted test parameter tuples.
    models_ops_test_params = []

    # Iterate over each top-level node in the AST.
    for node in tree.body:
        # Check if the node is an assignment (e.g., a variable assignment).
        if isinstance(node, ast.Assign):
            # Loop through each target of the assignment (in case of multiple targets).
            for target in node.targets:
                # Look for the specific variable name 'forge_modules_and_shapes_dtypes_list'.
                if isinstance(target, ast.Name) and target.id == "forge_modules_and_shapes_dtypes_list":
                    # Iterate over each element in the assigned list.
                    for elt in node.value.elts:
                        # Initialize the variable to hold the test parameter tuple.
                        param_tuple = None
                        # Check if the element is a call (e.g., a call to pytest.param(...)).
                        if (
                            isinstance(elt, ast.Call)
                            and isinstance(elt.func, ast.Attribute)
                            and elt.func.attr == "param"
                        ):
                            # Extract the first argument from the pytest.param(...) call, if available.
                            param_tuple = elt.args[0] if elt.args else None
                        # Otherwise, check if the element is directly a tuple.
                        elif isinstance(elt, ast.Tuple):
                            param_tuple = elt
                        else:
                            # If the element doesn't match the expected patterns, skip it.
                            param_tuple = None

                        # If a valid test parameter tuple was found, unparse each element back to source code.
                        if param_tuple:
                            tuple_elements = [ast.unparse(item) for item in param_tuple.elts]
                            models_ops_test_params.append(tuple_elements)

    # Return the list of extracted test parameter tuples.
    return models_ops_test_params


def update_params(models_ops_test_params, failed_test_configs):
    """
    Update test parameters by adding xfail marks for tests that have failed.

    This function iterates over the list of model operation test parameters (each represented as a list of strings)
    and checks if the first two elements of the parameter (assumed to be the module name and shapes/dtypes)
    match any of the failed test configurations. For matching parameters, a new parameter string is created using
    a pytest.param call with an xfail mark and the associated error message. If no match is found, the original
    parameter string is retained.

    Args:
        models_ops_test_params (List[List[str]]): A list of test parameter tuples (each tuple is a list of strings).
        failed_test_configs (List[Dict[str, str]]): A list of dictionaries where each dictionary contains
            the keys "module_name", "shapes_and_dtypes", and "error_message" representing a failed test.

    Returns:
        List[str]: A list of updated test parameter strings. Each string is either a plain tuple or a pytest.param
            call with an xfail mark.
    """
    new_models_ops_test_params = []

    for param in models_ops_test_params:

        param_str = ", ".join(param)
        matched = False

        for config in failed_test_configs:

            if param[0] == config["module_name"] and param[1] == config["shapes_and_dtypes"]:
                error_message = config["error_message"]
                new_models_ops_test_params.append(
                    f'pytest.param(({param_str}), marks=[pytest.mark.xfail(reason="{error_message}")])'
                )
                matched = True
                break

        if not matched:
            new_models_ops_test_params.append(f"({param_str})")
    return new_models_ops_test_params


def update_models_ops_tests_failures(failed_models_ops_tests_info: Dict[str, List[Dict[str, str]]]):
    """
    Update test files by modifying their test parameter lists to include xfail marks for failed tests.

    This function processes each test file (specified by the keys in failed_models_ops_tests_info), extracts
    the current test parameters using extract_models_ops_test_params, and updates these parameters using the
    failed test configurations. The updated parameter list is then injected back into the file, replacing the
    old list.

    Args:
        failed_models_ops_tests_info (Dict[str, List[Dict[str, str]]]): A dictionary mapping test file paths to a
            list of failed test configuration dictionaries. Each configuration should contain "module_name",
            "shapes_and_dtypes", and "error_message".
    """

    for idx, (failed_test_path, failed_test_config) in enumerate(failed_models_ops_tests_info.items()):
        # Extract the current list of test parameters from the file.
        models_ops_test_params = extract_models_ops_test_params(failed_test_path)
        # Update the test parameters with failure information.
        new_models_ops_test_params = update_params(models_ops_test_params, failed_test_config)

        lines = read_file(failed_test_path)

        new_lines = []
        is_pytest_params = False  # Flag to track if we are within the test parameter block.

        for line in lines:
            # When the marker "@pytest.mark.push" is encountered, insert the updated test parameters before it.
            if "@pytest.mark.push" in line:
                new_lines.append("forge_modules_and_shapes_dtypes_list = [\n")
                # Append each updated test parameter.
                for test_param in new_models_ops_test_params:
                    new_lines.append(f"\t{test_param},\n")
                new_lines.append("]\n")
                new_lines.append("\n")
                new_lines.append("\n")
                # Append the line containing the marker.
                new_lines.append(line)
                is_pytest_params = False  # End of the parameter block.
            # Skip the old test parameter block lines.
            elif "forge_modules_and_shapes_dtypes_list = [" in line or is_pytest_params:
                is_pytest_params = True
            else:
                # Retain other lines unchanged.
                new_lines.append(line)

        # Write the updated content back to the test file.
        with open(failed_test_path, "w") as file:
            file.writelines(new_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Update models ops tests with xfail marks based on pytest log failures."
    )
    parser.add_argument("--log_files", nargs="+", type=str, required=True, help="List of pytest log files")
    parser.add_argument(
        "--models_ops_test_dir_path",
        type=str,
        default="forge/test/models_ops/",
        required=False,
        help="Specify the directory path that contains generated models ops tests",
    )

    args = parser.parse_args()
    log_files = args.log_files
    models_ops_test_dir_path = args.models_ops_test_dir_path

    run_precommit(directory_path=models_ops_test_dir_path)

    # Extract failed tests and their failure reasons from the provided log files.
    failed_models_ops_tests = extract_failed_models_ops_tests_with_failure_reason(
        log_files=log_files, models_ops_test_dir_path=models_ops_test_dir_path
    )

    if len(failed_models_ops_tests) == 0:
        log_files_str = ", ".join(log_files)
        logger.error(f"There is no failures in the provided {log_files_str} log files")

    # Extract and organize failure configuration details from the failed tests.
    failed_models_ops_tests_info = extract_failed_models_ops_tests_config(
        failed_models_ops_tests=failed_models_ops_tests
    )

    # Update the test files with the new parameters (including xfail marks) based on the failures.
    update_models_ops_tests_failures(failed_models_ops_tests_info=failed_models_ops_tests_info)

    run_precommit(directory_path=models_ops_test_dir_path)


if __name__ == "__main__":
    main()
