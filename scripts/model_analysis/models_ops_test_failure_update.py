# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import ast
import re
from loguru import logger
import argparse
from utils import check_path, run_precommit
from typing import Dict, Optional, List
from forge.utils import create_excel_file
import pandas as pd


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


def rename_marker(marker: str) -> str:
    """
    Standardizes the test marker name to either 'xfail' or 'skip'.
    
    Args:
        marker (str): The original marker name.
    
    Returns:
        str: Normalized marker ('xfail' or 'skip') if it matches known values, 
             otherwise returns the lowercase version of the input.
    """
    marker_lower = marker.lower()
    if marker_lower in ["xfail", "failed"]:
        return "xfail"
    elif marker_lower in ["skip", "skipped"]:
        return "skip"
    else:
        return marker_lower

def extract_test_file_path_and_test_case_func(test_case: str):
    """
    Extracts the test file path and test case function name from a given test case string.

    Args:
        test_case (str): A test case string in the format 'file_path::test_function'.

    Returns:
        Tuple[Optional[str], Optional[str]]: A tuple containing:
            - test_file_path (str or None): The extracted file path if present.
            - test_case_func (str or None): The extracted test case function name if present.
    """
    test_file_path = None
    test_case_func = None

    # Check if test case contains '::' separator
    if "::" in test_case:
        test_file_path, test_case_func = test_case.split("::", 1)  # Splitting into two parts

    return test_file_path, test_case_func

def extract_unique_test_file_paths(test_cases: List[str]) -> set:
    """
    Extracts unique test file paths from a list of test cases.

    Args:
        test_cases (List[str]): A list of test case strings, each in the format 'file_path::test_function'.

    Returns:
        set: A set of unique test file paths.
    """
    unique_test_file_paths = set()

    for test_case in test_cases:
        test_file_path, _ = extract_test_file_path_and_test_case_func(test_case)
        
        # Add file path to the set if it exists
        if test_file_path is not None:
            unique_test_file_paths.add(test_file_path)

    return unique_test_file_paths


def extract_models_ops_test_params(file_path: str, return_marker_with_reason: bool = False):
    """
    Extract test parameter tuples and (optionally) their associated marker information from a Python file.

    This function reads a Python file, parses its content into an Abstract Syntax Tree (AST), and searches for an assignment
    to the variable 'forge_modules_and_shapes_dtypes_list'. It extracts test parameters defined either as direct tuples or via a call
    to pytest.param(...). When pytest.param is used, the function also collects marker information (e.g., 'skip', 'xfail') and any
    associated reason provided with the marker.

    Args:
        file_path (str): The path to the Python file containing test parameter definitions.
        return_marker_with_reason (bool, optional): If True, returns a tuple containing both the list of test parameter tuples and a
            dictionary mapping marker names (with reasons) to lists of corresponding test parameter tuples.
            Defaults to False.

    Returns:
        list: A list of test parameter tuples, where each tuple is represented as a list of source code strings.
        dict (optional): If return_marker_with_reason is True, also returns a dictionary with marker information. The dictionary maps
            each marker name to another dictionary that maps a reason (or "No_Reason" if unspecified) to a list of test parameter tuples.
    """
    # Open and read the file content
    with open(file_path, "r") as file:
        content = file.read()

    # Parse the file content into an Abstract Syntax Tree (AST)
    tree = ast.parse(content)

    # List to hold the extracted test parameter tuples
    models_ops_test_params = []

    # Dictionary to hold marker names along with their reasons and associated test parameter tuples
    marker_with_reason_and_params = {}

    # Iterate over each top-level node in the AST
    for node in tree.body:
        # Check if the node is an assignment (e.g., variable assignment)
        if isinstance(node, ast.Assign):
            # Loop through each target of the assignment (handle cases with multiple targets)
            for target in node.targets:
                # Look for the specific variable name 'forge_modules_and_shapes_dtypes_list'
                if isinstance(target, ast.Name) and target.id == "forge_modules_and_shapes_dtypes_list":
                    # Iterate over each element in the assigned list
                    for elt in node.value.elts:
                        # Initialize the variable to hold the test parameter tuple
                        param_tuple = None
                        # Flag to indicate if marker(s) were found for this element
                        marks_found = False
                        # Dictionary to collect markers and their reasons for the current element
                        marks_dict = {}

                        # Check if the element is a call (e.g., a call to pytest.param(...))
                        if (
                            isinstance(elt, ast.Call)
                            and isinstance(elt.func, ast.Attribute)
                            and elt.func.attr == "param"
                        ):
                            # Extract the first argument from the pytest.param(...) call, if available
                            param_tuple = elt.args[0] if elt.args else None

                            # Look for a 'marks' keyword in the pytest.param call
                            for kw in elt.keywords:
                                if kw.arg == "marks":
                                    marks_found = True
                                    # The marks can be provided as a list/tuple or as a single marker
                                    mark_nodes = kw.value.elts if isinstance(kw.value, (ast.List, ast.Tuple)) else [kw.value]
                                    
                                    # Process each marker node
                                    for mark_node in mark_nodes:
                                        # Initialize variables for marker name and reason
                                        marker_name = None
                                        reason = None

                                        # Check if the marker is a function call (e.g., pytest.mark.skip(reason="..."))
                                        if isinstance(mark_node, ast.Call) and isinstance(mark_node.func, ast.Attribute):
                                            # Extract the marker name from the function call's attribute (e.g., 'skip' in pytest.mark.skip)
                                            marker_name = mark_node.func.attr
                                            
                                            # Look for a 'reason' keyword argument in the marker call
                                            for m_kw in mark_node.keywords:
                                                if m_kw.arg == "reason":
                                                    # Unparse the reason AST node back to source code and strip surrounding quotes
                                                    reason = ast.unparse(m_kw.value).strip().strip('"').strip("'")
                                                    break

                                        # If the marker is not a call but a simple attribute (e.g., pytest.mark.skip)
                                        elif isinstance(mark_node, ast.Attribute):
                                            marker_name = mark_node.attr

                                        # If a marker name was successfully extracted, record it in marks_dict.
                                        # Note: If the same marker appears multiple times, the latest reason (or None) will overwrite previous entries.
                                        if marker_name:
                                            marks_dict[marker_name] = reason


                        # If the element is not a pytest.param call, check if it's directly a tuple
                        elif isinstance(elt, ast.Tuple):
                            param_tuple = elt
                        else:
                            # If the element doesn't match expected patterns, skip processing it
                            param_tuple = None

                        # If a valid test parameter tuple was found, process it further
                        if param_tuple:
                            # Convert each element of the tuple back to source code (as strings)
                            tuple_elements = [ast.unparse(item) for item in param_tuple.elts]
                            # Append the test parameter tuple to the list
                            models_ops_test_params.append(tuple_elements)
                            
                            # If markers were found, update the marker information dictionary accordingly
                            if marks_found and marks_dict:
                                for marker_name, reason in marks_dict.items():
                                    # Use a default reason if none is provided
                                    if reason is None:
                                        reason = "No_Reason"
                                    # Update the dictionary for the current marker and reason
                                    if marker_name in marker_with_reason_and_params.keys():
                                        if reason in marker_with_reason_and_params[marker_name].keys():
                                            marker_with_reason_and_params[marker_name][reason].append(tuple_elements)
                                        else:
                                            marker_with_reason_and_params[marker_name][reason] = [tuple_elements]
                                    else:
                                        marker_with_reason_and_params[marker_name] = {reason: [tuple_elements]}

    # Return both test parameters and marker info if requested
    if return_marker_with_reason:
        return models_ops_test_params, marker_with_reason_and_params

    # Otherwise, return only the list of extracted test parameter tuples
    return models_ops_test_params


def extract_test_cases_and_status(
    log_files: List[str],
    target_dirs: Optional[List[str]] = None,
    target_statuses: Optional[List[str]] = None
) -> Dict[str, Dict[str, str]]:
    """
    Extract test cases and their statuses from provided log files.

    This function processes each log file and extracts test case names along with their statuses 
    using a regular expression pattern. It supports filtering by target directories paths and target statuses.
    The extraction halts upon encountering the "==== FAILURES ====" marker in a log file.

    Args:
        log_files (List[str]): A list of file paths to the log files that need to be processed.
        target_dirs (Optional[List[str]]): A list of directory paths to filter test cases. 
            Only test cases whose paths contain one of these directory paths will be included.
            Defaults to None, which means no directory filtering is applied.
        target_statuses (Optional[List[str]]): A list of statuses to filter test cases. 
            Only test cases with one of these statuses will be included.
            Defaults to None, meaning no status filtering is applied.

    Returns:
        Dict[str, Dict[str, str]]: A dictionary where keys are test statuses (e.g., 'PASSED', 'FAILED', 'SKIP', 'XFAIL')
        and values are dictionaries mapping test case names (as strings) to their associated reason (as a string).
        An empty string is used as a placeholder for the reason if none is provided.
    """
    # Initialize a dictionary to map statuses to their corresponding test cases and reasons.
    status_to_tests_and_reason = {}

    # Compile a regular expression pattern to capture the test case and status from each line.
    # The pattern explanation:
    #   ^(.*?)             - Capture any characters from the beginning as the test case name.
    #   \s+                - One or more whitespace characters.
    #   (PASSED|FAILED|SKIPPED|XFAIL) - Capture one of the specified statuses.
    #   .*                 - Followed by any characters (the remainder of the line).
    pattern = re.compile(r"^(.*?)\s+(PASSED|FAILED|SKIPPED|XFAIL).*")
    
    # Process each log file provided in the log_files list.
    for log_file in log_files:
        # Check if the log file exists using the check_path function.
        if check_path(log_file):
            # Read all lines from the file using the read_file function.
            lines = read_file(log_file)
            for line in lines:
                # Attempt to match the line with the regex pattern.
                match = pattern.match(line)
                if match:
                    # Extract the test case name and status from the regex match groups.
                    test_case = match.group(1).strip()
                    status = match.group(2).strip()
                    
                    # convert 'SKIPPED' to 'SKIP'.
                    if status == "SKIPPED":
                        status = "SKIP"

                    # Filter by target directories if provided.
                    # Include the test case if target_dirs is None or if it contains any specified directory.
                    if target_dirs is None or any(dir_path in test_case for dir_path in target_dirs):
                        # Filter by target statuses if provided.
                        # Include the test case if target_statuses is None or if the status is in the provided list.
                        if target_statuses is None or status in target_statuses:
                            # If the status is already in the dictionary, add the test case to its inner dictionary.
                            if status in status_to_tests_and_reason:
                                status_to_tests_and_reason[status][test_case] = ""
                            else:
                                # Otherwise, create a new entry for this status with the test case.
                                status_to_tests_and_reason[status] = {test_case: ""}
                
                # Stop processing further lines once the failures section is reached.
                if "==== FAILURES ====" in line:
                    break
        else:
            # Log a warning if the provided log file path does not exist.
            logger.warning(f"Provided path {log_file} does not exist!")
    
    return status_to_tests_and_reason


def update_reason_for_xfailed_skipped_test(status_to_tests_and_reason: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    """
    Update the reasons for test cases marked as 'XFAIL' or 'SKIP' based on marker details extracted from test files.

    This function updates the given mapping of test statuses to test case strings and their associated reasons.
    It works as follows:
      1. Collects all test case strings from both 'SKIP' and 'XFAIL' entries.
      2. Extracts unique test file paths from these test case strings.
      3. For each unique test file, it extracts marker information (marker name, reason, and associated parameters)
         using the `extract_models_ops_test_params` function.
      4. For each marker found in the test file, if the corresponding test case matches the parameters from the marker,
         update the test case's reason in the input dictionary.

    Args:
        status_to_tests_and_reason (Dict[str, Dict[str, str]]): 
            A dictionary mapping statuses (e.g., 'SKIP', 'XFAIL') to dictionaries of test case strings and their reasons.
    
    Returns:
        Dict[str, Dict[str, str]]: The updated dictionary with reasons for test cases modified based on the marker data.
    """
    test_cases = []  # Accumulate test case strings for 'SKIP' and 'XFAIL'

    # Collect test cases for 'SKIP' status if present.
    if "SKIP" in status_to_tests_and_reason.keys():
        test_cases.extend(status_to_tests_and_reason["SKIP"])
    
    # Collect test cases for 'XFAIL' status if present.
    if "XFAIL" in status_to_tests_and_reason.keys():
        test_cases.extend(status_to_tests_and_reason["XFAIL"])
    
    # Remove duplicate test cases.
    test_cases = list(set(test_cases))
    
    # Extract unique file paths from the collected test case strings.
    unique_test_file_paths = extract_unique_test_file_paths(test_cases=test_cases)
    
    # For each unique test file path, extract marker information along with reasons.
    for test_file_path in unique_test_file_paths:
        # The function returns a tuple; we are interested in the marker information dictionary.
        _, marker_with_reason_and_params = extract_models_ops_test_params(
            file_path=test_file_path, 
            return_marker_with_reason=True
        )
        # Iterate over each marker (e.g., 'skip', 'xfail') and its associated reason and parameter list.
        for marker, reason_with_params in marker_with_reason_and_params.items():
            for reason, params in reason_with_params.items():
                for param in params:
                    # Check if the uppercase version of the marker exists in our input dictionary.
                    if marker.upper() in status_to_tests_and_reason.keys():
                        # Iterate through test cases for this marker.
                        for test in status_to_tests_and_reason[marker.upper()].keys():
                            # If both parts of the test parameter (e.g., module and function names) are found in the test string,
                            # update the reason for that test case.
                            if param[0] in test and param[1] in test:
                                status_to_tests_and_reason[marker.upper()][test] = reason
                                break  # Once updated, no need to check further for this test case.
    return status_to_tests_and_reason

def extract_xfailed_and_skipped_tests_with_reason(log_files: List[str], models_ops_test_dir_path: str) -> Dict[str, Dict[str, str]]:
    """
    Extract and update test cases with statuses 'XFAIL' and 'SKIP' from log files, including detailed reasons.

    This function performs the following steps:
      1. Uses `extract_test_cases_and_status` to retrieve test cases and their statuses from the provided log files,
         filtering by the target directory paths and statuses ('XFAIL' and 'SKIP').
      2. If any test cases are found, it updates these entries with detailed reason information by calling
         `update_reason_for_xfailed_skipped_test`.

    Args:
        log_files (List[str]): A list of log file paths containing test execution information.
        models_ops_test_dir_path (str): The directory path to filter test cases (typically where the tests reside).

    Returns:
        Dict[str, Dict[str, str]]: A dictionary mapping statuses ('XFAIL', 'SKIP') to dictionaries that map test case
            strings to their corresponding reasons.
    """
    # Extract test cases with statuses 'XFAIL' and 'SKIP' from the log files,
    # filtering for those test cases within the specified directory.
    status_to_tests_and_reason = extract_test_cases_and_status(
        log_files=log_files, 
        target_dirs=[models_ops_test_dir_path], 
        target_statuses=["XFAIL", "SKIP"]
    )
    
    # If test cases were found, update them with the marker-based reason information.
    if len(status_to_tests_and_reason) != 0:
        status_to_tests_and_reason = update_reason_for_xfailed_skipped_test(
            status_to_tests_and_reason=status_to_tests_and_reason
        )
    
    return status_to_tests_and_reason


def extract_failed_tests_with_failure_reason(log_files: List[str], target_dirs: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Extract failed test cases along with their failure reasons from pytest log files.

    This function processes one or more pytest log files to identify failed tests and extract their 
    corresponding error messages. It operates in two main phases:
    
    1. Short Test Summary Extraction:
       - Scans each log file for the "short test summary info" section.
       - Identifies test cases marked as FAILED and filters them based on the provided target 
         directories (if any).
       - Stores these failed test cases in a dictionary with an empty string as a placeholder 
         for the failure reason.

    2. Detailed Failure Reason Extraction:
       - After the summary, the function parses the "==== FAILURES ====" section.
       - It uses a regular expression pattern to locate the test function names within the failure details.
       - For lines starting with "E  " (indicating error messages), it collects up to a specified maximum 
         number of consecutive error lines to form a comprehensive error message.
       - An external helper, `ErrorMessageUpdater`, is used to further refine or update the error message 
         with additional context from subsequent lines.
       - The refined error message is then matched with the corresponding failed test case (using substring 
         checks) and updated in the dictionary.

    Args:
        log_files (List[str]): A list of file paths to pytest log files.
        target_dirs (Optional[List[str]], optional): A list of directory paths to filter the failed tests.
            Only test cases whose identifiers include one of these directory strings will be processed.
            If None, all failed tests from the log files are considered.

    Returns:
        Dict[str, str]: A dictionary mapping each failed test case (as a string) to its corresponding 
        failure reason (error message). If a test case's error message could not be determined, its value 
        remains an empty string.
    """
    # Instantiate the helper object for refining error messages with additional context.
    error_message_updater = ErrorMessageUpdater()

    # Regular expression to capture the test function name from lines in the detailed failure section.
    test_func_pattern = r"^_+\s+(test_[a-zA-Z0-9_]+\[.*\])\s+_+$"

    # Dictionary to store mapping from each failed test case to its failure message.
    failed_tests_with_reason: Dict[str, str] = {}

    # Define the maximum number of consecutive error lines to consider for assembling a complete error message.
    maximum_error_lines = 3

    # Process each log file provided.
    for log_file in log_files:
        if check_path(log_file):
            # Read all lines from the log file.
            lines = read_file(log_file)

            # Flag indicating the start of the "short test summary info" section.
            collect_failed_tests = False

            # First pass: Extract test cases marked as FAILED from the summary section.
            for line in lines:
                # Start collecting when the short test summary section begins.
                if "==== short test summary info ====" in line:
                    collect_failed_tests = True

                # Within the summary section, identify lines that indicate a test failure.
                elif collect_failed_tests and "FAILED" in line:
                    # Remove the "FAILED" tag and trim any whitespace/newlines.
                    failed_test = line.replace("FAILED", "").strip("\n").strip()

                    # If target_dirs filter is provided, only include tests that match the directory criteria.
                    if target_dirs is None or any(dir_path in failed_test for dir_path in target_dirs):
                        failed_tests_with_reason[failed_test] = ""

            # If no failed tests were found in the current log file, log a warning and skip detailed extraction.
            if len(failed_tests_with_reason) == 0:
                logger.warning(f"There is no failure in the {log_file}")
                continue

            # Optionally sort the dictionary (here by value and then key) for consistency.
            failed_tests_with_reason = dict(sorted(failed_tests_with_reason.items(), key=lambda kv: (kv[1], kv[0])))

            # Initialize variables for the detailed extraction phase.
            test_case_func = ""  # To store the current test function name.
            collect_failure_reason = False  # Flag indicating that the detailed failure section has begun.

            # Second pass: Extract detailed failure messages from the "==== FAILURES ====" section.
            for current_line, line in enumerate(lines):
                # Detect the start of the detailed failures section.
                if "==== FAILURES ====" in line:
                    collect_failure_reason = True

                elif collect_failure_reason:
                    # If the test function name hasn't been captured yet, try to extract it.
                    if len(test_case_func) == 0:
                        match = re.search(test_func_pattern, line)
                        if match:
                            # Capture the test function name (including parameterized details).
                            test_case_func = match.group(1)
                    else:
                        # When the test function is identified, look for lines starting with "E  " which denote error messages.
                        if line.startswith("E  "):
                            # Check if subsequent lines (up to maximum_error_lines) are also error lines.
                            if all(
                                error_line.startswith("E  ")
                                for error_line in lines[current_line : current_line + maximum_error_lines]
                            ):
                                # Extract and clean each of the consecutive error lines.
                                error_message = [
                                    error_line.replace("E  ", "").strip("\n").strip()
                                    for error_line in lines[current_line : current_line + maximum_error_lines]
                                ]
                                # Combine these lines into a single error message string.
                                error_message = " ".join(error_message)
                            else:
                                # If not all subsequent lines are error lines, use the current line as the error message.
                                error_message = line.replace("E  ", "").strip("\n").strip()

                            # Update/enhance the error message with additional context from the following lines.
                            error_message = error_message_updater.update(
                                error_message, lines[current_line + 1 : current_line + 11]
                            )

                            # Match the current test function with each failed test from the summary.
                            for failed_test in failed_tests_with_reason.keys():
                                if test_case_func in failed_test:
                                    # Update the failure reason for the matching test case.
                                    failed_tests_with_reason[failed_test] = error_message

                            # Reset the test function tracker for processing the next failure.
                            test_case_func = ""
        else:
            logger.warning(f"Provided {log_file} path doesn't exist!")

    return failed_tests_with_reason


def update_params(models_ops_test_params, marker_with_test_config, marker_with_reason_and_params):
    """
    Update test parameters by appending marker configuration and reason information.

    This function iterates over each test parameter tuple in `models_ops_test_params` and updates it
    based on two sources of marker data:
    
      1. `marker_with_test_config`: A dictionary mapping marker names to lists of configuration dictionaries.
         Each configuration is expected to contain:
            - "module_name": the module name associated with the test parameter.
            - "shapes_and_dtypes": the corresponding shapes and dtypes string.
            - "reason": the reason to be applied for the marker.
         If a test parameter matches a configuration (i.e. its module name and shapes_and_dtypes match the config),
         a marker string (with reason) is created and appended.
      
      2. `marker_with_reason_and_params`: Additional marker information, where for each marker, a set of reasons
         and associated parameter tuples is provided. If the current test parameter matches one of these tuples,
         an additional marker string is appended. If the reason is not "No_Reason", it is included in the marker string.

    The updated test parameters are formatted as strings using the pytest.param syntax when marker information is present,
    otherwise they remain as a simple tuple string.

    Args:
        models_ops_test_params: A list of test parameter tuples (each tuple is typically a list of strings) that define the tests.
        marker_with_test_config: A dictionary where keys are marker names (e.g., "skip", "xfail") and values are lists of
                                   configuration dictionaries. Each configuration dictionary must include "module_name",
                                   "shapes_and_dtypes", and "reason".
        marker_with_reason_and_params: Additional marker information containing reason details and associated parameter tuples.
                                         This is typically structured as an iterable of pairs where each pair consists of a marker
                                         name and a nested structure mapping reasons to lists of parameter tuples.

    Returns:
        A list of strings representing the updated test parameter definitions, with marker annotations added where applicable.
    """
    new_models_ops_test_params = []  # List to store updated test parameter definitions

    # Process each test parameter tuple
    for param in models_ops_test_params:
        # Convert the test parameter tuple into a comma-separated string representation
        param_str = ", ".join(param)

        markers_str = []  # List to collect marker strings for this parameter

        # Check marker configurations from marker_with_test_config
        for marker, configs in marker_with_test_config.items():
            for config in configs:
                # If the test parameter matches the configuration based on module_name and shapes_and_dtypes...
                if param[0] == config["module_name"] and param[1] == config["shapes_and_dtypes"]:
                    reason = config["reason"]
                    # Append the marker with its reason using the rename_marker helper function
                    markers_str.append(f'pytest.mark.{rename_marker(marker)}(reason="{reason}")')
                    break  # Stop processing further configs for this marker if a match is found
            if len(markers_str) != 0:
                break  # If a marker is already found, exit the outer loop

        # Process additional marker reason information if provided
        if len(marker_with_reason_and_params) != 0:
            for marker_name, reason_with_params in marker_with_reason_and_params.items():
                for reason, params_list in reason_with_params.items():
                    for param_tuple in params_list:
                        # Check if the current test parameter matches the parameter tuple from marker reason info
                        if param_tuple[0] == param[0] and param_tuple[1] == param[1]:
                            # Build an additional marker string. Note: using marker_name might be intended instead of marker.
                            additional_param_str = f'pytest.mark.{rename_marker(marker_name)}'
                            if reason != "No_Reason":
                                additional_param_str += f'(reason="{reason}")'
                            markers_str.append(additional_param_str)
        
        # Format the updated test parameter with marker annotations if any markers were added.
        if len(markers_str) != 0:
            # Join all marker strings into a single comma-separated string
            markers_str = ", ".join(markers_str)
            new_models_ops_test_params.append(f"pytest.param(({param_str}), marks=[{markers_str}])")
        else:
            # If no markers apply, retain the original test parameter tuple format.
            new_models_ops_test_params.append(f"({param_str})")
    
    return new_models_ops_test_params


def update_models_ops_tests(models_ops_test_update_info: Dict[str, Dict[str, List[Dict[str, str]]]]):
    """
    Update test files with new test parameters that include marker configurations and failure reasons.

    For each test file specified in the `models_ops_test_update_info` dictionary, this function:
    
      1. Extracts the current test parameters and marker reason information by calling `extract_models_ops_test_params`
         with `return_marker_with_reason=True`.
      2. Removes any existing "skip" or "xfail" entries from the marker reason information.
      3. Calls `update_params` to generate updated test parameter definitions that include the correct marker annotations.
      4. Reads the test file's content, locates the block containing the test parameter definitions (identified by the variable
         "forge_modules_and_shapes_dtypes_list"), and replaces that block with the updated definitions.
      5. Writes the updated content back to the file, effectively updating the tests in place.

    Args:
        models_ops_test_update_info (Dict[str, Dict[str, List[Dict[str, str]]]]): 
            A dictionary where each key is a test file path and each value is a dictionary mapping marker names to a list
            of configuration dictionaries. Each configuration dictionary should include information such as "module_name",
            "shapes_and_dtypes", and "reason".

    Returns:
        None. The function updates the test files directly.
    """
    # Iterate over each test file path and its corresponding marker test configuration
    for idx, (test_file_path, marker_with_test_config) in enumerate(models_ops_test_update_info.items()):
        # Extract current test parameters and marker reason information from the test file
        models_ops_test_params, marker_with_reason_and_params = extract_models_ops_test_params(
            test_file_path, return_marker_with_reason=True
        )

        # Remove 'skip' and 'xfail' markers from marker reason information if they exist
        if "skip" in marker_with_reason_and_params.keys():
            marker_with_reason_and_params.pop("skip")
        if "xfail" in marker_with_reason_and_params.keys():
            marker_with_reason_and_params.pop("xfail")

        # Generate updated test parameter definitions with the new marker information
        new_models_ops_test_params = update_params(
            models_ops_test_params, marker_with_test_config, marker_with_reason_and_params
        )

        # Read the entire content of the test file
        lines = read_file(test_file_path)

        new_lines = []         # Will store the updated file content
        is_pytest_params = False  # Flag to indicate if we are inside the old test parameter block

        # Process each line of the file
        for line in lines:
            # When encountering the marker "@pytest.mark.push", insert the updated test parameters before it
            if "@pytest.mark.push" in line:
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
        with open(test_file_path, "w") as file:
            file.writelines(new_lines)

    
def create_report(report_file_path: str, 
                  failed_models_ops_tests: Dict[str, str], 
                  status_to_tests_and_reason: Optional[Dict[str, Dict[str, str]]] = None):
    """
    Create an Excel report summarizing failed tests and tests with specific markers and reasons.

    This function generates an Excel report that consolidates:
      1. Failed test cases (from `failed_models_ops_tests`), each marked as "FAILED".
      2. Additional test cases with marker information (from `status_to_tests_and_reason`, if provided).
         For these tests, the corresponding marker (e.g., "SKIP", "XFAIL") and the reason are included.

    The report is written to the file specified by `report_file_path` using the `create_excel_file` helper.

    Args:
        report_file_path (str): The file path where the Excel report will be saved.
        failed_models_ops_tests (Dict[str, str]): A dictionary mapping failed test cases to their failure reasons.
        status_to_tests_and_reason (Optional[Dict[str, Dict[str, str]]], optional):
            A dictionary mapping marker names to a dictionary of test cases and their associated reasons.
            Defaults to None, meaning only failed tests will be reported.

    Returns:
        None. The function writes the report to the specified file.
    """
    sheet_title = "model_ops_test"  # Title of the Excel sheet
    headers = ["TestCases", "Marker", "Reason"]  # Column headers for the Excel report
    data = []  # List to hold rows of data for the report

    # Add each failed test case to the report data with a "FAILED" marker.
    for failed_test, failure_reason in failed_models_ops_tests.items():
        data.append([failed_test, "FAILED", failure_reason])

    # If additional test status information is provided, add those test cases to the report.
    # Note: The check ensures that the dictionary is not None and not empty.
    if status_to_tests_and_reason is not None and len(status_to_tests_and_reason) != 0:
        for marker, test_with_reason in status_to_tests_and_reason.items():
            for test, reason in test_with_reason.items():
                data.append([test, marker, reason])

    # Create the Excel report using the provided helper function.
    create_excel_file(
        title=sheet_title,
        headers=headers,
        rows=data,
        output_file_path=report_file_path,
    )


def extract_data_from_report(report_file_path: str):
    """
    Extract test update configuration information from an Excel report.

    This function reads an Excel report that contains test case details, markers, and reasons,
    then parses and organizes the data to generate a mapping of test file paths to marker configuration
    details for updating model ops tests.
    
    Args:
        report_file_path (str): The file path to the Excel report.

    Returns:
        Dict[str, Dict[str, List[Dict[str, str]]]]:
            A dictionary mapping each test file path to another dictionary that maps markers
            to lists of test configuration dictionaries. Each configuration dictionary contains:
                - "reason": The associated failure reason.
                - "module_name": The extracted module name.
                - "shapes_and_dtypes": The extracted shapes and dtypes information.
    """
    # Read the Excel report into a DataFrame, selecting only the required columns.
    report_df = pd.read_excel(
        report_file_path,
        header=0,
        usecols=["TestCases", "Marker", "Reason"],
    )

    # Regex pattern to extract module name and shapes/dtypes from the test case function string.
    regex = r"\[([^\[]+)-\[(.*)\]\]"

    # Dictionary to hold the update configuration info.
    models_ops_test_update_info = {}

    # Iterate over each row in the DataFrame.
    for index, row in report_df.iterrows():
        # Clean and extract the test case string.
        test_cases = str(row.TestCases).strip("\n")
        # Normalize the marker using the helper function.
        marker = rename_marker(str(row.Marker).strip("\n"))
        # Extract the reason.
        reason = str(row.Reason).strip("\n")

        # Extract the test file path and test case function using a helper function.
        test_file_path, test_case_func = extract_test_file_path_and_test_case_func(test_cases)
        # Skip processing if either the test file path or test case function is missing.
        if test_file_path is None or test_case_func is None:
            continue

        # Initialize a dictionary to hold test configuration details.
        test_config = {}
        test_config["reason"] = reason

        # Use regex to extract module name and shapes/dtypes from the test case function string.
        match = re.search(regex, test_case_func)
        if match:
            module_name = match.group(1)
            shapes_and_dtypes = match.group(2)
            test_config["module_name"] = module_name
            test_config["shapes_and_dtypes"] = f"[{shapes_and_dtypes}]"

        # Update the configuration dictionary for the test file.
        # If the test file path already exists, update the marker configuration list.
        if test_file_path in models_ops_test_update_info.keys():
            if marker in models_ops_test_update_info[test_file_path].keys():
                models_ops_test_update_info[test_file_path][marker].append(test_config)
            else:
                models_ops_test_update_info[test_file_path][marker] = [test_config]
        else:
            # Create a new entry for the test file path with the marker and configuration.
            models_ops_test_update_info[test_file_path] = {marker: [test_config]}

    return models_ops_test_update_info




#python scripts/model_analysis/models_ops_test_failure_update.py --log_files ci_logs/pytest_1.log ci_logs/pytest_2.log ci_logs/pytest_3.log ci_logs/pytest_4.log
#python scripts/model_analysis/models_ops_test_failure_update.py --report_file_path model_ops_tests_report.xlsx --use_report
def main():
    """
    Main function to update model ops tests based on pytest log failures.

    This utility can operate in two modes:
      1. Use an existing Excel report (via --use_report flag) to update tests.
      2. Parse provided pytest log files directly to extract failed tests and test markers,
         then generate an Excel report and update tests accordingly.

    Pre-commit checks are run on the models ops test directory both before and after updates.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Update model ops tests with xfail marks based on pytest log failures. "
        )
    )
    parser.add_argument(
        "--log_files",
        nargs="+",
        type=str,
        required=False,
        help=(
            "A space-separated list of paths to pytest log files containing test execution output. "
            "These logs are used to extract information about failed tests and tests marked as XFAIL/SKIP."
        ),
    )
    parser.add_argument(
        "--models_ops_test_dir_path",
        type=str,
        default="forge/test/models_ops/",
        required=False,
        help=(
            "The directory path that contains the generated model ops tests. "
            "Defaults to 'forge/test/models_ops/'."
        ),
    )
    parser.add_argument(
        "--report_file_path",
        type=str,
        default=os.path.join(os.getcwd(), "model_ops_tests_report.xlsx"),
        required=False,
        help=(
            "The file path to the Excel report containing test case details (test cases, markers, and reasons). "
            "When used with the '--use_report' flag, this report is used to update the tests. "
            "Defaults to 'model_ops_tests_report.xlsx' in the current working directory."
        ),
    )
    parser.add_argument(
        "--use_report",
        action="store_true",
        help=(
            "If set, the utility will use the provided Excel report file to update model ops tests. "
            "If not set, the utility will parse the provided pytest log files directly to extract failure and marker information."
        ),
    )

    args = parser.parse_args()
    log_files = args.log_files
    models_ops_test_dir_path = args.models_ops_test_dir_path
    report_file_path = args.report_file_path
    use_report = args.use_report

    run_precommit(directory_path=models_ops_test_dir_path)

    if use_report and check_path(report_file_path):
        models_ops_test_update_info = extract_data_from_report(report_file_path=report_file_path)
        update_models_ops_tests(models_ops_test_update_info=models_ops_test_update_info)
        run_precommit(directory_path=models_ops_test_dir_path)
    else:
        status_to_tests_and_reason = extract_xfailed_and_skipped_tests_with_reason(
            log_files=log_files, models_ops_test_dir_path=models_ops_test_dir_path
        )

        failed_models_ops_tests = extract_failed_tests_with_failure_reason(
            log_files=log_files, target_dirs=[models_ops_test_dir_path]
        )

        if len(failed_models_ops_tests) == 0:
            log_files_str = ", ".join(log_files)
            logger.error(f"There are no failures in the provided {log_files_str} log files")

        create_report(
            report_file_path=report_file_path,
            failed_models_ops_tests=failed_models_ops_tests,
            status_to_tests_and_reason=status_to_tests_and_reason,
        )



if __name__ == "__main__":
    main()
