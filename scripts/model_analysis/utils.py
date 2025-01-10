# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
from loguru import logger
import subprocess
from enum import IntEnum
from typing import Union, Dict, List, Tuple
import shutil

import torch


class CompilerComponent(IntEnum):
    FORGE = 0
    MLIR = 1
    TT_METAL = 2
    UNKNOWN = 4


class MatchingExceptionRule:
    """
    Represents a rule for matching exceptions based on specific tokens.

    Attributes:
        rule_name (str): Name of the rule.
        rule_tokens (List[str]): List of tokens to match in an exception message.
    """

    def __init__(self, rule_name: str, rule_tokens: List[str]):
        self.rule_name = rule_name
        self.rule_tokens = rule_tokens

    def match_rule(self, exception: str):
        """
        Matches the rule tokens against the given exception string.

        Args:
            exception (str): Exception message to match against.
        """
        # Check if all tokens in rule_tokens exist in the exception message and
        # return the rule_token if matches otherwise return None
        matched_token = all([True if token in exception else False for token in self.rule_tokens])
        if matched_token:
            return " ".join(self.rule_tokens)
        else:
            return None


class MatchingCompilerComponentException:
    """
    Represents exception matching for a specific compiler component.

    Attributes:
        compiler_component (CompilerComponent): Compiler component associated with this exception matching.
        exception_rules (List[MatchingExceptionRule]): List of exception rules for this component.
    """

    def __init__(self, compiler_component: CompilerComponent, exception_rules: List[MatchingExceptionRule]):
        self.compiler_component = compiler_component
        self.exception_rules = exception_rules

    def match_rule(self, exception: str):
        """
        Matches the given exception against the exception rules of this compiler component.
        Args:
            exception (str): Exception message to be checked against the rules.
        """
        # Iterate over all exception rules for this compiler component.
        for rule in self.exception_rules:
            # Attempt to match the current rule against the exception and If a match is found,
            # return the compiler component and the constructed error message.
            if rule.match_rule(exception) is not None:
                match_err_msg = (
                    f"[{self.compiler_component.name}] "
                    if rule.rule_name is None or rule.rule_name == ""
                    else f"[{self.compiler_component.name}][{rule.rule_name}] "
                )
                match_err_msg += rule.match_rule(exception)
                return self.compiler_component, match_err_msg

        return None, None


def check_path(directory_or_file_path: str):

    # Check if a file or directory exists, return True otherwise return False
    if os.path.exists(directory_or_file_path):
        logger.info(f"{directory_or_file_path} exists!")
        return True

    logger.info(f"{directory_or_file_path} does not exist.")
    return False


def create_python_package(package_path: str, package_name: str):
    package_path = os.path.join(package_path, package_name)
    if not check_path(package_path):
        os.makedirs(package_path, exist_ok=True)
    if not check_path(os.path.join(package_path, "__init__.py")):
        try:
            f = open(os.path.join(package_path, "__init__.py"), "x")
            f.close()
            logger.info(f"Created package in this path {package_path}")
        except FileExistsError:
            logger.info(f"__init__.py file already exists inside {package_path} path")


def dump_logs(log_files: Union[str, List[str]], content: str):
    if isinstance(log_files, str):
        log_files = [log_files]
    for log_file in log_files:
        log_file_dir_path = "/".join(log_file.split("/")[:-1])
        os.makedirs(log_file_dir_path, exist_ok=True)
        with open(log_file, "w") as f:
            f.write(content)
            logger.info(f"Dumped test logs in {log_file}")


def collect_all_model_analysis_test(directory_or_file_path: str, output_directory_path: str):
    """
    Collect all the tests that doesn't contains skip_model_analysis marker in a specified directory or file.
    """

    # Ensure the directory or file path exists
    assert check_path(
        directory_or_file_path
    ), f"The directory path for collecting test {directory_or_file_path} doesn't exists"

    logger.info(
        f"Collecting all the tests that doesn't contains skip_model_analysis marker in {directory_or_file_path}"
    )

    collected_test_outputs = ""
    try:
        # Run pytest to collect tests with the `model_analysis` marker
        result = subprocess.run(
            ["pytest", directory_or_file_path, "-m", "not skip_model_analysis", "--collect-only"],
            capture_output=True,
            text=True,
            check=True,
        )

        # Append stdout and stderr to the collected outputs
        collected_test_outputs += "STDOUT:\n" + result.stdout
        collected_test_outputs += "STDERR:\n" + result.stderr

    except subprocess.CalledProcessError as e:
        collected_test_outputs += e.output

    # Save the collected test outputs to a file
    collected_test_file_path = os.path.join(output_directory_path, "collected_tests.txt")
    dump_logs(collected_test_file_path, collected_test_outputs)

    # Extract tests from the collected test outputs
    test_list = []
    with open(collected_test_file_path, "r") as collected_test_file:
        lines = collected_test_file.readlines()
        test_lines = False
        for line in lines:
            if "Automatic Model Analysis Collected tests:" in line:
                test_lines = True
            elif "Automatic Model Analysis Collected test count:" in line:
                test_lines = False
                break
            elif test_lines:
                test_list.append(str(line).replace("\n", ""))

    return test_list


def run_command(command: str):
    command_outputs = ""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=True,
        )

        # Append stdout and stderr to the command outputs
        command_outputs += result.stdout + result.stderr

    except subprocess.CalledProcessError as e:
        command_outputs += e.output

    logger.info(f"Running the {command}\n{command_outputs}")


def run_precommit(directory_path: str):
    run_command("pip install pre-commit")
    run_command(f"pre-commit run --files $(find {directory_path} -type f)")


def remove_directory(directory_path: str):
    if check_path(directory_path):
        try:
            shutil.rmtree(directory_path)
            print(f"The directory path '{directory_path}' and its contents have been removed.")
        except Exception as e:
            print(f"An error occurred while removing the directory path {directory_path}: {e}")
