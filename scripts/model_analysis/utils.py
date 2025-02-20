# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
from loguru import logger
import subprocess
from enum import IntEnum
from typing import Union, Dict, List, Tuple, Optional, Callable
import shutil
from git import Repo

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
        extract_additional_info_func (Optional[Callable[[List[str], str], str]]):
            A callable function to extract additional information for the matched exception.
            This function takes two arguments:
                - rule_tokens (List[str]): List of matched tokens in an exception message.
                - exception (str): The original exception message.
            It returns a string with the additional extracted information.
    """

    def __init__(
        self,
        rule_name: str,
        rule_tokens: List[str],
        extract_additional_info_func: Optional[Callable[[List[str], str], str]] = None,
    ):
        self.rule_name = rule_name
        self.rule_tokens = rule_tokens
        self.extract_additional_info_func = extract_additional_info_func

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
            matched_expection = ""
            if self.extract_additional_info_func is not None:
                matched_expection = self.extract_additional_info_func(self.rule_tokens, exception)
            if len(matched_expection) == 0:
                matched_expection = " ".join(self.rule_tokens)
            return matched_expection
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


class CompilerComponentFailureAnalysis:
    """
    Analyzes and maintains failure details for different compiler components.

    Attributes:
        compiler_component_and_failure_details (Dict[CompilerComponent, Dict[str, List[str]]]):
            A dictionary mapping each compiler component to its associated failure reasons and the corresponding model variant names.
    """

    def __init__(self):
        self.compiler_component_and_failure_details = {}

    def update(self, compiler_component: CompilerComponent, failure_reason: str, model_variant_names: List[str]):
        """
        Updates the failure details for a given compiler component.

        Args:
            compiler_component (CompilerComponent): The compiler component to update.
            failure_reason (str): The reason for the failure.
            model_variant_names (List[str]): A list of model variant names associated with the failure.
        """
        if compiler_component in self.compiler_component_and_failure_details.keys():
            if failure_reason in self.compiler_component_and_failure_details[compiler_component].keys():
                self.compiler_component_and_failure_details[compiler_component][failure_reason].extend(
                    model_variant_names
                )
                self.compiler_component_and_failure_details[compiler_component][failure_reason] = list(
                    set(self.compiler_component_and_failure_details[compiler_component][failure_reason])
                )
            else:
                self.compiler_component_and_failure_details[compiler_component][failure_reason] = model_variant_names
        else:
            self.compiler_component_and_failure_details[compiler_component] = {failure_reason: model_variant_names}

    def sort_by_model_variant_names_length(self, reverse: bool = False):
        """
        Sorts the failure reasons for each compiler component based on the number of associated model variant names.

        Args:
            reverse (bool): If True, sorts in descending order; otherwise, in ascending order. Default is False.
        """
        self.compiler_component_and_failure_details = {
            compiler_component: dict(sorted(failure_details.items(), key=lambda item: len(item[1]), reverse=reverse))
            for compiler_component, failure_details in sorted(self.compiler_component_and_failure_details.items())
        }

    def get_compiler_component_and_failure_details(self):
        return self.compiler_component_and_failure_details


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


def sort_model_variant_info_list(model_variant_info_list):
    """
    Sorts a list of ModelVariantInfo objects first by model_name and then by variant_name.

    Args:
        model_variant_info_list (List[ModelVariantInfo]): The list of ModelVariantInfo objects to be sorted.

    Returns:
        List[ModelVariantInfo]: The sorted list of ModelVariantInfo objects.
    """
    # Sort the list by model_name and then by variant_name
    return sorted(
        model_variant_info_list,
        key=lambda model_variant_info: (model_variant_info.model_name, model_variant_info.variant_name),
    )


def get_commit_id(repo_path="."):
    """
    Retrieves the latest commit ID (SHA-1 hash) and commit ID URL from the specified Git repository.

    Args:
        repo_path (str): Path to the Git repository. Defaults to the current directory.

    Returns:
        tuple[str, ...]: Returns commit ID and URL if successful; otherwise, None.
    """
    try:
        repo = Repo(repo_path)
        commit_id = repo.head.commit.hexsha

        # Get the repository's remote URL
        remote_url = next(repo.remote().urls)

        # Convert SSH URL to HTTPS
        if remote_url.startswith("git@"):
            remote_url = remote_url.replace(":", "/").replace("git@", "https://")

        if remote_url.endswith(".git"):
            remote_url = remote_url[:-4]

        # Construct the commit URL
        commit_url = f"{remote_url}/commit/{commit_id}"
        return commit_id, commit_url

    except Exception as e:

        logger.warning(f"Error while fetching commit id: {e}")

        return None, None
