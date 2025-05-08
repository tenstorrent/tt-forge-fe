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
import ast

import torch
from forge.tvm_unique_op_generation import UniqueOperations


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
    """Checks if pre-commit is installed and runs it on all files in the given directory."""

    # Check if pre-commit is installed
    if shutil.which("pre-commit") is None:
        logger.info("pre-commit is not installed. Installing...")
        run_command("pip install pre-commit")
    else:
        logger.info("pre-commit is already installed.")

    # Run pre-commit on all files in the directory
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


def extract_framework_from_test_file_path(test_file_path: str):
    if "forge/test/models/pytorch" in test_file_path:
        framework = "pytorch"
    elif "forge/test/models/onnx" in test_file_path:
        framework = "onnx"
    elif "forge/test/models/paddlepaddle" in test_file_path:
        framework = "paddlepaddle"
    else:
        framework = "unknown"
    return framework


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


def filter_unique_operations(unique_operations: UniqueOperations, ops_to_filter: Optional[List[str]] = None):
    """
    Select and return only the unique operations whose forge operation names match those specified in `ops_to_filter`.
    """
    if not unique_operations or ops_to_filter is None or not ops_to_filter:
        return unique_operations

    if not any(
        [
            True if forge_op_function_name.split(".")[-1] in ops_to_filter else False
            for forge_op_function_name in unique_operations.keys()
        ]
    ):
        raise ValueError(f"Provided op names {filter_ops} are not found in unique ops extracted across all models")

    return {
        forge_op_function_name: unique_operands_and_opargs_opmetadata
        for forge_op_function_name, unique_operands_and_opargs_opmetadata in unique_operations.items()
        if forge_op_function_name.split(".")[-1] in ops_to_filter
    }


def filter_tests(tests: List[str], tests_to_filter: Optional[List[str]] = None) -> List[str]:
    """
    Return a list of tests that match any of the filter tests list.
    """
    if not tests or tests_to_filter is None or not tests_to_filter:
        return tests

    filtered_tests = [test for test in tests if any(filter_test in test for filter_test in tests_to_filter)]
    if not filtered_tests:
        raise ValueError("None of the specified tests match the collected model analysis tests.")

    filtered_tests.sort()

    return filtered_tests


def extract_models_ops_test_params(pytest_file_path: str):
    """
    Parse the given pytest file and extract, for each entry in
    'forge_modules_and_shapes_dtypes_list':
      - module_class_name: Name of the ForgeModule subclass
      - forge_op_name: The fully-qualified forge.op.* call
      - operand_shapes: List of tuple shapes passed to the op
      - operand_types: "Activation", "Constant", or "Parameter"
      - operand_dtypes: Corresponding dtype strings
      - op_kwargs: Keyword arguments passed to the opname
      - test_metadata: Dictionary of model_names, pcc, max_int, and custom args
      - markers: pytest.mark annotations with names and reasons

    Args:
        pytest_file_path (str): Path to the pytest file (.py)

    Returns:
        List[dict]: Each dict contains all extracted fields for one test param.
    """
    # Read file content and build AST
    with open(pytest_file_path, "r") as f:
        file_content = f.read()
    ast_tree = ast.parse(file_content)

    # Map each class name to its AST ClassDef node
    class_defs = {node.name: node for node in ast_tree.body if isinstance(node, ast.ClassDef)}

    # Pre-extract constants and parameters from each class's __init__
    init_data = {}
    for class_name, class_node in class_defs.items():
        constants_map = {}
        parameters_map = {}

        # Find __init__ method
        for stmt in class_node.body:
            if not (isinstance(stmt, ast.FunctionDef) and stmt.name == "__init__"):
                continue
            # Process each call inside __init__
            for init_call in stmt.body:
                if not (isinstance(init_call, ast.Expr) and isinstance(init_call.value, ast.Call)):
                    continue
                call_node = init_call.value

                # add_constant(name, shape=..., dtype=...)
                if isinstance(call_node.func, ast.Attribute) and call_node.func.attr == "add_constant":
                    const_name = ast.literal_eval(call_node.args[0])
                    const_shape = next(
                        (ast.literal_eval(kw.value) for kw in call_node.keywords if kw.arg == "shape"), None
                    )
                    const_dtype = next((ast.unparse(kw.value) for kw in call_node.keywords if kw.arg == "dtype"), None)
                    constants_map[const_name] = (const_shape, const_dtype)

                # add_parameter(name, forge.Parameter(*shape, ...))
                elif isinstance(call_node.func, ast.Attribute) and call_node.func.attr == "add_parameter":
                    param_name = ast.literal_eval(call_node.args[0])
                    parameter_ctor = call_node.args[1]
                    param_shape = next(
                        (ast.literal_eval(arg.value) for arg in parameter_ctor.args if isinstance(arg, ast.Starred)),
                        None,
                    )
                    param_dtype = next(
                        (ast.unparse(kw.value) for kw in parameter_ctor.keywords if kw.arg == "dev_data_format"), None
                    )
                    parameters_map[param_name] = (param_shape, param_dtype)

        init_data[class_name] = {
            "constants": constants_map,
            "parameters": parameters_map,
        }

    # List to collect extracted parameter info
    results = []

    # Locate the pytest parameter list assignment
    for node in ast_tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not (isinstance(target, ast.Name) and target.id == "forge_modules_and_shapes_dtypes_list"):
                continue

            # Iterate each element in the list
            for list_element in node.value.elts:
                # Initialize marker list
                marker_list = []

                # Handle pytest.param(..., marks=[...]) wrapper
                if (
                    isinstance(list_element, ast.Call)
                    and isinstance(list_element.func, ast.Attribute)
                    and list_element.func.attr == "param"
                ):
                    param_call = list_element
                    # Extract markers if present
                    for kw in param_call.keywords:
                        if kw.arg == "marks":
                            mark_nodes = kw.value.elts if isinstance(kw.value, (ast.List, ast.Tuple)) else [kw.value]
                            for mark_node in mark_nodes:
                                if isinstance(mark_node, ast.Call) and isinstance(mark_node.func, ast.Attribute):
                                    mark_name = mark_node.func.attr
                                    mark_reason = None
                                    # Check for reason=...
                                    for mk_kw in mark_node.keywords:
                                        if mk_kw.arg == "reason":
                                            mark_reason = ast.literal_eval(mk_kw.value)
                                    marker_list.append(
                                        {
                                            "maker_name": mark_name,
                                            "reason": mark_reason,
                                        }
                                    )
                                elif isinstance(mark_node, ast.Attribute):
                                    mark_name = mark_node.attr
                                    marker_list.append({"maker_name": mark_name, "reason": None})

                    # The real tuple is the first positional arg
                    param_tuple = param_call.args[0]
                elif isinstance(list_element, ast.Tuple):
                    param_tuple = list_element
                else:
                    # Skip unexpected elements
                    continue

                # Unpack the tuple: (ModuleClass, [ (shape,dtype)... ], metadata_dict)
                module_class_name = ast.unparse(param_tuple.elts[0])
                shapes_and_dtypes = [
                    (ast.literal_eval(s.elts[0]), ast.unparse(s.elts[1])) for s in param_tuple.elts[1].elts
                ]
                metadata = ast.literal_eval(param_tuple.elts[2])

                # Extract common metadata fields
                model_names = metadata.get("model_names", [])
                pcc_value = metadata.get("pcc", None)
                max_int_value = metadata.get("max_int", None)
                op_args = metadata.get("args", {})

                # Find the op call inside the forward method of this class
                class_node = class_defs.get(module_class_name)
                op_call_node = None
                for fn in class_node.body:
                    if isinstance(fn, ast.FunctionDef) and fn.name == "forward":
                        for stmt in ast.walk(fn):
                            if (
                                isinstance(stmt, ast.Call)
                                and isinstance(stmt.func, ast.Attribute)
                                and isinstance(stmt.func.value, ast.Attribute)
                                and isinstance(stmt.func.value.value, ast.Name)
                                and stmt.func.value.value.id == "forge"
                                and stmt.func.value.attr == "op"
                            ):
                                op_call_node = stmt
                                break
                        break

                # Iterate over positional args to collect operand info
                shape_iter = iter(shapes_and_dtypes)
                operand_shapes = []
                operand_types = []
                operand_dtypes = []
                operand_names = []
                const_map = init_data[module_class_name]["constants"]
                param_map = init_data[module_class_name]["parameters"]

                for operand in op_call_node.args[1:]:
                    # Activation inputs are plain variables
                    if isinstance(operand, ast.Name):
                        operand_types.append("Activation")
                        operand_names.append(operand.id)
                        shape, dtype = next(shape_iter)
                        operand_shapes.append(tuple(shape))
                        operand_dtypes.append(dtype)

                    # Constants or Parameters fetched via self.get_*()
                    elif isinstance(operand, ast.Call) and isinstance(operand.func, ast.Attribute):
                        if operand.func.attr == "get_constant":
                            operand_types.append("Constant")
                            const_key = ast.literal_eval(operand.args[0])
                            operand_names.append(const_key)
                            shape, dtype = const_map.get(const_key, (None, None))
                            operand_shapes.append(tuple(shape))
                            operand_dtypes.append(dtype)
                        elif operand.func.attr == "get_parameter":
                            operand_types.append("Parameter")
                            param_key = ast.literal_eval(operand.args[0])
                            operand_names.append(param_key)
                            shape, dtype = param_map.get(param_key, (None, None))
                            operand_shapes.append(tuple(shape))
                            operand_dtypes.append(dtype)

                # Aggregate all extracted details
                results.append(
                    {
                        "module_name": module_class_name,
                        "forge_op_name": ast.unparse(op_call_node.func),
                        "operand_types": operand_types,
                        "operand_shapes": operand_shapes,
                        "operand_dtypes": operand_dtypes,
                        "operand_names": operand_names,
                        "op_args": op_args,
                        "model_names": model_names,
                        "pcc": pcc_value,
                        "max_int": max_int_value,
                        "markers": marker_list,
                    }
                )

    return results
