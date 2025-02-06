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


def read_file(file_path: str):
    with open(file_path, "r") as file:
        lines = file.readlines()
    return lines

def extract_failed_models_ops_tests_with_failure_reason(log_files: List[str], models_ops_test_dir_path: str):

    test_func_pattern = r"^_+\s+(test_[a-zA-Z0-9_]+\[.*\])\s+_+$"

    failed_models_ops_tests = {}
    maximum_error_lines = 3

    for log_file in log_files:

        logger.info("log_file={}",log_file)

        if check_path(log_file):
            
            lines = read_file(log_file)

            collect_failed_models_ops_tests = False
            for line in lines:
                if "==== short test summary info ====" in line:
                    collect_failed_models_ops_tests = True
                elif collect_failed_models_ops_tests and "FAILED" in line:
                    failed_test = line.replace("FAILED", "").strip("\n").strip()
                    if models_ops_test_dir_path in failed_test and failed_test not in failed_models_ops_tests.keys():
                        failed_models_ops_tests[failed_test] = ""

            if len(failed_models_ops_tests) == 0:
                logger.warning(f"There is no failure in the {log_file}")
                continue
            
            failed_models_ops_tests = dict(sorted(failed_models_ops_tests.items(), key=lambda kv: (kv[1], kv[0])))


            test_case_func = ""
            collect_failure_reason = False

            for current_line, line in enumerate(lines):
                if "==== FAILURES ====" in line:
                    collect_failure_reason = True
                elif collect_failure_reason:
                    if len(test_case_func) == 0:
                        match = re.search(test_func_pattern, line)
                        if match:
                            test_case_func = match.group(1)
                    else:
                        if line.startswith("E  "):
                            if all([True if error_line.startswith("E  ") else False for error_line in lines[current_line : current_line+maximum_error_lines]]):
                                error_message = [error_line.replace("E  ", "").strip("\n").strip() for error_line in lines[current_line : current_line+maximum_error_lines]]
                                error_message = "\n".join(error_message)
                            else:
                                error_message = line.replace("E  ", "").strip("\n").strip()

                            for failed_test in failed_models_ops_tests.keys():
                                if test_case_func in failed_test:
                                    failed_models_ops_tests[failed_test] = error_message
                            
                            test_case_func = ""

        else:
            logger.warning(f"Provided {log_file} path doesn't exists!!")

    return failed_models_ops_tests

def extract_failed_models_ops_tests_config(failed_models_ops_tests: Dict[str, str]):
    failed_models_ops_tests_info = {}
    # Regular expression to capture model name and operand shapes
    regex = r"\[([^\[]+)-\[(.*)\]\]"
    for idx, (failed_test, error_message) in enumerate(failed_models_ops_tests.items()):
        failed_test_path, failed_test_cases = failed_test.split("::")
        match = re.search(regex, failed_test_cases)
        failed_test_config = {}
        failed_test_config["error_message"] = error_message
        if match:
            module_name = match.group(1)
            shapes_and_dtypes = match.group(2)
            failed_test_config["module_name"] = module_name
            failed_test_config["shapes_and_dtypes"] = f"[{shapes_and_dtypes}]"
        if failed_test_path in failed_models_ops_tests_info.keys():
            failed_models_ops_tests_info[failed_test_path].append(failed_test_config)
        else:
            failed_models_ops_tests_info[failed_test_path] = [failed_test_config]
    return failed_models_ops_tests_info

    def add_params(forge_modules, failed_details):
        new_forge_modules = []
        for forge_module in forge_modules:
            matched = False
            for failed_detail in failed_details:
                if forge_module[0] == failed_detail["module_name"] and forge_module[1] == failed_detail["shapes_and_dtypes"]:
                    forge_module_str = ", ".join(forge_module)
                    error_message = failed_detail["error_message"]
                    new_forge_modules.append(f'pytest.param(({forge_module_str}), marks=[pytest.mark.xfail(reason="{error_message}")])')
                    matched = True
            if not matched:
                forge_module_str = ", ".join(forge_module)
                new_forge_modules.append(f"({forge_module_str})")
        return new_forge_modules

def extract_models_ops_test_params(file_path: str):
    with open(file_path, "r") as file:
        content = file.read()
    
    tree = ast.parse(content)
    
    models_ops_test_params = []
    
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "forge_modules_and_shapes_dtypes_list":
                    for elt in node.value.elts:
                        if isinstance(elt, ast.Call) and isinstance(elt.func, ast.Attribute) and elt.func.attr == "param":
                            # Extract the first argument of pytest.param(...)
                            param_tuple = elt.args[0] if elt.args else None
                        elif isinstance(elt, ast.Tuple):
                            param_tuple = elt
                        else:
                            param_tuple = None
                        
                        if param_tuple:
                            tuple_elements = [ast.unparse(item) for item in param_tuple.elts]
                            models_ops_test_params.append(tuple_elements)
    return models_ops_test_params

def update_params(models_ops_test_params, failed_test_configs):
    new_models_ops_test_params = []
    for param in models_ops_test_params:
        param_str = ", ".join(param)
        matched = False
        for config in failed_test_configs:
            if param[0] == config["module_name"] and param[1] == config["shapes_and_dtypes"]:
                error_message = config["error_message"]
                new_models_ops_test_params.append(f'pytest.param(({param_str}), marks=[pytest.mark.xfail(reason="""{error_message}""")])')
                matched = True
        if not matched:
            new_models_ops_test_params.append(f"({param_str})")
    return new_models_ops_test_params

def update_models_ops_tests_failures(failed_models_ops_tests_info: Dict[str, List[Dict[str, str]]]):

    for idx, (failed_test_path, failed_test_config) in enumerate(failed_models_ops_tests_info.items()):
        models_ops_test_params = extract_models_ops_test_params(failed_test_path)
        new_models_ops_test_params = update_params(models_ops_test_params, failed_test_config)

        lines = read_file(failed_test_path)

        new_lines = []
        is_pytest_params = False
        for line in lines:
            if "@pytest.mark.push" in line:
                new_lines.append(f"forge_modules_and_shapes_dtypes_list = [\n")
                for test_param in new_models_ops_test_params:
                    new_lines.append(f"\t{test_param},\n")
                new_lines.append("]\n")
                new_lines.append("\n")
                new_lines.append("\n")
                new_lines.append(line)
                is_pytest_params = False
            elif "forge_modules_and_shapes_dtypes_list = [" in line or is_pytest_params:
                is_pytest_params = True   
            else:
                new_lines.append(line)

        with open(failed_test_path, "w") as file:
            file.writelines(new_lines)

# python scripts/model_analysis/models_ops_test_failure_update.py --log_files models_ops_test.log &> automatic_script.log
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_files", 
        nargs="+", 
        type=str,
        required=True,
        help="List of pytest log files", 
    )
    parser.add_argument(
        "--models_ops_test_dir_path", 
        type=str,
        default="forge/test/models_ops/"
        required=False,
        help="Specify the directory path contains generated models ops test", 
    )

    args = parser.parse_args()
    log_files=args.log_files
    models_ops_test_dir_path=args.models_ops_test_dir_path

    run_precommit(directory_path=models_ops_test_dir_path)
    failed_models_ops_tests = extract_failed_models_ops_tests_with_failure_reason(log_files=log_files, models_ops_test_dir_path=models_ops_test_dir_path)
    if len(failed_models_ops_tests) == 0:
        log_files_str = ", ".join(log_files)
        logger.error(f"There is no failures in the provided {log_files_str} log files")
    
    failed_models_ops_tests_info = extract_failed_models_ops_tests_config(failed_models_ops_tests=failed_models_ops_tests)
    update_models_ops_tests_failures(failed_models_ops_tests_info=failed_models_ops_tests_info)
    run_precommit(directory_path=models_ops_test_dir_path)

if __name__ == "__main__":
    main()