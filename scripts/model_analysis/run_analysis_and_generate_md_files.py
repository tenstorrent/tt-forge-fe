# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import subprocess
import os
import time
import json
from loguru import logger
import argparse
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
import ast

import torch

from forge.tvm_unique_op_generation import Operation, NodeType, UniqueOperations

from exception_rules import common_failure_matching_rules_list
from markdown import HtmlSymbol, MarkDownWriter
from unique_ops_utils import generate_and_export_unique_ops_tests, extract_unique_op_tests_from_models
from utils import (
    CompilerComponent,
    CompilerComponentFailureAnalysis,
    sort_model_variant_info_list,
    check_path,
    dump_logs,
    remove_directory,
    get_commit_id,
)


class UniqueOpTestInfo:
    """
    Represents information about a unique operation test, that includes op name, operands
    arguments, and the status of various compiler components.

    Attributes:
        op (str): The name of the operation.
        operands (List[str]): List of operands associated with the operation.
        args (List[str]): List of Operation Arguments if any
        components (dict): A dictionary indicating the support status for each compiler component.
        failure_reason (str): The reason for failure, if any, during testing.
    """

    def __init__(
        self,
        op: str,
        operands: List[str],
        args: List[str],
    ):
        self.op = str(op)
        self.operands = operands
        self.args = args
        self.components = {}
        for compiler_component in CompilerComponent:
            self.components[str(compiler_component.name)] = False
        self.failure_reason = ""

    @classmethod
    def create(cls, op_name, operand_names, operand_types, operand_shapes, operand_dtypes, args):

        operands = cls.create_operands(operand_names, operand_types, operand_shapes, operand_dtypes)

        args = cls.create_args(args)

        return cls(op=op_name, operands=operands, args=args)

    @classmethod
    def create_operands(cls, operand_names, operand_types, operand_shapes, operand_dtypes):
        operands = []
        for operand_name, operand_type, operand_shape, operand_dtype in zip(
            operand_names, operand_types, operand_shapes, operand_dtypes
        ):
            if isinstance(operand_shape, torch.Tensor):
                operands.append(f"Operand(type={operand_type}, name={operand_name}, dtype={operand_dtype})")
            else:
                operands.append(f"Operand(type={operand_type}, shape={operand_shape}, dtype={operand_dtype})")
        return operands

    @classmethod
    def create_args(cls, args):
        arg_info = []
        if not args.is_empty():
            for arg_name, arg_value in args.items():
                arg_info.append(f"{arg_name} : {arg_value}")
        return arg_info

    def update_compiler_components(self, error_message: str = ""):
        if error_message:
            updated_compiler_component_status = False
            # Iterate over all failure matching rules to find a match.
            for rule in common_failure_matching_rules_list:
                matched_compiler_component, match_err_msg = rule.match_rule(error_message)
                if matched_compiler_component is not None:
                    updated_compiler_component_status = True
                    self.failure_reason = match_err_msg
                    # Set all the compiler components less than matched compiler component to True.
                    for compiler_component in CompilerComponent:
                        if compiler_component < matched_compiler_component:
                            self.components[str(compiler_component.name)] = True
                    break
            # If no match is found, mark the UNKNOWN compiler component alone to True.
            if not updated_compiler_component_status:
                self.components[str(CompilerComponent.UNKNOWN.name)] = True
                self.failure_reason = "[UNKNOWN] The failure does not match any known compiler component exception rules. Please review the failure log to identify the component"
            return matched_compiler_component, match_err_msg

        else:
            # If no error message is provided, mark all compiler components (except UNKNOWN) to True.
            for compiler_component in CompilerComponent:
                if compiler_component != CompilerComponent.UNKNOWN:
                    self.components[str(compiler_component.name)] = True

            return None, None

    def __str__(self):
        return f"UniqueOpTestInfo(op={self.op}, operands={self.operands}, args={self.args}, components={self.components}, self.failure_reason={self.failure_reason})"


@dataclass
class ModelVariantInfo:
    """
    Stores information about a model, variant, framework of the model, including its support rates for different compiler components.

    Attributes:
        model_name (str): The name of the model.
        variant_name (str): The name of the model variant.
        framework (str): The framework used for the model.
        unique_ops (List[UniqueOpTestInfo]): List of unique op configuration test info
        forge_support_rate (float): The support rate for the Forge compiler component. Defaults to 0.0.
        mlir_support_rate (float): The support rate for the MLIR compiler component. Defaults to 0.0.
        ttmetal_support_rate (float): The support rate for the TT_METAL compiler component. Defaults to 0.0.
        unknown_rate (float): The support rate for an unknown compiler component. Defaults to 0.0.
    """

    model_name: str
    variant_name: str
    framework: str
    unique_ops: List[UniqueOpTestInfo]
    forge_support_rate: float = 0.0
    mlir_support_rate: float = 0.0
    ttmetal_support_rate: float = 0.0
    unknown_rate: float = 0.0

    def get_support_rate(self, compiler_component: CompilerComponent):
        # Check and return the appropriate support rate based on the compiler component.
        if compiler_component == CompilerComponent.FORGE:
            return self.forge_support_rate
        elif compiler_component == CompilerComponent.MLIR:
            return self.mlir_support_rate
        elif compiler_component == CompilerComponent.TT_METAL:
            return self.ttmetal_support_rate
        elif compiler_component == CompilerComponent.UNKNOWN:
            return self.unknown_rate
        else:
            logger.error(f"There is no compilercomponent {compiler_component.name}")

    def update_support_rate(self, compiler_component: CompilerComponent, support_rate: float):
        # Update the appropriate support rate based on the compiler component.
        if compiler_component == CompilerComponent.FORGE:
            self.forge_support_rate = support_rate
        elif compiler_component == CompilerComponent.MLIR:
            self.mlir_support_rate = support_rate
        elif compiler_component == CompilerComponent.TT_METAL:
            self.ttmetal_support_rate = support_rate
        elif compiler_component == CompilerComponent.UNKNOWN:
            self.unknown_rate = support_rate
        else:
            logger.error(f"There is no compilercomponent {compiler_component.name}")

    def __str__(self):
        model_variant_info = ""
        model_variant_info += f"\t\tModel : {model_name}\n"
        model_variant_info += f"\t\tVariant : {variant_name}\n"
        model_variant_info += f"\t\tframework : {framework}\n"
        model_variant_info += f"\t\tforge_support_rate : {forge_support_rate}\n"
        model_variant_info += f"\t\tmlir_support_rate : {mlir_support_rate}\n"
        model_variant_info += f"\t\tttmetal_support_rate : {ttmetal_support_rate}\n"
        model_variant_info += f"\t\tunknown_rate : {unknown_rate}\n"
        for idx, unique_op in enumerate(unique_ops):
            model_variant_info += f"\t\t\t\t{idx}){str(unique_op)}\n"


def run_models_unique_op_tests(unique_operations, unique_ops_output_directory_path, dump_failure_logs):
    """
    Run unique op configuration test that has been collected across all the models and populate the test result in the model variants
    """

    models_details = {}

    compiler_component_failure_analysis = CompilerComponentFailureAnalysis()

    # Iterate over each unique operation
    for forge_op_function_name in sorted(unique_operations):

        # Extract operation name from forge op function name
        op_name = forge_op_function_name.split(".")[-1]

        # Get the unique operands and operation arguments assiocated the operand metadata
        unique_operands_and_opargs_opmetadata = unique_operations[
            forge_op_function_name
        ].get_unique_operands_and_opargs_opmetadata()

        for operands, opargs_opmetadata in unique_operands_and_opargs_opmetadata:

            for args, operation_metadata in opargs_opmetadata.get_op_args_and_metadata():

                # Extract operands details such as names types, shapes, and data types
                operand_types = [NodeType.to_json(operand_type) for operand_type in operands.get_operand_types()]
                operand_shapes = operands.get_operand_shapes()
                operand_dtypes = operands.get_operand_dtypes()

                # Extract model varaiant info such as model, variant and framework name
                model_variant_info_list = operation_metadata["model_variant_info"]
                framework = model_variant_info_list[0]["framework"]
                operand_names = operation_metadata["operand_names"][0]

                # Create a UniqueOpTestInfo object to store details about the operation (name, operands, args)
                unique_op_test_info = UniqueOpTestInfo.create(
                    op_name=op_name,
                    operand_names=operand_names,
                    operand_types=operand_types,
                    operand_shapes=operand_shapes,
                    operand_dtypes=operand_dtypes,
                    args=args,
                )

                # Extract the test file path
                test = model_variant_info_list[0]["Testfile"]
                logger.info(f"Running the test: {test}")

                # If dump_failure_logs is set to True, prepare the log file paths for storing logs
                if dump_failure_logs:
                    log_files = []
                    for model_variant_info in model_variant_info_list:
                        log_file_dir_path = os.path.join(
                            unique_ops_output_directory_path,
                            model_variant_info["model_name"],
                            model_variant_info["variant_name"],
                            op_name,
                        )
                        test_name = model_variant_info["Testfile"].split("::")[
                            -1
                        ]  # Extract the test name from the test path
                        log_file_name = str(test_name) + "_log.txt"
                        log_file = os.path.join(log_file_dir_path, log_file_name)
                        log_files.append(log_file)

                # Start the timer to measure test execution time
                start_time = time.time()

                try:
                    # Run the unique op test by using subprocess libary run method.
                    result = subprocess.run(
                        ["pytest", test, "-vss"],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=180,
                        env=dict(
                            os.environ,
                            FORGE_DISABLE_REPORTIFY_DUMP="1",
                        ),
                    )

                    # Calculate elapsed time after test execution
                    elapsed_time = time.time() - start_time

                    if result.returncode != 0:
                        logger.info(f"\tFailed ({elapsed_time:.2f} seconds)")

                        # If the result.returncode is not equal to zero, collect the test stdout and stderr
                        error_message = ""
                        if result.stderr:
                            error_message += "STDERR: \n\n"
                            error_message += result.stderr
                        if result.stdout:
                            error_message += "STDOUT: \n\n"
                            error_message += result.stdout

                        # Update the instance of the UniqueOpTestInfo components datamember status
                        # for each compiler component and error message in failure_reason datamember
                        matched_compiler_component, match_err_msg = unique_op_test_info.update_compiler_components(
                            error_message
                        )

                        # Save failure logs if dump_failure_logs is set to True
                        if dump_failure_logs:
                            dump_logs(log_files, error_message)

                    else:
                        # If the test passed (return code is 0), update the UniqueOpTestInfo instance
                        # components datamember for all compiler component to True expect COMPILERCOMPONENT.UNKNOWN
                        logger.info(f"\tPassed ({elapsed_time:.2f} seconds)")
                        matched_compiler_component, match_err_msg = unique_op_test_info.update_compiler_components()

                # Handle timeout exceptions if the test exceeds the allowed 60-second time limit
                except subprocess.TimeoutExpired as e:
                    elapsed_time = time.time() - start_time

                    error_message = "Test timed out after 180 seconds"
                    matched_compiler_component, match_err_msg = unique_op_test_info.update_compiler_components(
                        error_message
                    )

                    logger.info(f"\tFailed ({elapsed_time:.2f} seconds) due to {error_message}")

                    if dump_failure_logs:
                        dump_logs(log_files, error_message)

                    # Do WH warm reset (potentially hang occurred)
                    logger.info("\tWarm reset...")
                    os.system("/home/software/syseng/wh/tt-smi -lr all")

                # Handle other exceptions during unique op test execution
                except subprocess.CalledProcessError as e:
                    elapsed_time = time.time() - start_time
                    logger.info(f"\tFailed ({elapsed_time:.2f} seconds)")

                    error_message = ""
                    if e.stderr:
                        error_message += "STDERR: \n\n"
                        error_message += e.stderr
                    if e.stdout:
                        error_message += "STDOUT: \n\n"
                        error_message += e.stdout

                    # Update the UniqueOpTestInfo instance components datamember status
                    # for each compiler component and error message in failure_reason datamember
                    matched_compiler_component, match_err_msg = unique_op_test_info.update_compiler_components(
                        error_message
                    )

                    if dump_failure_logs:
                        dump_logs(log_files, error_message)

                # Handle unexpected exceptions
                except Exception as ex:
                    elapsed_time = time.time() - start_time
                    error_message = (
                        f"An unexpected error occurred while running {test}: {ex} ({elapsed_time:.2f} seconds)"
                    )
                    matched_compiler_component, match_err_msg = unique_op_test_info.update_compiler_components(
                        error_message
                    )
                    logger.info(error_message)

                    if dump_failure_logs:
                        dump_logs(log_files, error_message)

                # Update the compiler component and failure reason and model variant names inside the compiler_component_failure_analysis
                model_variant_names = [
                    model_variant_info["variant_name"] for model_variant_info in model_variant_info_list
                ]
                if matched_compiler_component is not None and match_err_msg is not None:
                    compiler_component_failure_analysis.update(
                        compiler_component=matched_compiler_component,
                        failure_reason=match_err_msg,
                        model_variant_names=model_variant_names,
                    )

                # Update model details dictionary with variant name as key and ModelVariantInfo as values
                for model_variant_info in model_variant_info_list:
                    if model_variant_info["variant_name"] in models_details.keys():
                        models_details[model_variant_info["variant_name"]].unique_ops.append(unique_op_test_info)
                    else:
                        models_details[model_variant_info["variant_name"]] = ModelVariantInfo(
                            model_name=model_variant_info["model_name"],
                            variant_name=model_variant_info["variant_name"],
                            framework=model_variant_info["framework"],
                            unique_ops=[unique_op_test_info],
                        )

    # Calculate and update the compiler support rates for each component for all the model variants
    for variant_name, model_variant_info in models_details.items():
        for compiler_component in CompilerComponent:
            compiler_component_passed_test_count = sum(
                [
                    int(unique_op_test_info.components[str(compiler_component.name)])
                    for unique_op_test_info in model_variant_info.unique_ops
                ]
            )
            total_num_of_test = len(model_variant_info.unique_ops)
            compiler_component_pass_percentage = round(
                (compiler_component_passed_test_count / total_num_of_test) * 100.0
            )
            models_details[variant_name].update_support_rate(compiler_component, compiler_component_pass_percentage)

    model_variant_info_list = list(models_details.values())

    # Sort a list of ModelVariantInfo objects first by model_name and then by variant_name.
    model_variant_info_list = sort_model_variant_info_list(model_variant_info_list)

    # Sort the failure reasons for each compiler component based on the number of associated model variant names.
    compiler_component_failure_analysis.sort_by_model_variant_names_length(reverse=True)

    return model_variant_info_list, compiler_component_failure_analysis


def generate_markdown(
    markdown_file_name: str,
    markdown_file_dir_path: str,
    table_heading: str,
    table_headers: Dict[str, List[str]],
    table_rows: List[List[str]],
    lines_after_table_heading: Optional[List[str]] = None,
):
    """
    Generates a Markdown file that contains an HTML table with the given headers and rows.
    """
    # Create a markdown file for summarizing the results for all models in a single file
    markdown_writer = MarkDownWriter(markdown_file_name, markdown_file_dir_path)

    # Write a heading for the HTML table
    markdown_writer.write_html_table_heading(table_heading)

    if lines_after_table_heading is not None:
        for line in lines_after_table_heading:
            markdown_writer.write(line)

    # Generate and write the HTML table to the Markdown file
    markdown_writer.create_html_table_and_write(headers=table_headers, rows=table_rows)

    # Close the Markdown file after writing the table
    markdown_writer.close_file()


def create_root_and_sub_markdown_file(model_variant_info_list, markdown_directory_path):
    """
    Creates root and sub Markdown files summarizing the models and their unique operation test results.

    The root Markdown file contains an overview of all models and their compiler support rates.
    The sub Markdown files contain detailed results for each model variant's unique operation tests.

    """
    # Root markdown file name and directory path for saving the md file
    root_markdown_file_name = "ModelsInfo"
    root_markdown_directory_path = markdown_directory_path

    # HTML table heading for the root markdown and sub markdown files
    root_markdown_table_heading = "List of models and current compiler support rates"
    sub_markdown_table_heading = "Unique ops configuration and compiler support info"

    # List of compiler component names for table headers
    compiler_component_names = [
        MarkDownWriter.get_component_names_for_header(compiler_component) for compiler_component in CompilerComponent
    ]

    # HTML table header for the root markdown and sub markdown files
    root_markdown_table_headers = {
        "Model Details": ["Name", "Variant", "Framework"],
        "Passing rate of unique ops for each component": compiler_component_names,
    }

    sub_markdown_table_headers = {
        "Operation Details": ["Name", "Operands", "Arguments"],
        "Component Passing Check": compiler_component_names,
        "Issues": ["Failure Reason"],
    }

    root_markdown_table_rows = []

    remove_directory(directory_path=os.path.join(markdown_directory_path, "models"))

    # Iterate over model variants to generate sub markdown files and populate root markdown rows
    for model_variant_info in model_variant_info_list:

        # Prepare the path for the sub markdown file to store test results for this model variant
        sub_markdown_file_name = model_variant_info.variant_name
        sub_markdown_directory_path = os.path.join(markdown_directory_path, "models", model_variant_info.model_name)

        # List to store table rows for the sub markdown file
        sub_markdown_table_rows = []

        # Iterate over the unique operation test information to populate table rows for sub markdown
        for unique_op_test_info in model_variant_info.unique_ops:

            table_data = [unique_op_test_info.op]
            table_data.append("<br><div align='center'>X</div>".join(unique_op_test_info.operands))
            table_data.append("<br>".join(unique_op_test_info.args))

            # If unknown compiler component is set to True in unique_op_test_info, use the unknown symbol for indicating unknown compiler component status and for other compiler components set empty string
            # else for unknown compiler component  set empty string for indicating status and for other compiler component set pass or fail symbol
            if unique_op_test_info.components[str(CompilerComponent.UNKNOWN.name)]:
                for component_name, test_status in unique_op_test_info.components.items():
                    test_status = (
                        HtmlSymbol.UNKNOWN.value if component_name == str(CompilerComponent.UNKNOWN.name) else " "
                    )
                    table_data.append(test_status)
            else:
                for component_name, test_status in unique_op_test_info.components.items():
                    if component_name == str(CompilerComponent.UNKNOWN.name):
                        test_status = " "
                    else:
                        test_status = HtmlSymbol.PASS.value if test_status else HtmlSymbol.FAIL.value
                    table_data.append(test_status)
            table_data.append(unique_op_test_info.failure_reason)
            sub_markdown_table_rows.append(table_data)

        # Generate sub markdown file that contain model variant unique op test info
        generate_markdown(
            markdown_file_name=sub_markdown_file_name,
            markdown_file_dir_path=sub_markdown_directory_path,
            table_heading=sub_markdown_table_heading,
            table_headers=sub_markdown_table_headers,
            table_rows=sub_markdown_table_rows,
        )

        # Prepare a row for the root markdown file with summary details of the model variant
        table_data = [model_variant_info.model_name]

        # Create an HTML link for the variant name, linking to its corresponding model variant markdown file
        table_data.append(
            MarkDownWriter.create_html_link(
                model_variant_info.variant_name,
                os.path.join("./models", model_variant_info.model_name, model_variant_info.variant_name + ".md"),
            )
        )

        table_data.append(model_variant_info.framework)
        for compiler_component in CompilerComponent:
            table_data.append(str(model_variant_info.get_support_rate(compiler_component)) + " %")
        root_markdown_table_rows.append(table_data)

    commit_id, commit_url = get_commit_id()
    if commit_id is None and commit_url is None:
        commit_id = "Unknown"
    else:
        commit_id = MarkDownWriter.create_html_link(link_text=commit_id, url_or_path=commit_url)

    commit_id_str = f"<p><b>Commit Id :</b> {str(commit_id)}</p>"
    last_update_datetime = (
        f'<p><b>Last updated date and time(in GMT) :</b> {time.strftime("%A, %d %b %Y %I:%M:%S %p", time.gmtime())}</p>'
    )
    statistics_report_description = '<p><b>Note:</b> For detailed insights into compiler failures and their effects on models, please refer to the <a href="./stats/compiler_analysis_report.md">compiler_analysis_report.md</a>.</p>'

    content_after_table_heading = []
    content_after_table_heading.append(last_update_datetime)
    content_after_table_heading.append(commit_id_str)
    content_after_table_heading.append(statistics_report_description)

    # Generate root markdown file that contain all the model variants result
    generate_markdown(
        markdown_file_name=root_markdown_file_name,
        markdown_file_dir_path=root_markdown_directory_path,
        table_heading=root_markdown_table_heading,
        lines_after_table_heading=content_after_table_heading,
        table_headers=root_markdown_table_headers,
        table_rows=root_markdown_table_rows,
    )


def calculate_statistical_data(model_variant_info_list):
    """
    Calculate statistical data for compiler components based on model variant information.

    Args:
        model_variant_info_list (list): A list of model variant information objects.

    Returns:
        dict: A dictionary containing statistical data for each compiler component.
              Includes models pass count, models pass percentage, and average pass percentage.
    """

    # Initialize statistical data for each compiler component expect UNKNOWN compiler component
    statistical_data = {}
    for compiler_component in CompilerComponent:
        if compiler_component != CompilerComponent.UNKNOWN:
            statistical_data[compiler_component] = {"models_pass_count": 0, "average_pass_percentage": 0.0}

    # Calculate pass counts and average pass percentage
    for model_variant_info in model_variant_info_list:
        for compiler_component in CompilerComponent:
            if compiler_component != CompilerComponent.UNKNOWN:
                compiler_support_rate = model_variant_info.get_support_rate(compiler_component)
                statistical_data[compiler_component]["average_pass_percentage"] += compiler_support_rate
                if int(compiler_support_rate) == 100:
                    statistical_data[compiler_component]["models_pass_count"] += 1

    # Calculate percentages by dividing sums by the total number of models
    for compiler_component in CompilerComponent:
        if compiler_component != CompilerComponent.UNKNOWN:
            statistical_data[compiler_component]["models_pass_percentage"] = round(
                (statistical_data[compiler_component]["models_pass_count"] / len(model_variant_info_list)) * 100.0
            )
            statistical_data[compiler_component]["average_pass_percentage"] = round(
                statistical_data[compiler_component]["average_pass_percentage"] / len(model_variant_info_list)
            )

    return statistical_data


def calculate_top_n_blocked_models(model_variant_info_list: List[ModelVariantInfo], n: int):
    """
    Calculates the top N models with the least support rate for each compiler component (excluding UNKNOWN).

    Args:
    - model_variant_info_list (List[ModelVariantInfo]): A list of ModelVariantInfo objects containing model data and support rates.
    - n (int): The number of top models to return for each compiler component.

    Returns:
    - dict: A dictionary where the keys are compiler components (excluding UNKNOWN) and the values are lists of
            the top N models' variant names, sorted by the least support rate for that component.
    """

    # Initialize an empty dictionary to store the top N blocked models for each compiler component
    compiler_top_n_blocked_models = {}

    # Iterate over each compiler component
    for compiler_component in CompilerComponent:

        # Skip the UNKNOWN compiler component as we do not need to calculate for it
        if compiler_component != CompilerComponent.UNKNOWN:

            # Sort the model variants by their support rate for the current compiler component in ascending order
            sorted_model_variant_info_list = sorted(
                model_variant_info_list,
                key=lambda model_variant_info: model_variant_info.get_support_rate(compiler_component),
            )

            # Add the top N models (least support rate) for the current compiler component to the dictionary
            compiler_top_n_blocked_models[compiler_component] = [
                model_variant_info.variant_name
                + " ("
                + str(int(round(model_variant_info.get_support_rate(compiler_component))))
                + " %)"
                for model_variant_info in sorted_model_variant_info_list[:n]
            ]

    # Return the dictionary containing the top N models for each compiler component
    return compiler_top_n_blocked_models


def create_statistics_report_markdown_file(
    model_variant_info_list, compiler_component_failure_analysis, markdown_directory_path
):

    """
    Create a markdown report summarizing compiler statistics and failure analysis.

    Args:
        model_variant_info_list (list): A list of model variant information objects.
        compiler_component_failure_analysis (CompilerComponentFailureAnalysis):
            Object containing failure analysis for compiler components.
        markdown_directory_path (str): Directory path where the markdown report will be saved.
    """

    statistics_report_markdown_file_name = "compiler_analysis_report"
    statistics_report_directory_path = os.path.join(markdown_directory_path, "stats")
    markdown_writer = MarkDownWriter(statistics_report_markdown_file_name, statistics_report_directory_path)

    # Create failure analysis table
    table_heading = "Compiler Component Failure Analysis by Model Impacts"
    table_headers = ["Compiler Component", "Failure", "Number of Impacted Models", "Impacted Models"]
    compiler_component_failure_analysis = (
        compiler_component_failure_analysis.get_compiler_component_and_failure_details()
    )
    table_rows = []
    for compiler_component, failure_details in compiler_component_failure_analysis.items():
        component_name = MarkDownWriter.get_component_names_for_header(compiler_component)
        for failure, model_variant_names in failure_details.items():
            model_variant_names_html_list = [
                f"<li>{model_variant_name}</li>" for model_variant_name in model_variant_names
            ]
            model_variant_names_html_list = "<ul>" + "".join(model_variant_names_html_list) + "</ul>"
            table_rows.append(
                [
                    component_name,
                    failure,
                    len(model_variant_names),
                    model_variant_names_html_list,
                ]
            )

    # Write the failure analysis table to markdown
    markdown_writer.write_html_table_heading(table_heading)
    markdown_writer.write_html_table(table_headers=table_headers, table_rows=table_rows, rowspan_columns=[0])

    # Create compiler statistics table
    top_blocked_models_count = 10
    table_heading = "Compiler-Specific Model Statistics"
    table_headers = {
        f"Total no of models : {len(model_variant_info_list)}": [
            "Compiler Component",
            "Models Passed",
            "Pass Rate (%)",
            "Average Pass Rate (%)",
            f"Top-{top_blocked_models_count} Blocked Models (pass rate in %)",
        ]
    }
    statistical_data = calculate_statistical_data(model_variant_info_list=model_variant_info_list)
    compiler_top_n_blocked_models = calculate_top_n_blocked_models(
        model_variant_info_list=model_variant_info_list, n=top_blocked_models_count
    )
    table_rows = []
    for compiler_component in CompilerComponent:
        if compiler_component != CompilerComponent.UNKNOWN:
            component_name = MarkDownWriter.get_component_names_for_header(compiler_component)
            blocked_model_variant_names = compiler_top_n_blocked_models[compiler_component]
            blocked_model_variant_names = [
                f"<li>{blocked_model_variant_name}</li>" for blocked_model_variant_name in blocked_model_variant_names
            ]
            blocked_model_variant_names = "<ul>" + "".join(blocked_model_variant_names) + "</ul>"
            table_rows.append(
                [
                    component_name,
                    str(statistical_data[compiler_component]["models_pass_count"]),
                    str(statistical_data[compiler_component]["models_pass_percentage"]) + " %",
                    str(statistical_data[compiler_component]["average_pass_percentage"]) + " %",
                    blocked_model_variant_names,
                ]
            )

    # Write the compiler statistics table to markdown
    markdown_writer.write_html_table_heading(table_heading)
    markdown_writer.write_html_table(table_headers=table_headers, table_rows=table_rows)

    # Close the markdown file
    markdown_writer.close_file()


def main():
    parser = argparse.ArgumentParser(
        description="""Generate unique ops test for the models present in the test_directory_or_file_path
        specified by the user and run the unique ops test and generate markdown files, the root markdown file contains model name,
        variant name, framework and compiler components supported rate and sub-markdown file contains model variant unique op tests info"""
    )

    parser.add_argument(
        "--test_directory_or_file_path",
        type=str,
        default=os.path.join(os.getcwd(), "forge/test/models"),
        help="Specify the directory or file path containing models test",
    )
    parser.add_argument(
        "--dump_failure_logs",
        action="store_true",
        help="Specify the flag to dump the unique ops test failure logs.",
    )
    parser.add_argument(
        "--markdown_directory_path",
        default=os.path.join(os.getcwd(), "model_analysis_docs"),
        required=False,
        help="Specify the directory path for saving models information as markdowns file",
    )
    parser.add_argument(
        "--unique_ops_output_directory_path",
        default=os.path.join(os.getcwd(), "models_unique_ops_output"),
        required=False,
        help="Specify the output directory path for saving models unique op tests outputs(i.e failure logs, xlsx file)",
    )

    args = parser.parse_args()

    model_output_dir_paths = generate_and_export_unique_ops_tests(
        test_directory_or_file_path=args.test_directory_or_file_path,
        unique_ops_output_directory_path=args.unique_ops_output_directory_path,
    )

    unique_ops_config_across_all_models_file_path = os.path.join(
        args.unique_ops_output_directory_path, "extracted_unique_op_config_across_all_models.log"
    )
    unique_operations = extract_unique_op_tests_from_models(
        model_output_dir_paths=model_output_dir_paths,
        unique_ops_config_file_path=unique_ops_config_across_all_models_file_path,
    )

    model_variant_info_list, compiler_component_failure_analysis = run_models_unique_op_tests(
        unique_operations=unique_operations,
        unique_ops_output_directory_path=args.unique_ops_output_directory_path,
        dump_failure_logs=args.dump_failure_logs,
    )

    create_root_and_sub_markdown_file(
        model_variant_info_list=model_variant_info_list, markdown_directory_path=args.markdown_directory_path
    )

    create_statistics_report_markdown_file(
        model_variant_info_list=model_variant_info_list,
        compiler_component_failure_analysis=compiler_component_failure_analysis,
        markdown_directory_path=args.markdown_directory_path,
    )


if __name__ == "__main__":
    main()
