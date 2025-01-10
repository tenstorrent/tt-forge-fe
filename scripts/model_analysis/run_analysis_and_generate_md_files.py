# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import subprocess
import os
import time
import json
from loguru import logger
import math
import argparse
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass
import ast

import torch

from forge.tvm_unique_op_generation import Operation, NodeType, UniqueOperations

from exception_rules import common_failure_matching_rules_list
from markdown import HtmlSymbol, MarkDownWriter
from unique_ops_utils import generate_and_export_unique_ops_tests, extract_unique_op_tests_from_models
from utils import CompilerComponent, check_path, dump_logs, remove_directory


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
        else:
            # If no error message is provided, mark all compiler components (except UNKNOWN) to True.
            for compiler_component in CompilerComponent:
                if compiler_component != CompilerComponent.UNKNOWN:
                    self.components[str(compiler_component.name)] = True

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
    last_update_datetime: str = ""

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
        model_variant_info += f"\t\tlast_update_datetime : {last_update_datetime}\n"
        for idx, unique_op in enumerate(unique_ops):
            model_variant_info += f"\t\t\t\t{idx}){str(unique_op)}\n"


def run_models_unique_op_tests(unique_operations, unique_ops_output_directory_path, dump_failure_logs):
    """
    Run unique op configuration test that has been collected across all the models and populate the test result in the model variants
    """

    models_details = {}

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
                        unique_op_test_info.update_compiler_components(error_message)

                        # Save failure logs if dump_failure_logs is set to True
                        if dump_failure_logs:
                            dump_logs(log_files, error_message)

                    else:
                        # If the test passed (return code is 0), update the UniqueOpTestInfo instance
                        # components datamember for all compiler component to True expect COMPILERCOMPONENT.UNKNOWN
                        logger.info(f"\tPassed ({elapsed_time:.2f} seconds)")
                        unique_op_test_info.update_compiler_components()

                # Handle timeout exceptions if the test exceeds the allowed 60-second time limit
                except subprocess.TimeoutExpired as e:
                    elapsed_time = time.time() - start_time

                    error_message = "Test timed out after 180 seconds"
                    unique_op_test_info.update_compiler_components(error_message)

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
                    unique_op_test_info.update_compiler_components(error_message)

                    if dump_failure_logs:
                        dump_logs(log_files, error_message)

                # Handle unexpected exceptions
                except Exception as ex:
                    elapsed_time = time.time() - start_time
                    error_message = (
                        f"An unexpected error occurred while running {test}: {ex} ({elapsed_time:.2f} seconds)"
                    )
                    unique_op_test_info.update_compiler_components(error_message)
                    logger.info(error_message)

                    if dump_failure_logs:
                        dump_logs(log_files, error_message)

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
            compiler_component_pass_percentage = (
                str(math.ceil((compiler_component_passed_test_count / total_num_of_test) * 100.0)) + " %"
            )
            models_details[variant_name].update_support_rate(compiler_component, compiler_component_pass_percentage)

        models_details[variant_name].last_update_datetime = time.strftime("%A, %d %b %Y %I:%M:%S %p", time.gmtime())

    return models_details


def generate_markdown(
    markdown_file_name: str,
    markdown_file_dir_path: str,
    table_heading: str,
    table_headers: Dict[str, List[str]],
    table_rows: List[List[str]],
):
    """
    Generates a Markdown file that contains an HTML table with the given headers and rows.
    """
    # Create a markdown file for summarizing the results for all models in a single file
    markdown_writer = MarkDownWriter(markdown_file_name, markdown_file_dir_path)

    # Write a heading for the HTML table
    markdown_writer.write_html_table_heading(table_heading)

    # Generate and write the HTML table to the Markdown file
    markdown_writer.create_html_table_and_write(headers=table_headers, rows=table_rows)

    # Close the Markdown file after writing the table
    markdown_writer.close_file()


def create_root_and_sub_markdown_file(models_details, markdown_directory_path):
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
        "Last update(in GMT)": ["Date & time"],
    }

    sub_markdown_table_headers = {
        "Operation Details": ["Name", "Operands", "Arguments"],
        "Component Passing Check": compiler_component_names,
        "Issues": ["Failure Reason"],
    }

    root_markdown_table_rows = []

    remove_directory(directory_path=os.path.join(markdown_directory_path, "Models"))

    # Iterate over model variants to generate sub markdown files and populate root markdown rows
    for model_variant_info in models_details.values():

        # Prepare the path for the sub markdown file to store test results for this model variant
        sub_markdown_file_name = model_variant_info.variant_name
        sub_markdown_directory_path = os.path.join(markdown_directory_path, "Models", model_variant_info.model_name)

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
                os.path.join("./Models", model_variant_info.model_name, model_variant_info.variant_name + ".md"),
            )
        )

        table_data.append(model_variant_info.framework)
        for compiler_component in CompilerComponent:
            table_data.append(model_variant_info.get_support_rate(compiler_component))
        table_data.append(model_variant_info.last_update_datetime)
        root_markdown_table_rows.append(table_data)

    # Generate root markdown file that contain all the model variants result
    generate_markdown(
        markdown_file_name=root_markdown_file_name,
        markdown_file_dir_path=root_markdown_directory_path,
        table_heading=root_markdown_table_heading,
        table_headers=root_markdown_table_headers,
        table_rows=root_markdown_table_rows,
    )


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

    models_details = run_models_unique_op_tests(
        unique_operations=unique_operations,
        unique_ops_output_directory_path=args.unique_ops_output_directory_path,
        dump_failure_logs=args.dump_failure_logs,
    )

    create_root_and_sub_markdown_file(
        models_details=models_details, markdown_directory_path=args.markdown_directory_path
    )


if __name__ == "__main__":
    main()
