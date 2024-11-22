# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import subprocess
import os
import time
from loguru import logger
import math
import argparse
import pandas as pd
from tabulate import tabulate
from enum import IntEnum
from typing import Union, Dict, List, Tuple
from dataclasses import dataclass, asdict
import inspect

# Unicode for symbols
# pass_symbol = "\u2705"        # Checkmark
# fail_symbol = "\u274C"        # Crossmark
# unknown_symbol = "\uFFFD"     # Question mark

# Html symbols:
pass_symbol = "&#x2705;"  # Checkmark
fail_symbol = "&#x274C;"  # Crossmark
unknown_symbol = "&#xFFFD;"  # Question mark


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
                    self.compiler_component.name + " "
                    if rule.rule_name is None or rule.rule_name == ""
                    else self.compiler_component.name + ": " + rule.rule_name + " "
                )
                match_err_msg += rule.match_rule(exception)
                return self.compiler_component, match_err_msg

        return None, None


common_failure_matching_rules_list = [
    MatchingCompilerComponentException(
        CompilerComponent.FORGE,
        [
            MatchingExceptionRule(
                "forge_module evaluation", ["AssertionError", "Setting a tensor value of incorrect shape"]
            ),
            MatchingExceptionRule(
                "post_initial_graph_passes",
                [
                    "RuntimeError",
                    "has_newstyle_interface(std::get<std::string>(type), false)",
                    "decomposing a type with old OpType interface, expects new OpType interface",
                ],
            ),
            MatchingExceptionRule(
                "lower_to_mlir",
                ["RuntimeError", "Found Unsupported operations while lowering from TTForge to TTIR in forward graph"],
            ),
            MatchingExceptionRule(
                "mlir generation failure", ["RuntimeError", "Generated MLIR module failed verification"]
            ),
            MatchingExceptionRule(
                "Convert tt-forge attribute to an MLIR attribute", ["RuntimeError", "Unhandled attribute type"]
            ),
            MatchingExceptionRule(
                "ttmetal vs Forge Output Data mismatch",
                ["AssertionError", "assert False", "where False = all([False])"],
            ),
            # Compiled model Runtime
            MatchingExceptionRule(
                "Runtime Data mismatch", ["RuntimeError", "Tensor", "data type mismatch: expected", "got"]
            ),
            MatchingExceptionRule(
                "Runtime Shape mismatch", ["RuntimeError", "Tensor", "shape mismatch: expected", "got"]
            ),
            MatchingExceptionRule(
                "Runtime stride mismatch", ["RuntimeError", "Tensor", "stride mismatch: expected", "got"]
            ),
            MatchingExceptionRule(
                "Runtime Input count mismatch", ["RuntimeError", "Input count mismatch: expected", "got"]
            ),
            MatchingExceptionRule(
                "post_const_eval_tensors", ["RuntimeError", "unsupported memory format option Contiguous"]
            ),
        ],
    ),
    MatchingCompilerComponentException(
        CompilerComponent.MLIR,
        [
            MatchingExceptionRule(
                "ttnn.reshape mlir pipeline",
                [
                    "RuntimeError",
                    "'ttnn.reshape' op Shape attribute size must match output tensor rank",
                    "Failed to run MLIR compiler pass pipeline",
                ],
            ),
            MatchingExceptionRule(
                "ttnn.maxpool2d mlir pipeline",
                [
                    "RuntimeError",
                    "ttnn.max_pool2d currently only supports an input type of bfloat16",
                    "Failed to run MLIR compiler pass pipeline",
                ],
            ),
            MatchingExceptionRule("mlir pipeline", ["RuntimeError", "Failed to run MLIR compiler pass pipeline"]),
        ],
    ),
    MatchingCompilerComponentException(
        CompilerComponent.TT_METAL,
        [
            MatchingExceptionRule(
                "ttnn.tilize validation",
                [
                    "RuntimeError",
                    "tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_op.cpp",
                    "input_tensor_a.get_dtype() == DataType::BFLOAT16",
                ],
            ),
            MatchingExceptionRule(
                "ttnn.tilize_with_val_padding validation",
                [
                    "RuntimeError",
                    "tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp",
                    "input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32",
                    "Can only tilize bfloat16 or uint32 tensors",
                ],
            ),
            MatchingExceptionRule(
                "ttnn.embedding validation",
                [
                    "RuntimeError",
                    "tt-metal/ttnn/cpp/ttnn/operations/embedding/device/embedding_device_operation.cpp",
                    "weights.get_dtype() == DataType::BFLOAT16",
                ],
            ),
            MatchingExceptionRule(
                "ttnn.embedding validation",
                [
                    "RuntimeError",
                    "tt-metal/ttnn/cpp/ttnn/operations/embedding/device/embedding_device_operation.cpp",
                    "a.get_dtype() == DataType::UINT32 or a.get_dtype() == DataType::BFLOAT16",
                    "Input must be UINT32 or BFLOAT16",
                ],
            ),
            MatchingExceptionRule(
                "ttnn elementwise binary", ["RuntimeError", "BinaryOpType cannot be mapped to BcastOpMath"]
            ),
            MatchingExceptionRule(
                "ttnn.concat validation",
                ["RuntimeError", "Tile padding along concatenated dim", "not supported for concat yet"],
            ),
            MatchingExceptionRule(
                "ttnn.reshape validation",
                [
                    "RuntimeError",
                    "tt-metal/ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/device/reshape_op.cpp",
                    "input_tensor_a.get_dtype() == DataType::BFLOAT16",
                ],
            ),
            MatchingExceptionRule(
                "ttnn.matmul",
                [
                    "RuntimeError",
                    "tt-metal/ttnn/cpp/ttnn/operations/matmul/device/matmul_op_multi_core_reuse_program_factory.cpp",
                    "Mt % per_core_M == 0",
                ],
            ),
            MatchingExceptionRule(
                "ttnn.reshape",
                [
                    "RuntimeError",
                    "tt-metal/ttnn/cpp/ttnn/tensor/tensor_utils.cpp",
                    "new_volume == old_volume",
                    "Invalid arguments to reshape",
                ],
            ),
            MatchingExceptionRule("ttmetal allocations", ["RuntimeError", "Statically allocated circular buffers"]),
            MatchingExceptionRule(
                "ttmetal allocations",
                [
                    "RuntimeError",
                    "tt-metal/tt_metal/impl/allocator/allocator.cpp",
                    "Out of Memory: Not enough space to allocate",
                ],
            ),
        ],
    ),
]


@dataclass
class ModelVariantInfo:
    """
    Stores information about a model, variant, framework of the model, including its support rates for different compiler components.

    Attributes:
        model_name (str): The name of the model.
        variant_name (str): The name of the model variant.
        framework (str): The framework used for the model.
        forge_support_rate (float): The support rate for the Forge compiler component. Defaults to 0.0.
        mlir_support_rate (float): The support rate for the MLIR compiler component. Defaults to 0.0.
        ttmetal_support_rate (float): The support rate for the TT_METAL compiler component. Defaults to 0.0.
        unknown_rate (float): The support rate for an unknown compiler component. Defaults to 0.0.
    """

    model_name: str
    variant_name: str
    framework: str
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


class UniqueOpTestInfo:
    """
    Represents information about a unique operation test, that includes op name, operands
    arguments, and the status of various compiler components.

    Attributes:
        Op (str): The name of the operation.
        Operands (str): The operands associated with the operation.
        Args (str): Operation Arguments if any
        components (dict): A dictionary indicating the support status for each compiler component.
        failure_reason (str): The reason for failure, if any, during testing.
    """

    def __init__(
        self,
        Op: str,
        Operands: str,
        Args: str,
    ):
        self.Op = str(Op)
        self.Operands = str(Operands)
        self.Args = " " if pd.isna(Args) else str(Args)
        self.components = {}
        for compiler_component in CompilerComponent:
            self.components[str(compiler_component.name)] = False
        self.failure_reason = ""

    @classmethod
    def create_from_dict(cls, data: Dict[str, str]):

        # Extract the names of parameters for the __init__ method (excluding 'self').
        unique_op_test_info_params = list(inspect.signature(cls.__init__).parameters.keys())[1:]

        # Filter the dictionary to include only relevant keys for initialization.
        unique_op_test_data = {key: data[key] for key in unique_op_test_info_params if key in data}

        # Create and return an instance of the UniqueOpTestInfo class.
        return cls(**unique_op_test_data)

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
        return f"UniqueOpTestInfo(op={self.Op}, Operands={self.Operands}, Args={self.Args}, components={self.components}, self.failure_reason={self.failure_reason})"


class MarkDownWriter:
    """
    A utility class for writing Markdown files, including headings, tables, and links.

    Attributes:
        markdown_file_name (str): The name of the Markdown file (without extension).
        markdown_file_dir_path (str): The directory path where the Markdown file is created.
    """

    def __init__(self, markdown_file_name: str, markdown_file_dir_path: str = None, open_file: bool = True):
        self.markdown_file_name = markdown_file_name
        self.markdown_file = self.markdown_file_name + ".md"
        if markdown_file_dir_path is not None:
            self.markdown_file_dir_path = markdown_file_dir_path
        else:
            self.markdown_file_dir_path = os.getcwd()
        os.makedirs(self.markdown_file_dir_path, exist_ok=True)
        if open_file:
            self.file = open(os.path.join(self.markdown_file_dir_path, self.markdown_file), "w")

    def write(self, data: str):
        self.file.write(data)

    def write_line(self, data: str):
        self.write(data + "\n")

    def write_table_heading(self, table_heading: str, heading_rank: int = 1):
        table_heading = str("#" * heading_rank) + " " + table_heading
        self.write_line(table_heading)

    def write_table(self, headers, rows):
        # Create a Markdown table using the tabulate library with GitHub-flavored table formatting.
        markdown_table = tabulate(rows, headers, tablefmt="github", colalign=("center",) * len(headers))
        self.write_line(markdown_table)

    def get_component_names_for_header(self, compiler_component: CompilerComponent):
        if compiler_component == CompilerComponent.FORGE:
            return "Forge-Fe"
        elif compiler_component == CompilerComponent.MLIR:
            return "MLIR"
        elif compiler_component == CompilerComponent.TT_METAL:
            return "Metalium"
        elif compiler_component == CompilerComponent.UNKNOWN:
            return "N/A"
        else:
            logger.error(f"There is no compilercomponent {compiler_component.name}")

    def write_html_table_heading(self, table_heading: str, heading_rank: int = 1):
        table_heading = f"<h{heading_rank}>{table_heading}</h{heading_rank}>"
        self.write_line(table_heading)

    def create_html_table_and_write(self, headers: Dict[str, List[str]], rows: List[List[str]]):
        sub_headers = []
        for headers_list in headers.values():
            sub_headers.extend(headers_list)

        sub_header_row_data_length_match = all([True if len(row) == len(sub_headers) else False for row in rows])

        assert sub_header_row_data_length_match, "Sub headers and table row length is not matched"

        table_df = pd.DataFrame(data=rows, columns=sub_headers)

        top_headers = [
            (main_header, sub_header) for main_header, sub_headers in headers.items() for sub_header in sub_headers
        ]
        table_df.columns = pd.MultiIndex.from_tuples(top_headers)

        html_table = table_df.to_html(index=False, na_rep=" ", justify="center", escape=False)

        self.write_line(html_table)

    @classmethod
    def create_md_link(cls, link_text: str, url_or_path: str):
        return f"[{link_text}]({url_or_path})"

    @classmethod
    def create_html_link(cls, link_text: str, url_or_path: str):
        return f'<a href="{url_or_path}">{link_text}</a>'

    def close_file(self):
        self.file.close()


def dump_logs(log_file_dir_path: str, log_file_name: str, content: str):
    os.makedirs(log_file_dir_path, exist_ok=True)
    log_file = os.path.join(log_file_dir_path, log_file_name)
    with open(log_file, "w") as f:
        f.write(content)
        logger.info(f"Dumped test logs in {log_file}")


def collect_all_pytests(root_dir_path):

    logger.info(f"Collecting all pytests in {root_dir_path}")

    try:
        res = subprocess.check_output(["pytest", root_dir_path, "--setup-plan"], stderr=subprocess.STDOUT).decode(
            "utf-8"
        )
    except subprocess.CalledProcessError as e:
        output = e.output.decode("utf-8")
        logger.error(f"[Error!] output = {output}")
        return []

    test_list = []
    lines = res.split("\n")
    for line in lines:
        if "warnings summary" in line or "slowest durations" in line:
            break

        if line and line.startswith("        " + root_dir_path) and "::" in line and "training" not in line:
            line = line.strip()
            line = line.split(" (fixtures used:")[0] if " (fixtures used:" in line else line
            if "Grayskull" not in line and "Wormhole_B0" not in line:
                test_list.append(line)

    return test_list


def generate_and_export_unique_ops_tests(pytest_directory_path, model_file_path, unique_ops_output_directory_path):

    # If model_file_path is specified, collect all the tests in the model_file_path parent directory path
    # and in the test_list will only include the tests matching with model_file_path,
    # otherwise collect all the tests in the pytest_directory_path specified by the user
    if model_file_path:
        model_file_path_list = model_file_path.split("/")[:-1]
        tests_directory_path = "/".join(model_file_path_list)
    else:
        tests_directory_path = pytest_directory_path

    test_list = collect_all_pytests(tests_directory_path)

    if model_file_path:
        test_list = [test for test in test_list if test.split("::")[0] == model_file_path]

    assert test_list != [], f"No tests found in the {tests_directory_path} path"

    # Create a dictonary contains model_name as key and model tests(i.e include variant, task) as values
    model_name_to_tests = {}
    for test in test_list:
        model_name = test.split("::")[0].split("/")[-1].replace(".py", "").replace("test_", "")
        if model_name not in model_name_to_tests.keys():
            model_name_to_tests[model_name] = [test]
        else:
            model_name_to_tests[model_name].append(test)

    # Generate unique op test for the all collected test and save the models unique ops test information in the unique_ops_output_directory_path
    model_output_dir_paths = []
    for model_name, tests in model_name_to_tests.items():
        model_output_dir_path = os.path.join(unique_ops_output_directory_path, model_name)
        os.makedirs(model_output_dir_path, exist_ok=True)
        model_output_dir_paths.append(model_output_dir_path)
        for test in tests:
            logger.info(f"Running the tests : {test}")
            try:
                result = subprocess.run(
                    ["pytest", test, "-vss", "--generate-unique-op-tests"],
                    capture_output=True,
                    text=True,
                    check=True,
                    env=dict(
                        os.environ,
                        FORGE_EXPORT_TVM_GENERATED_UNIQUE_OP_TESTS_DETAILS="1",
                        FORGE_EXPORT_TVM_GENERATED_UNIQUE_OP_TESTS_DETAILS_DIR_PATH=model_output_dir_path,
                    ),
                )
                if result.returncode != 0:
                    logger.error(f"Error while running the pytest:\n stdout: {result.stdout}\n stderr: {result.stderr}")
                else:
                    logger.info(f"Successfully generated and exported unique ops test")

            except subprocess.CalledProcessError as e:
                logger.error(f"Error while running the pytest:\n {e.output}")

    return model_output_dir_paths


def run_model_unique_op_tests_and_generate_markdowns(
    model_output_dir_paths, markdown_directory_path, dump_failure_logs
):
    """
    Execute unique operation tests for specified models, gather compiler support details and
    generate detailed Markdown reports summarizing the results.

    Workflow:
        1. Locate and Process Model Variants:
            - Iterate through the list of model directories (`model_output_dir_paths`).
            - For each model:
                - Identify its variants by listing the contents of the model directory.
                - Search for the unique operation tests information file (`.xlsx`) for each variant.
        2. Extract test details and Run unique operation tests:
            - Load the `.xlsx` file for each model variant and extract the operation test information (e.g., framework, ops, operands, args, and test files).
            - For each test in the extracted details:
                - Execute the test and track pass or fail status for each compiler component (e.g., Forge-FE, MLIR, Metal) based on the test results.
                - if `dump_failure_logs` is True, save the failure logs
            - Calculate the percentage of successful tests for each compiler component.
        3. Generate Markdown Reports:
            - Sub Markdown Files:
                - Create a Markdown file for each model variant.
                - Include details about unique operation configurations, pass/fail status for each compiler component, and failure reasons.
            - Root Markdown File:
                - Summarize the results for all models in a single file (`ModelsInfo.md`).
                - Include details such as the model name, its variants, framework, and passing rate percentages for each compiler component.
    """

    # List to store information about all processed model variants
    models_details = []

    # Iterate through all provided model directories
    for model_output_dir_path in model_output_dir_paths:

        # Extract the model name from the directory path
        model_name = model_output_dir_path.split("/")[-1]

        # List all model variants in the directory
        model_variants = os.listdir(model_output_dir_path)

        # Process each model variant
        for model_variant in model_variants:

            model_variant_dir_path = os.path.join(model_output_dir_path, model_variant)

            # Look for a single `.xlsx` file containing unique operation test details
            model_variant_tvm_generated_op_test_file = [
                f for f in os.listdir(model_variant_dir_path) if f.endswith(".xlsx")
            ]
            if len(model_variant_tvm_generated_op_test_file) != 1:
                continue

            # Read the `.xlsx` file for the model variant
            model_variant_tvm_generated_op_test_file_path = os.path.join(
                model_variant_dir_path, model_variant_tvm_generated_op_test_file[0]
            )
            model_variant_df = pd.read_excel(
                model_variant_tvm_generated_op_test_file_path,
                header=0,
                usecols=["Framework", "Op", "Operands", "Args", "Testfile"],
            )

            # Create a UniqueOpTestInfo object to store details about model and variant name and framework of the model variant.
            model_variant_info: ModelVariantInfo = ModelVariantInfo(
                model_name=model_name,
                variant_name=model_variant,
                framework=model_variant_df["Framework"].unique()[0],
            )

            # List to store unique operation test results for the model variant
            model_variant_unique_op_tests_info = []

            # Iterate over each row in the DataFrame (each row corresponds to a test for a specific operation)
            for index, row in model_variant_df.iterrows():
                row = row.to_dict()

                # Create a UniqueOpTestInfo object to store details about the operation (name, operands, args)
                unique_op_test_info = UniqueOpTestInfo.create_from_dict(row)

                # Extract the test file path
                test = row["Testfile"]
                logger.info(f"Running the test: {test}")

                # If dump_failure_logs is set to True, prepare the log file path for storing logs
                if dump_failure_logs:
                    op_name = row["Op"]  # Get the operation name
                    log_file_dir_path = os.path.join(model_variant_dir_path, op_name)

                    test_name = test.split("::")[-1]  # Extract the test name from the test path
                    log_file_name = str(test_name) + "_log.txt"

                # Start the timer to measure test execution time
                start_time = time.time()

                try:
                    # Run the unique op test by using subprocess libary run method.
                    result = subprocess.run(
                        ["pytest", test, "-vss"], check=True, capture_output=True, text=True, timeout=60
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
                            dump_logs(log_file_dir_path, log_file_name, error_message)

                    else:
                        # If the test passed (return code is 0), update the UniqueOpTestInfo instance
                        # components datamember for all compiler component to True expect COMPILERCOMPONENT.UNKNOWN
                        logger.info(f"\tPassed ({elapsed_time:.2f} seconds)")
                        unique_op_test_info.update_compiler_components()

                # Handle timeout exceptions if the test exceeds the allowed 60-second time limit
                except subprocess.TimeoutExpired as e:
                    elapsed_time = time.time() - start_time

                    error_message = "Test timed out after 60 seconds"
                    unique_op_test_info.update_compiler_components(error_message)

                    logger.info(f"\tFailed ({elapsed_time:.2f} seconds) due to {error_message}")

                    if dump_failure_logs:
                        dump_logs(log_file_dir_path, log_file_name, error_message)

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
                        dump_logs(log_file_dir_path, log_file_name, error_message)

                # Handle unexpected exceptions
                except Exception as ex:
                    elapsed_time = time.time() - start_time
                    error_message = (
                        f"An unexpected error occurred while running {test}: {ex} ({elapsed_time:.2f} seconds)"
                    )
                    unique_op_test_info.update_compiler_components(error_message)
                    logger.info(error_message)

                # Append the current test's info to the list of tests for this model variant
                model_variant_unique_op_tests_info.append(unique_op_test_info)

            # Prepare the path for the Markdown file to store test results for this model variant
            model_variant_md_file_directory_path = os.path.join(markdown_directory_path, "Models", model_name)

            # Create a Markdown file for saving model variant unique op test information
            markdown_writer = MarkDownWriter(model_variant, model_variant_md_file_directory_path)

            # Write a heading for the HTML table in the model variant markdown file
            markdown_writer.write_html_table_heading("Unique ops configuration and compiler support info")

            # Get the list of compiler component names to use in the table header
            compiler_component_names = [
                markdown_writer.get_component_names_for_header(compiler_component)
                for compiler_component in CompilerComponent
            ]

            # Define the table header with three main sections
            table_header = {
                "Operation Details": ["Name", "Operands", "Arguments"],
                "Component Passing Check": compiler_component_names,
                "Issues": ["Failure Reason"],
            }

            # List to store table rows
            table_rows = []

            # Iterate over the unique operation test information to populate table rows
            for unique_op_test_info in model_variant_unique_op_tests_info:

                # Replace newline in Operands with line breaker and X character and Arguments with line breaker
                unique_op_test_info.Operands = unique_op_test_info.Operands.replace(
                    "\n", "<br><div align='center'>X</div>"
                )
                unique_op_test_info.Args = unique_op_test_info.Args.replace("\n", "<br>")
                table_data = [unique_op_test_info.Op, unique_op_test_info.Operands, unique_op_test_info.Args]

                # If unknown compiler component is set to True in unique_op_test_info, use the unknown symbol for indicating unknown compiler component status and for other compiler components set empty string
                # else for unknown compiler component  set empty string for indicating status and for other compiler component set pass or fail symbol
                if unique_op_test_info.components[str(CompilerComponent.UNKNOWN.name)]:
                    for component_name, test_status in unique_op_test_info.components.items():
                        test_status = unknown_symbol if component_name == str(CompilerComponent.UNKNOWN.name) else " "
                        table_data.append(test_status)
                else:
                    for component_name, test_status in unique_op_test_info.components.items():
                        if component_name == str(CompilerComponent.UNKNOWN.name):
                            test_status = " "
                        else:
                            test_status = pass_symbol if test_status else fail_symbol
                        table_data.append(test_status)
                table_data.append(unique_op_test_info.failure_reason)
                table_rows.append(table_data)

            # Create and write the HTML table to the Markdown file
            markdown_writer.create_html_table_and_write(headers=table_header, rows=table_rows)

            # Close the Markdown file after writing the table
            markdown_writer.close_file()

            # Calculate and update the compiler support rates for each component
            for compiler_component in CompilerComponent:
                compiler_component_passed_test_count = sum(
                    [
                        int(unique_op_test_info.components[str(compiler_component.name)])
                        for unique_op_test_info in model_variant_unique_op_tests_info
                    ]
                )
                total_num_of_test = len(model_variant_unique_op_tests_info)
                compiler_component_pass_percentage = (
                    str(math.ceil((compiler_component_passed_test_count / total_num_of_test) * 100.0)) + " %"
                )
                model_variant_info.update_support_rate(compiler_component, compiler_component_pass_percentage)

            # Append the model variant information to the list
            models_details.append(model_variant_info)

    # Create a markdown file for summarizing the results for all models in a single file
    markdown_writer = MarkDownWriter("ModelsInfo", markdown_directory_path)

    # Write a heading for the HTML table
    markdown_writer.write_html_table_heading("List of models and current compiler support rates")

    # Get the list of compiler component names to use in the table header
    compiler_component_names = [
        markdown_writer.get_component_names_for_header(compiler_component) for compiler_component in CompilerComponent
    ]

    # Define the table header with two main sections
    table_header = {
        "Model Details": ["Name", "Variant", "Framework"],
        "Passing rate of unique ops for each component": compiler_component_names,
    }

    # List to store table rows
    table_rows = []

    # Iterate over the model variant information list to populate table rows
    for model_variant_info in models_details:

        # Create an HTML link for the variant name, linking to its corresponding model variant markdown file
        model_variant_info.variant_name = MarkDownWriter.create_html_link(
            model_variant_info.variant_name,
            os.path.join("./Models", model_variant_info.model_name, model_variant_info.variant_name + ".md"),
        )
        table_data = [model_variant_info.model_name, model_variant_info.variant_name, model_variant_info.framework]
        for compiler_component in CompilerComponent:
            table_data.append(model_variant_info.get_support_rate(compiler_component))
        table_rows.append(table_data)

    # Generate and write the HTML table to the Markdown file
    markdown_writer.create_html_table_and_write(headers=table_header, rows=table_rows)

    # Close the Markdown file after writing the table
    markdown_writer.close_file()


def main():
    parser = argparse.ArgumentParser(
        description="""Generate unique ops test for the models present in the  pytest_directory_path or model_file_path
        specified by the user and run the unique ops test and generate markdown files, the root markdown file contains model name,
        variant name, framework and compiler components supported rate and sub-markdown file contains  model variant unique op tests info"""
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--pytest_directory_path",
        type=str,
        help="Specify the directory path containing models to test.",
    )
    group.add_argument(
        "--model_file_path",
        type=str,
        help="Specify the model file path to generate unique op tests and markdown file.",
    )
    parser.add_argument(
        "--dump_failure_logs",
        action="store_true",
        help="Specify the flag to dump the unique ops test failure logs.",
    )
    parser.add_argument(
        "--markdown_directory_path",
        default=os.path.join(os.getcwd(), "models_analysis_docs"),
        required=False,
        help="Specify the directory path for saving models information as markdowns file",
    )
    parser.add_argument(
        "--unique_ops_output_directory_path",
        default=os.path.join(os.getcwd(), "unique_ops"),
        required=False,
        help="Specify the output directory path for saving models unique op tests outputs(i.e failure logs, xlsx file)",
    )

    args = parser.parse_args()

    model_output_dir_paths = generate_and_export_unique_ops_tests(
        pytest_directory_path=args.pytest_directory_path,
        model_file_path=args.model_file_path,
        unique_ops_output_directory_path=args.unique_ops_output_directory_path,
    )
    logger.info(f"model_output_dir_paths={model_output_dir_paths}")
    run_model_unique_op_tests_and_generate_markdowns(
        model_output_dir_paths=model_output_dir_paths,
        markdown_directory_path=args.markdown_directory_path,
        dump_failure_logs=args.dump_failure_logs,
    )


if __name__ == "__main__":
    main()
