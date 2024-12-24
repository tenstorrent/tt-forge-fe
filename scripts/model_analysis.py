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
from tabulate import tabulate
from enum import IntEnum, Enum
from typing import Union, Dict, List, Tuple
from dataclasses import dataclass, asdict
import inspect
import ast

import torch

from forge.tvm_unique_op_generation import Operation, NodeType, UniqueOperations


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


common_failure_matching_rules_list = [
    MatchingCompilerComponentException(
        CompilerComponent.FORGE,
        [
            MatchingExceptionRule(
                "forge_module evaluation", ["AssertionError", "Setting a tensor value of incorrect shape"]
            ),
            MatchingExceptionRule(
                "embedding indicies tensor",
                ["IndexError", "forge/forge/op/eval/forge/embedding.py", "index out of range in self"],
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
                "lower_to_mlir",
                ["RuntimeError", "Unsupported data format during lowering from TTForge to TTIR"],
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
            MatchingExceptionRule(
                "Forge Verification Data mismatch",
                ["ValueError", "forge/forge/verify/verify.py", "Data mismatch (compare_with_golden)"],
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
                "TTIR to TTNN Conv2dOpConversionPattern",
                [
                    "tt_forge_signal_handler",
                    "tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp",
                    "Conv2dOpConversionPattern::matchAndRewrite(ttir::Conv2dOp, OpAdaptor, ConversionPatternRewriter &)",
                    "adaptor.getPaddingBottom() == adaptor.getPaddingTop()",
                    "TTNN only supports padding height/width attributes. Thus, padding_top",
                    "must equal padding_bottom for the op to execute as expected",
                ],
            ),
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
            MatchingExceptionRule(
                "MLIR runtime ttnn ", ["tt::exception", "tt-mlir/runtime/lib/ttnn/runtime.cpp", "Unsupported data type"]
            ),
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
                "ttnn elementwise binary",
                [
                    "RuntimeError",
                    "tt-metal/ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_device_operation.cpp",
                    "ttnn::operations::binary::BinaryDeviceOperation: unsupported broadcast",
                ],
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
                "ttnn.matmul",
                [
                    "RuntimeError",
                    "tt-metal/ttnn/cpp/ttnn/operations/matmul/device/matmul_op_multi_core_reuse_program_factory.cpp",
                    "Nt % per_core_N == 0",
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
            MatchingExceptionRule(
                "ttnn.reshape",
                [
                    "RuntimeError",
                    "tt-metal/ttnn/cpp/ttnn/operations/data_movement/reshape_view/reshape.cpp",
                    "tensor_shape.rank() <= 4",
                    "Only up to 4D tensors",
                ],
            ),
            MatchingExceptionRule(
                "ttnn permute",
                [
                    "RuntimeError",
                    "tt-metal/ttnn/cpp/ttnn/operations/data_movement/permute/device/permute_device_operation.cpp",
                    "attributes.dims.back() == tensor_args.input_tensor.get_logical_shape().rank() - 1",
                    "Last dimension of permute must be the last dimension of the input tensor as page-breaking is not supported at the moment",
                ],
            ),
            MatchingExceptionRule(
                "ttnn.pad",
                [
                    "RuntimeError",
                    "tt-metal/ttnn/cpp/ttnn/operations/data_movement/pad/pad.cpp",
                    "Tensor rank is not 4",
                ],
            ),
            MatchingExceptionRule(
                "TTNN tensor types",
                [
                    "RuntimeError",
                    "tt-metal/ttnn/cpp/ttnn/tensor/types.cpp",
                    "normalized_index >= 0 and normalized_index < rank",
                    "Index is out of bounds for the rank",
                ],
            ),
            MatchingExceptionRule(
                "TTNN tensor types",
                [
                    "RuntimeError",
                    "tt-metal/ttnn/cpp/ttnn/tensor/types.cpp",
                    "shape[cur_idx] == 1",
                    "Can't convert shape rank",
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
            MatchingExceptionRule(
                "ttnn core",
                [
                    "RuntimeError",
                    "tt-metal/ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp",
                    "logical_shape.rank() >= 2 && logical_shape.rank() <= 4",
                    "Only 2D, 3D, and 4D tensors are supported",
                ],
            ),
        ],
    ),
]


class UniqueOpTestInfo:
    """
    Represents information about a unique operation test, that includes op name, operands
    arguments, and the status of various compiler components.

    Attributes:
        Op (str): The name of the operation.
        Operands (List[str]): List of operands associated with the operation.
        Args (List[str]): List of Operation Arguments if any
        components (dict): A dictionary indicating the support status for each compiler component.
        failure_reason (str): The reason for failure, if any, during testing.
    """

    def __init__(
        self,
        Op: str,
        Operands: List[str],
        Args: List[str],
    ):
        self.Op = str(Op)
        self.Operands = Operands
        self.Args = Args
        self.components = {}
        for compiler_component in CompilerComponent:
            self.components[str(compiler_component.name)] = False
        self.failure_reason = ""

    @classmethod
    def create(cls, op_name, operand_names, operand_types, operand_shapes, operand_dtypes, args):

        operands = UniqueOpTestInfo.create_operands(operand_names, operand_types, operand_shapes, operand_dtypes)

        args = UniqueOpTestInfo.create_args(args)

        return cls(Op=op_name, Operands=operands, Args=args)

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
        return f"UniqueOpTestInfo(op={self.Op}, Operands={self.Operands}, Args={self.Args}, components={self.components}, self.failure_reason={self.failure_reason})"


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


class HtmlSymbol(Enum):
    PASS = "&#x2705;"  # Checkmark
    FAIL = "&#x274C;"  # Crossmark
    UNKNOWN = "&#xFFFD;"  # Question mark


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

    @classmethod
    def get_component_names_for_header(cls, compiler_component: CompilerComponent):
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


def check_path(directory_or_file_path: str):

    # Check if a file or directory exists, return True otherwise return False
    if os.path.exists(directory_or_file_path):
        logger.info(f"{directory_or_file_path} exists!")
        return True

    logger.info(f"{directory_or_file_path} does not exist.")
    return False


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
    Collect all the tests marked with the `model_analysis` marker in a specified directory or file.
    """

    # Ensure the directory or file path exists
    assert check_path(
        directory_or_file_path
    ), f"The directory path for collecting test {directory_or_file_path} doesn't exists"

    logger.info(f"Collecting all the test that has model_analysis marker in {directory_or_file_path}")

    collected_test_outputs = ""
    try:
        # Run pytest to collect tests with the `model_analysis` marker
        result = subprocess.run(
            ["pytest", directory_or_file_path, "-m", "model_analysis", "--collect-only"],
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


def generate_and_export_unique_ops_tests(test_directory_or_file_path: str, unique_ops_output_directory_path: str):
    """
    Collect the test with model_analysis marker in the test_directory_or_file_path specified by the user
    and then generate unique op test for all the collected test and return the list of directory path
    containing exported models unique op configuration as xlsx file
    """

    # Collect all the pytest inside the test_directory_or_file_path specified by the user with model_analysis marker
    test_list = collect_all_model_analysis_test(test_directory_or_file_path, unique_ops_output_directory_path)

    assert (
        test_list != []
    ), f"No tests found in the {test_directory_or_file_path} path with model_analysis pytest marker"

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
                        FORGE_DISABLE_REPORTIFY_DUMP="1",
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


def extract_unique_op_tests_from_models(model_output_dir_paths: List[str], unique_ops_output_directory_path: str):
    """
    Extract unique op configuration across all the models which will avoid running the redudant
    op configuration again by using the exported unique op configuration test details and models metadata
    """

    # Dictionary to store all the operations found in model variants
    models_operations = {}
    unique_op_count = 0

    # Dictionary to store constants (name and tensor) used in the model variants
    models_contants = {}

    # Iterate through all provided model directories
    for model_output_dir_path in model_output_dir_paths:

        # Extract the model name from the directory path
        model_name = model_output_dir_path.split("/")[-1]

        # List all model variants in the directory
        model_variants = os.listdir(model_output_dir_path)

        # Process each model variant
        for model_variant in model_variants:

            model_variant_dir_path = os.path.join(model_output_dir_path, model_variant)

            # Look for `.xlsx` and `.json` file containing unique operation details and metadata
            model_variant_tvm_generated_unique_op_xslx_file_path = None
            model_variant_tvm_generated_unique_op_metadata_file_path = None
            for f in os.listdir(model_variant_dir_path):
                if f.endswith(".xlsx"):
                    model_variant_tvm_generated_unique_op_xslx_file_path = os.path.join(model_variant_dir_path, f)
                elif f.endswith(".json"):
                    model_variant_tvm_generated_unique_op_metadata_file_path = os.path.join(model_variant_dir_path, f)

            # Skip if either `.xlsx` or `.json` file is missing
            if (
                model_variant_tvm_generated_unique_op_xslx_file_path is None
                or model_variant_tvm_generated_unique_op_metadata_file_path is None
            ):
                continue

            # Read the `.xlsx` file contains model variant unique op configuration details
            model_variant_df = pd.read_excel(
                model_variant_tvm_generated_unique_op_xslx_file_path,
                header=0,
                usecols=[
                    "Op",
                    "Operand_Names",
                    "Operand_Shapes",
                    "Operand_Types",
                    "Operand_Dtypes",
                    "Args",
                    "Testfile",
                ],
            )

            # Read the `.json` file contains model variant metadata information
            with open(model_variant_tvm_generated_unique_op_metadata_file_path, "r") as json_file:
                model_variant_metadata = json.load(json_file)

            # Load model variants parameters and buffers as tensors from specified files
            named_parameters = torch.load(model_variant_metadata["named_params_file_name"])
            if model_variant_metadata["param_file_name"] is not None:
                serialized_params = torch.load(model_variant_metadata["param_file_name"])
                named_parameters.update(serialized_params)
            named_buffers = torch.load(model_variant_metadata["named_buffers_file_name"])
            named_parameters.update(named_buffers)

            # Process each row in the `.xlsx` file to extract operation configurations
            for index, row in model_variant_df.iterrows():
                row = row.to_dict()
                unique_op_count += 1

                operand_names = ast.literal_eval(row["Operand_Names"])
                operand_types = ast.literal_eval(row["Operand_Types"])
                operand_types = [NodeType.from_json(operand_type) for operand_type in operand_types]

                # Prepare metadata associated with the operation
                metadata = {}
                metadata["model_variant_info"] = {}
                metadata["model_variant_info"]["model_name"] = model_name
                metadata["model_variant_info"]["variant_name"] = model_variant_metadata["module_name"]
                metadata["model_variant_info"]["framework"] = model_variant_metadata["framework"]
                metadata["model_variant_info"]["Testfile"] = row["Testfile"]

                # Create an Operation object with op name, shape, nodetype, dtype, arguments and operation metadata
                models_operations[unique_op_count] = Operation(
                    function_name=row["Op"],
                    input_names=operand_names,
                    args=ast.literal_eval(row["Args"]),
                    input_shapes=ast.literal_eval(row["Operand_Shapes"]),
                    input_dtypes=ast.literal_eval(row["Operand_Dtypes"]),
                    input_node_types=operand_types,
                    metadata=metadata,
                )

                # Store tensor which has constant nodetype as operands
                for operand_type, operand_name in zip(operand_types, operand_names):
                    if operand_type == NodeType.Constant:
                        models_contants[operand_name] = named_parameters[operand_name]

    # Extract unique operation configuration configuration across all the model variants
    unique_operations = UniqueOperations.create_unique_operations(models_operations, models_contants)

    # Dump the extracted unique operation configurations across all the model variants to a log file
    models_unique_op_config_file_path = os.path.join(
        unique_ops_output_directory_path, "extracted_unique_configuration_across_models.log"
    )
    dump_logs(models_unique_op_config_file_path, str(unique_operations))

    return unique_operations


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

    # Iterate over model variants to generate sub markdown files and populate root markdown rows
    for model_variant_info in models_details.values():

        # Prepare the path for the sub markdown file to store test results for this model variant
        sub_markdown_file_name = model_variant_info.variant_name
        sub_markdown_directory_path = os.path.join(markdown_directory_path, "Models", model_variant_info.model_name)

        # List to store table rows for the sub markdown file
        sub_markdown_table_rows = []

        # Iterate over the unique operation test information to populate table rows for sub markdown
        for unique_op_test_info in model_variant_info.unique_ops:

            table_data = [unique_op_test_info.Op]
            table_data.append("<br><div align='center'>X</div>".join(unique_op_test_info.Operands))
            table_data.append("<br>".join(unique_op_test_info.Args))

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
        default=os.path.join(os.getcwd(), "forge/test"),
        help="Specify the directory or file path containing models test with model_analysis pytest marker",
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
        test_directory_or_file_path=args.test_directory_or_file_path,
        unique_ops_output_directory_path=args.unique_ops_output_directory_path,
    )

    unique_operations = extract_unique_op_tests_from_models(
        model_output_dir_paths=model_output_dir_paths,
        unique_ops_output_directory_path=args.unique_ops_output_directory_path,
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
