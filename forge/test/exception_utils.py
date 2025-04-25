# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import List


class MatchingExceptionRule:
    """
    Represents a rule for matching exception messages based on specific tokens.

    Attributes:
        rule_name (str): The descriptive name of the rule.
        rule_tokens (List[str]): A list of strings that must all appear in an exception message to consider it a match.
    """

    def __init__(
        self,
        rule_name: str,
        rule_tokens: List[str],
    ):
        self.rule_name = rule_name
        self.rule_tokens = rule_tokens

    def match_rule(self, exception: str):
        """
        Checks if all tokens in self.rule_tokens are present in the given exception message.

        Args:
            exception (str): The exception message to be evaluated.

        Returns:
            str: A formatted string including the rule name and tokens if all tokens match, otherwise None.
        """
        # Evaluate whether every token from rule_tokens is found in the exception message.
        # Using 'all' with a generator expression streamlines the process.
        matched_tokens = all(token in exception for token in self.rule_tokens)
        if matched_tokens:
            # If all tokens match, format and return a string that identifies the rule.
            rule_token_str = " ".join(self.rule_tokens)
            return f"[{self.rule_name}] {rule_token_str}"
        else:
            # If one or more tokens do not match, return None to indicate no match.
            return None


compiler_exception_rules = [
    MatchingExceptionRule(
        "Framework vs Compiled Model Output Data mismatch",
        [
            "ValueError",
            "Data mismatch -> AutomaticValueChecker (compare_with_golden)",
        ],
    ),
    MatchingExceptionRule("Forge Module Evaluation", ["AssertionError", "Setting a tensor value of incorrect shape"]),
    MatchingExceptionRule(
        "Forge Compilation",
        ["RuntimeError", "Node not found"],
    ),
    MatchingExceptionRule(
        "Upsample Op Validation",
        ["AssertionError", "Only support upsample with integer scale factor"],
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
        "Convert tt-forge attribute to an MLIR attribute", ["RuntimeError", "Unhandled attribute type"]
    ),
    MatchingExceptionRule(
        "lower_to_mlir",
        ["RuntimeError", "Found Unsupported operations while lowering from TTForge to TTIR"],
    ),
    MatchingExceptionRule(
        "lower_to_mlir",
        ["RuntimeError", "Unsupported data format during lowering from TTForge to TTIR"],
    ),
    MatchingExceptionRule("lower_to_mlir", ["RuntimeError", "Generated MLIR module failed verification"]),
    MatchingExceptionRule(
        "run_mlir_passes",
        ["RuntimeError", "Failed to run MLIR compiler pass pipeline"],
    ),
    MatchingExceptionRule("Runtime Datatype Unsupported", ["RuntimeError", "Unhandled dtype"]),
    MatchingExceptionRule(
        "Runtime Datatype mismatch",
        ["RuntimeError", "Tensor", "data type mismatch: expected", "got"],
    ),
    MatchingExceptionRule("Runtime Shape mismatch", ["RuntimeError", "Tensor", "shape mismatch: expected", "got"]),
    MatchingExceptionRule(
        "Runtime stride mismatch",
        ["RuntimeError", "Tensor", "stride mismatch: expected", "got"],
    ),
    MatchingExceptionRule("Runtime Input count mismatch", ["RuntimeError", "Input count mismatch: expected", "got"]),
    MatchingExceptionRule(
        "Runtime Program Execution",
        ["RuntimeError", "Cannot access data pointer of Tensor that doesn't have storage"],
    ),
    MatchingExceptionRule(
        "tt-metal buffer allocation",
        [
            "RuntimeError",
            "tt-metal/tt_metal/impl/allocator/bank_manager.cpp",
            "Out of Memory: Not enough space to allocate",
        ],
    ),
    MatchingExceptionRule(
        "tt-metal kernel",
        [
            "RuntimeError",
            "tt-metal/tt_metal/impl/kernels/kernel.cpp",
            "unique+common runtime args targeting kernel",
            "are too large",
        ],
    ),
    MatchingExceptionRule(
        "ttnn.tilize validation",
        [
            "RuntimeError",
            "tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_op.cpp",
            "input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::FLOAT32",
            "data type must be bfloat16 or float32",
        ],
    ),
    MatchingExceptionRule(
        "ttnn.tilize_with_val_padding validation",
        [
            "RuntimeError",
            "tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp",
            "input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32",
            "or",
            "input_tensor_a.get_dtype() == DataType::FLOAT32",
            "Can only tilize bfloat16/float32 or uint32 tensors",
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
        "ttnn elementwise binary",
        [
            "RuntimeError",
            "tt-metal/ttnn/cpp/ttnn/operations/eltwise/binary/device/broadcast_height_and_width_multi_core_program_factory.cpp",
            "BinaryOpType cannot be mapped to BcastOpMath",
        ],
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
        "ttnn.reshape validation",
        [
            "RuntimeError",
            "tt-metal/ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/device/reshape_op.cpp",
            "input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::FLOAT32",
        ],
    ),
    MatchingExceptionRule(
        "ttnn.reshape validation",
        [
            "RuntimeError",
            "tt-metal/ttnn/cpp/ttnn/operations/data_movement/reshape_view/device/reshape_rm_op.cpp",
            "input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 or input_tensor_a.get_dtype() == DataType::FLOAT32",
            "Can only work with bfloat16/float32 or uint32 tensors",
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
        "ttmetal circular buffer validation",
        [
            "RuntimeError",
            "tt-metal/tt_metal/impl/program/program.cpp",
            "Statically allocated circular buffers in program",
        ],
    ),
    MatchingExceptionRule(
        "ttmetal circular buffer validation",
        ["RuntimeError", "tt-metal/tt_metal/impl/program/program.cpp", "Statically allocated circular buffers on core"],
    ),
    MatchingExceptionRule(
        "ttnn softmax",
        [
            "RuntimeError",
            "tt-metal/ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/moreh_softmax_device_operation.cpp",
            "input.get_dtype() == DataType::BFLOAT16 || input.get_dtype() == DataType::BFLOAT8_B",
            "Inputs must be of bfloat16 or bfloat8_b type",
        ],
    ),
    MatchingExceptionRule(
        "ttnn unsqueeze_to_4D",
        [
            "RuntimeError",
            "tt-metal/ttnn/cpp/ttnn/operations/core/core.cpp",
            "Tensor rank is greater than 4",
        ],
    ),
    MatchingExceptionRule(
        "ttnn shared operation",
        [
            "RuntimeError",
            "tt-metal/ttnn/cpp/ttnn/operations/data_movement/sharded/interleaved_to_sharded/device/interleaved_to_sharded_op.cpp",
            "(*this->output_mem_config.shard_spec).shape[1] * input_tensor.element_size() % hal::get_l1_alignment() == 0",
            "Shard page size must currently have L1 aligned page size",
        ],
    ),
    MatchingExceptionRule(
        "ttnn pool2d validation",
        [
            "RuntimeError",
            "tt-metal/ttnn/cpp/ttnn/operations/pool/generic/device/pool_op.cpp",
            "(input_shape[3] % tt::constants::TILE_WIDTH == 0) || (input_shape[3] == 16)",
            "Input channels",
            "should be padded to nearest TILE_WIDTH",
            "or should be 16",
        ],
    ),
    MatchingExceptionRule(
        "ttnn pool",
        [
            "RuntimeError",
            "tt-metal/ttnn/cpp/ttnn/operations/pool/generic/device/pool_multi_core_program_factory.cpp",
            "input_shape[3] == 16",
        ],
    ),
    MatchingExceptionRule(
        "ttnn conv2d",
        [
            "RuntimeError",
            "tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/device/conv2d_op_sharded_program_factory.cpp",
            "act_block_w_datums == round_up(conv_act_size_c * filter_w, TILE_WIDTH)",
        ],
    ),
    MatchingExceptionRule(
        "ttnn unsample",
        [
            "RuntimeError",
            "tt-metal/ttnn/cpp/ttnn/operations/pool/upsample/device/upsample_op.cpp",
            "Unsupported mode",
        ],
    ),
]


def extract_refined_error_message(error_message: str):
    """
    Extract and refine an error message from a multiline string.

    This function searches through the given error message for lines that start with the prefix "E ".
    If a matching line is found, it attempts to extract up to a predefined number of consecutive error lines
    (defined by max_error_lines). If all of the extracted lines start with "E ", each line is cleaned by
    removing the prefix "E " and any extra whitespace, then these lines are concatenated into a single string.
    If only one error line is found, it will return that single cleaned line.

    Args:
        error_message (str): A multiline error message string typically containing multiple lines.

    Returns:
        The refined error message if an error line is found, otherwise None.
    """
    # Define the maximum number of consecutive error lines to extract.
    max_error_lines = 3

    # Split the error message into individual lines, preserving newline characters.
    lines = error_message.splitlines(True)

    # Iterate over each line and its index.
    for idx, line in enumerate(lines):

        # Check if the current line is marked as an error line with the prefix "E ".
        if line.startswith("E "):

            # Retrieve a block of lines starting from the current error line up to the max defined.
            next_lines = lines[idx : idx + max_error_lines]

            # Verify if all lines in the extracted block start with "E "
            if all(l.startswith("E ") for l in next_lines):
                # Verify if all lines in the extracted block start with "E ".
                refined_error_message = [l.replace("E ", "").strip("\n").strip() for l in next_lines]
                return " ".join(refined_error_message)

            # If only a single error line is found, clean and return that line.
            return line.replace("E ", "").strip("\n").strip()

    # If no error lines with the expected prefix are found, return None.
    return None


def extract_failure_category(refined_error_message: str):
    """
    Iterates over defined compiler exception rules to find a rule that matches the refined error message.

    Args:
        refined_error_message (str): The error message after refinement for clarity.

    Returns:
        str or None: A formatted string representing the failure category if a match is found; otherwise, None.
    """
    # Loop through each exception rule and attempt to match the refined error message.
    for exception_rule in compiler_exception_rules:
        matched_exception = exception_rule.match_rule(refined_error_message)
        if matched_exception is not None:
            # Return the formatted rule string if a match is found.
            return matched_exception
    # If no rule matches, return None.
    return None
