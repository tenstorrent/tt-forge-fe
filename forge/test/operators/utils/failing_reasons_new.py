# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Failing reasons definition


from enum import Enum
from loguru import logger
from typing import Type, Optional
from dataclasses import dataclass, field
from typing import Union, List, Generator


@dataclass
class ExceptionData:
    # operator: str
    class_name: str
    message: str


@dataclass
class FailingReasonDef:
    description: str
    checks: List["ExceptionCheck"] = field(default_factory=list)


@dataclass
class ExceptionCheck:
    # operators: List[str]
    class_name: str
    messages: List[str]

    # def __contains__(self, message: Union[str, List[str]]) -> bool:
    #     if isinstance(message, str):
    #         return message in self.message
    #     if isinstance(message, list):
    #         return all(item in self.message for item in message)


class FailingReasonsEnum(Enum):
    NOT_IMPLEMENTED = "Not implemented operator"

    BUGGY_SHAPE = "Buggy shape"

    MICROBATCHING_UNSUPPORTED = "Higher microbatch size is not supported"

    UNSUPPORTED_DATA_FORMAT = "Data format is not supported"

    UNSUPPORTED_DIMENSION = "Unsupported dimension"

    UNSUPORTED_AXIS = "Unsupported axis parameter"

    UNSUPPORTED_PARAMETER_VALUE = "Unsupported parameter value"

    UNSUPPORTED_SPECIAL_CASE = "Unsupported special case"

    # Error message: E           RuntimeError: TT_ASSERT @ pybuda/csrc/passes/lowering_context.cpp:28: old_node->node_type() != graphlib::NodeType::kPyOp
    # Error for input shape (1, 1, 10000, 1). Error message: RuntimeError: TT_ASSERT @ pybuda/csrc/placer/lower_to_placer.cpp:245:
    COMPILATION_FAILED = "Model compilation failed"

    # Error message: E           AssertionError: Error during inference
    INFERENCE_FAILED = "Inference failed"

    # "Error message: E          AssertionError: Data mismatch detected"
    # Validation error caused by pcc threshold
    DATA_MISMATCH = "Verification failed due to data mismatch"

    # ValueError: Dtype mismatch: framework_model.dtype=torch.int8, compiled_model.dtype=torch.uint8
    DTYPE_MISMATCH = "Dtype mismatch"

    UNSUPPORTED_TYPE_FOR_VALIDATION = "Verification failed due to unsupported type in verify_module"

    # "Fatal python error - xfail does not work; UserWarning: resource_tracker: There appear to be 26 leaked semaphore objects to clean up at shutdown"
    # "Fatal python error - xfail does not work. Error message: Fatal Python error: Segmentation fault; UserWarning: resource_tracker: There appear to be 26 leaked semaphore objects to clean up at shutdown"
    SEMAPHORE_LEAK = "Semaphore leak"

    # RuntimeError: Fatal Python error: Segmentation fault
    SEG_FAULT = "Inference failed due to seg fault"

    # RuntimeError: Fatal Python error: Aborted
    FATAL_ERROR = "Fatal error occured"

    UNSUPPORTED_INPUT_SOURCE = "Unsupported input source"

    ATTRIBUTE_ERROR = "Attribute error"

    # INFO     | forge.compiled_graph_state:__call__:247  Running model forward on device...
    # Always | FATAL    | Out of Memory: Not enough space to allocate 896204800 B DRAM buffer across 12 banks, where each bank needs to store 74686464 B
    ALLOCATION_FAILED = "Out of Memory"

    INFERENCE_FROZE = "Inference froze without error message"

    ALLOCATION_CIRCULAR_BUFFER = "Allocation of circular buffer"


class FailingReasonsChecks(Enum):
    UNSUPPORTED_DATA_FORMAT = [
        # lambda ex: FailingReasonsValidation.validate_exception_message(ex, RuntimeError, "Unsupported data type"),
        lambda ex: ex.class_name == "RuntimeError" and "Unsupported data type" in ex.message,  # TODO: Check if this change is correct
        # lambda ex: ex.class_name == "RuntimeError" and "/forge/csrc/passes/lower_to_mlir.cpp:466: false" in ex.message,
        lambda ex: ex.class_name == "RuntimeError" and "/forge/csrc/passes/lower_to_mlir.cpp:473: false" in ex.message,
        lambda ex: ex.class_name == "RuntimeError"
        and ex.message == "Tensor 2 - data type mismatch: expected UInt32, got Float32",
        lambda ex: ex.class_name == "RuntimeError" and '"softmax_lastdim_kernel_impl" not implemented' in ex.message,
        lambda ex: ex.class_name == "RuntimeError" and "Unsupported data format" in ex.message,
        lambda ex: ex.class_name == "RuntimeError" and "\"bmm\" not implemented for 'Half'" in ex.message,
        lambda ex: ex.class_name == "RuntimeError" and "Input tensors must have the same data type, but got {} and {}" in ex.message,
    ]
    DATA_MISMATCH = [
        lambda ex: ex.class_name == "AssertionError" and ex.message == "PCC check failed",
        lambda ex: ex.class_name == "AssertionError" and ex.message.startswith("Data mismatch"),
        lambda ex: ex.class_name == "ValueError" and ex.message.startswith("Data mismatch"),
        lambda ex: ex.class_name == "RuntimeError" and "data type mismatch" in ex.message,
    ]
    DTYPE_MISMATCH = [
        lambda ex: ex.class_name == "ValueError" and ex.message.startswith("Dtype mismatch"),
    ]
    UNSUPPORTED_SPECIAL_CASE = [
        # lambda ex: ex.class_name == "AssertionError" and ex.message == "PCC check failed",
        lambda ex: ex.class_name == "AssertionError" and ex.message.startswith("Exponent value"),
        lambda ex: ex.class_name == "RuntimeError"
        and "normalized_index >= 0 and normalized_index < rank" in ex.message,
        # lambda ex: ex.class_name == "RuntimeError"
        # and "Statically allocated circular buffers on core range [(x=0,y=0) - (x=7,y=7)]" in ex.message,
        # Given weight of size [10, 2, 1, 1], expected bias to be 1-dimensional with 10 elements, but got bias of size [1, 1, 1, 10] instead
        lambda ex: ex.class_name == "RuntimeError"
        and "Given weight of size" in ex.message
        and "expected bias to be 1-dimensional with" in ex.message
        and "but got bias of size" in ex.message,
    ]
    NOT_IMPLEMENTED = [
        lambda ex: ex.class_name == "NotImplementedError"
        and ex.message.startswith("The following operators are not implemented:"),
        lambda ex: ex.class_name == "RuntimeError"
        and ex.message.startswith("Found Unsupported operations while lowering from TTForge to TTIR in forward graph"),
        lambda ex: ex.class_name == "RuntimeError"
        and ex.message.startswith("Unsupported operation for lowering from TTForge to TTIR:"),
        lambda ex: ex.class_name == "RuntimeError" and " not implemented for " in ex.message,
        lambda ex: ex.class_name == "AssertionError"
        and ex.message == "Encountered unsupported op types. Check error logs for more details",
        lambda ex: ex.class_name == "RuntimeError"
        and "!in_ref.get_shape().has_tile_padding(this->dim)"  # tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/data_movement/concat/device/concat_device_operation.cpp:47: !in_ref.get_shape().has_tile_padding(this->dim)
        in ex.message,
        lambda ex: ex.class_name == "RuntimeError"
        and "info:\nBinaryOpType cannot be mapped to BcastOpMath" in ex.message,
    ]
    ALLOCATION_FAILED = [
        lambda ex: ex.class_name == "RuntimeError" and "Out of Memory: Not enough space to allocate" in ex.message,
        # lambda ex: ex.class_name == "RuntimeError"
        # and "tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/impl/allocator/allocator.cpp:143"
        # in ex.message,
        # lambda ex: ex.class_name == "RuntimeError"
        # and "tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/impl/allocator/allocator.cpp:145"
        # in ex.message,
    ]
    ALLOCATION_CIRCULAR_BUFFER = [
        lambda ex: ex.class_name == "RuntimeError"
        and "Statically allocated circular buffers on core range" in ex.message,
    ]
    ATTRIBUTE_ERROR = [
        lambda ex: ex.class_name == "AttributeError",
    ]
    COMPILATION_FAILED = [
        lambda ex: ex.class_name == "RuntimeError"
        and "tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/core/core.cpp:49: tt::exception"
        in ex.message,
        lambda ex: ex.class_name == "RuntimeError"
        and "tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/core/core.cpp:60: tt::exception"
        in ex.message,
        lambda ex: ex.class_name == "RuntimeError" and "Generated MLIR module failed verification" in ex.message,
        lambda ex: f"InternalError: Check failed: *axis_ptr == X (Y vs. X) : cannot squeeze axis with dimension not equal to X"
        in ex.message,
    ]
    INFERENCE_FAILED = [
        lambda ex: ex.class_name == "AttributeError"
        and ex.message == "'TransposeTM' object has no attribute 'z_dim_slice' (via OpType cpp underlying class)",
        lambda ex: ex.class_name == "RuntimeError"
        and "tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_op.cpp"
        in ex.message,
        lambda ex: ex.class_name == "RuntimeError"
        and "tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp"
        in ex.message,
        lambda ex: ex.class_name == "RuntimeError"
        and "tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/reduction/generic/generic_reductions.cpp"
        in ex.message,
        lambda ex: ex.class_name == "RuntimeError"
        and "tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/data_movement/pad/device/pad_op.cpp"
        in ex.message,
        lambda ex: ex.class_name == "RuntimeError"
        and "tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/data_movement/transpose/device/transpose_op.cpp:106: input_tensor.get_dtype() == DataType::BFLOAT16 || input_tensor.get_dtype() == DataType::FLOAT32"
        in ex.message,
        lambda ex: ex.class_name == "RuntimeError"
        and "Statically allocated circular buffers on core range [(x=0,y=0) - (x=0,y=0)] grow to 2663200 B which is beyond max L1 size of 1499136 B"
        in ex.message,
        lambda ex: ex.class_name == "RuntimeError"
        and "Statically allocated circular buffers on core range [(x=0,y=0) - (x=7,y=7)] grow to 28100144 B which is beyond max L1 size of 1499136 B"
        in ex.message,
        lambda ex: ex.class_name == "RuntimeError"
        and "Index is out of bounds for the rank, should be between 0 and 0 however is 1" in ex.message,
        lambda ex: ex.class_name == "RuntimeError" and "Generated MLIR module failed verification." in ex.message,
        lambda ex: ex.class_name == "RuntimeError"
        and "Please look up dimensions by name, got: name = None" in ex.message,
        lambda ex: ex.class_name == "RuntimeError"
        and "293 unique+common runtime args targeting kernel reader_concat_stick_layout_interleaved_start_id on (x=0,y=0) are too large. Max allowable is 256"
        in ex.message,
        # lambda ex: ex.class_name == "RuntimeError"
        # and "input_tensor_arg.get_layout() == ttnn::ROW_MAJOR_LAYOUT" in ex.message,
        # lambda ex: ex.class_name == "RuntimeError" and "weights.get_dtype() == DataType::BFLOAT16" in ex.message,
        lambda ex: ex.class_name == "RuntimeError"
        and "Tensor 1 - data type mismatch: expected BFloat16, got Float32" in ex.message,
        lambda ex: ex.class_name == "RuntimeError"
        and "tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/data_movement/repeat/repeat.cpp:41: repeat_dims.rank() == input_rank"
        in ex.message,
        lambda ex: ex.class_name == "RuntimeError"
        and "tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/eltwise/binary/binary.cpp:124: second_shape[-3] == 1"
        in ex.message,
        lambda ex: ex.class_name == "RuntimeError"
        and "tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/tensor/types.cpp:192: normalized_index >= 0 and normalized_index < rank"
        in ex.message,
        lambda ex: ex.class_name == "RuntimeError" and "Fatal error" in ex.message,
        lambda ex: ex.class_name == "RuntimeError"
        and "mat1 and mat2 must have the same dtype, but got Int and Float" in ex.message,
    ]
    MICROBATCHING_UNSUPPORTED = [
        lambda ex: ex.class_name == "RuntimeError" and "The expanded size of the tensor" in ex.message,
    ]
    UNSUPORTED_AXIS = [
        lambda ex: ex.class_name == "RuntimeError" and "Inputs must be of bfloat16 or bfloat8_b type" in ex.message,
    ]


class FailingReasonsFinder:
    unique_messages = set()

    @classmethod
    def get_exception_data(cls, error_message: str) -> ExceptionData:
        error_message_short = error_message.split("\n")[0]
        class_name = error_message_short[:50].split(":")[0]
        # error_message = " ".join(error_message.split("\n")[:1])
        error_message = error_message[len(class_name) + 2 :]
        # logger.info(class_name)
        # logger.info(f"Line: {class_name} | {error_message}")
        ex = ExceptionData(class_name, error_message)
        return ex

    @classmethod
    def find_reason(cls, error_message: str):
        ex = cls.get_exception_data(error_message)
        reasons = list(cls.find_reasons(ex))
        if not reasons:
            return None
        if len(reasons) > 1:
            message = f"Multiple reasons found: {reasons} for ex: {ex}"
            if message not in cls.unique_messages:
                cls.unique_messages.add(message)
                logger.warning(message)
        return reasons[0]

    @classmethod
    def find_reasons(cls, ex: ExceptionData) -> Generator[str, None, None]:
        for xfail_reason in FailingReasonsChecks:
            xfail_reason_checks = xfail_reason.value
            for xfail_reason_check in xfail_reason_checks:
                if xfail_reason_check(ex):
                    yield xfail_reason.name
