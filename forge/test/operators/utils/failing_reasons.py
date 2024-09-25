# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Failing reasons for pytests marks


# Compilation failed
# Inference failed
# Validation failed
# Special case not supported
# ...


from loguru import logger
from typing import Type, Optional


class FailingReasons:
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

    UNSUPPORTED_TYPE_FOR_VALIDATION  = "Verification failed due to unsupported type in verify_module"

    # "Fatal python error - xfail does not work; UserWarning: resource_tracker: There appear to be 26 leaked semaphore objects to clean up at shutdown"
    # "Fatal python error - xfail does not work. Error message: Fatal Python error: Segmentation fault; UserWarning: resource_tracker: There appear to be 26 leaked semaphore objects to clean up at shutdown"
    SEMAPHORE_LEAK = "Semaphore leak"

    # RuntimeError: Fatal Python error: Segmentation fault
    SEG_FAULT = "Inference failed due to seg fault"

    UNSUPPORTED_INPUT_SOURCE = "Unsupported input source"


class FailingReasonsValidation:

    @classmethod
    def validate_exception_message(cls, exception_value: Exception, expected_message: str, exception_type: Optional[Type[Exception]]):
        ''' Validate exception message and type

        Args:
            exception_value (Exception): Raised exception to validate
            expected_message (str): Expected exception message
            exception_type (Optional[Type[Exception]]): Expected exception type

        Returns:
            bool: True if exception message and type match the expected values, False otherwise
        '''
        if exception_type is not None:
            return isinstance(exception_value, exception_type) and f"{exception_value}" == expected_message
        else:
            return f"{exception_value}" == expected_message

    # TODO: Add more checks
    # Unlisted reasons will not be checked
    XFAIL_REASON_CHECKS = {
        FailingReasons.UNSUPPORTED_DATA_FORMAT:
            [
                # lambda ex: FailingReasonsValidation.validate_exception_message(ex, RuntimeError, "Unsupported data type"),
                lambda ex: isinstance(ex, RuntimeError) and f"{ex}" == "Unsupported data type",
            ],
        FailingReasons.DATA_MISMATCH:
            [
                lambda ex: isinstance(ex, AssertionError) and f"{ex}" == "AssertionError: PCC check failed",
                lambda ex: isinstance(ex, AssertionError) and f"{ex}" == "PCC check failed",
            ],
        FailingReasons.UNSUPPORTED_SPECIAL_CASE:
            [
                lambda ex: isinstance(ex, AssertionError) and f"{ex}" == "PCC check failed",
            ],
        FailingReasons.NOT_IMPLEMENTED:
            [
                lambda ex: isinstance(ex, NotImplementedError) and f"{ex}".startswith("The following operators are not implemented:"),
                lambda ex: isinstance(ex, RuntimeError) and f"{ex}".startswith("Unsupported operation for lowering from TTForge to TTIR:"),
                lambda ex: isinstance(ex, AssertionError) and f"{ex}" == "Encountered unsupported op types. Check error logs for more details",
            ],
    }

    @classmethod
    def validate_exception(cls, exception_value: Exception, xfail_reason: str):
        ''' Validate exception based on xfail reason

        Args:
            exception_value (Exception): Raised exception to validate
            xfail_reason (str): Xfail reason

        Returns:
            bool: True if exception message and type match the expected values, False otherwise, None if no check is defined
        '''
        logger.debug(f"Validating xfail reason: '{xfail_reason}' for exception: {type(exception_value)} '{exception_value}'")

        if xfail_reason in cls.XFAIL_REASON_CHECKS:
            xfail_reason_checks = cls.XFAIL_REASON_CHECKS[xfail_reason]
            # Checking multiple conditions. If any of the conditions is met, return True
            for xfail_reason_check in xfail_reason_checks:
                if xfail_reason_check(exception_value):
                    logger.trace(f"Correct xfail reason: '{xfail_reason}' for exception: {type(exception_value)} '{exception_value}'")
                    return True
            logger.error(f"Wrong xfail reason: '{xfail_reason}' for exception: {type(exception_value)} '{exception_value}'")
            return False
        else:
            logger.warning(f"Test is marked with xfail reason: '{xfail_reason}' but no check performed for exception: {type(exception_value)} '{exception_value}'")
            return None
