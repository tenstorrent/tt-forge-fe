# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Validations of failing reasons


import traceback

from loguru import logger

from ..utils.features import TestSweepsFeatures
from ..utils.dual_output import global_string_buffer

from .failing_reasons import ExceptionData
from .failing_reasons import FailingReasons
from .pytest import PyTestUtils


class FailingReasonsValidation:
    @classmethod
    def validate_exception(cls, exception_value: Exception, exception_traceback: str, xfail_reason: str):
        """Validate exception based on xfail reason

        Args:
            exception_value (Exception): Raised exception to validate
            xfail_reason (str): Xfail reason

        Returns:
            bool: True if exception message and type match the expected values, False otherwise, None if no check is defined
        """
        with global_string_buffer.capture_output():
            return cls.validate_exception_body(exception_value, exception_traceback, xfail_reason)

    @classmethod
    def validate_exception_body(cls, exception_value: Exception, exception_traceback: str, xfail_reason: str):
        if TestSweepsFeatures.params.trace_xfail_validation:
            logger.debug(
                f"Validating xfail reason: '{xfail_reason}' for exception: {type(exception_value)} '{exception_value}'"
            )

            exception_stack = getattr(exception_value, "__traceback__", None)
            if exception_stack:
                formatted_stack = "".join(traceback.format_tb(exception_stack))
                logger.debug(f"Exception stack:\n{formatted_stack}")
            else:
                logger.debug("No traceback available for the exception.")

        failing_reason = FailingReasons.find_by_description(xfail_reason)
        if failing_reason is None:
            logger.error(
                f"Test is marked with xfail reason: '{xfail_reason}' but no failing reason found for exception: {type(exception_value)} '{exception_value}'"
            )
            return False

        if len(failing_reason.value.checks) > 0:
            ex_class_name = f"{type(exception_value).__module__}.{type(exception_value).__name__}"
            ex_class_name = ex_class_name.replace("builtins.", "")
            ex_message = f"{exception_value}"
            exception_traceback = PyTestUtils.remove_colors(exception_traceback)
            ex_data = ExceptionData(
                class_name=ex_class_name,
                message=ex_message,
                error_log=exception_traceback,
            )

            # Checking if exception data matches the failing reason
            if ex_data in failing_reason.value:
                logger.trace(
                    f"Correct xfail reason: '{xfail_reason}' for exception: {type(exception_value)} '{exception_value}'"
                )
                return True
            logger.error(
                f"Wrong xfail reason: '{xfail_reason}' for exception: {type(exception_value)} '{exception_value}'"
            )
            logger.debug(f"Exception data: {ex_data} failing reason: {failing_reason}")
            return False
        else:
            logger.warning(
                f"Test is marked with xfail reason: '{xfail_reason}' but no check performed for exception: {type(exception_value)} '{exception_value}'"
            )
            return None
