# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Validations of failing reasons


from loguru import logger

from .failing_reasons_new import ExceptionData
from .failing_reasons_new import FailingReasonsDefs
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
        xfail_reason_str = xfail_reason
        failing_reasons = [
            xfail_reason for xfail_reason in FailingReasonsDefs if xfail_reason.value.description == xfail_reason_str
        ]
        if len(failing_reasons) == 0:
            logger.error(
                f"Test is marked with xfail reason: '{xfail_reason}' but no failing reason found for exception: {type(exception_value)} '{exception_value}'"
            )
            return False
        elif len(failing_reasons) > 1:
            logger.error(
                f"Test is marked with xfail reason: '{xfail_reason}' but multiple failing reasons found for exception: {type(exception_value)} '{exception_value}'"
            )
            return False

        failing_reason = failing_reasons[0]

        if len(failing_reason.value.checks) > 0:
            ex_class_name = type(exception_value).__name__
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
