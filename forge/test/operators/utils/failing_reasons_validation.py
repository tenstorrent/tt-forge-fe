# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Validations of failing reasons


from loguru import logger
from typing import Optional

from forge.forge_property_utils import record_sweeps_detected_failing_reason

from .failing_reasons import ExceptionData
from .failing_reasons import FailingReasons
from .failing_reasons_finder import FailingReasonsFinder
from .pytest import PyTestUtils


class FailingReasonsValidation:
    @classmethod
    def build_ex_data(cls, exception_value: Exception, exception_traceback: str) -> ExceptionData:
        """Convert exception to ExceptionData object

        Args:
            exception_value (Exception): Raised exception
            exception_traceback (str): Exception traceback

        Returns:
            ExceptionData: Exception data object
        """
        ex_class_name = f"{type(exception_value).__module__}.{type(exception_value).__name__}"
        ex_class_name = ex_class_name.replace("builtins.", "")
        ex_message = f"{exception_value}"
        exception_traceback = PyTestUtils.remove_colors(exception_traceback)
        ex_data = ExceptionData(
            class_name=ex_class_name,
            message=ex_message,
            error_log=exception_traceback,
        )
        return ex_data

    @classmethod
    def find_reason_by_ex_data(cls, ex_data: ExceptionData) -> Optional[FailingReasons]:
        xfail_reason = FailingReasonsFinder.find_reason_by_ex_data(ex_data)
        xfail_reason = xfail_reason.name if xfail_reason else None
        if xfail_reason is None:
            xfail_reason = "UNCLASSIFIED"
        xfail_reason = FailingReasons[xfail_reason]
        return xfail_reason

    @classmethod
    def record_detected_failing_reason(cls, ex_data: ExceptionData):
        detected_xfail_reason = cls.find_reason_by_ex_data(ex_data)
        detected_xfail_reason_desc = detected_xfail_reason.value.description if detected_xfail_reason else None
        detected_xfail_reason = detected_xfail_reason.name if detected_xfail_reason else None
        record_sweeps_detected_failing_reason(
            detected_failing_reason=detected_xfail_reason,
            detected_failing_reason_desc=detected_xfail_reason_desc,
        )

    @classmethod
    def validate_exception(cls, exception_value: Exception, exception_traceback: str, xfail_reason: str):
        """Validate exception based on xfail reason

        Args:
            exception_value (Exception): Raised exception to validate
            xfail_reason (str): Xfail reason

        Returns:
            bool: True if exception message and type match the expected values, False otherwise, None if no check is defined
        """
        failing_reason = FailingReasons.find_by_description(xfail_reason)
        if failing_reason is None:
            logger.error(
                f"Test is marked with xfail reason: '{xfail_reason}' but no failing reason found for exception: {type(exception_value)} '{exception_value}'"
            )
            return False

        if len(failing_reason.value.checks) > 0:
            ex_data = cls.build_ex_data(exception_value, exception_traceback)

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
            # TODO calculate detected failing reason and log it
            # if ex_data.message is not None:
            #     FailingReasonsValidation.record_detected_failing_reason(ex_data)

            return False
        else:
            logger.warning(
                f"Test is marked with xfail reason: '{xfail_reason}' but no check performed for exception: {type(exception_value)} '{exception_value}'"
            )
            return None

    @classmethod
    def get_xfail_reason(cls, error_message_full, error_log) -> Optional[str]:
        if error_message_full is None:
            return None
        xfail_reason = FailingReasonsFinder.find_reason_by_ex_data(error_message_full, error_log)
        if xfail_reason is None:
            return "UNCLASSIFIED"
        return xfail_reason
