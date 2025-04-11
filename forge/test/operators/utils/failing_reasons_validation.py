# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Validations of failing reasons


import traceback

from loguru import logger
from typing import Generator

from ..utils.features import TestFeaturesConfiguration
from ..utils.dual_output import global_string_buffer

from .failing_reasons_new import ExceptionData
from .failing_reasons_new import FailingReasonsEnum
from .failing_reasons_new import FailingReasonsChecks
from .failing_reasons_new import FailingReasonsFinder


class FailingReasonsValidation:
    @classmethod
    def validate_exception(cls, exception_value: Exception, xfail_reason: str):
        """Validate exception based on xfail reason

        Args:
            exception_value (Exception): Raised exception to validate
            xfail_reason (str): Xfail reason

        Returns:
            bool: True if exception message and type match the expected values, False otherwise, None if no check is defined
        """
        with global_string_buffer.capture_output():
            return cls.validate_exception_body(exception_value, xfail_reason)

    @classmethod
    def validate_exception_body(cls, exception_value: Exception, xfail_reason: str):
        if TestFeaturesConfiguration.TRACE_XFAIL_VALIDATION:
            logger.debug(
                f"Validating xfail reason: '{xfail_reason}' for exception: {type(exception_value)} '{exception_value}'"
            )

            exception_stack = getattr(exception_value, "__traceback__", None)
            if exception_stack:
                formatted_stack = "".join(traceback.format_tb(exception_stack))
                logger.debug(f"Exception stack:\n{formatted_stack}")
            else:
                logger.debug("No traceback available for the exception.")

        ex_message = f"{exception_value}"

        xfail_reason_str = xfail_reason
        xfail_reasons = [xfail_reason for xfail_reason in FailingReasonsEnum if xfail_reason.value == xfail_reason_str]

        if len(xfail_reasons) > 0:
            xfail_reason = xfail_reasons[0]
            xfail_reason_checks = FailingReasonsChecks[xfail_reason.name].value
            ex_data = FailingReasonsFinder.get_exception_data(ex_message)
            # Checking multiple conditions. If any of the conditions is met, return True
            for xfail_reason_check in xfail_reason_checks:
                if xfail_reason_check(ex_data):
                    logger.trace(
                        f"Correct xfail reason: '{xfail_reason}' for exception: {type(exception_value)} '{exception_value}'"
                    )
                    return True
            logger.error(
                f"Wrong xfail reason: '{xfail_reason}' for exception: {type(exception_value)} '{exception_value}'"
            )
            return False
        else:
            logger.warning(
                f"Test is marked with xfail reason: '{xfail_reason}' but no check performed for exception: {type(exception_value)} '{exception_value}'"
            )
            return None
