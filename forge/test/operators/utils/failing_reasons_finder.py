# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from loguru import logger
from typing import Generator, Optional

from .failing_reasons import ExceptionData
from .failing_reasons import FailingReasons
from .failing_reasons import FailingReason


class FailingReasonsFinder:
    unique_messages = set()

    @classmethod
    def get_exception_data(cls, error_message: str, error_log: str) -> ExceptionData:
        error_message_line = error_message.split("\n")[0]
        class_name = error_message_line[:50].split(":")[0]
        error_message = error_message[len(class_name) + 2 :]
        # logger.info(class_name)
        # logger.info(f"Line: {class_name} | {error_message}")
        ex = ExceptionData(class_name, error_message, error_log)
        return ex

    @classmethod
    def find_reason(cls, error_message: str, error_log: str):
        ex = cls.get_exception_data(error_message, error_log)
        return cls.find_reason_by_ex_data(ex)

    @classmethod
    def find_reason_by_ex_data(cls, ex: ExceptionData) -> Optional[FailingReasons]:
        reasons = list(cls.find_reasons(ex))
        if not reasons:
            return None
        if len(reasons) > 1:
            message_line = ex.message.split("\n")[0]
            message_id = f"{reasons} {ex.class_name} {message_line}"
            if message_id not in cls.unique_messages:
                cls.unique_messages.add(message_id)
                logger.warning(f"Multiple reasons found: {reasons} for ex: {ex}")
        return reasons[0]

    @classmethod
    def find_reasons(cls, ex: ExceptionData) -> Generator[FailingReasons, None, None]:
        for failing_reason in FailingReasons:
            # Checking if exception data matches the failing reason
            if ex in failing_reason.value:
                yield failing_reason
                # Skip other failing checks for the same xfail reason
                # Uncomment the next line to stop verbose logging
                # break
