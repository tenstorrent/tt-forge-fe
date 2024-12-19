# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from test.operators.utils import FailingReasonsValidation, PyTestUtils

import _pytest
import _pytest.python
import _pytest.reports
import _pytest.runner
import pluggy.callers
import pytest
from loguru import logger


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: _pytest.python.Function, call: _pytest.runner.CallInfo):
    outcome: pluggy.callers._Result = yield
    report: _pytest.reports.TestReport = outcome.get_result()

    # This hook function is called after each step of the test execution (setup, call, teardown)
    if call.when == "call":  # 'call' is a phase when the test is actually executed

        if call.excinfo is not None:  # an exception occurred during the test execution

            logger.trace(
                f"Test: skipped: {report.skipped} failed: {report.failed} passed: {report.passed} report: {report}"
            )

            exception_value = call.excinfo.value
            xfail_reason = PyTestUtils.get_xfail_reason(item)
            if xfail_reason is not None:  # an xfail reason is defined for the test
                valid_reason = FailingReasonsValidation.validate_exception(exception_value, xfail_reason)

                # if reason is not valid, mark the test as failed
                if valid_reason == False:
                    report.outcome = "failed"
