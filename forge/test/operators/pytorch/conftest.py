# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import _pytest
import _pytest.python
import _pytest.reports
import _pytest.runner
import pluggy.callers

from loguru import logger

from test.operators.utils import PyTestUtils
from test.operators.utils import FailingReasonsValidation


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

                # if reason is not valid, mark the test as failed and keep the original exception
                if valid_reason == False:
                    # Replace test report with a new one with outcome set to 'failed' and exception details
                    new_report = _pytest.reports.TestReport(
                        item=item,
                        when=call.when,
                        outcome="failed",
                        longrepr=call.excinfo.getrepr(style="long"),
                        sections=report.sections,
                        nodeid=report.nodeid,
                        location=report.location,
                        keywords=report.keywords,
                    )
                    outcome.force_result(new_report)
            else:
                logger.debug(f"Test '{item.name}' failed with exception: {type(exception_value)} '{exception_value}'")
