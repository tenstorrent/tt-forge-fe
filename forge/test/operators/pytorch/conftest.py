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

from ..utils import TestPlanUtils


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: _pytest.python.Function, call: _pytest.runner.CallInfo):
    outcome: pluggy.callers._Result = yield
    report: _pytest.reports.TestReport = outcome.get_result()

    if report.when == "call" or (report.when == "setup" and report.skipped):
        xfail_reason = PyTestUtils.get_xfail_reason(item)

    # This hook function is called after each step of the test execution (setup, call, teardown)
    if call.when == "call":  # 'call' is a phase when the test is actually executed

        if call.excinfo is not None:  # an exception occurred during the test execution

            logger.trace(
                f"Test: skipped: {report.skipped} failed: {report.failed} passed: {report.passed} report: {report}"
            )

            exception_value = call.excinfo.value

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

    if report.when == "call" or (report.when == "setup" and report.skipped):
        try:
            log_test_vector_properties(item, report, xfail_reason)
        except Exception as e:
            logger.error(f"Failed to log test vector properties: {e}")
            logger.exception(e)
            pass


def log_test_vector_properties(item: _pytest.python.Function, report: _pytest.reports.TestReport, xfail_reason: str):
    original_name = item.originalname
    test_id = item.name
    test_id = test_id.replace(f"{original_name}[", "")
    test_id = test_id.replace("]", "")
    if test_id == "no_device-test_vector0":
        # This is not a valid test id. It happens when no tests are selected to run.
        return
    test_vector = TestPlanUtils.test_id_to_test_vector(test_id)

    item.user_properties.append(("id", test_id))
    item.user_properties.append(("operator", test_vector.operator))
    item.user_properties.append(
        ("input_source", test_vector.input_source.name if test_vector.input_source is not None else None)
    )
    item.user_properties.append(
        ("dev_data_format", test_vector.dev_data_format.name if test_vector.dev_data_format is not None else None)
    )
    item.user_properties.append(
        ("math_fidelity", test_vector.math_fidelity.name if test_vector.math_fidelity is not None else None)
    )
    item.user_properties.append(("input_shape", test_vector.input_shape))
    item.user_properties.append(("kwargs", test_vector.kwargs))
    if xfail_reason is not None:
        item.user_properties.append(("xfail_reason", xfail_reason))
    item.user_properties.append(("outcome", report.outcome))
