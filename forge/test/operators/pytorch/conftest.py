# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import _pytest
import _pytest.python
import _pytest.reports
import _pytest.runner
import pluggy.callers

import json
import pandas as pd

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

                # if reason is not valid, mark the test as failed
                if valid_reason == False:
                    report.outcome = "failed"


def analyze_report(report_path):
    with open(report_path, "r") as f:
        data = json.load(f)
    tests = data["tests"]

    df = pd.DataFrame(tests)

    df.drop(labels=["lineno", "keywords", "setup", "call", "user_properties", "teardown"], axis=1, inplace=True)
    df[["operator", "input_source"]] = df["nodeid"].str.extract(r"-(\w+)-(\w+)-")
    df.drop("nodeid", axis=1, inplace=True)
    df = df[["operator", "input_source", "outcome"]]
    df = df.rename(columns={"outcome": "test_result"})

    aggregated_df = df.groupby(["operator", "input_source", "test_result"]).size().reset_index(name="count")

    return aggregated_df.to_string()


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    report_path = "report.json"

    try:
        output = analyze_report(report_path)
        session.config._final_report = output
    except FileNotFoundError:
        session.config._final_report = "\n ANALYSIS REPORT: \nReport file not found!"


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    if hasattr(config, "_final_report"):
        print("\n ANALYSIS REPORT: \n")
        print(config._final_report)
        print("\n\n\n")
