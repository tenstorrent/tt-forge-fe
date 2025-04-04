# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time
import pytest
import psutil
import threading
from loguru import logger
from datetime import datetime
from forge.forge_property_utils import ForgePropertyHandler, ForgePropertyStore, ExecutionStage
from forge._C import ExecutionDepth


@pytest.fixture(scope="function", autouse=True)
def record_test_timestamp(record_property):
    start_timestamp = datetime.strftime(datetime.now(), "%Y-%m-%dT%H:%M:%S%z")
    record_property("start_timestamp", start_timestamp)
    yield
    end_timestamp = datetime.strftime(datetime.now(), "%Y-%m-%dT%H:%M:%S%z")
    record_property("end_timestamp", end_timestamp)


def refine_failure(failure_message: str):
    refined_failure_message = ""
    maximum_error_lines = 3
    lines = failure_message.splitlines(True)
    for idx, line in enumerate(lines):
        if refined_failure_message:
            return refined_failure_message
        if line.startswith("E "):
            if all(error_line.startswith("E ") for error_line in lines[idx : idx + maximum_error_lines]):
                refined_failure_message = [
                    error_line.replace("E ", "").strip("\n").strip()
                    for error_line in lines[idx : idx + maximum_error_lines]
                ]
                refined_failure_message = " ".join(refined_failure_message)
            else:
                refined_failure_message = line.replace("E ", "").strip("\n").strip()
    else:
        return None


@pytest.fixture(scope="function")
def forge_property_recorder(request, record_property):
    forge_property_store = ForgePropertyStore()

    forge_property_handler = ForgePropertyHandler(forge_property_store)

    # Set CI_FAILURE as default execution depth and FAILED_BEFORE_FORGE_COMPILATION_INITIATION as default execution stage
    forge_property_handler.record_execution_depth(ExecutionDepth.CI_FAILURE)
    forge_property_handler.record_execution_stage(ExecutionStage.FAILED_BEFORE_FORGE_COMPILATION_INITIATION)

    yield forge_property_handler

    report = getattr(request.node, "rep_call", None)
    if (
        report
        and report.when == "call"
        and (report.failed or (hasattr(report, "wasxfail") and report.outcome == "skipped"))
    ):
        failure_message = getattr(report, "longreprtext", None)
        if failure_message:
            refined_failure_message = refine_failure(failure_message)
            if refined_failure_message is not None and forge_property_handler.record_single_op_details:
                forge_property_handler.record_refined_error_message(refined_failure_message)

    forge_property_handler.store_property(record_property)


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    setattr(item, "rep_" + rep.when, rep)


@pytest.fixture(autouse=True)
def memory_usage_tracker():
    """
    A pytest fixture that tracks memory usage during the execution of a test.

    This fixture automatically tracks the memory usage of the process running the tests.
    It starts tracking before the test runs, continues tracking in a background thread during the test,
    and stops tracking after the test completes. It logs the memory usage statistics including the
    minimum, maximum, average, and total memory usage by the test.

    The memory usage is measured in megabytes (MB).

    Note:
        - This fixture is automatically used for all tests due to the `autouse=True` parameter.
        - The interval for memory readings can be adjusted by changing the sleep duration in the `track_memory` function.
        - Min, max, and avg memory usage are calculated based on the recorded memory readings from system memory.
    """
    process = psutil.Process()

    # Initialize memory tracking variables
    start_mem = process.memory_info().rss / (1024 * 1024)  # MB
    min_mem = start_mem
    max_mem = start_mem
    total_mem = start_mem
    count = 1

    # Start a background thread or loop to collect memory usage over time
    tracking = True

    def track_memory():
        nonlocal min_mem, max_mem, total_mem, count
        while tracking:
            current_mem = process.memory_info().rss / (1024 * 1024)
            min_mem = min(min_mem, current_mem)
            max_mem = max(max_mem, current_mem)
            total_mem += current_mem
            count += 1
            time.sleep(0.1)  # Adjust the interval as needed

    # Start tracking in a background thread
    import threading

    tracker_thread = threading.Thread(target=track_memory)
    tracker_thread.start()

    # Run the test
    yield

    # Stop tracking and wait for the thread to finish
    tracking = False
    tracker_thread.join()

    # Calculate end memory and memory usage stats
    end_mem = process.memory_info().rss / (1024 * 1024)  # MB
    min_mem = min(min_mem, end_mem)
    max_mem = max(max_mem, end_mem)
    total_mem += end_mem
    count += 1
    avg_mem = total_mem / count

    # Log memory usage statistics
    logger.info(f"Test memory usage:")
    logger.info(f"    By test: {end_mem - start_mem:.2f} MB")
    logger.info(f"    Minimum: {min_mem:.2f} MB")
    logger.info(f"    Maximum: {max_mem:.2f} MB")
    logger.info(f"    Average: {avg_mem:.2f} MB")
