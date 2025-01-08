# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time
import pytest
import psutil
import threading
from loguru import logger
from datetime import datetime


@pytest.fixture(scope="function", autouse=True)
def record_test_timestamp(record_property):
    start_timestamp = datetime.strftime(datetime.now(), "%Y-%m-%dT%H:%M:%S%z")
    record_property("start_timestamp", start_timestamp)
    yield
    end_timestamp = datetime.strftime(datetime.now(), "%Y-%m-%dT%H:%M:%S%z")
    record_property("end_timestamp", end_timestamp)


@pytest.fixture(scope="function", autouse=True)
def record_forge_property(record_property):
    """
    A pytest fixture that automatically records a property named 'frontend' with the value 'tt-forge-fe'
    for each test function. This fixture is applied to all test functions due to `autouse=True`.

    Parameters:
    ----------
    record_property : function
        A pytest built-in function used to record test metadata, such as custom properties or
        additional information about the test execution.

    Yields:
    -------
    function
        The `record_property` function, allowing tests to add additional properties if needed.

    Usage:
    ------
    def test_model(record_forge_property):
        # Record Forge Property
        record_forge_property("key", value)
    """
    # Record default properties for forge
    record_property("frontend", "tt-forge-fe")

    yield record_property


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


@pytest.mark.hookwrapper
def pytest_runtest_makereport(item, call):
    """
    A pytest hookwrapper for customizing the test report. This hook is triggered after each
    test function executes and allows access to the test's outcome and additional user-defined properties.

    Parameters:
    ----------
    item : pytest.Item
        The test item object representing the test function or parameterized test case.
    call : pytest.CallInfo
        An object containing information about the test function's call, including its result
        or exception details.

    Yields:
    -------
    None
        The hook yields control back to pytest, allowing it to finalize the test report.

    Behavior:
    ---------
    - Retrieves the test result via `outcome.get_result()`.
    - Checks for user-defined properties in `item.user_properties`.
    - If properties exist, converts them into a dictionary and prints them to the console.

    Notes:
    ------
    - User-defined properties can be set using `record_property` or added to `item.user_properties`.
    - This hookwrapper is useful for augmenting test reports with additional metadata.
    """
    outcome = yield
    report = outcome.get_result()

    if item.user_properties:
        test_properties = {prop[0]: prop[1] for prop in item.user_properties}
        print(test_properties)
