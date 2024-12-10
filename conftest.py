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


@pytest.fixture(scope="function")
def forge_property_recorder(request, record_property):
    """
    Pytest fixture to initialize and manage a ForgePropertyHandler instance for recording test properties.

    The fixture sets up a property store along with a handler configured with default parameters:
      - Execution depth is set to 'CI_FAILURE'
      - Execution stage is set to 'FAILED_BEFORE_FORGE_COMPILATION_INITIATION'

    After the test runs, the fixture extracts any refined error message and failure category attached to the test item
    (if available) and records these details. Finally, the properties are stored using the provided record_property
    function regardless of the test outcome.

    Yields:
        An instance of ForgePropertyHandler.
    """

    # Create a property store that will hold all the properties recorded during test execution.
    forge_property_store = ForgePropertyStore()

    # Create a handler that uses the property store; the handler is responsible for recording and managing property details.
    forge_property_handler = ForgePropertyHandler(forge_property_store)

    # Set CI_FAILURE as default execution depth and FAILED_BEFORE_FORGE_COMPILATION_INITIATION as default execution stage
    forge_property_handler.record_execution_depth(ExecutionDepth.CI_FAILURE)
    forge_property_handler.record_execution_stage(ExecutionStage.FAILED_BEFORE_FORGE_COMPILATION_INITIATION)

    # Provide the handler instance to the test function so it can record properties during test execution.
    yield forge_property_handler

    try:
        # Retrieve any refined error message that might have been set during the test execution
        refined_error_message = getattr(request.node, "refined_error_message", None)

        # Check if:
        # 1. The refined error message exists.
        # 2. The handler is configured to record single operation details (record_single_op_details flag is True).
        # If either of these checks fail, exit without further recording.
        if refined_error_message is None or not forge_property_handler.record_single_op_details:
            return

        # Record the refined error message in the handler's property store.
        forge_property_handler.record_refined_error_message(refined_error_message)

        # Retrieve and record failure category if failure category exists
        failure_category = getattr(request.node, "failure_category", None)
        if failure_category is not None:
            forge_property_handler.record_failure_category(failure_category)

    finally:
        # Store the recorded properties using the 'record_property' function from pytest.
        forge_property_handler.store_property(record_property)


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
    logger.info(f"    Start:   {start_mem:.2f} MB")
    logger.info(f"    End:     {end_mem:.2f} MB")
    logger.info(f"    By test: {end_mem - start_mem:.2f} MB")
    logger.info(f"    Minimum: {min_mem:.2f} MB")
    logger.info(f"    Maximum: {max_mem:.2f} MB")
    logger.info(f"    Average: {avg_mem:.2f} MB")
