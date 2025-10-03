# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import argparse
from loguru import logger
from utils import check_path
from models_ops_test_failure_update import read_file, remove_timestamp
from typing import List
import re


def detect_crash(lines: List[str]):
    """Detect whether the given log lines report crashed tests.

    The function looks for a crash header line matching ``CRASHED TESTS, found N :``
    (case-insensitive). If the captured count is > 0, it scans subsequent lines
    for a FAILED entry that also contains the substring ``forge/test`` which is
    used to identify relevant testcases.

    Args:
        lines: List of lines from a pytest log file (newline preserved is OK).

    Returns:
        True if a crash header with at least one following FAILED forge/test line
        is present; False otherwise.
    """
    crash_header_pattern = re.compile(r"^\s*={3,}\s*CRASHED TESTS,\s*found\s*(\d+)\s*:\s*$", re.IGNORECASE)
    contains_crashed_tests = False

    # Iterate lines and search for the crash header. Once found, scan following
    # lines to check whether any of them indicate a FAILED test for forge/test.
    for idx, line in enumerate(lines):
        match = crash_header_pattern.match(line.rstrip("\n"))
        if match:
            # Extract the numeric count from the header (defensive parse).
            count = int(match.group(1)) if match.group(1).isdigit() else 0
            if count > 0:
                # Search lines after the header for evidence of FAILED tests.
                for crahed_cases in lines[idx + 1 :]:

                    # We only flag a crash as relevant if it contains both
                    # "FAILED" and the path portion "forge/test".
                    if "FAILED" in crahed_cases and "forge/test" in crahed_cases:
                        contains_crashed_tests = True
                        break
            # If we already found a crashed test, stop scanning further.
            if contains_crashed_tests:
                break
    return contains_crashed_tests


def parse_failed_tests(lines: List[str]):
    """Parse FAILED tests from pytest output lines.

    This function looks for lines that include a test node id followed by a
    test status token (PASSED, FAILED, SKIPPED, XFAIL). When a status of
    "FAILED" is found, the preceding token (test node id) is collected.

    Args:
        lines: List of lines from the (timestamp-stripped) log.

    Returns:
        A set of unique test node ids that reported FAILED.
    """
    test_result_pattern = re.compile(r"^(.*?)\s+(PASSED|FAILED|SKIPPED|XFAIL).*")
    failed_tests = set()
    for line in lines:
        match = test_result_pattern.match(line)
        if match:
            test_case = match.group(1).strip()
            status = match.group(2).strip()
            if status == "FAILED":
                failed_tests.add(test_case)
    return failed_tests


def collect_failed_tests_from_logs(log_files: List[str]):
    """Scan multiple log files and collect FAILED test node ids when crashes occur.

    For every provided log file this function will:
    1. verify the path exists using ``check_path``;
    2. read the file using ``read_file``;
    3. remove timestamps using ``remove_timestamp``;
    4. call ``detect_crash``; and
    5. parse FAILED tests with ``parse_failed_tests`` if a crash is detected.

    Args:
        log_files: List of file paths to pytest log files.

    Returns:
        A sorted list of unique FAILED test node ids collected from all logs.
    """
    failed_test_cases = set()
    for log_file in log_files:
        try:
            if not check_path(log_file):
                logger.warning(f"Provided path does not exist: {log_file}")
                continue

            logger.info(f"Processing log file: {log_file}")
            lines = read_file(log_file)
            lines = remove_timestamp(lines)
            if detect_crash(lines):
                logger.info(f"Crash detected in {log_file} — collecting FAILED tests")
                failed_tests = parse_failed_tests(lines)
                if failed_tests:
                    failed_test_cases.update(failed_tests)
                else:
                    logger.info(f"No failed tests found in {log_file}")
            else:
                logger.info(f"No crash detected in {log_file} — skipping FAILED test collection.")

        except Exception as e:
            logger.exception(f"Error while processing {log_file}: {e}")

    return sorted(failed_test_cases)


def write_results(tests: List[str], out_file: str):
    """
    Write collected test node ids to an output file, one per line.

    If the tests list is empty the function will only log an informational
    message and not create/overwrite the output file.

    Args:
        tests: A list of test node ids to write.
        out_file: Path to the output file to write the test ids to.
    """
    if not tests:
        logger.info("No FAILED tests found in provided logs.")
        return
    else:
        logger.info(f"Collected {len(tests)} FAILED tests:")
        for t in tests:
            logger.info(f"{t}")

    try:
        with open(out_file, "w") as f:
            for t in tests:
                f.write(t + "\n")
        logger.info(f"Wrote results to {out_file}")
    except Exception as e:
        logger.exception(f"Failed to write results to {out_file}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Collect failed tests from pytest logs only when a crash is detected.")
    parser.add_argument(
        "--file_paths",
        nargs="+",
        type=str,
        required=True,
        help="List of pytest log file paths to scan",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="failed_tests_collected.txt",
        help="Output file path for the collected failed tests",
    )
    args = parser.parse_args()
    failed_tests = collect_failed_tests_from_logs(args.file_paths)
    write_results(failed_tests, args.out_file)


if __name__ == "__main__":
    main()
