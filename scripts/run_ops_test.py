# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import subprocess
import os
import time
from datetime import datetime
from loguru import logger
import argparse


def collect_all_pytests(root_dir_path):

    logger.info(f"Collecting all pytests in {root_dir_path}")

    try:
        res = subprocess.check_output(["pytest", root_dir_path, "--setup-plan"], stderr=subprocess.STDOUT).decode(
            "utf-8"
        )
    except subprocess.CalledProcessError as e:
        output = e.output.decode("utf-8")
        logger.error(f"[Error!] output = {output}")
        return []

    test_list = []
    lines = res.split("\n")
    for line in lines:
        if "warnings summary" in line or "slowest durations" in line:
            break

        if line and line.startswith("        " + root_dir_path) and "::" in line and "training" not in line:
            line = line.strip()
            line = line.split(" (fixtures used:")[0] if " (fixtures used:" in line else line
            if "Grayskull" not in line and "Wormhole_B0" not in line:
                test_list.append(line)

    return test_list


def extract_error_context(log_file, keyword="error", num_lines_before=0, num_lines_after=0, max_errors=None):
    """
    Extracts lines around the keyword from the log file.
    """
    context_lines = []
    error_count = 0
    with open(log_file, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if keyword in line.lower():
                if max_errors is not None and error_count >= max_errors:
                    break

                start = max(i - num_lines_before, 0)
                end = min(i + num_lines_after + 1, len(lines))

                context_lines.append(f"\nError context around line {i + 1}:\n")
                context_lines.append("---\n")  # Divider between error contexts
                context_lines.append("".join(lines[start:end]))
                context_lines.append("\n---\n")  # Divider between error contexts
                error_count += 1

    return context_lines


def run_tests(
    test_directory,
    single_op_test,
    unique_op_test,
    log_directory,
    num_lines_before,
    num_lines_after,
    max_errors,
    max_tests_to_run,
):
    """
    Runs all pytest files in the given directory, logging each test's output separately.
    Creates a summary with pass/fail counts and specific error messages for failures.
    """
    if not (single_op_test or unique_op_test):
        logger.warning("Set single_op_test or unique_op_test argument to True.")

    ops_test_directories = []
    for folder_name in os.listdir(test_directory):
        if folder_name == "single_ops" and single_op_test:
            ops_test_directories.append(os.path.join(test_directory, folder_name))
        elif folder_name == "unique_ops" and unique_op_test:
            ops_test_directories.append(os.path.join(test_directory, folder_name))

    if len(ops_test_directories) == 0:
        logger.error(f"There is no unique_ops/single_ops folder inside {test_directory} so please generate op tests")

    for ops_test_directory in ops_test_directories:

        module_names = os.listdir(ops_test_directory)

        for module_name in module_names:

            module_path = os.path.join(ops_test_directory, module_name)

            test_list = collect_all_pytests(module_path)

            if len(test_list) == 0:
                logger.warning(f"No pytests found inside {module_path}")
                continue

            test_files = {}
            for test in test_list:
                test_file = test.split("::")[0]
                if test_file not in test_files.keys():
                    test_files[test_file] = [test]
                else:
                    test_files[test_file].append(test)

            module_log_directory = os.path.join(log_directory, module_path)
            summary = {"passed": 0, "failed": 0, "failures": {}}
            test_count = 0
            for test_file, tests in test_files.items():

                if test_count > max_tests_to_run and max_tests_to_run > 0:
                    break

                for test_idx, test in enumerate(tests):

                    test_count += 1
                    log_file_dir = module_log_directory
                    test_name = test_file.split("/")[-1].split(".")[0]
                    ops_tests_category = ops_test_directory.split("/")[-1]
                    if ops_tests_category == "unique_ops":
                        op_name = test_name.split("_")[-1]
                        log_file_dir = os.path.join(log_file_dir, op_name)
                        log_file = os.path.join(log_file_dir, f"{test_name}_{str(test_idx)}_log.txt")
                    else:
                        log_file = os.path.join(log_file_dir, f"{test_name}_log.txt")

                    os.makedirs(log_file_dir, exist_ok=True)

                    logger.info(f"Running the test: {test}")

                    start_time = time.time()
                    try:
                        # Run each test file as a separate subprocess with a timeout of 30 seconds
                        result = subprocess.run(
                            ["pytest", test, "-vss"], check=True, capture_output=True, text=True, timeout=60
                        )

                        # Log output to a file
                        with open(log_file, "w") as f:
                            if result.stderr:
                                f.write("=== STDERR ===\n")
                                f.write(result.stderr)
                            if result.stdout:
                                f.write("=== STDOUT ===\n")
                                f.write(result.stdout)

                        elapsed_time = time.time() - start_time
                        # Print pass message with clear formatting
                        logger.info(f"\tPassed ({elapsed_time:.2f} seconds)")
                        logger.info(f"Dumped test logs in {log_file}")
                        summary["passed"] += 1

                    except subprocess.TimeoutExpired as e:
                        elapsed_time = time.time() - start_time
                        error_message = "Test timed out after 30 seconds"

                        # Do WH warm reset (potentially hang occurred)
                        logger.info("\tWarm reset...")
                        os.system("/home/software/syseng/wh/tt-smi -lr all")

                        # Log timeout error to a file
                        with open(log_file, "w") as f:
                            f.write("=== TIMEOUT ===\n")
                            f.write(error_message)

                        # Print timeout message with clear formatting
                        logger.info(f"\tFailed ({elapsed_time:.2f} seconds) - {error_message}")
                        logger.info(f"Dumped test logs in {log_file}")
                        summary["failed"] += 1
                        summary["failures"][test] = log_file

                    except subprocess.CalledProcessError as e:
                        # Log output to a file
                        with open(log_file, "w") as f:
                            if e.stderr:
                                f.write("=== STDERR ===\n")
                                f.write(e.stderr)
                            if e.stdout:
                                f.write("=== STDOUT ===\n")
                                f.write(e.stdout)

                        elapsed_time = time.time() - start_time
                        error_message = e.stderr

                        # Print fail message with clear formatting
                        logger.info(f"\tFailed ({elapsed_time:.2f} seconds)")
                        logger.info(f"Dumped test logs in {log_file}")
                        summary["failed"] += 1
                        summary["failures"][test] = log_file

                    except Exception as ex:
                        elapsed_time = time.time() - start_time
                        logger.info(
                            f"An unexpected error occurred while running {test}: {ex} ({elapsed_time:.2f} seconds)"
                        )

            # Print and log summary
            logger.info(f"==============={module_name} Test Summary ===============")
            logger.info(f"Total tests run: {test_count}")
            logger.info(f"Tests passed: {summary['passed']}")
            logger.info(f"Tests failed: {summary['failed']}")

            # Write summary to a file with a timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = os.path.join(module_log_directory, f"summary_{timestamp}.txt")

            with open(summary_file, "w") as f:
                f.write(f"Total tests run: {test_count}\n")
                f.write(f"Tests passed: {summary['passed']}\n")
                f.write(f"Tests failed: {summary['failed']}\n")

                if summary["failed"] > 0:
                    f.write("\nFailed Tests:\n")
                    for test, test_log_file in summary["failures"].items():
                        f.write(f"\n{'#' * 9}\n")
                        f.write(f"\nTest name: {test}\n")
                        f.write(f"\n{'#' * 9}\n\n")
                        error_context = extract_error_context(
                            test_log_file,
                            num_lines_before=num_lines_before,
                            num_lines_after=num_lines_after,
                            max_errors=max_errors,
                        )
                        f.writelines(error_context)


def main():
    parser = argparse.ArgumentParser(
        description="Run generated single/unique ops test and store the test logs and summary"
    )

    parser.add_argument(
        "--test_directory",
        type=str,
        default="generated_modules",
        help="Path to the directory containing single/unique ops tests",
    )
    parser.add_argument(
        "--single_op_test",
        action="store_true",
        help="Run the tests inside the single_ops directory (default: False).",
    )
    parser.add_argument(
        "--unique_op_test",
        action="store_true",
        help="Run the tests inside the unique_ops directory (default: False).",
    )
    parser.add_argument(
        "--log_directory",
        type=str,
        default="test_logs",
        help="Directory to store test logs and summary (default: 'test_logs').",
    )
    parser.add_argument(
        "--num_lines_before",
        type=int,
        default=1,
        help="Number of lines to include before an error in summary (default: 1).",
    )
    parser.add_argument(
        "--num_lines_after",
        type=int,
        default=5,
        help="Number of lines to include after an error in summary (default: 5).",
    )
    parser.add_argument(
        "--max_errors",
        type=int,
        default=1,
        help="Maximum number of errors allowed before stopping (default: 1).",
    )
    parser.add_argument(
        "--max_tests_to_run",
        type=int,
        default=-1,
        help="Maximum number of tests to run (default: -1 for no limit).",
    )

    args = parser.parse_args()

    run_tests(
        test_directory=args.test_directory,
        single_op_test=args.single_op_test,
        unique_op_test=args.unique_op_test,
        log_directory=args.log_directory,
        num_lines_before=args.num_lines_before,
        num_lines_after=args.num_lines_after,
        max_errors=args.max_errors,
        max_tests_to_run=args.max_tests_to_run,
    )


if __name__ == "__main__":
    main()
