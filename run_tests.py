# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: Apache-2.0
import subprocess
import os
import time
from datetime import datetime


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
    log_directory="test_logs",
    num_lines_before=1,
    num_lines_after=5,
    max_errors=None,
    max_tests_to_run=-1,
):
    """
    Runs all pytest files in the given directory, logging each test's output separately.
    Creates a summary with pass/fail counts and specific error messages for failures.
    """
    # Ensure the log directory exists
    os.makedirs(log_directory, exist_ok=True)

    test_files = [f for f in os.listdir(test_directory) if f.startswith("test_") or f.endswith("_test.py")]
    test_files = sorted(test_files)
    summary = {"passed": 0, "failed": 0, "failures": {}}

    for test_id, test_file in enumerate(test_files):
        if test_id > max_tests_to_run and max_tests_to_run > 0:
            break

        test_path = os.path.join(test_directory, test_file)
        log_file = os.path.join(log_directory, f"{test_file}_log.txt")

        print(f"Running test: {test_file}")

        start_time = time.time()

        try:
            # Run each test file as a separate subprocess with a timeout of 30 seconds
            result = subprocess.run(["pytest", test_path], check=True, capture_output=True, text=True, timeout=30)

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
            print(f"\tPassed ({elapsed_time:.2f} seconds)")
            summary["passed"] += 1

        except subprocess.TimeoutExpired as e:
            elapsed_time = time.time() - start_time
            error_message = "Test timed out after 30 seconds"

            # Do WH warm reset (potentially hang occurred)
            print("\tWarm reset...")
            os.system("/home/software/syseng/wh/tt-smi -lr all")

            # Log timeout error to a file
            with open(log_file, "w") as f:
                f.write("=== TIMEOUT ===\n")
                f.write(error_message)

            # Print timeout message with clear formatting
            print(f"\tFailed ({elapsed_time:.2f} seconds) - {error_message}")
            summary["failed"] += 1
            summary["failures"][test_file] = error_message

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
            print(f"\tFailed ({elapsed_time:.2f} seconds)")
            summary["failed"] += 1
            summary["failures"][test_file] = error_message

        except Exception as ex:
            elapsed_time = time.time() - start_time
            print(f"An unexpected error occurred while running {test_file}: {ex} ({elapsed_time:.2f} seconds)")

    # Print and log summary
    print("\n=== Test Summary ===")
    print(f"Total tests run: {len(test_files)}")
    print(f"Tests passed: {summary['passed']}")
    print(f"Tests failed: {summary['failed']}")

    # Write summary to a file with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(log_directory, f"summary_{timestamp}.txt")

    with open(summary_file, "w") as f:
        f.write(f"Total tests run: {len(test_files)}\n")
        f.write(f"Tests passed: {summary['passed']}\n")
        f.write(f"Tests failed: {summary['failed']}\n")

        if summary["failed"] > 0:
            f.write("\nFailed Tests:\n")
            for test, message in summary["failures"].items():
                f.write(f"\n{'#' * 9}\n")
                f.write(f"\nTest name: {test}\n")
                f.write(f"\n{'#' * 9}\n\n")
                error_context = extract_error_context(
                    os.path.join(log_directory, f"{test}_log.txt"),
                    num_lines_before=num_lines_before,
                    num_lines_after=num_lines_after,
                    max_errors=max_errors,
                )
                f.writelines(error_context)


if __name__ == "__main__":
    # Set your test directory here
    test_directory = "./generated_modules"  # Adjust this path to your test directory
    run_tests(test_directory, max_errors=1, max_tests_to_run=-1)  # Adjust max_errors as needed
