# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import subprocess
import sys
from pathlib import Path


def run_pytest(args=None, crashed_tests=None):
    """Run pytest and return the output and exit code"""
    cmd = ["pytest"] + (args or [])

    # Add deselects from crashed tests file if it exists
    cmd.extend(f"--deselect={test}" for test in crashed_tests)

    # Delete .pytest_current_test_executing file if it exists
    Path(".pytest_current_test_executing").unlink(missing_ok=True)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout + result.stderr, result.returncode
    except Exception as e:
        print(f"Error running pytest: {e}")
        return "", 1


def main():
    # Get command line arguments
    args = sys.argv[1:]
    crashed_tests = []

    while True:
        print("========================\nRunning pytest...")
        output, exit_code = run_pytest(args, crashed_tests)

        if exit_code <= 130:
            print("No crashes detected.")
            break

        print(f"Crash detected with exit code {exit_code}")
        # If .pytest_current_test_executing exists, append its content to crashed tests list
        current_test_file = Path(".pytest_current_test_executing")
        if current_test_file.exists():
            test_path = current_test_file.read_text().strip()
            if test_path and test_path not in crashed_tests:
                crashed_tests.append(test_path)
                print(f"Added crashed test: {test_path}")

        else:
            print("CRASH detected, but no current test executing file found.")
            break

    # Run only crashed tests with
    crashed_tests_twice = []
    for test in crashed_tests:
        print(f"========================\nRunning crashed test again: {test}")
        result = subprocess.run(["pytest", test], capture_output=True, text=True)
        print(result.stdout)
        if result.returncode == 0:
            print(f"Test {test} succeeded")
        else:
            print(f"Test {test} failed again with exit code {result.returncode}")
            crashed_tests_twice.append(test)
            print(result.stderr)

    print(output)
    if len(crashed_tests_twice) > 0:
        print(f"\n========================\nFound {len(crashed_tests_twice)} crashed tests (twice):")
        for test in crashed_tests_twice:
            print(f"FAILURE {test}")


if __name__ == "__main__":
    main()
