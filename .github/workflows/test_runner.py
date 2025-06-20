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
    print(f"Running command: {' '.join(cmd)}")

    # Delete .pytest_current_test_executing file if it exists
    Path(".pytest_current_test_executing").unlink(missing_ok=True)

    try:
        # Redirect both stdout and stderr to the same stream
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        output = []
        # Print output line by line in real-time
        for line in process.stdout:
            print(line, end="")  # Print to stdout
            output.append(line)  # Capture the output

        process.wait()  # Wait for the process to complete
        return output, process.returncode
    except Exception as e:
        err = f"Error running pytest: {e}"
        print(err)
        return err, 7


def main():
    # Get command line arguments
    args = sys.argv[1:]
    crashed_tests = []

    # Check for --run-crashed-tests option
    if "--run-crashed-tests" in args:
        args.remove("--run-crashed-tests")
        print("Running with --run-crashed-tests option")
        run_crashed_tests = True
    else:
        run_crashed_tests = False

    while True:
        print("======================== Running pytest...")
        output, exit_code = run_pytest(args, crashed_tests)

        if exit_code <= 130:
            print(f"======================== No crashes detected (exit code {exit_code}).")
            with open("pytest.log", "w") as f:
                for line in output:
                    f.write(line)
            break

        print(f"======================== Crash detected with exit code {exit_code}")
        # If .pytest_current_test_executing exists, append its content to crashed tests list
        current_test_file = Path(".pytest_current_test_executing")
        if current_test_file.exists():
            test_path = current_test_file.read_text().strip()
            if test_path:
                if test_path not in crashed_tests:
                    crashed_tests.append(test_path)
                    print(f"    Crashed test: {test_path}")
                else:
                    # If the test is already in the crashed tests list, print a message and exit
                    print(f"    Test {test_path} already in crashed tests list\nInternal error, exiting.")
                    run_crashed_tests = False
                    break
            else:
                print("CRASH detected, but .pytest_current_test_executing file is empty. Exiting.")
                run_crashed_tests = False
                break
        else:
            print("CRASH detected, but no current test executing file found. Exiting.")
            run_crashed_tests = False
            break

    # Run only crashed tests one by one (if --run-crashed-tests is specified)
    if run_crashed_tests:
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

        if len(crashed_tests_twice) > 0:
            print(f"\n========================\nFound {len(crashed_tests_twice)} crashed tests (twice):")
            # Append failures to the pytest.log file
            with open("pytest.log", "a") as f:
                for test in crashed_tests_twice:
                    print(f"FAILURE {test}")
                    f.write(f"FAILURE {test}\n")

    # Delete .pytest_current_test_executing file if it exists
    Path(".pytest_current_test_executing").unlink(missing_ok=True)

    # Write the crashed tests to a file (if any)
    if len(crashed_tests) > 0:
        with open("crashed_pytest.log", "w") as f:
            for line in crashed_tests:
                f.write(line)


if __name__ == "__main__":
    main()
