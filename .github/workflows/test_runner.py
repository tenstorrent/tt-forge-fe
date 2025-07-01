# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import subprocess
import sys
from pathlib import Path


def run_pytest(args=None):
    """Run pytest and return the output and exit code"""
    script_dir = Path(__file__).parent
    cmd = [sys.executable, script_dir / "run_tests.py"] + (args or [])

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
            sys.stdout.flush()  # Ensure real-time printing (or CI appears to hang)

        exit_code = process.wait()  # Wait for the process to complete
        return output, exit_code
    except Exception as e:
        err = f"Error running pytest: {e}"
        print(err)
        return err, 7


def remove_test(test_path, restart=False):
    """Remove the test from the crashed tests list"""
    file_path = Path(".pytest_tests_to_run")
    if not file_path.exists():
        return

    # Read all lines from file
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f.readlines()]

    # Filter lines based on restart flag
    if restart:
        filtered_lines = [line for line in lines if test_path not in line]
    else:
        # Find index of test_path line
        try:
            idx = next(i for i, line in enumerate(lines) if test_path in line) + 1
            filtered_lines = lines[idx:]
        except StopIteration:
            print("error: test_path not found in .pytest_tests_to_run")
            exit(8)

    # Write filtered lines back to file
    with open(file_path, "w") as f:
        for line in filtered_lines:
            f.write(line + "\n")


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
    # Check for --continue-after-crash option
    if "--continue-after-crash" in args:
        args.remove("--continue-after-crash")
        print("Running with --continue-after-crash option")
        restart_afer_crash = False
    else:
        restart_afer_crash = True

    while True:
        print("======================== Running pytest...")
        output, exit_code = run_pytest(args)

        if exit_code < 128 and exit_code >= 0:
            print(f"======================== No crashes detected (exit code {exit_code}).")
            with open("pytest.log", "w") as f:
                for line in output:
                    f.write(line + "\n")
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
                    remove_test(test_path, restart_afer_crash)
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
                    print(f"FAILED {test}")
                    f.write(f"FAILED {test}\n")

            if exit_code == 0:
                exit_code = 1

    # Delete .pytest_current_test_executing file if it exists
    Path(".pytest_current_test_executing").unlink(missing_ok=True)

    # Write the crashed tests to a file (if any)
    if len(crashed_tests) > 0:
        with open("crashed_pytest.log", "w") as f:
            for line in crashed_tests:
                f.write(line)

        if not run_crashed_tests:
            # This summary header for pytest.log is requuired for compatibility with Fail Inspector
            str = f"\n=========================== short test summary info ============================\n=== CRASHED TESTS, found {len(crashed_tests)}:"
            print(str)
            # Append failures to the pytest.log file
            with open("pytest.log", "a") as f:
                f.write(str + "\n")
                for test in crashed_tests:
                    print(f"FAILED {test}")
                    f.write(f"FAILED {test}\n")

            # If there are crashed tests change exit code to failure
            if exit_code == 0:
                exit_code = 1

    exit(exit_code)


if __name__ == "__main__":
    main()
