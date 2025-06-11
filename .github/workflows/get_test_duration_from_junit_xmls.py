# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import xml.etree.ElementTree as ET
import json
import sys


def extract_test_case_info(xml_file):
    """
    Extract test case names and their execution times from a JUnit XML report.

    Args:
        xml_file (str): Path to the XML file.

    Returns:
        dict: A dictionary with test case names as keys and their execution time in seconds as values.
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        test_cases_info = {}

        for testsuite in root.findall("testsuite"):
            # Iterate over all <testcase> elements within the current <testsuite>
            for testcase in testsuite.findall("testcase"):
                try:
                    path = testcase.get("classname").replace(".", "/")
                    name = testcase.get("name")
                    test_cases_info[f"{path}.py::{name}"] = float(testcase.get("time", 0))
                except ValueError:
                    print(f"Warning: Non-numeric time value encountered in {xml_file} for test case '{name}'")

    except ET.ParseError as e:
        print(f"Error parsing XML file {xml_file}: {e}")

    return test_cases_info


def process_directory(directory):
    """
    Process all JUnit XML files in the directory and subdirectories.

    Args:
        directory (str): Path to the root directory.

    Returns:
        dict: A dictionary with test case names as keys and their execution times as values across all files.
    """
    all_test_cases = {}

    for subdir, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".xml"):
                xml_file_path = os.path.join(subdir, file)
                # Check if it's a JUnit XML report by looking for <testsuite> tag
                try:
                    test_cases_info = extract_test_case_info(xml_file_path)
                    all_test_cases.update(test_cases_info)
                except Exception as e:
                    print(f"Error reading file {xml_file_path}: {e}")

    return all_test_cases


def main():
    if len(sys.argv) != 3:
        print("Usage: python .github/workflows/get_test_duration_from_junit_xmls.py <directory> <json_output>")
        sys.exit(1)

    root_directory = sys.argv[1]
    output_file = sys.argv[2]
    print(f"Dir to process {root_directory}, output file {output_file}")

    test_case_data = process_directory(root_directory)
    json_output = json.dumps(test_case_data, indent=4)

    with open(output_file, "w") as f:
        f.write(json_output)

    print(f"Test case data has been written to {output_file}")


if __name__ == "__main__":
    main()
