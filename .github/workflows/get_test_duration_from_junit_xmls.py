# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import xml.etree.ElementTree as ET
import json


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

        # Namespace map for parsing JUnit XML
        namespace = {"ns": "http://xmlns.jenkins.io/manifests/junit"}

        test_cases_info = {}

        # Iterate through each testcase element in the XML file
        for testcase in root.findall("ns:testsuite/ns:testcase", namespace):
            name = testcase.get("name")
            time_str = testcase.get("time", "0")  # Default to 0 if time not specified

            try:
                time_seconds = float(time_str)
                test_cases_info[name] = time_seconds
            except ValueError:
                print(f"Warning: Non-numeric time value encountered in {xml_file} for test case '{name}'")

        return test_cases_info

    except ET.ParseError as e:
        print(f"Error parsing XML file {xml_file}: {e}")
        return {}


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
    test_case_data = process_directory(root_directory)

    json_output = json.dumps(test_case_data, indent=4)

    output_file = sys.argv[2]
    with open(output_file, "w") as f:
        f.write(json_output)

    print(f"Test case data has been written to {output_file}")


if __name__ == "__main__":
    main()
