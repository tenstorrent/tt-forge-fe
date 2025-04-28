# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Any

from loguru import logger

import os
import re

import xml.etree.ElementTree as ET


class FilesUtils:

    # Read env var DOCKER_TOOLS_DIR
    DOCKER_TOOLS_DIR = os.getenv("DOCKER_TOOLS_DIR")

    @classmethod
    def log_file_path(cls, file: str) -> str:

        if not file.startswith("/"):
            file = f"{cls.DOCKER_TOOLS_DIR}/log/{file}"

        return file

    @classmethod
    def is_sweeps_file(cls, xml_file: str) -> bool:
        return not "random_test_graphs" in xml_file


class ReportUtils:
    @classmethod
    def get_property_value(cls, testcase: ET.Element, property_name: str) -> str:
        properties = testcase.find("properties")
        classname = testcase.get("classname")
        name = testcase.get("name")
        if properties is None:
            logger.warning(f"No properties found in testcase {classname} {name}")
            return None
        properties = [prop for prop in properties.findall("property") if prop.get("name") == property_name]
        if not properties:
            return None
        property_value = properties[0].get("value")
        if property_value == "None":
            return None
        return property_value

    @classmethod
    def load_from_xml_file(cls, xml_file: str) -> List[Dict[str, Any]]:

        xml_file = FilesUtils.log_file_path(xml_file)

        is_sweeps_file = True

        logger.info(f"Load {xml_file}")

        tree = ET.parse(xml_file)
        root = tree.getroot()

        data = []
        for testcase in root.findall(".//testcase"):
            name = testcase.get("name", "Unknown")
            classname = testcase.get("classname", "Unknown")
            time = testcase.get("time", "0")
            error_message = None
            error_log = None
            captured_output = None

            status = "passed"
            if testcase.find("failure") is not None:
                status = "failed"
                error_message = testcase.find("failure").get("message")
                error_log = testcase.find("failure").text
                # if error_log:
                #     error_log = cls.remove_colors(error_log)
            elif testcase.find("skipped") is not None:
                if testcase.find("skipped").get("type") == "pytest.xfail":
                    status = "xfailed"
                elif testcase.find("skipped").get("message") == "xfail-marked test passes unexpectedly":
                    status = "xpassed"
                else:
                    status = "skipped"

            if is_sweeps_file:

                id = cls.get_property_value(testcase, "id")
                if id is None:
                    logger.warning(f"No id found in testcase {classname} {name}")
                else:
                    id = id.replace("no_device-", "")
                operator = cls.get_property_value(testcase, "operator")
                input_source = cls.get_property_value(testcase, "input_source")
                input_shape = cls.get_property_value(testcase, "input_shape")
                kwargs = cls.get_property_value(testcase, "kwargs")
                dev_data_format = cls.get_property_value(testcase, "dev_data_format")
                math_fidelity = cls.get_property_value(testcase, "math_fidelity")
                xfail_reason = cls.get_property_value(testcase, "xfail_reason")
                # exception_type = cls.get_property_value(testcase, "exception_type")
                # exception_message_short = cls.get_property_value(testcase, "exception_message_short")
                # exception_message = cls.get_property_value(testcase, "exception_message")
                rtol = cls.get_property_value(testcase, "all_close_rtol")
                atol = cls.get_property_value(testcase, "all_close_atol")
                captured_output = cls.get_property_value(testcase, "captured_output")
                # if captured_output:
                #     captured_output = cls.remove_colors(captured_output)
                if rtol is not None:
                    rtol = float(rtol)
                if atol is not None:
                    atol = float(atol)

                outcome = cls.get_property_value(testcase, "outcome")

                if status == "passed" and xfail_reason is not None:
                    status = "xpassed"

                data.append(
                    {
                        "id": id,
                        # "name": name,
                        # "classname": classname,
                        "operator": operator,
                        "input_source": input_source,
                        "input_shape": input_shape,
                        "kwargs": kwargs,
                        "dev_data_format": dev_data_format,
                        "math_fidelity": math_fidelity,
                        "xfail_reason": xfail_reason,
                        "outcome": outcome,
                        "status": status,
                        "error_message": error_message,
                        "error_log": error_log,
                        "captured_output": captured_output,
                        # "exception_type": exception_type,
                        # "exception_message_short": exception_message_short,
                        # # "exception_message": exception_message,
                        "rtol": rtol,
                        "atol": atol,
                        "time": float(time),
                    }
                )
            else:
                test_file = cls.get_property_value(testcase, "test_file")
                test_code = cls.get_property_value(testcase, "test_code")
                data.append(
                    {
                        "name": name,
                        # "classname": classname,
                        "status": status,
                        "error_message": error_message,
                        "error_log": error_log,
                        "test_file": test_file,
                        "test_code": test_code,
                        "time": float(time),
                    }
                )

        for row in data:
            error_message = row.get("error_message")
            row["error_message_full"] = error_message

        return data


class EmulatorError(Exception):
    def __init__(self, error_message: str, error_log: str = None):
        super().__init__(error_message)
        self.error_message = error_message
        self.error_log = error_log


class ReportCache:
    def __init__(self, xml_files: List[str]):
        xml_files = [xml_file.replace(".log", "_format.xml") for xml_file in xml_files]
        self.xml_files = xml_files
        self.data = []
        for xml_file in xml_files:
            logger.info(f"Loading {xml_file}")
            self.data += ReportUtils.load_from_xml_file(xml_file)
        self.test_ids = {}
        for row in self.data:
            if "id" in row:
                test_id = row["id"]
                test_id = self.fix_test_id(test_id)
                self.test_ids[test_id] = row

    @classmethod
    def fix_test_id(cls, test_id: str) -> str:
        return re.sub(r"HiFi4\d", "HiFi4", test_id)

    def get_data(self, test_id) -> Dict[str, Any]:
        return self.test_ids.get(test_id, None)

    def emulate_run(self, test_id: str):
        test_id = self.fix_test_id(test_id)
        test_data = self.get_data(test_id)
        if test_data is None:
            raise ValueError(f"Test id {test_id} not found in report data")
        error_message = test_data["error_message_full"]
        error_log = test_data["error_log"]
        if error_message:
            raise EmulatorError(error_message, error_log)
