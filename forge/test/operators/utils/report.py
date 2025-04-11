#!./env/bin/python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# from collections import OrderedDict
from typing import Dict, List, Optional, Union, Tuple, Generator, Callable, Any, Type
from dataclasses import dataclass

# from datetime import datetime
from loguru import logger

# from IPython.display import display, HTML

import os
import re

# import psutil
# import pandas as pd

import xml.etree.ElementTree as ET


class FilesUtils:

    # Read env var DOCKER_TOOLS_DIR
    DOCKER_TOOLS_DIR = os.getenv("DOCKER_TOOLS_DIR")

    @classmethod
    def save_text(cls, file, content):
        with open(file, "w", encoding="utf-8") as f:
            f.write(content)

    @classmethod
    def log_file_path(cls, file: str) -> str:

        if not file.startswith("/"):
            file = f"{cls.DOCKER_TOOLS_DIR}/log/{file}"

        return file

    @classmethod
    def is_sweeps_file(cls, xml_file: str) -> bool:
        return not "random_test_graphs" in xml_file

    @classmethod
    def log_to_xml_file(cls, log_file: str) -> str:
        return log_file.replace(".log", "_format.xml")


class ReportUtils:

    ERROR_MESSAGE_LIMIT = 200

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
    def remove_colors(cls, text: str) -> str:
        # Remove colors from text
        text = re.sub(r"#x1B\[\d+m", "", text)
        text = re.sub(r"#x1B\[\d+;\d+;\d+m", "", text)
        text = re.sub(r"#x1B\[\d+;\d+;\d+;\d+;\d+m", "", text)

        text = re.sub(r"\[\d+m", "", text)
        text = re.sub(r"\[\d+;\d+;\d+m", "", text)
        text = re.sub(r"\[\d+;\d+;\d+;\d+;\d+m", "", text)

        text = re.sub(r"\[1A", "", text)
        text = re.sub(r"\[1B", "", text)
        text = re.sub(r"\[2K", "", text)

        return text

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
            error_message = error_message.split("\n")[0] if error_message is not None else None
            # Limit number of characters to 200
            error_message = error_message[: cls.ERROR_MESSAGE_LIMIT] if error_message is not None else None
            row["error_message"] = error_message

        return data


@dataclass
class ExceptionData:
    # operator: str
    class_name: str
    message: str


# class FailingReasonsValidation:

#     @classmethod
#     def get_exception_data(cls, error_message: str) -> ExceptionData:
#         error_message_short = error_message.split("\n")[0]
#         class_name = error_message_short[:50].split(":")[0]
#         # error_message = " ".join(error_message.split("\n")[:1])
#         error_message = error_message[len(class_name)+2:]
#         # logger.info(class_name)
#         # logger.info(f"Line: {class_name} | {error_message}")
#         ex = ExceptionData(class_name, error_message)
#         return ex


class EmulatorError(Exception):

    pass


# class RuntimeError(Exception):

#     pass


# class AssertionError(Exception):

#     pass


# class ValueError(Exception):

#     pass


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
                self.test_ids[row["id"]] = row

    def get_data(self, test_id) -> Dict[str, Any]:
        return self.test_ids.get(test_id, None)

    # @classmethod
    # def get_class(cls, class_name) -> Optional[Type[Exception]]:
    #     if class_name == "RuntimeError":
    #         return RuntimeError
    #     if class_name == "AssertionError":
    #         return AssertionError
    #     if class_name == "ValueError":
    #         return ValueError
    #     return None

    def emulate_run(self, test_id: str):
        test_data = self.get_data(test_id)
        error_message = test_data["error_message_full"]
        if error_message:
            # ex = FailingReasonsValidation.get_exception_data(error_message)
            raise EmulatorError(error_message)
            # class_type = self.get_class(ex.class_name)
            # if class_type is not None:
            #     raise class_type(error_message)
            # else:
            #     logger.warning(f"Unknown exception class: {ex.class_name}")
