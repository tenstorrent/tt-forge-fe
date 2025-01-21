# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import List


def collect_mlir_unsupported_ops(rule_tokens: List[str], error_message: str):
    unsupported_ops = []
    collect_ops_status = False
    for line in error_message.split("\n"):
        if "Unsupported Ops at: RUN_MLIR_COMPILER stage" in line:
            collect_ops_status = True
        elif "FAILED" in line or "==== FAILURES ====" in line:
            collect_ops_status = False
            break
        elif collect_ops_status and "Input_shape" not in line and "Attributes" not in line:
            unsupported_ops.append(line.strip())
    if len(unsupported_ops) != 0:
        matched_exception = " ".join(rule_tokens)
        matched_exception += " Unsupported Ops: "
        matched_exception += str(", ".join(unsupported_ops))
    return matched_exception


def collect_error_msg_from_line(rule_tokens: List[str], error_message: str):
    matched_exception = ""
    for line in error_message.split("\n"):
        if all([True if token in line else False for token in rule_tokens]):
            matched_exception += line
            break
    return matched_exception
