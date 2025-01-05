# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import re
from enum import Enum


class StrEnum(str, Enum):
    def __str__(self):
        return self.value


class Framework(StrEnum):
    PYTORCH = "pt"
    TENSORFLOW = "tf"
    ONNX = "onnx"


def build_module_name(
    framework: Framework,
    model: str,
    variant: str | None = None,
    task: str | None = None,
    source: str | None = None,
    suffix: str | None = None,
) -> str:
    module_name = f"{framework}_{model}"
    if variant is not None:
        module_name += f"_{variant}"
    if task is not None:
        module_name += f"_{task}"
    if source is not None:
        module_name += f"_{source}"
    if suffix is not None:
        module_name += f"_{suffix}"

    module_name = re.sub(r"[^a-zA-Z0-9_]", "_", module_name)
    module_name = re.sub(r"_+", "_", module_name)
    return module_name
