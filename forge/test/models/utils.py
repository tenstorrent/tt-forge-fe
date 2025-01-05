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


class Task(StrEnum):
    SPEECH_TRANSLATE = "speech_translate"
    QA = "qa"
    MASKED_LM = "mlm"
    CAUSAL_LM = "clm"
    TOKEN_CLASSIFICATION = "token_cls"
    SEQUENCE_CLASSIFICATION = "seq_cls"
    IMAGE_CLASSIFICATION = "img_cls"
    TEXT_GENERATION = "text_gen"
    OBJECT_DETECTION = "obj_det"
    SEMANTIC_SEGMENTATION = "sem_seg"
    MASKED_IMAGE_MODELLING = "masked_img"


class Source(StrEnum):
    HUGGINGFACE = "hf"
    TORCH_HUB = "torchhub"
    TIMM = "timm"
    OSMR = "osmr"
    TORCHVISION = "torchvision"


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
