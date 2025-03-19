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
    MUSIC_GENERATION = "music_generation"
    SPEECH_RECOGNITION = "speech_recognition"
    QA = "qa"
    MASKED_LM = "mlm"
    CAUSAL_LM = "clm"
    TOKEN_CLASSIFICATION = "token_cls"
    SEQUENCE_CLASSIFICATION = "seq_cls"
    IMAGE_CLASSIFICATION = "img_cls"
    IMAGE_SEGMENTATION = "img_seg"
    POSE_ESTIMATION = "pose_estimation"
    DEPTH_PREDICTION = "depth_prediction"
    TEXT_GENERATION = "text_gen"
    OBJECT_DETECTION = "obj_det"
    SEMANTIC_SEGMENTATION = "sem_seg"
    MASKED_IMAGE_MODELLING = "masked_img"
    CONDITIONAL_GENERATION = "cond_gen"
    IMAGE_ENCODING = "img_enc"
    VISUAL_BACKBONE = "visual_bb"
    DEPTH_ESTIMATION = "depth_estimation"
    SCENE_TEXT_RECOGNITION = "scene_text_recognition"
    TEXT_TO_SPEECH = "text_to_speech"
    SENTENCE_EMBEDDING_GENERATION = "sentence_embed_gen"
    MULTIMODAL_TEXT_GENERATION = "multimodal_text_gen"
    OPTICAL_CHARACTER_RECOGNITION = "optical_char_reg"


class Source(StrEnum):
    HUGGINGFACE = "hf"
    TORCH_HUB = "torchhub"
    TIMM = "timm"
    OSMR = "osmr"
    TORCHVISION = "torchvision"
    GITHUB = "github"


def build_module_name(
    framework: Framework,
    model: str,
    task: Task,
    source: Source,
    variant: str = "base",
    suffix: str | None = None,
) -> str:
    module_name = f"{framework}_{model}"
    if variant is not None:
        module_name += f"_{variant}"
    module_name += f"_{task}"
    module_name += f"_{source}"
    if suffix is not None:
        module_name += f"_{suffix}"

    module_name = re.sub(r"[^a-zA-Z0-9_]", "_", module_name)
    module_name = re.sub(r"_+", "_", module_name)
    module_name = module_name.lower()
    return module_name
