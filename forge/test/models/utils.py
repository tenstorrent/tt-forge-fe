# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import re
from enum import Enum
import torch
import requests
from tabulate import tabulate
import json


class StrEnum(str, Enum):
    def __str__(self):
        return self.value


class Framework(StrEnum):
    PYTORCH = "pt"
    TENSORFLOW = "tf"
    ONNX = "onnx"
    PADDLE = "pd"


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
    ATOMIC_ML = "atomic_ml"


class Source(StrEnum):
    HUGGINGFACE = "hf"
    TORCH_HUB = "torchhub"
    TIMM = "timm"
    OSMR = "osmr"
    TORCHVISION = "torchvision"
    GITHUB = "github"
    PADDLE = "paddlemodels"


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


def print_cls_results(fw_out, compiled_model_out):
    fw_top1_probabilities, fw_top1_class_indices = torch.topk(fw_out.softmax(dim=1) * 100, k=1)
    compiled_model_top1_probabilities, compiled_model_top1_class_indices = torch.topk(
        compiled_model_out.softmax(dim=1) * 100, k=1
    )

    # Directly get the top 1 predicted class and its probability for both models
    fw_top1_class_idx = fw_top1_class_indices[0, 0].item()
    compiled_model_top1_class_idx = compiled_model_top1_class_indices[0, 0].item()
    fw_top1_class_prob = fw_top1_probabilities[0, 0].item()
    compiled_model_top1_class_prob = compiled_model_top1_probabilities[0, 0].item()

    # Directly load ImageNet class labels inside post-process
    class_labels_url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    response = requests.get(class_labels_url)
    class_idx = response.json()
    class_labels = [class_idx[str(i)][1] for i in range(len(class_idx))]
    fw_top1_class_label = class_labels[fw_top1_class_idx]
    compiled_model_top1_class_label = class_labels[compiled_model_top1_class_idx]

    # Prepare the results for displaying
    table = [
        ["Metric", "Framework Model", "Compiled Model"],
        ["Top 1 Predicted Class Label", fw_top1_class_label, compiled_model_top1_class_label],
        ["Top 1 Predicted Class Probability", fw_top1_class_prob, compiled_model_top1_class_prob],
    ]
    print(tabulate(table, headers="firstrow", tablefmt="grid"))
