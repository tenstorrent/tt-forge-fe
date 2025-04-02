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


class BaseEnum(Enum):
    """Base Enum to handle short and full name attributes."""

    def __init__(self, short: str, full: str):
        self.short = short  # Short identifier (e.g., "hf" for Hugging Face)
        self.full = full  # Full descriptive name (e.g., "Hugging Face")


class Framework(BaseEnum):
    PYTORCH = ("pt", "PyTorch")
    TENSORFLOW = ("tf", "TensorFlow")
    ONNX = ("onnx", "ONNX")
    PADDLE = ("pd", "PaddlePaddle")


class Task(BaseEnum):
    SPEECH_TRANSLATE = ("speech_translate", "Speech Translation")
    MUSIC_GENERATION = ("music_generation", "Music Generation")
    SPEECH_RECOGNITION = ("speech_recognition", "Speech Recognition")
    QA = ("qa", "Question Answering")
    MASKED_LM = ("mlm", "Masked Language Modeling")
    CAUSAL_LM = ("clm", "Causal Language Modeling")
    TOKEN_CLASSIFICATION = ("token_cls", "Token Classification")
    SEQUENCE_CLASSIFICATION = ("seq_cls", "Sequence Classification")
    IMAGE_CLASSIFICATION = ("img_cls", "Image Classification")
    IMAGE_SEGMENTATION = ("img_seg", "Image Segmentation")
    POSE_ESTIMATION = ("pose_estimation", "Pose Estimation")
    DEPTH_PREDICTION = ("depth_prediction", "Depth Prediction")
    TEXT_GENERATION = ("text_gen", "Text Generation")
    OBJECT_DETECTION = ("obj_det", "Object Detection")
    SEMANTIC_SEGMENTATION = ("sem_seg", "Semantic Segmentation")
    MASKED_IMAGE_MODELING = ("masked_img", "Masked Image Modeling")
    CONDITIONAL_GENERATION = ("cond_gen", "Conditional Generation")
    IMAGE_ENCODING = ("img_enc", "Image Encoding")
    VISUAL_BACKBONE = ("visual_bb", "Visual Backbone")
    DEPTH_ESTIMATION = ("depth_estimation", "Depth Estimation")
    SCENE_TEXT_RECOGNITION = ("scene_text_recognition", "Scene Text Recognition")
    TEXT_TO_SPEECH = ("text_to_speech", "Text to Speech")
    SENTENCE_EMBEDDING_GENERATION = ("sentence_embed_gen", "Sentence Embedding Generation")
    MULTIMODAL_TEXT_GENERATION = ("multimodal_text_gen", "Multimodal Text Generation")
    ATOMIC_ML = ("atomic_ml", "Atomic Machine Learning")


class Source(BaseEnum):
    HUGGINGFACE = ("hf", "Hugging Face")
    TORCH_HUB = ("torchhub", "Torch Hub")
    TIMM = ("timm", "TIMM")
    OSMR = ("osmr", "OSMR")
    TORCHVISION = ("torchvision", "Torchvision")
    GITHUB = ("github", "GitHub")
    PADDLE = ("paddlemodels", "Paddle Models")


def build_module_name(
    framework: Framework,
    model: str,
    task: Task,
    source: Source,
    variant: str = "base",
    suffix: str | None = None,
) -> str:
    module_name = f"{framework}_{model}_{variant}_{task}_{source}"
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
