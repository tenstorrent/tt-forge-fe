# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import re
from enum import Enum
import torch
import requests
from PIL import Image
from torchvision import transforms


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


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def preprocess_input_data(image_url):
    input_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    input_tensor = transforms.ToTensor()(input_image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch


def build_optimum_cli_command(variant, tmp_path):
    """
    Constructs the command list for exporting a model using optimum-cli.

    Args:
        variant (str): The model variant to export.
        tmp_path (str): The temporary path where the model will be saved.

    Returns:
        list: A list of command components.
    """
    return [
        "optimum-cli",
        "export",
        "onnx",
        "--model",
        variant,
        tmp_path,
        "--opset",
        "17",
        "--monolith",
        "--framework",
        "pt",
        "--trust-remote-code",
        "--task",
        "text-generation",
        "--library-name",
        "transformers",
        "--batch_size",
        "1",
        "--legacy",
    ]
