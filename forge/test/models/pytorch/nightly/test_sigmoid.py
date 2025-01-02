# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import forge
import forge.op
from forge import ForgeModule

from loguru import logger
import torch

from forge import Tensor, compile
from forge.verify.compare import compare_with_golden
from forge.verify.verify import verify
from forge.verify.config import VerifyConfig
import pytest


class Sigmoid0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, sigmoid_input_0):
        sigmoid_output_1 = forge.op.Sigmoid("", sigmoid_input_0)
        return sigmoid_output_1


def ids_func(param):
    forge_module, shapes_dtypes, _ = param
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (Sigmoid0, [((2, 7, 2048), torch.float32)], {"model_name": ["pt_clip_vit_base_patch32_text"]}),
    (
        Sigmoid0,
        [((1, 256, 14336), torch.float32)],
        {
            "model_name": [
                "pt_Llama_3_1_8B_Instruct_causal_lm",
                "pt_Meta_Llama_3_8B_causal_lm",
                "pt_Meta_Llama_3_8B_Instruct_causal_lm",
                "pt_Llama_3_1_8B_causal_lm",
            ]
        },
    ),
    (
        Sigmoid0,
        [((1, 256, 8192), torch.float32)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_causal_lm", "pt_Llama_3_2_1B_causal_lm"]},
    ),
    (
        Sigmoid0,
        [((1, 4, 14336), torch.float32)],
        {
            "model_name": [
                "pt_Llama_3_1_8B_Instruct_seq_cls",
                "pt_Meta_Llama_3_8B_seq_cls",
                "pt_Llama_3_1_8B_seq_cls",
                "pt_Meta_Llama_3_8B_Instruct_seq_cls",
            ]
        },
    ),
    (
        Sigmoid0,
        [((1, 4, 8192), torch.float32)],
        {"model_name": ["pt_Llama_3_2_1B_Instruct_seq_cls", "pt_Llama_3_2_1B_seq_cls"]},
    ),
    (Sigmoid0, [((1, 128, 14336), torch.float32)], {"model_name": ["pt_Mistral_7B_v0_1"]}),
    (Sigmoid0, [((1, 29, 2816), torch.float32)], {"model_name": ["pt_qwen_chat"]}),
    (Sigmoid0, [((1, 6, 2816), torch.float32)], {"model_name": ["pt_qwen_causal_lm"]}),
    (
        Sigmoid0,
        [((1, 35, 8960), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_1_5B_Instruct", "pt_Qwen_Qwen2_5_Coder_1_5B"]},
    ),
    (
        Sigmoid0,
        [((1, 35, 11008), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_3B", "pt_Qwen_Qwen2_5_Coder_3B_Instruct"]},
    ),
    (
        Sigmoid0,
        [((1, 35, 18944), torch.float32)],
        {"model_name": ["pt_Qwen_Qwen2_5_Coder_7B", "pt_Qwen_Qwen2_5_Coder_7B_Instruct"]},
    ),
    (Sigmoid0, [((1, 35, 4864), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_Coder_0_5B"]}),
    (Sigmoid0, [((1, 39, 8960), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B_Instruct"]}),
    (Sigmoid0, [((1, 29, 8960), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_1_5B"]}),
    (Sigmoid0, [((1, 29, 18944), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B"]}),
    (Sigmoid0, [((1, 39, 4864), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B_Instruct"]}),
    (Sigmoid0, [((1, 29, 4864), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_0_5B"]}),
    (Sigmoid0, [((1, 39, 11008), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_3B_Instruct"]}),
    (Sigmoid0, [((1, 39, 18944), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_7B_Instruct"]}),
    (Sigmoid0, [((1, 29, 11008), torch.float32)], {"model_name": ["pt_Qwen_Qwen2_5_3B"]}),
    (
        Sigmoid0,
        [((1, 48, 160, 160), torch.float32)],
        {"model_name": ["pt_efficientnet_b4_timm", "pt_yolov5m_320x320", "pt_yolov5m_640x640", "pt_yolox_m"]},
    ),
    (
        Sigmoid0,
        [((1, 12, 1, 1), torch.float32)],
        {"model_name": ["pt_efficientnet_b4_timm", "pt_efficientnet_b4_torchvision"]},
    ),
    (
        Sigmoid0,
        [((1, 48, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_b4_timm",
                "pt_efficientnet_b4_torchvision",
                "pt_efficientnet_b0_torchvision",
                "pt_efficientnet_b0_timm",
            ]
        },
    ),
    (Sigmoid0, [((1, 24, 160, 160), torch.float32)], {"model_name": ["pt_efficientnet_b4_timm"]}),
    (
        Sigmoid0,
        [((1, 6, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_b4_timm",
                "pt_efficientnet_b4_torchvision",
                "pt_efficientnet_b0_torchvision",
                "pt_efficientnet_b0_timm",
            ]
        },
    ),
    (
        Sigmoid0,
        [((1, 24, 1, 1), torch.float32)],
        {"model_name": ["pt_efficientnet_b4_timm", "pt_efficientnet_b4_torchvision"]},
    ),
    (Sigmoid0, [((1, 144, 160, 160), torch.float32)], {"model_name": ["pt_efficientnet_b4_timm"]}),
    (Sigmoid0, [((1, 144, 80, 80), torch.float32)], {"model_name": ["pt_efficientnet_b4_timm"]}),
    (
        Sigmoid0,
        [((1, 144, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_b4_timm",
                "pt_efficientnet_b4_torchvision",
                "pt_efficientnet_b0_torchvision",
                "pt_efficientnet_b0_timm",
            ]
        },
    ),
    (
        Sigmoid0,
        [((1, 192, 80, 80), torch.float32)],
        {"model_name": ["pt_efficientnet_b4_timm", "pt_yolov5m_640x640", "pt_yolox_m"]},
    ),
    (
        Sigmoid0,
        [((1, 8, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_b4_timm",
                "pt_efficientnet_b4_torchvision",
                "pt_efficientnet_b0_torchvision",
                "pt_efficientnet_b0_timm",
            ]
        },
    ),
    (
        Sigmoid0,
        [((1, 192, 1, 1), torch.float32)],
        {"model_name": ["pt_efficientnet_b4_timm", "pt_efficientnet_b4_torchvision", "pt_regnet_y_040"]},
    ),
    (
        Sigmoid0,
        [((1, 192, 40, 40), torch.float32)],
        {"model_name": ["pt_efficientnet_b4_timm", "pt_yolov5m_320x320", "pt_yolov5m_640x640", "pt_yolox_m"]},
    ),
    (Sigmoid0, [((1, 336, 40, 40), torch.float32)], {"model_name": ["pt_efficientnet_b4_timm"]}),
    (
        Sigmoid0,
        [((1, 14, 1, 1), torch.float32)],
        {"model_name": ["pt_efficientnet_b4_timm", "pt_efficientnet_b4_torchvision"]},
    ),
    (
        Sigmoid0,
        [((1, 336, 1, 1), torch.float32)],
        {"model_name": ["pt_efficientnet_b4_timm", "pt_efficientnet_b4_torchvision"]},
    ),
    (Sigmoid0, [((1, 336, 20, 20), torch.float32)], {"model_name": ["pt_efficientnet_b4_timm"]}),
    (Sigmoid0, [((1, 672, 20, 20), torch.float32)], {"model_name": ["pt_efficientnet_b4_timm"]}),
    (
        Sigmoid0,
        [((1, 28, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_b4_timm",
                "pt_efficientnet_b4_torchvision",
                "pt_efficientnet_b0_torchvision",
                "pt_efficientnet_b0_timm",
            ]
        },
    ),
    (
        Sigmoid0,
        [((1, 672, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_b4_timm",
                "pt_efficientnet_b4_torchvision",
                "pt_efficientnet_b0_torchvision",
                "pt_efficientnet_b0_timm",
            ]
        },
    ),
    (Sigmoid0, [((1, 960, 20, 20), torch.float32)], {"model_name": ["pt_efficientnet_b4_timm"]}),
    (
        Sigmoid0,
        [((1, 40, 1, 1), torch.float32)],
        {"model_name": ["pt_efficientnet_b4_timm", "pt_efficientnet_b4_torchvision"]},
    ),
    (
        Sigmoid0,
        [((1, 960, 1, 1), torch.float32)],
        {"model_name": ["pt_efficientnet_b4_timm", "pt_efficientnet_b4_torchvision"]},
    ),
    (Sigmoid0, [((1, 960, 10, 10), torch.float32)], {"model_name": ["pt_efficientnet_b4_timm"]}),
    (Sigmoid0, [((1, 1632, 10, 10), torch.float32)], {"model_name": ["pt_efficientnet_b4_timm"]}),
    (
        Sigmoid0,
        [((1, 68, 1, 1), torch.float32)],
        {"model_name": ["pt_efficientnet_b4_timm", "pt_efficientnet_b4_torchvision"]},
    ),
    (
        Sigmoid0,
        [((1, 1632, 1, 1), torch.float32)],
        {"model_name": ["pt_efficientnet_b4_timm", "pt_efficientnet_b4_torchvision"]},
    ),
    (Sigmoid0, [((1, 2688, 10, 10), torch.float32)], {"model_name": ["pt_efficientnet_b4_timm"]}),
    (
        Sigmoid0,
        [((1, 112, 1, 1), torch.float32)],
        {"model_name": ["pt_efficientnet_b4_timm", "pt_efficientnet_b4_torchvision"]},
    ),
    (
        Sigmoid0,
        [((1, 2688, 1, 1), torch.float32)],
        {"model_name": ["pt_efficientnet_b4_timm", "pt_efficientnet_b4_torchvision"]},
    ),
    (Sigmoid0, [((1, 1792, 10, 10), torch.float32)], {"model_name": ["pt_efficientnet_b4_timm"]}),
    (Sigmoid0, [((1, 48, 112, 112), torch.float32)], {"model_name": ["pt_efficientnet_b4_torchvision"]}),
    (Sigmoid0, [((1, 24, 112, 112), torch.float32)], {"model_name": ["pt_efficientnet_b4_torchvision"]}),
    (Sigmoid0, [((1, 144, 112, 112), torch.float32)], {"model_name": ["pt_efficientnet_b4_torchvision"]}),
    (
        Sigmoid0,
        [((1, 144, 56, 56), torch.float32)],
        {"model_name": ["pt_efficientnet_b4_torchvision", "pt_efficientnet_b0_torchvision", "pt_efficientnet_b0_timm"]},
    ),
    (Sigmoid0, [((1, 192, 56, 56), torch.float32)], {"model_name": ["pt_efficientnet_b4_torchvision"]}),
    (Sigmoid0, [((1, 192, 28, 28), torch.float32)], {"model_name": ["pt_efficientnet_b4_torchvision"]}),
    (Sigmoid0, [((1, 336, 28, 28), torch.float32)], {"model_name": ["pt_efficientnet_b4_torchvision"]}),
    (Sigmoid0, [((1, 336, 14, 14), torch.float32)], {"model_name": ["pt_efficientnet_b4_torchvision"]}),
    (
        Sigmoid0,
        [((1, 672, 14, 14), torch.float32)],
        {"model_name": ["pt_efficientnet_b4_torchvision", "pt_efficientnet_b0_torchvision", "pt_efficientnet_b0_timm"]},
    ),
    (Sigmoid0, [((1, 960, 14, 14), torch.float32)], {"model_name": ["pt_efficientnet_b4_torchvision"]}),
    (Sigmoid0, [((1, 960, 7, 7), torch.float32)], {"model_name": ["pt_efficientnet_b4_torchvision"]}),
    (Sigmoid0, [((1, 1632, 7, 7), torch.float32)], {"model_name": ["pt_efficientnet_b4_torchvision"]}),
    (Sigmoid0, [((1, 2688, 7, 7), torch.float32)], {"model_name": ["pt_efficientnet_b4_torchvision"]}),
    (Sigmoid0, [((1, 1792, 7, 7), torch.float32)], {"model_name": ["pt_efficientnet_b4_torchvision"]}),
    (
        Sigmoid0,
        [((1, 32, 112, 112), torch.float32)],
        {"model_name": ["pt_efficientnet_b0_torchvision", "pt_efficientnet_b0_timm"]},
    ),
    (
        Sigmoid0,
        [((1, 32, 1, 1), torch.float32)],
        {"model_name": ["pt_efficientnet_b0_torchvision", "pt_efficientnet_b0_timm"]},
    ),
    (
        Sigmoid0,
        [((1, 96, 112, 112), torch.float32)],
        {"model_name": ["pt_efficientnet_b0_torchvision", "pt_efficientnet_b0_timm"]},
    ),
    (
        Sigmoid0,
        [((1, 96, 56, 56), torch.float32)],
        {"model_name": ["pt_efficientnet_b0_torchvision", "pt_efficientnet_b0_timm"]},
    ),
    (
        Sigmoid0,
        [((1, 4, 1, 1), torch.float32)],
        {"model_name": ["pt_efficientnet_b0_torchvision", "pt_efficientnet_b0_timm"]},
    ),
    (
        Sigmoid0,
        [((1, 96, 1, 1), torch.float32)],
        {"model_name": ["pt_efficientnet_b0_torchvision", "pt_efficientnet_b0_timm"]},
    ),
    (
        Sigmoid0,
        [((1, 144, 28, 28), torch.float32)],
        {"model_name": ["pt_efficientnet_b0_torchvision", "pt_efficientnet_b0_timm"]},
    ),
    (
        Sigmoid0,
        [((1, 240, 28, 28), torch.float32)],
        {"model_name": ["pt_efficientnet_b0_torchvision", "pt_efficientnet_b0_timm"]},
    ),
    (
        Sigmoid0,
        [((1, 10, 1, 1), torch.float32)],
        {"model_name": ["pt_efficientnet_b0_torchvision", "pt_efficientnet_b0_timm"]},
    ),
    (
        Sigmoid0,
        [((1, 240, 1, 1), torch.float32)],
        {"model_name": ["pt_efficientnet_b0_torchvision", "pt_efficientnet_b0_timm"]},
    ),
    (
        Sigmoid0,
        [((1, 240, 14, 14), torch.float32)],
        {"model_name": ["pt_efficientnet_b0_torchvision", "pt_efficientnet_b0_timm"]},
    ),
    (
        Sigmoid0,
        [((1, 480, 14, 14), torch.float32)],
        {"model_name": ["pt_efficientnet_b0_torchvision", "pt_efficientnet_b0_timm"]},
    ),
    (
        Sigmoid0,
        [((1, 20, 1, 1), torch.float32)],
        {"model_name": ["pt_efficientnet_b0_torchvision", "pt_efficientnet_b0_timm"]},
    ),
    (
        Sigmoid0,
        [((1, 480, 1, 1), torch.float32)],
        {"model_name": ["pt_efficientnet_b0_torchvision", "pt_efficientnet_b0_timm"]},
    ),
    (
        Sigmoid0,
        [((1, 672, 7, 7), torch.float32)],
        {"model_name": ["pt_efficientnet_b0_torchvision", "pt_efficientnet_b0_timm"]},
    ),
    (
        Sigmoid0,
        [((1, 1152, 7, 7), torch.float32)],
        {"model_name": ["pt_efficientnet_b0_torchvision", "pt_efficientnet_b0_timm"]},
    ),
    (
        Sigmoid0,
        [((1, 1152, 1, 1), torch.float32)],
        {"model_name": ["pt_efficientnet_b0_torchvision", "pt_efficientnet_b0_timm"]},
    ),
    (
        Sigmoid0,
        [((1, 1280, 7, 7), torch.float32)],
        {"model_name": ["pt_efficientnet_b0_torchvision", "pt_efficientnet_b0_timm"]},
    ),
    (Sigmoid0, [((1, 128, 1, 1), torch.float32)], {"model_name": ["pt_regnet_y_040"]}),
    (Sigmoid0, [((1, 512, 1, 1), torch.float32)], {"model_name": ["pt_regnet_y_040"]}),
    (Sigmoid0, [((1, 1088, 1, 1), torch.float32)], {"model_name": ["pt_regnet_y_040"]}),
    (
        Sigmoid0,
        [((1, 720, 60, 80), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_rn18fpn",
                "pt_retinanet_rn152fpn",
                "pt_retinanet_rn101fpn",
                "pt_retinanet_rn50fpn",
                "pt_retinanet_rn34fpn",
            ]
        },
    ),
    (
        Sigmoid0,
        [((1, 720, 30, 40), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_rn18fpn",
                "pt_retinanet_rn152fpn",
                "pt_retinanet_rn101fpn",
                "pt_retinanet_rn50fpn",
                "pt_retinanet_rn34fpn",
            ]
        },
    ),
    (
        Sigmoid0,
        [((1, 720, 15, 20), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_rn18fpn",
                "pt_retinanet_rn152fpn",
                "pt_retinanet_rn101fpn",
                "pt_retinanet_rn50fpn",
                "pt_retinanet_rn34fpn",
            ]
        },
    ),
    (
        Sigmoid0,
        [((1, 720, 8, 10), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_rn18fpn",
                "pt_retinanet_rn152fpn",
                "pt_retinanet_rn101fpn",
                "pt_retinanet_rn50fpn",
                "pt_retinanet_rn34fpn",
            ]
        },
    ),
    (
        Sigmoid0,
        [((1, 720, 4, 5), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_rn18fpn",
                "pt_retinanet_rn152fpn",
                "pt_retinanet_rn101fpn",
                "pt_retinanet_rn50fpn",
                "pt_retinanet_rn34fpn",
            ]
        },
    ),
    (Sigmoid0, [((3, 64, 64), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Sigmoid0, [((6, 64, 64), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Sigmoid0, [((12, 64, 64), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Sigmoid0, [((24, 64, 64), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Sigmoid0, [((1, 1, 256, 256), torch.float32)], {"model_name": ["pt_unet_torchhub"]}),
    (Sigmoid0, [((1, 80, 320, 320), torch.float32)], {"model_name": ["pt_yolov5x_640x640", "pt_yolox_x"]}),
    (Sigmoid0, [((1, 160, 160, 160), torch.float32)], {"model_name": ["pt_yolov5x_640x640", "pt_yolox_x"]}),
    (
        Sigmoid0,
        [((1, 80, 160, 160), torch.float32)],
        {"model_name": ["pt_yolov5x_640x640", "pt_yolov5x_320x320", "pt_yolox_x"]},
    ),
    (Sigmoid0, [((1, 320, 80, 80), torch.float32)], {"model_name": ["pt_yolov5x_640x640", "pt_yolox_x"]}),
    (
        Sigmoid0,
        [((1, 160, 80, 80), torch.float32)],
        {"model_name": ["pt_yolov5x_640x640", "pt_yolov5x_320x320", "pt_yolox_x"]},
    ),
    (Sigmoid0, [((1, 640, 40, 40), torch.float32)], {"model_name": ["pt_yolov5x_640x640", "pt_yolox_x"]}),
    (
        Sigmoid0,
        [((1, 320, 40, 40), torch.float32)],
        {"model_name": ["pt_yolov5x_640x640", "pt_yolov5x_320x320", "pt_yolox_x"]},
    ),
    (Sigmoid0, [((1, 1280, 20, 20), torch.float32)], {"model_name": ["pt_yolov5x_640x640", "pt_yolox_x"]}),
    (
        Sigmoid0,
        [((1, 640, 20, 20), torch.float32)],
        {"model_name": ["pt_yolov5x_640x640", "pt_yolov5x_320x320", "pt_yolox_x"]},
    ),
    (
        Sigmoid0,
        [((1, 255, 80, 80), torch.float32)],
        {
            "model_name": [
                "pt_yolov5x_640x640",
                "pt_yolov5s_1280x1280",
                "pt_yolov5s_640x640",
                "pt_yolov5m_640x640",
                "pt_yolov5l_640x640",
                "pt_yolov5n_640x640",
            ]
        },
    ),
    (
        Sigmoid0,
        [((1, 255, 40, 40), torch.float32)],
        {
            "model_name": [
                "pt_yolov5x_640x640",
                "pt_yolov5l_320x320",
                "pt_yolov5s_1280x1280",
                "pt_yolov5m_320x320",
                "pt_yolov5s_640x640",
                "pt_yolov5x_320x320",
                "pt_yolov5s_320x320",
                "pt_yolov5m_640x640",
                "pt_yolov5l_640x640",
                "pt_yolov5n_640x640",
                "pt_yolov5n_320x320",
            ]
        },
    ),
    (
        Sigmoid0,
        [((1, 255, 20, 20), torch.float32)],
        {
            "model_name": [
                "pt_yolov5x_640x640",
                "pt_yolov5l_320x320",
                "pt_yolov5m_320x320",
                "pt_yolov5s_640x640",
                "pt_yolov5x_320x320",
                "pt_yolov5s_320x320",
                "pt_yolov5m_640x640",
                "pt_yolov5l_640x640",
                "pt_yolov5n_640x640",
                "pt_yolov5n_320x320",
            ]
        },
    ),
    (
        Sigmoid0,
        [((1, 64, 160, 160), torch.float32)],
        {
            "model_name": [
                "pt_yolov5l_320x320",
                "pt_yolov5s_1280x1280",
                "pt_yolov5s_640x640",
                "pt_yolov5l_640x640",
                "pt_yolox_s",
                "pt_yolox_l",
            ]
        },
    ),
    (
        Sigmoid0,
        [((1, 128, 80, 80), torch.float32)],
        {
            "model_name": [
                "pt_yolov5l_320x320",
                "pt_yolov5s_1280x1280",
                "pt_yolov5s_640x640",
                "pt_yolov5l_640x640",
                "pt_yolox_s",
                "pt_yolox_l",
            ]
        },
    ),
    (
        Sigmoid0,
        [((1, 64, 80, 80), torch.float32)],
        {
            "model_name": [
                "pt_yolov5l_320x320",
                "pt_yolov5s_640x640",
                "pt_yolov5s_320x320",
                "pt_yolov5n_640x640",
                "pt_yolox_s",
            ]
        },
    ),
    (
        Sigmoid0,
        [((1, 256, 40, 40), torch.float32)],
        {
            "model_name": [
                "pt_yolov5l_320x320",
                "pt_yolov5s_1280x1280",
                "pt_yolov5s_640x640",
                "pt_yolov5l_640x640",
                "pt_yolox_s",
                "pt_yolox_l",
            ]
        },
    ),
    (
        Sigmoid0,
        [((1, 128, 40, 40), torch.float32)],
        {
            "model_name": [
                "pt_yolov5l_320x320",
                "pt_yolov5s_640x640",
                "pt_yolov5s_320x320",
                "pt_yolov5n_640x640",
                "pt_yolox_s",
            ]
        },
    ),
    (
        Sigmoid0,
        [((1, 512, 20, 20), torch.float32)],
        {"model_name": ["pt_yolov5l_320x320", "pt_yolov5s_640x640", "pt_yolov5l_640x640", "pt_yolox_s", "pt_yolox_l"]},
    ),
    (
        Sigmoid0,
        [((1, 256, 20, 20), torch.float32)],
        {
            "model_name": [
                "pt_yolov5l_320x320",
                "pt_yolov5s_640x640",
                "pt_yolov5s_320x320",
                "pt_yolov5n_640x640",
                "pt_yolox_s",
                "pt_yolox_l",
            ]
        },
    ),
    (Sigmoid0, [((1, 1024, 10, 10), torch.float32)], {"model_name": ["pt_yolov5l_320x320"]}),
    (Sigmoid0, [((1, 512, 10, 10), torch.float32)], {"model_name": ["pt_yolov5l_320x320", "pt_yolov5s_320x320"]}),
    (
        Sigmoid0,
        [((1, 255, 10, 10), torch.float32)],
        {
            "model_name": [
                "pt_yolov5l_320x320",
                "pt_yolov5m_320x320",
                "pt_yolov5x_320x320",
                "pt_yolov5s_320x320",
                "pt_yolov5n_320x320",
            ]
        },
    ),
    (Sigmoid0, [((1, 32, 640, 640), torch.float32)], {"model_name": ["pt_yolov5s_1280x1280"]}),
    (
        Sigmoid0,
        [((1, 64, 320, 320), torch.float32)],
        {"model_name": ["pt_yolov5s_1280x1280", "pt_yolov5l_640x640", "pt_yolox_l"]},
    ),
    (
        Sigmoid0,
        [((1, 32, 320, 320), torch.float32)],
        {"model_name": ["pt_yolov5s_1280x1280", "pt_yolov5s_640x640", "pt_yolox_s"]},
    ),
    (
        Sigmoid0,
        [((1, 128, 160, 160), torch.float32)],
        {"model_name": ["pt_yolov5s_1280x1280", "pt_yolov5l_640x640", "pt_yolox_l"]},
    ),
    (
        Sigmoid0,
        [((1, 256, 80, 80), torch.float32)],
        {"model_name": ["pt_yolov5s_1280x1280", "pt_yolov5l_640x640", "pt_yolox_l"]},
    ),
    (
        Sigmoid0,
        [((1, 512, 40, 40), torch.float32)],
        {"model_name": ["pt_yolov5s_1280x1280", "pt_yolov5l_640x640", "pt_yolox_l"]},
    ),
    (Sigmoid0, [((1, 255, 160, 160), torch.float32)], {"model_name": ["pt_yolov5s_1280x1280"]}),
    (
        Sigmoid0,
        [((1, 96, 80, 80), torch.float32)],
        {"model_name": ["pt_yolov5m_320x320", "pt_yolov5m_640x640", "pt_yolox_m"]},
    ),
    (Sigmoid0, [((1, 48, 80, 80), torch.float32)], {"model_name": ["pt_yolov5m_320x320"]}),
    (Sigmoid0, [((1, 96, 40, 40), torch.float32)], {"model_name": ["pt_yolov5m_320x320"]}),
    (
        Sigmoid0,
        [((1, 384, 20, 20), torch.float32)],
        {"model_name": ["pt_yolov5m_320x320", "pt_yolov5m_640x640", "pt_yolox_m"]},
    ),
    (Sigmoid0, [((1, 192, 20, 20), torch.float32)], {"model_name": ["pt_yolov5m_320x320", "pt_yolox_m"]}),
    (Sigmoid0, [((1, 768, 10, 10), torch.float32)], {"model_name": ["pt_yolov5m_320x320"]}),
    (Sigmoid0, [((1, 384, 10, 10), torch.float32)], {"model_name": ["pt_yolov5m_320x320"]}),
    (
        Sigmoid0,
        [((1, 32, 160, 160), torch.float32)],
        {"model_name": ["pt_yolov5s_640x640", "pt_yolov5s_320x320", "pt_yolov5n_640x640", "pt_yolox_s"]},
    ),
    (
        Sigmoid0,
        [((1, 80, 80, 80), torch.float32)],
        {
            "model_name": [
                "pt_yolov5x_320x320",
                "pt_yolox_m",
                "pt_yolox_s",
                "pt_yolox_darknet",
                "pt_yolox_x",
                "pt_yolox_l",
            ]
        },
    ),
    (Sigmoid0, [((1, 160, 40, 40), torch.float32)], {"model_name": ["pt_yolov5x_320x320"]}),
    (Sigmoid0, [((1, 320, 20, 20), torch.float32)], {"model_name": ["pt_yolov5x_320x320", "pt_yolox_x"]}),
    (Sigmoid0, [((1, 1280, 10, 10), torch.float32)], {"model_name": ["pt_yolov5x_320x320"]}),
    (Sigmoid0, [((1, 640, 10, 10), torch.float32)], {"model_name": ["pt_yolov5x_320x320"]}),
    (Sigmoid0, [((1, 48, 240, 240), torch.float32)], {"model_name": ["pt_yolov5m_480x480"]}),
    (Sigmoid0, [((1, 96, 120, 120), torch.float32)], {"model_name": ["pt_yolov5m_480x480"]}),
    (Sigmoid0, [((1, 48, 120, 120), torch.float32)], {"model_name": ["pt_yolov5m_480x480"]}),
    (Sigmoid0, [((1, 192, 60, 60), torch.float32)], {"model_name": ["pt_yolov5m_480x480"]}),
    (Sigmoid0, [((1, 96, 60, 60), torch.float32)], {"model_name": ["pt_yolov5m_480x480"]}),
    (Sigmoid0, [((1, 384, 30, 30), torch.float32)], {"model_name": ["pt_yolov5m_480x480"]}),
    (Sigmoid0, [((1, 192, 30, 30), torch.float32)], {"model_name": ["pt_yolov5m_480x480"]}),
    (Sigmoid0, [((1, 768, 15, 15), torch.float32)], {"model_name": ["pt_yolov5m_480x480"]}),
    (Sigmoid0, [((1, 384, 15, 15), torch.float32)], {"model_name": ["pt_yolov5m_480x480"]}),
    (
        Sigmoid0,
        [((1, 255, 60, 60), torch.float32)],
        {
            "model_name": [
                "pt_yolov5m_480x480",
                "pt_yolov5n_480x480",
                "pt_yolov5l_480x480",
                "pt_yolov5x_480x480",
                "pt_yolov5s_480x480",
            ]
        },
    ),
    (
        Sigmoid0,
        [((1, 255, 30, 30), torch.float32)],
        {
            "model_name": [
                "pt_yolov5m_480x480",
                "pt_yolov5n_480x480",
                "pt_yolov5l_480x480",
                "pt_yolov5x_480x480",
                "pt_yolov5s_480x480",
            ]
        },
    ),
    (
        Sigmoid0,
        [((1, 255, 15, 15), torch.float32)],
        {
            "model_name": [
                "pt_yolov5m_480x480",
                "pt_yolov5n_480x480",
                "pt_yolov5l_480x480",
                "pt_yolov5x_480x480",
                "pt_yolov5s_480x480",
            ]
        },
    ),
    (
        Sigmoid0,
        [((1, 32, 80, 80), torch.float32)],
        {"model_name": ["pt_yolov5s_320x320", "pt_yolov5n_640x640", "pt_yolov5n_320x320"]},
    ),
    (
        Sigmoid0,
        [((1, 64, 40, 40), torch.float32)],
        {"model_name": ["pt_yolov5s_320x320", "pt_yolov5n_640x640", "pt_yolov5n_320x320"]},
    ),
    (
        Sigmoid0,
        [((1, 128, 20, 20), torch.float32)],
        {"model_name": ["pt_yolov5s_320x320", "pt_yolov5n_640x640", "pt_yolov5n_320x320", "pt_yolox_s"]},
    ),
    (Sigmoid0, [((1, 256, 10, 10), torch.float32)], {"model_name": ["pt_yolov5s_320x320", "pt_yolov5n_320x320"]}),
    (Sigmoid0, [((1, 16, 240, 240), torch.float32)], {"model_name": ["pt_yolov5n_480x480"]}),
    (Sigmoid0, [((1, 32, 120, 120), torch.float32)], {"model_name": ["pt_yolov5n_480x480", "pt_yolov5s_480x480"]}),
    (Sigmoid0, [((1, 16, 120, 120), torch.float32)], {"model_name": ["pt_yolov5n_480x480"]}),
    (Sigmoid0, [((1, 64, 60, 60), torch.float32)], {"model_name": ["pt_yolov5n_480x480", "pt_yolov5s_480x480"]}),
    (Sigmoid0, [((1, 32, 60, 60), torch.float32)], {"model_name": ["pt_yolov5n_480x480"]}),
    (Sigmoid0, [((1, 128, 30, 30), torch.float32)], {"model_name": ["pt_yolov5n_480x480", "pt_yolov5s_480x480"]}),
    (Sigmoid0, [((1, 64, 30, 30), torch.float32)], {"model_name": ["pt_yolov5n_480x480"]}),
    (Sigmoid0, [((1, 256, 15, 15), torch.float32)], {"model_name": ["pt_yolov5n_480x480", "pt_yolov5s_480x480"]}),
    (Sigmoid0, [((1, 128, 15, 15), torch.float32)], {"model_name": ["pt_yolov5n_480x480"]}),
    (Sigmoid0, [((1, 48, 320, 320), torch.float32)], {"model_name": ["pt_yolov5m_640x640", "pt_yolox_m"]}),
    (Sigmoid0, [((1, 96, 160, 160), torch.float32)], {"model_name": ["pt_yolov5m_640x640", "pt_yolox_m"]}),
    (Sigmoid0, [((1, 384, 40, 40), torch.float32)], {"model_name": ["pt_yolov5m_640x640", "pt_yolox_m"]}),
    (Sigmoid0, [((1, 768, 20, 20), torch.float32)], {"model_name": ["pt_yolov5m_640x640", "pt_yolox_m"]}),
    (Sigmoid0, [((1, 64, 240, 240), torch.float32)], {"model_name": ["pt_yolov5l_480x480"]}),
    (Sigmoid0, [((1, 128, 120, 120), torch.float32)], {"model_name": ["pt_yolov5l_480x480"]}),
    (Sigmoid0, [((1, 64, 120, 120), torch.float32)], {"model_name": ["pt_yolov5l_480x480", "pt_yolov5s_480x480"]}),
    (Sigmoid0, [((1, 256, 60, 60), torch.float32)], {"model_name": ["pt_yolov5l_480x480"]}),
    (Sigmoid0, [((1, 128, 60, 60), torch.float32)], {"model_name": ["pt_yolov5l_480x480", "pt_yolov5s_480x480"]}),
    (Sigmoid0, [((1, 512, 30, 30), torch.float32)], {"model_name": ["pt_yolov5l_480x480"]}),
    (Sigmoid0, [((1, 256, 30, 30), torch.float32)], {"model_name": ["pt_yolov5l_480x480", "pt_yolov5s_480x480"]}),
    (Sigmoid0, [((1, 1024, 15, 15), torch.float32)], {"model_name": ["pt_yolov5l_480x480"]}),
    (Sigmoid0, [((1, 512, 15, 15), torch.float32)], {"model_name": ["pt_yolov5l_480x480", "pt_yolov5s_480x480"]}),
    (Sigmoid0, [((1, 80, 240, 240), torch.float32)], {"model_name": ["pt_yolov5x_480x480"]}),
    (Sigmoid0, [((1, 160, 120, 120), torch.float32)], {"model_name": ["pt_yolov5x_480x480"]}),
    (Sigmoid0, [((1, 80, 120, 120), torch.float32)], {"model_name": ["pt_yolov5x_480x480"]}),
    (Sigmoid0, [((1, 320, 60, 60), torch.float32)], {"model_name": ["pt_yolov5x_480x480"]}),
    (Sigmoid0, [((1, 160, 60, 60), torch.float32)], {"model_name": ["pt_yolov5x_480x480"]}),
    (Sigmoid0, [((1, 640, 30, 30), torch.float32)], {"model_name": ["pt_yolov5x_480x480"]}),
    (Sigmoid0, [((1, 320, 30, 30), torch.float32)], {"model_name": ["pt_yolov5x_480x480"]}),
    (Sigmoid0, [((1, 1280, 15, 15), torch.float32)], {"model_name": ["pt_yolov5x_480x480"]}),
    (Sigmoid0, [((1, 640, 15, 15), torch.float32)], {"model_name": ["pt_yolov5x_480x480"]}),
    (Sigmoid0, [((1, 1024, 20, 20), torch.float32)], {"model_name": ["pt_yolov5l_640x640", "pt_yolox_l"]}),
    (Sigmoid0, [((1, 16, 320, 320), torch.float32)], {"model_name": ["pt_yolov5n_640x640"]}),
    (Sigmoid0, [((1, 16, 160, 160), torch.float32)], {"model_name": ["pt_yolov5n_640x640", "pt_yolov5n_320x320"]}),
    (Sigmoid0, [((1, 32, 240, 240), torch.float32)], {"model_name": ["pt_yolov5s_480x480"]}),
    (Sigmoid0, [((1, 16, 80, 80), torch.float32)], {"model_name": ["pt_yolov5n_320x320"]}),
    (Sigmoid0, [((1, 32, 40, 40), torch.float32)], {"model_name": ["pt_yolov5n_320x320"]}),
    (Sigmoid0, [((1, 64, 20, 20), torch.float32)], {"model_name": ["pt_yolov5n_320x320"]}),
    (Sigmoid0, [((1, 128, 10, 10), torch.float32)], {"model_name": ["pt_yolov5n_320x320"]}),
    (Sigmoid0, [((1, 96, 56, 80), torch.float32)], {"model_name": ["pt_yolov6m"]}),
    (Sigmoid0, [((1, 192, 28, 40), torch.float32)], {"model_name": ["pt_yolov6m"]}),
    (Sigmoid0, [((1, 384, 14, 20), torch.float32)], {"model_name": ["pt_yolov6m"]}),
    (
        Sigmoid0,
        [((1, 80, 56, 80), torch.float32)],
        {"model_name": ["pt_yolov6m", "pt_yolov6n", "pt_yolov6l", "pt_yolov6s"]},
    ),
    (
        Sigmoid0,
        [((1, 80, 28, 40), torch.float32)],
        {"model_name": ["pt_yolov6m", "pt_yolov6n", "pt_yolov6l", "pt_yolov6s"]},
    ),
    (
        Sigmoid0,
        [((1, 80, 14, 20), torch.float32)],
        {"model_name": ["pt_yolov6m", "pt_yolov6n", "pt_yolov6l", "pt_yolov6s"]},
    ),
    (Sigmoid0, [((1, 32, 56, 80), torch.float32)], {"model_name": ["pt_yolov6n"]}),
    (Sigmoid0, [((1, 64, 28, 40), torch.float32)], {"model_name": ["pt_yolov6n"]}),
    (Sigmoid0, [((1, 128, 14, 20), torch.float32)], {"model_name": ["pt_yolov6n"]}),
    (Sigmoid0, [((1, 64, 224, 320), torch.float32)], {"model_name": ["pt_yolov6l"]}),
    (Sigmoid0, [((1, 128, 112, 160), torch.float32)], {"model_name": ["pt_yolov6l"]}),
    (Sigmoid0, [((1, 64, 112, 160), torch.float32)], {"model_name": ["pt_yolov6l"]}),
    (Sigmoid0, [((1, 256, 56, 80), torch.float32)], {"model_name": ["pt_yolov6l"]}),
    (Sigmoid0, [((1, 128, 56, 80), torch.float32)], {"model_name": ["pt_yolov6l"]}),
    (Sigmoid0, [((1, 512, 28, 40), torch.float32)], {"model_name": ["pt_yolov6l"]}),
    (Sigmoid0, [((1, 256, 28, 40), torch.float32)], {"model_name": ["pt_yolov6l"]}),
    (Sigmoid0, [((1, 1024, 14, 20), torch.float32)], {"model_name": ["pt_yolov6l"]}),
    (Sigmoid0, [((1, 512, 14, 20), torch.float32)], {"model_name": ["pt_yolov6l"]}),
    (Sigmoid0, [((1, 128, 28, 40), torch.float32)], {"model_name": ["pt_yolov6l", "pt_yolov6s"]}),
    (Sigmoid0, [((1, 64, 56, 80), torch.float32)], {"model_name": ["pt_yolov6l", "pt_yolov6s"]}),
    (Sigmoid0, [((1, 256, 14, 20), torch.float32)], {"model_name": ["pt_yolov6l", "pt_yolov6s"]}),
    (
        Sigmoid0,
        [((1, 1, 80, 80), torch.float32)],
        {"model_name": ["pt_yolox_m", "pt_yolox_s", "pt_yolox_darknet", "pt_yolox_x", "pt_yolox_l"]},
    ),
    (
        Sigmoid0,
        [((1, 1, 40, 40), torch.float32)],
        {"model_name": ["pt_yolox_m", "pt_yolox_s", "pt_yolox_darknet", "pt_yolox_x", "pt_yolox_l"]},
    ),
    (
        Sigmoid0,
        [((1, 80, 40, 40), torch.float32)],
        {"model_name": ["pt_yolox_m", "pt_yolox_s", "pt_yolox_darknet", "pt_yolox_x", "pt_yolox_l"]},
    ),
    (
        Sigmoid0,
        [((1, 1, 20, 20), torch.float32)],
        {"model_name": ["pt_yolox_m", "pt_yolox_s", "pt_yolox_darknet", "pt_yolox_x", "pt_yolox_l"]},
    ),
    (
        Sigmoid0,
        [((1, 80, 20, 20), torch.float32)],
        {"model_name": ["pt_yolox_m", "pt_yolox_s", "pt_yolox_darknet", "pt_yolox_x", "pt_yolox_l"]},
    ),
    (Sigmoid0, [((1, 16, 208, 208), torch.float32)], {"model_name": ["pt_yolox_nano"]}),
    (Sigmoid0, [((1, 16, 104, 104), torch.float32)], {"model_name": ["pt_yolox_nano"]}),
    (Sigmoid0, [((1, 32, 104, 104), torch.float32)], {"model_name": ["pt_yolox_nano"]}),
    (Sigmoid0, [((1, 32, 52, 52), torch.float32)], {"model_name": ["pt_yolox_nano"]}),
    (Sigmoid0, [((1, 64, 52, 52), torch.float32)], {"model_name": ["pt_yolox_nano"]}),
    (Sigmoid0, [((1, 64, 26, 26), torch.float32)], {"model_name": ["pt_yolox_nano"]}),
    (Sigmoid0, [((1, 128, 26, 26), torch.float32)], {"model_name": ["pt_yolox_nano"]}),
    (Sigmoid0, [((1, 128, 13, 13), torch.float32)], {"model_name": ["pt_yolox_nano"]}),
    (Sigmoid0, [((1, 256, 13, 13), torch.float32)], {"model_name": ["pt_yolox_nano"]}),
    (Sigmoid0, [((1, 1, 52, 52), torch.float32)], {"model_name": ["pt_yolox_nano", "pt_yolox_tiny"]}),
    (Sigmoid0, [((1, 80, 52, 52), torch.float32)], {"model_name": ["pt_yolox_nano", "pt_yolox_tiny"]}),
    (Sigmoid0, [((1, 1, 26, 26), torch.float32)], {"model_name": ["pt_yolox_nano", "pt_yolox_tiny"]}),
    (Sigmoid0, [((1, 80, 26, 26), torch.float32)], {"model_name": ["pt_yolox_nano", "pt_yolox_tiny"]}),
    (Sigmoid0, [((1, 64, 13, 13), torch.float32)], {"model_name": ["pt_yolox_nano"]}),
    (Sigmoid0, [((1, 1, 13, 13), torch.float32)], {"model_name": ["pt_yolox_nano", "pt_yolox_tiny"]}),
    (Sigmoid0, [((1, 80, 13, 13), torch.float32)], {"model_name": ["pt_yolox_nano", "pt_yolox_tiny"]}),
    (Sigmoid0, [((1, 24, 208, 208), torch.float32)], {"model_name": ["pt_yolox_tiny"]}),
    (Sigmoid0, [((1, 48, 104, 104), torch.float32)], {"model_name": ["pt_yolox_tiny"]}),
    (Sigmoid0, [((1, 24, 104, 104), torch.float32)], {"model_name": ["pt_yolox_tiny"]}),
    (Sigmoid0, [((1, 96, 52, 52), torch.float32)], {"model_name": ["pt_yolox_tiny"]}),
    (Sigmoid0, [((1, 48, 52, 52), torch.float32)], {"model_name": ["pt_yolox_tiny"]}),
    (Sigmoid0, [((1, 192, 26, 26), torch.float32)], {"model_name": ["pt_yolox_tiny"]}),
    (Sigmoid0, [((1, 96, 26, 26), torch.float32)], {"model_name": ["pt_yolox_tiny"]}),
    (Sigmoid0, [((1, 384, 13, 13), torch.float32)], {"model_name": ["pt_yolox_tiny"]}),
    (Sigmoid0, [((1, 192, 13, 13), torch.float32)], {"model_name": ["pt_yolox_tiny"]}),
    (Sigmoid0, [((1, 96, 13, 13), torch.float32)], {"model_name": ["pt_yolox_tiny"]}),
]


@pytest.mark.push
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes, record_property):
    record_property("frontend", "tt-forge-fe")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes

    for metadata_name, metadata_value in metadata.items():
        record_property(metadata_name, metadata_value)

    inputs = [
        Tensor.create_from_shape(operand_shape, operand_dtype) for operand_shape, operand_dtype in operand_shapes_dtypes
    ]

    framework_model = forge_module(forge_module.__name__)
    framework_model.process_framework_parameters()

    compiled_model = compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)
