# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import forge
import forge.op
from forge import ForgeModule

from loguru import logger
import torch

from forge import Tensor, compile
from forge.verify.verify import verify
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.config import VerifyConfig
import pytest


class Sigmoid0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, sigmoid_input_0):
        sigmoid_output_1 = forge.op.Sigmoid("", sigmoid_input_0)
        return sigmoid_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Sigmoid0,
        [((2, 7, 2048), torch.float32)],
        {"model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 39, 11008), torch.float32)],
        {
            "model_name": [
                "pt_deepseek_deepseek_math_7b_instruct_qa_hf",
                "DeepSeekWrapper_decoder",
                "pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 10, 9216), torch.float32)],
        {"model_name": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 10, 8192), torch.float32)],
        {"model_name": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 10, 23040), torch.float32)],
        {"model_name": ["pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 4, 8192), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 256, 8192), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 4, 14336), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 128, 14336), torch.float32)],
        {"model_name": ["pt_mistral_mistralai_mistral_7b_v0_1_clm_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 13, 8192), torch.float32)],
        {"model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 5, 8192), torch.float32)],
        {"model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (Sigmoid0, [((1, 6, 2816), torch.float32)], {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99}),
    (
        Sigmoid0,
        [((1, 29, 2816), torch.float32)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 35, 18944), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 35, 8960), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 35, 11008), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 35, 4864), torch.float32)],
        {"model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (Sigmoid0, [((1, 29, 8960), torch.float32)], {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99}),
    (
        Sigmoid0,
        [((1, 39, 8960), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 39, 18944), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (Sigmoid0, [((1, 29, 18944), torch.float32)], {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99}),
    (Sigmoid0, [((1, 29, 11008), torch.float32)], {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99}),
    (Sigmoid0, [((1, 29, 4864), torch.float32)], {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99}),
    (
        Sigmoid0,
        [((1, 39, 4864), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 48, 160, 160), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_640x640",
                "pt_yolox_yolox_m_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 12, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 48, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 24, 160, 160), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 6, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 24, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 144, 160, 160), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 144, 80, 80), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 144, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 192, 80, 80), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_640x640",
                "pt_yolox_yolox_m_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 8, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 192, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 192, 40, 40), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_640x640",
                "pt_yolox_yolox_m_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 336, 40, 40), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 14, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 336, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 336, 20, 20), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 672, 20, 20), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 28, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 672, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 960, 20, 20), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 40, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 960, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 960, 10, 10), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1632, 10, 10), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 68, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 1632, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 2688, 10, 10), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 112, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 2688, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 1792, 10, 10), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 32, 112, 112), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 32, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 96, 112, 112), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 96, 56, 56), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 4, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 96, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 144, 56, 56), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 144, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 240, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 10, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 240, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 240, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 480, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 20, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 480, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 672, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 672, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 1152, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 1152, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 1280, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 48, 112, 112), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 24, 112, 112), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 144, 112, 112), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 192, 56, 56), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 192, 28, 28), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 336, 28, 28), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 336, 14, 14), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 960, 14, 14), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 960, 7, 7), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1632, 7, 7), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 2688, 7, 7), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1792, 7, 7), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1, 320, 1024), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 1, 192, 640), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 128, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_facebook_regnet_y_040_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 512, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_facebook_regnet_y_040_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1088, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_facebook_regnet_y_040_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 720, 60, 80), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 720, 30, 40), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 720, 15, 20), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 720, 8, 10), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 720, 4, 5), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (Sigmoid0, [((1, 1, 256, 256), torch.float32)], {"model_name": ["pt_unet_base_img_seg_torchhub"], "pcc": 0.99}),
    (
        Sigmoid0,
        [((1, 32, 640, 640), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 64, 320, 320), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_640x640",
                "pt_yolox_yolox_l_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 32, 320, 320), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_640x640",
                "pt_yolox_yolox_s_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 128, 160, 160), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_640x640",
                "pt_yolox_yolox_l_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 64, 160, 160), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_320x320",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 256, 80, 80), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_640x640",
                "pt_yolox_yolox_l_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 128, 80, 80), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_320x320",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 512, 40, 40), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_640x640",
                "pt_yolox_yolox_l_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 256, 40, 40), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_320x320",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 255, 160, 160), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 255, 80, 80), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_640x640",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 255, 40, 40), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_640x640",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 80, 320, 320), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_640x640", "pt_yolox_yolox_x_obj_det_torchhub"],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 160, 160, 160), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_640x640", "pt_yolox_yolox_x_obj_det_torchhub"],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 80, 160, 160), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5x_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_320x320",
                "pt_yolox_yolox_x_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 320, 80, 80), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_640x640", "pt_yolox_yolox_x_obj_det_torchhub"],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 160, 80, 80), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5x_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_320x320",
                "pt_yolox_yolox_x_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 640, 40, 40), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_640x640", "pt_yolox_yolox_x_obj_det_torchhub"],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 320, 40, 40), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5x_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_320x320",
                "pt_yolox_yolox_x_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 1280, 20, 20), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_640x640", "pt_yolox_yolox_x_obj_det_torchhub"],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 640, 20, 20), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5x_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_320x320",
                "pt_yolox_yolox_x_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 255, 20, 20), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5x_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_640x640",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 80, 240, 240), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 160, 120, 120), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 80, 120, 120), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 320, 60, 60), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 160, 60, 60), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 640, 30, 30), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 320, 30, 30), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1280, 15, 15), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 640, 15, 15), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 255, 60, 60), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5x_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_480x480",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 255, 30, 30), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5x_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_480x480",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 255, 15, 15), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5x_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_480x480",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 32, 240, 240), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5s_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 64, 120, 120), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5s_imgcls_torchhub_480x480", "pt_yolo_v5_yolov5l_imgcls_torchhub_480x480"],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 32, 120, 120), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5s_imgcls_torchhub_480x480", "pt_yolo_v5_yolov5n_imgcls_torchhub_480x480"],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 128, 60, 60), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5s_imgcls_torchhub_480x480", "pt_yolo_v5_yolov5l_imgcls_torchhub_480x480"],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 64, 60, 60), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5s_imgcls_torchhub_480x480", "pt_yolo_v5_yolov5n_imgcls_torchhub_480x480"],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 256, 30, 30), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5s_imgcls_torchhub_480x480", "pt_yolo_v5_yolov5l_imgcls_torchhub_480x480"],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 128, 30, 30), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5s_imgcls_torchhub_480x480", "pt_yolo_v5_yolov5n_imgcls_torchhub_480x480"],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 512, 15, 15), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5s_imgcls_torchhub_480x480", "pt_yolo_v5_yolov5l_imgcls_torchhub_480x480"],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 256, 15, 15), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5s_imgcls_torchhub_480x480", "pt_yolo_v5_yolov5n_imgcls_torchhub_480x480"],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 96, 80, 80), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5m_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_640x640",
                "pt_yolox_yolox_m_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 48, 80, 80), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 96, 40, 40), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 384, 20, 20), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5m_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_imgcls_torchhub_640x640",
                "pt_yolox_yolox_m_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 192, 20, 20), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_320x320", "pt_yolox_yolox_m_obj_det_torchhub"],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 768, 10, 10), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 384, 10, 10), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 255, 10, 10), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5m_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_320x320",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 16, 160, 160), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_320x320", "pt_yolo_v5_yolov5n_imgcls_torchhub_640x640"],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 32, 80, 80), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5n_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_320x320",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 16, 80, 80), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 64, 40, 40), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5n_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_320x320",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 32, 40, 40), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 128, 20, 20), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5n_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_320x320",
                "pt_yolox_yolox_s_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 64, 20, 20), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 256, 10, 10), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_320x320", "pt_yolo_v5_yolov5s_imgcls_torchhub_320x320"],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 128, 10, 10), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 16, 320, 320), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 32, 160, 160), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5n_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_320x320",
                "pt_yolox_yolox_s_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 64, 80, 80), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5n_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_320x320",
                "pt_yolox_yolox_s_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 128, 40, 40), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5n_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_320x320",
                "pt_yolox_yolox_s_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 256, 20, 20), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5n_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_320x320",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 512, 20, 20), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5s_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_320x320",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 48, 240, 240), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 96, 120, 120), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 48, 120, 120), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 192, 60, 60), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 96, 60, 60), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 384, 30, 30), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 192, 30, 30), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 768, 15, 15), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 384, 15, 15), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1024, 20, 20), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5l_imgcls_torchhub_640x640", "pt_yolox_yolox_l_obj_det_torchhub"],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 1024, 10, 10), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5l_imgcls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 512, 10, 10), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5l_imgcls_torchhub_320x320", "pt_yolo_v5_yolov5s_imgcls_torchhub_320x320"],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 80, 80, 80), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5x_imgcls_torchhub_320x320",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 160, 40, 40), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 320, 20, 20), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_320x320", "pt_yolox_yolox_x_obj_det_torchhub"],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 1280, 10, 10), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 640, 10, 10), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 64, 240, 240), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5l_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 128, 120, 120), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5l_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 256, 60, 60), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5l_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 512, 30, 30), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5l_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1024, 15, 15), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5l_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 16, 240, 240), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 16, 120, 120), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 32, 60, 60), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 64, 30, 30), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 128, 15, 15), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 48, 320, 320), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_640x640", "pt_yolox_yolox_m_obj_det_torchhub"],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 96, 160, 160), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_640x640", "pt_yolox_yolox_m_obj_det_torchhub"],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 384, 40, 40), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_640x640", "pt_yolox_yolox_m_obj_det_torchhub"],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 768, 20, 20), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_640x640", "pt_yolox_yolox_m_obj_det_torchhub"],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 64, 56, 80), torch.float32)],
        {"model_name": ["pt_yolo_v6_yolov6s_obj_det_torchhub", "pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 128, 28, 40), torch.float32)],
        {"model_name": ["pt_yolo_v6_yolov6s_obj_det_torchhub", "pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 256, 14, 20), torch.float32)],
        {"model_name": ["pt_yolo_v6_yolov6s_obj_det_torchhub", "pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 80, 56, 80), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 80, 28, 40), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 80, 14, 20), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 64, 224, 320), torch.float32)],
        {"model_name": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 128, 112, 160), torch.float32)],
        {"model_name": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 64, 112, 160), torch.float32)],
        {"model_name": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 256, 56, 80), torch.float32)],
        {"model_name": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 128, 56, 80), torch.float32)],
        {"model_name": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 512, 28, 40), torch.float32)],
        {"model_name": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 256, 28, 40), torch.float32)],
        {"model_name": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1024, 14, 20), torch.float32)],
        {"model_name": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 512, 14, 20), torch.float32)],
        {"model_name": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 32, 56, 80), torch.float32)],
        {"model_name": ["pt_yolo_v6_yolov6n_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 64, 28, 40), torch.float32)],
        {"model_name": ["pt_yolo_v6_yolov6n_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 128, 14, 20), torch.float32)],
        {"model_name": ["pt_yolo_v6_yolov6n_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 96, 56, 80), torch.float32)],
        {"model_name": ["pt_yolo_v6_yolov6m_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 192, 28, 40), torch.float32)],
        {"model_name": ["pt_yolo_v6_yolov6m_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 384, 14, 20), torch.float32)],
        {"model_name": ["pt_yolo_v6_yolov6m_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1, 80, 80), torch.float32)],
        {
            "model_name": [
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 1, 40, 40), torch.float32)],
        {
            "model_name": [
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 80, 40, 40), torch.float32)],
        {
            "model_name": [
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 1, 20, 20), torch.float32)],
        {
            "model_name": [
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 80, 20, 20), torch.float32)],
        {
            "model_name": [
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 24, 208, 208), torch.float32)],
        {"model_name": ["pt_yolox_yolox_tiny_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 48, 104, 104), torch.float32)],
        {"model_name": ["pt_yolox_yolox_tiny_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 24, 104, 104), torch.float32)],
        {"model_name": ["pt_yolox_yolox_tiny_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 96, 52, 52), torch.float32)],
        {"model_name": ["pt_yolox_yolox_tiny_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 48, 52, 52), torch.float32)],
        {"model_name": ["pt_yolox_yolox_tiny_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 192, 26, 26), torch.float32)],
        {"model_name": ["pt_yolox_yolox_tiny_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 96, 26, 26), torch.float32)],
        {"model_name": ["pt_yolox_yolox_tiny_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 384, 13, 13), torch.float32)],
        {"model_name": ["pt_yolox_yolox_tiny_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 192, 13, 13), torch.float32)],
        {"model_name": ["pt_yolox_yolox_tiny_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1, 52, 52), torch.float32)],
        {"model_name": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 80, 52, 52), torch.float32)],
        {"model_name": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1, 26, 26), torch.float32)],
        {"model_name": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 80, 26, 26), torch.float32)],
        {"model_name": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 96, 13, 13), torch.float32)],
        {"model_name": ["pt_yolox_yolox_tiny_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1, 13, 13), torch.float32)],
        {"model_name": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 80, 13, 13), torch.float32)],
        {"model_name": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 16, 208, 208), torch.float32)],
        {"model_name": ["pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 16, 104, 104), torch.float32)],
        {"model_name": ["pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 32, 104, 104), torch.float32)],
        {"model_name": ["pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 32, 52, 52), torch.float32)],
        {"model_name": ["pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 64, 52, 52), torch.float32)],
        {"model_name": ["pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 64, 26, 26), torch.float32)],
        {"model_name": ["pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 128, 26, 26), torch.float32)],
        {"model_name": ["pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 128, 13, 13), torch.float32)],
        {"model_name": ["pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 256, 13, 13), torch.float32)],
        {"model_name": ["pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 64, 13, 13), torch.float32)],
        {"model_name": ["pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99},
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes, forge_property_recorder):
    forge_property_recorder.record_op_name("Sigmoid")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes

    pcc = metadata.pop("pcc")

    for metadata_name, metadata_value in metadata.items():
        forge_property_recorder("tags." + str(metadata_name), metadata_value)

    max_int = 1000
    inputs = [
        Tensor.create_from_shape(operand_shape, operand_dtype, max_int=max_int)
        for operand_shape, operand_dtype in operand_shapes_dtypes
    ]

    framework_model = forge_module(forge_module.__name__)
    framework_model.process_framework_parameters()

    for name, parameter in framework_model._parameters.items():
        parameter_tensor = Tensor.create_torch_tensor(
            shape=parameter.shape.get_pytorch_shape(), dtype=parameter.pt_data_format, max_int=max_int
        )
        framework_model.set_parameter(name, parameter_tensor)

    for name, constant in framework_model._constants.items():
        constant_tensor = Tensor.create_torch_tensor(
            shape=constant.shape.get_pytorch_shape(), dtype=constant.pt_data_format, max_int=max_int
        )
        framework_model.set_constant(name, constant_tensor)

    compiled_model = compile(framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder)

    verify(
        inputs,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
        forge_property_handler=forge_property_recorder,
    )
