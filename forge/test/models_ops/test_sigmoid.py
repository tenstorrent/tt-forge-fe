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
        [((1, 588, 5504), torch.float32)],
        {"model_name": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 39, 11008), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_math_7b_instruct_qa_hf", "pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 577, 4096), torch.float32)],
        {"model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 596, 11008), torch.float32)],
        {"model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 10, 23040), torch.float32)],
        {"model_name": ["pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 10, 8192), torch.float32)],
        {"model_name": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 10, 9216), torch.float32)],
        {"model_name": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf"], "pcc": 0.99},
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
        [((1, 4, 8192), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 4, 11008), torch.float32)],
        {"model_name": ["pt_llama3_huggyllama_llama_7b_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 256, 14336), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
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
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 32, 8192), torch.float32)],
        {"model_name": ["pt_llama3_meta_llama_llama_3_2_3b_clm_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 32, 11008), torch.float32)],
        {"model_name": ["pt_llama3_huggyllama_llama_7b_clm_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 128, 14336), torch.float32)],
        {"model_name": ["pt_mistral_mistralai_mistral_7b_v0_1_clm_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 5, 8192), torch.float32)],
        {"model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 13, 8192), torch.float32)],
        {"model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf"], "pcc": 0.99},
    ),
    (Sigmoid0, [((1, 6, 2816), torch.float32)], {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99}),
    (
        Sigmoid0,
        [((1, 29, 2816), torch.float32)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 35, 11008), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 35, 4864), torch.float32)],
        {"model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 35, 8960), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
            ],
            "pcc": 0.99,
        },
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
    (Sigmoid0, [((1, 29, 8960), torch.float32)], {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99}),
    (
        Sigmoid0,
        [((1, 39, 8960), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 13, 18944), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 39, 18944), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 39, 4864), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (Sigmoid0, [((1, 29, 4864), torch.float32)], {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99}),
    (Sigmoid0, [((1, 29, 11008), torch.float32)], {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99}),
    (Sigmoid0, [((1, 29, 18944), torch.float32)], {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99}),
    (
        Sigmoid0,
        [((1, 18), torch.float32)],
        {"model_name": ["pt_densenet_densenet121_hf_xray_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 32, 112, 112), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 8, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 32, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 96, 112, 112), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 96, 56, 56), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 4, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 96, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 144, 56, 56), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 6, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 144, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_regnet_regnet_y_800mf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 144, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 240, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 10, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 240, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 240, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 480, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 20, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 480, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 672, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 28, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 672, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 672, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 1152, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 48, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_regnet_regnet_y_1_6gf_img_cls_torchvision",
                "pt_regnet_regnet_y_400mf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 1152, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 1280, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
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
        [((1, 12, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 24, 112, 112), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 24, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
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
        [((1, 192, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
            ],
            "pcc": 0.99,
        },
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
        [((1, 14, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 336, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_regnet_regnet_y_1_6gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
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
        [((1, 40, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 960, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
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
        [((1, 68, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 1632, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 2688, 7, 7), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 112, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 2688, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 1792, 7, 7), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 48, 160, 160), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 24, 160, 160), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
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
        [((1, 192, 80, 80), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 192, 40, 40), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 336, 40, 40), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
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
        [((1, 960, 20, 20), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
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
        [((1, 2688, 10, 10), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1792, 10, 10), torch.float32)],
        {"model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 72, 28, 28), torch.float32)],
        {"model_name": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 120, 14, 14), torch.float32)],
        {"model_name": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 200, 7, 7), torch.float32)],
        {"model_name": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 184, 7, 7), torch.float32)],
        {"model_name": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 480, 7, 7), torch.float32)],
        {"model_name": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 960, 3, 3), torch.float32)],
        {"model_name": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 2, 30, 40), torch.float32)],
        {"model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 2, 60, 80), torch.float32)],
        {"model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 2, 120, 160), torch.float32)],
        {"model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1, 480, 640), torch.float32)],
        {"model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1, 320, 1024), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 1, 192, 640), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 224, 1, 1), torch.float32)],
        {
            "model_name": ["pt_regnet_regnet_y_16gf_img_cls_torchvision", "pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 448, 1, 1), torch.float32)],
        {
            "model_name": ["pt_regnet_regnet_y_16gf_img_cls_torchvision", "pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 1232, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_16gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 3024, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_16gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 120, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_1_6gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 888, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_1_6gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 232, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_32gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 696, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_32gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1392, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_32gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 3712, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_32gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 64, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_800mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 320, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_800mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 784, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_800mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 528, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1056, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 2904, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 7392, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 104, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_400mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 208, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_400mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 440, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_400mf_img_cls_torchvision"], "pcc": 0.99},
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
        [((1, 896, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_8gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 2016, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_8gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 72, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_3_2gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 216, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_3_2gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 576, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_3_2gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1512, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_3_2gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 720, 60, 80), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
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
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
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
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
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
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
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
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (Sigmoid0, [((1, 1, 256, 256), torch.float32)], {"model_name": ["pt_unet_base_img_seg_torchhub"], "pcc": 0.99}),
    (
        Sigmoid0,
        [((1, 16, 160, 160), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 32, 80, 80), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 16, 80, 80), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 64, 40, 40), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 32, 40, 40), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 128, 20, 20), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 64, 20, 20), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 256, 10, 10), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 128, 10, 10), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 255, 40, 40), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 255, 20, 20), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 255, 10, 10), torch.float32)],
        {"model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_320x320"], "pcc": 0.99},
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes, forge_property_recorder):
    forge_property_recorder("tags.op_name", "Sigmoid")

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
