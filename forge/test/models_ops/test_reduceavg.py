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
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.config import VerifyConfig
import pytest


class Reduceavg0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reduceavg_input_0):
        reduceavg_output_1 = forge.op.ReduceAvg("", reduceavg_input_0, dim=-1, keep_dim=True)
        return reduceavg_output_1


class Reduceavg1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reduceavg_input_0):
        reduceavg_output_1 = forge.op.ReduceAvg("", reduceavg_input_0, dim=-2, keep_dim=True)
        return reduceavg_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Reduceavg0,
        [((2, 13, 768), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 39, 4096), torch.float32)],
        {
            "model_name": ["pt_deepseek_deepseek_math_7b_instruct_qa_hf", "DeepSeekWrapper_decoder"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 10, 3072), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 10, 2048), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 7, 2048), torch.float32)],
        {
            "model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 4, 2048), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 256, 2048), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 4, 4096), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 128, 4096), torch.float32)],
        {
            "model_name": ["pt_mistral_mistralai_mistral_7b_v0_1_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 256, 3072), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 13, 3072), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 5, 3072), torch.float32)],
        {
            "model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 6, 1024), torch.float32)],
        {
            "model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 29, 1024), torch.float32)],
        {
            "model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 35, 3584), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 35, 1536), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 35, 2048), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 35, 896), torch.float32)],
        {
            "model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 29, 1536), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 39, 1536), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 39, 3584), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 29, 3584), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 29, 2048), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 39, 2048), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 29, 896), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 39, 896), torch.float32)],
        {
            "model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 1, 1024), torch.float32)],
        {
            "model_name": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 61, 1024), torch.float32)],
        {
            "model_name": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 1, 512), torch.float32)],
        {
            "model_name": ["pt_t5_t5_small_text_gen_hf", "pt_t5_google_flan_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 61, 512), torch.float32)],
        {
            "model_name": ["pt_t5_t5_small_text_gen_hf", "pt_t5_google_flan_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 1, 768), torch.float32)],
        {
            "model_name": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 61, 768), torch.float32)],
        {
            "model_name": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 48, 160, 160), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 48, 1, 160), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 24, 160, 160), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 24, 1, 160), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 144, 80, 80), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 144, 1, 80), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 192, 80, 80), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 192, 1, 80), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 192, 40, 40), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 192, 1, 40), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 336, 40, 40), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 336, 1, 40), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 336, 20, 20), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 336, 1, 20), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 672, 20, 20), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 672, 1, 20), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 960, 20, 20), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 960, 1, 20), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 960, 10, 10), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 960, 1, 10), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 1632, 10, 10), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 1632, 1, 10), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 2688, 10, 10), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 2688, 1, 10), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 32, 112, 112), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b0_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 32, 1, 112), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b0_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 96, 56, 56), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b0_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 96, 1, 56), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b0_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 144, 56, 56), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b0_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 144, 1, 56), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b0_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 144, 28, 28), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b0_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 144, 1, 28), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b0_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 240, 28, 28), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b0_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 240, 1, 28), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b0_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 240, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 240, 1, 14), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 480, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 480, 1, 14), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 672, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 672, 1, 14), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 672, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 672, 1, 7), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 1152, 7, 7), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b0_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 1152, 1, 7), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b0_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 72, 28, 28), torch.float32)],
        {
            "model_name": ["pt_ghostnet_ghostnet_100_img_cls_timm", "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 72, 1, 28), torch.float32)],
        {
            "model_name": ["pt_ghostnet_ghostnet_100_img_cls_timm", "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 120, 28, 28), torch.float32)],
        {
            "model_name": ["pt_ghostnet_ghostnet_100_img_cls_timm", "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 120, 1, 28), torch.float32)],
        {
            "model_name": ["pt_ghostnet_ghostnet_100_img_cls_timm", "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 960, 7, 7), torch.float32)],
        {
            "model_name": ["pt_ghostnet_ghostnet_100_img_cls_timm", "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 960, 1, 7), torch.float32)],
        {
            "model_name": ["pt_ghostnet_ghostnet_100_img_cls_timm", "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 49, 768), torch.float32)],
        {
            "model_name": ["pt_mlp_mixer_mixer_b32_224_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 196, 512), torch.float32)],
        {
            "model_name": ["pt_mlp_mixer_mixer_s16_224_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 49, 512), torch.float32)],
        {
            "model_name": ["pt_mlp_mixer_mixer_s32_224_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 196, 768), torch.float32)],
        {
            "model_name": [
                "pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 49, 1024), torch.float32)],
        {
            "model_name": ["pt_mlp_mixer_mixer_l32_224_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 196, 1024), torch.float32)],
        {
            "model_name": ["pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm", "pt_mlp_mixer_mixer_l16_224_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 16, 56, 56), torch.float32)],
        {
            "model_name": ["pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 16, 1, 56), torch.float32)],
        {
            "model_name": ["pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 96, 14, 14), torch.float32)],
        {
            "model_name": ["pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 96, 1, 14), torch.float32)],
        {
            "model_name": ["pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 120, 14, 14), torch.float32)],
        {
            "model_name": ["pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 120, 1, 14), torch.float32)],
        {
            "model_name": ["pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 144, 14, 14), torch.float32)],
        {
            "model_name": ["pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 144, 1, 14), torch.float32)],
        {
            "model_name": ["pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 288, 7, 7), torch.float32)],
        {
            "model_name": ["pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 288, 1, 7), torch.float32)],
        {
            "model_name": ["pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 576, 7, 7), torch.float32)],
        {
            "model_name": ["pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 576, 1, 7), torch.float32)],
        {
            "model_name": ["pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 256, 256), torch.float32)],
        {
            "model_name": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 256, 512), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 256, 56, 56), torch.float32)],
        {
            "model_name": [
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 256, 1, 56), torch.float32)],
        {
            "model_name": [
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 512, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 512, 1, 28), torch.float32)],
        {
            "model_name": [
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 768, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 768, 1, 14), torch.float32)],
        {
            "model_name": [
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 1024, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 1024, 1, 7), torch.float32)],
        {
            "model_name": [
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "keep_dim": "True"},
        },
    ),
]


@pytest.mark.push
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes, record_forge_property):
    record_forge_property("op_name", "ReduceAvg")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes

    pcc = metadata.pop("pcc")

    for metadata_name, metadata_value in metadata.items():
        record_forge_property(metadata_name, metadata_value)

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

    compiled_model = compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model, VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)))
