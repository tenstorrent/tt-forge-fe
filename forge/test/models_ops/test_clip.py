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


class Clip0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, clip_input_0):
        clip_output_1 = forge.op.Clip("", clip_input_0, min=0.0, max=1.0)
        return clip_output_1


class Clip1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, clip_input_0):
        clip_output_1 = forge.op.Clip("", clip_input_0, min=0.0, max=6.0)
        return clip_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Clip0,
        [((2, 1, 1, 13), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
            ],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip0,
        [((2, 1, 7, 7), torch.float32)],
        {
            "model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip0,
        [((1, 1, 256, 256), torch.float32)],
        {
            "model_name": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_xglm_facebook_xglm_1_7b_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip0,
        [((1, 12, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip0,
        [((1, 12, 384, 384), torch.float32)],
        {
            "model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip0,
        [((1, 1, 32, 32), torch.float32)],
        {
            "model_name": [
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip1,
        [((1, 72, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 120, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 480, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 672, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 960, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 24, 96, 96), torch.float32)],
        {
            "model_name": ["pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 48, 96, 96), torch.float32)],
        {
            "model_name": ["pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 48, 48, 48), torch.float32)],
        {
            "model_name": [
                "pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 96, 48, 48), torch.float32)],
        {
            "model_name": ["pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 96, 24, 24), torch.float32)],
        {
            "model_name": ["pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 192, 24, 24), torch.float32)],
        {
            "model_name": ["pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 192, 12, 12), torch.float32)],
        {
            "model_name": ["pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 384, 12, 12), torch.float32)],
        {
            "model_name": ["pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 384, 6, 6), torch.float32)],
        {
            "model_name": ["pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 768, 6, 6), torch.float32)],
        {
            "model_name": ["pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 32, 112, 112), torch.float32)],
        {
            "model_name": [
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 64, 112, 112), torch.float32)],
        {
            "model_name": ["pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 64, 56, 56), torch.float32)],
        {
            "model_name": ["pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 128, 56, 56), torch.float32)],
        {
            "model_name": ["pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 128, 28, 28), torch.float32)],
        {
            "model_name": ["pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 256, 28, 28), torch.float32)],
        {
            "model_name": ["pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 256, 14, 14), torch.float32)],
        {
            "model_name": ["pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 512, 14, 14), torch.float32)],
        {
            "model_name": ["pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 512, 7, 7), torch.float32)],
        {
            "model_name": ["pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 1024, 7, 7), torch.float32)],
        {
            "model_name": ["pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 16, 48, 48), torch.float32)],
        {
            "model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 48, 24, 24), torch.float32)],
        {
            "model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 48, 12, 12), torch.float32)],
        {
            "model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 96, 12, 12), torch.float32)],
        {
            "model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 96, 6, 6), torch.float32)],
        {
            "model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 144, 6, 6), torch.float32)],
        {
            "model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 192, 6, 6), torch.float32)],
        {
            "model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 192, 3, 3), torch.float32)],
        {
            "model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 336, 3, 3), torch.float32)],
        {
            "model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 1280, 3, 3), torch.float32)],
        {
            "model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 96, 112, 112), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 96, 56, 56), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 144, 56, 56), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 144, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 192, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 192, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 384, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 576, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 576, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 960, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 1280, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 24, 80, 80), torch.float32)],
        {
            "model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 96, 80, 80), torch.float32)],
        {
            "model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 96, 40, 40), torch.float32)],
        {
            "model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 144, 40, 40), torch.float32)],
        {
            "model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 144, 20, 20), torch.float32)],
        {
            "model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 144, 10, 10), torch.float32)],
        {
            "model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 288, 10, 10), torch.float32)],
        {
            "model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 432, 10, 10), torch.float32)],
        {
            "model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 432, 5, 5), torch.float32)],
        {
            "model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 720, 5, 5), torch.float32)],
        {
            "model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 1280, 5, 5), torch.float32)],
        {
            "model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 384, 28, 28), torch.float32)],
        {
            "model_name": ["pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 576, 28, 28), torch.float32)],
        {
            "model_name": ["pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 960, 28, 28), torch.float32)],
        {
            "model_name": ["pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 16, 112, 112), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 16, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 96, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 96, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 96, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 240, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 240, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 120, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 144, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 144, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 288, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 288, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 288, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 576, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 1024), torch.float32)],
        {
            "model_name": ["pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 1024, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 240, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 200, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 184, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 480, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 672, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 672, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 1280, 1, 1), torch.float32)],
        {
            "model_name": ["pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 1280), torch.float32)],
        {
            "model_name": ["pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub"],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 256, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 512, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 768, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"min": "0.0", "max": "6.0"},
        },
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes, record_forge_property):
    record_forge_property("tags.op_name", "Clip")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes

    pcc = metadata.pop("pcc")

    for metadata_name, metadata_value in metadata.items():
        record_forge_property("tags." + str(metadata_name), metadata_value)

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
