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
from forge.forge_property_utils import (
    record_forge_op_name,
    record_op_model_names,
    record_forge_op_args,
    record_single_op_operands_info,
)
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


class Reduceavg2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reduceavg_input_0):
        reduceavg_output_1 = forge.op.ReduceAvg("", reduceavg_input_0, dim=-3, keep_dim=True)
        return reduceavg_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Reduceavg0,
        [((1, 11, 128), torch.float32)],
        {
            "model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 11, 312), torch.float32)],
        {
            "model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 4, 2048), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 512, 256), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_base_img_cls_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 196, 512), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_s16_224_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 256, 256), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 256, 56, 56), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 256, 1, 56), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 512, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 512, 1, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 768, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 768, 1, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 1024, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 1024, 1, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 9, 768), torch.float32)],
        {
            "model_names": [
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 196, 1024), torch.bfloat16)],
        {
            "model_names": [
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 48, 160, 160), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 48, 1, 160), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 24, 160, 160), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 24, 1, 160), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 144, 80, 80), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 144, 1, 80), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 192, 80, 80), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 192, 1, 80), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 192, 40, 40), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 192, 1, 40), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 336, 40, 40), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 336, 1, 40), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 336, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 336, 1, 20), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 672, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 672, 1, 20), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 960, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 960, 1, 20), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 960, 10, 10), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 960, 1, 10), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 1632, 10, 10), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 1632, 1, 10), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 2688, 10, 10), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 2688, 1, 10), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 72, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 72, 1, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 120, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 120, 1, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 480, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 480, 1, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 672, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 672, 1, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 672, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 672, 1, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 960, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 960, 1, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 196, 768), torch.bfloat16)],
        {
            "model_names": [
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 49, 512), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_s32_224_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 6, 1024), torch.float32)],
        {
            "model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 35, 1536), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 29, 1536), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 513, 768), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99, "args": {"dim": "-1", "keep_dim": "True"}},
    ),
    (
        Reduceavg0,
        [((1, 61, 768), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99, "args": {"dim": "-1", "keep_dim": "True"}},
    ),
    (
        Reduceavg1,
        [((1, 16, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 16, 1, 56), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 96, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 96, 1, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 240, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 240, 1, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 120, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 120, 1, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 144, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 144, 1, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 288, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 288, 1, 7), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 576, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 576, 1, 7), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 29, 896), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 256, 512), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b1_img_cls_hf", "pt_segformer_nvidia_mit_b3_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg2,
        [((1, 7, 7, 2048), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim": "-3", "keep_dim": "True"}},
    ),
    (
        Reduceavg1,
        [((1, 1, 7, 2048), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim": "-2", "keep_dim": "True"}},
    ),
    (
        Reduceavg1,
        [((1, 48, 224, 224), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 48, 1, 224), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 24, 224, 224), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 24, 1, 224), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 144, 112, 112), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 144, 1, 112), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 240, 112, 112), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 240, 1, 112), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 240, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 240, 1, 56), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 384, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 384, 1, 56), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 384, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 384, 1, 28), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 768, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 768, 1, 28), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 1056, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 1056, 1, 28), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 1056, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 1056, 1, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 1824, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 1824, 1, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 3072, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 3072, 1, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 513, 512), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"dim": "-1", "keep_dim": "True"}},
    ),
    (
        Reduceavg0,
        [((1, 61, 512), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"dim": "-1", "keep_dim": "True"}},
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    record_forge_op_name("ReduceAvg")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes

    pcc = metadata.pop("pcc")

    for metadata_name, metadata_value in metadata.items():
        if metadata_name == "model_names":
            record_op_model_names(metadata_value)
        elif metadata_name == "args":
            record_forge_op_args(metadata_value)
        else:
            logger.warning(
                "No utility function available in forge property handler to record %s property", metadata_name
            )

    max_int = 1000
    inputs = [
        Tensor.create_from_shape(operand_shape, operand_dtype, max_int=max_int)
        for operand_shape, operand_dtype in operand_shapes_dtypes
    ]

    framework_model = forge_module(forge_module.__name__)

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

    record_single_op_operands_info(framework_model, inputs)

    compiler_cfg = forge.config.CompilerConfig()
    if "default_df_override" in metadata.keys():
        compiler_cfg.default_df_override = forge.DataFormat.from_json(metadata["default_df_override"])

    compiled_model = compile(framework_model, sample_inputs=inputs, compiler_cfg=compiler_cfg)

    verify(inputs, framework_model, compiled_model, VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)))
