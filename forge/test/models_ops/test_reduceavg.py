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
        reduceavg_output_1 = forge.op.ReduceAvg("", reduceavg_input_0, dim=-2, keep_dim=True)
        return reduceavg_output_1


class Reduceavg1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reduceavg_input_0):
        reduceavg_output_1 = forge.op.ReduceAvg("", reduceavg_input_0, dim=-1, keep_dim=True)
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
        [((1, 196, 512), torch.float32)],
        {
            "model_names": ["onnx_mlp_mixer_mixer_s16_224_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 12, 120), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 48, 160, 160), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 48, 1, 160), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 24, 160, 160), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 24, 1, 160), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 144, 80, 80), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 144, 1, 80), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 192, 80, 80), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 192, 1, 80), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 192, 40, 40), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 192, 1, 40), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 336, 40, 40), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 336, 1, 40), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 336, 20, 20), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 336, 1, 20), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 672, 20, 20), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 672, 1, 20), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 960, 20, 20), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 960, 1, 20), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 960, 10, 10), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 960, 1, 10), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 1632, 10, 10), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 1632, 1, 10), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 2688, 10, 10), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 2688, 1, 10), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 32, 112, 112), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 32, 1, 112), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 96, 56, 56), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 96, 1, 56), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 144, 56, 56), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 144, 1, 56), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 144, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 144, 1, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 240, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 240, 1, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 240, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b0_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 240, 1, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b0_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 480, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 480, 1, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 672, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 672, 1, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 672, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 672, 1, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b0_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 1152, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 1152, 1, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 256, 512), torch.bfloat16)],
        {
            "model_names": [
                "pt_mlp_mixer_base_img_cls_github",
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 196, 768), torch.bfloat16)],
        {
            "model_names": [
                "pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
                "pt_beit_base_img_cls_hf",
                "pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 196, 512), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_s16_224_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 49, 512), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_s32_224_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 35, 896), torch.float32)],
        {"model_names": ["pt_qwen_coder_0_5b_clm_hf"], "pcc": 0.99, "args": {"dim": "-1", "keep_dim": "True"}},
    ),
    (
        Reduceavg1,
        [((1, 25, 768), torch.float32)],
        {
            "model_names": [
                "pt_stereo_medium_music_generation_hf",
                "pt_stereo_small_music_generation_hf",
                "pt_stereo_large_music_generation_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 168, 56, 56), torch.float32)],
        {
            "model_names": ["regnet_regnety_080_onnx", "onnx_regnet_facebook_regnet_y_080_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 168, 1, 56), torch.float32)],
        {
            "model_names": ["regnet_regnety_080_onnx", "onnx_regnet_facebook_regnet_y_080_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 448, 28, 28), torch.float32)],
        {
            "model_names": [
                "regnet_regnety_080_onnx",
                "regnet_regnety_160_onnx",
                "regnet_regnety_120_onnx",
                "onnx_regnet_facebook_regnet_y_080_img_cls_hf",
                "onnx_regnet_facebook_regnet_y_120_img_cls_hf",
                "onnx_regnet_facebook_regnet_y_160_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 448, 1, 28), torch.float32)],
        {
            "model_names": [
                "regnet_regnety_080_onnx",
                "regnet_regnety_160_onnx",
                "regnet_regnety_120_onnx",
                "onnx_regnet_facebook_regnet_y_080_img_cls_hf",
                "onnx_regnet_facebook_regnet_y_120_img_cls_hf",
                "onnx_regnet_facebook_regnet_y_160_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 896, 14, 14), torch.float32)],
        {
            "model_names": [
                "regnet_regnety_080_onnx",
                "regnet_regnety_120_onnx",
                "onnx_regnet_facebook_regnet_y_080_img_cls_hf",
                "onnx_regnet_facebook_regnet_y_120_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 896, 1, 14), torch.float32)],
        {
            "model_names": [
                "regnet_regnety_080_onnx",
                "regnet_regnety_120_onnx",
                "onnx_regnet_facebook_regnet_y_080_img_cls_hf",
                "onnx_regnet_facebook_regnet_y_120_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 2016, 7, 7), torch.float32)],
        {
            "model_names": ["regnet_regnety_080_onnx", "onnx_regnet_facebook_regnet_y_080_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 2016, 1, 7), torch.float32)],
        {
            "model_names": ["regnet_regnety_080_onnx", "onnx_regnet_facebook_regnet_y_080_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 40, 144, 144), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 40, 1, 144), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 24, 144, 144), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 24, 1, 144), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 144, 72, 72), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 144, 1, 72), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 192, 72, 72), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 192, 1, 72), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 192, 36, 36), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 192, 1, 36), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 288, 36, 36), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 288, 1, 36), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 288, 18, 18), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 288, 1, 18), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 576, 18, 18), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 576, 1, 18), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 816, 18, 18), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 816, 1, 18), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 816, 9, 9), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 816, 1, 9), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 1392, 9, 9), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 1392, 1, 9), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 2304, 9, 9), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 2304, 1, 9), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 72, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_ghostnet_ghostnet_100_in1k_img_cls_timm", "onnx_ghostnet_ghostnet_100_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 72, 1, 28), torch.float32)],
        {
            "model_names": ["onnx_ghostnet_ghostnet_100_in1k_img_cls_timm", "onnx_ghostnet_ghostnet_100_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 120, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_ghostnet_ghostnet_100_in1k_img_cls_timm", "onnx_ghostnet_ghostnet_100_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 120, 1, 28), torch.float32)],
        {
            "model_names": ["onnx_ghostnet_ghostnet_100_in1k_img_cls_timm", "onnx_ghostnet_ghostnet_100_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 480, 14, 14), torch.float32)],
        {
            "model_names": [
                "onnx_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "onnx_ghostnet_ghostnet_100_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 480, 1, 14), torch.float32)],
        {
            "model_names": [
                "onnx_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "onnx_ghostnet_ghostnet_100_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 672, 14, 14), torch.float32)],
        {
            "model_names": [
                "onnx_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "onnx_ghostnet_ghostnet_100_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 672, 1, 14), torch.float32)],
        {
            "model_names": [
                "onnx_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "onnx_ghostnet_ghostnet_100_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 672, 7, 7), torch.float32)],
        {
            "model_names": [
                "onnx_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "onnx_ghostnet_ghostnet_100_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 672, 1, 7), torch.float32)],
        {
            "model_names": [
                "onnx_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "onnx_ghostnet_ghostnet_100_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 960, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_ghostnet_ghostnet_100_in1k_img_cls_timm", "onnx_ghostnet_ghostnet_100_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 960, 1, 7), torch.float32)],
        {
            "model_names": ["onnx_ghostnet_ghostnet_100_in1k_img_cls_timm", "onnx_ghostnet_ghostnet_100_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 196, 768), torch.float32)],
        {
            "model_names": [
                "onnx_mlp_mixer_mixer_b16_224_img_cls_timm",
                "onnx_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                "onnx_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "onnx_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "onnx_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
                "onnx_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 14, 768), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_qa_padlenlp"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 9, 768), torch.float32)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_qa_padlenlp",
                "pd_ernie_1_0_mlm_padlenlp",
                "pd_ernie_1_0_qa_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_ernie_1_0_seq_cls_padlenlp",
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 197, 768), torch.float32)],
        {
            "model_names": [
                "pd_clip_vision_openai_clip_vit_base_patch16_img_enc_padlenlp",
                "pd_chineseclip_ofa_sys_chinese_clip_vit_base_patch16_img_text_pairing_padlenlp",
                "pd_chineseclip_vision_ofa_sys_chinese_clip_vit_base_patch16_img_enc_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 768), torch.float32)],
        {
            "model_names": [
                "pd_clip_vision_openai_clip_vit_base_patch16_img_enc_padlenlp",
                "pd_chineseclip_ofa_sys_chinese_clip_vit_base_patch16_img_text_pairing_padlenlp",
                "pd_chineseclip_vision_ofa_sys_chinese_clip_vit_base_patch16_img_enc_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 196, 1024), torch.bfloat16)],
        {
            "model_names": [
                "pt_beit_large_img_cls_hf",
                "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_l16_224_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 588, 2048), torch.float32)],
        {"model_names": ["pt_deepseek_1_3b_instruct_qa_hf"], "pcc": 0.99, "args": {"dim": "-1", "keep_dim": "True"}},
    ),
    (
        Reduceavg0,
        [((1, 49, 768), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_b32_224_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 224, 56, 56), torch.float32)],
        {
            "model_names": [
                "regnet_regnety_160_onnx",
                "regnet_regnety_120_onnx",
                "onnx_regnet_facebook_regnet_y_120_img_cls_hf",
                "onnx_regnet_facebook_regnet_y_160_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 224, 1, 56), torch.float32)],
        {
            "model_names": [
                "regnet_regnety_160_onnx",
                "regnet_regnety_120_onnx",
                "onnx_regnet_facebook_regnet_y_120_img_cls_hf",
                "onnx_regnet_facebook_regnet_y_160_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 1232, 14, 14), torch.float32)],
        {
            "model_names": ["regnet_regnety_160_onnx", "onnx_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 1232, 1, 14), torch.float32)],
        {
            "model_names": ["regnet_regnety_160_onnx", "onnx_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 3024, 7, 7), torch.float32)],
        {
            "model_names": ["regnet_regnety_160_onnx", "onnx_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 3024, 1, 7), torch.float32)],
        {
            "model_names": ["regnet_regnety_160_onnx", "onnx_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 49, 1024), torch.float32)],
        {
            "model_names": ["onnx_mlp_mixer_mixer_l32_224_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 1, 512), torch.float32)],
        {"model_names": ["onnx_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"dim": "-1", "keep_dim": "True"}},
    ),
    (
        Reduceavg1,
        [((1, 61, 512), torch.float32)],
        {
            "model_names": [
                "onnx_t5_t5_small_text_gen_hf",
                "pt_t5_t5_small_text_gen_hf",
                "pt_t5_google_flan_t5_small_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 29, 1536), torch.float32)],
        {"model_names": ["pt_qwen_v2_1_5b_clm_hf"], "pcc": 0.99, "args": {"dim": "-1", "keep_dim": "True"}},
    ),
    (
        Reduceavg0,
        [((1, 256, 56, 56), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet39b_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_img_cls_timm",
                "pt_vovnet_ese_vovnet99b_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 256, 1, 56), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet39b_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_img_cls_timm",
                "pt_vovnet_ese_vovnet99b_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 512, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet39b_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_img_cls_timm",
                "pt_vovnet_ese_vovnet99b_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 512, 1, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet39b_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_img_cls_timm",
                "pt_vovnet_ese_vovnet99b_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 768, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet39b_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_img_cls_timm",
                "pt_vovnet_ese_vovnet99b_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 768, 1, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet39b_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_img_cls_timm",
                "pt_vovnet_ese_vovnet99b_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 1024, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet39b_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_img_cls_timm",
                "pt_vovnet_ese_vovnet99b_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 1024, 1, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet39b_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_img_cls_timm",
                "pt_vovnet_ese_vovnet99b_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 144, 56, 56), torch.float32)],
        {
            "model_names": [
                "regnet_regnety_064_onnx",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_regnet_facebook_regnet_y_064_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 144, 1, 56), torch.float32)],
        {
            "model_names": [
                "regnet_regnety_064_onnx",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_regnet_facebook_regnet_y_064_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 288, 28, 28), torch.float32)],
        {
            "model_names": ["regnet_regnety_064_onnx", "onnx_regnet_facebook_regnet_y_064_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 288, 1, 28), torch.float32)],
        {
            "model_names": ["regnet_regnety_064_onnx", "onnx_regnet_facebook_regnet_y_064_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 576, 14, 14), torch.float32)],
        {
            "model_names": ["regnet_regnety_064_onnx", "onnx_regnet_facebook_regnet_y_064_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 576, 1, 14), torch.float32)],
        {
            "model_names": ["regnet_regnety_064_onnx", "onnx_regnet_facebook_regnet_y_064_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 1296, 7, 7), torch.float32)],
        {
            "model_names": ["regnet_regnety_064_onnx", "onnx_regnet_facebook_regnet_y_064_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 1296, 1, 7), torch.float32)],
        {
            "model_names": ["regnet_regnety_064_onnx", "onnx_regnet_facebook_regnet_y_064_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 49, 768), torch.float32)],
        {
            "model_names": ["onnx_mlp_mixer_mixer_b32_224_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 256, 256), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 256, 512), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_mit_b2_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 11, 128), torch.float32)],
        {
            "model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 11, 312), torch.float32)],
        {
            "model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 15, 768), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 8, 768), torch.float32)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
                "pd_blip_text_salesforce_blip_image_captioning_base_text_enc_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 522, 2048), torch.float32)],
        {
            "model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 32, 128, 128), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 32, 1, 128), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 16, 128, 128), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 16, 1, 128), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 96, 64, 64), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 96, 1, 64), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 144, 64, 64), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 144, 1, 64), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 144, 32, 32), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 144, 1, 32), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 288, 32, 32), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 288, 1, 32), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 288, 16, 16), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 288, 1, 16), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 528, 16, 16), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 528, 1, 16), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 720, 16, 16), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 720, 1, 16), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 720, 8, 8), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 720, 1, 8), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 1248, 8, 8), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 1248, 1, 8), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 2112, 8, 8), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 2112, 1, 8), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 48, 160, 160), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 48, 1, 160), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 24, 160, 160), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 24, 1, 160), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 144, 80, 80), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 144, 1, 80), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 192, 80, 80), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 192, 1, 80), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 192, 40, 40), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 192, 1, 40), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 336, 40, 40), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 336, 1, 40), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 336, 20, 20), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 336, 1, 20), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 672, 20, 20), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 672, 1, 20), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 960, 20, 20), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 960, 1, 20), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 960, 10, 10), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 960, 1, 10), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 1632, 10, 10), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 1632, 1, 10), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 2688, 10, 10), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 2688, 1, 10), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 196, 1024), torch.float32)],
        {
            "model_names": [
                "onnx_mlp_mixer_mixer_l16_224_img_cls_timm",
                "onnx_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "onnx_mlp_mixer_mixer_l16_224_in21k_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 577, 768), torch.float32)],
        {
            "model_names": [
                "pd_blip_vision_salesforce_blip_image_captioning_base_img_enc_padlenlp",
                "pd_blip_salesforce_blip_image_captioning_base_img_captioning_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((4, 5, 768), torch.float32)],
        {
            "model_names": ["pd_chineseclip_ofa_sys_chinese_clip_vit_base_patch16_img_text_pairing_padlenlp"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 11, 768), torch.float32)],
        {
            "model_names": [
                "pd_roberta_rbt4_ch_clm_padlenlp",
                "pd_bert_chinese_roberta_base_qa_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 49, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_l32_224_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 6, 1024), torch.float32)],
        {"model_names": ["pt_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"dim": "-1", "keep_dim": "True"}},
    ),
    (
        Reduceavg1,
        [((4, 31, 2560), torch.bfloat16)],
        {
            "model_names": ["pt_qwen_v3_embedding_4b_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((4, 31, 32, 128), torch.bfloat16)],
        {
            "model_names": ["pt_qwen_v3_embedding_4b_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((4, 31, 8, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_qwen_v3_embedding_4b_sentence_embed_gen_hf",
                "pt_qwen_v3_embedding_0_6b_sentence_embed_gen_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 513, 1024), torch.float32)],
        {
            "model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 61, 1024), torch.float32)],
        {
            "model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 48, 224, 224), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 48, 1, 224), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 24, 224, 224), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 24, 1, 224), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 144, 112, 112), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 144, 1, 112), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 240, 112, 112), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 240, 1, 112), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 240, 56, 56), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 240, 1, 56), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 384, 56, 56), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 384, 1, 56), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 384, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 384, 1, 28), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 768, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 768, 1, 28), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 1056, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 1056, 1, 28), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 1056, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 1056, 1, 14), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 1824, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 1824, 1, 14), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 3072, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 3072, 1, 14), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 256, 19, 19), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 256, 1, 19), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 512, 19, 19), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 512, 1, 19), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 768, 19, 19), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 768, 1, 19), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 960, 19, 19), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 960, 1, 19), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 1536, 10, 10), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 1536, 1, 10), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 256, 256), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 72, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 72, 1, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 120, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 120, 1, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 960, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 960, 1, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 4, 2048), torch.float32)],
        {
            "model_names": ["pt_llama3_llama_3_2_1b_seq_cls_hf", "pt_llama3_llama_3_2_1b_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 128, 1024), torch.float32)],
        {"model_names": ["pt_qwen_v3_0_6b_clm_hf"], "pcc": 0.99, "args": {"dim": "-1", "keep_dim": "True"}},
    ),
    (
        Reduceavg1,
        [((1, 128, 16, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v3_0_6b_clm_hf", "pt_qwen_v3_1_7b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 128, 8, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v3_0_6b_clm_hf", "pt_qwen_v3_1_7b_clm_hf", "pt_qwen_v3_4b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 232, 56, 56), torch.float32)],
        {
            "model_names": ["regnet_regnety_320_onnx", "onnx_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 232, 1, 56), torch.float32)],
        {
            "model_names": ["regnet_regnety_320_onnx", "onnx_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 696, 28, 28), torch.float32)],
        {
            "model_names": ["regnet_regnety_320_onnx", "onnx_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 696, 1, 28), torch.float32)],
        {
            "model_names": ["regnet_regnety_320_onnx", "onnx_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 1392, 14, 14), torch.float32)],
        {
            "model_names": ["regnet_regnety_320_onnx", "onnx_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 1392, 1, 14), torch.float32)],
        {
            "model_names": ["regnet_regnety_320_onnx", "onnx_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 3712, 7, 7), torch.float32)],
        {
            "model_names": ["regnet_regnety_320_onnx", "onnx_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 3712, 1, 7), torch.float32)],
        {
            "model_names": ["regnet_regnety_320_onnx", "onnx_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 32, 112, 112), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 32, 1, 112), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 96, 56, 56), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 96, 1, 56), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 144, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 144, 1, 28), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 240, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 240, 1, 28), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 240, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 240, 1, 14), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 1152, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 1152, 1, 7), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 256, 56, 56), torch.float32)],
        {
            "model_names": ["onnx_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 256, 1, 56), torch.float32)],
        {
            "model_names": ["onnx_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 512, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 512, 1, 28), torch.float32)],
        {
            "model_names": ["onnx_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 768, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 768, 1, 14), torch.float32)],
        {
            "model_names": ["onnx_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 1024, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 1024, 1, 7), torch.float32)],
        {
            "model_names": ["onnx_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 256, 18, 18), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 256, 1, 18), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 512, 18, 18), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 512, 1, 18), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 768, 18, 18), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 768, 1, 18), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 960, 18, 18), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 960, 1, 18), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 960, 9, 9), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 960, 1, 9), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 1632, 9, 9), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 1632, 1, 9), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 128, 2048), torch.float32)],
        {"model_names": ["pt_qwen_v3_1_7b_clm_hf"], "pcc": 0.99, "args": {"dim": "-1", "keep_dim": "True"}},
    ),
    (
        Reduceavg1,
        [((1, 513, 768), torch.float32)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 61, 768), torch.float32)],
        {
            "model_names": ["pt_t5_t5_base_text_gen_hf", "pt_t5_google_flan_t5_base_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 49, 512), torch.float32)],
        {
            "model_names": ["onnx_mlp_mixer_mixer_s32_224_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 10, 768), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 16, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 16, 1, 56), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 96, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 96, 1, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 120, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 120, 1, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 144, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 144, 1, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 288, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 288, 1, 7), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 576, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 576, 1, 7), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 35, 1536), torch.float32)],
        {"model_names": ["pt_qwen_coder_1_5b_clm_hf"], "pcc": 0.99, "args": {"dim": "-1", "keep_dim": "True"}},
    ),
    (
        Reduceavg1,
        [((1, 513, 512), torch.float32)],
        {
            "model_names": ["pt_t5_t5_small_text_gen_hf", "pt_t5_google_flan_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 2240, 7, 7), torch.float32)],
        {
            "model_names": ["regnet_regnety_120_onnx", "onnx_regnet_facebook_regnet_y_120_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 2240, 1, 7), torch.float32)],
        {
            "model_names": ["regnet_regnety_120_onnx", "onnx_regnet_facebook_regnet_y_120_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 32, 120, 120), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 32, 1, 120), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 16, 120, 120), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 16, 1, 120), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 96, 60, 60), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 96, 1, 60), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 144, 60, 60), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 144, 1, 60), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 144, 30, 30), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 144, 1, 30), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 240, 30, 30), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 240, 1, 30), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 240, 15, 15), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 240, 1, 15), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 480, 15, 15), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 480, 1, 15), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 672, 15, 15), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 672, 1, 15), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 672, 8, 8), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 672, 1, 8), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 1152, 8, 8), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 1152, 1, 8), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 1920, 8, 8), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 1920, 1, 8), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((2, 4, 768), torch.float32)],
        {
            "model_names": ["pd_blip_salesforce_blip_image_captioning_base_img_captioning_padlenlp"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 48, 224, 224), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 48, 1, 224), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 24, 224, 224), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 24, 1, 224), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 144, 112, 112), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 144, 1, 112), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 240, 112, 112), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 240, 1, 112), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 240, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 240, 1, 56), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 384, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 384, 1, 56), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 384, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 384, 1, 28), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 768, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 768, 1, 28), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 1056, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 1056, 1, 28), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 1056, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 1056, 1, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 1824, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 1824, 1, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 3072, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 3072, 1, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 29, 1024), torch.float32)],
        {"model_names": ["pt_qwen1_5_0_5b_chat_clm_hf"], "pcc": 0.99, "args": {"dim": "-1", "keep_dim": "True"}},
    ),
    (
        Reduceavg1,
        [((1, 39, 896), torch.float32)],
        {"model_names": ["pt_qwen_v2_0_5b_instruct_clm_hf"], "pcc": 0.99, "args": {"dim": "-1", "keep_dim": "True"}},
    ),
    (
        Reduceavg0,
        [((1, 128, 56, 56), torch.float32)],
        {
            "model_names": ["regnet_regnety_040_onnx", "onnx_regnet_facebook_regnet_y_040_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 128, 1, 56), torch.float32)],
        {
            "model_names": ["regnet_regnety_040_onnx", "onnx_regnet_facebook_regnet_y_040_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 192, 28, 28), torch.float32)],
        {
            "model_names": ["regnet_regnety_040_onnx", "onnx_regnet_facebook_regnet_y_040_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 192, 1, 28), torch.float32)],
        {
            "model_names": ["regnet_regnety_040_onnx", "onnx_regnet_facebook_regnet_y_040_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 512, 14, 14), torch.float32)],
        {
            "model_names": ["regnet_regnety_040_onnx", "onnx_regnet_facebook_regnet_y_040_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 512, 1, 14), torch.float32)],
        {
            "model_names": ["regnet_regnety_040_onnx", "onnx_regnet_facebook_regnet_y_040_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((1, 1088, 7, 7), torch.float32)],
        {
            "model_names": ["regnet_regnety_040_onnx", "onnx_regnet_facebook_regnet_y_040_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 1088, 1, 7), torch.float32)],
        {
            "model_names": ["regnet_regnety_040_onnx", "onnx_regnet_facebook_regnet_y_040_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg2,
        [((1, 7, 7, 2048), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim": "-3", "keep_dim": "True"}},
    ),
    (
        Reduceavg0,
        [((1, 1, 7, 2048), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim": "-2", "keep_dim": "True"}},
    ),
    (
        Reduceavg1,
        [((1, 596, 4096), torch.float32)],
        {"model_names": ["pt_llava_1_5_7b_cond_gen_hf"], "pcc": 0.99, "args": {"dim": "-1", "keep_dim": "True"}},
    ),
    (
        Reduceavg1,
        [((1, 12, 4096), torch.float32)],
        {
            "model_names": [
                "pt_ministral_ministral_3b_instruct_clm_hf",
                "pt_ministral_ministral_8b_instruct_clm_hf",
                "pt_mistral_7b_clm_hf",
                "pt_mistral_7b_instruct_v03_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 256, 3072), torch.float32)],
        {
            "model_names": [
                "pt_phi3_5_mini_instruct_clm_hf",
                "pt_llama3_llama_3_2_3b_clm_hf",
                "pt_llama3_llama_3_2_3b_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((2, 32, 10, 16384), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((2, 32, 1, 16384), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((2, 32, 10, 4096), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((2, 32, 1, 4096), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((2, 32, 20, 4096), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((2, 32, 20, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((2, 32, 1, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((2, 32, 40, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((2, 32, 80, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((2, 32, 60, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((2, 32, 60, 4096), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((2, 32, 40, 4096), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((2, 32, 30, 4096), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((2, 32, 30, 16384), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((2, 32, 20, 16384), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((100, 8, 9240), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((100, 8, 4480), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((100, 8, 8640), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((100, 8, 17280), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((100, 8, 34240), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((100, 8, 33, 850), torch.bfloat16)],
        {
            "model_names": ["pt_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((100, 8, 1, 850), torch.bfloat16)],
        {
            "model_names": ["pt_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((100, 8, 16, 850), torch.bfloat16)],
        {
            "model_names": ["pt_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((100, 8, 8, 3350), torch.bfloat16)],
        {
            "model_names": ["pt_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((100, 8, 1, 3350), torch.bfloat16)],
        {
            "model_names": ["pt_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((100, 8, 4, 13400), torch.bfloat16)],
        {
            "model_names": ["pt_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((100, 8, 1, 13400), torch.bfloat16)],
        {
            "model_names": ["pt_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg0,
        [((100, 8, 2, 53400), torch.bfloat16)],
        {
            "model_names": ["pt_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((100, 8, 1, 53400), torch.bfloat16)],
        {
            "model_names": ["pt_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 256, 2048), torch.float32)],
        {
            "model_names": ["pt_llama3_llama_3_2_1b_clm_hf", "pt_llama3_llama_3_2_1b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 1063, 4096), torch.float32)],
        {"model_names": ["pt_deepseek_7b_instruct_qa_hf"], "pcc": 0.99, "args": {"dim": "-1", "keep_dim": "True"}},
    ),
    (
        Reduceavg1,
        [((1, 522, 3072), torch.float32)],
        {
            "model_names": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 356, 2048), torch.float32)],
        {
            "model_names": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 356, 3072), torch.float32)],
        {
            "model_names": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 512, 2048), torch.float32)],
        {
            "model_names": ["pt_gemma_google_gemma_2b_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 256, 4096), torch.float32)],
        {
            "model_names": [
                "pt_llama3_huggyllama_7b_clm_hf",
                "pt_llama3_llama_3_1_8b_clm_hf",
                "pt_llama3_llama_3_1_8b_instruct_clm_hf",
                "pt_llama3_llama_3_8b_clm_hf",
                "pt_llama3_llama_3_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 4, 4096), torch.float32)],
        {
            "model_names": [
                "pt_llama3_huggyllama_7b_seq_cls_hf",
                "pt_llama3_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 4, 3072), torch.float32)],
        {
            "model_names": ["pt_llama3_llama_3_2_3b_instruct_seq_cls_hf", "pt_llama3_llama_3_2_3b_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 5, 3072), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 13, 3072), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_token_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 5, 5120), torch.float32)],
        {"model_names": ["pt_phi4_microsoft_phi_4_seq_cls_hf"], "pcc": 0.99, "args": {"dim": "-1", "keep_dim": "True"}},
    ),
    (
        Reduceavg1,
        [((1, 12, 5120), torch.float32)],
        {
            "model_names": ["pt_phi4_microsoft_phi_4_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 13, 3584), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((1, 128, 2560), torch.float32)],
        {"model_names": ["pt_qwen_v3_4b_clm_hf"], "pcc": 0.99, "args": {"dim": "-1", "keep_dim": "True"}},
    ),
    (
        Reduceavg1,
        [((1, 128, 32, 128), torch.float32)],
        {"model_names": ["pt_qwen_v3_4b_clm_hf"], "pcc": 0.99, "args": {"dim": "-1", "keep_dim": "True"}},
    ),
    (
        Reduceavg1,
        [((4, 31, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_qwen_v3_embedding_0_6b_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reduceavg1,
        [((4, 31, 16, 128), torch.bfloat16)],
        {
            "model_names": ["pt_qwen_v3_embedding_0_6b_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
@pytest.mark.parametrize("training_test", [False, True], ids=["inference", "training"])
def test_module(forge_module_and_shapes_dtypes, training_test):

    record_forge_op_name("ReduceAvg")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes

    pcc = metadata.get("pcc")

    for metadata_name, metadata_value in metadata.items():
        if metadata_name in ["pcc"]:
            continue
        elif metadata_name == "model_names":
            record_op_model_names(metadata_value)
        elif metadata_name == "args":
            record_forge_op_args(metadata_value)
        else:
            logger.warning(
                "No utility function available in forge property handler to record %s property", metadata_name
            )

    max_int = 1000
    inputs = [
        Tensor.create_from_shape(operand_shape, operand_dtype, max_int=max_int, requires_grad=training_test)
        for operand_shape, operand_dtype in operand_shapes_dtypes
    ]

    framework_model = forge_module(forge_module.__name__)

    for name, parameter in framework_model._parameters.items():
        parameter_tensor = Tensor.create_torch_tensor(
            shape=parameter.shape.get_pytorch_shape(),
            dtype=parameter.pt_data_format,
            max_int=max_int,
            requires_grad=training_test,
        )
        framework_model.set_parameter(name, parameter_tensor)

    for name, constant in framework_model._constants.items():
        constant_tensor = Tensor.create_torch_tensor(
            shape=constant.shape.get_pytorch_shape(),
            dtype=constant.pt_data_format,
            max_int=max_int,
            requires_grad=training_test,
        )
        framework_model.set_constant(name, constant_tensor)

    record_single_op_operands_info(framework_model, inputs)

    compiler_cfg = forge.config.CompilerConfig()
    if "default_df_override" in metadata.keys():
        compiler_cfg.default_df_override = forge.DataFormat.from_json(metadata["default_df_override"])

    compiled_model = compile(framework_model, sample_inputs=inputs, compiler_cfg=compiler_cfg, training=training_test)

    verify(
        inputs,
        framework_model,
        compiled_model,
        with_backward=training_test,
        verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
    )
