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


class Clip2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, clip_input_0):
        clip_output_1 = forge.op.Clip("", clip_input_0, min=1e-12, max=3.4028234663852886e38)
        return clip_output_1


class Clip3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, clip_input_0):
        clip_output_1 = forge.op.Clip("", clip_input_0, min=-3.4028234663852886e38, max=4.605170185988092)
        return clip_output_1


class Clip4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, clip_input_0):
        clip_output_1 = forge.op.Clip("", clip_input_0, min=-3.0, max=3.0)
        return clip_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Clip0,
        [((1, 1, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip1,
        [((1, 32, 112, 112), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 64, 112, 112), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 64, 56, 56), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 128, 56, 56), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 128, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 256, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 256, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 512, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 512, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 1024, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip0,
        [((1, 1, 256, 256), torch.float32)],
        {
            "model_names": [
                "pt_phi1_microsoft_phi_1_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip1,
        [((1, 256, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 512, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 768, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 1024, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 72, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 120, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 480, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 672, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 960, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 24, 96, 96), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 48, 96, 96), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 48, 48, 48), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 96, 48, 48), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 96, 24, 24), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 192, 24, 24), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 192, 12, 12), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 384, 12, 12), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 384, 6, 6), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 768, 6, 6), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 24, 80, 80), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 96, 80, 80), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 96, 40, 40), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 144, 40, 40), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 144, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 144, 10, 10), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 288, 10, 10), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 432, 10, 10), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 432, 5, 5), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 720, 5, 5), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 1280, 5, 5), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 96, 112, 112), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 96, 56, 56), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 144, 56, 56), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 144, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 192, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 192, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_mobilenet_v2_img_cls_torchvision", "pt_mobilenetv2_basic_img_cls_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 384, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_mobilenet_v2_img_cls_torchvision", "pt_mobilenetv2_basic_img_cls_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 576, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_mobilenet_v2_img_cls_torchvision", "pt_mobilenetv2_basic_img_cls_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 576, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 960, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 1280, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_mobilenet_v2_img_cls_torchvision", "pt_mobilenetv2_basic_img_cls_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 16, 112, 112), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 240, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 240, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 200, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 184, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 480, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 672, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 672, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 1280, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip0,
        [((1, 1, 6, 6), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"min": "0.0", "max": "1.0"}},
    ),
    (
        Clip0,
        [((1, 1, 35, 35), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip0,
        [((1, 1, 29, 29), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf", "pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip2,
        [((64, 3, 64, 1), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"min": "1e-12", "max": "3.4028234663852886e+38"},
        },
    ),
    (
        Clip3,
        [((3, 1, 1), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"min": "-3.4028234663852886e+38", "max": "4.605170185988092"},
        },
    ),
    (
        Clip2,
        [((16, 6, 64, 1), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"min": "1e-12", "max": "3.4028234663852886e+38"},
        },
    ),
    (
        Clip3,
        [((6, 1, 1), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"min": "-3.4028234663852886e+38", "max": "4.605170185988092"},
        },
    ),
    (
        Clip2,
        [((4, 12, 64, 1), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"min": "1e-12", "max": "3.4028234663852886e+38"},
        },
    ),
    (
        Clip3,
        [((12, 1, 1), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"min": "-3.4028234663852886e+38", "max": "4.605170185988092"},
        },
    ),
    (
        Clip2,
        [((1, 24, 64, 1), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"min": "1e-12", "max": "3.4028234663852886e+38"},
        },
    ),
    (
        Clip3,
        [((24, 1, 1), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"min": "-3.4028234663852886e+38", "max": "4.605170185988092"},
        },
    ),
    (
        Clip1,
        [((1, 32, 150, 150), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 144, 150, 150), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 144, 75, 75), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 192, 75, 75), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 192, 38, 38), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 288, 38, 38), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 288, 19, 19), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 576, 19, 19), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 816, 19, 19), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 816, 10, 10), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 1392, 10, 10), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 1280, 10, 10), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 384, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 576, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 960, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 16, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 96, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 96, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 96, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 240, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 120, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 144, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 144, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 288, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 288, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 288, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 576, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip0,
        [((1, 16, 112, 112), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip0,
        [((1, 16, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip0,
        [((1, 96, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip0,
        [((1, 96, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip0,
        [((1, 96, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip0,
        [((1, 240, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip0,
        [((1, 240, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip0,
        [((1, 120, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip0,
        [((1, 120, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip0,
        [((1, 144, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip0,
        [((1, 144, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip0,
        [((1, 288, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip0,
        [((1, 288, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip0,
        [((1, 288, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip0,
        [((1, 576, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip0,
        [((1, 576, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip0,
        [((1, 1024), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip1,
        [((1, 32, 112, 112), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 96, 112, 112), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 96, 56, 56), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 144, 56, 56), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 144, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 192, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 192, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 432, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 624, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 624, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 1056, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 1280, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip4,
        [((1, 8, 16, 50), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"min": "-3.0", "max": "3.0"},
        },
    ),
    (
        Clip0,
        [((1, 8, 1, 1), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip4,
        [((1, 48, 8, 50), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"min": "-3.0", "max": "3.0"},
        },
    ),
    (
        Clip4,
        [((1, 48, 4, 50), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"min": "-3.0", "max": "3.0"},
        },
    ),
    (
        Clip0,
        [((1, 48, 1, 1), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip4,
        [((1, 120, 4, 50), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"min": "-3.0", "max": "3.0"},
        },
    ),
    (
        Clip4,
        [((1, 64, 4, 50), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"min": "-3.0", "max": "3.0"},
        },
    ),
    (
        Clip0,
        [((1, 64, 1, 1), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip4,
        [((1, 72, 4, 50), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"min": "-3.0", "max": "3.0"},
        },
    ),
    (
        Clip0,
        [((1, 72, 1, 1), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip4,
        [((1, 144, 4, 50), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"min": "-3.0", "max": "3.0"},
        },
    ),
    (
        Clip4,
        [((1, 144, 2, 50), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"min": "-3.0", "max": "3.0"},
        },
    ),
    (
        Clip4,
        [((1, 288, 2, 50), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"min": "-3.0", "max": "3.0"},
        },
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    record_forge_op_name("Clip")

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
