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
        clip_output_1 = forge.op.Clip("", clip_input_0, min=9.999999960041972e-13, max=3.4028234663852886e38)
        return clip_output_1


class Clip3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, clip_input_0):
        clip_output_1 = forge.op.Clip("", clip_input_0, min=1e-12, max=3.4028234663852886e38)
        return clip_output_1


class Clip4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, clip_input_0):
        clip_output_1 = forge.op.Clip("", clip_input_0, min=-3.4028234663852886e38, max=4.605170249938965)
        return clip_output_1


class Clip5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, clip_input_0):
        clip_output_1 = forge.op.Clip("", clip_input_0, min=-3.4028234663852886e38, max=4.605170185988092)
        return clip_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Clip0,
        [((1, 1, 256, 256), torch.float32)],
        {
            "model_names": [
                "onnx_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "onnx_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf",
                "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
                "pt_phi2_microsoft_phi_2_clm_hf",
                "pt_phi2_microsoft_phi_2_pytdml_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
                "pt_xglm_facebook_xglm_1_7b_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip1,
        [((1, 32, 112, 112), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_lite0_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 96, 112, 112), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_lite0_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 96, 56, 56), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_lite0_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 144, 56, 56), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_lite0_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_140_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 144, 28, 28), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_lite0_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 240, 28, 28), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_lite0_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 240, 14, 14), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_lite0_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 480, 14, 14), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_lite0_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 672, 14, 14), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_lite0_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 672, 7, 7), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_lite0_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 1152, 7, 7), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_lite0_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 1280, 7, 7), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_lite0_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 48, 112, 112), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_140_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 144, 112, 112), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_140_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 192, 56, 56), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_140_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 192, 28, 28), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_140_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 288, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_140_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 288, 14, 14), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_140_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 528, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_140_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 816, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_140_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 816, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_140_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 1344, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_140_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 1792, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_140_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 16, 112, 112), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 48, 56, 56), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 96, 28, 28), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 96, 14, 14), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 192, 14, 14), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 288, 7, 7), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 480, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 384, 14, 14), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 576, 14, 14), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 576, 7, 7), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 960, 7, 7), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
            ],
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
        Clip2,
        [((64, 3, 64, 1), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"min": "9.999999960041972e-13", "max": "3.4028234663852886e+38"},
        },
    ),
    (
        Clip3,
        [((64, 3, 64, 1), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"min": "1e-12", "max": "3.4028234663852886e+38"},
        },
    ),
    (
        Clip4,
        [((3, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"min": "-3.4028234663852886e+38", "max": "4.605170249938965"},
        },
    ),
    (
        Clip5,
        [((3, 1, 1), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"min": "-3.4028234663852886e+38", "max": "4.605170185988092"},
        },
    ),
    (
        Clip2,
        [((16, 6, 64, 1), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"min": "9.999999960041972e-13", "max": "3.4028234663852886e+38"},
        },
    ),
    (
        Clip3,
        [((16, 6, 64, 1), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"min": "1e-12", "max": "3.4028234663852886e+38"},
        },
    ),
    (
        Clip4,
        [((6, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"min": "-3.4028234663852886e+38", "max": "4.605170249938965"},
        },
    ),
    (
        Clip5,
        [((6, 1, 1), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"min": "-3.4028234663852886e+38", "max": "4.605170185988092"},
        },
    ),
    (
        Clip2,
        [((4, 12, 64, 1), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"min": "9.999999960041972e-13", "max": "3.4028234663852886e+38"},
        },
    ),
    (
        Clip3,
        [((4, 12, 64, 1), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"min": "1e-12", "max": "3.4028234663852886e+38"},
        },
    ),
    (
        Clip4,
        [((12, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"min": "-3.4028234663852886e+38", "max": "4.605170249938965"},
        },
    ),
    (
        Clip5,
        [((12, 1, 1), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"min": "-3.4028234663852886e+38", "max": "4.605170185988092"},
        },
    ),
    (
        Clip2,
        [((1, 24, 64, 1), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"min": "9.999999960041972e-13", "max": "3.4028234663852886e+38"},
        },
    ),
    (
        Clip3,
        [((1, 24, 64, 1), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"min": "1e-12", "max": "3.4028234663852886e+38"},
        },
    ),
    (
        Clip4,
        [((24, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"min": "-3.4028234663852886e+38", "max": "4.605170249938965"},
        },
    ),
    (
        Clip5,
        [((24, 1, 1), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"min": "-3.4028234663852886e+38", "max": "4.605170185988092"},
        },
    ),
    (
        Clip0,
        [((1, 256, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip1,
        [((1, 256, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip0,
        [((1, 512, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip1,
        [((1, 512, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip0,
        [((1, 768, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip1,
        [((1, 768, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip0,
        [((1, 1024, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip1,
        [((1, 1024, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip0,
        [((2, 1, 1, 13), torch.float32)],
        {
            "model_names": [
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip0,
        [((2, 1, 7, 7), torch.float32)],
        {
            "model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip0,
        [((1, 1, 32, 32), torch.float32)],
        {
            "model_names": [
                "pt_bloom_bigscience_bloom_1b1_clm_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf",
                "pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf",
                "pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf",
                "pt_llama3_huggyllama_llama_7b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_clm_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip0,
        [((1, 12, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip0,
        [((1, 12, 384, 384), torch.float32)],
        {
            "model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip1,
        [((1, 32, 150, 150), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 144, 150, 150), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 144, 75, 75), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 192, 75, 75), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 192, 38, 38), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 288, 38, 38), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 288, 19, 19), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 576, 19, 19), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 816, 19, 19), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 816, 10, 10), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 1392, 10, 10), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 1280, 10, 10), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 32, 190, 190), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 144, 190, 190), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 144, 95, 95), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 192, 95, 95), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 192, 48, 48), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 336, 48, 48), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 336, 24, 24), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 672, 24, 24), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 960, 24, 24), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 960, 12, 12), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 1632, 12, 12), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 1280, 12, 12), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 32, 130, 130), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 96, 130, 130), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 96, 65, 65), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 144, 65, 65), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 144, 33, 33), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 288, 33, 33), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 288, 17, 17), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 528, 17, 17), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 720, 17, 17), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 720, 9, 9), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 1248, 9, 9), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 1280, 9, 9), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 32, 120, 120), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 96, 120, 120), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 96, 60, 60), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 144, 60, 60), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 144, 30, 30), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 240, 30, 30), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 240, 15, 15), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 480, 15, 15), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 672, 15, 15), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 672, 8, 8), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 1152, 8, 8), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 1280, 8, 8), torch.float32)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 72, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 120, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 480, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 672, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 960, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 64, 112, 112), torch.float32)],
        {
            "model_names": [
                "pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 64, 56, 56), torch.float32)],
        {
            "model_names": [
                "pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 128, 56, 56), torch.float32)],
        {
            "model_names": [
                "pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 128, 28, 28), torch.float32)],
        {
            "model_names": [
                "pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 256, 28, 28), torch.float32)],
        {
            "model_names": [
                "pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 256, 14, 14), torch.float32)],
        {
            "model_names": [
                "pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 512, 14, 14), torch.float32)],
        {
            "model_names": [
                "pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 512, 7, 7), torch.float32)],
        {
            "model_names": [
                "pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 1024, 7, 7), torch.float32)],
        {
            "model_names": [
                "pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 24, 96, 96), torch.float32)],
        {
            "model_names": ["pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 48, 96, 96), torch.float32)],
        {
            "model_names": ["pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 48, 48, 48), torch.float32)],
        {
            "model_names": [
                "pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 96, 48, 48), torch.float32)],
        {
            "model_names": ["pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 96, 24, 24), torch.float32)],
        {
            "model_names": ["pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 192, 24, 24), torch.float32)],
        {
            "model_names": ["pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 192, 12, 12), torch.float32)],
        {
            "model_names": ["pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 384, 12, 12), torch.float32)],
        {
            "model_names": ["pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 384, 6, 6), torch.float32)],
        {
            "model_names": ["pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 768, 6, 6), torch.float32)],
        {
            "model_names": ["pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 24, 80, 80), torch.float32)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 96, 80, 80), torch.float32)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 96, 40, 40), torch.float32)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 144, 40, 40), torch.float32)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 144, 20, 20), torch.float32)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 144, 10, 10), torch.float32)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 288, 10, 10), torch.float32)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 432, 10, 10), torch.float32)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 432, 5, 5), torch.float32)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 720, 5, 5), torch.float32)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 1280, 5, 5), torch.float32)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 16, 48, 48), torch.float32)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 48, 24, 24), torch.float32)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 48, 12, 12), torch.float32)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 96, 12, 12), torch.float32)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 96, 6, 6), torch.float32)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 144, 6, 6), torch.float32)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 192, 6, 6), torch.float32)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 192, 3, 3), torch.float32)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 336, 3, 3), torch.float32)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 1280, 3, 3), torch.float32)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 384, 28, 28), torch.float32)],
        {
            "model_names": ["pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 576, 28, 28), torch.float32)],
        {
            "model_names": ["pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 960, 28, 28), torch.float32)],
        {
            "model_names": ["pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 16, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 96, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 240, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 120, 14, 14), torch.float32)],
        {
            "model_names": [
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 144, 14, 14), torch.float32)],
        {
            "model_names": [
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 144, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 288, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 576, 1, 1), torch.float32)],
        {
            "model_names": [
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 200, 14, 14), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 184, 14, 14), torch.float32)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 1280), torch.float32)],
        {
            "model_names": ["pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 1280, 1, 1), torch.float32)],
        {
            "model_names": ["pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((1, 1024), torch.float32)],
        {
            "model_names": ["pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip3,
        [((64, 4, 64, 1), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"min": "1e-12", "max": "3.4028234663852886e+38"},
        },
    ),
    (
        Clip5,
        [((4, 1, 1), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"min": "-3.4028234663852886e+38", "max": "4.605170185988092"},
        },
    ),
    (
        Clip3,
        [((16, 8, 64, 1), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"min": "1e-12", "max": "3.4028234663852886e+38"},
        },
    ),
    (
        Clip5,
        [((8, 1, 1), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"min": "-3.4028234663852886e+38", "max": "4.605170185988092"},
        },
    ),
    (
        Clip3,
        [((4, 16, 64, 1), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"min": "1e-12", "max": "3.4028234663852886e+38"},
        },
    ),
    (
        Clip5,
        [((16, 1, 1), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"min": "-3.4028234663852886e+38", "max": "4.605170185988092"},
        },
    ),
    (
        Clip3,
        [((1, 32, 64, 1), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"min": "1e-12", "max": "3.4028234663852886e+38"},
        },
    ),
    (
        Clip5,
        [((32, 1, 1), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {"min": "-3.4028234663852886e+38", "max": "4.605170185988092"},
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

    record_single_op_operands_info(framework_model, inputs)

    compiled_model = compile(framework_model, sample_inputs=inputs)

    verify(
        inputs,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
    )
