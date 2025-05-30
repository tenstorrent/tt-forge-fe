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
        clip_output_1 = forge.op.Clip("", clip_input_0, min=-3.0, max=3.0)
        return clip_output_1


class Clip1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, clip_input_0):
        clip_output_1 = forge.op.Clip("", clip_input_0, min=0.0, max=1.0)
        return clip_output_1


class Clip2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, clip_input_0):
        clip_output_1 = forge.op.Clip("", clip_input_0, min=0.0, max=6.0)
        return clip_output_1


class Clip3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, clip_input_0):
        clip_output_1 = forge.op.Clip("", clip_input_0, min=9.999999960041972e-13, max=3.4028234663852886e38)
        return clip_output_1


class Clip4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, clip_input_0):
        clip_output_1 = forge.op.Clip("", clip_input_0, min=-3.4028234663852886e38, max=4.605170249938965)
        return clip_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Clip0,
        [((1, 16, 240, 240), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99, "args": {"min": "-3.0", "max": "3.0"}},
    ),
    (
        Clip0,
        [((1, 32, 240, 240), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99, "args": {"min": "-3.0", "max": "3.0"}},
    ),
    (
        Clip0,
        [((1, 48, 120, 120), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99, "args": {"min": "-3.0", "max": "3.0"}},
    ),
    (
        Clip0,
        [((1, 96, 60, 60), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99, "args": {"min": "-3.0", "max": "3.0"}},
    ),
    (
        Clip0,
        [((1, 192, 30, 30), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99, "args": {"min": "-3.0", "max": "3.0"}},
    ),
    (
        Clip1,
        [((1, 192, 1, 1), torch.float32)],
        {
            "model_names": ["TranslatedLayer", "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip0,
        [((1, 384, 15, 15), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99, "args": {"min": "-3.0", "max": "3.0"}},
    ),
    (
        Clip1,
        [((1, 384, 1, 1), torch.float32)],
        {
            "model_names": ["TranslatedLayer", "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip1,
        [((1, 96, 1, 1), torch.float32)],
        {
            "model_names": ["TranslatedLayer", "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip1,
        [((1, 24, 1, 1), torch.float32)],
        {
            "model_names": ["TranslatedLayer", "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip0,
        [((1, 16, 16, 50), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"min": "-3.0", "max": "3.0"},
        },
    ),
    (
        Clip0,
        [((1, 32, 16, 50), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"min": "-3.0", "max": "3.0"},
        },
    ),
    (
        Clip0,
        [((1, 64, 16, 50), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"min": "-3.0", "max": "3.0"},
        },
    ),
    (
        Clip0,
        [((1, 64, 8, 50), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"min": "-3.0", "max": "3.0"},
        },
    ),
    (
        Clip0,
        [((1, 128, 8, 50), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"min": "-3.0", "max": "3.0"},
        },
    ),
    (
        Clip0,
        [((1, 128, 8, 25), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"min": "-3.0", "max": "3.0"},
        },
    ),
    (
        Clip0,
        [((1, 240, 8, 25), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"min": "-3.0", "max": "3.0"},
        },
    ),
    (
        Clip0,
        [((1, 240, 4, 25), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"min": "-3.0", "max": "3.0"},
        },
    ),
    (
        Clip1,
        [((1, 240, 1, 1), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip0,
        [((1, 480, 4, 25), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"min": "-3.0", "max": "3.0"},
        },
    ),
    (
        Clip1,
        [((1, 480, 1, 1), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip0,
        [((1, 480, 2, 25), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"min": "-3.0", "max": "3.0"},
        },
    ),
    (
        Clip1,
        [((1, 1, 256, 256), torch.float32)],
        {
            "model_names": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_phi2_microsoft_phi_2_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip1,
        [((1, 12, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip1,
        [((1, 1, 32, 32), torch.float32)],
        {
            "model_names": [
                "pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_bloom_bigscience_bloom_1b1_clm_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip2,
        [((1, 16, 112, 112), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip2,
        [((1, 48, 112, 112), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip0,
        [((1, 48, 112, 112), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {"min": "-3.0", "max": "3.0"},
        },
    ),
    (
        Clip2,
        [((1, 48, 56, 56), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip2,
        [((1, 96, 56, 56), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip0,
        [((1, 96, 56, 56), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {"min": "-3.0", "max": "3.0"},
        },
    ),
    (
        Clip2,
        [((1, 96, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip2,
        [((1, 96, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip2,
        [((1, 192, 14, 14), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip2,
        [((1, 288, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip2,
        [((1, 288, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip2,
        [((1, 480, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip2,
        [((1, 1280, 7, 7), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip0,
        [((1, 16, 224, 224), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {"min": "-3.0", "max": "3.0"},
        },
    ),
    (
        Clip0,
        [((1, 32, 224, 224), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {"min": "-3.0", "max": "3.0"},
        },
    ),
    (
        Clip0,
        [((1, 192, 28, 28), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {"min": "-3.0", "max": "3.0"},
        },
    ),
    (
        Clip2,
        [((1, 192, 28, 28), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip0,
        [((1, 384, 14, 14), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {"min": "-3.0", "max": "3.0"},
        },
    ),
    (
        Clip2,
        [((1, 384, 14, 14), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip1,
        [((2, 1, 7, 7), torch.float32)],
        {
            "model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip1,
        [((1, 12, 384, 384), torch.float32)],
        {
            "model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip1,
        [((1, 1, 7, 7), torch.float32)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_clm_hf"], "pcc": 0.99, "args": {"min": "0.0", "max": "1.0"}},
    ),
    (
        Clip2,
        [((1, 32, 112, 112), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip2,
        [((1, 96, 112, 112), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip2,
        [((1, 144, 56, 56), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip2,
        [((1, 144, 28, 28), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip2,
        [((1, 576, 14, 14), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip2,
        [((1, 576, 7, 7), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip2,
        [((1, 960, 7, 7), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip0,
        [((1, 8, 16, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"min": "-3.0", "max": "3.0"},
        },
    ),
    (
        Clip1,
        [((1, 8, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip0,
        [((1, 48, 8, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"min": "-3.0", "max": "3.0"},
        },
    ),
    (
        Clip0,
        [((1, 48, 4, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"min": "-3.0", "max": "3.0"},
        },
    ),
    (
        Clip1,
        [((1, 48, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip0,
        [((1, 120, 4, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"min": "-3.0", "max": "3.0"},
        },
    ),
    (
        Clip1,
        [((1, 120, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip0,
        [((1, 64, 4, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"min": "-3.0", "max": "3.0"},
        },
    ),
    (
        Clip1,
        [((1, 64, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip0,
        [((1, 72, 4, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"min": "-3.0", "max": "3.0"},
        },
    ),
    (
        Clip1,
        [((1, 72, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip0,
        [((1, 144, 4, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"min": "-3.0", "max": "3.0"},
        },
    ),
    (
        Clip0,
        [((1, 144, 2, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"min": "-3.0", "max": "3.0"},
        },
    ),
    (
        Clip1,
        [((1, 144, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip0,
        [((1, 288, 2, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"min": "-3.0", "max": "3.0"},
        },
    ),
    (
        Clip1,
        [((1, 288, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
        },
    ),
    (
        Clip2,
        [((1, 432, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip2,
        [((1, 624, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip2,
        [((1, 624, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip2,
        [((1, 1056, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "6.0"},
        },
    ),
    (
        Clip3,
        [((64, 3, 64, 1), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"min": "9.999999960041972e-13", "max": "3.4028234663852886e+38"},
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
        Clip3,
        [((16, 6, 64, 1), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"min": "9.999999960041972e-13", "max": "3.4028234663852886e+38"},
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
        Clip3,
        [((4, 12, 64, 1), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"min": "9.999999960041972e-13", "max": "3.4028234663852886e+38"},
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
        Clip3,
        [((1, 24, 64, 1), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"min": "9.999999960041972e-13", "max": "3.4028234663852886e+38"},
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
        Clip1,
        [((2, 1, 1, 13), torch.float32)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"min": "0.0", "max": "1.0"},
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

    verify(inputs, framework_model, compiled_model, VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)))
