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
    (Sigmoid0, [((1, 1, 480, 480), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (
        Sigmoid0,
        [((1, 100, 4), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 40, 144, 144), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 10, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 40, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 24, 144, 144), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 6, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 24, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 144, 144, 144), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 144, 72, 72), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 144, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 192, 72, 72), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 8, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 192, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 192, 36, 36), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 288, 36, 36), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 12, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 288, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 288, 18, 18), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 576, 18, 18), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 576, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 816, 18, 18), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 34, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 816, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 816, 9, 9), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 1392, 9, 9), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 58, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 1392, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 2304, 9, 9), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 96, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 2304, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 1536, 9, 9), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 60, 1, 12), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 120, 1, 12), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 12, 240), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 480, 1, 12), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 256, 8192), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_clm_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 32, 112, 112), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 32, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 96, 112, 112), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 96, 56, 56), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 4, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 144, 56, 56), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 144, 28, 28), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 240, 28, 28), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 240, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 240, 14, 14), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 480, 14, 14), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 20, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 480, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 672, 14, 14), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 28, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 672, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 672, 7, 7), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1152, 7, 7), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 48, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 1152, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 1280, 7, 7), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1, 448, 448), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((2, 7, 2048), torch.float32)],
        {"model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 4, 8192), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 32, 120, 120), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 16, 120, 120), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 16, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 96, 120, 120), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 96, 60, 60), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 144, 60, 60), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 144, 30, 30), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 240, 30, 30), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 240, 15, 15), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 480, 15, 15), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 672, 15, 15), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 672, 8, 8), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1152, 8, 8), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1920, 8, 8), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 80, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1920, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1280, 8, 8), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 48, 160, 160), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 24, 160, 160), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 144, 160, 160), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 144, 80, 80), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 192, 80, 80), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 192, 40, 40), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 336, 40, 40), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 14, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 336, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 336, 20, 20), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 672, 20, 20), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 960, 20, 20), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 960, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 960, 10, 10), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1632, 10, 10), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 68, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1632, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 2688, 10, 10), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 112, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 2688, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1792, 10, 10), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 16, 320, 320), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 32, 160, 160), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 16, 160, 160), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 64, 80, 80), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 32, 80, 80), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 128, 40, 40), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 64, 40, 40), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 256, 20, 20), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 128, 20, 20), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 80, 80, 80), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 80, 40, 40), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 64, 20, 20), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 80, 20, 20), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (Sigmoid0, [((1, 80, 8400), torch.float32)], {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99}),
    (
        Sigmoid0,
        [((1, 48), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 32, 128, 128), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 16, 128, 128), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 96, 128, 128), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 96, 64, 64), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 144, 64, 64), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 144, 32, 32), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 288, 32, 32), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 288, 16, 16), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 528, 16, 16), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 22, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 528, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 720, 16, 16), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 30, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 720, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 720, 8, 8), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1248, 8, 8), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 52, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1248, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 2112, 8, 8), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 88, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 2112, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1408, 8, 8), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 48, 224, 224), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 24, 224, 224), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 144, 224, 224), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 144, 112, 112), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 240, 112, 112), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 240, 56, 56), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 384, 56, 56), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 384, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 384, 28, 28), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 768, 28, 28), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 768, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1056, 28, 28), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 44, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1056, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1056, 14, 14), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1824, 14, 14), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 76, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1824, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 3072, 14, 14), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 128, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 3072, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 2048, 14, 14), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((3, 64, 64), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((6, 64, 64), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((12, 64, 64), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((24, 64, 64), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 588, 5504), torch.float32)],
        {"model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 2048, 6), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"], "pcc": 0.99},
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    record_forge_op_name("Sigmoid")

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
