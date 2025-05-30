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


class Relu0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, relu_input_0):
        relu_output_1 = forge.op.Relu("", relu_input_0)
        return relu_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Relu0,
        [((1, 48, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer", "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 96, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer", "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 24, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer", "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 6, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer", "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (Relu0, [((1, 24, 120, 120), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (Relu0, [((1, 24, 240, 240), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (
        Relu0,
        [((1, 64, 214, 320), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 64, 107, 160), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 256, 107, 160), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 128, 107, 160), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 128, 54, 80), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 512, 54, 80), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 256, 54, 80), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 256, 27, 40), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1024, 27, 40), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 512, 27, 40), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 512, 14, 20), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 2048, 14, 20), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 280, 2048), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 100, 2048), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 100, 256), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 16, 224, 224), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla46x_c_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "onnx_dla_dla34_visual_bb_torchvision",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 32, 112, 112), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla46x_c_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "onnx_dla_dla34_visual_bb_torchvision",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 256, 112, 112), torch.float32)],
        {"model_names": ["onnx_dla_dla102x2_visual_bb_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 256, 56, 56), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 128, 56, 56), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_vovnet_vovnet27s_obj_det_osmr",
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 512, 56, 56), torch.float32)],
        {"model_names": ["onnx_dla_dla102x2_visual_bb_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 512, 28, 28), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 256, 28, 28), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_vovnet_vovnet27s_obj_det_osmr",
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1024, 28, 28), torch.float32)],
        {"model_names": ["onnx_dla_dla102x2_visual_bb_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1024, 14, 14), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 512, 14, 14), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 2048, 14, 14), torch.float32)],
        {"model_names": ["onnx_dla_dla102x2_visual_bb_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 2048, 7, 7), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1024, 7, 7), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 64, 112, 112), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla46x_c_visual_bb_torchvision",
                "onnx_vovnet_vovnet27s_obj_det_osmr",
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_dla_dla169_visual_bb_torchvision",
                "pd_resnet_18_img_cls_paddlemodels",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 64, 56, 56), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla46x_c_visual_bb_torchvision",
                "onnx_vovnet_vovnet27s_obj_det_osmr",
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_dla_dla60_visual_bb_torchvision",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_dla_dla169_visual_bb_torchvision",
                "pd_resnet_18_img_cls_paddlemodels",
                "onnx_dla_dla34_visual_bb_torchvision",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 64, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_dla_dla46x_c_visual_bb_torchvision", "onnx_dla_dla60x_c_visual_bb_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 128, 28, 28), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla46x_c_visual_bb_torchvision",
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_dla_dla60_visual_bb_torchvision",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_dla_dla169_visual_bb_torchvision",
                "pd_resnet_18_img_cls_paddlemodels",
                "onnx_dla_dla34_visual_bb_torchvision",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 128, 14, 14), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla46x_c_visual_bb_torchvision",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 256, 14, 14), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla46x_c_visual_bb_torchvision",
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_dla_dla60_visual_bb_torchvision",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_dla_dla169_visual_bb_torchvision",
                "pd_resnet_18_img_cls_paddlemodels",
                "onnx_dla_dla34_visual_bb_torchvision",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 256, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_dla_dla46x_c_visual_bb_torchvision", "onnx_dla_dla60x_c_visual_bb_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 768, 128, 128), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (Relu0, [((1, 80, 28, 28), torch.float32)], {"model_names": ["onnx_vovnet_vovnet27s_obj_det_osmr"], "pcc": 0.99}),
    (Relu0, [((1, 96, 14, 14), torch.float32)], {"model_names": ["onnx_vovnet_vovnet27s_obj_det_osmr"], "pcc": 0.99}),
    (
        Relu0,
        [((1, 384, 14, 14), torch.float32)],
        {"model_names": ["onnx_vovnet_vovnet27s_obj_det_osmr", "pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (Relu0, [((1, 112, 7, 7), torch.float32)], {"model_names": ["onnx_vovnet_vovnet27s_obj_det_osmr"], "pcc": 0.99}),
    (
        Relu0,
        [((1, 512, 7, 7), torch.float32)],
        {
            "model_names": [
                "onnx_vovnet_vovnet27s_obj_det_osmr",
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_dla_dla60_visual_bb_torchvision",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_dla_dla169_visual_bb_torchvision",
                "pd_resnet_18_img_cls_paddlemodels",
                "onnx_dla_dla34_visual_bb_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 9, 3072), torch.float32)],
        {"model_names": ["pd_ernie_1_0_mlm_padlenlp", "pd_ernie_1_0_qa_padlenlp"], "pcc": 0.99},
    ),
    (Relu0, [((1, 9, 768), torch.float32)], {"model_names": ["pd_ernie_1_0_mlm_padlenlp"], "pcc": 0.99}),
    (
        Relu0,
        [((1, 60, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 120, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (Relu0, [((32, 3072), torch.float32)], {"model_names": ["pt_opt_facebook_opt_125m_qa_hf"], "pcc": 0.99}),
    (
        Relu0,
        [((1, 128, 112, 112), torch.float32)],
        {
            "model_names": ["onnx_dla_dla102x_visual_bb_torchvision", "onnx_dla_dla60x_visual_bb_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 160, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet_v1_57_obj_det_torchhub", "pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 192, 14, 14), torch.float32)],
        {"model_names": ["onnx_vovnet_vovnet_v1_57_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 768, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet_v1_57_obj_det_torchhub", "pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 224, 7, 7), torch.float32)],
        {"model_names": ["onnx_vovnet_vovnet_v1_57_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 24, 112, 112), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 24, 224, 224), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (Relu0, [((256, 8192), torch.float32)], {"model_names": ["pt_opt_facebook_opt_1_3b_clm_hf"], "pcc": 0.99}),
    (Relu0, [((32, 4096), torch.float32)], {"model_names": ["pt_opt_facebook_opt_350m_qa_hf"], "pcc": 0.99}),
    (
        Relu0,
        [((100, 264, 14, 20), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((100, 128, 14, 20), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((100, 64, 27, 40), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((100, 32, 54, 80), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((100, 16, 107, 160), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 8, 16, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 2, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 40, 16, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 40, 8, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 48, 8, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 12, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 30, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 16, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 18, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 36, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 72, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 256, 128, 128), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 15, 15, 512), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (Relu0, [((1, 64, 55, 55), torch.float32)], {"model_names": ["pd_alexnet_base_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Relu0,
        [((1, 192, 27, 27), torch.float32)],
        {"model_names": ["pd_alexnet_base_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 384, 13, 13), torch.float32)],
        {"model_names": ["pd_alexnet_base_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 256, 13, 13), torch.float32)],
        {"model_names": ["pd_alexnet_base_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (Relu0, [((1, 4096), torch.float32)], {"model_names": ["pd_alexnet_base_img_cls_paddlemodels"], "pcc": 0.99}),
    (Relu0, [((1, 96, 56, 56), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Relu0,
        [((1, 160, 56, 56), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 192, 56, 56), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 224, 56, 56), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 192, 28, 28), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 224, 28, 28), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 288, 28, 28), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 320, 28, 28), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 352, 28, 28), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 384, 28, 28), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 416, 28, 28), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 448, 28, 28), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 480, 28, 28), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 288, 14, 14), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 320, 14, 14), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 352, 14, 14), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 416, 14, 14), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 448, 14, 14), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 480, 14, 14), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 544, 14, 14), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 576, 14, 14), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 608, 14, 14), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 640, 14, 14), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 672, 14, 14), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 704, 14, 14), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 736, 14, 14), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 800, 14, 14), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 832, 14, 14), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 864, 14, 14), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 896, 14, 14), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 928, 14, 14), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 960, 14, 14), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 992, 14, 14), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (Relu0, [((1, 128, 7, 7), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (Relu0, [((1, 544, 7, 7), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (Relu0, [((1, 576, 7, 7), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (Relu0, [((1, 608, 7, 7), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (Relu0, [((1, 640, 7, 7), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (Relu0, [((1, 672, 7, 7), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (Relu0, [((1, 704, 7, 7), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (Relu0, [((1, 736, 7, 7), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (Relu0, [((1, 768, 7, 7), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (Relu0, [((1, 800, 7, 7), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (Relu0, [((1, 832, 7, 7), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (Relu0, [((1, 864, 7, 7), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (Relu0, [((1, 896, 7, 7), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (Relu0, [((1, 928, 7, 7), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (Relu0, [((1, 960, 7, 7), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (Relu0, [((1, 992, 7, 7), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Relu0,
        [((2, 13, 3072), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    record_forge_op_name("Relu")

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
