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
        [((1, 64, 214, 320), torch.float32)],
        {
            "model_name": [
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
            "model_name": [
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
            "model_name": [
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
            "model_name": [
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
            "model_name": [
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
            "model_name": [
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
            "model_name": [
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
            "model_name": [
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
            "model_name": [
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
            "model_name": [
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
            "model_name": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 2048, 14, 20), torch.float32)],
        {
            "model_name": [
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
            "model_name": [
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
            "model_name": [
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
            "model_name": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((100, 264, 14, 20), torch.float32)],
        {"model_name": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((100, 128, 14, 20), torch.float32)],
        {"model_name": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((100, 64, 27, 40), torch.float32)],
        {"model_name": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((100, 32, 54, 80), torch.float32)],
        {"model_name": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((100, 16, 107, 160), torch.float32)],
        {"model_name": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 64, 112, 112), torch.float32)],
        {
            "model_name": [
                "onnx_resnet_50_img_cls_hf",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_18_img_cls_paddlemodels",
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_resnet_50_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm",
                "pt_mobilenet_v1_basic_img_cls_torchvision",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_regnet_regnet_x_800mf_img_cls_torchvision",
                "pt_regnet_regnet_y_800mf_img_cls_torchvision",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_resnet_resnet18_img_cls_torchvision",
                "pt_resnet_resnet34_img_cls_torchvision",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_unet_cityscape_img_seg_osmr",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_vovnet_vovnet57_obj_det_osmr",
                "pt_vovnet_vovnet39_obj_det_osmr",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_vovnet27s_obj_det_osmr",
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 64, 56, 56), torch.float32)],
        {
            "model_name": [
                "onnx_resnet_50_img_cls_hf",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_18_img_cls_paddlemodels",
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_resnet_50_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_dla_dla34_in1k_img_cls_timm",
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm",
                "pt_mobilenet_v1_basic_img_cls_torchvision",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_monodle_base_obj_det_torchvision",
                "pt_rcnn_base_obj_det_torchvision_rect_0",
                "pt_regnet_regnet_x_800mf_img_cls_torchvision",
                "pt_regnet_regnet_y_800mf_img_cls_torchvision",
                "pt_regnet_regnet_x_400mf_img_cls_torchvision",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_resnet_resnet18_img_cls_torchvision",
                "pt_resnet_resnet34_img_cls_torchvision",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_50_img_cls_timm",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_vovnet27s_obj_det_osmr",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 256, 56, 56), torch.float32)],
        {
            "model_name": [
                "onnx_resnet_50_img_cls_hf",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_resnet_50_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm",
                "pt_monodle_base_obj_det_torchvision",
                "pt_regnet_regnet_x_16gf_img_cls_torchvision",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_unet_cityscape_img_seg_osmr",
                "pt_unet_carvana_base_img_seg_github",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_vgg_vgg13_img_cls_torchvision",
                "pt_vgg_vgg19_img_cls_torchvision",
                "pt_vgg_vgg19_obj_det_osmr",
                "pt_vgg_vgg11_obj_det_osmr",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vgg_vgg13_bn_img_cls_torchvision",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg13_obj_det_osmr",
                "pt_vgg_vgg16_bn_img_cls_torchvision",
                "pt_vgg_vgg11_bn_img_cls_torchvision",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vgg_vgg19_bn_obj_det_timm",
                "pt_vovnet_vovnet57_obj_det_osmr",
                "pt_vovnet_vovnet39_obj_det_osmr",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 128, 56, 56), torch.float32)],
        {
            "model_name": [
                "onnx_resnet_50_img_cls_hf",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_resnet_50_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm",
                "pt_mobilenet_v1_basic_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
                "pt_regnet_regnet_x_800mf_img_cls_torchvision",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_unet_cityscape_img_seg_osmr",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_vovnet_vovnet57_obj_det_osmr",
                "pt_vovnet_vovnet39_obj_det_osmr",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_vovnet27s_obj_det_osmr",
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 128, 28, 28), torch.float32)],
        {
            "model_name": [
                "onnx_resnet_50_img_cls_hf",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_18_img_cls_paddlemodels",
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_resnet_50_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_dla_dla34_in1k_img_cls_timm",
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_mobilenet_v1_basic_img_cls_torchvision",
                "pt_monodle_base_obj_det_torchvision",
                "pt_regnet_regnet_x_800mf_img_cls_torchvision",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_resnet_resnet18_img_cls_torchvision",
                "pt_resnet_resnet34_img_cls_torchvision",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_50_img_cls_timm",
                "pt_unet_qubvel_img_seg_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 512, 28, 28), torch.float32)],
        {
            "model_name": [
                "onnx_resnet_50_img_cls_hf",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_resnet_50_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
                "pt_regnet_regnet_x_16gf_img_cls_torchvision",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_unet_cityscape_img_seg_osmr",
                "pt_unet_carvana_base_img_seg_github",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_vgg_vgg13_img_cls_torchvision",
                "pt_vgg_vgg19_img_cls_torchvision",
                "pt_vgg_vgg19_obj_det_osmr",
                "pt_vgg_vgg11_obj_det_osmr",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vgg_vgg13_bn_img_cls_torchvision",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg13_obj_det_osmr",
                "pt_vgg_vgg16_bn_img_cls_torchvision",
                "pt_vgg_vgg11_bn_img_cls_torchvision",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vgg_vgg19_bn_obj_det_timm",
                "pt_vovnet_vovnet57_obj_det_osmr",
                "pt_vovnet_vovnet39_obj_det_osmr",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 256, 28, 28), torch.float32)],
        {
            "model_name": [
                "onnx_resnet_50_img_cls_hf",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_resnet_50_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm",
                "pt_mobilenet_v1_basic_img_cls_torchvision",
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_unet_cityscape_img_seg_osmr",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_vovnet_vovnet27s_obj_det_osmr",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 256, 14, 14), torch.float32)],
        {
            "model_name": [
                "onnx_resnet_50_img_cls_hf",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_18_img_cls_paddlemodels",
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_resnet_50_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_dla_dla34_in1k_img_cls_timm",
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_mobilenet_v1_basic_img_cls_torchvision",
                "pt_monodle_base_obj_det_torchvision",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_resnet_resnet18_img_cls_torchvision",
                "pt_resnet_resnet34_img_cls_torchvision",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_50_img_cls_timm",
                "pt_unet_qubvel_img_seg_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1024, 14, 14), torch.float32)],
        {
            "model_name": [
                "onnx_resnet_50_img_cls_hf",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_resnet_50_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_unet_carvana_base_img_seg_github",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 512, 14, 14), torch.float32)],
        {
            "model_name": [
                "onnx_resnet_50_img_cls_hf",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_resnet_50_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm",
                "pt_mobilenet_v1_basic_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_unet_cityscape_img_seg_osmr",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_vgg_vgg13_img_cls_torchvision",
                "pt_vgg_vgg19_img_cls_torchvision",
                "pt_vgg_vgg19_obj_det_osmr",
                "pt_vgg_vgg11_obj_det_osmr",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vgg_vgg13_bn_img_cls_torchvision",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg13_obj_det_osmr",
                "pt_vgg_vgg16_bn_img_cls_torchvision",
                "pt_vgg_vgg11_bn_img_cls_torchvision",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vgg_vgg19_bn_obj_det_timm",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 512, 7, 7), torch.float32)],
        {
            "model_name": [
                "onnx_resnet_50_img_cls_hf",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_18_img_cls_paddlemodels",
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_resnet_50_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_dla_dla34_in1k_img_cls_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_mobilenet_v1_basic_img_cls_torchvision",
                "pt_monodle_base_obj_det_torchvision",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_resnet_resnet18_img_cls_torchvision",
                "pt_resnet_resnet34_img_cls_torchvision",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_50_img_cls_timm",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_vovnet_vovnet27s_obj_det_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 2048, 7, 7), torch.float32)],
        {
            "model_name": [
                "onnx_resnet_50_img_cls_hf",
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_resnet_50_img_cls_paddlemodels",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm",
                "pt_regnet_regnet_x_16gf_img_cls_torchvision",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 64, 55, 55), torch.float32)],
        {"model_name": ["pd_alexnet_base_img_cls_paddlemodels", "pt_alexnet_alexnet_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 192, 27, 27), torch.float32)],
        {
            "model_name": [
                "pd_alexnet_base_img_cls_paddlemodels",
                "pt_alexnet_alexnet_img_cls_torchhub",
                "pt_rcnn_base_obj_det_torchvision_rect_0",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 384, 13, 13), torch.float32)],
        {
            "model_name": [
                "pd_alexnet_base_img_cls_paddlemodels",
                "pt_alexnet_alexnet_img_cls_torchhub",
                "pt_alexnet_base_img_cls_osmr",
                "pt_rcnn_base_obj_det_torchvision_rect_0",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 256, 13, 13), torch.float32)],
        {
            "model_name": [
                "pd_alexnet_base_img_cls_paddlemodels",
                "pt_alexnet_alexnet_img_cls_torchhub",
                "pt_alexnet_base_img_cls_osmr",
                "pt_rcnn_base_obj_det_torchvision_rect_0",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 4096), torch.float32)],
        {
            "model_name": [
                "pd_alexnet_base_img_cls_paddlemodels",
                "pt_alexnet_alexnet_img_cls_torchhub",
                "pt_alexnet_base_img_cls_osmr",
                "pt_rcnn_base_obj_det_torchvision_rect_0",
                "pt_vgg_vgg13_img_cls_torchvision",
                "pt_vgg_vgg19_img_cls_torchvision",
                "pt_vgg_vgg19_obj_det_osmr",
                "pt_vgg_vgg11_obj_det_osmr",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vgg_vgg13_bn_img_cls_torchvision",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg13_obj_det_osmr",
                "pt_vgg_vgg16_bn_img_cls_torchvision",
                "pt_vgg_vgg11_bn_img_cls_torchvision",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg16_obj_det_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 96, 56, 56), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 160, 56, 56), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 192, 56, 56), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 224, 56, 56), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
                "pt_regnet_regnet_y_16gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 160, 28, 28), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_regnet_regnet_x_400mf_img_cls_torchvision",
                "pt_vovnet_vovnet57_obj_det_osmr",
                "pt_vovnet_vovnet39_obj_det_osmr",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 192, 28, 28), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 224, 28, 28), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 288, 28, 28), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_regnet_regnet_x_800mf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 320, 28, 28), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_regnet_regnet_y_800mf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 352, 28, 28), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 384, 28, 28), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 416, 28, 28), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 448, 28, 28), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
                "pt_regnet_regnet_y_16gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 480, 28, 28), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 128, 14, 14), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm",
                "pt_monodle_base_obj_det_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 288, 14, 14), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_regnet_regnet_x_800mf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 320, 14, 14), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_regnet_regnet_y_800mf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 352, 14, 14), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 384, 14, 14), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_vovnet_vovnet27s_obj_det_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 416, 14, 14), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 448, 14, 14), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 480, 14, 14), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 544, 14, 14), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 576, 14, 14), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_regnet_regnet_y_3_2gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 608, 14, 14), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 640, 14, 14), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 672, 14, 14), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_regnet_regnet_x_800mf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 704, 14, 14), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 736, 14, 14), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 768, 14, 14), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_vovnet_vovnet57_obj_det_osmr",
                "pt_vovnet_vovnet39_obj_det_osmr",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 800, 14, 14), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 832, 14, 14), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 864, 14, 14), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 896, 14, 14), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_regnet_regnet_x_16gf_img_cls_torchvision",
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 928, 14, 14), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 960, 14, 14), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 992, 14, 14), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 128, 7, 7), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 544, 7, 7), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 576, 7, 7), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 608, 7, 7), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 640, 7, 7), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 672, 7, 7), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_regnet_regnet_x_800mf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 704, 7, 7), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 736, 7, 7), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 768, 7, 7), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 800, 7, 7), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 832, 7, 7), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 864, 7, 7), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 896, 7, 7), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 928, 7, 7), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 960, 7, 7), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 992, 7, 7), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1024, 7, 7), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm",
                "pt_mobilenet_v1_basic_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_vovnet_vovnet57_obj_det_osmr",
                "pt_vovnet_vovnet39_obj_det_osmr",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 256, 27, 27), torch.float32)],
        {"model_name": ["pd_googlenet_base_img_cls_paddlemodels", "pt_alexnet_base_img_cls_osmr"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 480, 27, 27), torch.float32)],
        {"model_name": ["pd_googlenet_base_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 512, 13, 13), torch.float32)],
        {"model_name": ["pd_googlenet_base_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 528, 13, 13), torch.float32)],
        {"model_name": ["pd_googlenet_base_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 832, 13, 13), torch.float32)],
        {"model_name": ["pd_googlenet_base_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (Relu0, [((1, 832, 6, 6), torch.float32)], {"model_name": ["pd_googlenet_base_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Relu0,
        [((1, 1024, 6, 6), torch.float32)],
        {"model_name": ["pd_googlenet_base_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (Relu0, [((1, 1024), torch.float32)], {"model_name": ["pd_googlenet_base_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Relu0,
        [((1, 32, 112, 112), torch.float32)],
        {
            "model_name": [
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_dla_dla34_in1k_img_cls_timm",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_mobilenet_v1_basic_img_cls_torchvision",
                "pt_monodle_base_obj_det_torchvision",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
                "pt_regnet_regnet_y_1_6gf_img_cls_torchvision",
                "pt_regnet_regnet_y_32gf_img_cls_torchvision",
                "pt_regnet_regnet_y_400mf_img_cls_torchvision",
                "pt_regnet_regnet_x_8gf_img_cls_torchvision",
                "pt_regnet_regnet_x_32gf_img_cls_torchvision",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
                "pt_regnet_regnet_x_16gf_img_cls_torchvision",
                "pt_regnet_regnet_x_1_6gf_img_cls_torchvision",
                "pt_regnet_regnet_x_800mf_img_cls_torchvision",
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
                "pt_regnet_regnet_y_800mf_img_cls_torchvision",
                "pt_regnet_regnet_y_3_2gf_img_cls_torchvision",
                "pt_regnet_regnet_x_400mf_img_cls_torchvision",
                "pt_regnet_regnet_y_16gf_img_cls_torchvision",
                "pt_unet_qubvel_img_seg_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((2, 13, 3072), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 768), torch.float32)],
        {"model_name": ["pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf"], "pcc": 0.99},
    ),
    (Relu0, [((1, 334, 16384), torch.float32)], {"model_name": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99}),
    (
        Relu0,
        [((32, 8192), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_1_3b_qa_hf", "pt_opt_facebook_opt_1_3b_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((32, 4096), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_350m_seq_cls_hf", "pt_opt_facebook_opt_350m_qa_hf"], "pcc": 0.99},
    ),
    (Relu0, [((256, 8192), torch.float32)], {"model_name": ["pt_opt_facebook_opt_1_3b_clm_hf"], "pcc": 0.99}),
    (Relu0, [((256, 4096), torch.float32)], {"model_name": ["pt_opt_facebook_opt_350m_clm_hf"], "pcc": 0.99}),
    (
        Relu0,
        [((32, 3072), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_125m_qa_hf", "pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99},
    ),
    (Relu0, [((256, 3072), torch.float32)], {"model_name": ["pt_opt_facebook_opt_125m_clm_hf"], "pcc": 0.99}),
    (Relu0, [((1, 61, 2048), torch.float32)], {"model_name": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99}),
    (Relu0, [((1, 1, 2048), torch.float32)], {"model_name": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99}),
    (Relu0, [((1, 61, 4096), torch.float32)], {"model_name": ["pt_t5_t5_large_text_gen_hf"], "pcc": 0.99}),
    (Relu0, [((1, 1, 4096), torch.float32)], {"model_name": ["pt_t5_t5_large_text_gen_hf"], "pcc": 0.99}),
    (Relu0, [((1, 61, 3072), torch.float32)], {"model_name": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99}),
    (Relu0, [((1, 1, 3072), torch.float32)], {"model_name": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99}),
    (Relu0, [((1024, 512), torch.float32)], {"model_name": ["pt_nbeats_generic_basis_clm_hf"], "pcc": 0.99}),
    (Relu0, [((1024, 256), torch.float32)], {"model_name": ["pt_nbeats_trend_basis_clm_hf"], "pcc": 0.99}),
    (Relu0, [((1024, 2048), torch.float32)], {"model_name": ["pt_nbeats_seasionality_basis_clm_hf"], "pcc": 0.99}),
    (Relu0, [((1, 96, 54, 54), torch.float32)], {"model_name": ["pt_alexnet_base_img_cls_osmr"], "pcc": 0.99}),
    (
        Relu0,
        [((1, 16, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_autoencoder_conv_img_enc_github",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (Relu0, [((1, 4, 14, 14), torch.float32)], {"model_name": ["pt_autoencoder_conv_img_enc_github"], "pcc": 0.99}),
    (
        Relu0,
        [((1, 16, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_autoencoder_conv_img_enc_github",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 128), torch.float32)],
        {"model_name": ["pt_autoencoder_linear_img_enc_github", "pt_mnist_base_img_cls_github"], "pcc": 0.99},
    ),
    (Relu0, [((1, 64), torch.float32)], {"model_name": ["pt_autoencoder_linear_img_enc_github"], "pcc": 0.99}),
    (Relu0, [((1, 12), torch.float32)], {"model_name": ["pt_autoencoder_linear_img_enc_github"], "pcc": 0.99}),
    (
        Relu0,
        [((1, 96, 112, 112), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 144, 56, 56), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_regnet_regnet_y_800mf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 240, 56, 56), torch.float32)],
        {
            "model_name": ["pt_densenet_densenet161_img_cls_torchvision", "pt_regnet_regnet_x_8gf_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 288, 56, 56), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 336, 56, 56), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_regnet_regnet_x_32gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 384, 56, 56), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 240, 28, 28), torch.float32)],
        {
            "model_name": ["pt_densenet_densenet161_img_cls_torchvision", "pt_regnet_regnet_x_8gf_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 336, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_regnet_regnet_y_1_6gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 432, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 528, 28, 28), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 576, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_regnet_regnet_y_3_2gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 624, 28, 28), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 672, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_regnet_regnet_x_32gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 720, 28, 28), torch.float32)],
        {
            "model_name": ["pt_densenet_densenet161_img_cls_torchvision", "pt_regnet_regnet_x_8gf_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 768, 28, 28), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 192, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_vovnet_vovnet57_obj_det_osmr",
                "pt_vovnet_vovnet39_obj_det_osmr",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 432, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 528, 14, 14), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 624, 14, 14), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 720, 14, 14), torch.float32)],
        {
            "model_name": ["pt_densenet_densenet161_img_cls_torchvision", "pt_regnet_regnet_x_8gf_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 816, 14, 14), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 912, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_regnet_regnet_x_1_6gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1008, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1056, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1104, 14, 14), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1152, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1200, 14, 14), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1248, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1296, 14, 14), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1344, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_regnet_regnet_x_32gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1392, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_regnet_regnet_y_32gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1440, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1488, 14, 14), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1536, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1584, 14, 14), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1632, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1680, 14, 14), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1728, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1776, 14, 14), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1824, 14, 14), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1872, 14, 14), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1920, 14, 14), torch.float32)],
        {
            "model_name": ["pt_densenet_densenet161_img_cls_torchvision", "pt_regnet_regnet_x_8gf_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1968, 14, 14), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 2016, 14, 14), torch.float32)],
        {
            "model_name": ["pt_densenet_densenet161_img_cls_torchvision", "pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 2064, 14, 14), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 2112, 14, 14), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1056, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 192, 7, 7), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1104, 7, 7), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1152, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1200, 7, 7), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1248, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1296, 7, 7), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1344, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1392, 7, 7), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1440, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1488, 7, 7), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1536, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1584, 7, 7), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1632, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1680, 7, 7), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1728, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1776, 7, 7), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1824, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1872, 7, 7), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1920, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_regnet_regnet_x_8gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1968, 7, 7), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 2016, 7, 7), torch.float32)],
        {
            "model_name": ["pt_densenet_densenet161_img_cls_torchvision", "pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 2064, 7, 7), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 2112, 7, 7), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 2160, 7, 7), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 2208, 7, 7), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1088, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1120, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1184, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1216, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1280, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1312, 14, 14), torch.float32)],
        {"model_name": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1376, 14, 14), torch.float32)],
        {"model_name": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1408, 14, 14), torch.float32)],
        {"model_name": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1472, 14, 14), torch.float32)],
        {"model_name": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1504, 14, 14), torch.float32)],
        {"model_name": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1568, 14, 14), torch.float32)],
        {"model_name": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1600, 14, 14), torch.float32)],
        {"model_name": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1664, 14, 14), torch.float32)],
        {"model_name": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1696, 14, 14), torch.float32)],
        {"model_name": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1760, 14, 14), torch.float32)],
        {"model_name": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1792, 14, 14), torch.float32)],
        {"model_name": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1088, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1120, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1184, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1216, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1280, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1312, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1376, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1408, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1472, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1504, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1568, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1600, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1664, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1696, 7, 7), torch.float32)],
        {"model_name": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1760, 7, 7), torch.float32)],
        {"model_name": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1792, 7, 7), torch.float32)],
        {"model_name": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1856, 7, 7), torch.float32)],
        {"model_name": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1888, 7, 7), torch.float32)],
        {"model_name": ["pt_densenet_densenet201_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 16, 224, 224), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_dla_dla34_in1k_img_cls_timm",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_monodle_base_obj_det_torchvision",
                "pt_unet_qubvel_img_seg_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 128, 112, 112), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
                "pt_unet_cityscape_img_seg_osmr",
                "pt_unet_carvana_base_img_seg_github",
                "pt_vgg_vgg13_img_cls_torchvision",
                "pt_vgg_vgg19_img_cls_torchvision",
                "pt_vgg_vgg19_obj_det_osmr",
                "pt_vgg_vgg11_obj_det_osmr",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vgg_vgg13_bn_img_cls_torchvision",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg13_obj_det_osmr",
                "pt_vgg_vgg16_bn_img_cls_torchvision",
                "pt_vgg_vgg11_bn_img_cls_torchvision",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vgg_vgg19_bn_obj_det_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 64, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm",
                "pt_monodle_base_obj_det_torchvision",
                "pt_regnet_regnet_x_400mf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 256, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm",
                "pt_monodle_base_obj_det_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 256, 112, 112), torch.float32)],
        {
            "model_name": ["pt_dla_dla102x2_visual_bb_torchvision", "pt_regnet_regnet_x_16gf_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 512, 56, 56), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_regnet_regnet_x_16gf_img_cls_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1024, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 2048, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_regnet_regnet_x_16gf_img_cls_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 32, 56, 56), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm",
                "pt_regnet_regnet_x_400mf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 32, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 64, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 16, 112, 112), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 8, 112, 112), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 24, 112, 112), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 36, 56, 56), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 20, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 60, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 32, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 120, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_regnet_regnet_y_1_6gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 100, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 92, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 240, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 120, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 336, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_regnet_regnet_y_1_6gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 168, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 480, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 240, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1280, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 64, 30, 40), torch.float32)],
        {"model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 32, 30, 40), torch.float32)],
        {"model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 64, 60, 80), torch.float32)],
        {"model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 32, 60, 80), torch.float32)],
        {"model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 64, 120, 160), torch.float32)],
        {
            "model_name": [
                "pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 32, 120, 160), torch.float32)],
        {"model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 64, 480, 640), torch.float32)],
        {"model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 176, 28, 28), torch.float32)],
        {"model_name": ["pt_googlenet_base_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 96, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 304, 14, 14), torch.float32)],
        {"model_name": ["pt_googlenet_base_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 208, 14, 14), torch.float32)],
        {
            "model_name": ["pt_googlenet_base_img_cls_torchvision", "pt_regnet_regnet_y_400mf_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 48, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 296, 14, 14), torch.float32)],
        {"model_name": ["pt_googlenet_base_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 224, 14, 14), torch.float32)],
        {"model_name": ["pt_googlenet_base_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 280, 14, 14), torch.float32)],
        {"model_name": ["pt_googlenet_base_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Relu0, [((1, 448, 7, 7), torch.float32)], {"model_name": ["pt_googlenet_base_img_cls_torchvision"], "pcc": 0.99}),
    (
        Relu0,
        [((1, 320, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (Relu0, [((1, 624, 7, 7), torch.float32)], {"model_name": ["pt_googlenet_base_img_cls_torchvision"], "pcc": 0.99}),
    (
        Relu0,
        [((1, 384, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_googlenet_base_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 18, 56, 56), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 36, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 72, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 18, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 144, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 18, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 36, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 48, 56, 56), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_regnet_regnet_y_1_6gf_img_cls_torchvision",
                "pt_regnet_regnet_y_400mf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 48, 28, 28), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnetv2_w48_pose_estimation_osmr", "pt_hrnet_hrnet_w48_pose_estimation_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 96, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_vovnet_vovnet27s_obj_det_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 16, 56, 56), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 32, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 40, 56, 56), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnet_w40_pose_estimation_timm", "pt_hrnet_hrnetv2_w40_pose_estimation_osmr"],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 80, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_vovnet_vovnet27s_obj_det_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 160, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_regnet_regnet_x_400mf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 40, 28, 28), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnet_w40_pose_estimation_timm", "pt_hrnet_hrnetv2_w40_pose_estimation_osmr"],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 40, 14, 14), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnet_w40_pose_estimation_timm", "pt_hrnet_hrnetv2_w40_pose_estimation_osmr"],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 80, 14, 14), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnet_w40_pose_estimation_timm", "pt_hrnet_hrnetv2_w40_pose_estimation_osmr"],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 44, 56, 56), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnetv2_w44_pose_estimation_osmr", "pt_hrnet_hrnet_w44_pose_estimation_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 88, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 176, 14, 14), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnetv2_w44_pose_estimation_osmr", "pt_hrnet_hrnet_w44_pose_estimation_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 44, 28, 28), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnetv2_w44_pose_estimation_osmr", "pt_hrnet_hrnet_w44_pose_estimation_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 352, 7, 7), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnetv2_w44_pose_estimation_osmr", "pt_hrnet_hrnet_w44_pose_estimation_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 44, 14, 14), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnetv2_w44_pose_estimation_osmr", "pt_hrnet_hrnet_w44_pose_estimation_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 88, 14, 14), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnetv2_w44_pose_estimation_osmr", "pt_hrnet_hrnet_w44_pose_estimation_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 30, 56, 56), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnet_w30_pose_estimation_timm", "pt_hrnet_hrnetv2_w30_pose_estimation_osmr"],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 120, 14, 14), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnet_w30_pose_estimation_timm", "pt_hrnet_hrnetv2_w30_pose_estimation_osmr"],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 30, 28, 28), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnet_w30_pose_estimation_timm", "pt_hrnet_hrnetv2_w30_pose_estimation_osmr"],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 240, 7, 7), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnet_w30_pose_estimation_timm", "pt_hrnet_hrnetv2_w30_pose_estimation_osmr"],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 30, 14, 14), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnet_w30_pose_estimation_timm", "pt_hrnet_hrnetv2_w30_pose_estimation_osmr"],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 60, 14, 14), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnet_w30_pose_estimation_timm", "pt_hrnet_hrnetv2_w30_pose_estimation_osmr"],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 32, 149, 149), torch.float32)],
        {
            "model_name": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 32, 147, 147), torch.float32)],
        {
            "model_name": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 64, 147, 147), torch.float32)],
        {
            "model_name": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 96, 73, 73), torch.float32)],
        {
            "model_name": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 64, 73, 73), torch.float32)],
        {
            "model_name": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 96, 71, 71), torch.float32)],
        {
            "model_name": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 192, 35, 35), torch.float32)],
        {
            "model_name": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 224, 35, 35), torch.float32)],
        {
            "model_name": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 96, 35, 35), torch.float32)],
        {
            "model_name": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 384, 17, 17), torch.float32)],
        {
            "model_name": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 256, 17, 17), torch.float32)],
        {
            "model_name": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 768, 17, 17), torch.float32)],
        {
            "model_name": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 224, 17, 17), torch.float32)],
        {
            "model_name": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 192, 17, 17), torch.float32)],
        {
            "model_name": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 128, 17, 17), torch.float32)],
        {
            "model_name": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 192, 8, 8), torch.float32)],
        {
            "model_name": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 320, 17, 17), torch.float32)],
        {
            "model_name": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 320, 8, 8), torch.float32)],
        {
            "model_name": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1024, 8, 8), torch.float32)],
        {
            "model_name": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 256, 8, 8), torch.float32)],
        {
            "model_name": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 448, 8, 8), torch.float32)],
        {
            "model_name": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 512, 8, 8), torch.float32)],
        {
            "model_name": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (Relu0, [((1, 32, 26, 26), torch.float32)], {"model_name": ["pt_mnist_base_img_cls_github"], "pcc": 0.99}),
    (Relu0, [((1, 64, 24, 24), torch.float32)], {"model_name": ["pt_mnist_base_img_cls_github"], "pcc": 0.99}),
    (
        Relu0,
        [((1, 256, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_ssd300_resnet50_base_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 8, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
                "pt_regnet_regnet_y_128gf_img_cls_torchvision",
                "pt_regnet_regnet_y_1_6gf_img_cls_torchvision",
                "pt_regnet_regnet_y_32gf_img_cls_torchvision",
                "pt_regnet_regnet_y_400mf_img_cls_torchvision",
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
                "pt_regnet_regnet_y_800mf_img_cls_torchvision",
                "pt_regnet_regnet_y_3_2gf_img_cls_torchvision",
                "pt_regnet_regnet_y_16gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 72, 56, 56), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_regnet_regnet_x_1_6gf_img_cls_torchvision",
                "pt_regnet_regnet_y_3_2gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 72, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 24, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 64, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 40, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 72, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 144, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_regnet_regnet_y_3_2gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 64, 160, 160), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_ssd_resnet18_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet34_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 64, 80, 80), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_ssd_resnet18_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet34_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 128, 40, 40), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_ssd_resnet18_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet34_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 256, 20, 20), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_ssd_resnet18_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet34_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 512, 10, 10), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_ssd_resnet18_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet34_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
                "pt_ssd300_resnet50_base_img_cls_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 256, 80, 80), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 128, 80, 80), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 512, 40, 40), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 256, 40, 40), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1024, 20, 20), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 512, 20, 20), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 2048, 10, 10), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 64, 96, 320), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 64, 48, 160), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 128, 24, 80), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 256, 12, 40), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 512, 6, 20), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 64, 160, 512), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 64, 80, 256), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 128, 40, 128), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 256, 20, 64), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 512, 10, 32), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 48, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_facebook_regnet_y_040_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 128, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_facebook_regnet_y_040_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 272, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_facebook_regnet_y_040_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 32, 192, 192), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 528, 192, 192), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 528, 96, 96), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 132, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1056, 96, 96), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1056, 48, 48), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 264, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 2904, 48, 48), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 2904, 24, 24), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 726, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 7392, 24, 24), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 7392, 12, 12), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 48, 112, 112), torch.float32)],
        {
            "model_name": [
                "pt_regnet_regnet_y_1_6gf_img_cls_torchvision",
                "pt_regnet_regnet_y_400mf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 12, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_regnet_regnet_y_1_6gf_img_cls_torchvision",
                "pt_regnet_regnet_y_400mf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 120, 56, 56), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_1_6gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 30, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_1_6gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 84, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_1_6gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 888, 14, 14), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_1_6gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 888, 7, 7), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_1_6gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 222, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_1_6gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 232, 112, 112), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_32gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 232, 56, 56), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_32gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 58, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_32gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 696, 56, 56), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_32gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 696, 28, 28), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_32gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 174, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_32gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1392, 28, 28), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_32gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 348, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_32gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 3712, 14, 14), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_32gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 3712, 7, 7), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_32gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 104, 56, 56), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_400mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 104, 28, 28), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_400mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 26, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_400mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 208, 28, 28), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_400mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 52, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_400mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 440, 14, 14), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_400mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 440, 7, 7), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_400mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 110, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_400mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 80, 112, 112), torch.float32)],
        {"model_name": ["pt_regnet_regnet_x_8gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 80, 56, 56), torch.float32)],
        {"model_name": ["pt_regnet_regnet_x_8gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 336, 112, 112), torch.float32)],
        {"model_name": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 672, 56, 56), torch.float32)],
        {"model_name": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1344, 28, 28), torch.float32)],
        {"model_name": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 2520, 14, 14), torch.float32)],
        {"model_name": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 2520, 7, 7), torch.float32)],
        {"model_name": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1008, 7, 7), torch.float32)],
        {"model_name": ["pt_regnet_regnet_x_3_2gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 896, 28, 28), torch.float32)],
        {
            "model_name": ["pt_regnet_regnet_x_16gf_img_cls_torchvision", "pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 72, 112, 112), torch.float32)],
        {
            "model_name": [
                "pt_regnet_regnet_x_1_6gf_img_cls_torchvision",
                "pt_regnet_regnet_y_3_2gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 168, 56, 56), torch.float32)],
        {"model_name": ["pt_regnet_regnet_x_1_6gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 168, 28, 28), torch.float32)],
        {"model_name": ["pt_regnet_regnet_x_1_6gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 408, 28, 28), torch.float32)],
        {"model_name": ["pt_regnet_regnet_x_1_6gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 408, 14, 14), torch.float32)],
        {"model_name": ["pt_regnet_regnet_x_1_6gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 912, 7, 7), torch.float32)],
        {"model_name": ["pt_regnet_regnet_x_1_6gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 224, 112, 112), torch.float32)],
        {
            "model_name": ["pt_regnet_regnet_y_8gf_img_cls_torchvision", "pt_regnet_regnet_y_16gf_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 56, 1, 1), torch.float32)],
        {
            "model_name": ["pt_regnet_regnet_y_8gf_img_cls_torchvision", "pt_regnet_regnet_y_16gf_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 448, 56, 56), torch.float32)],
        {
            "model_name": ["pt_regnet_regnet_y_8gf_img_cls_torchvision", "pt_regnet_regnet_y_16gf_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 112, 1, 1), torch.float32)],
        {
            "model_name": ["pt_regnet_regnet_y_8gf_img_cls_torchvision", "pt_regnet_regnet_y_16gf_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 224, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_8gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 144, 28, 28), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_800mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 16, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_800mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 36, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_800mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 80, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_800mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 784, 14, 14), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_800mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 784, 7, 7), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_800mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 196, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_800mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 18, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_3_2gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 216, 56, 56), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_3_2gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 216, 28, 28), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_3_2gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 54, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_3_2gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1512, 14, 14), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_3_2gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1512, 7, 7), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_3_2gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 400, 14, 14), torch.float32)],
        {"model_name": ["pt_regnet_regnet_x_400mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 400, 7, 7), torch.float32)],
        {"model_name": ["pt_regnet_regnet_x_400mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1232, 28, 28), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_16gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1232, 14, 14), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_16gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 308, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_16gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 3024, 14, 14), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_16gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 3024, 7, 7), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_16gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 64, 240, 320), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 256, 120, 160), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 128, 120, 160), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 128, 60, 80), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 512, 60, 80), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 256, 60, 80), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 256, 30, 40), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1024, 30, 40), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 512, 30, 40), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 512, 15, 20), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 2048, 15, 20), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 256, 15, 20), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 256, 8, 10), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 256, 4, 5), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 768, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 256, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 64, 150, 150), torch.float32)],
        {
            "model_name": [
                "pt_ssd300_resnet50_base_img_cls_torchhub",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 64, 75, 75), torch.float32)],
        {"model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 256, 75, 75), torch.float32)],
        {
            "model_name": [
                "pt_ssd300_resnet50_base_img_cls_torchhub",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 128, 75, 75), torch.float32)],
        {
            "model_name": [
                "pt_ssd300_resnet50_base_img_cls_torchhub",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 128, 38, 38), torch.float32)],
        {"model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 512, 38, 38), torch.float32)],
        {"model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 256, 38, 38), torch.float32)],
        {
            "model_name": [
                "pt_ssd300_resnet50_base_img_cls_torchhub",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1024, 38, 38), torch.float32)],
        {"model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 512, 19, 19), torch.float32)],
        {"model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 256, 19, 19), torch.float32)],
        {"model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 128, 10, 10), torch.float32)],
        {"model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 256, 5, 5), torch.float32)],
        {"model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 128, 5, 5), torch.float32)],
        {"model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 256, 3, 3), torch.float32)],
        {"model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 128, 3, 3), torch.float32)],
        {"model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (Relu0, [((1, 32, 256, 256), torch.float32)], {"model_name": ["pt_unet_base_img_seg_torchhub"], "pcc": 0.99}),
    (Relu0, [((1, 64, 128, 128), torch.float32)], {"model_name": ["pt_unet_base_img_seg_torchhub"], "pcc": 0.99}),
    (Relu0, [((1, 128, 64, 64), torch.float32)], {"model_name": ["pt_unet_base_img_seg_torchhub"], "pcc": 0.99}),
    (Relu0, [((1, 256, 32, 32), torch.float32)], {"model_name": ["pt_unet_base_img_seg_torchhub"], "pcc": 0.99}),
    (Relu0, [((1, 512, 16, 16), torch.float32)], {"model_name": ["pt_unet_base_img_seg_torchhub"], "pcc": 0.99}),
    (
        Relu0,
        [((1, 64, 224, 224), torch.float32)],
        {
            "model_name": [
                "pt_unet_cityscape_img_seg_osmr",
                "pt_unet_carvana_base_img_seg_github",
                "pt_vgg_vgg13_img_cls_torchvision",
                "pt_vgg_vgg19_img_cls_torchvision",
                "pt_vgg_vgg19_obj_det_osmr",
                "pt_vgg_vgg11_obj_det_osmr",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vgg_vgg13_bn_img_cls_torchvision",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg13_obj_det_osmr",
                "pt_vgg_vgg16_bn_img_cls_torchvision",
                "pt_vgg_vgg11_bn_img_cls_torchvision",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vgg_vgg19_bn_obj_det_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (Relu0, [((1, 4096, 1, 1), torch.float32)], {"model_name": ["pt_vgg_vgg19_bn_obj_det_timm"], "pcc": 0.99}),
    (
        Relu0,
        [((1, 224, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_vovnet_vovnet57_obj_det_osmr",
                "pt_vovnet_vovnet39_obj_det_osmr",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (Relu0, [((1, 112, 7, 7), torch.float32)], {"model_name": ["pt_vovnet_vovnet27s_obj_det_osmr"], "pcc": 0.99}),
    (
        Relu0,
        [((1, 32, 150, 150), torch.float32)],
        {
            "model_name": [
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 128, 150, 150), torch.float32)],
        {
            "model_name": [
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 728, 38, 38), torch.float32)],
        {
            "model_name": [
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 728, 19, 19), torch.float32)],
        {
            "model_name": [
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1024, 19, 19), torch.float32)],
        {
            "model_name": [
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1024, 10, 10), torch.float32)],
        {
            "model_name": [
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 1536, 10, 10), torch.float32)],
        {
            "model_name": [
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (Relu0, [((1, 128, 147, 147), torch.float32)], {"model_name": ["pt_xception_xception_img_cls_timm"], "pcc": 0.99}),
    (Relu0, [((1, 128, 74, 74), torch.float32)], {"model_name": ["pt_xception_xception_img_cls_timm"], "pcc": 0.99}),
    (Relu0, [((1, 256, 74, 74), torch.float32)], {"model_name": ["pt_xception_xception_img_cls_timm"], "pcc": 0.99}),
    (Relu0, [((1, 256, 37, 37), torch.float32)], {"model_name": ["pt_xception_xception_img_cls_timm"], "pcc": 0.99}),
    (Relu0, [((1, 728, 37, 37), torch.float32)], {"model_name": ["pt_xception_xception_img_cls_timm"], "pcc": 0.99}),
    (Relu0, [((1, 16, 224, 320), torch.float32)], {"model_name": ["pt_yolo_v6_yolov6n_obj_det_torchhub"], "pcc": 0.99}),
    (Relu0, [((1, 32, 112, 160), torch.float32)], {"model_name": ["pt_yolo_v6_yolov6n_obj_det_torchhub"], "pcc": 0.99}),
    (
        Relu0,
        [((1, 64, 56, 80), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 128, 28, 40), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 256, 14, 20), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 128, 14, 20), torch.float32)],
        {"model_name": ["pt_yolo_v6_yolov6n_obj_det_torchhub", "pt_yolo_v6_yolov6s_obj_det_torchhub"], "pcc": 0.99},
    ),
    (Relu0, [((1, 64, 14, 20), torch.float32)], {"model_name": ["pt_yolo_v6_yolov6n_obj_det_torchhub"], "pcc": 0.99}),
    (
        Relu0,
        [((1, 64, 28, 40), torch.float32)],
        {"model_name": ["pt_yolo_v6_yolov6n_obj_det_torchhub", "pt_yolo_v6_yolov6s_obj_det_torchhub"], "pcc": 0.99},
    ),
    (Relu0, [((1, 32, 28, 40), torch.float32)], {"model_name": ["pt_yolo_v6_yolov6n_obj_det_torchhub"], "pcc": 0.99}),
    (Relu0, [((1, 32, 56, 80), torch.float32)], {"model_name": ["pt_yolo_v6_yolov6n_obj_det_torchhub"], "pcc": 0.99}),
    (Relu0, [((1, 48, 224, 320), torch.float32)], {"model_name": ["pt_yolo_v6_yolov6m_obj_det_torchhub"], "pcc": 0.99}),
    (Relu0, [((1, 96, 112, 160), torch.float32)], {"model_name": ["pt_yolo_v6_yolov6m_obj_det_torchhub"], "pcc": 0.99}),
    (
        Relu0,
        [((1, 64, 112, 160), torch.float32)],
        {"model_name": ["pt_yolo_v6_yolov6m_obj_det_torchhub", "pt_yolo_v6_yolov6s_obj_det_torchhub"], "pcc": 0.99},
    ),
    (Relu0, [((1, 192, 56, 80), torch.float32)], {"model_name": ["pt_yolo_v6_yolov6m_obj_det_torchhub"], "pcc": 0.99}),
    (
        Relu0,
        [((1, 128, 56, 80), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (Relu0, [((1, 384, 28, 40), torch.float32)], {"model_name": ["pt_yolo_v6_yolov6m_obj_det_torchhub"], "pcc": 0.99}),
    (
        Relu0,
        [((1, 256, 28, 40), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (Relu0, [((1, 768, 14, 20), torch.float32)], {"model_name": ["pt_yolo_v6_yolov6m_obj_det_torchhub"], "pcc": 0.99}),
    (Relu0, [((1, 384, 14, 20), torch.float32)], {"model_name": ["pt_yolo_v6_yolov6m_obj_det_torchhub"], "pcc": 0.99}),
    (Relu0, [((1, 192, 14, 20), torch.float32)], {"model_name": ["pt_yolo_v6_yolov6m_obj_det_torchhub"], "pcc": 0.99}),
    (Relu0, [((1, 192, 28, 40), torch.float32)], {"model_name": ["pt_yolo_v6_yolov6m_obj_det_torchhub"], "pcc": 0.99}),
    (Relu0, [((1, 96, 28, 40), torch.float32)], {"model_name": ["pt_yolo_v6_yolov6m_obj_det_torchhub"], "pcc": 0.99}),
    (Relu0, [((1, 96, 56, 80), torch.float32)], {"model_name": ["pt_yolo_v6_yolov6m_obj_det_torchhub"], "pcc": 0.99}),
    (Relu0, [((1, 256, 56, 80), torch.float32)], {"model_name": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99}),
    (
        Relu0,
        [((1, 128, 112, 160), torch.float32)],
        {"model_name": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99},
    ),
    (Relu0, [((1, 32, 224, 320), torch.float32)], {"model_name": ["pt_yolo_v6_yolov6s_obj_det_torchhub"], "pcc": 0.99}),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes, forge_property_recorder):

    forge_property_recorder.enable_single_op_details_recording()
    forge_property_recorder.record_forge_op_name("Relu")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes

    pcc = metadata.pop("pcc")

    for metadata_name, metadata_value in metadata.items():
        if metadata_name == "model_name":
            forge_property_recorder.record_op_model_names(metadata_value)
        elif metadata_name == "op_params":
            forge_property_recorder.record_forge_op_args(metadata_value)
        else:
            logger.warning("no utility function in forge property handler")

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

    forge_property_recorder.record_single_op_operands_info(framework_model, inputs)

    compiled_model = compile(framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder)

    verify(
        inputs,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
        forge_property_handler=forge_property_recorder,
    )
