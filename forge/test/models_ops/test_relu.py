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
        [((1, 64, 112, 112), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_vovnet_vovnet27s_obj_det_osmr",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 64, 56, 56), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_vovnet_vovnet27s_obj_det_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 256, 56, 56), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 128, 56, 56), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_vovnet_vovnet27s_obj_det_osmr",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 128, 28, 28), torch.float32)],
        {"model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 512, 28, 28), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 256, 28, 28), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_vovnet_vovnet27s_obj_det_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 256, 14, 14), torch.float32)],
        {"model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1024, 14, 14), torch.float32)],
        {"model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 512, 14, 14), torch.float32)],
        {"model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 512, 7, 7), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_vovnet_vovnet27s_obj_det_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 2048, 7, 7), torch.float32)],
        {"model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 16, 224, 224), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 32, 112, 112), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_regnet_facebook_regnet_y_160_img_cls_hf",
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_regnet_regnet_x_32gf_img_cls_torchvision",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 64, 112, 112), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 64, 56, 56), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 128, 56, 56), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 128, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 256, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_resnet_50_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 256, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 512, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_vgg_vgg13_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 512, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1024, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 256, 56, 56), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_vgg_vgg13_img_cls_torchvision",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 30, 56, 56), torch.bfloat16)],
        {"model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Relu0,
        [((1, 60, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm", "pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 120, 14, 14), torch.bfloat16)],
        {"model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Relu0,
        [((1, 30, 28, 28), torch.bfloat16)],
        {"model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Relu0,
        [((1, 240, 7, 7), torch.bfloat16)],
        {"model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Relu0,
        [((1, 30, 14, 14), torch.bfloat16)],
        {"model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Relu0,
        [((1, 60, 14, 14), torch.bfloat16)],
        {"model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Relu0,
        [((1, 256, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 128, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 64, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 32, 56, 56), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 2048, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_resnet_50_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 64, 160, 160), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_ssd_resnet34_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 64, 80, 80), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_ssd_resnet34_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 128, 40, 40), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_ssd_resnet34_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 256, 20, 20), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_ssd_resnet34_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 512, 10, 10), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_ssd_resnet34_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 32, 192, 192), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 528, 192, 192), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 528, 96, 96), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 8, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_regnet_y_128gf_img_cls_torchvision",
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_160_img_cls_hf",
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 132, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1056, 96, 96), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1056, 48, 48), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 264, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 2904, 48, 48), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 2904, 24, 24), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 726, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 7392, 24, 24), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 7392, 12, 12), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 224, 112, 112), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_8gf_img_cls_torchvision", "pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 224, 56, 56), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_160_img_cls_hf",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 56, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_8gf_img_cls_torchvision", "pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 448, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_8gf_img_cls_torchvision", "pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 448, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_160_img_cls_hf",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 112, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_8gf_img_cls_torchvision", "pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 896, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 896, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 224, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 2016, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 2016, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 512, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_vgg_vgg13_img_cls_torchvision",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1024, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 64, 240, 320), torch.bfloat16)],
        {"model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Relu0,
        [((1, 64, 120, 160), torch.bfloat16)],
        {"model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Relu0,
        [((1, 256, 120, 160), torch.bfloat16)],
        {"model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Relu0,
        [((1, 128, 120, 160), torch.bfloat16)],
        {"model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Relu0,
        [((1, 128, 60, 80), torch.bfloat16)],
        {"model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Relu0,
        [((1, 512, 60, 80), torch.bfloat16)],
        {"model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Relu0,
        [((1, 256, 60, 80), torch.bfloat16)],
        {"model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Relu0,
        [((1, 256, 30, 40), torch.bfloat16)],
        {"model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Relu0,
        [((1, 1024, 30, 40), torch.bfloat16)],
        {"model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Relu0,
        [((1, 512, 30, 40), torch.bfloat16)],
        {"model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Relu0,
        [((1, 512, 15, 20), torch.bfloat16)],
        {"model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Relu0,
        [((1, 2048, 15, 20), torch.bfloat16)],
        {"model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Relu0,
        [((1, 256, 15, 20), torch.bfloat16)],
        {"model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Relu0,
        [((1, 256, 8, 10), torch.bfloat16)],
        {"model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Relu0,
        [((1, 256, 4, 5), torch.bfloat16)],
        {"model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Relu0,
        [((1, 64, 224, 224), torch.bfloat16)],
        {
            "model_names": [
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vgg_vgg13_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 128, 112, 112), torch.bfloat16)],
        {
            "model_names": [
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vgg_vgg13_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 4096), torch.bfloat16)],
        {
            "model_names": [
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vgg_vgg13_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 160, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 192, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 768, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 224, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 80, 28, 28), torch.bfloat16)],
        {"model_names": ["pt_vovnet_vovnet27s_img_cls_osmr"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Relu0,
        [((1, 96, 14, 14), torch.bfloat16)],
        {"model_names": ["pt_vovnet_vovnet27s_img_cls_osmr"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Relu0,
        [((1, 384, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 112, 7, 7), torch.bfloat16)],
        {"model_names": ["pt_vovnet_vovnet27s_img_cls_osmr"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Relu0,
        [((1, 32, 150, 150), torch.bfloat16)],
        {
            "model_names": ["pt_xception_xception41_img_cls_timm", "pt_xception_xception71_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 64, 150, 150), torch.bfloat16)],
        {
            "model_names": ["pt_xception_xception41_img_cls_timm", "pt_xception_xception71_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 128, 150, 150), torch.bfloat16)],
        {
            "model_names": ["pt_xception_xception41_img_cls_timm", "pt_xception_xception71_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 128, 75, 75), torch.bfloat16)],
        {
            "model_names": ["pt_xception_xception41_img_cls_timm", "pt_xception_xception71_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 256, 75, 75), torch.bfloat16)],
        {
            "model_names": ["pt_xception_xception41_img_cls_timm", "pt_xception_xception71_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 256, 38, 38), torch.bfloat16)],
        {
            "model_names": ["pt_xception_xception41_img_cls_timm", "pt_xception_xception71_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 728, 38, 38), torch.bfloat16)],
        {
            "model_names": ["pt_xception_xception41_img_cls_timm", "pt_xception_xception71_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 728, 19, 19), torch.bfloat16)],
        {
            "model_names": [
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1024, 19, 19), torch.bfloat16)],
        {
            "model_names": ["pt_xception_xception41_img_cls_timm", "pt_xception_xception71_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1024, 10, 10), torch.bfloat16)],
        {
            "model_names": ["pt_xception_xception41_img_cls_timm", "pt_xception_xception71_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1536, 10, 10), torch.bfloat16)],
        {
            "model_names": [
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 2048, 10, 10), torch.bfloat16)],
        {
            "model_names": [
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_xception_xception_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 768, 128, 128), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 96, 56, 56), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 160, 56, 56), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 192, 56, 56), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 192, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 224, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 288, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 320, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 352, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 384, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 416, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 480, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 288, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 320, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 352, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 416, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 448, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 480, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 544, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 576, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 608, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 640, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 672, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 704, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 736, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 800, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 832, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 864, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 928, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 960, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 992, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 128, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 544, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 576, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 608, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 640, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 672, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 704, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 736, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 768, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 800, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 832, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 864, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 896, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 928, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 960, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 992, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 256, 112, 112), torch.bfloat16)],
        {"model_names": ["pt_dla_dla102x2_visual_bb_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Relu0,
        [((1, 512, 56, 56), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1024, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 2048, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 32, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 64, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 16, 112, 112), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 8, 112, 112), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 24, 112, 112), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 36, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 20, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 32, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 120, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 100, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 92, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 240, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 120, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 336, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 168, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 480, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 240, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1280, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 16, 56, 56), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 16, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 16, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 32, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 18, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 36, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 72, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 18, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 144, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 18, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 36, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 72, 56, 56), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 72, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 24, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 256, 80, 80), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 128, 80, 80), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 512, 40, 40), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 256, 40, 40), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1024, 20, 20), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 512, 20, 20), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 64, 96, 320), torch.bfloat16)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 64, 48, 160), torch.bfloat16)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 128, 24, 80), torch.bfloat16)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 256, 12, 40), torch.bfloat16)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 512, 6, 20), torch.bfloat16)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1232, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1232, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 308, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 3024, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 3024, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 232, 112, 112), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 232, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 58, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 696, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 696, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 174, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1392, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1392, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 348, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 3712, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 3712, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 336, 112, 112), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 336, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 672, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 672, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1344, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1344, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_regnet_x_32gf_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 2520, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 2520, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 96, 112, 112), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_3_2gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 432, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_3_2gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 432, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_3_2gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1008, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_3_2gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1008, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_3_2gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 15, 15, 512), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Relu0, [((1, 61, 3072), torch.float32)], {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99}),
    (Relu0, [((1, 513, 3072), torch.float32)], {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99}),
    (
        Relu0,
        [((1, 32, 150, 150), torch.float32)],
        {"model_names": ["onnx_xception_xception71_tf_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 64, 150, 150), torch.float32)],
        {"model_names": ["onnx_xception_xception71_tf_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 128, 150, 150), torch.float32)],
        {"model_names": ["onnx_xception_xception71_tf_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 128, 75, 75), torch.float32)],
        {"model_names": ["onnx_xception_xception71_tf_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 256, 75, 75), torch.float32)],
        {"model_names": ["onnx_xception_xception71_tf_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 256, 38, 38), torch.float32)],
        {"model_names": ["onnx_xception_xception71_tf_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 728, 38, 38), torch.float32)],
        {"model_names": ["onnx_xception_xception71_tf_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 728, 19, 19), torch.float32)],
        {"model_names": ["onnx_xception_xception71_tf_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1024, 19, 19), torch.float32)],
        {"model_names": ["onnx_xception_xception71_tf_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1024, 10, 10), torch.float32)],
        {"model_names": ["onnx_xception_xception71_tf_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1536, 10, 10), torch.float32)],
        {"model_names": ["onnx_xception_xception71_tf_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 2048, 10, 10), torch.float32)],
        {"model_names": ["onnx_xception_xception71_tf_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 32, 149, 149), torch.bfloat16)],
        {
            "model_names": [
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 32, 147, 147), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 64, 147, 147), torch.bfloat16)],
        {
            "model_names": [
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 96, 73, 73), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 64, 73, 73), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 96, 71, 71), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 192, 35, 35), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 224, 35, 35), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 96, 35, 35), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 384, 17, 17), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 256, 17, 17), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 768, 17, 17), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 224, 17, 17), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 192, 17, 17), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 128, 17, 17), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 192, 8, 8), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 320, 17, 17), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 320, 8, 8), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1024, 8, 8), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 256, 8, 8), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 448, 8, 8), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 512, 8, 8), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 256, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 88, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 64, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 40, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 72, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 144, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 768, 128, 128), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 128, 147, 147), torch.bfloat16)],
        {"model_names": ["pt_xception_xception_img_cls_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Relu0,
        [((1, 128, 74, 74), torch.bfloat16)],
        {"model_names": ["pt_xception_xception_img_cls_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Relu0,
        [((1, 256, 74, 74), torch.bfloat16)],
        {"model_names": ["pt_xception_xception_img_cls_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Relu0,
        [((1, 256, 37, 37), torch.bfloat16)],
        {"model_names": ["pt_xception_xception_img_cls_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Relu0,
        [((1, 728, 37, 37), torch.bfloat16)],
        {"model_names": ["pt_xception_xception_img_cls_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Relu0,
        [((1, 256, 14, 20), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Relu0,
        [((1, 256, 28, 40), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Relu0,
        [((1, 256, 56, 80), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Relu0,
        [((1, 128, 28, 40), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Relu0,
        [((1, 128, 56, 80), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Relu0,
        [((1, 128, 112, 160), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (Relu0, [((1, 112, 109, 64), torch.float32)], {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99}),
    (Relu0, [((1, 56, 54, 64), torch.float32)], {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99}),
    (Relu0, [((1, 56, 54, 256), torch.float32)], {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99}),
    (Relu0, [((1, 28, 27, 128), torch.float32)], {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99}),
    (Relu0, [((1, 28, 27, 512), torch.float32)], {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99}),
    (Relu0, [((1, 14, 14, 256), torch.float32)], {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99}),
    (Relu0, [((1, 14, 14, 1024), torch.float32)], {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99}),
    (Relu0, [((1, 7, 7, 512), torch.float32)], {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99}),
    (Relu0, [((1, 7, 7, 2048), torch.float32)], {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99}),
    (
        Relu0,
        [((1, 16, 56, 56), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 8, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 72, 56, 56), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 72, 28, 28), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 88, 28, 28), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 24, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 64, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 32, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 40, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 72, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Relu0,
        [((1, 144, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (Relu0, [((1, 80, 28, 28), torch.float32)], {"model_names": ["onnx_vovnet_vovnet27s_obj_det_osmr"], "pcc": 0.99}),
    (Relu0, [((1, 96, 14, 14), torch.float32)], {"model_names": ["onnx_vovnet_vovnet27s_obj_det_osmr"], "pcc": 0.99}),
    (Relu0, [((1, 384, 14, 14), torch.float32)], {"model_names": ["onnx_vovnet_vovnet27s_obj_det_osmr"], "pcc": 0.99}),
    (Relu0, [((1, 112, 7, 7), torch.float32)], {"model_names": ["onnx_vovnet_vovnet27s_obj_det_osmr"], "pcc": 0.99}),
    (
        Relu0,
        [((1, 160, 28, 28), torch.float32)],
        {"model_names": ["onnx_vovnet_vovnet_v1_57_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 192, 14, 14), torch.float32)],
        {"model_names": ["onnx_vovnet_vovnet_v1_57_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 768, 14, 14), torch.float32)],
        {"model_names": ["onnx_vovnet_vovnet_v1_57_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 224, 7, 7), torch.float32)],
        {"model_names": ["onnx_vovnet_vovnet_v1_57_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 1024, 7, 7), torch.float32)],
        {"model_names": ["onnx_vovnet_vovnet_v1_57_obj_det_torchhub"], "pcc": 0.99},
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
    (
        Relu0,
        [((1, 8, 16, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 2, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 40, 16, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 40, 8, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 48, 8, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 12, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 30, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 16, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 18, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Relu0,
        [((1, 36, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (Relu0, [((1, 128), torch.float32)], {"model_names": ["pt_autoencoder_linear_img_enc_github"], "pcc": 0.99}),
    (Relu0, [((1, 64), torch.float32)], {"model_names": ["pt_autoencoder_linear_img_enc_github"], "pcc": 0.99}),
    (Relu0, [((1, 12), torch.float32)], {"model_names": ["pt_autoencoder_linear_img_enc_github"], "pcc": 0.99}),
    (
        Relu0,
        [((1, 1056, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1088, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1120, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1152, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1184, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1216, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1248, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1280, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1312, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1376, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1408, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1440, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1472, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1504, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1536, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1568, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1600, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1632, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1664, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1696, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1728, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1760, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1792, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1056, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1088, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1120, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1152, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1184, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1216, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1248, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1280, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1312, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1344, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1376, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1408, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1440, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1472, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1504, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1536, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1568, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1600, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1632, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1664, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1696, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1728, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1760, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1792, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1824, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1856, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1888, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Relu0,
        [((1, 1920, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (Relu0, [((256, 3072), torch.float32)], {"model_names": ["pt_opt_facebook_opt_125m_clm_hf"], "pcc": 0.99}),
    (Relu0, [((32, 3072), torch.float32)], {"model_names": ["pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99}),
    (Relu0, [((1, 61, 2048), torch.float32)], {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99}),
    (Relu0, [((1, 513, 2048), torch.float32)], {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99}),
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
