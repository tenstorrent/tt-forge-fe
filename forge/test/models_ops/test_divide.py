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


class Divide0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, divide_input_0, divide_input_1):
        divide_output_1 = forge.op.Divide("", divide_input_0, divide_input_1)
        return divide_output_1


class Divide1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("divide1_const_0", shape=(1,), dtype=torch.float32)

    def forward(self, divide_input_1):
        divide_output_1 = forge.op.Divide("", self.get_constant("divide1_const_0"), divide_input_1)
        return divide_output_1


class Divide2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("divide2_const_0", shape=(1,), dtype=torch.bfloat16)

    def forward(self, divide_input_1):
        divide_output_1 = forge.op.Divide("", self.get_constant("divide2_const_0"), divide_input_1)
        return divide_output_1


class Divide3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("divide3_const_1", shape=(1,), dtype=torch.bfloat16)

    def forward(self, divide_input_0):
        divide_output_1 = forge.op.Divide("", divide_input_0, self.get_constant("divide3_const_1"))
        return divide_output_1


class Divide4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("divide4_const_1", shape=(1,), dtype=torch.float32)

    def forward(self, divide_input_0):
        divide_output_1 = forge.op.Divide("", divide_input_0, self.get_constant("divide4_const_1"))
        return divide_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Divide0,
        [((1, 11, 128), torch.float32), ((1, 11, 1), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Divide0,
        [((1, 12, 11, 11), torch.float32), ((1, 12, 11, 1), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Divide0,
        [((1, 11, 312), torch.float32), ((1, 11, 1), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((64,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "tf_resnet_resnet50_img_cls_keras",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((256,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "tf_resnet_resnet50_img_cls_keras",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((128,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "tf_resnet_resnet50_img_cls_keras",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((512,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "tf_resnet_resnet50_img_cls_keras",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((1024,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "tf_resnet_resnet50_img_cls_keras",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((2048,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "tf_resnet_resnet50_img_cls_keras",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide2,
        [((16,), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((32,), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_regnet_regnet_y_128gf_img_cls_torchvision",
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_160_img_cls_hf",
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_regnet_regnet_x_32gf_img_cls_torchvision",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
                "pt_yolo_v4_default_obj_det_github",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_xception_xception_img_cls_timm",
                "pt_yolo_v3_default_obj_det_github",
                "pt_yolov9_default_obj_det_github",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((64,), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilenetv3_ssd_resnet34_img_cls_torchvision",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_yolo_v4_default_obj_det_github",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_xception_xception_img_cls_timm",
                "pt_yolo_v3_default_obj_det_github",
                "pt_yolov9_default_obj_det_github",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((128,), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilenetv3_ssd_resnet34_img_cls_torchvision",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_yolo_v4_default_obj_det_github",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_xception_xception_img_cls_timm",
                "pt_yolo_v3_default_obj_det_github",
                "pt_yolov9_default_obj_det_github",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((256,), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilenetv3_ssd_resnet34_img_cls_torchvision",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_fpn_base_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_yolo_v4_default_obj_det_github",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_xception_xception_img_cls_timm",
                "pt_yolo_v3_default_obj_det_github",
                "pt_yolov9_default_obj_det_github",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((512,), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilenetv3_ssd_resnet34_img_cls_torchvision",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_yolo_v4_default_obj_det_github",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_yolo_v3_default_obj_det_github",
                "pt_yolov9_default_obj_det_github",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((1024,), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_yolo_v4_default_obj_det_github",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_xception_xception_img_cls_timm",
                "pt_yolo_v3_default_obj_det_github",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
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
        Divide2,
        [((40,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((24,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((144,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((192,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((48,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((288,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((96,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((576,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((136,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((816,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((232,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((1392,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((384,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((2304,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((1536,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((56,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((336,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_regnet_regnet_x_32gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((112,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((672,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_regnet_regnet_x_32gf_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((160,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_yolov8_yolov8x_obj_det_github",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((960,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((272,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((1632,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((448,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_regnet_facebook_regnet_y_160_img_cls_hf",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((2688,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((1792,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((240,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((72,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((432,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((864,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((200,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((1200,), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((344,), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((2064,), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((3456,), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((30,), torch.bfloat16)],
        {"model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Divide2,
        [((60,), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm", "pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((120,), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((2048,), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((528,), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_regnet_y_128gf_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((1056,), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_regnet_y_128gf_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((2904,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((7392,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((224,), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_160_img_cls_hf",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((896,), torch.bfloat16)],
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
        Divide2,
        [((2016,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((1, 1, 16384, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((1, 2, 4096, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((1, 5, 1024, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((1, 8, 256, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((1, 256, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((1, 512, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((768,), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((1, 768, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((1, 1024, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((80,), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_yolov8_yolov8x_obj_det_github",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((728,), torch.bfloat16)],
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
        Divide2,
        [((320,), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov8_yolov8x_obj_det_github",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((640,), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov8_yolov8x_obj_det_github",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((1, 2, 8400), torch.bfloat16)],
        {
            "model_names": ["pt_yolov8_yolov8x_obj_det_github", "pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide4,
        [((1, 1, 16384, 256), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Divide4,
        [((1, 16384, 256), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Divide4,
        [((1, 2, 4096, 256), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Divide4,
        [((1, 4096, 512), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Divide4,
        [((1, 5, 1024, 256), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Divide4,
        [((1, 1024, 1280), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Divide4,
        [((1, 8, 256, 256), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Divide4,
        [((1, 256, 2048), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Divide0,
        [((1, 9, 768), torch.float32), ((1, 9, 1), torch.float32)],
        {
            "model_names": [
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 12, 9, 9), torch.float32), ((1, 12, 9, 1), torch.float32)],
        {
            "model_names": [
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide2,
        [((352,), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((416,), torch.bfloat16)],
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
        Divide2,
        [((480,), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((544,), torch.bfloat16)],
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
        Divide2,
        [((608,), torch.bfloat16)],
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
        Divide2,
        [((704,), torch.bfloat16)],
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
        Divide2,
        [((736,), torch.bfloat16)],
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
        Divide2,
        [((800,), torch.bfloat16)],
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
        Divide2,
        [((832,), torch.bfloat16)],
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
        Divide2,
        [((928,), torch.bfloat16)],
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
        Divide2,
        [((992,), torch.bfloat16)],
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
        Divide2,
        [((1344,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_regnet_regnet_x_32gf_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((3840,), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((2560,), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((8,), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((12,), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((36,), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((1, 72, 1, 1), torch.bfloat16)],
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
        Divide2,
        [((20,), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((1, 120, 1, 1), torch.bfloat16)],
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
        Divide2,
        [((100,), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((92,), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((1, 480, 1, 1), torch.bfloat16)],
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
        Divide3,
        [((1, 672, 1, 1), torch.bfloat16)],
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
        Divide3,
        [((1, 960, 1, 1), torch.bfloat16)],
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
        Divide2,
        [((18,), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((720,), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((1280,), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((1, 16, 112, 112), torch.bfloat16)],
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
        Divide3,
        [((1, 240, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((1, 240, 14, 14), torch.bfloat16)],
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
        Divide3,
        [((1, 200, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((184,), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((1, 184, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((1, 480, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((1, 672, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((1, 672, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((1, 960, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((1, 1280, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((1, 3, 192, 640), torch.bfloat16)],
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
        Divide2,
        [((1232,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((3024,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((696,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((3712,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((2520,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((1008,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_3_2gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide0,
        [((64, 3, 64, 32), torch.float32), ((64, 3, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Divide0,
        [((16, 6, 64, 32), torch.float32), ((16, 6, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Divide0,
        [((4, 12, 64, 32), torch.float32), ((4, 12, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Divide0,
        [((1, 24, 64, 32), torch.float32), ((1, 24, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Divide2,
        [((176,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((304,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((1824,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((3072,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((1, 16, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((88,), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((1, 96, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((1, 96, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((1, 96, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((1, 240, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((1, 120, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((1, 144, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((1, 144, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((1, 288, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((1, 288, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((1, 288, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((1, 576, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((1, 576, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide4,
        [((1, 12, 128, 128), torch.float32)],
        {"model_names": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Divide3,
        [((1, 5880, 2), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Divide4,
        [((1, 197, 768), torch.float32)],
        {"model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (Divide4, [((1, 2, 8400), torch.float32)], {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99}),
    (
        Divide1,
        [((8,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Divide4,
        [((1, 8, 16, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((40,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((16,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((48,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Divide4,
        [((1, 48, 8, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Divide4,
        [((1, 48, 4, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((24,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((120,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Divide4,
        [((1, 120, 4, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Divide4,
        [((1, 64, 4, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((72,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Divide4,
        [((1, 72, 4, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((144,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Divide4,
        [((1, 144, 4, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Divide4,
        [((1, 144, 2, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((288,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Divide4,
        [((1, 288, 2, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Divide0,
        [((1, 25, 6625), torch.float32), ((1, 25, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Divide4,
        [((1, 16, 256, 256), torch.float32)],
        {"model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"], "pcc": 0.99},
    ),
    (
        Divide2,
        [((1088,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((1120,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((1152,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((1184,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((1216,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((1248,), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((1312,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((1376,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((1408,), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((1440,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((1472,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((1504,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((1568,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((1600,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((1664,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((1696,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((1728,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((1760,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((1856,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((1888,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((1920,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((208,), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((2112,), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    record_forge_op_name("Divide")

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
