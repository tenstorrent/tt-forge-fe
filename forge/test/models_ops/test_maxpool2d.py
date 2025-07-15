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


class Maxpool2D0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, maxpool2d_input_0):
        maxpool2d_output_1 = forge.op.MaxPool2d(
            "",
            maxpool2d_input_0,
            kernel_size=3,
            stride=2,
            padding=[1, 1, 1, 1],
            dilation=1,
            ceil_mode=False,
            channel_last=0,
        )
        return maxpool2d_output_1


class Maxpool2D1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, maxpool2d_input_0):
        maxpool2d_output_1 = forge.op.MaxPool2d(
            "",
            maxpool2d_input_0,
            kernel_size=2,
            stride=2,
            padding=[0, 0, 0, 0],
            dilation=1,
            ceil_mode=False,
            channel_last=0,
        )
        return maxpool2d_output_1


class Maxpool2D2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, maxpool2d_input_0):
        maxpool2d_output_1 = forge.op.MaxPool2d(
            "",
            maxpool2d_input_0,
            kernel_size=3,
            stride=2,
            padding=[0, 0, 0, 0],
            dilation=1,
            ceil_mode=True,
            channel_last=0,
        )
        return maxpool2d_output_1


class Maxpool2D3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, maxpool2d_input_0):
        maxpool2d_output_1 = forge.op.MaxPool2d(
            "",
            maxpool2d_input_0,
            kernel_size=5,
            stride=1,
            padding=[2, 2, 2, 2],
            dilation=1,
            ceil_mode=False,
            channel_last=0,
        )
        return maxpool2d_output_1


class Maxpool2D4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, maxpool2d_input_0):
        maxpool2d_output_1 = forge.op.MaxPool2d(
            "",
            maxpool2d_input_0,
            kernel_size=9,
            stride=1,
            padding=[4, 4, 4, 4],
            dilation=1,
            ceil_mode=False,
            channel_last=0,
        )
        return maxpool2d_output_1


class Maxpool2D5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, maxpool2d_input_0):
        maxpool2d_output_1 = forge.op.MaxPool2d(
            "",
            maxpool2d_input_0,
            kernel_size=13,
            stride=1,
            padding=[6, 6, 6, 6],
            dilation=1,
            ceil_mode=False,
            channel_last=0,
        )
        return maxpool2d_output_1


class Maxpool2D6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, maxpool2d_input_0):
        maxpool2d_output_1 = forge.op.MaxPool2d(
            "",
            maxpool2d_input_0,
            kernel_size=1,
            stride=2,
            padding=[0, 0, 0, 0],
            dilation=1,
            ceil_mode=False,
            channel_last=0,
        )
        return maxpool2d_output_1


class Maxpool2D7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, maxpool2d_input_0):
        maxpool2d_output_1 = forge.op.MaxPool2d(
            "",
            maxpool2d_input_0,
            kernel_size=3,
            stride=2,
            padding=[0, 0, 0, 0],
            dilation=1,
            ceil_mode=False,
            channel_last=0,
        )
        return maxpool2d_output_1


class Maxpool2D8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, maxpool2d_input_0):
        maxpool2d_output_1 = forge.op.MaxPool2d(
            "",
            maxpool2d_input_0,
            kernel_size=3,
            stride=2,
            padding=[0, 0, 1, 1],
            dilation=1,
            ceil_mode=False,
            channel_last=1,
        )
        return maxpool2d_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Maxpool2D0,
        [((1, 64, 112, 112), torch.float32)],
        {
            "model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D1,
        [((1, 32, 112, 112), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D1,
        [((1, 128, 56, 56), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D2,
        [((1, 128, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_vovnet_vovnet27s_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D1,
        [((1, 256, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D2,
        [((1, 256, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_vovnet_vovnet27s_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D1,
        [((1, 512, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vgg_vgg13_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D0,
        [((1, 64, 160, 160), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_ssd_resnet34_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D0,
        [((1, 64, 112, 112), torch.bfloat16)],
        {
            "model_names": [
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
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
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D0,
        [((1, 64, 240, 320), torch.bfloat16)],
        {
            "model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D1,
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
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D1,
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
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D1,
        [((1, 256, 56, 56), torch.bfloat16)],
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
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D2,
        [((1, 256, 56, 56), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D1,
        [((1, 512, 28, 28), torch.bfloat16)],
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
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D2,
        [((1, 512, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D2,
        [((1, 768, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D2,
        [((1, 384, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_vovnet_vovnet27s_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D3,
        [((1, 512, 20, 20), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_640x640"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "5",
                "stride": "1",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D3,
        [((1, 128, 15, 15), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "5",
                "stride": "1",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D3,
        [((1, 320, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolov8_yolov8x_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "5",
                "stride": "1",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D3,
        [((1, 512, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_l_obj_det_torchhub", "pt_yolox_yolox_darknet_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "5",
                "stride": "1",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D4,
        [((1, 512, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_l_obj_det_torchhub", "pt_yolox_yolox_darknet_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "9",
                "stride": "1",
                "padding": "[4, 4, 4, 4]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D5,
        [((1, 512, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_l_obj_det_torchhub", "pt_yolox_yolox_darknet_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "13",
                "stride": "1",
                "padding": "[6, 6, 6, 6]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D3,
        [((1, 384, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_m_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "5",
                "stride": "1",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D4,
        [((1, 384, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_m_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "9",
                "stride": "1",
                "padding": "[4, 4, 4, 4]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D5,
        [((1, 384, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_m_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "13",
                "stride": "1",
                "padding": "[6, 6, 6, 6]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D1,
        [((1, 64, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_dla_dla46_c_visual_bb_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D1,
        [((1, 64, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_dla_dla46_c_visual_bb_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D1,
        [((1, 128, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_dla_dla46_c_visual_bb_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D6,
        [((1, 256, 8, 8), torch.bfloat16)],
        {
            "model_names": ["pt_fpn_base_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "1",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D0,
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
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D5,
        [((1, 512, 15, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v4_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "13",
                "stride": "1",
                "padding": "[6, 6, 6, 6]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D4,
        [((1, 512, 15, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v4_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "9",
                "stride": "1",
                "padding": "[4, 4, 4, 4]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D3,
        [((1, 512, 15, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v4_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "5",
                "stride": "1",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D3,
        [((1, 640, 10, 10), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_320x320"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "5",
                "stride": "1",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D7,
        [((1, 64, 147, 147), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D7,
        [((1, 192, 71, 71), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D7,
        [((1, 384, 35, 35), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D7,
        [((1, 1024, 17, 17), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D0,
        [((1, 128, 147, 147), torch.bfloat16)],
        {
            "model_names": ["pt_xception_xception_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D0,
        [((1, 256, 74, 74), torch.bfloat16)],
        {
            "model_names": ["pt_xception_xception_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D0,
        [((1, 728, 37, 37), torch.bfloat16)],
        {
            "model_names": ["pt_xception_xception_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D0,
        [((1, 1024, 19, 19), torch.bfloat16)],
        {
            "model_names": ["pt_xception_xception_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D3,
        [((1, 384, 20, 20), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_640x640"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "5",
                "stride": "1",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D3,
        [((1, 256, 10, 10), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5s_img_cls_torchhub_320x320"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "5",
                "stride": "1",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D3,
        [((1, 512, 14, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "5",
                "stride": "1",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D0,
        [((1, 128, 159, 159), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D0,
        [((1, 256, 79, 79), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D0,
        [((1, 256, 39, 39), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D3,
        [((1, 256, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "5",
                "stride": "1",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D0,
        [((1, 128, 79, 79), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D8,
        [((1, 112, 109, 64), torch.float32)],
        {
            "model_names": ["tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 1, 1]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "1",
            },
        },
    ),
    (
        Maxpool2D2,
        [((1, 128, 56, 56), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet27s_obj_det_osmr"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D2,
        [((1, 256, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet27s_obj_det_osmr"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D2,
        [((1, 384, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet27s_obj_det_osmr"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D2,
        [((1, 256, 56, 56), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet_v1_57_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D2,
        [((1, 512, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet_v1_57_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D2,
        [((1, 768, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet_v1_57_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D3,
        [((1, 128, 20, 20), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "5",
                "stride": "1",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D7,
        [((1, 64, 55, 55), torch.float32)],
        {
            "model_names": ["pd_alexnet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D7,
        [((1, 192, 27, 27), torch.float32)],
        {
            "model_names": ["pd_alexnet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D7,
        [((1, 256, 13, 13), torch.float32)],
        {
            "model_names": ["pd_alexnet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D1,
        [((1, 288, 2, 50), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    record_forge_op_name("MaxPool2d")

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
