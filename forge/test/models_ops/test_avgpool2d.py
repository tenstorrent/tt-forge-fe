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


class Avgpool2D0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, avgpool2d_input_0):
        avgpool2d_output_1 = forge.op.AvgPool2d(
            "",
            avgpool2d_input_0,
            kernel_size=[1, 1],
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
            channel_last=0,
        )
        return avgpool2d_output_1


class Avgpool2D1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, avgpool2d_input_0):
        avgpool2d_output_1 = forge.op.AvgPool2d(
            "",
            avgpool2d_input_0,
            kernel_size=[2, 2],
            stride=[2, 2],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
            channel_last=0,
        )
        return avgpool2d_output_1


class Avgpool2D2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, avgpool2d_input_0):
        avgpool2d_output_1 = forge.op.AvgPool2d(
            "",
            avgpool2d_input_0,
            kernel_size=[56, 56],
            stride=[56, 56],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
            channel_last=0,
        )
        return avgpool2d_output_1


class Avgpool2D3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, avgpool2d_input_0):
        avgpool2d_output_1 = forge.op.AvgPool2d(
            "",
            avgpool2d_input_0,
            kernel_size=[7, 7],
            stride=[7, 7],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
            channel_last=0,
        )
        return avgpool2d_output_1


class Avgpool2D4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, avgpool2d_input_0):
        avgpool2d_output_1 = forge.op.AvgPool2d(
            "",
            avgpool2d_input_0,
            kernel_size=[14, 14],
            stride=[14, 14],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
            channel_last=0,
        )
        return avgpool2d_output_1


class Avgpool2D5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, avgpool2d_input_0):
        avgpool2d_output_1 = forge.op.AvgPool2d(
            "",
            avgpool2d_input_0,
            kernel_size=[7, 7],
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
            channel_last=0,
        )
        return avgpool2d_output_1


class Avgpool2D6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, avgpool2d_input_0):
        avgpool2d_output_1 = forge.op.AvgPool2d(
            "",
            avgpool2d_input_0,
            kernel_size=[10, 10],
            stride=[10, 10],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
            channel_last=0,
        )
        return avgpool2d_output_1


class Avgpool2D7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, avgpool2d_input_0):
        avgpool2d_output_1 = forge.op.AvgPool2d(
            "",
            avgpool2d_input_0,
            kernel_size=[112, 112],
            stride=[112, 112],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
            channel_last=0,
        )
        return avgpool2d_output_1


class Avgpool2D8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, avgpool2d_input_0):
        avgpool2d_output_1 = forge.op.AvgPool2d(
            "",
            avgpool2d_input_0,
            kernel_size=[28, 28],
            stride=[28, 28],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
            channel_last=0,
        )
        return avgpool2d_output_1


class Avgpool2D9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, avgpool2d_input_0):
        avgpool2d_output_1 = forge.op.AvgPool2d(
            "",
            avgpool2d_input_0,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            ceil_mode=False,
            count_include_pad=False,
            channel_last=0,
        )
        return avgpool2d_output_1


class Avgpool2D10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, avgpool2d_input_0):
        avgpool2d_output_1 = forge.op.AvgPool2d(
            "",
            avgpool2d_input_0,
            kernel_size=[8, 8],
            stride=[8, 8],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
            channel_last=0,
        )
        return avgpool2d_output_1


class Avgpool2D11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, avgpool2d_input_0):
        avgpool2d_output_1 = forge.op.AvgPool2d(
            "",
            avgpool2d_input_0,
            kernel_size=[8, 8],
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
            channel_last=0,
        )
        return avgpool2d_output_1


class Avgpool2D12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, avgpool2d_input_0):
        avgpool2d_output_1 = forge.op.AvgPool2d(
            "",
            avgpool2d_input_0,
            kernel_size=[6, 6],
            stride=[6, 6],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
            channel_last=0,
        )
        return avgpool2d_output_1


class Avgpool2D13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, avgpool2d_input_0):
        avgpool2d_output_1 = forge.op.AvgPool2d(
            "",
            avgpool2d_input_0,
            kernel_size=[3, 3],
            stride=[3, 3],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
            channel_last=0,
        )
        return avgpool2d_output_1


class Avgpool2D14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, avgpool2d_input_0):
        avgpool2d_output_1 = forge.op.AvgPool2d(
            "",
            avgpool2d_input_0,
            kernel_size=[5, 5],
            stride=[5, 5],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
            channel_last=0,
        )
        return avgpool2d_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Avgpool2D0,
        [((1, 256, 6, 6), torch.float32)],
        {
            "model_name": ["pt_alexnet_alexnet_img_cls_torchhub", "pt_rcnn_base_obj_det_torchvision_rect_0"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[1, 1]",
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D1,
        [((1, 192, 56, 56), torch.float32)],
        {
            "model_name": ["pt_densenet_densenet161_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[2, 2]",
                "stride": "[2, 2]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D2,
        [((1, 192, 56, 56), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[56, 56]",
                "stride": "[56, 56]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D1,
        [((1, 384, 28, 28), torch.float32)],
        {
            "model_name": ["pt_densenet_densenet161_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[2, 2]",
                "stride": "[2, 2]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D1,
        [((1, 1056, 14, 14), torch.float32)],
        {
            "model_name": ["pt_densenet_densenet161_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[2, 2]",
                "stride": "[2, 2]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D3,
        [((1, 2208, 7, 7), torch.float32)],
        {
            "model_name": ["pt_densenet_densenet161_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[7, 7]",
                "stride": "[7, 7]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D1,
        [((1, 128, 56, 56), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[2, 2]",
                "stride": "[2, 2]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D2,
        [((1, 128, 56, 56), torch.float32)],
        {
            "model_name": ["pt_regnet_facebook_regnet_y_040_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[56, 56]",
                "stride": "[56, 56]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D1,
        [((1, 256, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[2, 2]",
                "stride": "[2, 2]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D1,
        [((1, 896, 14, 14), torch.float32)],
        {
            "model_name": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[2, 2]",
                "stride": "[2, 2]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D3,
        [((1, 1920, 7, 7), torch.float32)],
        {
            "model_name": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[7, 7]",
                "stride": "[7, 7]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D1,
        [((1, 640, 14, 14), torch.float32)],
        {
            "model_name": ["pt_densenet_densenet169_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[2, 2]",
                "stride": "[2, 2]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D3,
        [((1, 1664, 7, 7), torch.float32)],
        {
            "model_name": ["pt_densenet_densenet169_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[7, 7]",
                "stride": "[7, 7]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D1,
        [((1, 512, 14, 14), torch.float32)],
        {
            "model_name": ["pt_densenet_densenet121_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[2, 2]",
                "stride": "[2, 2]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D4,
        [((1, 512, 14, 14), torch.float32)],
        {
            "model_name": ["pt_regnet_facebook_regnet_y_040_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[14, 14]",
                "stride": "[14, 14]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D3,
        [((1, 1024, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenet_v1_basic_img_cls_torchvision",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[7, 7]",
                "stride": "[7, 7]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D5,
        [((1, 1024, 7, 7), torch.float32)],
        {
            "model_name": ["pt_vovnet_vovnet39_obj_det_osmr", "pt_vovnet_vovnet57_obj_det_osmr"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[7, 7]",
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D3,
        [((1, 512, 7, 7), torch.float32)],
        {
            "model_name": ["pt_dla_dla34_visual_bb_torchvision"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[7, 7]",
                "stride": "[7, 7]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D0,
        [((1, 512, 7, 7), torch.float32)],
        {
            "model_name": ["pt_vgg_19_obj_det_hf", "pt_vgg_vgg19_bn_obj_det_torchhub"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[1, 1]",
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D5,
        [((1, 512, 7, 7), torch.float32)],
        {
            "model_name": ["pt_vovnet_vovnet27s_obj_det_osmr"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[7, 7]",
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D3,
        [((1, 256, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[7, 7]",
                "stride": "[7, 7]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D6,
        [((1, 1792, 10, 10), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[10, 10]",
                "stride": "[10, 10]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D7,
        [((1, 32, 112, 112), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b0_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[112, 112]",
                "stride": "[112, 112]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D2,
        [((1, 96, 56, 56), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b0_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[56, 56]",
                "stride": "[56, 56]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D2,
        [((1, 144, 56, 56), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[56, 56]",
                "stride": "[56, 56]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D8,
        [((1, 144, 28, 28), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b0_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[28, 28]",
                "stride": "[28, 28]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D8,
        [((1, 240, 28, 28), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b0_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[28, 28]",
                "stride": "[28, 28]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D4,
        [((1, 240, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[14, 14]",
                "stride": "[14, 14]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D4,
        [((1, 480, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[14, 14]",
                "stride": "[14, 14]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D4,
        [((1, 672, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[14, 14]",
                "stride": "[14, 14]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D3,
        [((1, 672, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[7, 7]",
                "stride": "[7, 7]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D3,
        [((1, 1152, 7, 7), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b0_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[7, 7]",
                "stride": "[7, 7]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D3,
        [((1, 1280, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[7, 7]",
                "stride": "[7, 7]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D7,
        [((1, 48, 112, 112), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[112, 112]",
                "stride": "[112, 112]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D7,
        [((1, 24, 112, 112), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[112, 112]",
                "stride": "[112, 112]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D8,
        [((1, 192, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[28, 28]",
                "stride": "[28, 28]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D8,
        [((1, 336, 28, 28), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[28, 28]",
                "stride": "[28, 28]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D4,
        [((1, 336, 14, 14), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[14, 14]",
                "stride": "[14, 14]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D4,
        [((1, 960, 14, 14), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[14, 14]",
                "stride": "[14, 14]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D3,
        [((1, 960, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[7, 7]",
                "stride": "[7, 7]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D3,
        [((1, 1632, 7, 7), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[7, 7]",
                "stride": "[7, 7]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D3,
        [((1, 2688, 7, 7), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[7, 7]",
                "stride": "[7, 7]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D3,
        [((1, 1792, 7, 7), torch.float32)],
        {
            "model_name": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[7, 7]",
                "stride": "[7, 7]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D3,
        [((1, 2048, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "ResNetForImageClassification",
                "ResNet",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[7, 7]",
                "stride": "[7, 7]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D5,
        [((1, 2048, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
            ],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[7, 7]",
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    pytest.param(
        (
            Avgpool2D9,
            [((1, 384, 35, 35), torch.float32)],
            {
                "model_name": ["pt_inception_v4_img_cls_timm", "pt_inception_v4_img_cls_osmr"],
                "pcc": 0.99,
                "op_params": {
                    "kernel_size": "[3, 3]",
                    "stride": "[1, 1]",
                    "padding": "[1, 1, 1, 1]",
                    "ceil_mode": "False",
                    "count_include_pad": "False",
                    "channel_last": "0",
                },
            },
        ),
        marks=[pytest.mark.xfail(reason="RuntimeError: Tensor 1 - stride mismatch: expected [1225, 1], got [0, 0]")],
    ),
    pytest.param(
        (
            Avgpool2D9,
            [((1, 1024, 17, 17), torch.float32)],
            {
                "model_name": ["pt_inception_v4_img_cls_timm", "pt_inception_v4_img_cls_osmr"],
                "pcc": 0.99,
                "op_params": {
                    "kernel_size": "[3, 3]",
                    "stride": "[1, 1]",
                    "padding": "[1, 1, 1, 1]",
                    "ceil_mode": "False",
                    "count_include_pad": "False",
                    "channel_last": "0",
                },
            },
        ),
        marks=[pytest.mark.xfail(reason="RuntimeError: Tensor 1 - stride mismatch: expected [289, 1], got [0, 0]")],
    ),
    pytest.param(
        (
            Avgpool2D9,
            [((1, 1536, 8, 8), torch.float32)],
            {
                "model_name": ["pt_inception_v4_img_cls_timm", "pt_inception_v4_img_cls_osmr"],
                "pcc": 0.99,
                "op_params": {
                    "kernel_size": "[3, 3]",
                    "stride": "[1, 1]",
                    "padding": "[1, 1, 1, 1]",
                    "ceil_mode": "False",
                    "count_include_pad": "False",
                    "channel_last": "0",
                },
            },
        ),
        marks=[pytest.mark.xfail(reason="RuntimeError: Tensor 1 - stride mismatch: expected [64, 1], got [0, 0]")],
    ),
    (
        Avgpool2D10,
        [((1, 1536, 8, 8), torch.float32)],
        {
            "model_name": ["pt_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[8, 8]",
                "stride": "[8, 8]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D11,
        [((1, 1536, 8, 8), torch.float32)],
        {
            "model_name": ["pt_inception_v4_img_cls_osmr"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[8, 8]",
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D12,
        [((1, 768, 6, 6), torch.float32)],
        {
            "model_name": ["pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[6, 6]",
                "stride": "[6, 6]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D13,
        [((1, 1280, 3, 3), torch.float32)],
        {
            "model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[3, 3]",
                "stride": "[3, 3]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D14,
        [((1, 1280, 5, 5), torch.float32)],
        {
            "model_name": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[5, 5]",
                "stride": "[5, 5]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D8,
        [((1, 320, 28, 28), torch.float32)],
        {
            "model_name": ["pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[28, 28]",
                "stride": "[28, 28]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D2,
        [((1, 16, 56, 56), torch.float32)],
        {
            "model_name": ["pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[56, 56]",
                "stride": "[56, 56]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D4,
        [((1, 96, 14, 14), torch.float32)],
        {
            "model_name": ["pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[14, 14]",
                "stride": "[14, 14]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D4,
        [((1, 120, 14, 14), torch.float32)],
        {
            "model_name": ["pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[14, 14]",
                "stride": "[14, 14]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D4,
        [((1, 144, 14, 14), torch.float32)],
        {
            "model_name": ["pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[14, 14]",
                "stride": "[14, 14]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D3,
        [((1, 288, 7, 7), torch.float32)],
        {
            "model_name": ["pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[7, 7]",
                "stride": "[7, 7]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D3,
        [((1, 576, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[7, 7]",
                "stride": "[7, 7]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D8,
        [((1, 72, 28, 28), torch.float32)],
        {
            "model_name": ["pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[28, 28]",
                "stride": "[28, 28]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D8,
        [((1, 120, 28, 28), torch.float32)],
        {
            "model_name": ["pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[28, 28]",
                "stride": "[28, 28]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D3,
        [((1, 1088, 7, 7), torch.float32)],
        {
            "model_name": ["pt_regnet_facebook_regnet_y_040_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[7, 7]",
                "stride": "[7, 7]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D0,
        [((1, 4096, 1, 1), torch.float32)],
        {
            "model_name": ["pt_vgg_vgg19_bn_obj_det_timm"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[1, 1]",
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D6,
        [((1, 2048, 10, 10), torch.float32)],
        {
            "model_name": [
                "pt_xception_xception_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "[10, 10]",
                "stride": "[10, 10]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes, record_forge_property):
    record_forge_property("tags.op_name", "AvgPool2d")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes

    pcc = metadata.pop("pcc")

    for metadata_name, metadata_value in metadata.items():
        record_forge_property("tags." + str(metadata_name), metadata_value)

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

    compiled_model = compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model, VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)))
