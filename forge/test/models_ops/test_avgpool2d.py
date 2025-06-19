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


class Avgpool2D0(ForgeModule):
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


class Avgpool2D1(ForgeModule):
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


class Avgpool2D2(ForgeModule):
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


class Avgpool2D3(ForgeModule):
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
            count_include_pad=False,
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
            kernel_size=[112, 112],
            stride=[112, 112],
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
            kernel_size=[56, 56],
            stride=[56, 56],
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
            kernel_size=[2, 2],
            stride=[2, 2],
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
            kernel_size=[28, 28],
            stride=[28, 28],
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
            kernel_size=[14, 14],
            stride=[14, 14],
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
            kernel_size=[10, 10],
            stride=[10, 10],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
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
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            ceil_mode=False,
            count_include_pad=False,
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
            stride=[8, 8],
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
            kernel_size=[8, 8],
            stride=[1, 1],
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
            kernel_size=[6, 6],
            stride=[6, 6],
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
            stride=[3, 3],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=False,
            channel_last=0,
        )
        return avgpool2d_output_1


class Avgpool2D15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, avgpool2d_input_0):
        avgpool2d_output_1 = forge.op.AvgPool2d(
            "",
            avgpool2d_input_0,
            kernel_size=[16, 50],
            stride=[16, 50],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
            channel_last=0,
        )
        return avgpool2d_output_1


class Avgpool2D16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, avgpool2d_input_0):
        avgpool2d_output_1 = forge.op.AvgPool2d(
            "",
            avgpool2d_input_0,
            kernel_size=[4, 50],
            stride=[4, 50],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
            channel_last=0,
        )
        return avgpool2d_output_1


class Avgpool2D17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, avgpool2d_input_0):
        avgpool2d_output_1 = forge.op.AvgPool2d(
            "",
            avgpool2d_input_0,
            kernel_size=[2, 50],
            stride=[2, 50],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
            channel_last=0,
        )
        return avgpool2d_output_1


class Avgpool2D18(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, avgpool2d_input_0):
        avgpool2d_output_1 = forge.op.AvgPool2d(
            "",
            avgpool2d_input_0,
            kernel_size=[15, 15],
            stride=[15, 15],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
            channel_last=0,
        )
        return avgpool2d_output_1


class Avgpool2D19(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, avgpool2d_input_0):
        avgpool2d_output_1 = forge.op.AvgPool2d(
            "",
            avgpool2d_input_0,
            kernel_size=[30, 30],
            stride=[30, 30],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
            channel_last=0,
        )
        return avgpool2d_output_1


class Avgpool2D20(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, avgpool2d_input_0):
        avgpool2d_output_1 = forge.op.AvgPool2d(
            "",
            avgpool2d_input_0,
            kernel_size=[60, 60],
            stride=[60, 60],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
            channel_last=0,
        )
        return avgpool2d_output_1


class Avgpool2D21(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, avgpool2d_input_0):
        avgpool2d_output_1 = forge.op.AvgPool2d(
            "",
            avgpool2d_input_0,
            kernel_size=[120, 120],
            stride=[120, 120],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
            channel_last=0,
        )
        return avgpool2d_output_1


class Avgpool2D22(ForgeModule):
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
            count_include_pad=False,
            channel_last=0,
        )
        return avgpool2d_output_1


class Avgpool2D23(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, avgpool2d_input_0):
        avgpool2d_output_1 = forge.op.AvgPool2d(
            "",
            avgpool2d_input_0,
            kernel_size=[9, 9],
            stride=[9, 9],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
            channel_last=0,
        )
        return avgpool2d_output_1


class Avgpool2D24(ForgeModule):
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
            count_include_pad=False,
            channel_last=1,
        )
        return avgpool2d_output_1


class Avgpool2D25(ForgeModule):
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
            count_include_pad=False,
            channel_last=0,
        )
        return avgpool2d_output_1


class Avgpool2D26(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, avgpool2d_input_0):
        avgpool2d_output_1 = forge.op.AvgPool2d(
            "",
            avgpool2d_input_0,
            kernel_size=[9, 9],
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=False,
            channel_last=0,
        )
        return avgpool2d_output_1


class Avgpool2D27(ForgeModule):
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


class Avgpool2D28(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, avgpool2d_input_0):
        avgpool2d_output_1 = forge.op.AvgPool2d(
            "",
            avgpool2d_input_0,
            kernel_size=[4, 25],
            stride=[4, 25],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
            channel_last=0,
        )
        return avgpool2d_output_1


class Avgpool2D29(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, avgpool2d_input_0):
        avgpool2d_output_1 = forge.op.AvgPool2d(
            "",
            avgpool2d_input_0,
            kernel_size=[2, 2],
            stride=[3, 2],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=False,
            channel_last=0,
        )
        return avgpool2d_output_1


class Avgpool2D30(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, avgpool2d_input_0):
        avgpool2d_output_1 = forge.op.AvgPool2d(
            "",
            avgpool2d_input_0,
            kernel_size=[12, 12],
            stride=[12, 12],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
            channel_last=0,
        )
        return avgpool2d_output_1


class Avgpool2D31(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, avgpool2d_input_0):
        avgpool2d_output_1 = forge.op.AvgPool2d(
            "",
            avgpool2d_input_0,
            kernel_size=[10, 10],
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=False,
            channel_last=0,
        )
        return avgpool2d_output_1


class Avgpool2D32(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, avgpool2d_input_0):
        avgpool2d_output_1 = forge.op.AvgPool2d(
            "",
            avgpool2d_input_0,
            kernel_size=[2, 2],
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
            channel_last=0,
        )
        return avgpool2d_output_1


class Avgpool2D33(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, avgpool2d_input_0):
        avgpool2d_output_1 = forge.op.AvgPool2d(
            "",
            avgpool2d_input_0,
            kernel_size=[14, 14],
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=False,
            channel_last=0,
        )
        return avgpool2d_output_1


class Avgpool2D34(ForgeModule):
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


class Avgpool2D35(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, avgpool2d_input_0):
        avgpool2d_output_1 = forge.op.AvgPool2d(
            "",
            avgpool2d_input_0,
            kernel_size=[96, 96],
            stride=[96, 96],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
            channel_last=0,
        )
        return avgpool2d_output_1


class Avgpool2D36(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, avgpool2d_input_0):
        avgpool2d_output_1 = forge.op.AvgPool2d(
            "",
            avgpool2d_input_0,
            kernel_size=[48, 48],
            stride=[48, 48],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
            channel_last=0,
        )
        return avgpool2d_output_1


class Avgpool2D37(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, avgpool2d_input_0):
        avgpool2d_output_1 = forge.op.AvgPool2d(
            "",
            avgpool2d_input_0,
            kernel_size=[24, 24],
            stride=[24, 24],
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
        [((1, 672, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_regnet_x_800mf_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {
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
        [((1, 2048, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_resnet_50_img_cls_hf",
                "pt_regnet_regnet_x_16gf_img_cls_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_resnet_50_img_cls_timm",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {
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
        [((1, 2048, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D2,
        [((1, 512, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_vgg_vgg11_bn_img_cls_torchvision",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vgg_vgg19_img_cls_torchvision",
                "pt_vgg_vgg16_bn_img_cls_torchvision",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_vgg13_img_cls_torchvision",
                "pt_vgg_vgg13_bn_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D0,
        [((1, 512, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla34_in1k_img_cls_timm",
                "pt_resnet_resnet18_img_cls_torchvision",
                "pt_resnet_resnet34_img_cls_torchvision",
                "pt_dla_dla34_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {
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
        [((1, 512, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_vovnet_vovnet27s_img_cls_osmr"],
            "pcc": 0.99,
            "args": {
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
        [((1, 1024, 7, 7), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla102_visual_bb_torchvision",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "onnx_vovnet_v1_vovnet39_obj_det_torchhub",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[7, 7]",
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D0,
        [((1, 1024, 7, 7), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv1_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
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
            "model_names": ["pd_resnet_18_img_cls_paddlemodels", "pd_resnet_34_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
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
        [((1, 512, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_dla_dla34_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[7, 7]",
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D1,
        [((1, 512, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet27s_obj_det_osmr"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D0,
        [((1, 256, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {
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
        [((1, 1024, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_swin_swin_b_img_cls_torchvision",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "args": {
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
        [((1, 1024, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_vovnet_vovnet39_img_cls_osmr", "pt_vovnet_vovnet57_img_cls_osmr"],
            "pcc": 0.99,
            "args": {
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
            Avgpool2D4,
            [((1, 40, 112, 112), torch.bfloat16)],
            {
                "model_names": ["pt_efficientnet_efficientnet_b3_img_cls_torchvision"],
                "pcc": 0.99,
                "args": {
                    "kernel_size": "[112, 112]",
                    "stride": "[112, 112]",
                    "padding": "[0, 0, 0, 0]",
                    "ceil_mode": "False",
                    "count_include_pad": "True",
                    "channel_last": "0",
                },
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Avgpool2D4,
            [((1, 24, 112, 112), torch.bfloat16)],
            {
                "model_names": [
                    "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "args": {
                    "kernel_size": "[112, 112]",
                    "stride": "[112, 112]",
                    "padding": "[0, 0, 0, 0]",
                    "ceil_mode": "False",
                    "count_include_pad": "True",
                    "channel_last": "0",
                },
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Avgpool2D5,
            [((1, 144, 56, 56), torch.bfloat16)],
            {
                "model_names": [
                    "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                    "pt_regnet_facebook_regnet_y_064_img_cls_hf",
                    "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "args": {
                    "kernel_size": "[56, 56]",
                    "stride": "[56, 56]",
                    "padding": "[0, 0, 0, 0]",
                    "ceil_mode": "False",
                    "count_include_pad": "True",
                    "channel_last": "0",
                },
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Avgpool2D5,
            [((1, 192, 56, 56), torch.bfloat16)],
            {
                "model_names": [
                    "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "args": {
                    "kernel_size": "[56, 56]",
                    "stride": "[56, 56]",
                    "padding": "[0, 0, 0, 0]",
                    "ceil_mode": "False",
                    "count_include_pad": "True",
                    "channel_last": "0",
                },
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Avgpool2D6,
        [((1, 192, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet161_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[2, 2]",
                "stride": "[2, 2]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    pytest.param(
        (
            Avgpool2D7,
            [((1, 192, 28, 28), torch.bfloat16)],
            {
                "model_names": [
                    "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                    "pt_regnet_facebook_regnet_y_040_img_cls_hf",
                ],
                "pcc": 0.99,
                "args": {
                    "kernel_size": "[28, 28]",
                    "stride": "[28, 28]",
                    "padding": "[0, 0, 0, 0]",
                    "ceil_mode": "False",
                    "count_include_pad": "True",
                    "channel_last": "0",
                },
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Avgpool2D7,
        [((1, 288, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_064_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {
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
        [((1, 288, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D8,
        [((1, 576, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_regnet_regnet_y_3_2gf_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_064_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D8,
        [((1, 816, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b3_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D0,
        [((1, 816, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b3_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
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
        [((1, 1392, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b3_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
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
        [((1, 2304, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {
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
        [((1, 1536, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b3_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D9,
        [((1, 1280, 10, 10), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D0,
        [((1, 960, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D10,
        [((1, 384, 35, 35), torch.bfloat16)],
        {
            "model_names": [
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[3, 3]",
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "ceil_mode": "False",
                "count_include_pad": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D10,
        [((1, 1024, 17, 17), torch.bfloat16)],
        {
            "model_names": [
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[3, 3]",
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "ceil_mode": "False",
                "count_include_pad": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D10,
        [((1, 1536, 8, 8), torch.bfloat16)],
        {
            "model_names": [
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[3, 3]",
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "ceil_mode": "False",
                "count_include_pad": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D11,
        [((1, 1536, 8, 8), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D12,
        [((1, 1536, 8, 8), torch.bfloat16)],
        {
            "model_names": ["pt_inception_v4_img_cls_osmr"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D7,
        [((1, 320, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[28, 28]",
                "stride": "[28, 28]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    pytest.param(
        (
            Avgpool2D5,
            [((1, 16, 56, 56), torch.bfloat16)],
            {
                "model_names": ["pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"],
                "pcc": 0.99,
                "args": {
                    "kernel_size": "[56, 56]",
                    "stride": "[56, 56]",
                    "padding": "[0, 0, 0, 0]",
                    "ceil_mode": "False",
                    "count_include_pad": "True",
                    "channel_last": "0",
                },
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Avgpool2D8,
        [((1, 96, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D8,
        [((1, 240, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D8,
        [((1, 120, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D8,
        [((1, 144, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D0,
        [((1, 288, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"],
            "pcc": 0.99,
            "args": {
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
        [((1, 576, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
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
        [((1, 912, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_1_6gf_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
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
        [((1, 400, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_400mf_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
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
        [((1, 1920, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_regnet_x_8gf_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {
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
        [((1, 256, 7, 7), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla60x_c_visual_bb_torchvision",
                "onnx_dla_dla46x_c_visual_bb_torchvision",
                "onnx_dla_dla46_c_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[7, 7]",
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D3,
        [((1, 1280, 7, 7), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_lite0_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[7, 7]",
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D0,
        [((1, 1280, 7, 7), torch.float32)],
        {
            "model_names": ["pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
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
            "model_names": ["onnx_resnet_50_img_cls_hf"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[7, 7]",
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D0,
        [((1, 2048, 7, 7), torch.float32)],
        {
            "model_names": [
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_50_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D13,
        [((1, 1024, 6, 6), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D14,
        [((1, 512, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[5, 5]",
                "stride": "[3, 3]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D14,
        [((1, 528, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[5, 5]",
                "stride": "[3, 3]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D15,
        [((1, 8, 16, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[16, 50]",
                "stride": "[16, 50]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D16,
        [((1, 48, 4, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[4, 50]",
                "stride": "[4, 50]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D16,
        [((1, 120, 4, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[4, 50]",
                "stride": "[4, 50]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D16,
        [((1, 64, 4, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[4, 50]",
                "stride": "[4, 50]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D16,
        [((1, 72, 4, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[4, 50]",
                "stride": "[4, 50]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D17,
        [((1, 144, 2, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[2, 50]",
                "stride": "[2, 50]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D17,
        [((1, 288, 2, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[2, 50]",
                "stride": "[2, 50]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    pytest.param(
        (
            Avgpool2D4,
            [((1, 32, 112, 112), torch.bfloat16)],
            {
                "model_names": [
                    "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "args": {
                    "kernel_size": "[112, 112]",
                    "stride": "[112, 112]",
                    "padding": "[0, 0, 0, 0]",
                    "ceil_mode": "False",
                    "count_include_pad": "True",
                    "channel_last": "0",
                },
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Avgpool2D5,
            [((1, 96, 56, 56), torch.bfloat16)],
            {
                "model_names": [
                    "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "args": {
                    "kernel_size": "[56, 56]",
                    "stride": "[56, 56]",
                    "padding": "[0, 0, 0, 0]",
                    "ceil_mode": "False",
                    "count_include_pad": "True",
                    "channel_last": "0",
                },
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Avgpool2D7,
            [((1, 144, 28, 28), torch.bfloat16)],
            {
                "model_names": [
                    "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                    "pt_regnet_regnet_y_800mf_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "args": {
                    "kernel_size": "[28, 28]",
                    "stride": "[28, 28]",
                    "padding": "[0, 0, 0, 0]",
                    "ceil_mode": "False",
                    "count_include_pad": "True",
                    "channel_last": "0",
                },
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Avgpool2D7,
            [((1, 240, 28, 28), torch.bfloat16)],
            {
                "model_names": [
                    "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "args": {
                    "kernel_size": "[28, 28]",
                    "stride": "[28, 28]",
                    "padding": "[0, 0, 0, 0]",
                    "ceil_mode": "False",
                    "count_include_pad": "True",
                    "channel_last": "0",
                },
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Avgpool2D8,
        [((1, 480, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D8,
        [((1, 672, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D0,
        [((1, 1152, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {
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
        [((1, 1280, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[7, 7]",
                "stride": "[7, 7]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    pytest.param(
        (
            Avgpool2D4,
            [((1, 48, 112, 112), torch.bfloat16)],
            {
                "model_names": [
                    "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "args": {
                    "kernel_size": "[112, 112]",
                    "stride": "[112, 112]",
                    "padding": "[0, 0, 0, 0]",
                    "ceil_mode": "False",
                    "count_include_pad": "True",
                    "channel_last": "0",
                },
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Avgpool2D5,
            [((1, 240, 56, 56), torch.bfloat16)],
            {
                "model_names": [
                    "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "args": {
                    "kernel_size": "[56, 56]",
                    "stride": "[56, 56]",
                    "padding": "[0, 0, 0, 0]",
                    "ceil_mode": "False",
                    "count_include_pad": "True",
                    "channel_last": "0",
                },
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Avgpool2D7,
        [((1, 384, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b5_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D6,
        [((1, 384, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet161_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D8,
        [((1, 384, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b5_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D8,
        [((1, 768, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b5_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D8,
        [((1, 1056, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b5_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D6,
        [((1, 1056, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet161_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D0,
        [((1, 1056, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b5_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
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
        [((1, 1824, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b5_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
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
        [((1, 3072, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b5_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D9,
        [((1, 512, 10, 10), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_ssd_resnet18_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet34_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[10, 10]",
                "stride": "[10, 10]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    pytest.param(
        (
            Avgpool2D5,
            [((1, 48, 56, 56), torch.bfloat16)],
            {
                "model_names": [
                    "pt_regnet_regnet_y_1_6gf_img_cls_torchvision",
                    "pt_regnet_regnet_y_400mf_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "args": {
                    "kernel_size": "[56, 56]",
                    "stride": "[56, 56]",
                    "padding": "[0, 0, 0, 0]",
                    "ceil_mode": "False",
                    "count_include_pad": "True",
                    "channel_last": "0",
                },
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Avgpool2D7,
        [((1, 120, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_regnet_y_1_6gf_img_cls_torchvision",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "args": {
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
        [((1, 336, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_regnet_y_1_6gf_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D0,
        [((1, 888, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_1_6gf_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[7, 7]",
                "stride": "[7, 7]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    pytest.param(
        (
            Avgpool2D5,
            [((1, 72, 56, 56), torch.bfloat16)],
            {
                "model_names": ["pt_regnet_regnet_y_3_2gf_img_cls_torchvision"],
                "pcc": 0.99,
                "args": {
                    "kernel_size": "[56, 56]",
                    "stride": "[56, 56]",
                    "padding": "[0, 0, 0, 0]",
                    "ceil_mode": "False",
                    "count_include_pad": "True",
                    "channel_last": "0",
                },
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Avgpool2D7,
            [((1, 216, 28, 28), torch.bfloat16)],
            {
                "model_names": ["pt_regnet_regnet_y_3_2gf_img_cls_torchvision"],
                "pcc": 0.99,
                "args": {
                    "kernel_size": "[28, 28]",
                    "stride": "[28, 28]",
                    "padding": "[0, 0, 0, 0]",
                    "ceil_mode": "False",
                    "count_include_pad": "True",
                    "channel_last": "0",
                },
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Avgpool2D0,
        [((1, 1512, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_3_2gf_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[7, 7]",
                "stride": "[7, 7]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    pytest.param(
        (
            Avgpool2D5,
            [((1, 64, 56, 56), torch.bfloat16)],
            {
                "model_names": ["pt_regnet_regnet_y_800mf_img_cls_torchvision"],
                "pcc": 0.99,
                "args": {
                    "kernel_size": "[56, 56]",
                    "stride": "[56, 56]",
                    "padding": "[0, 0, 0, 0]",
                    "ceil_mode": "False",
                    "count_include_pad": "True",
                    "channel_last": "0",
                },
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Avgpool2D8,
        [((1, 320, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_800mf_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D0,
        [((1, 784, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_800mf_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D9,
        [((1, 2048, 10, 10), torch.bfloat16)],
        {
            "model_names": [
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_xception_xception71_img_cls_timm",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D18,
        [((1, 192, 15, 15), torch.float32)],
        {
            "model_names": ["TranslatedLayer"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[15, 15]",
                "stride": "[15, 15]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D18,
        [((1, 384, 15, 15), torch.float32)],
        {
            "model_names": ["TranslatedLayer"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[15, 15]",
                "stride": "[15, 15]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D18,
        [((1, 96, 15, 15), torch.float32)],
        {
            "model_names": ["TranslatedLayer"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[15, 15]",
                "stride": "[15, 15]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D18,
        [((1, 24, 15, 15), torch.float32)],
        {
            "model_names": ["TranslatedLayer"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[15, 15]",
                "stride": "[15, 15]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D19,
        [((1, 96, 30, 30), torch.float32)],
        {
            "model_names": ["TranslatedLayer"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[30, 30]",
                "stride": "[30, 30]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D19,
        [((1, 24, 30, 30), torch.float32)],
        {
            "model_names": ["TranslatedLayer"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[30, 30]",
                "stride": "[30, 30]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D20,
        [((1, 96, 60, 60), torch.float32)],
        {
            "model_names": ["TranslatedLayer"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[60, 60]",
                "stride": "[60, 60]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D20,
        [((1, 24, 60, 60), torch.float32)],
        {
            "model_names": ["TranslatedLayer"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[60, 60]",
                "stride": "[60, 60]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D21,
        [((1, 96, 120, 120), torch.float32)],
        {
            "model_names": ["TranslatedLayer"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[120, 120]",
                "stride": "[120, 120]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D21,
        [((1, 24, 120, 120), torch.float32)],
        {
            "model_names": ["TranslatedLayer"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[120, 120]",
                "stride": "[120, 120]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D22,
        [((1, 128, 56, 56), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[2, 2]",
                "stride": "[2, 2]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D22,
        [((1, 256, 28, 28), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[2, 2]",
                "stride": "[2, 2]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D22,
        [((1, 512, 14, 14), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[2, 2]",
                "stride": "[2, 2]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "False",
                "channel_last": "0",
            },
        },
    ),
    pytest.param(
        (
            Avgpool2D4,
            [((1, 16, 112, 112), torch.bfloat16)],
            {
                "model_names": [
                    "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "args": {
                    "kernel_size": "[112, 112]",
                    "stride": "[112, 112]",
                    "padding": "[0, 0, 0, 0]",
                    "ceil_mode": "False",
                    "count_include_pad": "True",
                    "channel_last": "0",
                },
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Avgpool2D8,
        [((1, 528, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D8,
        [((1, 720, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D0,
        [((1, 720, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
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
        [((1, 1248, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
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
        [((1, 2112, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
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
        [((1, 1408, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[7, 7]",
                "stride": "[7, 7]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    pytest.param(
        (
            Avgpool2D4,
            [((1, 56, 112, 112), torch.bfloat16)],
            {
                "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
                "pcc": 0.99,
                "args": {
                    "kernel_size": "[112, 112]",
                    "stride": "[112, 112]",
                    "padding": "[0, 0, 0, 0]",
                    "ceil_mode": "False",
                    "count_include_pad": "True",
                    "channel_last": "0",
                },
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Avgpool2D7,
        [((1, 432, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
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
        [((1, 432, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D8,
        [((1, 864, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D8,
        [((1, 1200, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D0,
        [((1, 1200, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
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
        [((1, 2064, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
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
        [((1, 3456, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D9,
        [((1, 1792, 10, 10), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D23,
        [((1, 1792, 9, 9), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[9, 9]",
                "stride": "[9, 9]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D7,
        [((1, 104, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_400mf_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
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
        [((1, 208, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_400mf_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D0,
        [((1, 440, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_400mf_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
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
        [((1, 768, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D24,
        [((1, 7, 7, 2048), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[7, 7]",
                "stride": "[7, 7]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "False",
                "channel_last": "1",
            },
        },
    ),
    (
        Avgpool2D25,
        [((1, 1280, 8, 8), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[8, 8]",
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D26,
        [((1, 1536, 9, 9), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[9, 9]",
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D6,
        [((1, 128, 56, 56), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[2, 2]",
                "stride": "[2, 2]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    pytest.param(
        (
            Avgpool2D5,
            [((1, 128, 56, 56), torch.bfloat16)],
            {
                "model_names": ["pt_regnet_facebook_regnet_y_040_img_cls_hf"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
                "args": {
                    "kernel_size": "[56, 56]",
                    "stride": "[56, 56]",
                    "padding": "[0, 0, 0, 0]",
                    "ceil_mode": "False",
                    "count_include_pad": "True",
                    "channel_last": "0",
                },
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Avgpool2D6,
        [((1, 256, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D6,
        [((1, 512, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D8,
        [((1, 512, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_040_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
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
        Avgpool2D27,
        [((1, 1280, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[3, 3]",
                "stride": "[3, 3]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    pytest.param(
        (
            Avgpool2D5,
            [((1, 224, 56, 56), torch.bfloat16)],
            {
                "model_names": [
                    "pt_regnet_facebook_regnet_y_120_img_cls_hf",
                    "pt_regnet_regnet_y_16gf_img_cls_torchvision",
                    "pt_regnet_facebook_regnet_y_160_img_cls_hf",
                    "pt_regnet_regnet_y_8gf_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "args": {
                    "kernel_size": "[56, 56]",
                    "stride": "[56, 56]",
                    "padding": "[0, 0, 0, 0]",
                    "ceil_mode": "False",
                    "count_include_pad": "True",
                    "channel_last": "0",
                },
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Avgpool2D7,
        [((1, 448, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_facebook_regnet_y_120_img_cls_hf",
                "pt_regnet_regnet_y_16gf_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_160_img_cls_hf",
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_080_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {
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
        [((1, 896, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_facebook_regnet_y_120_img_cls_hf",
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_080_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D6,
        [((1, 896, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D0,
        [((1, 2240, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_120_img_cls_hf"],
            "pcc": 0.99,
            "args": {
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
        [((1, 1232, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_regnet_y_16gf_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_160_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D0,
        [((1, 3024, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_regnet_y_16gf_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_160_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D28,
        [((1, 240, 4, 25), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[4, 25]",
                "stride": "[4, 25]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D28,
        [((1, 480, 4, 25), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[4, 25]",
                "stride": "[4, 25]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D29,
        [((1, 480, 2, 25), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[2, 2]",
                "stride": "[3, 2]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D0,
        [((1, 2208, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet161_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D23,
        [((1, 1280, 9, 9), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[9, 9]",
                "stride": "[9, 9]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D0,
        [((1, 2016, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_8gf_img_cls_torchvision", "pt_regnet_facebook_regnet_y_080_img_cls_hf"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D2,
        [((1, 4096, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_vgg_vgg19_bn_obj_det_timm"],
            "pcc": 0.99,
            "args": {
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
        [((1, 640, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet169_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D0,
        [((1, 1664, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet169_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
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
        [((1, 72, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub"],
            "pcc": 0.99,
            "args": {
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
        [((1, 192, 14, 14), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D8,
        [((1, 384, 14, 14), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D8,
        [((1, 96, 14, 14), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D8,
        [((1, 24, 14, 14), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D7,
        [((1, 96, 28, 28), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D7,
        [((1, 24, 28, 28), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D5,
        [((1, 96, 56, 56), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D5,
        [((1, 24, 56, 56), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
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
        [((1, 96, 112, 112), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D4,
        [((1, 24, 112, 112), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[112, 112]",
                "stride": "[112, 112]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    pytest.param(
        (
            Avgpool2D4,
            [((1, 64, 112, 112), torch.bfloat16)],
            {
                "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
                "pcc": 0.99,
                "args": {
                    "kernel_size": "[112, 112]",
                    "stride": "[112, 112]",
                    "padding": "[0, 0, 0, 0]",
                    "ceil_mode": "False",
                    "count_include_pad": "True",
                    "channel_last": "0",
                },
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Avgpool2D5,
            [((1, 288, 56, 56), torch.bfloat16)],
            {
                "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
                "pcc": 0.99,
                "args": {
                    "kernel_size": "[56, 56]",
                    "stride": "[56, 56]",
                    "padding": "[0, 0, 0, 0]",
                    "ceil_mode": "False",
                    "count_include_pad": "True",
                    "channel_last": "0",
                },
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Avgpool2D7,
        [((1, 480, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
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
        [((1, 960, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D8,
        [((1, 1344, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D0,
        [((1, 1344, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
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
        [((1, 3840, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
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
        [((1, 2560, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D30,
        [((1, 1280, 12, 12), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[12, 12]",
                "stride": "[12, 12]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D6,
        [((1, 24, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D6,
        [((1, 40, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D6,
        [((1, 80, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D6,
        [((1, 112, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D6,
        [((1, 160, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D0,
        [((1, 1296, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_064_img_cls_hf"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[7, 7]",
                "stride": "[7, 7]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    pytest.param(
        (
            Avgpool2D5,
            [((1, 232, 56, 56), torch.bfloat16)],
            {
                "model_names": [
                    "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                    "pt_regnet_regnet_y_32gf_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "args": {
                    "kernel_size": "[56, 56]",
                    "stride": "[56, 56]",
                    "padding": "[0, 0, 0, 0]",
                    "ceil_mode": "False",
                    "count_include_pad": "True",
                    "channel_last": "0",
                },
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Avgpool2D7,
        [((1, 696, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_regnet_regnet_y_32gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {
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
        [((1, 1392, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_regnet_regnet_y_32gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D0,
        [((1, 3712, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_regnet_regnet_y_32gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D11,
        [((1, 1024, 8, 8), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
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
        [((1, 768, 8, 8), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D25,
        [((1, 1408, 8, 8), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[8, 8]",
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D31,
        [((1, 1792, 10, 10), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[10, 10]",
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D7,
        [((1, 336, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D0,
        [((1, 1632, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
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
        [((1, 2688, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
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
        [((1, 1792, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D11,
        [((1, 1280, 8, 8), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D0,
        [((1, 1008, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_3_2gf_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
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
        Avgpool2D32,
        [((1, 256, 160, 160), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[2, 2]",
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D32,
        [((1, 512, 80, 80), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[2, 2]",
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D32,
        [((1, 512, 40, 40), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[2, 2]",
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D32,
        [((1, 256, 80, 80), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[2, 2]",
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D33,
        [((1, 2048, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "[14, 14]",
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D2,
        [((1, 256, 6, 6), torch.bfloat16)],
        {
            "model_names": ["pt_alexnet_base_img_cls_torchhub", "pt_rcnn_base_obj_det_torchvision_rect_0"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
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
        Avgpool2D8,
        [((1, 2048, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
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
        Avgpool2D13,
        [((1, 768, 6, 6), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
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
        Avgpool2D34,
        [((1, 1280, 5, 5), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
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
        Avgpool2D0,
        [((1, 1088, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_040_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "[7, 7]",
                "stride": "[7, 7]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    pytest.param(
        (
            Avgpool2D5,
            [((1, 168, 56, 56), torch.bfloat16)],
            {
                "model_names": ["pt_regnet_facebook_regnet_y_080_img_cls_hf"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
                "args": {
                    "kernel_size": "[56, 56]",
                    "stride": "[56, 56]",
                    "padding": "[0, 0, 0, 0]",
                    "ceil_mode": "False",
                    "count_include_pad": "True",
                    "channel_last": "0",
                },
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Avgpool2D0,
        [((1, 2520, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "[7, 7]",
                "stride": "[7, 7]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    pytest.param(
        (
            Avgpool2D35,
            [((1, 528, 96, 96), torch.bfloat16)],
            {
                "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
                "args": {
                    "kernel_size": "[96, 96]",
                    "stride": "[96, 96]",
                    "padding": "[0, 0, 0, 0]",
                    "ceil_mode": "False",
                    "count_include_pad": "True",
                    "channel_last": "0",
                },
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Avgpool2D36,
            [((1, 1056, 48, 48), torch.bfloat16)],
            {
                "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
                "args": {
                    "kernel_size": "[48, 48]",
                    "stride": "[48, 48]",
                    "padding": "[0, 0, 0, 0]",
                    "ceil_mode": "False",
                    "count_include_pad": "True",
                    "channel_last": "0",
                },
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Avgpool2D37,
        [((1, 2904, 24, 24), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "[24, 24]",
                "stride": "[24, 24]",
                "padding": "[0, 0, 0, 0]",
                "ceil_mode": "False",
                "count_include_pad": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Avgpool2D30,
        [((1, 7392, 12, 12), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "[12, 12]",
                "stride": "[12, 12]",
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
def test_module(forge_module_and_shapes_dtypes):

    record_forge_op_name("AvgPool2d")

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

    compiler_cfg = forge.config.CompilerConfig()
    if "default_df_override" in metadata.keys():
        compiler_cfg.default_df_override = forge.DataFormat.from_json(metadata["default_df_override"])

    compiled_model = compile(framework_model, sample_inputs=inputs, compiler_cfg=compiler_cfg)

    verify(inputs, framework_model, compiled_model, VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)))
