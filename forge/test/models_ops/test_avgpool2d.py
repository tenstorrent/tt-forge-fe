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
            kernel_size=[15, 15],
            stride=[15, 15],
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
            kernel_size=[30, 30],
            stride=[30, 30],
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
            kernel_size=[60, 60],
            stride=[60, 60],
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
            kernel_size=[120, 120],
            stride=[120, 120],
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
            kernel_size=[7, 7],
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=False,
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
            stride=[7, 7],
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
            kernel_size=[9, 9],
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=False,
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
            kernel_size=[7, 7],
            stride=[1, 1],
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
            kernel_size=[4, 25],
            stride=[4, 25],
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
            kernel_size=[2, 2],
            stride=[3, 2],
            padding=[0, 0, 0, 0],
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
            kernel_size=[14, 14],
            stride=[14, 14],
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
            kernel_size=[28, 28],
            stride=[28, 28],
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
            kernel_size=[56, 56],
            stride=[56, 56],
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
            kernel_size=[112, 112],
            stride=[112, 112],
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
            kernel_size=[7, 7],
            stride=[7, 7],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=False,
            channel_last=1,
        )
        return avgpool2d_output_1


class Avgpool2D15(ForgeModule):
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


class Avgpool2D16(ForgeModule):
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


class Avgpool2D17(ForgeModule):
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


class Avgpool2D18(ForgeModule):
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


class Avgpool2D19(ForgeModule):
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


class Avgpool2D20(ForgeModule):
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


class Avgpool2D21(ForgeModule):
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


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Avgpool2D0,
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
        Avgpool2D0,
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
        Avgpool2D0,
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
        Avgpool2D0,
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
        Avgpool2D1,
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
        Avgpool2D1,
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
        Avgpool2D2,
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
        Avgpool2D2,
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
        Avgpool2D3,
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
        Avgpool2D3,
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
        Avgpool2D4,
        [((1, 1024, 7, 7), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
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
        Avgpool2D5,
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
        Avgpool2D4,
        [((1, 256, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_dla_dla46x_c_visual_bb_torchvision", "onnx_dla_dla60x_c_visual_bb_torchvision"],
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
        Avgpool2D6,
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
        Avgpool2D7,
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
        Avgpool2D5,
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
        Avgpool2D4,
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
        Avgpool2D8,
        [((1, 240, 4, 25), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
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
        Avgpool2D8,
        [((1, 480, 4, 25), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
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
        Avgpool2D9,
        [((1, 480, 2, 25), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
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
        Avgpool2D5,
        [((1, 2048, 7, 7), torch.float32)],
        {
            "model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"],
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
        Avgpool2D4,
        [((1, 1280, 7, 7), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
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
        Avgpool2D5,
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
        Avgpool2D10,
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
        Avgpool2D10,
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
        Avgpool2D10,
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
        Avgpool2D10,
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
        Avgpool2D11,
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
        Avgpool2D11,
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
        Avgpool2D12,
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
        Avgpool2D12,
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
        Avgpool2D13,
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
        Avgpool2D13,
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
    (
        Avgpool2D14,
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
        Avgpool2D15,
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
        Avgpool2D16,
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
        Avgpool2D17,
        [((1, 8, 16, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
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
        Avgpool2D18,
        [((1, 48, 4, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
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
        Avgpool2D18,
        [((1, 120, 4, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
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
        Avgpool2D18,
        [((1, 64, 4, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
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
        Avgpool2D18,
        [((1, 72, 4, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
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
        Avgpool2D19,
        [((1, 144, 2, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
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
        Avgpool2D19,
        [((1, 288, 2, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
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
        Avgpool2D15,
        [((1, 1408, 8, 8), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"],
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
        Avgpool2D20,
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
        Avgpool2D21,
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
        Avgpool2D21,
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
        Avgpool2D21,
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

    compiled_model = compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model, VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)))
