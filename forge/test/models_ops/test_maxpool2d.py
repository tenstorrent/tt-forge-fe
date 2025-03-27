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


class Maxpool2D0(ForgeModule):
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


class Maxpool2D1(ForgeModule):
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
            kernel_size=2,
            stride=2,
            padding=[0, 0, 0, 0],
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
            kernel_size=3,
            stride=1,
            padding=[1, 1, 1, 1],
            dilation=1,
            ceil_mode=True,
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
            kernel_size=1,
            stride=2,
            padding=[0, 0, 0, 0],
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
            kernel_size=2,
            stride=2,
            padding=[0, 0, 0, 0],
            dilation=1,
            ceil_mode=True,
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
            kernel_size=5,
            stride=1,
            padding=[2, 2, 2, 2],
            dilation=1,
            ceil_mode=False,
            channel_last=0,
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
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {
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
        [((1, 64, 112, 112), torch.float32)],
        {
            "model_name": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_resnet_resnet34_img_cls_torchvision",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_resnet_resnet18_img_cls_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_resnet_resnet152_img_cls_torchvision",
                "ResNetForImageClassification",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {
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
        Maxpool2D2,
        [((1, 64, 112, 112), torch.float32)],
        {
            "model_name": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {
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
        Maxpool2D0,
        [((1, 64, 55, 55), torch.float32)],
        {
            "model_name": ["pt_alexnet_alexnet_img_cls_torchhub"],
            "pcc": 0.99,
            "op_params": {
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
        [((1, 192, 27, 27), torch.float32)],
        {
            "model_name": ["pt_alexnet_alexnet_img_cls_torchhub", "pt_rcnn_base_obj_det_torchvision_rect_0"],
            "pcc": 0.99,
            "op_params": {
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
        [((1, 256, 13, 13), torch.float32)],
        {
            "model_name": ["pt_alexnet_alexnet_img_cls_torchhub", "pt_rcnn_base_obj_det_torchvision_rect_0"],
            "pcc": 0.99,
            "op_params": {
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
        Maxpool2D2,
        [((1, 256, 13, 13), torch.float32)],
        {
            "model_name": ["pt_alexnet_base_img_cls_osmr"],
            "pcc": 0.99,
            "op_params": {
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
        [((1, 96, 54, 54), torch.float32)],
        {
            "model_name": ["pt_alexnet_base_img_cls_osmr"],
            "pcc": 0.99,
            "op_params": {
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
        [((1, 256, 27, 27), torch.float32)],
        {
            "model_name": ["pt_alexnet_base_img_cls_osmr"],
            "pcc": 0.99,
            "op_params": {
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
        [((1, 16, 28, 28), torch.float32)],
        {
            "model_name": ["pt_autoencoder_conv_img_enc_github"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    pytest.param(
        (
            Maxpool2D3,
            [((1, 4, 14, 14), torch.float32)],
            {
                "model_name": ["pt_autoencoder_conv_img_enc_github"],
                "pcc": 0.99,
                "op_params": {
                    "kernel_size": "2",
                    "stride": "2",
                    "padding": "[0, 0, 0, 0]",
                    "dilation": "1",
                    "ceil_mode": "False",
                    "channel_last": "0",
                },
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_FATAL @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/data_movement/sharded/interleaved_to_sharded/device/interleaved_to_sharded_op.cpp:24: (*this->output_mem_config.shard_spec).shape[1] * input_tensor.element_size() % hal::get_l1_alignment() == 0 info: Shard page size must currently have L1 aligned page size"
            )
        ],
    ),
    (
        Maxpool2D1,
        [((1, 96, 112, 112), torch.float32)],
        {
            "model_name": ["pt_densenet_densenet161_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {
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
        [((1, 32, 112, 112), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla34_in1k_img_cls_timm",
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_monodle_base_obj_det_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {
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
        Maxpool2D3,
        [((1, 128, 56, 56), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {
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
        [((1, 128, 56, 56), torch.float32)],
        {
            "model_name": ["pt_vovnet_vovnet27s_obj_det_osmr"],
            "pcc": 0.99,
            "op_params": {
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
        [((1, 256, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {
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
        Maxpool2D4,
        [((1, 256, 28, 28), torch.float32)],
        {
            "model_name": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "3",
                "stride": "1",
                "padding": "[1, 1, 1, 1]",
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
            "model_name": ["pt_vovnet_vovnet27s_obj_det_osmr"],
            "pcc": 0.99,
            "op_params": {
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
        [((1, 512, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_vgg_vgg16_bn_img_cls_torchvision",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg11_bn_img_cls_torchvision",
                "pt_vgg_vgg13_img_cls_torchvision",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vgg_vgg11_obj_det_osmr",
                "pt_vgg_vgg13_obj_det_osmr",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vgg_vgg13_bn_img_cls_torchvision",
                "pt_vgg_vgg19_img_cls_torchvision",
                "pt_vgg_vgg19_obj_det_osmr",
                "pt_vgg_vgg19_bn_obj_det_timm",
                "pt_vgg_bn_vgg19_obj_det_osmr",
            ],
            "pcc": 0.99,
            "op_params": {
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
        Maxpool2D4,
        [((1, 512, 14, 14), torch.float32)],
        {
            "model_name": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "3",
                "stride": "1",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "ceil_mode": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D3,
        [((1, 64, 56, 56), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla34_in1k_img_cls_timm",
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_monodle_base_obj_det_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {
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
        [((1, 64, 56, 56), torch.float32)],
        {
            "model_name": ["pt_rcnn_base_obj_det_torchvision_rect_0"],
            "pcc": 0.99,
            "op_params": {
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
        Maxpool2D3,
        [((1, 128, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla34_in1k_img_cls_timm",
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_monodle_base_obj_det_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {
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
        Maxpool2D3,
        [((1, 256, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla34_in1k_img_cls_timm",
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_monodle_base_obj_det_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {
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
        Maxpool2D3,
        [((1, 64, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {
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
        Maxpool2D3,
        [((1, 128, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {
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
        Maxpool2D5,
        [((1, 256, 8, 8), torch.float32)],
        {
            "model_name": ["pt_fpn_base_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {
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
        Maxpool2D2,
        [((1, 192, 56, 56), torch.float32)],
        {
            "model_name": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {
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
        Maxpool2D4,
        [((1, 192, 28, 28), torch.float32)],
        {
            "model_name": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "3",
                "stride": "1",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "ceil_mode": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D2,
        [((1, 480, 28, 28), torch.float32)],
        {
            "model_name": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {
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
        Maxpool2D4,
        [((1, 480, 14, 14), torch.float32)],
        {
            "model_name": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "3",
                "stride": "1",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "ceil_mode": "True",
                "channel_last": "0",
            },
        },
    ),
    pytest.param(
        (
            Maxpool2D4,
            [((1, 528, 14, 14), torch.float32)],
            {
                "model_name": ["pt_googlenet_base_img_cls_torchvision"],
                "pcc": 0.99,
                "op_params": {
                    "kernel_size": "3",
                    "stride": "1",
                    "padding": "[1, 1, 1, 1]",
                    "dilation": "1",
                    "ceil_mode": "True",
                    "channel_last": "0",
                },
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_FATAL @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/pool/generic/device/pool_op.cpp:37: (input_shape[3] % tt::constants::TILE_WIDTH == 0) || (input_shape[3] == 16) info: Input channels (528) should be padded to nearest TILE_WIDTH (32) or should be 16"
            )
        ],
    ),
    (
        Maxpool2D6,
        [((1, 832, 14, 14), torch.float32)],
        {
            "model_name": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D4,
        [((1, 832, 7, 7), torch.float32)],
        {
            "model_name": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "3",
                "stride": "1",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "ceil_mode": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D0,
        [((1, 64, 147, 147), torch.float32)],
        {
            "model_name": [
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {
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
        [((1, 192, 71, 71), torch.float32)],
        {
            "model_name": [
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {
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
        [((1, 384, 35, 35), torch.float32)],
        {
            "model_name": [
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {
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
        [((1, 1024, 17, 17), torch.float32)],
        {
            "model_name": [
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {
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
        Maxpool2D3,
        [((1, 64, 24, 24), torch.float32)],
        {
            "model_name": ["pt_mnist_base_img_cls_github"],
            "pcc": 0.99,
            "op_params": {
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
        [((1, 64, 160, 160), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet18_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet34_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {
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
        [((1, 64, 160, 512), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {
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
        [((1, 64, 96, 320), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {
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
        [((1, 64, 240, 320), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
            ],
            "pcc": 0.99,
            "op_params": {
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
        [((1, 64, 150, 150), torch.float32)],
        {
            "model_name": ["pt_ssd300_resnet50_base_img_cls_torchhub"],
            "pcc": 0.99,
            "op_params": {
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
        [((1, 32, 256, 256), torch.float32)],
        {
            "model_name": ["pt_unet_base_img_seg_torchhub"],
            "pcc": 0.99,
            "op_params": {
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
        Maxpool2D3,
        [((1, 64, 128, 128), torch.float32)],
        {
            "model_name": ["pt_unet_base_img_seg_torchhub"],
            "pcc": 0.99,
            "op_params": {
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
        Maxpool2D3,
        [((1, 128, 64, 64), torch.float32)],
        {
            "model_name": ["pt_unet_base_img_seg_torchhub"],
            "pcc": 0.99,
            "op_params": {
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
        Maxpool2D3,
        [((1, 256, 32, 32), torch.float32)],
        {
            "model_name": ["pt_unet_base_img_seg_torchhub"],
            "pcc": 0.99,
            "op_params": {
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
        Maxpool2D3,
        [((1, 64, 224, 224), torch.float32)],
        {
            "model_name": [
                "pt_unet_carvana_base_img_seg_github",
                "pt_unet_cityscape_img_seg_osmr",
                "pt_vgg_vgg16_bn_img_cls_torchvision",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg11_bn_img_cls_torchvision",
                "pt_vgg_vgg13_img_cls_torchvision",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vgg_vgg11_obj_det_osmr",
                "pt_vgg_vgg13_obj_det_osmr",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vgg_vgg13_bn_img_cls_torchvision",
                "pt_vgg_vgg19_img_cls_torchvision",
                "pt_vgg_vgg19_obj_det_osmr",
                "pt_vgg_vgg19_bn_obj_det_timm",
                "pt_vgg_bn_vgg19_obj_det_osmr",
            ],
            "pcc": 0.99,
            "op_params": {
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
        Maxpool2D3,
        [((1, 128, 112, 112), torch.float32)],
        {
            "model_name": [
                "pt_unet_carvana_base_img_seg_github",
                "pt_unet_cityscape_img_seg_osmr",
                "pt_vgg_vgg16_bn_img_cls_torchvision",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg11_bn_img_cls_torchvision",
                "pt_vgg_vgg13_img_cls_torchvision",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vgg_vgg11_obj_det_osmr",
                "pt_vgg_vgg13_obj_det_osmr",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vgg_vgg13_bn_img_cls_torchvision",
                "pt_vgg_vgg19_img_cls_torchvision",
                "pt_vgg_vgg19_obj_det_osmr",
                "pt_vgg_vgg19_bn_obj_det_timm",
                "pt_vgg_bn_vgg19_obj_det_osmr",
            ],
            "pcc": 0.99,
            "op_params": {
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
        Maxpool2D3,
        [((1, 256, 56, 56), torch.float32)],
        {
            "model_name": [
                "pt_unet_carvana_base_img_seg_github",
                "pt_unet_cityscape_img_seg_osmr",
                "pt_vgg_vgg16_bn_img_cls_torchvision",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg11_bn_img_cls_torchvision",
                "pt_vgg_vgg13_img_cls_torchvision",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vgg_vgg11_obj_det_osmr",
                "pt_vgg_vgg13_obj_det_osmr",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vgg_vgg13_bn_img_cls_torchvision",
                "pt_vgg_vgg19_img_cls_torchvision",
                "pt_vgg_vgg19_obj_det_osmr",
                "pt_vgg_vgg19_bn_obj_det_timm",
                "pt_vgg_bn_vgg19_obj_det_osmr",
            ],
            "pcc": 0.99,
            "op_params": {
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
        [((1, 256, 56, 56), torch.float32)],
        {
            "model_name": [
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_vovnet57_obj_det_osmr",
                "pt_vovnet_vovnet39_obj_det_osmr",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {
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
        [((1, 512, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_unet_carvana_base_img_seg_github",
                "pt_unet_cityscape_img_seg_osmr",
                "pt_vgg_vgg16_bn_img_cls_torchvision",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg11_bn_img_cls_torchvision",
                "pt_vgg_vgg13_img_cls_torchvision",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vgg_vgg11_obj_det_osmr",
                "pt_vgg_vgg13_obj_det_osmr",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vgg_vgg13_bn_img_cls_torchvision",
                "pt_vgg_vgg19_img_cls_torchvision",
                "pt_vgg_vgg19_obj_det_osmr",
                "pt_vgg_vgg19_bn_obj_det_timm",
                "pt_vgg_bn_vgg19_obj_det_osmr",
            ],
            "pcc": 0.99,
            "op_params": {
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
        [((1, 512, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_vovnet57_obj_det_osmr",
                "pt_vovnet_vovnet39_obj_det_osmr",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {
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
            "model_name": [
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_vovnet57_obj_det_osmr",
                "pt_vovnet_vovnet39_obj_det_osmr",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {
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
            "model_name": ["pt_vovnet_vovnet27s_obj_det_osmr"],
            "pcc": 0.99,
            "op_params": {
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
        [((1, 128, 147, 147), torch.float32)],
        {
            "model_name": ["pt_xception_xception_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {
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
        [((1, 256, 74, 74), torch.float32)],
        {
            "model_name": ["pt_xception_xception_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    pytest.param(
        (
            Maxpool2D1,
            [((1, 728, 37, 37), torch.float32)],
            {
                "model_name": ["pt_xception_xception_img_cls_timm"],
                "pcc": 0.99,
                "op_params": {
                    "kernel_size": "3",
                    "stride": "2",
                    "padding": "[1, 1, 1, 1]",
                    "dilation": "1",
                    "ceil_mode": "False",
                    "channel_last": "0",
                },
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_FATAL @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/pool/generic/device/pool_op.cpp:37: (input_shape[3] % tt::constants::TILE_WIDTH == 0) || (input_shape[3] == 16) info: Input channels (728) should be padded to nearest TILE_WIDTH (32) or should be 16"
            )
        ],
    ),
    (
        Maxpool2D1,
        [((1, 1024, 19, 19), torch.float32)],
        {
            "model_name": ["pt_xception_xception_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {
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
        Maxpool2D7,
        [((1, 128, 10, 10), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_320x320"],
            "pcc": 0.99,
            "op_params": {
                "kernel_size": "5",
                "stride": "1",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes, forge_property_recorder):
    forge_property_recorder("tags.op_name", "MaxPool2d")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes

    pcc = metadata.pop("pcc")

    for metadata_name, metadata_value in metadata.items():
        forge_property_recorder("tags." + str(metadata_name), metadata_value)

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

    compiled_model = compile(framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder)

    verify(
        inputs,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
        forge_property_handler=forge_property_recorder,
    )
