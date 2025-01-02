# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import forge
import forge.op
from forge import ForgeModule

from loguru import logger
import torch

from forge import Tensor, compile
from forge.verify.compare import compare_with_golden
from forge.verify.verify import verify
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
            max_pool_add_sub_surround=False,
            max_pool_add_sub_surround_value=1.0,
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
            max_pool_add_sub_surround=False,
            max_pool_add_sub_surround_value=1.0,
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
            padding=[1, 1, 1, 1],
            dilation=1,
            ceil_mode=False,
            max_pool_add_sub_surround=False,
            max_pool_add_sub_surround_value=1.0,
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
            kernel_size=3,
            stride=2,
            padding=[0, 0, 0, 0],
            dilation=1,
            ceil_mode=True,
            max_pool_add_sub_surround=False,
            max_pool_add_sub_surround_value=1.0,
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
            max_pool_add_sub_surround=False,
            max_pool_add_sub_surround_value=1.0,
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
            max_pool_add_sub_surround=False,
            max_pool_add_sub_surround_value=1.0,
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
            max_pool_add_sub_surround=False,
            max_pool_add_sub_surround_value=1.0,
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
            max_pool_add_sub_surround=False,
            max_pool_add_sub_surround_value=1.0,
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
            kernel_size=9,
            stride=1,
            padding=[4, 4, 4, 4],
            dilation=1,
            ceil_mode=False,
            max_pool_add_sub_surround=False,
            max_pool_add_sub_surround_value=1.0,
            channel_last=0,
        )
        return maxpool2d_output_1


class Maxpool2D9(ForgeModule):
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
            max_pool_add_sub_surround=False,
            max_pool_add_sub_surround_value=1.0,
            channel_last=0,
        )
        return maxpool2d_output_1


def ids_func(param):
    forge_module, shapes_dtypes, _ = param
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (Maxpool2D0, [((1, 64, 55, 55), torch.float32)], {"model_name": ["pt_alexnet_torchhub"]}),
    (Maxpool2D0, [((1, 192, 27, 27), torch.float32)], {"model_name": ["pt_alexnet_torchhub", "pt_rcnn"]}),
    (Maxpool2D0, [((1, 256, 13, 13), torch.float32)], {"model_name": ["pt_alexnet_torchhub", "pt_rcnn"]}),
    (Maxpool2D1, [((1, 16, 28, 28), torch.float32)], {"model_name": ["pt_conv_ae"]}),
    (Maxpool2D1, [((1, 4, 14, 14), torch.float32)], {"model_name": ["pt_conv_ae"]}),
    (
        Maxpool2D2,
        [((1, 64, 112, 112), torch.float32)],
        {
            "model_name": [
                "pt_densenet121",
                "pt_densenet_169",
                "pt_densenet_201",
                "pt_resnet50_timm",
                "pt_resnet50",
                "pt_resnext50_torchhub",
                "pt_resnext101_torchhub",
                "pt_resnext14_osmr",
                "pt_resnext26_osmr",
                "pt_resnext101_fb_wsl",
                "pt_resnext101_osmr",
                "pt_resnext50_osmr",
                "pt_unet_qubvel_pt",
                "pt_wide_resnet101_2_timm",
                "pt_wide_resnet50_2_hub",
                "pt_wide_resnet50_2_timm",
                "pt_wide_resnet101_2_hub",
            ]
        },
    ),
    (Maxpool2D3, [((1, 64, 112, 112), torch.float32)], {"model_name": ["pt_googlenet"]}),
    (Maxpool2D0, [((1, 64, 112, 112), torch.float32)], {"model_name": ["pt_vision_perceiver_conv"]}),
    (Maxpool2D2, [((1, 96, 112, 112), torch.float32)], {"model_name": ["pt_densenet_161"]}),
    (
        Maxpool2D1,
        [((1, 32, 112, 112), torch.float32)],
        {
            "model_name": [
                "pt_dla102x",
                "pt_dla169",
                "pt_dla60",
                "pt_dla46_c",
                "pt_dla102x2",
                "pt_dla46x_c",
                "pt_dla60x",
                "pt_dla102",
                "pt_dla34",
                "pt_dla60x_c",
                "pt_monodle",
            ]
        },
    ),
    (
        Maxpool2D1,
        [((1, 128, 56, 56), torch.float32)],
        {"model_name": ["pt_dla102x", "pt_dla169", "pt_dla60", "pt_dla102x2", "pt_dla60x", "pt_dla102"]},
    ),
    (Maxpool2D3, [((1, 128, 56, 56), torch.float32)], {"model_name": ["pt_vovnet27s"]}),
    (
        Maxpool2D1,
        [((1, 256, 28, 28), torch.float32)],
        {"model_name": ["pt_dla102x", "pt_dla169", "pt_dla60", "pt_dla102x2", "pt_dla60x", "pt_dla102"]},
    ),
    (Maxpool2D4, [((1, 256, 28, 28), torch.float32)], {"model_name": ["pt_googlenet"]}),
    (Maxpool2D3, [((1, 256, 28, 28), torch.float32)], {"model_name": ["pt_vovnet27s"]}),
    (
        Maxpool2D1,
        [((1, 512, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_dla102x",
                "pt_dla169",
                "pt_dla60",
                "pt_dla102x2",
                "pt_dla60x",
                "pt_dla102",
                "pt_vgg13_osmr",
                "pt_bn_vgg19_osmr",
                "pt_bn_vgg19b_osmr",
                "pt_vgg16_osmr",
                "pt_vgg19_osmr",
                "pt_vgg_19_hf",
                "pt_vgg_bn19_torchhub",
                "pt_vgg11_osmr",
                "pt_vgg19_bn_timm",
            ]
        },
    ),
    (Maxpool2D4, [((1, 512, 14, 14), torch.float32)], {"model_name": ["pt_googlenet"]}),
    (
        Maxpool2D1,
        [((1, 64, 56, 56), torch.float32)],
        {"model_name": ["pt_dla46_c", "pt_dla46x_c", "pt_dla34", "pt_dla60x_c", "pt_monodle"]},
    ),
    (Maxpool2D0, [((1, 64, 56, 56), torch.float32)], {"model_name": ["pt_rcnn"]}),
    (Maxpool2D1, [((1, 64, 28, 28), torch.float32)], {"model_name": ["pt_dla46_c", "pt_dla46x_c", "pt_dla60x_c"]}),
    (Maxpool2D1, [((1, 128, 14, 14), torch.float32)], {"model_name": ["pt_dla46_c", "pt_dla46x_c", "pt_dla60x_c"]}),
    (Maxpool2D1, [((1, 128, 28, 28), torch.float32)], {"model_name": ["pt_dla34", "pt_monodle"]}),
    (Maxpool2D1, [((1, 256, 14, 14), torch.float32)], {"model_name": ["pt_dla34", "pt_monodle"]}),
    (Maxpool2D5, [((1, 256, 8, 8), torch.float32)], {"model_name": ["pt_fpn"]}),
    (Maxpool2D3, [((1, 192, 56, 56), torch.float32)], {"model_name": ["pt_googlenet"]}),
    (Maxpool2D4, [((1, 192, 28, 28), torch.float32)], {"model_name": ["pt_googlenet"]}),
    (Maxpool2D3, [((1, 480, 28, 28), torch.float32)], {"model_name": ["pt_googlenet"]}),
    (Maxpool2D4, [((1, 480, 14, 14), torch.float32)], {"model_name": ["pt_googlenet"]}),
    (Maxpool2D4, [((1, 528, 14, 14), torch.float32)], {"model_name": ["pt_googlenet"]}),
    (Maxpool2D6, [((1, 832, 14, 14), torch.float32)], {"model_name": ["pt_googlenet"]}),
    (Maxpool2D4, [((1, 832, 7, 7), torch.float32)], {"model_name": ["pt_googlenet"]}),
    (
        Maxpool2D0,
        [((1, 64, 147, 147), torch.float32)],
        {"model_name": ["pt_timm_inception_v4", "pt_osmr_inception_v4"]},
    ),
    (Maxpool2D0, [((1, 192, 71, 71), torch.float32)], {"model_name": ["pt_timm_inception_v4", "pt_osmr_inception_v4"]}),
    (Maxpool2D0, [((1, 384, 35, 35), torch.float32)], {"model_name": ["pt_timm_inception_v4", "pt_osmr_inception_v4"]}),
    (
        Maxpool2D0,
        [((1, 1024, 17, 17), torch.float32)],
        {"model_name": ["pt_timm_inception_v4", "pt_osmr_inception_v4"]},
    ),
    (
        Maxpool2D2,
        [((1, 64, 240, 320), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_rn18fpn",
                "pt_retinanet_rn152fpn",
                "pt_retinanet_rn101fpn",
                "pt_retinanet_rn50fpn",
                "pt_retinanet_rn34fpn",
            ]
        },
    ),
    (Maxpool2D2, [((1, 64, 150, 150), torch.float32)], {"model_name": ["pt_ssd300_resnet50"]}),
    (
        Maxpool2D1,
        [((1, 64, 224, 224), torch.float32)],
        {
            "model_name": [
                "pt_unet_cityscapes_osmr",
                "pt_vgg13_osmr",
                "pt_bn_vgg19_osmr",
                "pt_bn_vgg19b_osmr",
                "pt_vgg16_osmr",
                "pt_vgg19_osmr",
                "pt_vgg_19_hf",
                "pt_vgg_bn19_torchhub",
                "pt_vgg11_osmr",
                "pt_vgg19_bn_timm",
            ]
        },
    ),
    (
        Maxpool2D1,
        [((1, 128, 112, 112), torch.float32)],
        {
            "model_name": [
                "pt_unet_cityscapes_osmr",
                "pt_vgg13_osmr",
                "pt_bn_vgg19_osmr",
                "pt_bn_vgg19b_osmr",
                "pt_vgg16_osmr",
                "pt_vgg19_osmr",
                "pt_vgg_19_hf",
                "pt_vgg_bn19_torchhub",
                "pt_vgg11_osmr",
                "pt_vgg19_bn_timm",
            ]
        },
    ),
    (
        Maxpool2D1,
        [((1, 256, 56, 56), torch.float32)],
        {
            "model_name": [
                "pt_unet_cityscapes_osmr",
                "pt_vgg13_osmr",
                "pt_bn_vgg19_osmr",
                "pt_bn_vgg19b_osmr",
                "pt_vgg16_osmr",
                "pt_vgg19_osmr",
                "pt_vgg_19_hf",
                "pt_vgg_bn19_torchhub",
                "pt_vgg11_osmr",
                "pt_vgg19_bn_timm",
            ]
        },
    ),
    (
        Maxpool2D3,
        [((1, 256, 56, 56), torch.float32)],
        {
            "model_name": [
                "pt_ese_vovnet19b_dw",
                "pt_ese_vovnet39b",
                "pt_vovnet39",
                "vovnet_57_stigma_pt",
                "pt_ese_vovnet99b",
                "pt_vovnet_39_stigma",
                "pt_vovnet57",
            ]
        },
    ),
    (
        Maxpool2D1,
        [((1, 512, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_unet_cityscapes_osmr",
                "pt_vgg13_osmr",
                "pt_bn_vgg19_osmr",
                "pt_bn_vgg19b_osmr",
                "pt_vgg16_osmr",
                "pt_vgg19_osmr",
                "pt_vgg_19_hf",
                "pt_vgg_bn19_torchhub",
                "pt_vgg11_osmr",
                "pt_vgg19_bn_timm",
            ]
        },
    ),
    (
        Maxpool2D3,
        [((1, 512, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_ese_vovnet19b_dw",
                "pt_ese_vovnet39b",
                "pt_vovnet39",
                "vovnet_57_stigma_pt",
                "pt_ese_vovnet99b",
                "pt_vovnet_39_stigma",
                "pt_vovnet57",
            ]
        },
    ),
    (Maxpool2D1, [((1, 32, 256, 256), torch.float32)], {"model_name": ["pt_unet_torchhub"]}),
    (Maxpool2D1, [((1, 64, 128, 128), torch.float32)], {"model_name": ["pt_unet_torchhub"]}),
    (Maxpool2D1, [((1, 128, 64, 64), torch.float32)], {"model_name": ["pt_unet_torchhub"]}),
    (Maxpool2D1, [((1, 256, 32, 32), torch.float32)], {"model_name": ["pt_unet_torchhub"]}),
    (
        Maxpool2D3,
        [((1, 768, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_ese_vovnet19b_dw",
                "pt_ese_vovnet39b",
                "pt_vovnet39",
                "vovnet_57_stigma_pt",
                "pt_ese_vovnet99b",
                "pt_vovnet_39_stigma",
                "pt_vovnet57",
            ]
        },
    ),
    (Maxpool2D3, [((1, 384, 14, 14), torch.float32)], {"model_name": ["pt_vovnet27s"]}),
    (Maxpool2D2, [((1, 128, 147, 147), torch.float32)], {"model_name": ["pt_xception_timm"]}),
    (Maxpool2D2, [((1, 256, 74, 74), torch.float32)], {"model_name": ["pt_xception_timm"]}),
    (Maxpool2D2, [((1, 728, 37, 37), torch.float32)], {"model_name": ["pt_xception_timm"]}),
    (Maxpool2D2, [((1, 1024, 19, 19), torch.float32)], {"model_name": ["pt_xception_timm"]}),
    (Maxpool2D7, [((1, 640, 20, 20), torch.float32)], {"model_name": ["pt_yolov5x_640x640", "pt_yolox_x"]}),
    (Maxpool2D8, [((1, 640, 20, 20), torch.float32)], {"model_name": ["pt_yolox_x"]}),
    (Maxpool2D9, [((1, 640, 20, 20), torch.float32)], {"model_name": ["pt_yolox_x"]}),
    (Maxpool2D7, [((1, 512, 10, 10), torch.float32)], {"model_name": ["pt_yolov5l_320x320"]}),
    (Maxpool2D7, [((1, 256, 40, 40), torch.float32)], {"model_name": ["pt_yolov5s_1280x1280"]}),
    (Maxpool2D7, [((1, 384, 10, 10), torch.float32)], {"model_name": ["pt_yolov5m_320x320"]}),
    (Maxpool2D7, [((1, 256, 20, 20), torch.float32)], {"model_name": ["pt_yolov5s_640x640", "pt_yolox_s"]}),
    (Maxpool2D8, [((1, 256, 20, 20), torch.float32)], {"model_name": ["pt_yolox_s"]}),
    (Maxpool2D9, [((1, 256, 20, 20), torch.float32)], {"model_name": ["pt_yolox_s"]}),
    (Maxpool2D7, [((1, 640, 10, 10), torch.float32)], {"model_name": ["pt_yolov5x_320x320"]}),
    (Maxpool2D7, [((1, 384, 15, 15), torch.float32)], {"model_name": ["pt_yolov5m_480x480"]}),
    (Maxpool2D7, [((1, 256, 10, 10), torch.float32)], {"model_name": ["pt_yolov5s_320x320"]}),
    (Maxpool2D7, [((1, 128, 15, 15), torch.float32)], {"model_name": ["pt_yolov5n_480x480"]}),
    (Maxpool2D7, [((1, 384, 20, 20), torch.float32)], {"model_name": ["pt_yolov5m_640x640", "pt_yolox_m"]}),
    (Maxpool2D8, [((1, 384, 20, 20), torch.float32)], {"model_name": ["pt_yolox_m"]}),
    (Maxpool2D9, [((1, 384, 20, 20), torch.float32)], {"model_name": ["pt_yolox_m"]}),
    (Maxpool2D7, [((1, 512, 15, 15), torch.float32)], {"model_name": ["pt_yolov5l_480x480"]}),
    (Maxpool2D7, [((1, 640, 15, 15), torch.float32)], {"model_name": ["pt_yolov5x_480x480"]}),
    (
        Maxpool2D7,
        [((1, 512, 20, 20), torch.float32)],
        {"model_name": ["pt_yolov5l_640x640", "pt_yolox_darknet", "pt_yolox_l"]},
    ),
    (Maxpool2D8, [((1, 512, 20, 20), torch.float32)], {"model_name": ["pt_yolox_darknet", "pt_yolox_l"]}),
    (Maxpool2D9, [((1, 512, 20, 20), torch.float32)], {"model_name": ["pt_yolox_darknet", "pt_yolox_l"]}),
    (Maxpool2D7, [((1, 128, 20, 20), torch.float32)], {"model_name": ["pt_yolov5n_640x640"]}),
    (Maxpool2D7, [((1, 256, 15, 15), torch.float32)], {"model_name": ["pt_yolov5s_480x480"]}),
    (Maxpool2D7, [((1, 128, 10, 10), torch.float32)], {"model_name": ["pt_yolov5n_320x320"]}),
    (Maxpool2D7, [((1, 384, 14, 20), torch.float32)], {"model_name": ["pt_yolov6m"]}),
    (Maxpool2D7, [((1, 128, 14, 20), torch.float32)], {"model_name": ["pt_yolov6n"]}),
    (Maxpool2D7, [((1, 512, 14, 20), torch.float32)], {"model_name": ["pt_yolov6l"]}),
    (Maxpool2D7, [((1, 256, 14, 20), torch.float32)], {"model_name": ["pt_yolov6s"]}),
    (Maxpool2D7, [((1, 128, 13, 13), torch.float32)], {"model_name": ["pt_yolox_nano"]}),
    (Maxpool2D8, [((1, 128, 13, 13), torch.float32)], {"model_name": ["pt_yolox_nano"]}),
    (Maxpool2D9, [((1, 128, 13, 13), torch.float32)], {"model_name": ["pt_yolox_nano"]}),
    (Maxpool2D7, [((1, 192, 13, 13), torch.float32)], {"model_name": ["pt_yolox_tiny"]}),
    (Maxpool2D8, [((1, 192, 13, 13), torch.float32)], {"model_name": ["pt_yolox_tiny"]}),
    (Maxpool2D9, [((1, 192, 13, 13), torch.float32)], {"model_name": ["pt_yolox_tiny"]}),
]


@pytest.mark.push
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes, record_property):
    record_property("frontend", "tt-forge-fe")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes

    for metadata_name, metadata_value in metadata.items():
        record_property(metadata_name, metadata_value)

    inputs = [
        Tensor.create_from_shape(operand_shape, operand_dtype) for operand_shape, operand_dtype in operand_shapes_dtypes
    ]

    framework_model = forge_module(forge_module.__name__)
    framework_model.process_framework_parameters()

    compiled_model = compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)
