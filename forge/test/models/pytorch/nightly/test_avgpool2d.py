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
            kernel_size=[14, 14],
            stride=[14, 14],
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
            stride=[7, 7],
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
            kernel_size=[5, 5],
            stride=[5, 5],
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
            kernel_size=[3, 3],
            stride=[3, 3],
            padding=[0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
            channel_last=0,
        )
        return avgpool2d_output_1


def ids_func(param):
    forge_module, shapes_dtypes, _ = param
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (Avgpool2D0, [((1, 256, 6, 6), torch.float32)], {"model_name": ["pt_alexnet_torchhub", "pt_rcnn"]}),
    (
        Avgpool2D1,
        [((1, 128, 56, 56), torch.float32)],
        {"model_name": ["pt_densenet121", "pt_densenet_169", "pt_densenet_201"]},
    ),
    (Avgpool2D2, [((1, 128, 56, 56), torch.float32)], {"model_name": ["pt_regnet_y_040"]}),
    (
        Avgpool2D1,
        [((1, 256, 28, 28), torch.float32)],
        {"model_name": ["pt_densenet121", "pt_densenet_169", "pt_densenet_201"]},
    ),
    (Avgpool2D1, [((1, 512, 14, 14), torch.float32)], {"model_name": ["pt_densenet121"]}),
    (Avgpool2D3, [((1, 512, 14, 14), torch.float32)], {"model_name": ["pt_regnet_y_040"]}),
    (
        Avgpool2D4,
        [((1, 1024, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_densenet121",
                "pt_dla102x",
                "pt_dla169",
                "pt_dla60",
                "pt_dla102x2",
                "pt_dla60x",
                "pt_dla102",
                "pt_googlenet",
                "pt_mobilenet_v1_224",
                "pt_ese_vovnet19b_dw",
                "pt_ese_vovnet39b",
                "vovnet_57_stigma_pt",
                "pt_ese_vovnet99b",
                "pt_vovnet_39_stigma",
            ]
        },
    ),
    (Avgpool2D5, [((1, 1024, 7, 7), torch.float32)], {"model_name": ["pt_vovnet39", "pt_vovnet57"]}),
    (Avgpool2D1, [((1, 192, 56, 56), torch.float32)], {"model_name": ["pt_densenet_161"]}),
    (Avgpool2D2, [((1, 192, 56, 56), torch.float32)], {"model_name": ["pt_efficientnet_b4_torchvision"]}),
    (Avgpool2D1, [((1, 384, 28, 28), torch.float32)], {"model_name": ["pt_densenet_161"]}),
    (Avgpool2D1, [((1, 1056, 14, 14), torch.float32)], {"model_name": ["pt_densenet_161"]}),
    (Avgpool2D4, [((1, 2208, 7, 7), torch.float32)], {"model_name": ["pt_densenet_161"]}),
    (Avgpool2D1, [((1, 640, 14, 14), torch.float32)], {"model_name": ["pt_densenet_169"]}),
    (Avgpool2D4, [((1, 1664, 7, 7), torch.float32)], {"model_name": ["pt_densenet_169"]}),
    (Avgpool2D1, [((1, 896, 14, 14), torch.float32)], {"model_name": ["pt_densenet_201"]}),
    (Avgpool2D4, [((1, 1920, 7, 7), torch.float32)], {"model_name": ["pt_densenet_201"]}),
    (Avgpool2D4, [((1, 256, 7, 7), torch.float32)], {"model_name": ["pt_dla46_c", "pt_dla46x_c", "pt_dla60x_c"]}),
    (Avgpool2D4, [((1, 512, 7, 7), torch.float32)], {"model_name": ["pt_dla34"]}),
    (Avgpool2D0, [((1, 512, 7, 7), torch.float32)], {"model_name": ["pt_vgg_19_hf", "pt_vgg_bn19_torchhub"]}),
    (Avgpool2D5, [((1, 512, 7, 7), torch.float32)], {"model_name": ["pt_vovnet27s"]}),
    (Avgpool2D6, [((1, 1792, 10, 10), torch.float32)], {"model_name": ["pt_efficientnet_b4_timm"]}),
    (Avgpool2D7, [((1, 48, 112, 112), torch.float32)], {"model_name": ["pt_efficientnet_b4_torchvision"]}),
    (Avgpool2D7, [((1, 24, 112, 112), torch.float32)], {"model_name": ["pt_efficientnet_b4_torchvision"]}),
    (
        Avgpool2D2,
        [((1, 144, 56, 56), torch.float32)],
        {"model_name": ["pt_efficientnet_b4_torchvision", "pt_efficientnet_b0_torchvision"]},
    ),
    (
        Avgpool2D8,
        [((1, 192, 28, 28), torch.float32)],
        {"model_name": ["pt_efficientnet_b4_torchvision", "pt_regnet_y_040"]},
    ),
    (Avgpool2D8, [((1, 336, 28, 28), torch.float32)], {"model_name": ["pt_efficientnet_b4_torchvision"]}),
    (Avgpool2D3, [((1, 336, 14, 14), torch.float32)], {"model_name": ["pt_efficientnet_b4_torchvision"]}),
    (
        Avgpool2D3,
        [((1, 672, 14, 14), torch.float32)],
        {"model_name": ["pt_efficientnet_b4_torchvision", "pt_efficientnet_b0_torchvision", "pt_mobilenet_v3_large"]},
    ),
    (Avgpool2D3, [((1, 960, 14, 14), torch.float32)], {"model_name": ["pt_efficientnet_b4_torchvision"]}),
    (
        Avgpool2D4,
        [((1, 960, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_b4_torchvision",
                "pt_ghostnet_100",
                "pt_mobilenet_v3_large",
                "pt_mobilenetv3_large_100",
            ]
        },
    ),
    (Avgpool2D4, [((1, 1632, 7, 7), torch.float32)], {"model_name": ["pt_efficientnet_b4_torchvision"]}),
    (Avgpool2D4, [((1, 2688, 7, 7), torch.float32)], {"model_name": ["pt_efficientnet_b4_torchvision"]}),
    (Avgpool2D4, [((1, 1792, 7, 7), torch.float32)], {"model_name": ["pt_efficientnet_b4_torchvision"]}),
    (Avgpool2D7, [((1, 32, 112, 112), torch.float32)], {"model_name": ["pt_efficientnet_b0_torchvision"]}),
    (Avgpool2D2, [((1, 96, 56, 56), torch.float32)], {"model_name": ["pt_efficientnet_b0_torchvision"]}),
    (Avgpool2D8, [((1, 144, 28, 28), torch.float32)], {"model_name": ["pt_efficientnet_b0_torchvision"]}),
    (Avgpool2D8, [((1, 240, 28, 28), torch.float32)], {"model_name": ["pt_efficientnet_b0_torchvision"]}),
    (
        Avgpool2D3,
        [((1, 240, 14, 14), torch.float32)],
        {"model_name": ["pt_efficientnet_b0_torchvision", "pt_mobilenet_v3_small"]},
    ),
    (
        Avgpool2D3,
        [((1, 480, 14, 14), torch.float32)],
        {"model_name": ["pt_efficientnet_b0_torchvision", "pt_mobilenet_v3_large"]},
    ),
    (
        Avgpool2D4,
        [((1, 672, 7, 7), torch.float32)],
        {"model_name": ["pt_efficientnet_b0_torchvision", "pt_mobilenet_v3_large"]},
    ),
    (Avgpool2D4, [((1, 1152, 7, 7), torch.float32)], {"model_name": ["pt_efficientnet_b0_torchvision"]}),
    (
        Avgpool2D4,
        [((1, 1280, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_b0_torchvision",
                "pt_efficientnet_b0_timm",
                "mobilenetv2_basic",
                "mobilenetv2_timm",
                "mobilenetv2_224",
            ]
        },
    ),
    (
        Avgpool2D4,
        [((1, 2048, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_timm_hrnet_w18",
                "pt_hrnet_timm_hrnet_w30",
                "pt_hrnet_timm_hrnet_w32",
                "pt_hrnet_timm_hrnet_w48",
                "pt_hrnet_timm_hrnet_w40",
                "pt_hrnet_timm_hrnet_w44",
                "pt_hrnet_timm_hrnet_w18_small",
                "pt_hrnet_timm_hrnet_w64",
                "pt_hrnet_timm_hrnet_w18_small_v2",
                "pt_resnet50_timm",
                "pt_resnet50",
                "pt_resnext50_torchhub",
                "pt_resnext101_torchhub",
                "pt_resnext101_fb_wsl",
                "pt_wide_resnet101_2_timm",
                "pt_wide_resnet50_2_hub",
                "pt_wide_resnet50_2_timm",
                "pt_wide_resnet101_2_hub",
            ]
        },
    ),
    (
        Avgpool2D5,
        [((1, 2048, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_osmr_hrnet_w18_small_v2",
                "pt_hrnet_osmr_hrnetv2_w64",
                "pt_hrnet_osmr_hrnetv2_w40",
                "pt_hrnet_osmr_hrnetv2_w18",
                "pt_hrnet_osmr_hrnetv2_w32",
                "pt_hrnet_osmr_hrnetv2_w30",
                "pt_hrnet_osmr_hrnetv2_w44",
                "pt_hrnet_osmr_hrnetv2_w48",
                "pt_hrnet_osmr_hrnet_w18_small_v1",
                "pt_resnext14_osmr",
                "pt_resnext26_osmr",
                "pt_resnext101_osmr",
                "pt_resnext50_osmr",
            ]
        },
    ),
    (Avgpool2D9, [((1, 384, 35, 35), torch.float32)], {"model_name": ["pt_timm_inception_v4", "pt_osmr_inception_v4"]}),
    (
        Avgpool2D9,
        [((1, 1024, 17, 17), torch.float32)],
        {"model_name": ["pt_timm_inception_v4", "pt_osmr_inception_v4"]},
    ),
    (Avgpool2D9, [((1, 1536, 8, 8), torch.float32)], {"model_name": ["pt_timm_inception_v4", "pt_osmr_inception_v4"]}),
    (Avgpool2D10, [((1, 1536, 8, 8), torch.float32)], {"model_name": ["pt_timm_inception_v4"]}),
    (Avgpool2D11, [((1, 1536, 8, 8), torch.float32)], {"model_name": ["pt_osmr_inception_v4"]}),
    (Avgpool2D12, [((1, 768, 6, 6), torch.float32)], {"model_name": ["pt_mobilenet_v1_192"]}),
    (Avgpool2D1, [((1, 1024, 2, 2), torch.float32)], {"model_name": ["pt_mobilenet_v1_basic"]}),
    (Avgpool2D8, [((1, 320, 28, 28), torch.float32)], {"model_name": ["mobilenetv2_deeplabv3"]}),
    (Avgpool2D13, [((1, 1280, 5, 5), torch.float32)], {"model_name": ["mobilenetv2_160"]}),
    (Avgpool2D14, [((1, 1280, 3, 3), torch.float32)], {"model_name": ["mobilenetv2_96"]}),
    (Avgpool2D2, [((1, 16, 56, 56), torch.float32)], {"model_name": ["pt_mobilenet_v3_small"]}),
    (Avgpool2D3, [((1, 96, 14, 14), torch.float32)], {"model_name": ["pt_mobilenet_v3_small"]}),
    (Avgpool2D3, [((1, 120, 14, 14), torch.float32)], {"model_name": ["pt_mobilenet_v3_small"]}),
    (Avgpool2D3, [((1, 144, 14, 14), torch.float32)], {"model_name": ["pt_mobilenet_v3_small"]}),
    (Avgpool2D4, [((1, 288, 7, 7), torch.float32)], {"model_name": ["pt_mobilenet_v3_small"]}),
    (
        Avgpool2D4,
        [((1, 576, 7, 7), torch.float32)],
        {"model_name": ["pt_mobilenet_v3_small", "pt_mobilenetv3_small_100"]},
    ),
    (Avgpool2D8, [((1, 72, 28, 28), torch.float32)], {"model_name": ["pt_mobilenet_v3_large"]}),
    (Avgpool2D8, [((1, 120, 28, 28), torch.float32)], {"model_name": ["pt_mobilenet_v3_large"]}),
    (Avgpool2D4, [((1, 1088, 7, 7), torch.float32)], {"model_name": ["pt_regnet_y_040"]}),
    (Avgpool2D0, [((1, 4096, 1, 1), torch.float32)], {"model_name": ["pt_vgg19_bn_timm"]}),
    (
        Avgpool2D6,
        [((1, 2048, 10, 10), torch.float32)],
        {"model_name": ["pt_xception65_timm", "pt_xception71_timm", "pt_xception41_timm", "pt_xception_timm"]},
    ),
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
