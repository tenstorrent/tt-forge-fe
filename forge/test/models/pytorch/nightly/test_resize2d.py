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


class Resize2D0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[16, 16], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[64, 64], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[128, 128], method="linear", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[56, 56], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[28, 28], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[14, 14], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[28, 28], method="linear", align_corners=True, channel_last=0
        )
        return resize2d_output_1


class Resize2D7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[30, 40], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[60, 80], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[56, 56], method="linear", align_corners=True, channel_last=0
        )
        return resize2d_output_1


class Resize2D10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[112, 112], method="linear", align_corners=True, channel_last=0
        )
        return resize2d_output_1


class Resize2D11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[224, 224], method="linear", align_corners=True, channel_last=0
        )
        return resize2d_output_1


class Resize2D12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[112, 112], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[224, 224], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[40, 40], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[80, 80], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[20, 20], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[160, 160], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D18(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[30, 30], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D19(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[60, 60], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D20(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[26, 26], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D21(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[52, 52], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


def ids_func(param):
    forge_module, shapes_dtypes, _ = param
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (Resize2D0, [((1, 256, 8, 8), torch.float32)], {"model_name": ["pt_fpn"]}),
    (Resize2D1, [((1, 256, 16, 16), torch.float32)], {"model_name": ["pt_fpn"]}),
    (
        Resize2D2,
        [((1, 256, 16, 16), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_segformer_b1_finetuned_ade_512_512"]},
    ),
    (
        Resize2D3,
        [((1, 18, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_timm_hrnet_w18",
                "pt_hrnet_osmr_hrnet_w18_small_v2",
                "pt_hrnet_osmr_hrnetv2_w18",
                "pt_hrnet_timm_hrnet_w18_small_v2",
            ]
        },
    ),
    (
        Resize2D3,
        [((1, 18, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_timm_hrnet_w18",
                "pt_hrnet_osmr_hrnet_w18_small_v2",
                "pt_hrnet_osmr_hrnetv2_w18",
                "pt_hrnet_timm_hrnet_w18_small_v2",
            ]
        },
    ),
    (
        Resize2D4,
        [((1, 36, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_timm_hrnet_w18",
                "pt_hrnet_osmr_hrnet_w18_small_v2",
                "pt_hrnet_osmr_hrnetv2_w18",
                "pt_hrnet_timm_hrnet_w18_small_v2",
            ]
        },
    ),
    (
        Resize2D3,
        [((1, 18, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_timm_hrnet_w18",
                "pt_hrnet_osmr_hrnet_w18_small_v2",
                "pt_hrnet_osmr_hrnetv2_w18",
                "pt_hrnet_timm_hrnet_w18_small_v2",
            ]
        },
    ),
    (
        Resize2D4,
        [((1, 36, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_timm_hrnet_w18",
                "pt_hrnet_osmr_hrnet_w18_small_v2",
                "pt_hrnet_osmr_hrnetv2_w18",
                "pt_hrnet_timm_hrnet_w18_small_v2",
            ]
        },
    ),
    (
        Resize2D5,
        [((1, 72, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_timm_hrnet_w18",
                "pt_hrnet_osmr_hrnet_w18_small_v2",
                "pt_hrnet_osmr_hrnetv2_w18",
                "pt_hrnet_timm_hrnet_w18_small_v2",
            ]
        },
    ),
    (
        Resize2D3,
        [((1, 64, 28, 28), torch.float32)],
        {"model_name": ["pt_hrnet_osmr_hrnetv2_w64", "pt_hrnet_timm_hrnet_w64"]},
    ),
    (
        Resize2D3,
        [((1, 64, 14, 14), torch.float32)],
        {"model_name": ["pt_hrnet_osmr_hrnetv2_w64", "pt_hrnet_timm_hrnet_w64"]},
    ),
    (
        Resize2D4,
        [((1, 64, 14, 14), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w32", "pt_hrnet_osmr_hrnetv2_w32"]},
    ),
    (
        Resize2D4,
        [((1, 128, 14, 14), torch.float32)],
        {"model_name": ["pt_hrnet_osmr_hrnetv2_w64", "pt_hrnet_timm_hrnet_w64"]},
    ),
    (
        Resize2D3,
        [((1, 64, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_osmr_hrnetv2_w64", "pt_hrnet_timm_hrnet_w64"]},
    ),
    (
        Resize2D4,
        [((1, 64, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w32", "pt_hrnet_osmr_hrnetv2_w32"]},
    ),
    (
        Resize2D5,
        [((1, 64, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w18_small", "pt_hrnet_osmr_hrnet_w18_small_v1"]},
    ),
    (
        Resize2D4,
        [((1, 128, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_osmr_hrnetv2_w64", "pt_hrnet_timm_hrnet_w64"]},
    ),
    (
        Resize2D5,
        [((1, 128, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w32", "pt_hrnet_osmr_hrnetv2_w32"]},
    ),
    (
        Resize2D5,
        [((1, 256, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_osmr_hrnetv2_w64", "pt_hrnet_timm_hrnet_w64"]},
    ),
    (
        Resize2D3,
        [((1, 30, 28, 28), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w30", "pt_hrnet_osmr_hrnetv2_w30"]},
    ),
    (
        Resize2D3,
        [((1, 30, 14, 14), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w30", "pt_hrnet_osmr_hrnetv2_w30"]},
    ),
    (
        Resize2D4,
        [((1, 60, 14, 14), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w30", "pt_hrnet_osmr_hrnetv2_w30"]},
    ),
    (
        Resize2D3,
        [((1, 30, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w30", "pt_hrnet_osmr_hrnetv2_w30"]},
    ),
    (
        Resize2D4,
        [((1, 60, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w30", "pt_hrnet_osmr_hrnetv2_w30"]},
    ),
    (
        Resize2D5,
        [((1, 120, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w30", "pt_hrnet_osmr_hrnetv2_w30"]},
    ),
    (
        Resize2D3,
        [((1, 32, 28, 28), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w32", "pt_hrnet_osmr_hrnetv2_w32"]},
    ),
    (
        Resize2D3,
        [((1, 32, 14, 14), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w32", "pt_hrnet_osmr_hrnetv2_w32"]},
    ),
    (
        Resize2D4,
        [((1, 32, 14, 14), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w18_small", "pt_hrnet_osmr_hrnet_w18_small_v1"]},
    ),
    (
        Resize2D3,
        [((1, 32, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w32", "pt_hrnet_osmr_hrnetv2_w32"]},
    ),
    (
        Resize2D4,
        [((1, 32, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w18_small", "pt_hrnet_osmr_hrnet_w18_small_v1"]},
    ),
    (
        Resize2D3,
        [((1, 40, 28, 28), torch.float32)],
        {"model_name": ["pt_hrnet_osmr_hrnetv2_w40", "pt_hrnet_timm_hrnet_w40"]},
    ),
    (
        Resize2D3,
        [((1, 40, 14, 14), torch.float32)],
        {"model_name": ["pt_hrnet_osmr_hrnetv2_w40", "pt_hrnet_timm_hrnet_w40"]},
    ),
    (
        Resize2D4,
        [((1, 80, 14, 14), torch.float32)],
        {"model_name": ["pt_hrnet_osmr_hrnetv2_w40", "pt_hrnet_timm_hrnet_w40"]},
    ),
    (
        Resize2D3,
        [((1, 40, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_osmr_hrnetv2_w40", "pt_hrnet_timm_hrnet_w40"]},
    ),
    (
        Resize2D4,
        [((1, 80, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_osmr_hrnetv2_w40", "pt_hrnet_timm_hrnet_w40"]},
    ),
    (
        Resize2D5,
        [((1, 160, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_osmr_hrnetv2_w40", "pt_hrnet_timm_hrnet_w40"]},
    ),
    (
        Resize2D3,
        [((1, 48, 28, 28), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w48", "pt_hrnet_osmr_hrnetv2_w48"]},
    ),
    (
        Resize2D3,
        [((1, 48, 14, 14), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w48", "pt_hrnet_osmr_hrnetv2_w48"]},
    ),
    (
        Resize2D4,
        [((1, 96, 14, 14), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w48", "pt_hrnet_osmr_hrnetv2_w48"]},
    ),
    (
        Resize2D3,
        [((1, 48, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w48", "pt_hrnet_osmr_hrnetv2_w48"]},
    ),
    (
        Resize2D4,
        [((1, 96, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w48", "pt_hrnet_osmr_hrnetv2_w48"]},
    ),
    (
        Resize2D5,
        [((1, 192, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w48", "pt_hrnet_osmr_hrnetv2_w48"]},
    ),
    (
        Resize2D3,
        [((1, 44, 28, 28), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w44", "pt_hrnet_osmr_hrnetv2_w44"]},
    ),
    (
        Resize2D3,
        [((1, 44, 14, 14), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w44", "pt_hrnet_osmr_hrnetv2_w44"]},
    ),
    (
        Resize2D4,
        [((1, 88, 14, 14), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w44", "pt_hrnet_osmr_hrnetv2_w44"]},
    ),
    (
        Resize2D3,
        [((1, 44, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w44", "pt_hrnet_osmr_hrnetv2_w44"]},
    ),
    (
        Resize2D4,
        [((1, 88, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w44", "pt_hrnet_osmr_hrnetv2_w44"]},
    ),
    (
        Resize2D5,
        [((1, 176, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w44", "pt_hrnet_osmr_hrnetv2_w44"]},
    ),
    (
        Resize2D3,
        [((1, 16, 28, 28), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w18_small", "pt_hrnet_osmr_hrnet_w18_small_v1"]},
    ),
    (
        Resize2D3,
        [((1, 16, 14, 14), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w18_small", "pt_hrnet_osmr_hrnet_w18_small_v1"]},
    ),
    (
        Resize2D3,
        [((1, 16, 7, 7), torch.float32)],
        {"model_name": ["pt_hrnet_timm_hrnet_w18_small", "pt_hrnet_osmr_hrnet_w18_small_v1"]},
    ),
    (Resize2D6, [((1, 256, 1, 1), torch.float32)], {"model_name": ["mobilenetv2_deeplabv3"]}),
    (
        Resize2D7,
        [((1, 256, 15, 20), torch.float32)],
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
    (
        Resize2D8,
        [((1, 256, 30, 40), torch.float32)],
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
    (
        Resize2D2,
        [((1, 256, 32, 32), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_segformer_b1_finetuned_ade_512_512"]},
    ),
    (
        Resize2D2,
        [((1, 256, 64, 64), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_segformer_b1_finetuned_ade_512_512"]},
    ),
    (
        Resize2D2,
        [((1, 256, 128, 128), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_segformer_b1_finetuned_ade_512_512"]},
    ),
    (
        Resize2D2,
        [((1, 768, 16, 16), torch.float32)],
        {
            "model_name": [
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
            ]
        },
    ),
    (
        Resize2D2,
        [((1, 768, 32, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
            ]
        },
    ),
    (
        Resize2D2,
        [((1, 768, 64, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
            ]
        },
    ),
    (
        Resize2D2,
        [((1, 768, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
            ]
        },
    ),
    (Resize2D6, [((1, 512, 14, 14), torch.float32)], {"model_name": ["pt_unet_cityscapes_osmr"]}),
    (Resize2D9, [((1, 256, 28, 28), torch.float32)], {"model_name": ["pt_unet_cityscapes_osmr"]}),
    (Resize2D10, [((1, 128, 56, 56), torch.float32)], {"model_name": ["pt_unet_cityscapes_osmr"]}),
    (Resize2D11, [((1, 64, 112, 112), torch.float32)], {"model_name": ["pt_unet_cityscapes_osmr"]}),
    (Resize2D5, [((1, 2048, 7, 7), torch.float32)], {"model_name": ["pt_unet_qubvel_pt"]}),
    (Resize2D4, [((1, 256, 14, 14), torch.float32)], {"model_name": ["pt_unet_qubvel_pt"]}),
    (Resize2D3, [((1, 128, 28, 28), torch.float32)], {"model_name": ["pt_unet_qubvel_pt"]}),
    (Resize2D12, [((1, 64, 56, 56), torch.float32)], {"model_name": ["pt_unet_qubvel_pt"]}),
    (Resize2D13, [((1, 32, 112, 112), torch.float32)], {"model_name": ["pt_unet_qubvel_pt"]}),
    (Resize2D14, [((1, 640, 20, 20), torch.float32)], {"model_name": ["pt_yolov5x_640x640", "pt_yolox_x"]}),
    (Resize2D15, [((1, 320, 40, 40), torch.float32)], {"model_name": ["pt_yolov5x_640x640", "pt_yolox_x"]}),
    (Resize2D16, [((1, 512, 10, 10), torch.float32)], {"model_name": ["pt_yolov5l_320x320"]}),
    (
        Resize2D14,
        [((1, 256, 20, 20), torch.float32)],
        {"model_name": ["pt_yolov5l_320x320", "pt_yolov5s_640x640", "pt_yolox_s", "pt_yolox_darknet"]},
    ),
    (
        Resize2D15,
        [((1, 256, 40, 40), torch.float32)],
        {"model_name": ["pt_yolov5s_1280x1280", "pt_yolov5l_640x640", "pt_yolox_l"]},
    ),
    (Resize2D17, [((1, 128, 80, 80), torch.float32)], {"model_name": ["pt_yolov5s_1280x1280"]}),
    (Resize2D16, [((1, 384, 10, 10), torch.float32)], {"model_name": ["pt_yolov5m_320x320"]}),
    (Resize2D14, [((1, 192, 20, 20), torch.float32)], {"model_name": ["pt_yolov5m_320x320"]}),
    (
        Resize2D15,
        [((1, 128, 40, 40), torch.float32)],
        {"model_name": ["pt_yolov5s_640x640", "pt_yolox_s", "pt_yolox_darknet"]},
    ),
    (Resize2D16, [((1, 640, 10, 10), torch.float32)], {"model_name": ["pt_yolov5x_320x320"]}),
    (Resize2D14, [((1, 320, 20, 20), torch.float32)], {"model_name": ["pt_yolov5x_320x320"]}),
    (Resize2D18, [((1, 384, 15, 15), torch.float32)], {"model_name": ["pt_yolov5m_480x480"]}),
    (Resize2D19, [((1, 192, 30, 30), torch.float32)], {"model_name": ["pt_yolov5m_480x480"]}),
    (Resize2D16, [((1, 256, 10, 10), torch.float32)], {"model_name": ["pt_yolov5s_320x320"]}),
    (Resize2D14, [((1, 128, 20, 20), torch.float32)], {"model_name": ["pt_yolov5s_320x320", "pt_yolov5n_640x640"]}),
    (Resize2D18, [((1, 128, 15, 15), torch.float32)], {"model_name": ["pt_yolov5n_480x480"]}),
    (Resize2D19, [((1, 64, 30, 30), torch.float32)], {"model_name": ["pt_yolov5n_480x480"]}),
    (Resize2D14, [((1, 384, 20, 20), torch.float32)], {"model_name": ["pt_yolov5m_640x640", "pt_yolox_m"]}),
    (Resize2D15, [((1, 192, 40, 40), torch.float32)], {"model_name": ["pt_yolov5m_640x640", "pt_yolox_m"]}),
    (Resize2D18, [((1, 512, 15, 15), torch.float32)], {"model_name": ["pt_yolov5l_480x480"]}),
    (Resize2D19, [((1, 256, 30, 30), torch.float32)], {"model_name": ["pt_yolov5l_480x480"]}),
    (Resize2D18, [((1, 640, 15, 15), torch.float32)], {"model_name": ["pt_yolov5x_480x480"]}),
    (Resize2D19, [((1, 320, 30, 30), torch.float32)], {"model_name": ["pt_yolov5x_480x480"]}),
    (Resize2D14, [((1, 512, 20, 20), torch.float32)], {"model_name": ["pt_yolov5l_640x640", "pt_yolox_l"]}),
    (Resize2D15, [((1, 64, 40, 40), torch.float32)], {"model_name": ["pt_yolov5n_640x640"]}),
    (Resize2D18, [((1, 256, 15, 15), torch.float32)], {"model_name": ["pt_yolov5s_480x480"]}),
    (Resize2D19, [((1, 128, 30, 30), torch.float32)], {"model_name": ["pt_yolov5s_480x480"]}),
    (Resize2D16, [((1, 128, 10, 10), torch.float32)], {"model_name": ["pt_yolov5n_320x320"]}),
    (Resize2D14, [((1, 64, 20, 20), torch.float32)], {"model_name": ["pt_yolov5n_320x320"]}),
    (Resize2D20, [((1, 128, 13, 13), torch.float32)], {"model_name": ["pt_yolox_nano"]}),
    (Resize2D21, [((1, 64, 26, 26), torch.float32)], {"model_name": ["pt_yolox_nano"]}),
    (Resize2D20, [((1, 192, 13, 13), torch.float32)], {"model_name": ["pt_yolox_tiny"]}),
    (Resize2D21, [((1, 96, 26, 26), torch.float32)], {"model_name": ["pt_yolox_tiny"]}),
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
