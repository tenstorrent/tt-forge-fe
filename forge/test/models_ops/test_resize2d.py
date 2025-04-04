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


class Resize2D0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[27, 40], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[54, 80], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[107, 160], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[27, 27], method="linear", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[16, 16], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[64, 64], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[128, 128], method="linear", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[56, 56], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[28, 28], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[14, 14], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[7, 7], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[30, 40], method="linear", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[60, 80], method="linear", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[120, 160], method="linear", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[240, 320], method="linear", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[480, 640], method="linear", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[28, 28], method="linear", align_corners=True, channel_last=0
        )
        return resize2d_output_1


class Resize2D17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[12, 40], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D18(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[24, 80], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D19(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[48, 160], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D20(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[96, 320], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D21(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[192, 640], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D22(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[20, 64], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D23(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[40, 128], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D24(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[80, 256], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D25(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[160, 512], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D26(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[320, 1024], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D27(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[30, 40], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D28(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[60, 80], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D29(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[56, 56], method="linear", align_corners=True, channel_last=0
        )
        return resize2d_output_1


class Resize2D30(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[112, 112], method="linear", align_corners=True, channel_last=0
        )
        return resize2d_output_1


class Resize2D31(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[224, 224], method="linear", align_corners=True, channel_last=0
        )
        return resize2d_output_1


class Resize2D32(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[112, 112], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D33(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[224, 224], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D34(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[30, 30], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D35(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[60, 60], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D36(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[20, 20], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D37(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[40, 40], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D38(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[80, 80], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D39(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[160, 160], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D40(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[26, 26], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D41(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[52, 52], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    pytest.param(
        (
            Resize2D0,
            [((100, 128, 14, 20), torch.float32)],
            {
                "model_name": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
                "pcc": 0.99,
                "op_params": {
                    "sizes": "[27, 40]",
                    "method": '"nearest_neighbor"',
                    "align_corners": "False",
                    "channel_last": "0",
                },
            },
        ),
        marks=[pytest.mark.xfail(reason="AssertionError: Only support upsample with integer scale factor")],
    ),
    (
        Resize2D1,
        [((100, 64, 27, 40), torch.float32)],
        {
            "model_name": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[54, 80]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    pytest.param(
        (
            Resize2D2,
            [((100, 32, 54, 80), torch.float32)],
            {
                "model_name": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
                "pcc": 0.99,
                "op_params": {
                    "sizes": "[107, 160]",
                    "method": '"nearest_neighbor"',
                    "align_corners": "False",
                    "channel_last": "0",
                },
            },
        ),
        marks=[pytest.mark.xfail(reason="AssertionError: Only support upsample with integer scale factor")],
    ),
    pytest.param(
        (
            Resize2D3,
            [((1, 12, 27, 27), torch.float32)],
            {
                "model_name": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
                "pcc": 0.99,
                "op_params": {"sizes": "[27, 27]", "method": '"linear"', "align_corners": "False", "channel_last": "0"},
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/pool/upsample/device/upsample_op.cpp:118: tt::exception info: Unsupported mode"
            )
        ],
    ),
    pytest.param(
        (
            Resize2D3,
            [((1, 16, 27, 27), torch.float32)],
            {
                "model_name": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
                "pcc": 0.99,
                "op_params": {"sizes": "[27, 27]", "method": '"linear"', "align_corners": "False", "channel_last": "0"},
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/pool/upsample/device/upsample_op.cpp:118: tt::exception info: Unsupported mode"
            )
        ],
    ),
    (
        Resize2D4,
        [((1, 256, 8, 8), torch.float32)],
        {
            "model_name": ["pt_fpn_base_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[16, 16]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D5,
        [((1, 256, 16, 16), torch.float32)],
        {
            "model_name": ["pt_fpn_base_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[64, 64]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    pytest.param(
        (
            Resize2D6,
            [((1, 256, 16, 16), torch.float32)],
            {
                "model_name": [
                    "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                    "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                ],
                "pcc": 0.99,
                "op_params": {
                    "sizes": "[128, 128]",
                    "method": '"linear"',
                    "align_corners": "False",
                    "channel_last": "0",
                },
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/pool/upsample/device/upsample_op.cpp:118: tt::exception info: Unsupported mode"
            )
        ],
    ),
    (
        Resize2D7,
        [((1, 72, 28, 28), torch.float32)],
        {
            "model_name": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[56, 56]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D8,
        [((1, 120, 14, 14), torch.float32)],
        {
            "model_name": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[28, 28]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D8,
        [((1, 240, 14, 14), torch.float32)],
        {
            "model_name": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[28, 28]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D9,
        [((1, 200, 7, 7), torch.float32)],
        {
            "model_name": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[14, 14]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D9,
        [((1, 184, 7, 7), torch.float32)],
        {
            "model_name": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[14, 14]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D9,
        [((1, 480, 7, 7), torch.float32)],
        {
            "model_name": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[14, 14]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D9,
        [((1, 672, 7, 7), torch.float32)],
        {
            "model_name": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[14, 14]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    pytest.param(
        (
            Resize2D10,
            [((1, 960, 3, 3), torch.float32)],
            {
                "model_name": ["pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm"],
                "pcc": 0.99,
                "op_params": {
                    "sizes": "[7, 7]",
                    "method": '"nearest_neighbor"',
                    "align_corners": "False",
                    "channel_last": "0",
                },
            },
        ),
        marks=[pytest.mark.xfail(reason="AssertionError: Only support upsample with integer scale factor")],
    ),
    pytest.param(
        (
            Resize2D11,
            [((1, 64, 15, 20), torch.float32)],
            {
                "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
                "pcc": 0.99,
                "op_params": {"sizes": "[30, 40]", "method": '"linear"', "align_corners": "False", "channel_last": "0"},
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/pool/upsample/device/upsample_op.cpp:118: tt::exception info: Unsupported mode"
            )
        ],
    ),
    pytest.param(
        (
            Resize2D12,
            [((1, 64, 30, 40), torch.float32)],
            {
                "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
                "pcc": 0.99,
                "op_params": {"sizes": "[60, 80]", "method": '"linear"', "align_corners": "False", "channel_last": "0"},
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/pool/upsample/device/upsample_op.cpp:118: tt::exception info: Unsupported mode"
            )
        ],
    ),
    pytest.param(
        (
            Resize2D13,
            [((1, 64, 60, 80), torch.float32)],
            {
                "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
                "pcc": 0.99,
                "op_params": {
                    "sizes": "[120, 160]",
                    "method": '"linear"',
                    "align_corners": "False",
                    "channel_last": "0",
                },
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/pool/upsample/device/upsample_op.cpp:118: tt::exception info: Unsupported mode"
            )
        ],
    ),
    pytest.param(
        (
            Resize2D14,
            [((1, 64, 120, 160), torch.float32)],
            {
                "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
                "pcc": 0.99,
                "op_params": {
                    "sizes": "[240, 320]",
                    "method": '"linear"',
                    "align_corners": "False",
                    "channel_last": "0",
                },
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/pool/upsample/device/upsample_op.cpp:118: tt::exception info: Unsupported mode"
            )
        ],
    ),
    pytest.param(
        (
            Resize2D15,
            [((1, 64, 240, 320), torch.float32)],
            {
                "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
                "pcc": 0.99,
                "op_params": {
                    "sizes": "[480, 640]",
                    "method": '"linear"',
                    "align_corners": "False",
                    "channel_last": "0",
                },
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/pool/upsample/device/upsample_op.cpp:118: tt::exception info: Unsupported mode"
            )
        ],
    ),
    (
        Resize2D7,
        [((1, 64, 28, 28), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnetv2_w64_pose_estimation_osmr", "pt_hrnet_hrnet_w64_pose_estimation_timm"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[56, 56]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D7,
        [((1, 64, 14, 14), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnetv2_w64_pose_estimation_osmr", "pt_hrnet_hrnet_w64_pose_estimation_timm"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[56, 56]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D8,
        [((1, 64, 14, 14), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnetv2_w32_pose_estimation_osmr", "pt_hrnet_hrnet_w32_pose_estimation_timm"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[28, 28]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D8,
        [((1, 128, 14, 14), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnetv2_w64_pose_estimation_osmr", "pt_hrnet_hrnet_w64_pose_estimation_timm"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[28, 28]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D7,
        [((1, 64, 7, 7), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnetv2_w64_pose_estimation_osmr", "pt_hrnet_hrnet_w64_pose_estimation_timm"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[56, 56]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D9,
        [((1, 64, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
            ],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[14, 14]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D8,
        [((1, 64, 7, 7), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnetv2_w32_pose_estimation_osmr", "pt_hrnet_hrnet_w32_pose_estimation_timm"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[28, 28]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D8,
        [((1, 128, 7, 7), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnetv2_w64_pose_estimation_osmr", "pt_hrnet_hrnet_w64_pose_estimation_timm"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[28, 28]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D9,
        [((1, 128, 7, 7), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnetv2_w32_pose_estimation_osmr", "pt_hrnet_hrnet_w32_pose_estimation_timm"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[14, 14]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D9,
        [((1, 256, 7, 7), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnetv2_w64_pose_estimation_osmr", "pt_hrnet_hrnet_w64_pose_estimation_timm"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[14, 14]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D7,
        [((1, 18, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm",
            ],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[56, 56]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D7,
        [((1, 18, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm",
            ],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[56, 56]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D8,
        [((1, 36, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm",
            ],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[28, 28]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D7,
        [((1, 18, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm",
            ],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[56, 56]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D8,
        [((1, 36, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm",
            ],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[28, 28]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D9,
        [((1, 72, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm",
            ],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[14, 14]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D7,
        [((1, 48, 28, 28), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnetv2_w48_pose_estimation_osmr", "pt_hrnet_hrnet_w48_pose_estimation_timm"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[56, 56]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D7,
        [((1, 48, 14, 14), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnetv2_w48_pose_estimation_osmr", "pt_hrnet_hrnet_w48_pose_estimation_timm"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[56, 56]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D8,
        [((1, 96, 14, 14), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnetv2_w48_pose_estimation_osmr", "pt_hrnet_hrnet_w48_pose_estimation_timm"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[28, 28]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D7,
        [((1, 48, 7, 7), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnetv2_w48_pose_estimation_osmr", "pt_hrnet_hrnet_w48_pose_estimation_timm"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[56, 56]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D8,
        [((1, 96, 7, 7), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnetv2_w48_pose_estimation_osmr", "pt_hrnet_hrnet_w48_pose_estimation_timm"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[28, 28]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D9,
        [((1, 192, 7, 7), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnetv2_w48_pose_estimation_osmr", "pt_hrnet_hrnet_w48_pose_estimation_timm"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[14, 14]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D7,
        [((1, 16, 28, 28), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
            ],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[56, 56]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D7,
        [((1, 16, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
            ],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[56, 56]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D8,
        [((1, 32, 14, 14), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
            ],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[28, 28]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D7,
        [((1, 32, 14, 14), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnetv2_w32_pose_estimation_osmr", "pt_hrnet_hrnet_w32_pose_estimation_timm"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[56, 56]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D8,
        [((1, 32, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
            ],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[28, 28]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D7,
        [((1, 32, 7, 7), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnetv2_w32_pose_estimation_osmr", "pt_hrnet_hrnet_w32_pose_estimation_timm"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[56, 56]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D7,
        [((1, 16, 7, 7), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
            ],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[56, 56]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D7,
        [((1, 32, 28, 28), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnetv2_w32_pose_estimation_osmr", "pt_hrnet_hrnet_w32_pose_estimation_timm"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[56, 56]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D7,
        [((1, 40, 28, 28), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnet_w40_pose_estimation_timm", "pt_hrnet_hrnetv2_w40_pose_estimation_osmr"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[56, 56]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D7,
        [((1, 40, 14, 14), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnet_w40_pose_estimation_timm", "pt_hrnet_hrnetv2_w40_pose_estimation_osmr"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[56, 56]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D8,
        [((1, 80, 14, 14), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnet_w40_pose_estimation_timm", "pt_hrnet_hrnetv2_w40_pose_estimation_osmr"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[28, 28]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D7,
        [((1, 40, 7, 7), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnet_w40_pose_estimation_timm", "pt_hrnet_hrnetv2_w40_pose_estimation_osmr"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[56, 56]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D8,
        [((1, 80, 7, 7), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnet_w40_pose_estimation_timm", "pt_hrnet_hrnetv2_w40_pose_estimation_osmr"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[28, 28]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D9,
        [((1, 160, 7, 7), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnet_w40_pose_estimation_timm", "pt_hrnet_hrnetv2_w40_pose_estimation_osmr"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[14, 14]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D7,
        [((1, 44, 28, 28), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnetv2_w44_pose_estimation_osmr", "pt_hrnet_hrnet_w44_pose_estimation_timm"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[56, 56]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D7,
        [((1, 44, 14, 14), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnetv2_w44_pose_estimation_osmr", "pt_hrnet_hrnet_w44_pose_estimation_timm"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[56, 56]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D8,
        [((1, 88, 14, 14), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnetv2_w44_pose_estimation_osmr", "pt_hrnet_hrnet_w44_pose_estimation_timm"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[28, 28]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D7,
        [((1, 44, 7, 7), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnetv2_w44_pose_estimation_osmr", "pt_hrnet_hrnet_w44_pose_estimation_timm"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[56, 56]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D8,
        [((1, 88, 7, 7), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnetv2_w44_pose_estimation_osmr", "pt_hrnet_hrnet_w44_pose_estimation_timm"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[28, 28]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D9,
        [((1, 176, 7, 7), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnetv2_w44_pose_estimation_osmr", "pt_hrnet_hrnet_w44_pose_estimation_timm"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[14, 14]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D7,
        [((1, 30, 28, 28), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnet_w30_pose_estimation_timm", "pt_hrnet_hrnetv2_w30_pose_estimation_osmr"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[56, 56]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D7,
        [((1, 30, 14, 14), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnet_w30_pose_estimation_timm", "pt_hrnet_hrnetv2_w30_pose_estimation_osmr"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[56, 56]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D8,
        [((1, 60, 14, 14), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnet_w30_pose_estimation_timm", "pt_hrnet_hrnetv2_w30_pose_estimation_osmr"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[28, 28]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D7,
        [((1, 30, 7, 7), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnet_w30_pose_estimation_timm", "pt_hrnet_hrnetv2_w30_pose_estimation_osmr"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[56, 56]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D8,
        [((1, 60, 7, 7), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnet_w30_pose_estimation_timm", "pt_hrnet_hrnetv2_w30_pose_estimation_osmr"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[28, 28]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D9,
        [((1, 120, 7, 7), torch.float32)],
        {
            "model_name": ["pt_hrnet_hrnet_w30_pose_estimation_timm", "pt_hrnet_hrnetv2_w30_pose_estimation_osmr"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[14, 14]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    pytest.param(
        (
            Resize2D16,
            [((1, 256, 1, 1), torch.float32)],
            {
                "model_name": ["pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf"],
                "pcc": 0.99,
                "op_params": {"sizes": "[28, 28]", "method": '"linear"', "align_corners": "True", "channel_last": "0"},
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/pool/upsample/device/upsample_op.cpp:118: tt::exception info: Unsupported mode"
            )
        ],
    ),
    (
        Resize2D17,
        [((1, 256, 6, 20), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[12, 40]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D18,
        [((1, 128, 12, 40), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[24, 80]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D19,
        [((1, 64, 24, 80), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[48, 160]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D20,
        [((1, 32, 48, 160), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[96, 320]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D21,
        [((1, 16, 96, 320), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[192, 640]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D22,
        [((1, 256, 10, 32), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[20, 64]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D23,
        [((1, 128, 20, 64), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[40, 128]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D24,
        [((1, 64, 40, 128), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[80, 256]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D25,
        [((1, 32, 80, 256), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[160, 512]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D26,
        [((1, 16, 160, 512), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[320, 1024]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D27,
        [((1, 256, 15, 20), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
            ],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[30, 40]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D28,
        [((1, 256, 30, 40), torch.float32)],
        {
            "model_name": [
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
            ],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[60, 80]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    pytest.param(
        (
            Resize2D6,
            [((1, 768, 16, 16), torch.float32)],
            {
                "model_name": [
                    "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                    "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                    "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                ],
                "pcc": 0.99,
                "op_params": {
                    "sizes": "[128, 128]",
                    "method": '"linear"',
                    "align_corners": "False",
                    "channel_last": "0",
                },
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/pool/upsample/device/upsample_op.cpp:118: tt::exception info: Unsupported mode"
            )
        ],
    ),
    pytest.param(
        (
            Resize2D6,
            [((1, 768, 32, 32), torch.float32)],
            {
                "model_name": [
                    "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                    "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                    "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                ],
                "pcc": 0.99,
                "op_params": {
                    "sizes": "[128, 128]",
                    "method": '"linear"',
                    "align_corners": "False",
                    "channel_last": "0",
                },
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/pool/upsample/device/upsample_op.cpp:118: tt::exception info: Unsupported mode"
            )
        ],
    ),
    pytest.param(
        (
            Resize2D6,
            [((1, 768, 64, 64), torch.float32)],
            {
                "model_name": [
                    "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                    "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                    "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                ],
                "pcc": 0.99,
                "op_params": {
                    "sizes": "[128, 128]",
                    "method": '"linear"',
                    "align_corners": "False",
                    "channel_last": "0",
                },
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/pool/upsample/device/upsample_op.cpp:118: tt::exception info: Unsupported mode"
            )
        ],
    ),
    pytest.param(
        (
            Resize2D6,
            [((1, 768, 128, 128), torch.float32)],
            {
                "model_name": [
                    "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                    "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                    "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                ],
                "pcc": 0.99,
                "op_params": {
                    "sizes": "[128, 128]",
                    "method": '"linear"',
                    "align_corners": "False",
                    "channel_last": "0",
                },
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/pool/upsample/device/upsample_op.cpp:118: tt::exception info: Unsupported mode"
            )
        ],
    ),
    pytest.param(
        (
            Resize2D6,
            [((1, 256, 32, 32), torch.float32)],
            {
                "model_name": [
                    "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                    "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                ],
                "pcc": 0.99,
                "op_params": {
                    "sizes": "[128, 128]",
                    "method": '"linear"',
                    "align_corners": "False",
                    "channel_last": "0",
                },
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/pool/upsample/device/upsample_op.cpp:118: tt::exception info: Unsupported mode"
            )
        ],
    ),
    pytest.param(
        (
            Resize2D6,
            [((1, 256, 64, 64), torch.float32)],
            {
                "model_name": [
                    "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                    "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                ],
                "pcc": 0.99,
                "op_params": {
                    "sizes": "[128, 128]",
                    "method": '"linear"',
                    "align_corners": "False",
                    "channel_last": "0",
                },
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/pool/upsample/device/upsample_op.cpp:118: tt::exception info: Unsupported mode"
            )
        ],
    ),
    pytest.param(
        (
            Resize2D6,
            [((1, 256, 128, 128), torch.float32)],
            {
                "model_name": [
                    "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                    "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                ],
                "pcc": 0.99,
                "op_params": {
                    "sizes": "[128, 128]",
                    "method": '"linear"',
                    "align_corners": "False",
                    "channel_last": "0",
                },
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/pool/upsample/device/upsample_op.cpp:118: tt::exception info: Unsupported mode"
            )
        ],
    ),
    pytest.param(
        (
            Resize2D16,
            [((1, 512, 14, 14), torch.float32)],
            {
                "model_name": ["pt_unet_cityscape_img_seg_osmr"],
                "pcc": 0.99,
                "op_params": {"sizes": "[28, 28]", "method": '"linear"', "align_corners": "True", "channel_last": "0"},
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/pool/upsample/device/upsample_op.cpp:118: tt::exception info: Unsupported mode"
            )
        ],
    ),
    pytest.param(
        (
            Resize2D29,
            [((1, 256, 28, 28), torch.float32)],
            {
                "model_name": ["pt_unet_cityscape_img_seg_osmr"],
                "pcc": 0.99,
                "op_params": {"sizes": "[56, 56]", "method": '"linear"', "align_corners": "True", "channel_last": "0"},
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/pool/upsample/device/upsample_op.cpp:118: tt::exception info: Unsupported mode"
            )
        ],
    ),
    pytest.param(
        (
            Resize2D30,
            [((1, 128, 56, 56), torch.float32)],
            {
                "model_name": ["pt_unet_cityscape_img_seg_osmr"],
                "pcc": 0.99,
                "op_params": {
                    "sizes": "[112, 112]",
                    "method": '"linear"',
                    "align_corners": "True",
                    "channel_last": "0",
                },
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/pool/upsample/device/upsample_op.cpp:118: tt::exception info: Unsupported mode"
            )
        ],
    ),
    pytest.param(
        (
            Resize2D31,
            [((1, 64, 112, 112), torch.float32)],
            {
                "model_name": ["pt_unet_cityscape_img_seg_osmr"],
                "pcc": 0.99,
                "op_params": {
                    "sizes": "[224, 224]",
                    "method": '"linear"',
                    "align_corners": "True",
                    "channel_last": "0",
                },
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/pool/upsample/device/upsample_op.cpp:118: tt::exception info: Unsupported mode"
            )
        ],
    ),
    (
        Resize2D9,
        [((1, 2048, 7, 7), torch.float32)],
        {
            "model_name": ["pt_unet_qubvel_img_seg_torchhub"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[14, 14]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D8,
        [((1, 256, 14, 14), torch.float32)],
        {
            "model_name": ["pt_unet_qubvel_img_seg_torchhub"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[28, 28]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D7,
        [((1, 128, 28, 28), torch.float32)],
        {
            "model_name": ["pt_unet_qubvel_img_seg_torchhub"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[56, 56]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D32,
        [((1, 64, 56, 56), torch.float32)],
        {
            "model_name": ["pt_unet_qubvel_img_seg_torchhub"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[112, 112]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D33,
        [((1, 32, 112, 112), torch.float32)],
        {
            "model_name": ["pt_unet_qubvel_img_seg_torchhub"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[224, 224]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D34,
        [((1, 256, 15, 15), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5s_imgcls_torchhub_480x480"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[30, 30]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D35,
        [((1, 128, 30, 30), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5s_imgcls_torchhub_480x480"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[60, 60]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D36,
        [((1, 640, 10, 10), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_320x320"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[20, 20]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D37,
        [((1, 320, 20, 20), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_320x320"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[40, 40]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D36,
        [((1, 256, 10, 10), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5s_imgcls_torchhub_320x320"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[20, 20]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D37,
        [((1, 128, 20, 20), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5s_imgcls_torchhub_320x320", "pt_yolo_v5_yolov5n_imgcls_torchhub_640x640"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[40, 40]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D34,
        [((1, 512, 15, 15), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5l_imgcls_torchhub_480x480"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[30, 30]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D35,
        [((1, 256, 30, 30), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5l_imgcls_torchhub_480x480"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[60, 60]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D37,
        [((1, 384, 20, 20), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_640x640", "pt_yolox_yolox_m_obj_det_torchhub"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[40, 40]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D38,
        [((1, 192, 40, 40), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_640x640", "pt_yolox_yolox_m_obj_det_torchhub"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[80, 80]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D37,
        [((1, 512, 20, 20), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5l_imgcls_torchhub_640x640", "pt_yolox_yolox_l_obj_det_torchhub"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[40, 40]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D38,
        [((1, 256, 40, 40), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5l_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280",
                "pt_yolox_yolox_l_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[80, 80]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D37,
        [((1, 640, 20, 20), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_640x640", "pt_yolox_yolox_x_obj_det_torchhub"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[40, 40]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D38,
        [((1, 320, 40, 40), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_640x640", "pt_yolox_yolox_x_obj_det_torchhub"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[80, 80]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D37,
        [((1, 256, 20, 20), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5s_imgcls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_imgcls_torchhub_320x320",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[40, 40]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D38,
        [((1, 128, 40, 40), torch.float32)],
        {
            "model_name": [
                "pt_yolo_v5_yolov5s_imgcls_torchhub_640x640",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[80, 80]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D38,
        [((1, 64, 40, 40), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_640x640"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[80, 80]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D34,
        [((1, 640, 15, 15), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_480x480"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[30, 30]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D35,
        [((1, 320, 30, 30), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5x_imgcls_torchhub_480x480"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[60, 60]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D36,
        [((1, 384, 10, 10), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_320x320"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[20, 20]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D37,
        [((1, 192, 20, 20), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_320x320"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[40, 40]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D36,
        [((1, 512, 10, 10), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5l_imgcls_torchhub_320x320"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[20, 20]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D39,
        [((1, 128, 80, 80), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[160, 160]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D34,
        [((1, 384, 15, 15), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_480x480"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[30, 30]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D35,
        [((1, 192, 30, 30), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5m_imgcls_torchhub_480x480"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[60, 60]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D36,
        [((1, 128, 10, 10), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_320x320"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[20, 20]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D37,
        [((1, 64, 20, 20), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_320x320"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[40, 40]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D34,
        [((1, 128, 15, 15), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_480x480"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[30, 30]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D35,
        [((1, 64, 30, 30), torch.float32)],
        {
            "model_name": ["pt_yolo_v5_yolov5n_imgcls_torchhub_480x480"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[60, 60]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D40,
        [((1, 192, 13, 13), torch.float32)],
        {
            "model_name": ["pt_yolox_yolox_tiny_obj_det_torchhub"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[26, 26]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D41,
        [((1, 96, 26, 26), torch.float32)],
        {
            "model_name": ["pt_yolox_yolox_tiny_obj_det_torchhub"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[52, 52]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D40,
        [((1, 128, 13, 13), torch.float32)],
        {
            "model_name": ["pt_yolox_yolox_nano_obj_det_torchhub"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[26, 26]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Resize2D41,
        [((1, 64, 26, 26), torch.float32)],
        {
            "model_name": ["pt_yolox_yolox_nano_obj_det_torchhub"],
            "pcc": 0.99,
            "op_params": {
                "sizes": "[52, 52]",
                "method": '"nearest_neighbor"',
                "align_corners": "False",
                "channel_last": "0",
            },
        },
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes, forge_property_recorder):

    forge_property_recorder.enable_single_op_details_recording()

    forge_property_recorder.record_forge_op_name("Resize2d")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes

    pcc = metadata.pop("pcc")

    for metadata_name, metadata_value in metadata.items():
        if metadata_name == "model_name":
            forge_property_recorder.record_op_model_names(metadata_value)
        elif metadata_name == "op_params":
            forge_property_recorder.record_forge_op_args(metadata_value)
        else:
            print("there is no utility function available to record these details")

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
