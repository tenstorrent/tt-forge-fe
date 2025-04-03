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


class Conv2Dtranspose0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2dtranspose0.weight_1",
            forge.Parameter(*(4, 16, 2, 2), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2dtranspose_input_0):
        conv2dtranspose_output_1 = forge.op.Conv2dTranspose(
            "",
            conv2dtranspose_input_0,
            self.get_parameter("conv2dtranspose0.weight_1"),
            stride=2,
            padding=0,
            dilation=1,
            groups=1,
            channel_last=0,
            output_padding=[0, 0],
        )
        return conv2dtranspose_output_1


class Conv2Dtranspose1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2dtranspose1.weight_1",
            forge.Parameter(*(16, 1, 2, 2), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2dtranspose_input_0):
        conv2dtranspose_output_1 = forge.op.Conv2dTranspose(
            "",
            conv2dtranspose_input_0,
            self.get_parameter("conv2dtranspose1.weight_1"),
            stride=2,
            padding=0,
            dilation=1,
            groups=1,
            channel_last=0,
            output_padding=[0, 0],
        )
        return conv2dtranspose_output_1


class Conv2Dtranspose2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2dtranspose2.weight_1",
            forge.Parameter(*(64, 1, 4, 4), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2dtranspose_input_0):
        conv2dtranspose_output_1 = forge.op.Conv2dTranspose(
            "",
            conv2dtranspose_input_0,
            self.get_parameter("conv2dtranspose2.weight_1"),
            stride=2,
            padding=1,
            dilation=1,
            groups=64,
            channel_last=0,
            output_padding=[0, 0],
        )
        return conv2dtranspose_output_1


class Conv2Dtranspose3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2dtranspose3.weight_1",
            forge.Parameter(*(128, 1, 4, 4), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2dtranspose_input_0):
        conv2dtranspose_output_1 = forge.op.Conv2dTranspose(
            "",
            conv2dtranspose_input_0,
            self.get_parameter("conv2dtranspose3.weight_1"),
            stride=2,
            padding=1,
            dilation=1,
            groups=128,
            channel_last=0,
            output_padding=[0, 0],
        )
        return conv2dtranspose_output_1


class Conv2Dtranspose4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2dtranspose4.weight_1",
            forge.Parameter(*(256, 1, 4, 4), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2dtranspose_input_0):
        conv2dtranspose_output_1 = forge.op.Conv2dTranspose(
            "",
            conv2dtranspose_input_0,
            self.get_parameter("conv2dtranspose4.weight_1"),
            stride=2,
            padding=1,
            dilation=1,
            groups=256,
            channel_last=0,
            output_padding=[0, 0],
        )
        return conv2dtranspose_output_1


class Conv2Dtranspose5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2dtranspose5.weight_1",
            forge.Parameter(*(512, 256, 2, 2), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2dtranspose_input_0):
        conv2dtranspose_output_1 = forge.op.Conv2dTranspose(
            "",
            conv2dtranspose_input_0,
            self.get_parameter("conv2dtranspose5.weight_1"),
            stride=2,
            padding=0,
            dilation=1,
            groups=1,
            channel_last=0,
            output_padding=[0, 0],
        )
        return conv2dtranspose_output_1


class Conv2Dtranspose6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2dtranspose6.weight_1",
            forge.Parameter(*(256, 128, 2, 2), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2dtranspose_input_0):
        conv2dtranspose_output_1 = forge.op.Conv2dTranspose(
            "",
            conv2dtranspose_input_0,
            self.get_parameter("conv2dtranspose6.weight_1"),
            stride=2,
            padding=0,
            dilation=1,
            groups=1,
            channel_last=0,
            output_padding=[0, 0],
        )
        return conv2dtranspose_output_1


class Conv2Dtranspose7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2dtranspose7.weight_1",
            forge.Parameter(*(128, 64, 2, 2), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2dtranspose_input_0):
        conv2dtranspose_output_1 = forge.op.Conv2dTranspose(
            "",
            conv2dtranspose_input_0,
            self.get_parameter("conv2dtranspose7.weight_1"),
            stride=2,
            padding=0,
            dilation=1,
            groups=1,
            channel_last=0,
            output_padding=[0, 0],
        )
        return conv2dtranspose_output_1


class Conv2Dtranspose8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2dtranspose8.weight_1",
            forge.Parameter(*(64, 32, 2, 2), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2dtranspose_input_0):
        conv2dtranspose_output_1 = forge.op.Conv2dTranspose(
            "",
            conv2dtranspose_input_0,
            self.get_parameter("conv2dtranspose8.weight_1"),
            stride=2,
            padding=0,
            dilation=1,
            groups=1,
            channel_last=0,
            output_padding=[0, 0],
        )
        return conv2dtranspose_output_1


class Conv2Dtranspose9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2dtranspose9.weight_1",
            forge.Parameter(*(1024, 512, 2, 2), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2dtranspose_input_0):
        conv2dtranspose_output_1 = forge.op.Conv2dTranspose(
            "",
            conv2dtranspose_input_0,
            self.get_parameter("conv2dtranspose9.weight_1"),
            stride=2,
            padding=0,
            dilation=1,
            groups=1,
            channel_last=0,
            output_padding=[0, 0],
        )
        return conv2dtranspose_output_1


class Conv2Dtranspose10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2dtranspose10_const_1", shape=(64, 64, 2, 2), dtype=torch.float32)
        self.add_constant("conv2dtranspose10_const_2", shape=(64,), dtype=torch.float32)

    def forward(self, conv2dtranspose_input_0):
        conv2dtranspose_output_1 = forge.op.Conv2dTranspose(
            "",
            conv2dtranspose_input_0,
            self.get_constant("conv2dtranspose10_const_1"),
            self.get_constant("conv2dtranspose10_const_2"),
            stride=2,
            padding=0,
            dilation=1,
            groups=1,
            channel_last=0,
            output_padding=[0, 0],
        )
        return conv2dtranspose_output_1


class Conv2Dtranspose11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2dtranspose11_const_1", shape=(32, 32, 2, 2), dtype=torch.float32)
        self.add_constant("conv2dtranspose11_const_2", shape=(32,), dtype=torch.float32)

    def forward(self, conv2dtranspose_input_0):
        conv2dtranspose_output_1 = forge.op.Conv2dTranspose(
            "",
            conv2dtranspose_input_0,
            self.get_constant("conv2dtranspose11_const_1"),
            self.get_constant("conv2dtranspose11_const_2"),
            stride=2,
            padding=0,
            dilation=1,
            groups=1,
            channel_last=0,
            output_padding=[0, 0],
        )
        return conv2dtranspose_output_1


class Conv2Dtranspose12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2dtranspose12_const_1", shape=(192, 192, 2, 2), dtype=torch.float32)
        self.add_constant("conv2dtranspose12_const_2", shape=(192,), dtype=torch.float32)

    def forward(self, conv2dtranspose_input_0):
        conv2dtranspose_output_1 = forge.op.Conv2dTranspose(
            "",
            conv2dtranspose_input_0,
            self.get_constant("conv2dtranspose12_const_1"),
            self.get_constant("conv2dtranspose12_const_2"),
            stride=2,
            padding=0,
            dilation=1,
            groups=1,
            channel_last=0,
            output_padding=[0, 0],
        )
        return conv2dtranspose_output_1


class Conv2Dtranspose13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2dtranspose13_const_1", shape=(96, 96, 2, 2), dtype=torch.float32)
        self.add_constant("conv2dtranspose13_const_2", shape=(96,), dtype=torch.float32)

    def forward(self, conv2dtranspose_input_0):
        conv2dtranspose_output_1 = forge.op.Conv2dTranspose(
            "",
            conv2dtranspose_input_0,
            self.get_constant("conv2dtranspose13_const_1"),
            self.get_constant("conv2dtranspose13_const_2"),
            stride=2,
            padding=0,
            dilation=1,
            groups=1,
            channel_last=0,
            output_padding=[0, 0],
        )
        return conv2dtranspose_output_1


class Conv2Dtranspose14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2dtranspose14_const_1", shape=(256, 256, 2, 2), dtype=torch.float32)
        self.add_constant("conv2dtranspose14_const_2", shape=(256,), dtype=torch.float32)

    def forward(self, conv2dtranspose_input_0):
        conv2dtranspose_output_1 = forge.op.Conv2dTranspose(
            "",
            conv2dtranspose_input_0,
            self.get_constant("conv2dtranspose14_const_1"),
            self.get_constant("conv2dtranspose14_const_2"),
            stride=2,
            padding=0,
            dilation=1,
            groups=1,
            channel_last=0,
            output_padding=[0, 0],
        )
        return conv2dtranspose_output_1


class Conv2Dtranspose15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2dtranspose15_const_1", shape=(128, 128, 2, 2), dtype=torch.float32)
        self.add_constant("conv2dtranspose15_const_2", shape=(128,), dtype=torch.float32)

    def forward(self, conv2dtranspose_input_0):
        conv2dtranspose_output_1 = forge.op.Conv2dTranspose(
            "",
            conv2dtranspose_input_0,
            self.get_constant("conv2dtranspose15_const_1"),
            self.get_constant("conv2dtranspose15_const_2"),
            stride=2,
            padding=0,
            dilation=1,
            groups=1,
            channel_last=0,
            output_padding=[0, 0],
        )
        return conv2dtranspose_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    pytest.param(
        (
            Conv2Dtranspose0,
            [((1, 4, 7, 7), torch.float32)],
            {
                "model_name": ["pt_autoencoder_conv_img_enc_github"],
                "pcc": 0.99,
                "op_params": {
                    "stride": "2",
                    "padding": "0",
                    "dilation": "1",
                    "groups": "1",
                    "channel_last": "0",
                    "output_padding": "[0, 0]",
                },
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Conv2Dtranspose1,
            [((1, 16, 14, 14), torch.float32)],
            {
                "model_name": ["pt_autoencoder_conv_img_enc_github"],
                "pcc": 0.99,
                "op_params": {
                    "stride": "2",
                    "padding": "0",
                    "dilation": "1",
                    "groups": "1",
                    "channel_last": "0",
                    "output_padding": "[0, 0]",
                },
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/impl/program/program.cpp:980: tt::exception info: Statically allocated circular buffers in program 8502 clash with L1 buffers on core range [(x=0,y=0) - (x=0,y=0)]. L1 buffer allocated at 1227648 and static circular buffer region ends at 1342496"
            )
        ],
    ),
    pytest.param(
        (
            Conv2Dtranspose2,
            [((1, 64, 28, 28), torch.float32)],
            {
                "model_name": ["pt_monodle_base_obj_det_torchvision"],
                "pcc": 0.99,
                "op_params": {
                    "stride": "2",
                    "padding": "1",
                    "dilation": "1",
                    "groups": "64",
                    "channel_last": "0",
                    "output_padding": "[0, 0]",
                },
            },
        ),
        marks=[pytest.mark.skip(reason="Floating point exception")],
    ),
    pytest.param(
        (
            Conv2Dtranspose3,
            [((1, 128, 14, 14), torch.float32)],
            {
                "model_name": ["pt_monodle_base_obj_det_torchvision"],
                "pcc": 0.99,
                "op_params": {
                    "stride": "2",
                    "padding": "1",
                    "dilation": "1",
                    "groups": "128",
                    "channel_last": "0",
                    "output_padding": "[0, 0]",
                },
            },
        ),
        marks=[pytest.mark.skip(reason="Floating point exception")],
    ),
    pytest.param(
        (
            Conv2Dtranspose4,
            [((1, 256, 7, 7), torch.float32)],
            {
                "model_name": ["pt_monodle_base_obj_det_torchvision"],
                "pcc": 0.99,
                "op_params": {
                    "stride": "2",
                    "padding": "1",
                    "dilation": "1",
                    "groups": "256",
                    "channel_last": "0",
                    "output_padding": "[0, 0]",
                },
            },
        ),
        marks=[pytest.mark.skip(reason="Floating point exception")],
    ),
    pytest.param(
        (
            Conv2Dtranspose5,
            [((1, 512, 16, 16), torch.float32)],
            {
                "model_name": ["pt_unet_base_img_seg_torchhub"],
                "pcc": 0.99,
                "op_params": {
                    "stride": "2",
                    "padding": "0",
                    "dilation": "1",
                    "groups": "1",
                    "channel_last": "0",
                    "output_padding": "[0, 0]",
                },
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/impl/program/program.cpp:971: tt::exception info: Statically allocated circular buffers on core range [(x=0,y=0) - (x=7,y=1)] grow to 1686560 B which is beyond max L1 size of 1499136 B"
            )
        ],
    ),
    pytest.param(
        (
            Conv2Dtranspose6,
            [((1, 256, 32, 32), torch.float32)],
            {
                "model_name": ["pt_unet_base_img_seg_torchhub"],
                "pcc": 0.99,
                "op_params": {
                    "stride": "2",
                    "padding": "0",
                    "dilation": "1",
                    "groups": "1",
                    "channel_last": "0",
                    "output_padding": "[0, 0]",
                },
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/impl/program/program.cpp:971: tt::exception info: Statically allocated circular buffers on core range [(x=0,y=0) - (x=7,y=0)] grow to 6405152 B which is beyond max L1 size of 1499136 B"
            )
        ],
    ),
    pytest.param(
        (
            Conv2Dtranspose7,
            [((1, 128, 64, 64), torch.float32)],
            {
                "model_name": ["pt_unet_base_img_seg_torchhub"],
                "pcc": 0.99,
                "op_params": {
                    "stride": "2",
                    "padding": "0",
                    "dilation": "1",
                    "groups": "1",
                    "channel_last": "0",
                    "output_padding": "[0, 0]",
                },
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:140: tt::exception info: Out of Memory: Not enough space to allocate 8520192 B L1 buffer across 4 banks, where each bank needs to store 2130048 B"
            )
        ],
    ),
    pytest.param(
        (
            Conv2Dtranspose8,
            [((1, 64, 128, 128), torch.float32)],
            {
                "model_name": ["pt_unet_base_img_seg_torchhub"],
                "pcc": 0.99,
                "op_params": {
                    "stride": "2",
                    "padding": "0",
                    "dilation": "1",
                    "groups": "1",
                    "channel_last": "0",
                    "output_padding": "[0, 0]",
                },
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:140: tt::exception info: Out of Memory: Not enough space to allocate 4194304 B L1 buffer across 2 banks, where each bank needs to store 2097152 B"
            )
        ],
    ),
    pytest.param(
        (
            Conv2Dtranspose9,
            [((1, 1024, 14, 14), torch.float32)],
            {
                "model_name": ["pt_unet_carvana_base_img_seg_github"],
                "pcc": 0.99,
                "op_params": {
                    "stride": "2",
                    "padding": "0",
                    "dilation": "1",
                    "groups": "1",
                    "channel_last": "0",
                    "output_padding": "[0, 0]",
                },
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/impl/program/program.cpp:980: tt::exception info: Statically allocated circular buffers in program 8658 clash with L1 buffers on core range [(x=0,y=0) - (x=7,y=3)]. L1 buffer allocated at 1227648 and static circular buffer region ends at 1342496"
            )
        ],
    ),
    pytest.param(
        (
            Conv2Dtranspose5,
            [((1, 512, 28, 28), torch.float32)],
            {
                "model_name": ["pt_unet_carvana_base_img_seg_github"],
                "pcc": 0.99,
                "op_params": {
                    "stride": "2",
                    "padding": "0",
                    "dilation": "1",
                    "groups": "1",
                    "channel_last": "0",
                    "output_padding": "[0, 0]",
                },
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/impl/program/program.cpp:971: tt::exception info: Statically allocated circular buffers on core range [(x=0,y=0) - (x=7,y=1)] grow to 4930592 B which is beyond max L1 size of 1499136 B"
            )
        ],
    ),
    pytest.param(
        (
            Conv2Dtranspose6,
            [((1, 256, 56, 56), torch.float32)],
            {
                "model_name": ["pt_unet_carvana_base_img_seg_github"],
                "pcc": 0.99,
                "op_params": {
                    "stride": "2",
                    "padding": "0",
                    "dilation": "1",
                    "groups": "1",
                    "channel_last": "0",
                    "output_padding": "[0, 0]",
                },
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:140: tt::exception info: Out of Memory: Not enough space to allocate 13075456 B L1 buffer across 8 banks, where each bank needs to store 1634432 B"
            )
        ],
    ),
    pytest.param(
        (
            Conv2Dtranspose7,
            [((1, 128, 112, 112), torch.float32)],
            {
                "model_name": ["pt_unet_carvana_base_img_seg_github"],
                "pcc": 0.99,
                "op_params": {
                    "stride": "2",
                    "padding": "0",
                    "dilation": "1",
                    "groups": "1",
                    "channel_last": "0",
                    "output_padding": "[0, 0]",
                },
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:140: tt::exception info: Out of Memory: Not enough space to allocate 6422528 B L1 buffer across 4 banks, where each bank needs to store 1605632 B"
            )
        ],
    ),
    pytest.param(
        (
            Conv2Dtranspose10,
            [((1, 64, 14, 20), torch.float32)],
            {
                "model_name": ["pt_yolo_v6_yolov6n_obj_det_torchhub"],
                "pcc": 0.99,
                "op_params": {
                    "stride": "2",
                    "padding": "0",
                    "dilation": "1",
                    "groups": "1",
                    "channel_last": "0",
                    "output_padding": "[0, 0]",
                },
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/impl/program/program.cpp:971: tt::exception info: Statically allocated circular buffers on core range [(x=0,y=0) - (x=1,y=0)] grow to 1838112 B which is beyond max L1 size of 1499136 B"
            )
        ],
    ),
    pytest.param(
        (
            Conv2Dtranspose11,
            [((1, 32, 28, 40), torch.float32)],
            {
                "model_name": ["pt_yolo_v6_yolov6n_obj_det_torchhub"],
                "pcc": 0.99,
                "op_params": {
                    "stride": "2",
                    "padding": "0",
                    "dilation": "1",
                    "groups": "1",
                    "channel_last": "0",
                    "output_padding": "[0, 0]",
                },
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/impl/program/program.cpp:971: tt::exception info: Statically allocated circular buffers on core range [(x=0,y=0) - (x=0,y=0)] grow to 6999072 B which is beyond max L1 size of 1499136 B"
            )
        ],
    ),
    pytest.param(
        (
            Conv2Dtranspose12,
            [((1, 192, 14, 20), torch.float32)],
            {
                "model_name": ["pt_yolo_v6_yolov6m_obj_det_torchhub"],
                "pcc": 0.99,
                "op_params": {
                    "stride": "2",
                    "padding": "0",
                    "dilation": "1",
                    "groups": "1",
                    "channel_last": "0",
                    "output_padding": "[0, 0]",
                },
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/impl/program/program.cpp:971: tt::exception info: Statically allocated circular buffers on core range [(x=0,y=0) - (x=5,y=0)] grow to 1838112 B which is beyond max L1 size of 1499136 B"
            )
        ],
    ),
    pytest.param(
        (
            Conv2Dtranspose13,
            [((1, 96, 28, 40), torch.float32)],
            {
                "model_name": ["pt_yolo_v6_yolov6m_obj_det_torchhub"],
                "pcc": 0.99,
                "op_params": {
                    "stride": "2",
                    "padding": "0",
                    "dilation": "1",
                    "groups": "1",
                    "channel_last": "0",
                    "output_padding": "[0, 0]",
                },
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/impl/program/program.cpp:971: tt::exception info: Statically allocated circular buffers on core range [(x=0,y=0) - (x=2,y=0)] grow to 6999072 B which is beyond max L1 size of 1499136 B"
            )
        ],
    ),
    pytest.param(
        (
            Conv2Dtranspose14,
            [((1, 256, 14, 20), torch.float32)],
            {
                "model_name": ["pt_yolo_v6_yolov6l_obj_det_torchhub"],
                "pcc": 0.99,
                "op_params": {
                    "stride": "2",
                    "padding": "0",
                    "dilation": "1",
                    "groups": "1",
                    "channel_last": "0",
                    "output_padding": "[0, 0]",
                },
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/impl/program/program.cpp:971: tt::exception info: Statically allocated circular buffers on core range [(x=0,y=0) - (x=7,y=0)] grow to 1838112 B which is beyond max L1 size of 1499136 B"
            )
        ],
    ),
    pytest.param(
        (
            Conv2Dtranspose15,
            [((1, 128, 28, 40), torch.float32)],
            {
                "model_name": ["pt_yolo_v6_yolov6l_obj_det_torchhub"],
                "pcc": 0.99,
                "op_params": {
                    "stride": "2",
                    "padding": "0",
                    "dilation": "1",
                    "groups": "1",
                    "channel_last": "0",
                    "output_padding": "[0, 0]",
                },
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/impl/program/program.cpp:971: tt::exception info: Statically allocated circular buffers on core range [(x=0,y=0) - (x=3,y=0)] grow to 6999072 B which is beyond max L1 size of 1499136 B"
            )
        ],
    ),
    pytest.param(
        (
            Conv2Dtranspose15,
            [((1, 128, 14, 20), torch.float32)],
            {
                "model_name": ["pt_yolo_v6_yolov6s_obj_det_torchhub"],
                "pcc": 0.99,
                "op_params": {
                    "stride": "2",
                    "padding": "0",
                    "dilation": "1",
                    "groups": "1",
                    "channel_last": "0",
                    "output_padding": "[0, 0]",
                },
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/impl/program/program.cpp:971: tt::exception info: Statically allocated circular buffers on core range [(x=0,y=0) - (x=3,y=0)] grow to 1838112 B which is beyond max L1 size of 1499136 B"
            )
        ],
    ),
    pytest.param(
        (
            Conv2Dtranspose10,
            [((1, 64, 28, 40), torch.float32)],
            {
                "model_name": ["pt_yolo_v6_yolov6s_obj_det_torchhub"],
                "pcc": 0.99,
                "op_params": {
                    "stride": "2",
                    "padding": "0",
                    "dilation": "1",
                    "groups": "1",
                    "channel_last": "0",
                    "output_padding": "[0, 0]",
                },
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/impl/program/program.cpp:971: tt::exception info: Statically allocated circular buffers on core range [(x=0,y=0) - (x=1,y=0)] grow to 6999072 B which is beyond max L1 size of 1499136 B"
            )
        ],
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes, forge_property_recorder):
    forge_property_recorder("tags.op_name", "Conv2dTranspose")

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
