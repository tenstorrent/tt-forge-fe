# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from loguru import logger

import forge
import forge.op
from forge import ForgeModule, Tensor, compile
from forge.forge_property_utils import (
    record_forge_op_args,
    record_forge_op_name,
    record_op_model_names,
    record_single_op_operands_info,
)
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify


class Conv2D0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d0.weight_1",
            forge.Parameter(*(16, 3, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d0.weight_1"),
            stride=[2, 2],
            padding=[0, 0, 1, 1],
            dilation=1,
            groups=1,
            channel_last=False,
        )
        return conv2d_output_1


class Conv2D1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d1.weight_1",
            forge.Parameter(*(16, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d1.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=16,
            channel_last=False,
        )
        return conv2d_output_1


class Conv2D2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d2.weight_1",
            forge.Parameter(*(8, 16, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d2.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=False,
        )
        return conv2d_output_1


class Conv2D3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d3.weight_1",
            forge.Parameter(*(48, 8, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d3.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=False,
        )
        return conv2d_output_1


class Conv2D4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d4.weight_1",
            forge.Parameter(*(48, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d4.weight_1"),
            stride=[2, 2],
            padding=[0, 0, 1, 1],
            dilation=1,
            groups=48,
            channel_last=False,
        )
        return conv2d_output_1


class Conv2D5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d5.weight_1",
            forge.Parameter(*(8, 48, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d5.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=False,
        )
        return conv2d_output_1


class Conv2D6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d6.weight_1",
            forge.Parameter(*(48, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d6.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=48,
            channel_last=False,
        )
        return conv2d_output_1


class Conv2D7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d7.weight_1",
            forge.Parameter(*(16, 48, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d7.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=False,
        )
        return conv2d_output_1


class Conv2D8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d8.weight_1",
            forge.Parameter(*(96, 16, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d8.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=False,
        )
        return conv2d_output_1


class Conv2D9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d9.weight_1",
            forge.Parameter(*(96, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d9.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=96,
            channel_last=False,
        )
        return conv2d_output_1


class Conv2D10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d10.weight_1",
            forge.Parameter(*(96, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d10.weight_1"),
            stride=[2, 2],
            padding=[0, 0, 1, 1],
            dilation=1,
            groups=96,
            channel_last=False,
        )
        return conv2d_output_1


class Conv2D11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d11.weight_1",
            forge.Parameter(*(16, 96, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d11.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=False,
        )
        return conv2d_output_1


class Conv2D12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d12.weight_1",
            forge.Parameter(*(24, 96, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d12.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=False,
        )
        return conv2d_output_1


class Conv2D13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d13.weight_1",
            forge.Parameter(*(144, 24, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d13.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=False,
        )
        return conv2d_output_1


class Conv2D14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d14.weight_1",
            forge.Parameter(*(144, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d14.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=144,
            channel_last=False,
        )
        return conv2d_output_1


class Conv2D15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d15.weight_1",
            forge.Parameter(*(24, 144, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d15.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=False,
        )
        return conv2d_output_1


class Conv2D16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d16.weight_1",
            forge.Parameter(*(32, 144, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d16.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=False,
        )
        return conv2d_output_1


class Conv2D17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d17.weight_1",
            forge.Parameter(*(192, 32, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d17.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=False,
        )
        return conv2d_output_1


class Conv2D18(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d18.weight_1",
            forge.Parameter(*(192, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d18.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=192,
            channel_last=False,
        )
        return conv2d_output_1


class Conv2D19(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d19.weight_1",
            forge.Parameter(*(192, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d19.weight_1"),
            stride=[2, 2],
            padding=[0, 0, 1, 1],
            dilation=1,
            groups=192,
            channel_last=False,
        )
        return conv2d_output_1


class Conv2D20(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d20.weight_1",
            forge.Parameter(*(32, 192, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d20.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=False,
        )
        return conv2d_output_1


class Conv2D21(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d21.weight_1",
            forge.Parameter(*(56, 192, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d21.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=False,
        )
        return conv2d_output_1


class Conv2D22(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d22.weight_1",
            forge.Parameter(*(336, 56, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d22.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=False,
        )
        return conv2d_output_1


class Conv2D23(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d23.weight_1",
            forge.Parameter(*(336, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d23.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=336,
            channel_last=False,
        )
        return conv2d_output_1


class Conv2D24(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d24.weight_1",
            forge.Parameter(*(56, 336, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d24.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=False,
        )
        return conv2d_output_1


class Conv2D25(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d25.weight_1",
            forge.Parameter(*(112, 336, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d25.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=False,
        )
        return conv2d_output_1


class Conv2D26(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d26.weight_1",
            forge.Parameter(*(1280, 112, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d26.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=False,
        )
        return conv2d_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Conv2D0,
        [((1, 3, 96, 96), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[2, 2]",
                "padding": "[0, 0, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "False",
            },
        },
    ),
    (
        Conv2D1,
        [((1, 16, 48, 48), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "16",
                "channel_last": "False",
            },
        },
    ),
    (
        Conv2D2,
        [((1, 16, 48, 48), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "False",
            },
        },
    ),
    (
        Conv2D3,
        [((1, 8, 48, 48), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "False",
            },
        },
    ),
    (
        Conv2D4,
        [((1, 48, 48, 48), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[2, 2]",
                "padding": "[0, 0, 1, 1]",
                "dilation": "1",
                "groups": "48",
                "channel_last": "False",
            },
        },
    ),
    (
        Conv2D5,
        [((1, 48, 24, 24), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "False",
            },
        },
    ),
    (
        Conv2D3,
        [((1, 8, 24, 24), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "False",
            },
        },
    ),
    (
        Conv2D6,
        [((1, 48, 24, 24), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "48",
                "channel_last": "False",
            },
        },
    ),
    (
        Conv2D4,
        [((1, 48, 24, 24), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[2, 2]",
                "padding": "[0, 0, 1, 1]",
                "dilation": "1",
                "groups": "48",
                "channel_last": "False",
            },
        },
    ),
    (
        Conv2D7,
        [((1, 48, 12, 12), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "False",
            },
        },
    ),
    (
        Conv2D8,
        [((1, 16, 12, 12), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "False",
            },
        },
    ),
    (
        Conv2D9,
        [((1, 96, 12, 12), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "96",
                "channel_last": "False",
            },
        },
    ),
    (
        Conv2D10,
        [((1, 96, 12, 12), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[2, 2]",
                "padding": "[0, 0, 1, 1]",
                "dilation": "1",
                "groups": "96",
                "channel_last": "False",
            },
        },
    ),
    (
        Conv2D11,
        [((1, 96, 12, 12), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "False",
            },
        },
    ),
    (
        Conv2D12,
        [((1, 96, 6, 6), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "False",
            },
        },
    ),
    (
        Conv2D13,
        [((1, 24, 6, 6), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "False",
            },
        },
    ),
    (
        Conv2D14,
        [((1, 144, 6, 6), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "144",
                "channel_last": "False",
            },
        },
    ),
    (
        Conv2D15,
        [((1, 144, 6, 6), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "False",
            },
        },
    ),
    (
        Conv2D16,
        [((1, 144, 6, 6), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "False",
            },
        },
    ),
    (
        Conv2D17,
        [((1, 32, 6, 6), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "False",
            },
        },
    ),
    (
        Conv2D18,
        [((1, 192, 6, 6), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "192",
                "channel_last": "False",
            },
        },
    ),
    (
        Conv2D19,
        [((1, 192, 6, 6), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[2, 2]",
                "padding": "[0, 0, 1, 1]",
                "dilation": "1",
                "groups": "192",
                "channel_last": "False",
            },
        },
    ),
    (
        Conv2D20,
        [((1, 192, 6, 6), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "False",
            },
        },
    ),
    (
        Conv2D21,
        [((1, 192, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "False",
            },
        },
    ),
    (
        Conv2D22,
        [((1, 56, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "False",
            },
        },
    ),
    (
        Conv2D23,
        [((1, 336, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "336",
                "channel_last": "False",
            },
        },
    ),
    (
        Conv2D24,
        [((1, 336, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "False",
            },
        },
    ),
    (
        Conv2D25,
        [((1, 336, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "False",
            },
        },
    ),
    (
        Conv2D26,
        [((1, 112, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "False",
            },
        },
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    record_forge_op_name("Conv2d")

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
