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


class Conv2D0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d0.weight_1",
            forge.Parameter(*(16, 3, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d0.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d1.weight_1",
            forge.Parameter(*(16, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d2.weight_1",
            forge.Parameter(*(32, 16, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d3.weight_1",
            forge.Parameter(*(32, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d3.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=32,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d4.weight_1",
            forge.Parameter(*(48, 32, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d4.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d5.weight_1",
            forge.Parameter(*(48, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d5.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=48,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d6.weight_1",
            forge.Parameter(*(48, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d6.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=48,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d7.weight_1",
            forge.Parameter(*(48, 48, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d8.weight_1",
            forge.Parameter(*(96, 48, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d9.weight_1",
            forge.Parameter(*(96, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d10.weight_1",
            forge.Parameter(*(96, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d10.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=96,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d11.weight_1",
            forge.Parameter(*(96, 96, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d12.weight_1",
            forge.Parameter(*(192, 96, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d13.weight_1",
            forge.Parameter(*(192, 1, 5, 5), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d13.weight_1"),
            stride=[1, 1],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=192,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d14.weight_1",
            forge.Parameter(*(192, 1, 5, 5), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d14.weight_1"),
            stride=[2, 2],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=192,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d15.weight_1",
            forge.Parameter(*(192, 192, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d16.weight_1",
            forge.Parameter(*(48, 192, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d17.weight_1",
            forge.Parameter(*(192, 48, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D18(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d18.weight_1",
            forge.Parameter(*(384, 192, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d18.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D19(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d19.weight_1",
            forge.Parameter(*(384, 1, 5, 5), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d19.weight_1"),
            stride=[1, 1],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=384,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D20(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d20.weight_1",
            forge.Parameter(*(96, 384, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D21(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d21.weight_1",
            forge.Parameter(*(384, 96, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D22(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d22.weight_1",
            forge.Parameter(*(384, 384, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D23(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d23.weight_1",
            forge.Parameter(*(360, 384, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d23.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D24(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d24.weight_1",
            forge.Parameter(*(96, 360, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D25(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d25.weight_1",
            forge.Parameter(*(24, 96, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D26(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d26.weight_1",
            forge.Parameter(*(96, 24, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D27(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d27.weight_1",
            forge.Parameter(*(24, 96, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d27.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D28(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d28.weight_1",
            forge.Parameter(*(6, 24, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d28.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D29(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d29.weight_1",
            forge.Parameter(*(24, 6, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d29.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D30(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d30.weight_1",
            forge.Parameter(*(42, 192, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d30.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D31(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d31.weight_1",
            forge.Parameter(*(96, 42, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d31.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D32(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d32.weight_1",
            forge.Parameter(*(18, 96, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d32.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D33(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d33.weight_1",
            forge.Parameter(*(96, 18, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d33.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D34(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d34.weight_1",
            forge.Parameter(*(12, 48, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d34.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D35(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d35.weight_1",
            forge.Parameter(*(96, 12, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d35.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D36(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d36.weight_1",
            forge.Parameter(*(64, 3, 7, 7), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d36.weight_1"),
            stride=[2, 2],
            padding=[3, 3, 3, 3],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D37(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d37.weight_1",
            forge.Parameter(*(64, 64, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d37.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D38(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d38.weight_1",
            forge.Parameter(*(64, 64, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d38.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D39(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d39.weight_1",
            forge.Parameter(*(256, 64, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d39.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D40(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d40.weight_1",
            forge.Parameter(*(64, 256, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d40.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D41(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d41.weight_1",
            forge.Parameter(*(128, 256, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d41.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D42(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d42.weight_1",
            forge.Parameter(*(128, 128, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d42.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D43(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d43.weight_1",
            forge.Parameter(*(512, 128, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d43.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D44(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d44.weight_1",
            forge.Parameter(*(512, 256, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d44.weight_1"),
            stride=[2, 2],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D45(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d45.weight_1",
            forge.Parameter(*(128, 512, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d45.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D46(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d46.weight_1",
            forge.Parameter(*(128, 128, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d46.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D47(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d47.weight_1",
            forge.Parameter(*(256, 512, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d47.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D48(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d48.weight_1",
            forge.Parameter(*(256, 256, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d48.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D49(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d49.weight_1",
            forge.Parameter(*(1024, 256, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d49.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D50(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d50.weight_1",
            forge.Parameter(*(1024, 512, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d50.weight_1"),
            stride=[2, 2],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D51(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d51.weight_1",
            forge.Parameter(*(256, 1024, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d51.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D52(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d52.weight_1",
            forge.Parameter(*(256, 256, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d52.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D53(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d53.weight_1",
            forge.Parameter(*(512, 1024, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d53.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D54(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d54.weight_1",
            forge.Parameter(*(512, 512, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d54.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D55(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d55.weight_1",
            forge.Parameter(*(2048, 512, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d55.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D56(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d56.weight_1",
            forge.Parameter(*(2048, 1024, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d56.weight_1"),
            stride=[2, 2],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D57(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d57.weight_1",
            forge.Parameter(*(512, 2048, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d57.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D58(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d58.weight_1",
            forge.Parameter(*(512, 512, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d58.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D59(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d59.weight_1",
            forge.Parameter(*(256, 2048, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d59.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D60(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d60.weight_1",
            forge.Parameter(*(16, 3, 7, 7), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d60.weight_1"),
            stride=[1, 1],
            padding=[3, 3, 3, 3],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D61(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d61.weight_1",
            forge.Parameter(*(16, 16, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d61.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D62(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d62.weight_1",
            forge.Parameter(*(32, 16, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d62.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D63(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d63.weight_1",
            forge.Parameter(*(256, 32, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d63.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D64(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d64.weight_1",
            forge.Parameter(*(256, 4, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d64.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=64,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D65(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d65.weight_1",
            forge.Parameter(*(128, 32, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d65.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D66(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d66.weight_1",
            forge.Parameter(*(256, 128, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d66.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D67(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d67.weight_1",
            forge.Parameter(*(256, 4, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d67.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=64,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D68(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d68.weight_1",
            forge.Parameter(*(512, 8, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d68.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=64,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D69(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d69.weight_1",
            forge.Parameter(*(256, 128, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d69.weight_1"),
            stride=[2, 2],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D70(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d70.weight_1",
            forge.Parameter(*(512, 256, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d70.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D71(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d71.weight_1",
            forge.Parameter(*(512, 8, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d71.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=64,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D72(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d72.weight_1",
            forge.Parameter(*(256, 768, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d72.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D73(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d73.weight_1",
            forge.Parameter(*(256, 1152, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d73.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D74(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d74.weight_1",
            forge.Parameter(*(1024, 16, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d74.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=64,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D75(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d75.weight_1",
            forge.Parameter(*(1024, 512, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d75.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D76(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d76.weight_1",
            forge.Parameter(*(1024, 16, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d76.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=64,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D77(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d77.weight_1",
            forge.Parameter(*(512, 1536, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d77.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D78(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d78.weight_1",
            forge.Parameter(*(512, 2816, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d78.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D79(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d79.weight_1",
            forge.Parameter(*(2048, 32, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d79.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=64,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D80(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d80.weight_1",
            forge.Parameter(*(1024, 2048, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d80.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D81(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d81.weight_1",
            forge.Parameter(*(2048, 1024, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d81.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D82(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d82.weight_1",
            forge.Parameter(*(2048, 32, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d82.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=64,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D83(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d83.weight_1",
            forge.Parameter(*(1024, 2560, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d83.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D84(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d84.weight_1",
            forge.Parameter(*(1000, 1024, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d84.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D85(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d85.weight_1",
            forge.Parameter(*(64, 32, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d85.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D86(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d86.weight_1",
            forge.Parameter(*(64, 2, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d86.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=32,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D87(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d87.weight_1",
            forge.Parameter(*(64, 2, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d87.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=32,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D88(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d88.weight_1",
            forge.Parameter(*(64, 128, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d88.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D89(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d89.weight_1",
            forge.Parameter(*(128, 64, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d89.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D90(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d90.weight_1",
            forge.Parameter(*(128, 4, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d90.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=32,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D91(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d91.weight_1",
            forge.Parameter(*(128, 128, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d91.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D92(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d92.weight_1",
            forge.Parameter(*(128, 4, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d92.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=32,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D93(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d93.weight_1",
            forge.Parameter(*(128, 448, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d93.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D94(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d94.weight_1",
            forge.Parameter(*(256, 8, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d94.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=32,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D95(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d95.weight_1",
            forge.Parameter(*(256, 256, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d95.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D96(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d96.weight_1",
            forge.Parameter(*(256, 8, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d96.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=32,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D97(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d97.weight_1",
            forge.Parameter(*(256, 640, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d97.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D98(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d98.weight_1",
            forge.Parameter(*(1000, 256, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d98.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D99(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d99.weight_1",
            forge.Parameter(*(40, 3, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d99.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D100(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d100.weight_1",
            forge.Parameter(*(40, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d100.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=40,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D101(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d101.weight_1",
            forge.Parameter(*(10, 40, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d101.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D102(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d102.weight_1",
            forge.Parameter(*(40, 10, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d102.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D103(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d103.weight_1",
            forge.Parameter(*(24, 40, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d103.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D104(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d104.weight_1",
            forge.Parameter(*(24, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d104.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=24,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D105(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d105.weight_1",
            forge.Parameter(*(24, 24, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d105.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D106(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d106.weight_1",
            forge.Parameter(*(144, 24, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d106.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D107(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d107.weight_1",
            forge.Parameter(*(144, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d107.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=144,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D108(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d108.weight_1",
            forge.Parameter(*(6, 144, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d108.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D109(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d109.weight_1",
            forge.Parameter(*(144, 6, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d109.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D110(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d110.weight_1",
            forge.Parameter(*(32, 144, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d110.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D111(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d111.weight_1",
            forge.Parameter(*(192, 32, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d111.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D112(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d112.weight_1",
            forge.Parameter(*(192, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d112.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=192,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D113(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d113.weight_1",
            forge.Parameter(*(8, 192, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d113.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D114(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d114.weight_1",
            forge.Parameter(*(192, 8, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d114.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D115(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d115.weight_1",
            forge.Parameter(*(32, 192, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d115.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D116(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d116.weight_1",
            forge.Parameter(*(288, 48, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d116.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D117(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d117.weight_1",
            forge.Parameter(*(288, 1, 5, 5), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d117.weight_1"),
            stride=[1, 1],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=288,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D118(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d118.weight_1",
            forge.Parameter(*(12, 288, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d118.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D119(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d119.weight_1",
            forge.Parameter(*(288, 12, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d119.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D120(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d120.weight_1",
            forge.Parameter(*(48, 288, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d120.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D121(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d121.weight_1",
            forge.Parameter(*(288, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d121.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=288,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D122(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d122.weight_1",
            forge.Parameter(*(96, 288, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d122.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D123(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d123.weight_1",
            forge.Parameter(*(576, 96, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d123.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D124(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d124.weight_1",
            forge.Parameter(*(576, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d124.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=576,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D125(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d125.weight_1",
            forge.Parameter(*(24, 576, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d125.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D126(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d126.weight_1",
            forge.Parameter(*(576, 24, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d126.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D127(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d127.weight_1",
            forge.Parameter(*(96, 576, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d127.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D128(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d128.weight_1",
            forge.Parameter(*(576, 1, 5, 5), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d128.weight_1"),
            stride=[1, 1],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=576,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D129(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d129.weight_1",
            forge.Parameter(*(136, 576, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d129.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D130(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d130.weight_1",
            forge.Parameter(*(816, 136, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d130.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D131(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d131.weight_1",
            forge.Parameter(*(816, 1, 5, 5), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d131.weight_1"),
            stride=[1, 1],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=816,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D132(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d132.weight_1",
            forge.Parameter(*(816, 1, 5, 5), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d132.weight_1"),
            stride=[2, 2],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=816,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D133(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d133.weight_1",
            forge.Parameter(*(34, 816, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d133.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D134(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d134.weight_1",
            forge.Parameter(*(816, 34, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d134.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D135(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d135.weight_1",
            forge.Parameter(*(136, 816, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d135.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D136(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d136.weight_1",
            forge.Parameter(*(232, 816, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d136.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D137(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d137.weight_1",
            forge.Parameter(*(1392, 232, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d137.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D138(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d138.weight_1",
            forge.Parameter(*(1392, 1, 5, 5), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d138.weight_1"),
            stride=[1, 1],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=1392,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D139(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d139.weight_1",
            forge.Parameter(*(58, 1392, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d139.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D140(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d140.weight_1",
            forge.Parameter(*(1392, 58, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d140.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D141(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d141.weight_1",
            forge.Parameter(*(232, 1392, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d141.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D142(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d142.weight_1",
            forge.Parameter(*(1392, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d142.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1392,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D143(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d143.weight_1",
            forge.Parameter(*(384, 1392, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d143.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D144(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d144.weight_1",
            forge.Parameter(*(2304, 384, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d144.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D145(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d145.weight_1",
            forge.Parameter(*(2304, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d145.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=2304,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D146(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d146.weight_1",
            forge.Parameter(*(96, 2304, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d146.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D147(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d147.weight_1",
            forge.Parameter(*(2304, 96, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d147.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D148(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d148.weight_1",
            forge.Parameter(*(384, 2304, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d148.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D149(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d149.weight_1",
            forge.Parameter(*(1536, 384, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d149.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D150(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d150.weight_1",
            forge.Parameter(*(64, 3, 7, 7), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d150.weight_1"),
            stride=[4, 4],
            padding=[3, 3, 3, 3],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D151(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d151.weight_1",
            forge.Parameter(*(64, 64, 8, 8), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d151.weight_1"),
            stride=[8, 8],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D152(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d152.weight_1",
            forge.Parameter(*(256, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d152.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=256,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D153(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d153.weight_1",
            forge.Parameter(*(128, 64, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d153.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D154(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d154.weight_1",
            forge.Parameter(*(128, 128, 4, 4), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d154.weight_1"),
            stride=[4, 4],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D155(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d155.weight_1",
            forge.Parameter(*(512, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d155.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=512,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D156(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d156.weight_1",
            forge.Parameter(*(320, 128, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d156.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D157(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d157.weight_1",
            forge.Parameter(*(320, 320, 2, 2), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d157.weight_1"),
            stride=[2, 2],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D158(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d158.weight_1",
            forge.Parameter(*(1280, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d158.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1280,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D159(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d159.weight_1",
            forge.Parameter(*(512, 320, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d159.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D160(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d160.weight_1",
            forge.Parameter(*(2048, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d160.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=2048,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D161(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d161.weight_1",
            forge.Parameter(*(768, 3072, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d161.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D162(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d162.weight_1",
            forge.Parameter(*(150, 768, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d162.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D163(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d163.weight_1",
            forge.Parameter(*(64, 3, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d163.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D164(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d164.weight_1",
            forge.Parameter(*(64, 64, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d164.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D165(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d165.weight_1",
            forge.Parameter(*(64, 128, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d165.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D166(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d166.weight_1",
            forge.Parameter(*(80, 128, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d166.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D167(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d167.weight_1",
            forge.Parameter(*(80, 80, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d167.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D168(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d168.weight_1",
            forge.Parameter(*(256, 528, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d168.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D169(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d169.weight_1",
            forge.Parameter(*(96, 256, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d169.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D170(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d170.weight_1",
            forge.Parameter(*(96, 96, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d170.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D171(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d171.weight_1",
            forge.Parameter(*(384, 736, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d171.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D172(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d172.weight_1",
            forge.Parameter(*(112, 384, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d172.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D173(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d173.weight_1",
            forge.Parameter(*(112, 112, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d173.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D174(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d174.weight_1",
            forge.Parameter(*(512, 944, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d174.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D175(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d175.weight_1",
            forge.Parameter(*(32, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d175.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=32,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D176(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d176.weight_1",
            forge.Parameter(*(64, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d176.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=64,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D177(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d177.weight_1",
            forge.Parameter(*(64, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d177.weight_1"),
            stride=[2, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=64,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D178(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d178.weight_1",
            forge.Parameter(*(128, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d178.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=128,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D179(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d179.weight_1",
            forge.Parameter(*(128, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d179.weight_1"),
            stride=[1, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=128,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D180(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d180.weight_1",
            forge.Parameter(*(240, 128, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d180.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D181(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d181.weight_1",
            forge.Parameter(*(240, 1, 5, 5), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d181.weight_1"),
            stride=[1, 1],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=240,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D182(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d182.weight_1",
            forge.Parameter(*(240, 1, 5, 5), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d182.weight_1"),
            stride=[2, 1],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=240,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D183(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d183.weight_1",
            forge.Parameter(*(240, 240, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d183.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D184(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d184.weight_1",
            forge.Parameter(*(60, 240, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d184.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D185(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d185.weight_1",
            forge.Parameter(*(240, 60, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d185.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D186(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d186.weight_1",
            forge.Parameter(*(480, 240, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d186.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D187(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d187.weight_1",
            forge.Parameter(*(480, 1, 5, 5), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d187.weight_1"),
            stride=[1, 1],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=480,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D188(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d188.weight_1",
            forge.Parameter(*(480, 1, 5, 5), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d188.weight_1"),
            stride=[2, 1],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=480,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D189(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d189.weight_1",
            forge.Parameter(*(120, 480, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d189.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D190(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d190.weight_1",
            forge.Parameter(*(480, 120, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d190.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D191(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d191.weight_1",
            forge.Parameter(*(480, 480, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d191.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D192(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d192.weight_1",
            forge.Parameter(*(60, 480, 1, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d192.weight_1"),
            stride=[1, 1],
            padding=[0, 1, 0, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D193(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d193.weight_1",
            forge.Parameter(*(120, 60, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d193.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D194(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d194.weight_1",
            forge.Parameter(*(60, 960, 1, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d194.weight_1"),
            stride=[1, 1],
            padding=[0, 1, 0, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D195(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[1, 1],
            padding=[1, 0, 1, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D196(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[2, 1],
            padding=[1, 0, 1, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D197(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d197.weight_1",
            forge.Parameter(*(512, 16, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d197.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=32,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D198(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d198.weight_1",
            forge.Parameter(*(512, 512, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d198.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D199(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d199.weight_1",
            forge.Parameter(*(512, 16, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d199.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=32,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D200(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d200.weight_1",
            forge.Parameter(*(1024, 32, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d200.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=32,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D201(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d201.weight_1",
            forge.Parameter(*(1024, 1024, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d201.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D202(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d202.weight_1",
            forge.Parameter(*(1024, 32, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d202.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=32,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D203(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d203.weight_1",
            forge.Parameter(*(128, 64, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d203.weight_1"),
            stride=[2, 2],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D204(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d204.weight_1",
            forge.Parameter(*(256, 896, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d204.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D205(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d205.weight_1",
            forge.Parameter(*(512, 2304, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d205.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D206(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d206.weight_1",
            forge.Parameter(*(32, 3, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d206.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D207(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d207.weight_1",
            forge.Parameter(*(8, 32, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d207.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D208(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d208.weight_1",
            forge.Parameter(*(32, 8, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d208.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D209(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d209.weight_1",
            forge.Parameter(*(16, 32, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d209.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D210(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d210.weight_1",
            forge.Parameter(*(96, 16, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d210.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D211(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d211.weight_1",
            forge.Parameter(*(4, 96, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d211.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D212(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d212.weight_1",
            forge.Parameter(*(96, 4, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d212.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D213(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d213.weight_1",
            forge.Parameter(*(144, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d213.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=144,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D214(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d214.weight_1",
            forge.Parameter(*(24, 144, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d214.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D215(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d215.weight_1",
            forge.Parameter(*(144, 1, 5, 5), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d215.weight_1"),
            stride=[2, 2],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=144,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D216(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d216.weight_1",
            forge.Parameter(*(40, 144, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d216.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D217(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d217.weight_1",
            forge.Parameter(*(240, 40, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d217.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D218(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d218.weight_1",
            forge.Parameter(*(10, 240, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d218.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D219(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d219.weight_1",
            forge.Parameter(*(240, 10, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d219.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D220(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d220.weight_1",
            forge.Parameter(*(40, 240, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d220.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D221(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d221.weight_1",
            forge.Parameter(*(240, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d221.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=240,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D222(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d222.weight_1",
            forge.Parameter(*(80, 240, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d222.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D223(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d223.weight_1",
            forge.Parameter(*(480, 80, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d223.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D224(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d224.weight_1",
            forge.Parameter(*(480, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d224.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=480,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D225(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d225.weight_1",
            forge.Parameter(*(20, 480, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d225.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D226(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d226.weight_1",
            forge.Parameter(*(480, 20, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d226.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D227(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d227.weight_1",
            forge.Parameter(*(80, 480, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d227.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D228(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d228.weight_1",
            forge.Parameter(*(112, 480, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d228.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D229(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d229.weight_1",
            forge.Parameter(*(672, 112, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d229.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D230(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d230.weight_1",
            forge.Parameter(*(672, 1, 5, 5), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d230.weight_1"),
            stride=[1, 1],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=672,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D231(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d231.weight_1",
            forge.Parameter(*(672, 1, 5, 5), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d231.weight_1"),
            stride=[2, 2],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=672,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D232(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d232.weight_1",
            forge.Parameter(*(28, 672, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d232.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D233(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d233.weight_1",
            forge.Parameter(*(672, 28, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d233.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D234(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d234.weight_1",
            forge.Parameter(*(112, 672, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d234.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D235(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d235.weight_1",
            forge.Parameter(*(192, 672, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d235.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D236(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d236.weight_1",
            forge.Parameter(*(1152, 192, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d236.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D237(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d237.weight_1",
            forge.Parameter(*(1152, 1, 5, 5), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d237.weight_1"),
            stride=[1, 1],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=1152,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D238(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d238.weight_1",
            forge.Parameter(*(48, 1152, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d238.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D239(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d239.weight_1",
            forge.Parameter(*(1152, 48, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d239.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D240(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d240.weight_1",
            forge.Parameter(*(192, 1152, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d240.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D241(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d241.weight_1",
            forge.Parameter(*(1152, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d241.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1152,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D242(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d242.weight_1",
            forge.Parameter(*(320, 1152, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d242.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D243(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d243.weight_1",
            forge.Parameter(*(1280, 320, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d243.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D244(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d244.weight_1",
            forge.Parameter(*(8, 16, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d244.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D245(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d245.weight_1",
            forge.Parameter(*(48, 8, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d245.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D246(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d246.weight_1",
            forge.Parameter(*(16, 48, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d246.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D247(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d247.weight_1",
            forge.Parameter(*(16, 96, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d247.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D248(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d248.weight_1",
            forge.Parameter(*(32, 96, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d248.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D249(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d249.weight_1",
            forge.Parameter(*(288, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d249.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=288,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D250(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d250.weight_1",
            forge.Parameter(*(80, 288, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d250.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D251(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d251.weight_1",
            forge.Parameter(*(160, 480, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d251.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D252(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d252.weight_1",
            forge.Parameter(*(1280, 160, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d252.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D253(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d253.weight_1",
            forge.Parameter(*(160, 256, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d253.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D254(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d254.weight_1",
            forge.Parameter(*(160, 160, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d254.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D255(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d255.weight_1",
            forge.Parameter(*(512, 1056, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d255.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D256(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d256.weight_1",
            forge.Parameter(*(192, 512, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d256.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D257(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d257.weight_1",
            forge.Parameter(*(192, 192, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d257.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D258(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d258.weight_1",
            forge.Parameter(*(768, 1472, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d258.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D259(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d259.weight_1",
            forge.Parameter(*(192, 768, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d259.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D260(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d260.weight_1",
            forge.Parameter(*(768, 1728, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d260.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D261(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d261.weight_1",
            forge.Parameter(*(224, 768, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d261.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D262(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d262.weight_1",
            forge.Parameter(*(224, 224, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d262.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D263(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d263.weight_1",
            forge.Parameter(*(1024, 1888, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d263.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D264(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d264.weight_1",
            forge.Parameter(*(224, 1024, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d264.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D265(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d265.weight_1",
            forge.Parameter(*(1024, 2144, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d265.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D266(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[2, 2],
            padding=[3, 3, 3, 3],
            dilation=1,
            groups=1,
            channel_last=1,
        )
        return conv2d_output_1


class Conv2D267(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=1,
        )
        return conv2d_output_1


class Conv2D268(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=1,
        )
        return conv2d_output_1


class Conv2D269(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=1,
        )
        return conv2d_output_1


class Conv2D270(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[2, 2],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=1,
        )
        return conv2d_output_1


class Conv2D271(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d271.weight_1",
            forge.Parameter(*(32, 256, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d271.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D272(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d272.weight_1",
            forge.Parameter(*(64, 512, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d272.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D273(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d273.weight_1",
            forge.Parameter(*(128, 1024, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d273.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D274(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d274.weight_1",
            forge.Parameter(*(264, 264, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d274.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D275(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d275.weight_1",
            forge.Parameter(*(128, 264, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d275.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D276(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d276.weight_1",
            forge.Parameter(*(32, 64, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d276.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D277(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d277.weight_1",
            forge.Parameter(*(16, 32, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d277.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D278(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d278.weight_1",
            forge.Parameter(*(1, 16, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d278.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D279(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d279.weight_1",
            forge.Parameter(*(128, 384, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d279.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D280(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d280.weight_1",
            forge.Parameter(*(512, 2560, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d280.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D281(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d281.weight_1",
            forge.Parameter(*(512, 3328, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d281.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D282(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d282.weight_1",
            forge.Parameter(*(4, 16, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d282.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D283(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d283.weight_1",
            forge.Parameter(*(16, 4, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d283.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D284(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d284.weight_1",
            forge.Parameter(*(16, 16, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d284.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D285(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d285.weight_1",
            forge.Parameter(*(1920, 320, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d285.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D286(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d286.weight_1",
            forge.Parameter(*(1920, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d286.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1920,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D287(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d287.weight_1",
            forge.Parameter(*(80, 1920, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d287.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D288(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d288.weight_1",
            forge.Parameter(*(1920, 80, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d288.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D289(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d289.weight_1",
            forge.Parameter(*(320, 1920, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d289.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D290(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d290.weight_1",
            forge.Parameter(*(48, 3, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d290.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D291(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d291.weight_1",
            forge.Parameter(*(48, 12, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d291.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D292(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d292.weight_1",
            forge.Parameter(*(24, 48, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d292.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D293(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d293.weight_1",
            forge.Parameter(*(56, 192, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d293.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D294(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d294.weight_1",
            forge.Parameter(*(336, 56, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d294.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D295(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d295.weight_1",
            forge.Parameter(*(336, 1, 5, 5), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d295.weight_1"),
            stride=[1, 1],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=336,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D296(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d296.weight_1",
            forge.Parameter(*(14, 336, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d296.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D297(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d297.weight_1",
            forge.Parameter(*(336, 14, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d297.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D298(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d298.weight_1",
            forge.Parameter(*(56, 336, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d298.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D299(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d299.weight_1",
            forge.Parameter(*(336, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d299.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=336,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D300(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d300.weight_1",
            forge.Parameter(*(112, 336, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d300.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D301(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d301.weight_1",
            forge.Parameter(*(672, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d301.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=672,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D302(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d302.weight_1",
            forge.Parameter(*(160, 672, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d302.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D303(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d303.weight_1",
            forge.Parameter(*(960, 160, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d303.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D304(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d304.weight_1",
            forge.Parameter(*(960, 1, 5, 5), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d304.weight_1"),
            stride=[1, 1],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=960,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D305(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d305.weight_1",
            forge.Parameter(*(960, 1, 5, 5), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d305.weight_1"),
            stride=[2, 2],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=960,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D306(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d306.weight_1",
            forge.Parameter(*(40, 960, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d306.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D307(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d307.weight_1",
            forge.Parameter(*(960, 40, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d307.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D308(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d308.weight_1",
            forge.Parameter(*(160, 960, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d308.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D309(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d309.weight_1",
            forge.Parameter(*(272, 960, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d309.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D310(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d310.weight_1",
            forge.Parameter(*(1632, 272, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d310.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D311(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d311.weight_1",
            forge.Parameter(*(1632, 1, 5, 5), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d311.weight_1"),
            stride=[1, 1],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=1632,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D312(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d312.weight_1",
            forge.Parameter(*(68, 1632, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d312.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D313(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d313.weight_1",
            forge.Parameter(*(1632, 68, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d313.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D314(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d314.weight_1",
            forge.Parameter(*(272, 1632, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d314.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D315(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d315.weight_1",
            forge.Parameter(*(1632, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d315.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1632,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D316(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d316.weight_1",
            forge.Parameter(*(448, 1632, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d316.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D317(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d317.weight_1",
            forge.Parameter(*(2688, 448, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d317.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D318(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d318.weight_1",
            forge.Parameter(*(2688, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d318.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=2688,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D319(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d319.weight_1",
            forge.Parameter(*(112, 2688, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d319.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D320(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d320.weight_1",
            forge.Parameter(*(2688, 112, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d320.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D321(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d321.weight_1",
            forge.Parameter(*(448, 2688, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d321.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D322(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d322.weight_1",
            forge.Parameter(*(1792, 448, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d322.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D323(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d323.weight_1",
            forge.Parameter(*(192, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d323.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=192,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D324(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d324.weight_1",
            forge.Parameter(*(64, 192, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d324.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D325(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d325.weight_1",
            forge.Parameter(*(384, 64, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d325.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D326(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d326.weight_1",
            forge.Parameter(*(384, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d326.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=384,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D327(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d327.weight_1",
            forge.Parameter(*(64, 384, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d327.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D328(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d328.weight_1",
            forge.Parameter(*(576, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d328.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=576,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D329(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d329.weight_1",
            forge.Parameter(*(160, 576, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d329.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D330(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d330.weight_1",
            forge.Parameter(*(960, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d330.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=960,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D331(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d331.weight_1",
            forge.Parameter(*(320, 960, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d331.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D332(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d332.weight_1",
            forge.Parameter(*(32, 3, 7, 7), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d332.weight_1"),
            stride=[4, 4],
            padding=[3, 3, 3, 3],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D333(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d333.weight_1",
            forge.Parameter(*(32, 32, 8, 8), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d333.weight_1"),
            stride=[8, 8],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D334(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d334.weight_1",
            forge.Parameter(*(64, 32, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d334.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D335(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d335.weight_1",
            forge.Parameter(*(64, 64, 4, 4), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d335.weight_1"),
            stride=[4, 4],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D336(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d336.weight_1",
            forge.Parameter(*(160, 64, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d336.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D337(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d337.weight_1",
            forge.Parameter(*(160, 160, 2, 2), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d337.weight_1"),
            stride=[2, 2],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D338(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d338.weight_1",
            forge.Parameter(*(640, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d338.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=640,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D339(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d339.weight_1",
            forge.Parameter(*(256, 160, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d339.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D340(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d340.weight_1",
            forge.Parameter(*(1024, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d340.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1024,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D341(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d341.weight_1",
            forge.Parameter(*(768, 3, 16, 16), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d341.weight_1"),
            stride=[16, 16],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D342(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d342.weight_1",
            forge.Parameter(*(32, 32, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d342.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D343(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d343.weight_1",
            forge.Parameter(*(32, 48, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d343.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D344(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d344.weight_1",
            forge.Parameter(*(32, 32, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d344.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D345(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d345.weight_1",
            forge.Parameter(*(256, 128, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d345.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D346(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d346.weight_1",
            forge.Parameter(*(256, 384, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d346.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D347(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d347.weight_1",
            forge.Parameter(*(128, 192, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d347.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D348(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d348.weight_1",
            forge.Parameter(*(64, 96, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d348.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D349(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d349.weight_1",
            forge.Parameter(*(80, 64, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d349.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D350(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d350.weight_1",
            forge.Parameter(*(80, 80, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d350.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D351(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d351.weight_1",
            forge.Parameter(*(64, 256, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d351.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D352(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d352.weight_1",
            forge.Parameter(*(80, 256, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d352.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D353(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d353.weight_1",
            forge.Parameter(*(1, 16, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d353.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D354(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d354.weight_1",
            forge.Parameter(*(8, 3, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d354.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D355(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d355.weight_1",
            forge.Parameter(*(8, 8, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d355.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D356(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d356.weight_1",
            forge.Parameter(*(8, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d356.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=8,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D357(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d357.weight_1",
            forge.Parameter(*(2, 8, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d357.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D358(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d358.weight_1",
            forge.Parameter(*(8, 2, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d358.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D359(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d359.weight_1",
            forge.Parameter(*(40, 8, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d359.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D360(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d360.weight_1",
            forge.Parameter(*(40, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d360.weight_1"),
            stride=[2, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=40,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D361(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d361.weight_1",
            forge.Parameter(*(16, 40, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d361.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D362(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d362.weight_1",
            forge.Parameter(*(48, 16, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d362.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D363(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d363.weight_1",
            forge.Parameter(*(48, 1, 5, 5), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d363.weight_1"),
            stride=[2, 1],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=48,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D364(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d364.weight_1",
            forge.Parameter(*(120, 24, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d364.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D365(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d365.weight_1",
            forge.Parameter(*(120, 1, 5, 5), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d365.weight_1"),
            stride=[1, 1],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=120,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D366(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d366.weight_1",
            forge.Parameter(*(30, 120, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d366.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D367(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d367.weight_1",
            forge.Parameter(*(120, 30, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d367.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D368(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d368.weight_1",
            forge.Parameter(*(24, 120, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d368.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D369(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d369.weight_1",
            forge.Parameter(*(64, 24, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d369.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D370(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d370.weight_1",
            forge.Parameter(*(64, 1, 5, 5), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d370.weight_1"),
            stride=[1, 1],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=64,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D371(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d371.weight_1",
            forge.Parameter(*(16, 64, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d371.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D372(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d372.weight_1",
            forge.Parameter(*(64, 16, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d372.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D373(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d373.weight_1",
            forge.Parameter(*(24, 64, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d373.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D374(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d374.weight_1",
            forge.Parameter(*(72, 24, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d374.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D375(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d375.weight_1",
            forge.Parameter(*(72, 1, 5, 5), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d375.weight_1"),
            stride=[1, 1],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=72,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D376(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d376.weight_1",
            forge.Parameter(*(18, 72, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d376.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D377(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d377.weight_1",
            forge.Parameter(*(72, 18, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d377.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D378(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d378.weight_1",
            forge.Parameter(*(24, 72, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d378.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D379(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d379.weight_1",
            forge.Parameter(*(144, 1, 5, 5), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d379.weight_1"),
            stride=[2, 1],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=144,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D380(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d380.weight_1",
            forge.Parameter(*(36, 144, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d380.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D381(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d381.weight_1",
            forge.Parameter(*(144, 36, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d381.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D382(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d382.weight_1",
            forge.Parameter(*(48, 144, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d382.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D383(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d383.weight_1",
            forge.Parameter(*(72, 288, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d383.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D384(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d384.weight_1",
            forge.Parameter(*(288, 72, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d384.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D385(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d385.weight_1",
            forge.Parameter(*(512, 256, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d385.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D386(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d386.weight_1",
            forge.Parameter(*(512, 1280, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d386.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D387(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d387.weight_1",
            forge.Parameter(*(1000, 512, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d387.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D388(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d388.weight_1",
            forge.Parameter(*(128, 576, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d388.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D389(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d389.weight_1",
            forge.Parameter(*(88, 288, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d389.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D390(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d390.weight_1",
            forge.Parameter(*(528, 88, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d390.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D391(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d391.weight_1",
            forge.Parameter(*(528, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d391.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=528,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D392(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d392.weight_1",
            forge.Parameter(*(22, 528, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d392.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D393(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d393.weight_1",
            forge.Parameter(*(528, 22, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d393.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D394(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d394.weight_1",
            forge.Parameter(*(88, 528, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d394.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D395(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d395.weight_1",
            forge.Parameter(*(528, 1, 5, 5), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d395.weight_1"),
            stride=[1, 1],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=528,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D396(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d396.weight_1",
            forge.Parameter(*(120, 528, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d396.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D397(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d397.weight_1",
            forge.Parameter(*(720, 120, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d397.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D398(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d398.weight_1",
            forge.Parameter(*(720, 1, 5, 5), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d398.weight_1"),
            stride=[1, 1],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=720,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D399(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d399.weight_1",
            forge.Parameter(*(720, 1, 5, 5), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d399.weight_1"),
            stride=[2, 2],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=720,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D400(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d400.weight_1",
            forge.Parameter(*(30, 720, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d400.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D401(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d401.weight_1",
            forge.Parameter(*(720, 30, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d401.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D402(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d402.weight_1",
            forge.Parameter(*(120, 720, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d402.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D403(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d403.weight_1",
            forge.Parameter(*(208, 720, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d403.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D404(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d404.weight_1",
            forge.Parameter(*(1248, 208, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d404.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D405(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d405.weight_1",
            forge.Parameter(*(1248, 1, 5, 5), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d405.weight_1"),
            stride=[1, 1],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=1248,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D406(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d406.weight_1",
            forge.Parameter(*(52, 1248, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d406.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D407(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d407.weight_1",
            forge.Parameter(*(1248, 52, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d407.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D408(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d408.weight_1",
            forge.Parameter(*(208, 1248, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d408.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D409(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d409.weight_1",
            forge.Parameter(*(1248, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d409.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1248,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D410(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d410.weight_1",
            forge.Parameter(*(352, 1248, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d410.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D411(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d411.weight_1",
            forge.Parameter(*(2112, 352, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d411.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D412(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d412.weight_1",
            forge.Parameter(*(2112, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d412.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=2112,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D413(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d413.weight_1",
            forge.Parameter(*(88, 2112, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d413.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D414(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d414.weight_1",
            forge.Parameter(*(2112, 88, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d414.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D415(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d415.weight_1",
            forge.Parameter(*(352, 2112, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d415.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D416(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d416.weight_1",
            forge.Parameter(*(1408, 352, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d416.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D417(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d417.weight_1",
            forge.Parameter(*(240, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d417.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=240,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D418(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d418.weight_1",
            forge.Parameter(*(240, 1, 5, 5), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d418.weight_1"),
            stride=[2, 2],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=240,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D419(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d419.weight_1",
            forge.Parameter(*(64, 240, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d419.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D420(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d420.weight_1",
            forge.Parameter(*(16, 384, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d420.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D421(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d421.weight_1",
            forge.Parameter(*(384, 16, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d421.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D422(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d422.weight_1",
            forge.Parameter(*(384, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d422.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=384,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D423(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d423.weight_1",
            forge.Parameter(*(768, 128, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d423.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D424(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d424.weight_1",
            forge.Parameter(*(768, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d424.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=768,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D425(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d425.weight_1",
            forge.Parameter(*(32, 768, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d425.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D426(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d426.weight_1",
            forge.Parameter(*(768, 32, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d426.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D427(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d427.weight_1",
            forge.Parameter(*(128, 768, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d427.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D428(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d428.weight_1",
            forge.Parameter(*(768, 1, 5, 5), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d428.weight_1"),
            stride=[1, 1],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=768,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D429(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d429.weight_1",
            forge.Parameter(*(176, 768, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d429.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D430(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d430.weight_1",
            forge.Parameter(*(1056, 176, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d430.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D431(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d431.weight_1",
            forge.Parameter(*(1056, 1, 5, 5), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d431.weight_1"),
            stride=[1, 1],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=1056,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D432(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d432.weight_1",
            forge.Parameter(*(1056, 1, 5, 5), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d432.weight_1"),
            stride=[2, 2],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=1056,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D433(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d433.weight_1",
            forge.Parameter(*(44, 1056, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d433.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D434(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d434.weight_1",
            forge.Parameter(*(1056, 44, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d434.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D435(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d435.weight_1",
            forge.Parameter(*(176, 1056, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d435.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D436(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d436.weight_1",
            forge.Parameter(*(304, 1056, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d436.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D437(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d437.weight_1",
            forge.Parameter(*(1824, 304, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d437.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D438(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d438.weight_1",
            forge.Parameter(*(1824, 1, 5, 5), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d438.weight_1"),
            stride=[1, 1],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=1824,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D439(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d439.weight_1",
            forge.Parameter(*(76, 1824, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d439.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D440(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d440.weight_1",
            forge.Parameter(*(1824, 76, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d440.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D441(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d441.weight_1",
            forge.Parameter(*(304, 1824, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d441.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D442(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d442.weight_1",
            forge.Parameter(*(1824, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d442.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1824,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D443(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d443.weight_1",
            forge.Parameter(*(512, 1824, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d443.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D444(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d444.weight_1",
            forge.Parameter(*(3072, 512, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d444.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D445(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d445.weight_1",
            forge.Parameter(*(3072, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d445.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=3072,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D446(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d446.weight_1",
            forge.Parameter(*(128, 3072, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d446.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D447(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d447.weight_1",
            forge.Parameter(*(3072, 128, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d447.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D448(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d448.weight_1",
            forge.Parameter(*(512, 3072, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d448.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D449(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d449.weight_1",
            forge.Parameter(*(72, 192, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d449.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D450(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d450.weight_1",
            forge.Parameter(*(432, 72, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d450.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D451(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d451.weight_1",
            forge.Parameter(*(432, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d451.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=432,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D452(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d452.weight_1",
            forge.Parameter(*(72, 432, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d452.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D453(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d453.weight_1",
            forge.Parameter(*(104, 432, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d453.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D454(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d454.weight_1",
            forge.Parameter(*(624, 104, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d454.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D455(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d455.weight_1",
            forge.Parameter(*(624, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d455.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=624,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D456(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d456.weight_1",
            forge.Parameter(*(624, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d456.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=624,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D457(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d457.weight_1",
            forge.Parameter(*(104, 624, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d457.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D458(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d458.weight_1",
            forge.Parameter(*(176, 624, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d458.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D459(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d459.weight_1",
            forge.Parameter(*(1056, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d459.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1056,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D460(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d460.weight_1",
            forge.Parameter(*(352, 1056, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d460.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D461(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d461.weight_1",
            forge.Parameter(*(1280, 352, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d461.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D462(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d462.weight_1",
            forge.Parameter(*(150, 256, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d462.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D463(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d463.weight_1",
            forge.Parameter(*(96, 3, 4, 4), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d463.weight_1"),
            stride=[4, 4],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D464(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d464.weight_1",
            forge.Parameter(*(64, 3, 11, 11), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "conv2d464.weight_2", forge.Parameter(*(64,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d464.weight_1"),
            self.get_parameter("conv2d464.weight_2"),
            stride=[4, 4],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D465(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d465.weight_1",
            forge.Parameter(*(192, 64, 5, 5), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "conv2d465.weight_2", forge.Parameter(*(192,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d465.weight_1"),
            self.get_parameter("conv2d465.weight_2"),
            stride=[1, 1],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D466(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d466.weight_1",
            forge.Parameter(*(384, 192, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "conv2d466.weight_2", forge.Parameter(*(384,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d466.weight_1"),
            self.get_parameter("conv2d466.weight_2"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D467(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d467.weight_1",
            forge.Parameter(*(256, 384, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "conv2d467.weight_2", forge.Parameter(*(256,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d467.weight_1"),
            self.get_parameter("conv2d467.weight_2"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D468(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d468.weight_1",
            forge.Parameter(*(256, 256, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "conv2d468.weight_2", forge.Parameter(*(256,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d468.weight_1"),
            self.get_parameter("conv2d468.weight_2"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D469(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d469.weight_1",
            forge.Parameter(*(32, 128, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d469.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D470(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d470.weight_1",
            forge.Parameter(*(128, 96, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d470.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D471(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d471.weight_1",
            forge.Parameter(*(128, 160, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d471.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D472(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d472.weight_1",
            forge.Parameter(*(128, 224, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d472.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D473(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d473.weight_1",
            forge.Parameter(*(128, 288, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d473.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D474(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d474.weight_1",
            forge.Parameter(*(128, 320, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d474.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D475(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d475.weight_1",
            forge.Parameter(*(128, 352, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d475.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D476(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d476.weight_1",
            forge.Parameter(*(128, 416, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d476.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D477(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d477.weight_1",
            forge.Parameter(*(128, 480, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d477.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D478(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d478.weight_1",
            forge.Parameter(*(128, 544, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d478.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D479(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d479.weight_1",
            forge.Parameter(*(128, 608, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d479.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D480(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d480.weight_1",
            forge.Parameter(*(128, 640, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d480.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D481(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d481.weight_1",
            forge.Parameter(*(128, 672, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d481.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D482(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d482.weight_1",
            forge.Parameter(*(128, 704, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d482.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D483(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d483.weight_1",
            forge.Parameter(*(128, 736, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d483.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D484(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d484.weight_1",
            forge.Parameter(*(128, 800, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d484.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D485(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d485.weight_1",
            forge.Parameter(*(128, 832, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d485.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D486(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d486.weight_1",
            forge.Parameter(*(128, 864, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d486.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D487(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d487.weight_1",
            forge.Parameter(*(128, 896, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d487.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D488(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d488.weight_1",
            forge.Parameter(*(128, 928, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d488.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D489(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d489.weight_1",
            forge.Parameter(*(128, 960, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d489.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D490(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d490.weight_1",
            forge.Parameter(*(128, 992, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d490.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D491(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d491.weight_1",
            forge.Parameter(*(64, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d491.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=64,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D492(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d492.weight_1",
            forge.Parameter(*(128, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d492.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=128,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D493(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d493.weight_1",
            forge.Parameter(*(256, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d493.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=256,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D494(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d494.weight_1",
            forge.Parameter(*(512, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d494.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=512,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D495(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[1, 1],
            padding=[0, 3, 0, 3],
            dilation=1,
            groups=2048,
            channel_last=0,
        )
        return conv2d_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Conv2D0,
        [((1, 3, 480, 480), torch.float32)],
        {
            "model_names": ["TranslatedLayer"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D1,
        [((1, 16, 240, 240), torch.float32)],
        {
            "model_names": ["TranslatedLayer"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "16",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D2,
        [((1, 16, 240, 240), torch.float32)],
        {
            "model_names": ["TranslatedLayer"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D3,
        [((1, 32, 240, 240), torch.float32)],
        {
            "model_names": ["TranslatedLayer"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "32",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D4,
        [((1, 32, 120, 120), torch.float32)],
        {
            "model_names": ["TranslatedLayer"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D5,
        [((1, 48, 120, 120), torch.float32)],
        {
            "model_names": ["TranslatedLayer"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "48",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D6,
        [((1, 48, 120, 120), torch.float32)],
        {
            "model_names": ["TranslatedLayer"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "48",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D7,
        [((1, 48, 120, 120), torch.float32)],
        {
            "model_names": ["TranslatedLayer"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D8,
        [((1, 48, 60, 60), torch.float32)],
        {
            "model_names": ["TranslatedLayer"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D9,
        [((1, 96, 60, 60), torch.float32)],
        {
            "model_names": ["TranslatedLayer"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "96",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D10,
        [((1, 96, 60, 60), torch.float32)],
        {
            "model_names": ["TranslatedLayer"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "96",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D11,
        [((1, 96, 60, 60), torch.float32)],
        {
            "model_names": ["TranslatedLayer"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D12,
        [((1, 96, 30, 30), torch.float32)],
        {
            "model_names": ["TranslatedLayer"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D13,
        [((1, 192, 30, 30), torch.float32)],
        {
            "model_names": ["TranslatedLayer"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "192",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D14,
        [((1, 192, 30, 30), torch.float32)],
        {
            "model_names": ["TranslatedLayer"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "192",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D15,
        [((1, 192, 30, 30), torch.float32)],
        {
            "model_names": ["TranslatedLayer"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D16,
        [((1, 192, 1, 1), torch.float32)],
        {
            "model_names": ["TranslatedLayer", "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D17,
        [((1, 48, 1, 1), torch.float32)],
        {
            "model_names": ["TranslatedLayer", "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D18,
        [((1, 192, 15, 15), torch.float32)],
        {
            "model_names": ["TranslatedLayer"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D19,
        [((1, 384, 15, 15), torch.float32)],
        {
            "model_names": ["TranslatedLayer"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "384",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D20,
        [((1, 384, 1, 1), torch.float32)],
        {
            "model_names": ["TranslatedLayer", "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D21,
        [((1, 96, 1, 1), torch.float32)],
        {
            "model_names": ["TranslatedLayer", "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D22,
        [((1, 384, 15, 15), torch.float32)],
        {
            "model_names": ["TranslatedLayer"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D23,
        [((1, 384, 15, 15), torch.float32)],
        {
            "model_names": ["TranslatedLayer"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D24,
        [((1, 360, 15, 15), torch.float32)],
        {
            "model_names": ["TranslatedLayer"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D25,
        [((1, 96, 1, 1), torch.float32)],
        {
            "model_names": ["TranslatedLayer", "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D26,
        [((1, 24, 1, 1), torch.float32)],
        {
            "model_names": ["TranslatedLayer", "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D27,
        [((1, 96, 15, 15), torch.float32)],
        {
            "model_names": ["TranslatedLayer"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D28,
        [((1, 24, 1, 1), torch.float32)],
        {
            "model_names": [
                "TranslatedLayer",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D29,
        [((1, 6, 1, 1), torch.float32)],
        {
            "model_names": [
                "TranslatedLayer",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D30,
        [((1, 192, 30, 30), torch.float32)],
        {
            "model_names": ["TranslatedLayer"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D31,
        [((1, 42, 30, 30), torch.float32)],
        {
            "model_names": ["TranslatedLayer"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D27,
        [((1, 96, 30, 30), torch.float32)],
        {
            "model_names": ["TranslatedLayer"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D32,
        [((1, 96, 60, 60), torch.float32)],
        {
            "model_names": ["TranslatedLayer"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D33,
        [((1, 18, 60, 60), torch.float32)],
        {
            "model_names": ["TranslatedLayer"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D27,
        [((1, 96, 60, 60), torch.float32)],
        {
            "model_names": ["TranslatedLayer"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D34,
        [((1, 48, 120, 120), torch.float32)],
        {
            "model_names": ["TranslatedLayer"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D35,
        [((1, 12, 120, 120), torch.float32)],
        {
            "model_names": ["TranslatedLayer"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D27,
        [((1, 96, 120, 120), torch.float32)],
        {
            "model_names": ["TranslatedLayer"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D36,
        [((1, 3, 427, 640), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[3, 3, 3, 3]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D37,
        [((1, 64, 107, 160), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D38,
        [((1, 64, 107, 160), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D39,
        [((1, 64, 107, 160), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D40,
        [((1, 256, 107, 160), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D41,
        [((1, 256, 107, 160), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D42,
        [((1, 128, 107, 160), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D43,
        [((1, 128, 54, 80), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D44,
        [((1, 256, 107, 160), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D45,
        [((1, 512, 54, 80), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D46,
        [((1, 128, 54, 80), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D47,
        [((1, 512, 54, 80), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D48,
        [((1, 256, 54, 80), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D49,
        [((1, 256, 27, 40), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D50,
        [((1, 512, 54, 80), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D51,
        [((1, 1024, 27, 40), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D52,
        [((1, 256, 27, 40), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D53,
        [((1, 1024, 27, 40), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D54,
        [((1, 512, 27, 40), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D55,
        [((1, 512, 14, 20), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D56,
        [((1, 1024, 27, 40), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D57,
        [((1, 2048, 14, 20), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D58,
        [((1, 512, 14, 20), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D59,
        [((1, 2048, 14, 20), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D60,
        [((1, 3, 224, 224), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla46x_c_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "onnx_dla_dla34_visual_bb_torchvision",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[3, 3, 3, 3]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D61,
        [((1, 16, 224, 224), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla46x_c_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "onnx_dla_dla34_visual_bb_torchvision",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D62,
        [((1, 16, 224, 224), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla46x_c_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "onnx_dla_dla34_visual_bb_torchvision",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D63,
        [((1, 32, 112, 112), torch.float32)],
        {
            "model_names": ["onnx_dla_dla102x2_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D64,
        [((1, 256, 112, 112), torch.float32)],
        {
            "model_names": ["onnx_dla_dla102x2_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "64",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D41,
        [((1, 256, 56, 56), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D65,
        [((1, 32, 56, 56), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D66,
        [((1, 128, 56, 56), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D67,
        [((1, 256, 56, 56), torch.float32)],
        {
            "model_names": ["onnx_dla_dla102x2_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "64",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D43,
        [((1, 128, 56, 56), torch.float32)],
        {
            "model_names": ["onnx_dla_dla102x2_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D68,
        [((1, 512, 56, 56), torch.float32)],
        {
            "model_names": ["onnx_dla_dla102x2_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "64",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D47,
        [((1, 512, 28, 28), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D66,
        [((1, 128, 28, 28), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D69,
        [((1, 128, 28, 28), torch.float32)],
        {
            "model_names": ["pd_resnet_18_img_cls_paddlemodels", "pd_resnet_34_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D70,
        [((1, 256, 28, 28), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D71,
        [((1, 512, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_dla_dla102x2_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "64",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D72,
        [((1, 768, 28, 28), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla169_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D73,
        [((1, 1152, 28, 28), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla169_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D49,
        [((1, 256, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_dla_dla102x2_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D74,
        [((1, 1024, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_dla_dla102x2_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "64",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D53,
        [((1, 1024, 14, 14), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D70,
        [((1, 256, 14, 14), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D44,
        [((1, 256, 14, 14), torch.float32)],
        {
            "model_names": ["pd_resnet_18_img_cls_paddlemodels", "pd_resnet_34_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D75,
        [((1, 512, 14, 14), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D76,
        [((1, 1024, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_dla_dla102x2_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "64",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D77,
        [((1, 1536, 14, 14), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D57,
        [((1, 2048, 14, 14), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla169_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D78,
        [((1, 2816, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_dla_dla102x2_visual_bb_torchvision", "onnx_dla_dla102x_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D55,
        [((1, 512, 14, 14), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D79,
        [((1, 2048, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_dla_dla102x2_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "64",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D80,
        [((1, 2048, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_dla_dla102x2_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D75,
        [((1, 512, 7, 7), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D81,
        [((1, 1024, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_dla_dla102x2_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D82,
        [((1, 2048, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_dla_dla102x2_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "64",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D83,
        [((1, 2560, 7, 7), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D84,
        [((1, 1024, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D85,
        [((1, 32, 112, 112), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla46x_c_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D86,
        [((1, 64, 112, 112), torch.float32)],
        {
            "model_names": ["onnx_dla_dla46x_c_visual_bb_torchvision", "onnx_dla_dla60x_c_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "32",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D37,
        [((1, 64, 56, 56), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla46x_c_visual_bb_torchvision",
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D85,
        [((1, 32, 56, 56), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla46x_c_visual_bb_torchvision",
                "onnx_dla_dla34_visual_bb_torchvision",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D87,
        [((1, 64, 56, 56), torch.float32)],
        {
            "model_names": ["onnx_dla_dla46x_c_visual_bb_torchvision", "onnx_dla_dla60x_c_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "32",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D86,
        [((1, 64, 56, 56), torch.float32)],
        {
            "model_names": ["onnx_dla_dla46x_c_visual_bb_torchvision", "onnx_dla_dla60x_c_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "32",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D88,
        [((1, 128, 56, 56), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla46x_c_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla34_visual_bb_torchvision",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D37,
        [((1, 64, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_dla_dla46x_c_visual_bb_torchvision", "onnx_dla_dla60x_c_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D87,
        [((1, 64, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_dla_dla46x_c_visual_bb_torchvision", "onnx_dla_dla60x_c_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "32",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D88,
        [((1, 128, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_dla_dla46x_c_visual_bb_torchvision", "onnx_dla_dla60x_c_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D40,
        [((1, 256, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_dla_dla46x_c_visual_bb_torchvision", "onnx_dla_dla60x_c_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D89,
        [((1, 64, 28, 28), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla46x_c_visual_bb_torchvision",
                "onnx_dla_dla34_visual_bb_torchvision",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D90,
        [((1, 128, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_dla_dla46x_c_visual_bb_torchvision", "onnx_dla_dla60x_c_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "32",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D91,
        [((1, 128, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_dla_dla46x_c_visual_bb_torchvision", "onnx_dla_dla60x_c_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D89,
        [((1, 64, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_dla_dla46x_c_visual_bb_torchvision", "onnx_dla_dla60x_c_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D92,
        [((1, 128, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_dla_dla46x_c_visual_bb_torchvision", "onnx_dla_dla60x_c_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "32",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D41,
        [((1, 256, 14, 14), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla46x_c_visual_bb_torchvision",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D93,
        [((1, 448, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_dla_dla46x_c_visual_bb_torchvision", "pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D66,
        [((1, 128, 14, 14), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla46x_c_visual_bb_torchvision",
                "onnx_dla_dla34_visual_bb_torchvision",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D94,
        [((1, 256, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_dla_dla46x_c_visual_bb_torchvision", "onnx_dla_dla60x_c_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "32",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D95,
        [((1, 256, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_dla_dla46x_c_visual_bb_torchvision", "onnx_dla_dla60x_c_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D66,
        [((1, 128, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_dla_dla46x_c_visual_bb_torchvision", "onnx_dla_dla60x_c_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D96,
        [((1, 256, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_dla_dla46x_c_visual_bb_torchvision", "onnx_dla_dla60x_c_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "32",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D97,
        [((1, 640, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_dla_dla46x_c_visual_bb_torchvision", "onnx_dla_dla60x_c_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D98,
        [((1, 256, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_dla_dla46x_c_visual_bb_torchvision", "onnx_dla_dla60x_c_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D99,
        [((1, 3, 288, 288), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D100,
        [((1, 40, 144, 144), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "40",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D101,
        [((1, 40, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D102,
        [((1, 10, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D103,
        [((1, 40, 144, 144), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D104,
        [((1, 24, 144, 144), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "24",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D105,
        [((1, 24, 144, 144), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D106,
        [((1, 24, 144, 144), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D107,
        [((1, 144, 144, 144), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "144",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D108,
        [((1, 144, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D109,
        [((1, 6, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D110,
        [((1, 144, 72, 72), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D111,
        [((1, 32, 72, 72), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D112,
        [((1, 192, 72, 72), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "192",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D113,
        [((1, 192, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D114,
        [((1, 8, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D115,
        [((1, 192, 72, 72), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D14,
        [((1, 192, 72, 72), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "192",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D16,
        [((1, 192, 36, 36), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D116,
        [((1, 48, 36, 36), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D117,
        [((1, 288, 36, 36), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "288",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D118,
        [((1, 288, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D119,
        [((1, 12, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D120,
        [((1, 288, 36, 36), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D121,
        [((1, 288, 36, 36), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "288",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D122,
        [((1, 288, 18, 18), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D123,
        [((1, 96, 18, 18), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D124,
        [((1, 576, 18, 18), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "576",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D125,
        [((1, 576, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D126,
        [((1, 24, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D127,
        [((1, 576, 18, 18), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D128,
        [((1, 576, 18, 18), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "576",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D129,
        [((1, 576, 18, 18), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D130,
        [((1, 136, 18, 18), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D131,
        [((1, 816, 18, 18), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "816",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D132,
        [((1, 816, 18, 18), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "816",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D133,
        [((1, 816, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D134,
        [((1, 34, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D135,
        [((1, 816, 18, 18), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D136,
        [((1, 816, 9, 9), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D137,
        [((1, 232, 9, 9), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D138,
        [((1, 1392, 9, 9), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "1392",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D139,
        [((1, 1392, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D140,
        [((1, 58, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D141,
        [((1, 1392, 9, 9), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D142,
        [((1, 1392, 9, 9), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1392",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D143,
        [((1, 1392, 9, 9), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D144,
        [((1, 384, 9, 9), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D145,
        [((1, 2304, 9, 9), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "2304",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D146,
        [((1, 2304, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D147,
        [((1, 96, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D148,
        [((1, 2304, 9, 9), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D149,
        [((1, 384, 9, 9), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D150,
        [((1, 3, 512, 512), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[4, 4]",
                "padding": "[3, 3, 3, 3]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D151,
        [((1, 64, 128, 128), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[8, 8]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D152,
        [((1, 256, 128, 128), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "256",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D153,
        [((1, 64, 128, 128), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D154,
        [((1, 128, 64, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[4, 4]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D155,
        [((1, 512, 64, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "512",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D156,
        [((1, 128, 64, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D157,
        [((1, 320, 32, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D158,
        [((1, 1280, 32, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1280",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D159,
        [((1, 320, 32, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D160,
        [((1, 2048, 16, 16), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "2048",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D161,
        [((1, 3072, 128, 128), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D162,
        [((1, 768, 128, 128), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D163,
        [((1, 3, 224, 224), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet27s_obj_det_osmr", "onnx_vovnet_vovnet_v1_57_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D38,
        [((1, 64, 112, 112), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet27s_obj_det_osmr", "onnx_vovnet_vovnet_v1_57_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D164,
        [((1, 64, 112, 112), torch.float32)],
        {
            "model_names": ["onnx_dla_dla60_visual_bb_torchvision", "onnx_dla_dla169_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D153,
        [((1, 64, 112, 112), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet27s_obj_det_osmr", "onnx_vovnet_vovnet_v1_57_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D165,
        [((1, 128, 56, 56), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet27s_obj_det_osmr"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D38,
        [((1, 64, 56, 56), torch.float32)],
        {
            "model_names": [
                "onnx_vovnet_vovnet27s_obj_det_osmr",
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_dla_dla60_visual_bb_torchvision",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_dla_dla169_visual_bb_torchvision",
                "pd_resnet_18_img_cls_paddlemodels",
                "onnx_dla_dla34_visual_bb_torchvision",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D93,
        [((1, 448, 56, 56), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet27s_obj_det_osmr"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D166,
        [((1, 128, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet27s_obj_det_osmr"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D167,
        [((1, 80, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet27s_obj_det_osmr"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D168,
        [((1, 528, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet27s_obj_det_osmr"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D169,
        [((1, 256, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet27s_obj_det_osmr"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D170,
        [((1, 96, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet27s_obj_det_osmr"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D171,
        [((1, 736, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet27s_obj_det_osmr"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D172,
        [((1, 384, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet27s_obj_det_osmr"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D173,
        [((1, 112, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet27s_obj_det_osmr"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D174,
        [((1, 944, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet27s_obj_det_osmr"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D0,
        [((1, 3, 32, 100), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D1,
        [((1, 16, 16, 50), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "16",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D2,
        [((1, 16, 16, 50), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D175,
        [((1, 32, 16, 50), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "32",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D85,
        [((1, 32, 16, 50), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D176,
        [((1, 64, 16, 50), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "64",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D177,
        [((1, 64, 16, 50), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "64",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D37,
        [((1, 64, 16, 50), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D89,
        [((1, 64, 8, 50), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D178,
        [((1, 128, 8, 50), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "128",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D179,
        [((1, 128, 8, 50), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "128",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D91,
        [((1, 128, 8, 50), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D180,
        [((1, 128, 8, 25), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D181,
        [((1, 240, 8, 25), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "240",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D182,
        [((1, 240, 8, 25), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 1]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "240",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D183,
        [((1, 240, 8, 25), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D184,
        [((1, 240, 1, 1), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D185,
        [((1, 60, 1, 1), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D186,
        [((1, 240, 4, 25), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D187,
        [((1, 480, 4, 25), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "480",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D188,
        [((1, 480, 4, 25), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 1]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "480",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D189,
        [((1, 480, 1, 1), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D190,
        [((1, 120, 1, 1), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D191,
        [((1, 480, 4, 25), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D191,
        [((1, 480, 2, 25), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D187,
        [((1, 480, 2, 25), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "480",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D192,
        [((1, 480, 1, 12), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 1, 0, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D193,
        [((1, 60, 1, 12), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D190,
        [((1, 120, 1, 12), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D194,
        [((1, 960, 1, 12), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 1, 0, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D36,
        [((1, 3, 224, 224), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_resnet_18_img_cls_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[3, 3, 3, 3]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D39,
        [((1, 64, 56, 56), torch.float32)],
        {
            "model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D40,
        [((1, 256, 56, 56), torch.float32)],
        {
            "model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D42,
        [((1, 128, 56, 56), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_dla_dla60_visual_bb_torchvision",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_dla_dla169_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D46,
        [((1, 128, 56, 56), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet_v1_57_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D43,
        [((1, 128, 28, 28), torch.float32)],
        {
            "model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D44,
        [((1, 256, 56, 56), torch.float32)],
        {
            "model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D45,
        [((1, 512, 28, 28), torch.float32)],
        {
            "model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D46,
        [((1, 128, 28, 28), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_dla_dla60_visual_bb_torchvision",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_dla_dla169_visual_bb_torchvision",
                "pd_resnet_18_img_cls_paddlemodels",
                "onnx_dla_dla34_visual_bb_torchvision",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D48,
        [((1, 256, 28, 28), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_dla_dla60_visual_bb_torchvision",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_dla_dla169_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D49,
        [((1, 256, 14, 14), torch.float32)],
        {
            "model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D50,
        [((1, 512, 28, 28), torch.float32)],
        {
            "model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D51,
        [((1, 1024, 14, 14), torch.float32)],
        {
            "model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D52,
        [((1, 256, 14, 14), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_dla_dla60_visual_bb_torchvision",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_dla_dla169_visual_bb_torchvision",
                "pd_resnet_18_img_cls_paddlemodels",
                "onnx_dla_dla34_visual_bb_torchvision",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D54,
        [((1, 512, 14, 14), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_dla_dla60_visual_bb_torchvision",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_dla_dla169_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D55,
        [((1, 512, 7, 7), torch.float32)],
        {
            "model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D56,
        [((1, 1024, 14, 14), torch.float32)],
        {
            "model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D57,
        [((1, 2048, 7, 7), torch.float32)],
        {
            "model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D58,
        [((1, 512, 7, 7), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_dla_dla60_visual_bb_torchvision",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_dla_dla169_visual_bb_torchvision",
                "pd_resnet_18_img_cls_paddlemodels",
                "onnx_dla_dla34_visual_bb_torchvision",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D195,
        [((1, 80, 3000, 1), torch.float32), ((768, 80, 3, 1), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 0, 1, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D196,
        [((1, 768, 3000, 1), torch.float32), ((768, 768, 3, 1), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 1]",
                "padding": "[1, 0, 1, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D65,
        [((1, 32, 112, 112), torch.float32)],
        {
            "model_names": ["onnx_dla_dla102x_visual_bb_torchvision", "onnx_dla_dla60x_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D90,
        [((1, 128, 112, 112), torch.float32)],
        {
            "model_names": ["onnx_dla_dla102x_visual_bb_torchvision", "onnx_dla_dla60x_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "32",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D91,
        [((1, 128, 56, 56), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D92,
        [((1, 128, 56, 56), torch.float32)],
        {
            "model_names": ["onnx_dla_dla102x_visual_bb_torchvision", "onnx_dla_dla60x_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "32",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D94,
        [((1, 256, 56, 56), torch.float32)],
        {
            "model_names": ["onnx_dla_dla102x_visual_bb_torchvision", "onnx_dla_dla60x_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "32",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D95,
        [((1, 256, 28, 28), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D96,
        [((1, 256, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_dla_dla102x_visual_bb_torchvision", "onnx_dla_dla60x_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "32",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D197,
        [((1, 512, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_dla_dla102x_visual_bb_torchvision", "onnx_dla_dla60x_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "32",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D198,
        [((1, 512, 14, 14), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D199,
        [((1, 512, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_dla_dla102x_visual_bb_torchvision", "onnx_dla_dla60x_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "32",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D200,
        [((1, 1024, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_dla_dla102x_visual_bb_torchvision", "onnx_dla_dla60x_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "32",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D201,
        [((1, 1024, 7, 7), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D202,
        [((1, 1024, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_dla_dla102x_visual_bb_torchvision", "onnx_dla_dla60x_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "32",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D89,
        [((1, 64, 56, 56), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_dla_dla169_visual_bb_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D203,
        [((1, 64, 56, 56), torch.float32)],
        {
            "model_names": ["pd_resnet_18_img_cls_paddlemodels", "pd_resnet_34_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D41,
        [((1, 256, 28, 28), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla34_visual_bb_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D204,
        [((1, 896, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_dla_dla60_visual_bb_torchvision", "onnx_dla_dla60x_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D47,
        [((1, 512, 14, 14), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla34_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D205,
        [((1, 2304, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_dla_dla60_visual_bb_torchvision", "onnx_dla_dla60x_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D53,
        [((1, 1024, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_dla_dla60_visual_bb_torchvision", "onnx_dla_dla169_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D206,
        [((1, 3, 224, 224), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D175,
        [((1, 32, 112, 112), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "32",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D207,
        [((1, 32, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D208,
        [((1, 8, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D209,
        [((1, 32, 112, 112), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D210,
        [((1, 16, 112, 112), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D10,
        [((1, 96, 112, 112), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "96",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D211,
        [((1, 96, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D212,
        [((1, 4, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D25,
        [((1, 96, 56, 56), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D106,
        [((1, 24, 56, 56), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D213,
        [((1, 144, 56, 56), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "144",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D107,
        [((1, 144, 56, 56), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "144",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D214,
        [((1, 144, 56, 56), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D215,
        [((1, 144, 56, 56), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "144",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D216,
        [((1, 144, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D217,
        [((1, 40, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D181,
        [((1, 240, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "240",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D218,
        [((1, 240, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D219,
        [((1, 10, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D220,
        [((1, 240, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D221,
        [((1, 240, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "240",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D222,
        [((1, 240, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D223,
        [((1, 80, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D224,
        [((1, 480, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "480",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D225,
        [((1, 480, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D226,
        [((1, 20, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D227,
        [((1, 480, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D187,
        [((1, 480, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "480",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D228,
        [((1, 480, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D229,
        [((1, 112, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D230,
        [((1, 672, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "672",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D231,
        [((1, 672, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "672",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D232,
        [((1, 672, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D233,
        [((1, 28, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D234,
        [((1, 672, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D235,
        [((1, 672, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D236,
        [((1, 192, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D237,
        [((1, 1152, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "1152",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D238,
        [((1, 1152, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D239,
        [((1, 48, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D240,
        [((1, 1152, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D241,
        [((1, 1152, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1152",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D242,
        [((1, 1152, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D243,
        [((1, 320, 7, 7), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D0,
        [((1, 3, 224, 224), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D1,
        [((1, 16, 112, 112), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "16",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D244,
        [((1, 16, 112, 112), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D245,
        [((1, 8, 112, 112), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D6,
        [((1, 48, 112, 112), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "48",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D5,
        [((1, 48, 112, 112), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "48",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D246,
        [((1, 48, 56, 56), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D210,
        [((1, 16, 56, 56), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D9,
        [((1, 96, 56, 56), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "96",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D10,
        [((1, 96, 56, 56), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "96",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D247,
        [((1, 96, 56, 56), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D247,
        [((1, 96, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D210,
        [((1, 16, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D9,
        [((1, 96, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "96",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D10,
        [((1, 96, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "96",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D248,
        [((1, 96, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D111,
        [((1, 32, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D112,
        [((1, 192, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "192",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D115,
        [((1, 192, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D16,
        [((1, 192, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D116,
        [((1, 48, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D249,
        [((1, 288, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "288",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D121,
        [((1, 288, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "288",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D120,
        [((1, 288, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D250,
        [((1, 288, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D223,
        [((1, 80, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D224,
        [((1, 480, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "480",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D227,
        [((1, 480, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D251,
        [((1, 480, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D252,
        [((1, 160, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D72,
        [((1, 768, 56, 56), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet_v1_57_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D253,
        [((1, 256, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet_v1_57_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D254,
        [((1, 160, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet_v1_57_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D255,
        [((1, 1056, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet_v1_57_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D256,
        [((1, 512, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet_v1_57_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D257,
        [((1, 192, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet_v1_57_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D258,
        [((1, 1472, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet_v1_57_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D259,
        [((1, 768, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet_v1_57_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D260,
        [((1, 1728, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet_v1_57_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D261,
        [((1, 768, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet_v1_57_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D262,
        [((1, 224, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet_v1_57_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D263,
        [((1, 1888, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet_v1_57_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D264,
        [((1, 1024, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet_v1_57_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D265,
        [((1, 2144, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet_v1_57_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D0,
        [((1, 3, 448, 448), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D1,
        [((1, 16, 224, 224), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "16",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D2,
        [((1, 16, 224, 224), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D3,
        [((1, 32, 224, 224), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "32",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D4,
        [((1, 32, 112, 112), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D7,
        [((1, 48, 112, 112), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D8,
        [((1, 48, 56, 56), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D11,
        [((1, 96, 56, 56), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D12,
        [((1, 96, 28, 28), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D13,
        [((1, 192, 28, 28), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "192",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D14,
        [((1, 192, 28, 28), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "192",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D15,
        [((1, 192, 28, 28), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D18,
        [((1, 192, 14, 14), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D19,
        [((1, 384, 14, 14), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "384",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D22,
        [((1, 384, 14, 14), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D23,
        [((1, 384, 14, 14), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D24,
        [((1, 360, 14, 14), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D27,
        [((1, 96, 14, 14), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D30,
        [((1, 192, 28, 28), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D31,
        [((1, 42, 28, 28), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D27,
        [((1, 96, 28, 28), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D32,
        [((1, 96, 56, 56), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D33,
        [((1, 18, 56, 56), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D27,
        [((1, 96, 56, 56), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D34,
        [((1, 48, 112, 112), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D35,
        [((1, 12, 112, 112), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D27,
        [((1, 96, 112, 112), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D195,
        [((1, 80, 3000, 1), torch.float32), ((384, 80, 3, 1), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 0, 1, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D196,
        [((1, 384, 3000, 1), torch.float32), ((384, 384, 3, 1), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 1]",
                "padding": "[1, 0, 1, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D266,
        [((1, 224, 224, 3), torch.float32), ((64, 3, 7, 7), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[3, 3, 3, 3]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "1",
            },
        },
    ),
    (
        Conv2D267,
        [((1, 56, 55, 64), torch.float32), ((64, 64, 1, 1), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "1",
            },
        },
    ),
    (
        Conv2D268,
        [((1, 56, 55, 64), torch.float32), ((64, 64, 3, 3), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "1",
            },
        },
    ),
    (
        Conv2D267,
        [((1, 56, 55, 64), torch.float32), ((256, 64, 1, 1), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "1",
            },
        },
    ),
    (
        Conv2D267,
        [((1, 56, 55, 256), torch.float32), ((64, 256, 1, 1), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "1",
            },
        },
    ),
    (
        Conv2D267,
        [((1, 56, 55, 256), torch.float32), ((128, 256, 1, 1), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "1",
            },
        },
    ),
    (
        Conv2D269,
        [((1, 56, 55, 128), torch.float32), ((128, 128, 3, 3), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "1",
            },
        },
    ),
    (
        Conv2D267,
        [((1, 28, 28, 128), torch.float32), ((512, 128, 1, 1), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "1",
            },
        },
    ),
    (
        Conv2D270,
        [((1, 56, 55, 256), torch.float32), ((512, 256, 1, 1), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "1",
            },
        },
    ),
    (
        Conv2D267,
        [((1, 28, 28, 512), torch.float32), ((128, 512, 1, 1), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "1",
            },
        },
    ),
    (
        Conv2D268,
        [((1, 28, 28, 128), torch.float32), ((128, 128, 3, 3), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "1",
            },
        },
    ),
    (
        Conv2D267,
        [((1, 28, 28, 512), torch.float32), ((256, 512, 1, 1), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "1",
            },
        },
    ),
    (
        Conv2D269,
        [((1, 28, 28, 256), torch.float32), ((256, 256, 3, 3), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "1",
            },
        },
    ),
    (
        Conv2D267,
        [((1, 14, 14, 256), torch.float32), ((1024, 256, 1, 1), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "1",
            },
        },
    ),
    (
        Conv2D270,
        [((1, 28, 28, 512), torch.float32), ((1024, 512, 1, 1), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "1",
            },
        },
    ),
    (
        Conv2D267,
        [((1, 14, 14, 1024), torch.float32), ((256, 1024, 1, 1), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "1",
            },
        },
    ),
    (
        Conv2D268,
        [((1, 14, 14, 256), torch.float32), ((256, 256, 3, 3), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "1",
            },
        },
    ),
    (
        Conv2D267,
        [((1, 14, 14, 1024), torch.float32), ((512, 1024, 1, 1), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "1",
            },
        },
    ),
    (
        Conv2D269,
        [((1, 14, 14, 512), torch.float32), ((512, 512, 3, 3), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "1",
            },
        },
    ),
    (
        Conv2D267,
        [((1, 7, 7, 512), torch.float32), ((2048, 512, 1, 1), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "1",
            },
        },
    ),
    (
        Conv2D270,
        [((1, 14, 14, 1024), torch.float32), ((2048, 1024, 1, 1), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "1",
            },
        },
    ),
    (
        Conv2D267,
        [((1, 7, 7, 2048), torch.float32), ((512, 2048, 1, 1), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "1",
            },
        },
    ),
    (
        Conv2D268,
        [((1, 7, 7, 512), torch.float32), ((512, 512, 3, 3), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "1",
            },
        },
    ),
    (
        Conv2D271,
        [((1, 256, 107, 160), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D272,
        [((1, 512, 54, 80), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D273,
        [((1, 1024, 27, 40), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D95,
        [((1, 256, 14, 20), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D274,
        [((100, 264, 14, 20), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D275,
        [((100, 264, 14, 20), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D165,
        [((100, 128, 27, 40), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D276,
        [((100, 64, 54, 80), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D277,
        [((100, 32, 107, 160), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D278,
        [((100, 16, 107, 160), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D279,
        [((1, 384, 56, 56), torch.float32)],
        {
            "model_names": ["onnx_dla_dla169_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D280,
        [((1, 2560, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_dla_dla169_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D281,
        [((1, 3328, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_dla_dla169_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D206,
        [((1, 3, 240, 240), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D175,
        [((1, 32, 120, 120), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "32",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D209,
        [((1, 32, 120, 120), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D1,
        [((1, 16, 120, 120), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "16",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D282,
        [((1, 16, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D283,
        [((1, 4, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D284,
        [((1, 16, 120, 120), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D210,
        [((1, 16, 120, 120), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D10,
        [((1, 96, 120, 120), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "96",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D25,
        [((1, 96, 60, 60), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D106,
        [((1, 24, 60, 60), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D213,
        [((1, 144, 60, 60), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "144",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D214,
        [((1, 144, 60, 60), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D215,
        [((1, 144, 60, 60), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "144",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D216,
        [((1, 144, 30, 30), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D217,
        [((1, 40, 30, 30), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D181,
        [((1, 240, 30, 30), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "240",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D220,
        [((1, 240, 30, 30), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D221,
        [((1, 240, 30, 30), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "240",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D222,
        [((1, 240, 15, 15), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D223,
        [((1, 80, 15, 15), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D224,
        [((1, 480, 15, 15), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "480",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D227,
        [((1, 480, 15, 15), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D187,
        [((1, 480, 15, 15), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "480",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D228,
        [((1, 480, 15, 15), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D229,
        [((1, 112, 15, 15), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D230,
        [((1, 672, 15, 15), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "672",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D231,
        [((1, 672, 15, 15), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "672",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D234,
        [((1, 672, 15, 15), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D235,
        [((1, 672, 8, 8), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D236,
        [((1, 192, 8, 8), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D237,
        [((1, 1152, 8, 8), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "1152",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D240,
        [((1, 1152, 8, 8), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D241,
        [((1, 1152, 8, 8), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1152",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D242,
        [((1, 1152, 8, 8), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D285,
        [((1, 320, 8, 8), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D286,
        [((1, 1920, 8, 8), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1920",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D287,
        [((1, 1920, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D288,
        [((1, 80, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D289,
        [((1, 1920, 8, 8), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D243,
        [((1, 320, 8, 8), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D290,
        [((1, 3, 320, 320), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D5,
        [((1, 48, 160, 160), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "48",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D34,
        [((1, 48, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D291,
        [((1, 12, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D292,
        [((1, 48, 160, 160), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D104,
        [((1, 24, 160, 160), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "24",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D105,
        [((1, 24, 160, 160), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D106,
        [((1, 24, 160, 160), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D107,
        [((1, 144, 160, 160), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "144",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D110,
        [((1, 144, 80, 80), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D111,
        [((1, 32, 80, 80), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D112,
        [((1, 192, 80, 80), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "192",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D115,
        [((1, 192, 80, 80), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D14,
        [((1, 192, 80, 80), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "192",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D293,
        [((1, 192, 40, 40), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D294,
        [((1, 56, 40, 40), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D295,
        [((1, 336, 40, 40), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "336",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D296,
        [((1, 336, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D297,
        [((1, 14, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D298,
        [((1, 336, 40, 40), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D299,
        [((1, 336, 40, 40), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "336",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D300,
        [((1, 336, 20, 20), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D229,
        [((1, 112, 20, 20), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D301,
        [((1, 672, 20, 20), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "672",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D234,
        [((1, 672, 20, 20), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D230,
        [((1, 672, 20, 20), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "672",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D302,
        [((1, 672, 20, 20), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D303,
        [((1, 160, 20, 20), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D304,
        [((1, 960, 20, 20), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "960",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D305,
        [((1, 960, 20, 20), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "960",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D306,
        [((1, 960, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D307,
        [((1, 40, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D308,
        [((1, 960, 20, 20), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D309,
        [((1, 960, 10, 10), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D310,
        [((1, 272, 10, 10), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D311,
        [((1, 1632, 10, 10), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "1632",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D312,
        [((1, 1632, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D313,
        [((1, 68, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D314,
        [((1, 1632, 10, 10), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D315,
        [((1, 1632, 10, 10), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1632",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D316,
        [((1, 1632, 10, 10), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D317,
        [((1, 448, 10, 10), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D318,
        [((1, 2688, 10, 10), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "2688",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D319,
        [((1, 2688, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D320,
        [((1, 112, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D321,
        [((1, 2688, 10, 10), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D322,
        [((1, 448, 10, 10), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D110,
        [((1, 144, 28, 28), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D111,
        [((1, 32, 28, 28), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D112,
        [((1, 192, 28, 28), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "192",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D323,
        [((1, 192, 28, 28), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "192",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D115,
        [((1, 192, 28, 28), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D324,
        [((1, 192, 14, 14), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D325,
        [((1, 64, 14, 14), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D326,
        [((1, 384, 14, 14), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "384",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D327,
        [((1, 384, 14, 14), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D20,
        [((1, 384, 14, 14), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D123,
        [((1, 96, 14, 14), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D124,
        [((1, 576, 14, 14), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "576",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D328,
        [((1, 576, 14, 14), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "576",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D127,
        [((1, 576, 14, 14), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D329,
        [((1, 576, 7, 7), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D303,
        [((1, 160, 7, 7), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D330,
        [((1, 960, 7, 7), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "960",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D308,
        [((1, 960, 7, 7), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D331,
        [((1, 960, 7, 7), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D332,
        [((1, 3, 512, 512), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[4, 4]",
                "padding": "[3, 3, 3, 3]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D333,
        [((1, 32, 128, 128), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[8, 8]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D178,
        [((1, 128, 128, 128), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "128",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D334,
        [((1, 32, 128, 128), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D335,
        [((1, 64, 64, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[4, 4]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D152,
        [((1, 256, 64, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "256",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D336,
        [((1, 64, 64, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D337,
        [((1, 160, 32, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D338,
        [((1, 640, 32, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "640",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D339,
        [((1, 160, 32, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D340,
        [((1, 1024, 16, 16), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1024",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D341,
        [((1, 3, 224, 224), torch.float32)],
        {
            "model_names": ["onnx_vit_base_google_vit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {
                "stride": "[16, 16]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D0,
        [((1, 3, 640, 640), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D62,
        [((1, 16, 320, 320), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D342,
        [((1, 32, 160, 160), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D61,
        [((1, 16, 160, 160), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D343,
        [((1, 48, 160, 160), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D334,
        [((1, 32, 160, 160), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D37,
        [((1, 64, 80, 80), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D344,
        [((1, 32, 80, 80), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D88,
        [((1, 128, 80, 80), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D153,
        [((1, 64, 80, 80), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D91,
        [((1, 128, 40, 40), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D38,
        [((1, 64, 40, 40), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D41,
        [((1, 256, 40, 40), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D345,
        [((1, 128, 40, 40), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D95,
        [((1, 256, 20, 20), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D46,
        [((1, 128, 20, 20), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D346,
        [((1, 384, 20, 20), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D41,
        [((1, 256, 20, 20), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D47,
        [((1, 512, 20, 20), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D279,
        [((1, 384, 40, 40), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D347,
        [((1, 192, 40, 40), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D324,
        [((1, 192, 80, 80), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D348,
        [((1, 96, 80, 80), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D38,
        [((1, 64, 80, 80), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D164,
        [((1, 64, 80, 80), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D349,
        [((1, 64, 80, 80), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D167,
        [((1, 80, 80, 80), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D350,
        [((1, 80, 80, 80), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D165,
        [((1, 128, 40, 40), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D37,
        [((1, 64, 40, 40), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D166,
        [((1, 128, 40, 40), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D167,
        [((1, 80, 40, 40), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D350,
        [((1, 80, 40, 40), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D42,
        [((1, 128, 40, 40), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D351,
        [((1, 256, 20, 20), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D38,
        [((1, 64, 20, 20), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D37,
        [((1, 64, 20, 20), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D352,
        [((1, 256, 20, 20), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D167,
        [((1, 80, 20, 20), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D350,
        [((1, 80, 20, 20), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D353,
        [((1, 16, 4, 8400), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D354,
        [((1, 3, 32, 100), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D355,
        [((1, 8, 16, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D356,
        [((1, 8, 16, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "8",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D357,
        [((1, 8, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D358,
        [((1, 2, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D359,
        [((1, 8, 16, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D360,
        [((1, 40, 16, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "40",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D361,
        [((1, 40, 8, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D362,
        [((1, 16, 8, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D5,
        [((1, 48, 8, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "48",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D246,
        [((1, 48, 8, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D363,
        [((1, 48, 8, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 1]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "48",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D292,
        [((1, 48, 4, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D364,
        [((1, 24, 4, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D365,
        [((1, 120, 4, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "120",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D366,
        [((1, 120, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D367,
        [((1, 30, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D368,
        [((1, 120, 4, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D369,
        [((1, 24, 4, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D370,
        [((1, 64, 4, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "64",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D371,
        [((1, 64, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D372,
        [((1, 16, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D373,
        [((1, 64, 4, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D374,
        [((1, 24, 4, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D375,
        [((1, 72, 4, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "72",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D376,
        [((1, 72, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D377,
        [((1, 18, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D378,
        [((1, 72, 4, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D106,
        [((1, 24, 4, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D379,
        [((1, 144, 4, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 1]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "144",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D380,
        [((1, 144, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D381,
        [((1, 36, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D382,
        [((1, 144, 2, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D116,
        [((1, 48, 2, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D117,
        [((1, 288, 2, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "288",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D383,
        [((1, 288, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D384,
        [((1, 72, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D120,
        [((1, 288, 2, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D153,
        [((1, 64, 56, 56), torch.float32)],
        {
            "model_names": [
                "pd_resnet_18_img_cls_paddlemodels",
                "onnx_dla_dla34_visual_bb_torchvision",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D345,
        [((1, 128, 28, 28), torch.float32)],
        {
            "model_names": [
                "pd_resnet_18_img_cls_paddlemodels",
                "onnx_dla_dla34_visual_bb_torchvision",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D385,
        [((1, 256, 14, 14), torch.float32)],
        {
            "model_names": [
                "pd_resnet_18_img_cls_paddlemodels",
                "onnx_dla_dla34_visual_bb_torchvision",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D195,
        [((1, 128, 3000, 1), torch.float32), ((1280, 128, 3, 1), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf", "onnx_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 0, 1, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D196,
        [((1, 1280, 3000, 1), torch.float32), ((1280, 1280, 3, 1), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf", "onnx_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 1]",
                "padding": "[1, 0, 1, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D334,
        [((1, 32, 112, 112), torch.float32)],
        {
            "model_names": ["onnx_dla_dla34_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D93,
        [((1, 448, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_dla_dla34_visual_bb_torchvision", "pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D204,
        [((1, 896, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_dla_dla34_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D70,
        [((1, 256, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_dla_dla34_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D386,
        [((1, 1280, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_dla_dla34_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D387,
        [((1, 512, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_dla_dla34_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D279,
        [((1, 384, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_dla_dla60x_c_visual_bb_torchvision", "pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D388,
        [((1, 576, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_dla_dla60x_c_visual_bb_torchvision", "pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D206,
        [((1, 3, 256, 256), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D175,
        [((1, 32, 128, 128), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "32",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D209,
        [((1, 32, 128, 128), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D1,
        [((1, 16, 128, 128), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "16",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D284,
        [((1, 16, 128, 128), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D210,
        [((1, 16, 128, 128), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D10,
        [((1, 96, 128, 128), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "96",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D25,
        [((1, 96, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D106,
        [((1, 24, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D213,
        [((1, 144, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "144",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D214,
        [((1, 144, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D215,
        [((1, 144, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "144",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D382,
        [((1, 144, 32, 32), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D116,
        [((1, 48, 32, 32), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D117,
        [((1, 288, 32, 32), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "288",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D120,
        [((1, 288, 32, 32), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D121,
        [((1, 288, 32, 32), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "288",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D389,
        [((1, 288, 16, 16), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D390,
        [((1, 88, 16, 16), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D391,
        [((1, 528, 16, 16), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "528",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D392,
        [((1, 528, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D393,
        [((1, 22, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D394,
        [((1, 528, 16, 16), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D395,
        [((1, 528, 16, 16), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "528",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D396,
        [((1, 528, 16, 16), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D397,
        [((1, 120, 16, 16), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D398,
        [((1, 720, 16, 16), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "720",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D399,
        [((1, 720, 16, 16), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "720",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D400,
        [((1, 720, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D401,
        [((1, 30, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D402,
        [((1, 720, 16, 16), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D403,
        [((1, 720, 8, 8), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D404,
        [((1, 208, 8, 8), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D405,
        [((1, 1248, 8, 8), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "1248",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D406,
        [((1, 1248, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D407,
        [((1, 52, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D408,
        [((1, 1248, 8, 8), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D409,
        [((1, 1248, 8, 8), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1248",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D410,
        [((1, 1248, 8, 8), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D411,
        [((1, 352, 8, 8), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D412,
        [((1, 2112, 8, 8), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "2112",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D413,
        [((1, 2112, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D414,
        [((1, 88, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D415,
        [((1, 2112, 8, 8), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D416,
        [((1, 352, 8, 8), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D290,
        [((1, 3, 448, 448), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D5,
        [((1, 48, 224, 224), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "48",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D292,
        [((1, 48, 224, 224), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D104,
        [((1, 24, 224, 224), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "24",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D105,
        [((1, 24, 224, 224), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D106,
        [((1, 24, 224, 224), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D107,
        [((1, 144, 224, 224), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "144",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D216,
        [((1, 144, 112, 112), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D217,
        [((1, 40, 112, 112), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D417,
        [((1, 240, 112, 112), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "240",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D220,
        [((1, 240, 112, 112), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D418,
        [((1, 240, 112, 112), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "240",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D419,
        [((1, 240, 56, 56), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D325,
        [((1, 64, 56, 56), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D19,
        [((1, 384, 56, 56), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "384",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D420,
        [((1, 384, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D421,
        [((1, 16, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D327,
        [((1, 384, 56, 56), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D422,
        [((1, 384, 56, 56), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "384",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D279,
        [((1, 384, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm", "pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D423,
        [((1, 128, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D424,
        [((1, 768, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "768",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D425,
        [((1, 768, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D426,
        [((1, 32, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D427,
        [((1, 768, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D428,
        [((1, 768, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "768",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D429,
        [((1, 768, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D430,
        [((1, 176, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D431,
        [((1, 1056, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "1056",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D432,
        [((1, 1056, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "1056",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D433,
        [((1, 1056, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D434,
        [((1, 44, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D435,
        [((1, 1056, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D436,
        [((1, 1056, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D437,
        [((1, 304, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D438,
        [((1, 1824, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "1824",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D439,
        [((1, 1824, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D440,
        [((1, 76, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D441,
        [((1, 1824, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D442,
        [((1, 1824, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1824",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D443,
        [((1, 1824, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D444,
        [((1, 512, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D445,
        [((1, 3072, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "3072",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D446,
        [((1, 3072, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D447,
        [((1, 128, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D448,
        [((1, 3072, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D449,
        [((1, 192, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D450,
        [((1, 72, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D451,
        [((1, 432, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "432",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D452,
        [((1, 432, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D453,
        [((1, 432, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D454,
        [((1, 104, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D455,
        [((1, 624, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "624",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D456,
        [((1, 624, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "624",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D457,
        [((1, 624, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D458,
        [((1, 624, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D430,
        [((1, 176, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D459,
        [((1, 1056, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1056",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D435,
        [((1, 1056, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D460,
        [((1, 1056, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D461,
        [((1, 352, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D51,
        [((1, 1024, 128, 128), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D462,
        [((1, 256, 128, 128), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D463,
        [((1, 3, 256, 256), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {
                "stride": "[4, 4]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D464,
        [((1, 3, 224, 224), torch.float32)],
        {
            "model_names": ["pd_alexnet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[4, 4]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D465,
        [((1, 64, 27, 27), torch.float32)],
        {
            "model_names": ["pd_alexnet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D466,
        [((1, 192, 13, 13), torch.float32)],
        {
            "model_names": ["pd_alexnet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D467,
        [((1, 384, 13, 13), torch.float32)],
        {
            "model_names": ["pd_alexnet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D468,
        [((1, 256, 13, 13), torch.float32)],
        {
            "model_names": ["pd_alexnet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D469,
        [((1, 128, 56, 56), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D470,
        [((1, 96, 56, 56), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D471,
        [((1, 160, 56, 56), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D347,
        [((1, 192, 56, 56), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D472,
        [((1, 224, 56, 56), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D91,
        [((1, 128, 28, 28), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D469,
        [((1, 128, 28, 28), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D471,
        [((1, 160, 28, 28), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D347,
        [((1, 192, 28, 28), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D472,
        [((1, 224, 28, 28), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D473,
        [((1, 288, 28, 28), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D474,
        [((1, 320, 28, 28), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D475,
        [((1, 352, 28, 28), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D476,
        [((1, 416, 28, 28), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D477,
        [((1, 480, 28, 28), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D469,
        [((1, 128, 14, 14), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D473,
        [((1, 288, 14, 14), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D474,
        [((1, 320, 14, 14), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D475,
        [((1, 352, 14, 14), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D476,
        [((1, 416, 14, 14), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D477,
        [((1, 480, 14, 14), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D45,
        [((1, 512, 14, 14), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D478,
        [((1, 544, 14, 14), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D479,
        [((1, 608, 14, 14), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D480,
        [((1, 640, 14, 14), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D481,
        [((1, 672, 14, 14), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D482,
        [((1, 704, 14, 14), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D483,
        [((1, 736, 14, 14), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D427,
        [((1, 768, 14, 14), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D484,
        [((1, 800, 14, 14), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D485,
        [((1, 832, 14, 14), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D486,
        [((1, 864, 14, 14), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D487,
        [((1, 896, 14, 14), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D488,
        [((1, 928, 14, 14), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D489,
        [((1, 960, 14, 14), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D490,
        [((1, 992, 14, 14), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D45,
        [((1, 512, 7, 7), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D469,
        [((1, 128, 7, 7), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D478,
        [((1, 544, 7, 7), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D388,
        [((1, 576, 7, 7), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D479,
        [((1, 608, 7, 7), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D480,
        [((1, 640, 7, 7), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D481,
        [((1, 672, 7, 7), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D482,
        [((1, 704, 7, 7), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D483,
        [((1, 736, 7, 7), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D427,
        [((1, 768, 7, 7), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D484,
        [((1, 800, 7, 7), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D485,
        [((1, 832, 7, 7), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D486,
        [((1, 864, 7, 7), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D487,
        [((1, 896, 7, 7), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D488,
        [((1, 928, 7, 7), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D489,
        [((1, 960, 7, 7), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D490,
        [((1, 992, 7, 7), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D491,
        [((1, 64, 112, 112), torch.float32)],
        {
            "model_names": ["pd_mobilenetv1_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "64",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D178,
        [((1, 128, 56, 56), torch.float32)],
        {
            "model_names": ["pd_mobilenetv1_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "128",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D492,
        [((1, 128, 56, 56), torch.float32)],
        {
            "model_names": ["pd_mobilenetv1_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "128",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D152,
        [((1, 256, 28, 28), torch.float32)],
        {
            "model_names": ["pd_mobilenetv1_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "256",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D493,
        [((1, 256, 28, 28), torch.float32)],
        {
            "model_names": ["pd_mobilenetv1_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "256",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D155,
        [((1, 512, 14, 14), torch.float32)],
        {
            "model_names": ["pd_mobilenetv1_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "512",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D494,
        [((1, 512, 14, 14), torch.float32)],
        {
            "model_names": ["pd_mobilenetv1_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "512",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D340,
        [((1, 1024, 7, 7), torch.float32)],
        {
            "model_names": ["pd_mobilenetv1_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1024",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D495,
        [((1, 2048, 1, 6), torch.float32), ((2048, 1, 1, 4), torch.float32)],
        {
            "model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 3, 0, 3]",
                "dilation": "1",
                "groups": "2048",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D195,
        [((1, 80, 3000, 1), torch.float32), ((512, 80, 3, 1), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 0, 1, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D196,
        [((1, 512, 3000, 1), torch.float32), ((512, 512, 3, 1), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {
                "stride": "[2, 1]",
                "padding": "[1, 0, 1, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
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
