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
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.config import VerifyConfig
import pytest


class Conv2D0(ForgeModule):
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
            groups=4,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D1(ForgeModule):
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
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d2.weight_1",
            forge.Parameter(*(64, 3, 11, 11), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d2.weight_1"),
            stride=[4, 4],
            padding=[2, 2, 2, 2],
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
            forge.Parameter(*(192, 64, 5, 5), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d3.weight_1"),
            stride=[1, 1],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d4.weight_1",
            forge.Parameter(*(384, 192, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d4.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(256, 384, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d5.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d6.weight_1",
            forge.Parameter(*(256, 256, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d6.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d7.weight_1",
            forge.Parameter(*(16, 1, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d7.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(4, 16, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d8.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(768, 3, 16, 16), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d9.weight_1"),
            stride=[16, 16],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d10.weight_1",
            forge.Parameter(*(192, 3, 16, 16), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d10.weight_1"),
            stride=[16, 16],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d11.weight_1",
            forge.Parameter(*(384, 3, 16, 16), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d11.weight_1"),
            stride=[16, 16],
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
            forge.Parameter(*(64, 3, 7, 7), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d12.weight_1"),
            stride=[2, 2],
            padding=[3, 3, 3, 3],
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
            forge.Parameter(*(128, 64, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d14.weight_1",
            forge.Parameter(*(32, 128, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d14.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d15.weight_1",
            forge.Parameter(*(32, 128, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d15.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(128, 96, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(128, 128, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(128, 160, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(128, 192, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d19.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D20(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d20.weight_1",
            forge.Parameter(*(128, 224, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(128, 256, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(128, 288, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(128, 320, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(128, 352, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(128, 384, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(128, 416, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(128, 448, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d27.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
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
            forge.Parameter(*(128, 480, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(256, 512, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(128, 512, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(128, 544, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(128, 576, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(128, 608, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(128, 640, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(128, 672, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(128, 704, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d36.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
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
            forge.Parameter(*(128, 736, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(128, 768, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d38.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
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
            forge.Parameter(*(128, 800, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(128, 832, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(128, 864, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(128, 896, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d42.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
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
            forge.Parameter(*(128, 928, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(128, 960, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d44.weight_1"),
            stride=[1, 1],
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
            forge.Parameter(*(128, 992, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(128, 1024, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d46.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
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
            forge.Parameter(*(128, 1056, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(128, 1088, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d48.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
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
            forge.Parameter(*(128, 1120, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(128, 1152, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d50.weight_1"),
            stride=[1, 1],
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
            forge.Parameter(*(128, 1184, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(128, 1216, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d52.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
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
            forge.Parameter(*(128, 1248, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(128, 1280, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d54.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
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
            forge.Parameter(*(128, 1312, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(128, 1344, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d56.weight_1"),
            stride=[1, 1],
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
            forge.Parameter(*(128, 1376, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(128, 1408, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d58.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
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
            forge.Parameter(*(128, 1440, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(128, 1472, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d60.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
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
            forge.Parameter(*(128, 1504, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d61.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
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
            forge.Parameter(*(128, 1536, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d62.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
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
            forge.Parameter(*(128, 1568, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(128, 1600, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d64.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D65(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d65.weight_1",
            forge.Parameter(*(128, 1632, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(128, 1664, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(128, 1696, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d67.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D68(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d68.weight_1",
            forge.Parameter(*(128, 1728, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d68.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D69(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d69.weight_1",
            forge.Parameter(*(128, 1760, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d69.weight_1"),
            stride=[1, 1],
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
            forge.Parameter(*(896, 1792, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(128, 1792, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d71.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D72(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d72.weight_1",
            forge.Parameter(*(128, 1824, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(128, 1856, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(128, 1888, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d74.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D75(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d75.weight_1",
            forge.Parameter(*(512, 1024, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(96, 3, 7, 7), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d76.weight_1"),
            stride=[2, 2],
            padding=[3, 3, 3, 3],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D77(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d77.weight_1",
            forge.Parameter(*(192, 96, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(48, 192, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d78.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(192, 144, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d79.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D80(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d80.weight_1",
            forge.Parameter(*(192, 192, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(192, 240, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(192, 288, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d82.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D83(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d83.weight_1",
            forge.Parameter(*(192, 336, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(192, 384, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(192, 432, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(192, 480, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d86.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D87(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d87.weight_1",
            forge.Parameter(*(192, 528, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d87.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D88(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d88.weight_1",
            forge.Parameter(*(192, 576, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(192, 624, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(192, 672, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d90.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D91(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d91.weight_1",
            forge.Parameter(*(192, 720, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(384, 768, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d92.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D93(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d93.weight_1",
            forge.Parameter(*(192, 768, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(192, 816, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d94.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D95(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d95.weight_1",
            forge.Parameter(*(192, 864, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(192, 912, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d96.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D97(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d97.weight_1",
            forge.Parameter(*(192, 960, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(192, 1008, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(192, 1056, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d99.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
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
            forge.Parameter(*(192, 1104, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d100.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D101(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d101.weight_1",
            forge.Parameter(*(192, 1152, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(192, 1200, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(192, 1248, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(192, 1296, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d104.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D105(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d105.weight_1",
            forge.Parameter(*(192, 1344, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(192, 1392, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(192, 1440, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d107.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D108(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d108.weight_1",
            forge.Parameter(*(192, 1488, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(192, 1536, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(192, 1584, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(192, 1632, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(192, 1680, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d112.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D113(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d113.weight_1",
            forge.Parameter(*(192, 1728, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(192, 1776, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(192, 1824, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(192, 1872, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(192, 1920, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d117.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D118(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d118.weight_1",
            forge.Parameter(*(192, 1968, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(192, 2016, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(192, 2064, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(1056, 2112, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d121.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D122(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d122.weight_1",
            forge.Parameter(*(192, 2112, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(192, 2160, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(640, 1280, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d124.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D125(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d125.weight_1",
            forge.Parameter(*(16, 3, 7, 7), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d125.weight_1"),
            stride=[1, 1],
            padding=[3, 3, 3, 3],
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
            forge.Parameter(*(16, 16, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d126.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(32, 16, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d127.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(64, 32, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d128.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D129(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d129.weight_1",
            forge.Parameter(*(64, 2, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d129.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=32,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D130(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d130.weight_1",
            forge.Parameter(*(64, 64, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(64, 2, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d131.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=32,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D132(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d132.weight_1",
            forge.Parameter(*(64, 128, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d132.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D133(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d133.weight_1",
            forge.Parameter(*(64, 256, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(128, 4, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d134.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=32,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D135(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d135.weight_1",
            forge.Parameter(*(128, 4, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d135.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=32,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D136(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d136.weight_1",
            forge.Parameter(*(256, 128, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(256, 8, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d137.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=32,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D138(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d138.weight_1",
            forge.Parameter(*(256, 256, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d138.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D139(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d139.weight_1",
            forge.Parameter(*(256, 8, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d139.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=32,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D140(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d140.weight_1",
            forge.Parameter(*(256, 640, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(1000, 256, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(32, 32, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d142.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D143(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d143.weight_1",
            forge.Parameter(*(32, 32, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d143.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(32, 32, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d144.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(32, 64, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d145.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D146(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d146.weight_1",
            forge.Parameter(*(64, 64, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d146.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(64, 64, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d147.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(128, 128, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d148.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(128, 128, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d149.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(256, 32, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d150.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
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
            forge.Parameter(*(256, 4, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d151.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=64,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D152(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d152.weight_1",
            forge.Parameter(*(128, 32, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d152.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D153(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d153.weight_1",
            forge.Parameter(*(256, 4, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d153.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=64,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D154(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d154.weight_1",
            forge.Parameter(*(512, 128, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d154.weight_1"),
            stride=[1, 1],
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
            forge.Parameter(*(512, 8, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d155.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=64,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D156(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d156.weight_1",
            forge.Parameter(*(512, 256, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d156.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
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
            forge.Parameter(*(512, 8, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d157.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=64,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D158(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d158.weight_1",
            forge.Parameter(*(256, 768, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d158.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D159(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d159.weight_1",
            forge.Parameter(*(256, 1152, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d159.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
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
            forge.Parameter(*(1024, 256, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d160.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D161(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d161.weight_1",
            forge.Parameter(*(1024, 16, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d161.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=64,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D162(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d162.weight_1",
            forge.Parameter(*(1024, 512, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(1024, 16, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d163.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=64,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D164(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d164.weight_1",
            forge.Parameter(*(512, 1536, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d164.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
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
            forge.Parameter(*(512, 2048, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d165.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
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
            forge.Parameter(*(512, 2816, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d166.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
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
            forge.Parameter(*(2048, 512, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d167.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
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
            forge.Parameter(*(2048, 32, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d168.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=64,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D169(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d169.weight_1",
            forge.Parameter(*(1024, 2048, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d169.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
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
            forge.Parameter(*(2048, 1024, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d170.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
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
            forge.Parameter(*(2048, 32, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d171.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=64,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D172(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d172.weight_1",
            forge.Parameter(*(1024, 2560, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d172.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
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
            forge.Parameter(*(1000, 1024, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d173.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
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
            forge.Parameter(*(256, 256, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d174.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(512, 2560, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d175.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D176(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d176.weight_1",
            forge.Parameter(*(512, 3328, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d176.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D177(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d177.weight_1",
            forge.Parameter(*(512, 512, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d177.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D178(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d178.weight_1",
            forge.Parameter(*(512, 512, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d178.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D179(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d179.weight_1",
            forge.Parameter(*(512, 512, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d179.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D180(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d180.weight_1",
            forge.Parameter(*(256, 896, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(512, 2304, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d181.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D182(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d182.weight_1",
            forge.Parameter(*(128, 32, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d182.weight_1"),
            stride=[2, 2],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D183(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d183.weight_1",
            forge.Parameter(*(512, 16, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d183.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=32,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D184(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d184.weight_1",
            forge.Parameter(*(512, 16, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d184.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=32,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D185(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d185.weight_1",
            forge.Parameter(*(1024, 32, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d185.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=32,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D186(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d186.weight_1",
            forge.Parameter(*(1024, 32, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d186.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=32,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D187(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d187.weight_1",
            forge.Parameter(*(1024, 1024, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d187.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D188(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d188.weight_1",
            forge.Parameter(*(64, 32, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d188.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D189(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d189.weight_1",
            forge.Parameter(*(128, 64, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d189.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(256, 128, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d190.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(512, 256, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d191.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(512, 1280, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d192.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
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
            forge.Parameter(*(1000, 512, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(48, 3, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d194.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=48,
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
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=48,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D197(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d197.weight_1",
            forge.Parameter(*(12, 48, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d197.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D198(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d198.weight_1",
            forge.Parameter(*(48, 12, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(24, 48, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d199.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D200(ForgeModule):
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
            groups=24,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D201(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d201.weight_1",
            forge.Parameter(*(6, 24, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(24, 6, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d202.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D203(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d203.weight_1",
            forge.Parameter(*(24, 24, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d203.weight_1"),
            stride=[1, 1],
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
            forge.Parameter(*(144, 24, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=144,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D206(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d206.weight_1",
            forge.Parameter(*(6, 144, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d206.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
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
            forge.Parameter(*(144, 6, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(32, 144, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(192, 32, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=192,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D211(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d211.weight_1",
            forge.Parameter(*(8, 192, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(192, 8, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(32, 192, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d213.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D214(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[2, 2],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=192,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D215(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d215.weight_1",
            forge.Parameter(*(56, 192, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d215.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D216(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d216.weight_1",
            forge.Parameter(*(336, 56, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[1, 1],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=336,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D218(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d218.weight_1",
            forge.Parameter(*(14, 336, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(336, 14, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(56, 336, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=336,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D222(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d222.weight_1",
            forge.Parameter(*(112, 336, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(672, 112, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=672,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D225(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d225.weight_1",
            forge.Parameter(*(28, 672, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(672, 28, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(112, 672, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[1, 1],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=672,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D229(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[2, 2],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=672,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D230(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d230.weight_1",
            forge.Parameter(*(160, 672, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d230.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D231(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d231.weight_1",
            forge.Parameter(*(960, 160, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d231.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D232(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[1, 1],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=960,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D233(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[2, 2],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=960,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D234(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d234.weight_1",
            forge.Parameter(*(40, 960, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(960, 40, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(160, 960, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(272, 960, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d237.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D238(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d238.weight_1",
            forge.Parameter(*(1632, 272, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[1, 1],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=1632,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D240(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d240.weight_1",
            forge.Parameter(*(68, 1632, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(1632, 68, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d241.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D242(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d242.weight_1",
            forge.Parameter(*(272, 1632, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1632,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D244(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d244.weight_1",
            forge.Parameter(*(448, 1632, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(2688, 448, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=2688,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D247(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d247.weight_1",
            forge.Parameter(*(112, 2688, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(2688, 112, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(448, 2688, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d249.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D250(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d250.weight_1",
            forge.Parameter(*(1792, 448, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(32, 3, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d251.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(32, 3, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d252.weight_1"),
            stride=[2, 2],
            padding=[0, 1, 0, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D253(ForgeModule):
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
            groups=32,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D254(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d254.weight_1",
            forge.Parameter(*(8, 32, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d254.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
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
            forge.Parameter(*(32, 8, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(16, 32, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d256.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
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
            forge.Parameter(*(96, 16, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d257.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D258(ForgeModule):
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
            groups=96,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D259(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[2, 2],
            padding=[0, 1, 0, 1],
            dilation=1,
            groups=96,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D260(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d260.weight_1",
            forge.Parameter(*(4, 96, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(96, 4, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d261.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
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
            forge.Parameter(*(24, 96, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d262.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D263(ForgeModule):
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
            groups=144,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D264(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[2, 2],
            padding=[0, 1, 0, 1],
            dilation=1,
            groups=144,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D265(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d265.weight_1",
            forge.Parameter(*(24, 144, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=144,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D267(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d267.weight_1",
            forge.Parameter(*(40, 144, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d267.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D268(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d268.weight_1",
            forge.Parameter(*(240, 40, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d268.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
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
            stride=[1, 1],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=240,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D270(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d270.weight_1",
            forge.Parameter(*(10, 240, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d270.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D271(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d271.weight_1",
            forge.Parameter(*(240, 10, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(40, 240, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=240,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D274(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d274.weight_1",
            forge.Parameter(*(80, 240, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d274.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
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
            forge.Parameter(*(480, 80, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d275.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D276(ForgeModule):
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
            groups=480,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D277(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d277.weight_1",
            forge.Parameter(*(20, 480, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d277.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
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
            forge.Parameter(*(480, 20, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d278.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
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
            forge.Parameter(*(80, 480, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[1, 1],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=480,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D281(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d281.weight_1",
            forge.Parameter(*(112, 480, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(1152, 192, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[1, 1],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=1152,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D284(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d284.weight_1",
            forge.Parameter(*(48, 1152, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(1152, 48, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1152,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D287(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d287.weight_1",
            forge.Parameter(*(320, 1152, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(1280, 320, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(256, 2048, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(16, 3, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(8, 16, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=8,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D293(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d293.weight_1",
            forge.Parameter(*(24, 16, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=12,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D295(ForgeModule):
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
            groups=16,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D296(ForgeModule):
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
            groups=16,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D297(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d297.weight_1",
            forge.Parameter(*(36, 24, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=36,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D299(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d299.weight_1",
            forge.Parameter(*(12, 72, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d299.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D300(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[2, 2],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=72,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D301(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d301.weight_1",
            forge.Parameter(*(20, 72, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d301.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D302(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d302.weight_1",
            forge.Parameter(*(72, 20, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=20,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D304(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[2, 2],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=24,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D305(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d305.weight_1",
            forge.Parameter(*(40, 24, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d305.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D306(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d306.weight_1",
            forge.Parameter(*(60, 40, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=60,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D308(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d308.weight_1",
            forge.Parameter(*(32, 120, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(120, 32, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(20, 120, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(120, 40, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d311.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D312(ForgeModule):
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
            groups=120,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D313(ForgeModule):
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
            groups=40,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D314(ForgeModule):
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
            groups=40,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D315(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d315.weight_1",
            forge.Parameter(*(80, 40, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d315.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D316(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d316.weight_1",
            forge.Parameter(*(100, 80, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=100,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D318(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d318.weight_1",
            forge.Parameter(*(40, 200, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d318.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D319(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d319.weight_1",
            forge.Parameter(*(92, 80, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=92,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D321(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d321.weight_1",
            forge.Parameter(*(40, 184, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(240, 80, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=240,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D324(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d324.weight_1",
            forge.Parameter(*(120, 480, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(480, 120, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(56, 480, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d326.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D327(ForgeModule):
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
            groups=56,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D328(ForgeModule):
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
            groups=80,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D329(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d329.weight_1",
            forge.Parameter(*(112, 80, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(336, 112, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d330.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D331(ForgeModule):
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
            groups=336,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D332(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d332.weight_1",
            forge.Parameter(*(168, 672, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d332.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
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
            forge.Parameter(*(672, 168, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d333.weight_1"),
            stride=[1, 1],
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
            forge.Parameter(*(56, 672, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d334.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
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
            forge.Parameter(*(80, 672, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d335.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D336(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[2, 2],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=112,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D337(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d337.weight_1",
            forge.Parameter(*(160, 112, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d337.weight_1"),
            stride=[1, 1],
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
            forge.Parameter(*(480, 160, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d338.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D339(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d339.weight_1",
            forge.Parameter(*(80, 960, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d339.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
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
            forge.Parameter(*(240, 960, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d340.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D341(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d341.weight_1",
            forge.Parameter(*(960, 240, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d341.weight_1"),
            stride=[1, 1],
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
            forge.Parameter(*(1280, 960, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(192, 64, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d343.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(128, 96, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(32, 16, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d345.weight_1"),
            stride=[1, 1],
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
            forge.Parameter(*(192, 128, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d346.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(96, 32, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d347.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(208, 96, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d348.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(48, 16, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(64, 480, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(224, 112, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(64, 24, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(64, 512, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(256, 128, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d354.weight_1"),
            stride=[1, 1],
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
            forge.Parameter(*(288, 144, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d355.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(64, 32, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d356.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D357(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d357.weight_1",
            forge.Parameter(*(320, 160, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d357.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(320, 160, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d358.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(128, 32, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d359.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(128, 32, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d360.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D361(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d361.weight_1",
            forge.Parameter(*(128, 528, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(128, 48, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d362.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(64, 3, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d363.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D364(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d364.weight_1",
            forge.Parameter(*(64, 3, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d364.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(256, 64, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d365.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D366(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d366.weight_1",
            forge.Parameter(*(18, 256, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d366.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(18, 18, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d367.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(18, 18, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d368.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(36, 256, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d369.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(36, 36, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d370.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D371(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d371.weight_1",
            forge.Parameter(*(36, 36, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d371.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(18, 36, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(36, 18, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d373.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(72, 36, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d374.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(72, 72, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d375.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
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
            forge.Parameter(*(36, 72, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(72, 18, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d378.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(144, 72, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d379.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D380(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d380.weight_1",
            forge.Parameter(*(144, 144, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d380.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D381(ForgeModule):
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
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D382(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d382.weight_1",
            forge.Parameter(*(144, 18, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d382.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(144, 36, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d383.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(256, 144, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(1024, 144, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d385.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
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
            forge.Parameter(*(128, 72, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(512, 72, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(64, 36, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(256, 36, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(32, 18, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(128, 18, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d391.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D392(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d392.weight_1",
            forge.Parameter(*(512, 256, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d392.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(1024, 512, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d393.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(40, 256, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d394.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(40, 40, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d395.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D396(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d396.weight_1",
            forge.Parameter(*(40, 40, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d396.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(80, 256, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d397.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(80, 80, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d398.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D399(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d399.weight_1",
            forge.Parameter(*(80, 80, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d399.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D400(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d400.weight_1",
            forge.Parameter(*(40, 80, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(80, 40, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d401.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(160, 80, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d402.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(160, 160, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d403.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(40, 160, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(80, 160, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d405.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D406(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d406.weight_1",
            forge.Parameter(*(160, 40, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d406.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(320, 320, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d407.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(320, 40, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d408.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(320, 80, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d409.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D410(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d410.weight_1",
            forge.Parameter(*(256, 320, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(1024, 320, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(512, 160, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d412.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D413(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d413.weight_1",
            forge.Parameter(*(64, 80, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(256, 80, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(32, 40, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(128, 40, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(64, 256, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d417.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D418(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d418.weight_1",
            forge.Parameter(*(64, 256, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d418.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D419(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d419.weight_1",
            forge.Parameter(*(128, 256, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d419.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(256, 64, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d420.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(512, 64, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d421.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(512, 128, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d422.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D423(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d423.weight_1",
            forge.Parameter(*(32, 256, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d423.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(32, 128, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d424.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D425(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d425.weight_1",
            forge.Parameter(*(256, 32, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d425.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(16, 128, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d426.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(16, 16, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d427.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(16, 64, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d428.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D429(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d429.weight_1",
            forge.Parameter(*(128, 16, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d429.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(64, 16, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d430.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(1024, 128, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d431.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D432(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d432.weight_1",
            forge.Parameter(*(512, 64, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d432.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D433(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d433.weight_1",
            forge.Parameter(*(32, 16, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(128, 16, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(44, 256, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d435.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(44, 44, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d436.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(44, 44, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d437.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(88, 256, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d438.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D439(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d439.weight_1",
            forge.Parameter(*(88, 88, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d439.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(88, 88, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d440.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(44, 88, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(88, 44, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d442.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D443(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d443.weight_1",
            forge.Parameter(*(176, 88, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d443.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(176, 176, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d444.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(44, 176, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d445.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D446(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d446.weight_1",
            forge.Parameter(*(88, 176, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(176, 44, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d447.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(352, 176, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d448.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(352, 352, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d449.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(352, 44, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d450.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(352, 88, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d451.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D452(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d452.weight_1",
            forge.Parameter(*(256, 352, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(1024, 352, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(128, 176, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(512, 176, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d455.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D456(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d456.weight_1",
            forge.Parameter(*(64, 88, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d456.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D457(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d457.weight_1",
            forge.Parameter(*(256, 88, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(32, 44, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(128, 44, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d459.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D460(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d460.weight_1",
            forge.Parameter(*(48, 256, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d460.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(48, 48, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d461.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(48, 48, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d462.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(96, 256, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d463.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(96, 96, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d464.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(96, 96, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d465.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(48, 96, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d466.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
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
            forge.Parameter(*(96, 48, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d467.weight_1"),
            stride=[2, 2],
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
            forge.Parameter(*(192, 96, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d468.weight_1"),
            stride=[2, 2],
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
            forge.Parameter(*(192, 192, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(48, 192, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(96, 192, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(192, 48, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d472.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(384, 192, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d473.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(384, 384, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d474.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(384, 48, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d475.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(384, 96, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d476.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(256, 384, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(1024, 384, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(512, 192, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(64, 96, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(256, 96, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(32, 48, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(128, 48, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(30, 256, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d484.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(30, 30, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d485.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(30, 30, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d486.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(60, 256, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d487.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(60, 60, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d488.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(60, 60, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d489.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
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
            forge.Parameter(*(30, 60, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
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
            forge.Parameter(*(60, 30, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d491.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D492(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d492.weight_1",
            forge.Parameter(*(120, 60, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d492.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D493(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d493.weight_1",
            forge.Parameter(*(120, 120, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d493.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D494(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d494.weight_1",
            forge.Parameter(*(30, 120, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d494.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D495(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d495.weight_1",
            forge.Parameter(*(60, 120, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d495.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D496(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d496.weight_1",
            forge.Parameter(*(120, 30, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d496.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D497(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d497.weight_1",
            forge.Parameter(*(240, 120, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d497.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D498(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d498.weight_1",
            forge.Parameter(*(240, 240, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d498.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D499(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d499.weight_1",
            forge.Parameter(*(240, 30, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d499.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D500(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d500.weight_1",
            forge.Parameter(*(240, 60, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d500.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D501(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d501.weight_1",
            forge.Parameter(*(256, 240, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d501.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D502(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d502.weight_1",
            forge.Parameter(*(1024, 240, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d502.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D503(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d503.weight_1",
            forge.Parameter(*(128, 120, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d503.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D504(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d504.weight_1",
            forge.Parameter(*(512, 120, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d504.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D505(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d505.weight_1",
            forge.Parameter(*(64, 60, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d505.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D506(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d506.weight_1",
            forge.Parameter(*(256, 60, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d506.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D507(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d507.weight_1",
            forge.Parameter(*(32, 30, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d507.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D508(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d508.weight_1",
            forge.Parameter(*(128, 30, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d508.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D509(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d509.weight_1",
            forge.Parameter(*(32, 3, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d509.weight_1"),
            stride=[2, 2],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D510(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d510.weight_1",
            forge.Parameter(*(32, 32, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d510.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D511(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d511.weight_1",
            forge.Parameter(*(96, 64, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d511.weight_1"),
            stride=[2, 2],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D512(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d512.weight_1",
            forge.Parameter(*(64, 160, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d512.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D513(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d513.weight_1",
            forge.Parameter(*(96, 64, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d513.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D514(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d514.weight_1",
            forge.Parameter(*(64, 64, 1, 7), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d514.weight_1"),
            stride=[1, 1],
            padding=[3, 3, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D515(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d515.weight_1",
            forge.Parameter(*(64, 64, 7, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d515.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 3, 3],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D516(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d516.weight_1",
            forge.Parameter(*(192, 192, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d516.weight_1"),
            stride=[2, 2],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D517(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d517.weight_1",
            forge.Parameter(*(96, 64, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d517.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D518(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d518.weight_1",
            forge.Parameter(*(96, 384, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d518.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D519(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d519.weight_1",
            forge.Parameter(*(384, 384, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d519.weight_1"),
            stride=[2, 2],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D520(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d520.weight_1",
            forge.Parameter(*(224, 192, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d520.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D521(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d521.weight_1",
            forge.Parameter(*(256, 224, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d521.weight_1"),
            stride=[2, 2],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D522(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d522.weight_1",
            forge.Parameter(*(224, 192, 1, 7), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d522.weight_1"),
            stride=[1, 1],
            padding=[3, 3, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D523(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d523.weight_1",
            forge.Parameter(*(256, 224, 7, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d523.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 3, 3],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D524(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d524.weight_1",
            forge.Parameter(*(192, 192, 7, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d524.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 3, 3],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D525(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d525.weight_1",
            forge.Parameter(*(224, 224, 7, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d525.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 3, 3],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D526(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d526.weight_1",
            forge.Parameter(*(256, 224, 1, 7), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d526.weight_1"),
            stride=[1, 1],
            padding=[3, 3, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D527(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d527.weight_1",
            forge.Parameter(*(192, 1024, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d527.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D528(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d528.weight_1",
            forge.Parameter(*(256, 1024, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d528.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D529(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d529.weight_1",
            forge.Parameter(*(256, 256, 1, 7), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d529.weight_1"),
            stride=[1, 1],
            padding=[3, 3, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D530(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d530.weight_1",
            forge.Parameter(*(320, 256, 7, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d530.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 3, 3],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D531(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d531.weight_1",
            forge.Parameter(*(320, 320, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d531.weight_1"),
            stride=[2, 2],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D532(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d532.weight_1",
            forge.Parameter(*(256, 384, 1, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d532.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D533(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d533.weight_1",
            forge.Parameter(*(256, 384, 3, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d533.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D534(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d534.weight_1",
            forge.Parameter(*(448, 384, 3, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d534.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D535(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d535.weight_1",
            forge.Parameter(*(512, 448, 1, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d535.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D536(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d536.weight_1",
            forge.Parameter(*(256, 512, 1, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d536.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D537(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d537.weight_1",
            forge.Parameter(*(256, 512, 3, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d537.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D538(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d538.weight_1",
            forge.Parameter(*(256, 1536, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d538.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D539(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d539.weight_1",
            forge.Parameter(*(768, 3, 32, 32), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d539.weight_1"),
            stride=[32, 32],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D540(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d540.weight_1",
            forge.Parameter(*(1024, 3, 32, 32), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d540.weight_1"),
            stride=[32, 32],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D541(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d541.weight_1",
            forge.Parameter(*(512, 3, 32, 32), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d541.weight_1"),
            stride=[32, 32],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D542(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d542.weight_1",
            forge.Parameter(*(512, 3, 16, 16), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d542.weight_1"),
            stride=[16, 16],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D543(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d543.weight_1",
            forge.Parameter(*(1024, 3, 16, 16), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d543.weight_1"),
            stride=[16, 16],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D544(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d544.weight_1",
            forge.Parameter(*(24, 3, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d544.weight_1"),
            stride=[2, 2],
            padding=[0, 1, 0, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D545(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d545.weight_1",
            forge.Parameter(*(48, 24, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d545.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D546(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[2, 2],
            padding=[0, 1, 0, 1],
            dilation=1,
            groups=48,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D547(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d547.weight_1",
            forge.Parameter(*(96, 48, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d547.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D548(ForgeModule):
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
            groups=96,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D549(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d549.weight_1",
            forge.Parameter(*(96, 96, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d549.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D550(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[2, 2],
            padding=[0, 1, 0, 1],
            dilation=1,
            groups=192,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D551(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d551.weight_1",
            forge.Parameter(*(384, 192, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d551.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D552(ForgeModule):
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
            groups=384,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D553(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[2, 2],
            padding=[0, 1, 0, 1],
            dilation=1,
            groups=384,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D554(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d554.weight_1",
            forge.Parameter(*(384, 384, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d554.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D555(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d555.weight_1",
            forge.Parameter(*(768, 384, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d555.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D556(ForgeModule):
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
            groups=768,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D557(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d557.weight_1",
            forge.Parameter(*(768, 768, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d557.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D558(ForgeModule):
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
            groups=64,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D559(ForgeModule):
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
            groups=128,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D560(ForgeModule):
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
            groups=128,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D561(ForgeModule):
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
            groups=256,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D562(ForgeModule):
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
            groups=256,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D563(ForgeModule):
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
            groups=512,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D564(ForgeModule):
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
            groups=512,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D565(ForgeModule):
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
            groups=1024,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D566(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[2, 2],
            padding=[0, 1, 0, 1],
            dilation=1,
            groups=64,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D567(ForgeModule):
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
            groups=64,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D568(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[2, 2],
            padding=[0, 1, 0, 1],
            dilation=1,
            groups=128,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D569(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[2, 2],
            padding=[0, 1, 0, 1],
            dilation=1,
            groups=256,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D570(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[2, 2],
            padding=[0, 1, 0, 1],
            dilation=1,
            groups=512,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D571(ForgeModule):
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
            groups=192,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D572(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d572.weight_1",
            forge.Parameter(*(64, 192, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d572.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D573(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d573.weight_1",
            forge.Parameter(*(384, 64, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d573.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D574(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d574.weight_1",
            forge.Parameter(*(64, 384, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d574.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D575(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d575.weight_1",
            forge.Parameter(*(576, 96, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d575.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D576(ForgeModule):
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
            groups=576,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D577(ForgeModule):
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
            groups=576,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D578(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[2, 2],
            padding=[0, 1, 0, 1],
            dilation=1,
            groups=576,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D579(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d579.weight_1",
            forge.Parameter(*(96, 576, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d579.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D580(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d580.weight_1",
            forge.Parameter(*(160, 576, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d580.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D581(ForgeModule):
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
            groups=960,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D582(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d582.weight_1",
            forge.Parameter(*(320, 960, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d582.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D583(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d583.weight_1",
            forge.Parameter(*(16, 24, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d583.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D584(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d584.weight_1",
            forge.Parameter(*(48, 144, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d584.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D585(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d585.weight_1",
            forge.Parameter(*(288, 48, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d585.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D586(ForgeModule):
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
            groups=288,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D587(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d587.weight_1",
            forge.Parameter(*(48, 288, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d587.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D588(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d588.weight_1",
            forge.Parameter(*(72, 288, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d588.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D589(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d589.weight_1",
            forge.Parameter(*(432, 72, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d589.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D590(ForgeModule):
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
            groups=432,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D591(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[2, 2],
            padding=[0, 1, 0, 1],
            dilation=1,
            groups=432,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D592(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d592.weight_1",
            forge.Parameter(*(72, 432, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d592.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D593(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d593.weight_1",
            forge.Parameter(*(120, 432, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d593.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D594(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d594.weight_1",
            forge.Parameter(*(720, 120, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d594.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D595(ForgeModule):
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
            groups=720,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D596(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d596.weight_1",
            forge.Parameter(*(120, 720, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d596.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D597(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d597.weight_1",
            forge.Parameter(*(240, 720, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d597.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D598(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d598.weight_1",
            forge.Parameter(*(1280, 240, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d598.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D599(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d599.weight_1",
            forge.Parameter(*(16, 3, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d599.weight_1"),
            stride=[2, 2],
            padding=[0, 1, 0, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D600(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d600.weight_1",
            forge.Parameter(*(48, 8, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d600.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D601(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d601.weight_1",
            forge.Parameter(*(8, 48, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d601.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D602(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d602.weight_1",
            forge.Parameter(*(16, 48, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d602.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D603(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d603.weight_1",
            forge.Parameter(*(16, 96, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d603.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D604(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d604.weight_1",
            forge.Parameter(*(1280, 112, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d604.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D605(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[1, 1],
            padding=[2, 2, 2, 2],
            dilation=2,
            groups=384,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D606(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[1, 1],
            padding=[2, 2, 2, 2],
            dilation=2,
            groups=576,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D607(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[1, 1],
            padding=[4, 4, 4, 4],
            dilation=4,
            groups=960,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D608(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d608.weight_1",
            forge.Parameter(*(21, 256, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d608.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D609(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d609.weight_1",
            forge.Parameter(*(16, 8, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d609.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D610(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d610.weight_1",
            forge.Parameter(*(16, 16, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d610.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D611(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d611.weight_1",
            forge.Parameter(*(72, 16, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d611.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D612(ForgeModule):
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
            groups=72,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D613(ForgeModule):
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
            groups=72,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D614(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d614.weight_1",
            forge.Parameter(*(24, 72, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d614.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D615(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d615.weight_1",
            forge.Parameter(*(88, 24, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d615.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D616(ForgeModule):
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
            groups=88,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D617(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d617.weight_1",
            forge.Parameter(*(24, 88, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d617.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D618(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d618.weight_1",
            forge.Parameter(*(96, 24, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d618.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D619(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[2, 2],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=96,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D620(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d620.weight_1",
            forge.Parameter(*(40, 96, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d620.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D621(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d621.weight_1",
            forge.Parameter(*(64, 240, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d621.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D622(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d622.weight_1",
            forge.Parameter(*(240, 64, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d622.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D623(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[1, 1],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=120,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D624(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d624.weight_1",
            forge.Parameter(*(48, 120, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d624.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D625(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d625.weight_1",
            forge.Parameter(*(144, 48, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d625.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D626(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[1, 1],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=144,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D627(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d627.weight_1",
            forge.Parameter(*(144, 40, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d627.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D628(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[2, 2],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=288,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D629(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d629.weight_1",
            forge.Parameter(*(288, 72, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d629.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D630(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d630.weight_1",
            forge.Parameter(*(96, 288, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d630.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D631(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[1, 1],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=576,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D632(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d632.weight_1",
            forge.Parameter(*(144, 576, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d632.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D633(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d633.weight_1",
            forge.Parameter(*(576, 144, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d633.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D634(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d634.weight_1",
            forge.Parameter(*(1024, 576, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d634.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D635(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d635.weight_1",
            forge.Parameter(*(64, 16, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d635.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D636(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d636.weight_1",
            forge.Parameter(*(24, 64, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d636.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D637(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d637.weight_1",
            forge.Parameter(*(72, 24, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d637.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D638(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d638.weight_1",
            forge.Parameter(*(40, 72, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d638.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D639(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d639.weight_1",
            forge.Parameter(*(40, 120, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d639.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D640(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d640.weight_1",
            forge.Parameter(*(200, 80, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d640.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D641(ForgeModule):
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
            groups=200,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D642(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d642.weight_1",
            forge.Parameter(*(80, 200, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d642.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D643(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d643.weight_1",
            forge.Parameter(*(184, 80, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d643.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D644(ForgeModule):
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
            groups=184,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D645(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d645.weight_1",
            forge.Parameter(*(80, 184, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d645.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D646(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d646.weight_1",
            forge.Parameter(*(128, 64, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d646.weight_1"),
            stride=[2, 2],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D647(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d647.weight_1",
            forge.Parameter(*(256, 128, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d647.weight_1"),
            stride=[2, 2],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D648(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d648.weight_1",
            forge.Parameter(*(512, 256, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d648.weight_1"),
            stride=[2, 2],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D649(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d649.weight_1",
            forge.Parameter(*(256, 512, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d649.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D650(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d650.weight_1",
            forge.Parameter(*(128, 256, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d650.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D651(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d651.weight_1",
            forge.Parameter(*(64, 128, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d651.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D652(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d652.weight_1",
            forge.Parameter(*(32, 64, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d652.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D653(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d653.weight_1",
            forge.Parameter(*(32, 96, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d653.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D654(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d654.weight_1",
            forge.Parameter(*(16, 32, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d654.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D655(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d655.weight_1",
            forge.Parameter(*(16, 16, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d655.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D656(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d656.weight_1",
            forge.Parameter(*(1, 16, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d656.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D657(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d657.weight_1",
            forge.Parameter(*(64, 128, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d657.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D658(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d658.weight_1",
            forge.Parameter(*(128, 256, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d658.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D659(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d659.weight_1",
            forge.Parameter(*(256, 512, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d659.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D660(ForgeModule):
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
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D661(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d661.weight_1",
            forge.Parameter(*(3, 256, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d661.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D662(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d662.weight_1",
            forge.Parameter(*(2, 256, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d662.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D663(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d663.weight_1",
            forge.Parameter(*(24, 256, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d663.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D664(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d664.weight_1",
            forge.Parameter(*(256, 3, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d664.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D665(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d665_const_1", shape=(64, 3, 11, 11), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d665_const_1"),
            stride=[4, 4],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D666(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d666_const_1", shape=(192, 64, 5, 5), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d666_const_1"),
            stride=[1, 1],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D667(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d667_const_1", shape=(384, 192, 3, 3), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d667_const_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D668(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d668_const_1", shape=(256, 384, 3, 3), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d668_const_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D669(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d669_const_1", shape=(256, 256, 3, 3), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d669_const_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D670(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d670.weight_1",
            forge.Parameter(*(128, 64, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d670.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=2,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D671(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d671.weight_1",
            forge.Parameter(*(8, 128, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d671.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D672(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d672.weight_1",
            forge.Parameter(*(128, 8, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d672.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D673(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d673.weight_1",
            forge.Parameter(*(128, 64, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d673.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=2,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D674(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d674.weight_1",
            forge.Parameter(*(192, 128, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d674.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D675(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d675.weight_1",
            forge.Parameter(*(192, 128, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d675.weight_1"),
            stride=[2, 2],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D676(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d676.weight_1",
            forge.Parameter(*(192, 64, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d676.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=3,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D677(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d677.weight_1",
            forge.Parameter(*(192, 64, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d677.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=3,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D678(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d678.weight_1",
            forge.Parameter(*(192, 48, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d678.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D679(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d679.weight_1",
            forge.Parameter(*(512, 192, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d679.weight_1"),
            stride=[2, 2],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D680(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d680.weight_1",
            forge.Parameter(*(512, 64, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d680.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=8,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D681(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d681.weight_1",
            forge.Parameter(*(48, 512, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d681.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D682(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d682.weight_1",
            forge.Parameter(*(512, 48, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d682.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D683(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d683.weight_1",
            forge.Parameter(*(512, 64, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d683.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=8,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D684(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d684.weight_1",
            forge.Parameter(*(1088, 512, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d684.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D685(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d685.weight_1",
            forge.Parameter(*(1088, 512, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d685.weight_1"),
            stride=[2, 2],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D686(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d686.weight_1",
            forge.Parameter(*(1088, 64, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d686.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=17,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D687(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d687.weight_1",
            forge.Parameter(*(1088, 128, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d687.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D688(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d688.weight_1",
            forge.Parameter(*(1088, 1088, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d688.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D689(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d689.weight_1",
            forge.Parameter(*(1088, 64, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d689.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=17,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D690(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d690.weight_1",
            forge.Parameter(*(272, 1088, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d690.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D691(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d691.weight_1",
            forge.Parameter(*(1088, 272, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d691.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D692(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d692.weight_1",
            forge.Parameter(*(1024, 512, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d692.weight_1"),
            stride=[2, 2],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D693(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d693.weight_1",
            forge.Parameter(*(2048, 1024, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d693.weight_1"),
            stride=[2, 2],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D694(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d694.weight_1",
            forge.Parameter(*(2048, 64, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d694.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=32,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D695(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d695.weight_1",
            forge.Parameter(*(2048, 2048, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d695.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D696(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d696.weight_1",
            forge.Parameter(*(2048, 64, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d696.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=32,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D697(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d697.weight_1",
            forge.Parameter(*(720, 256, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d697.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D698(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d698.weight_1",
            forge.Parameter(*(256, 512, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d698.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D699(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d699.weight_1",
            forge.Parameter(*(36, 256, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d699.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D700(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d700.weight_1",
            forge.Parameter(*(256, 2048, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d700.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D701(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d701.weight_1",
            forge.Parameter(*(32, 3, 7, 7), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d701.weight_1"),
            stride=[4, 4],
            padding=[3, 3, 3, 3],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D702(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d702.weight_1",
            forge.Parameter(*(32, 32, 8, 8), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d702.weight_1"),
            stride=[8, 8],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D703(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d703.weight_1",
            forge.Parameter(*(64, 64, 4, 4), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d703.weight_1"),
            stride=[4, 4],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D704(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d704.weight_1",
            forge.Parameter(*(160, 64, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d704.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D705(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d705.weight_1",
            forge.Parameter(*(160, 160, 2, 2), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d705.weight_1"),
            stride=[2, 2],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D706(ForgeModule):
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
            groups=640,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D707(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d707.weight_1",
            forge.Parameter(*(256, 160, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d707.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D708(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d708.weight_1",
            forge.Parameter(*(150, 256, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d708.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D709(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d709.weight_1",
            forge.Parameter(*(64, 3, 7, 7), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d709.weight_1"),
            stride=[4, 4],
            padding=[3, 3, 3, 3],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D710(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d710.weight_1",
            forge.Parameter(*(64, 64, 8, 8), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d710.weight_1"),
            stride=[8, 8],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D711(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d711.weight_1",
            forge.Parameter(*(128, 128, 4, 4), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d711.weight_1"),
            stride=[4, 4],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D712(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d712.weight_1",
            forge.Parameter(*(320, 128, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d712.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D713(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d713.weight_1",
            forge.Parameter(*(320, 320, 2, 2), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d713.weight_1"),
            stride=[2, 2],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D714(ForgeModule):
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
            groups=1280,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D715(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d715.weight_1",
            forge.Parameter(*(512, 320, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d715.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D716(ForgeModule):
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
            groups=2048,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D717(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d717.weight_1",
            forge.Parameter(*(768, 3072, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d717.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D718(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d718.weight_1",
            forge.Parameter(*(150, 768, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d718.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D719(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d719.weight_1",
            forge.Parameter(*(16, 1024, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d719.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D720(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d720.weight_1",
            forge.Parameter(*(24, 512, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d720.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D721(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d721.weight_1",
            forge.Parameter(*(24, 256, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d721.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D722(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d722.weight_1",
            forge.Parameter(*(256, 128, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d722.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D723(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d723.weight_1",
            forge.Parameter(*(16, 256, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d723.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D724(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d724.weight_1",
            forge.Parameter(*(324, 1024, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d724.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D725(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d725.weight_1",
            forge.Parameter(*(486, 512, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d725.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D726(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d726.weight_1",
            forge.Parameter(*(486, 256, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d726.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D727(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d727.weight_1",
            forge.Parameter(*(324, 256, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d727.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D728(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d728.weight_1",
            forge.Parameter(*(32, 3, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d728.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D729(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d729.weight_1",
            forge.Parameter(*(128, 64, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d729.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D730(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d730.weight_1",
            forge.Parameter(*(32, 64, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d730.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D731(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d731.weight_1",
            forge.Parameter(*(1, 32, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d731.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D732(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d732.weight_1",
            forge.Parameter(*(256, 3072, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d732.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D733(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d733.weight_1",
            forge.Parameter(*(128, 768, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d733.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D734(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d734.weight_1",
            forge.Parameter(*(64, 384, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d734.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D735(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d735.weight_1",
            forge.Parameter(*(16, 32, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d735.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D736(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d736.weight_1",
            forge.Parameter(*(1, 16, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d736.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D737(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d737.weight_1",
            forge.Parameter(*(256, 1024, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d737.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D738(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d738.weight_1",
            forge.Parameter(*(128, 512, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d738.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D739(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d739.weight_1",
            forge.Parameter(*(19, 64, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d739.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D740(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d740.weight_1",
            forge.Parameter(*(4096, 512, 7, 7), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d740.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D741(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d741.weight_1",
            forge.Parameter(*(4096, 4096, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d741.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D742(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d742.weight_1",
            forge.Parameter(*(160, 256, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d742.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D743(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d743.weight_1",
            forge.Parameter(*(512, 1056, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d743.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D744(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d744.weight_1",
            forge.Parameter(*(192, 512, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d744.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D745(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d745.weight_1",
            forge.Parameter(*(768, 1472, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d745.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D746(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d746.weight_1",
            forge.Parameter(*(192, 768, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d746.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D747(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d747.weight_1",
            forge.Parameter(*(768, 1728, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d747.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D748(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d748.weight_1",
            forge.Parameter(*(224, 768, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d748.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D749(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d749.weight_1",
            forge.Parameter(*(224, 224, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d749.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D750(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d750.weight_1",
            forge.Parameter(*(1024, 1888, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d750.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D751(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d751.weight_1",
            forge.Parameter(*(224, 1024, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d751.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D752(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d752.weight_1",
            forge.Parameter(*(1024, 2144, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d752.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D753(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d753.weight_1",
            forge.Parameter(*(160, 512, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d753.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D754(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d754.weight_1",
            forge.Parameter(*(512, 1312, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d754.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D755(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d755.weight_1",
            forge.Parameter(*(80, 128, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d755.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D756(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d756.weight_1",
            forge.Parameter(*(256, 528, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d756.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D757(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d757.weight_1",
            forge.Parameter(*(96, 256, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d757.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D758(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d758.weight_1",
            forge.Parameter(*(384, 736, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d758.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D759(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d759.weight_1",
            forge.Parameter(*(112, 384, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d759.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D760(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d760.weight_1",
            forge.Parameter(*(112, 112, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d760.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D761(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d761.weight_1",
            forge.Parameter(*(512, 944, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d761.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D762(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d762.weight_1",
            forge.Parameter(*(256, 448, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d762.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D763(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d763.weight_1",
            forge.Parameter(*(160, 256, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d763.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D764(ForgeModule):
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
            groups=160,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D765(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d765.weight_1",
            forge.Parameter(*(160, 160, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d765.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D766(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d766.weight_1",
            forge.Parameter(*(512, 736, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d766.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D767(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d767.weight_1",
            forge.Parameter(*(192, 512, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d767.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D768(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d768.weight_1",
            forge.Parameter(*(768, 1088, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d768.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D769(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d769.weight_1",
            forge.Parameter(*(224, 768, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d769.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D770(ForgeModule):
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
            groups=224,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D771(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d771.weight_1",
            forge.Parameter(*(224, 224, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d771.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D772(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d772.weight_1",
            forge.Parameter(*(1024, 1440, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d772.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D773(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d773.weight_1",
            forge.Parameter(*(1024, 1024, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d773.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D774(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d774.weight_1",
            forge.Parameter(*(1024, 1024, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d774.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D775(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d775.weight_1",
            forge.Parameter(*(256, 256, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d775.weight_1"),
            stride=[2, 2],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D776(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d776.weight_1",
            forge.Parameter(*(728, 256, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d776.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D777(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d777.weight_1",
            forge.Parameter(*(728, 256, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d777.weight_1"),
            stride=[2, 2],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D778(ForgeModule):
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
            groups=728,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D779(ForgeModule):
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
            groups=728,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D780(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d780.weight_1",
            forge.Parameter(*(728, 728, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d780.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D781(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d781.weight_1",
            forge.Parameter(*(728, 728, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d781.weight_1"),
            stride=[2, 2],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D782(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d782.weight_1",
            forge.Parameter(*(1024, 728, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d782.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D783(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d783.weight_1",
            forge.Parameter(*(1024, 728, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d783.weight_1"),
            stride=[2, 2],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D784(ForgeModule):
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
            groups=1024,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D785(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d785.weight_1",
            forge.Parameter(*(1536, 1024, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d785.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D786(ForgeModule):
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
            groups=1536,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D787(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d787.weight_1",
            forge.Parameter(*(1536, 1536, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d787.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D788(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d788.weight_1",
            forge.Parameter(*(2048, 1536, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d788.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D789(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d789.weight_1",
            forge.Parameter(*(64, 32, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d789.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D790(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d790_const_1", shape=(80, 3, 6, 6), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d790_const_1"),
            stride=[2, 2],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D791(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d791_const_1", shape=(160, 80, 3, 3), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d791_const_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D792(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d792_const_1", shape=(80, 160, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d792_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D793(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d793_const_1", shape=(80, 80, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d793_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D794(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d794_const_1", shape=(80, 80, 3, 3), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d794_const_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D795(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d795_const_1", shape=(160, 160, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d795_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D796(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d796_const_1", shape=(320, 160, 3, 3), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d796_const_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D797(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d797_const_1", shape=(160, 320, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d797_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D798(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d798_const_1", shape=(160, 160, 3, 3), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d798_const_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D799(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d799_const_1", shape=(320, 320, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d799_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D800(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d800_const_1", shape=(640, 320, 3, 3), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d800_const_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D801(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d801_const_1", shape=(320, 640, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d801_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D802(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d802_const_1", shape=(320, 320, 3, 3), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d802_const_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D803(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d803_const_1", shape=(320, 320, 3, 3), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d803_const_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D804(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d804_const_1", shape=(640, 640, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d804_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D805(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d805_const_1", shape=(1280, 640, 3, 3), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d805_const_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D806(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d806_const_1", shape=(640, 1280, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d806_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D807(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d807_const_1", shape=(640, 640, 3, 3), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d807_const_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D808(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d808_const_1", shape=(640, 640, 3, 3), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d808_const_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D809(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d809_const_1", shape=(1280, 1280, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d809_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D810(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d810_const_1", shape=(1280, 2560, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d810_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D811(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d811_const_1", shape=(320, 1280, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d811_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D812(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d812_const_1", shape=(160, 640, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d812_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D813(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d813_const_1", shape=(255, 320, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d813_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D814(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d814_const_1", shape=(255, 640, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d814_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D815(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d815_const_1", shape=(255, 1280, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d815_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D816(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d816_const_1", shape=(48, 3, 6, 6), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d816_const_1"),
            stride=[2, 2],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D817(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d817_const_1", shape=(96, 48, 3, 3), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d817_const_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D818(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d818_const_1", shape=(48, 96, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d818_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D819(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d819_const_1", shape=(48, 48, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d819_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D820(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d820_const_1", shape=(48, 48, 3, 3), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d820_const_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D821(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d821_const_1", shape=(96, 96, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d821_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D822(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d822_const_1", shape=(192, 96, 3, 3), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d822_const_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D823(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d823_const_1", shape=(96, 192, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d823_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D824(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d824_const_1", shape=(96, 96, 3, 3), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d824_const_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D825(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d825_const_1", shape=(192, 192, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d825_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D826(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d826_const_1", shape=(384, 192, 3, 3), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d826_const_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D827(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d827_const_1", shape=(192, 384, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d827_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D828(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d828_const_1", shape=(192, 192, 3, 3), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d828_const_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D829(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d829_const_1", shape=(384, 384, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d829_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D830(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d830_const_1", shape=(768, 384, 3, 3), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d830_const_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D831(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d831_const_1", shape=(384, 768, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d831_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D832(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d832_const_1", shape=(384, 384, 3, 3), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d832_const_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D833(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d833_const_1", shape=(768, 768, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d833_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D834(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d834_const_1", shape=(768, 1536, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d834_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D835(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d835_const_1", shape=(192, 768, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d835_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D836(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d836_const_1", shape=(96, 384, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d836_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D837(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d837_const_1", shape=(255, 192, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d837_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D838(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d838_const_1", shape=(192, 192, 3, 3), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d838_const_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D839(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d839_const_1", shape=(255, 384, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d839_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D840(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d840_const_1", shape=(384, 384, 3, 3), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d840_const_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D841(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d841_const_1", shape=(255, 768, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d841_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D842(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d842_const_1", shape=(32, 3, 6, 6), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d842_const_1"),
            stride=[2, 2],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D843(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d843_const_1", shape=(64, 32, 3, 3), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d843_const_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D844(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d844_const_1", shape=(32, 64, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d844_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D845(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d845_const_1", shape=(32, 32, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d845_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D846(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d846_const_1", shape=(32, 32, 3, 3), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d846_const_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D847(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d847_const_1", shape=(64, 64, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d847_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D848(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d848_const_1", shape=(128, 64, 3, 3), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d848_const_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D849(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d849_const_1", shape=(64, 128, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d849_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D850(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d850_const_1", shape=(64, 64, 3, 3), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d850_const_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D851(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d851_const_1", shape=(64, 64, 3, 3), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d851_const_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D852(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d852_const_1", shape=(128, 128, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d852_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D853(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d853_const_1", shape=(256, 128, 3, 3), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d853_const_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D854(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d854_const_1", shape=(128, 256, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d854_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D855(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d855_const_1", shape=(128, 128, 3, 3), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d855_const_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D856(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d856_const_1", shape=(128, 128, 3, 3), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d856_const_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D857(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d857_const_1", shape=(256, 256, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d857_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D858(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d858_const_1", shape=(512, 256, 3, 3), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d858_const_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D859(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d859_const_1", shape=(256, 512, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d859_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D860(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d860_const_1", shape=(256, 256, 3, 3), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d860_const_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D861(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d861_const_1", shape=(512, 512, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d861_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D862(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d862_const_1", shape=(512, 1024, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d862_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D863(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d863_const_1", shape=(128, 512, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d863_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D864(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d864_const_1", shape=(64, 256, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d864_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D865(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d865_const_1", shape=(255, 128, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d865_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D866(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d866_const_1", shape=(255, 256, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d866_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D867(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d867_const_1", shape=(255, 512, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d867_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D868(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d868_const_1", shape=(64, 3, 6, 6), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d868_const_1"),
            stride=[2, 2],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D869(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d869_const_1", shape=(1024, 512, 3, 3), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d869_const_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D870(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d870_const_1", shape=(512, 512, 3, 3), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d870_const_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D871(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d871_const_1", shape=(1024, 1024, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d871_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D872(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d872_const_1", shape=(1024, 2048, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d872_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D873(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d873_const_1", shape=(256, 1024, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d873_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D874(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d874_const_1", shape=(512, 512, 3, 3), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d874_const_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D875(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d875_const_1", shape=(255, 1024, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d875_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D876(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d876_const_1", shape=(16, 3, 6, 6), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d876_const_1"),
            stride=[2, 2],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D877(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d877_const_1", shape=(32, 16, 3, 3), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d877_const_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D878(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d878_const_1", shape=(16, 32, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d878_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D879(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d879_const_1", shape=(16, 16, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d879_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D880(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d880_const_1", shape=(16, 16, 3, 3), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d880_const_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D881(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d881_const_1", shape=(32, 128, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d881_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D882(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d882_const_1", shape=(255, 64, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d882_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D883(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d883_const_1", shape=(16, 3, 3, 3), dtype=torch.float32)
        self.add_constant("conv2d883_const_2", shape=(16,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d883_const_1"),
            self.get_constant("conv2d883_const_2"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D884(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d884_const_1", shape=(32, 16, 3, 3), dtype=torch.float32)
        self.add_constant("conv2d884_const_2", shape=(32,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d884_const_1"),
            self.get_constant("conv2d884_const_2"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D885(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d885_const_1", shape=(32, 32, 3, 3), dtype=torch.float32)
        self.add_constant("conv2d885_const_2", shape=(32,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d885_const_1"),
            self.get_constant("conv2d885_const_2"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D886(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d886_const_1", shape=(32, 32, 3, 3), dtype=torch.float32)
        self.add_constant("conv2d886_const_2", shape=(32,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d886_const_1"),
            self.get_constant("conv2d886_const_2"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D887(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d887_const_1", shape=(64, 32, 3, 3), dtype=torch.float32)
        self.add_constant("conv2d887_const_2", shape=(64,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d887_const_1"),
            self.get_constant("conv2d887_const_2"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D888(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d888_const_1", shape=(64, 64, 3, 3), dtype=torch.float32)
        self.add_constant("conv2d888_const_2", shape=(64,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d888_const_1"),
            self.get_constant("conv2d888_const_2"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D889(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d889_const_1", shape=(64, 64, 3, 3), dtype=torch.float32)
        self.add_constant("conv2d889_const_2", shape=(64,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d889_const_1"),
            self.get_constant("conv2d889_const_2"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D890(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d890_const_1", shape=(128, 64, 3, 3), dtype=torch.float32)
        self.add_constant("conv2d890_const_2", shape=(128,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d890_const_1"),
            self.get_constant("conv2d890_const_2"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D891(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d891_const_1", shape=(128, 128, 3, 3), dtype=torch.float32)
        self.add_constant("conv2d891_const_2", shape=(128,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d891_const_1"),
            self.get_constant("conv2d891_const_2"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D892(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d892_const_1", shape=(128, 128, 3, 3), dtype=torch.float32)
        self.add_constant("conv2d892_const_2", shape=(128,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d892_const_1"),
            self.get_constant("conv2d892_const_2"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D893(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d893_const_1", shape=(256, 128, 3, 3), dtype=torch.float32)
        self.add_constant("conv2d893_const_2", shape=(256,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d893_const_1"),
            self.get_constant("conv2d893_const_2"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D894(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d894_const_1", shape=(256, 256, 3, 3), dtype=torch.float32)
        self.add_constant("conv2d894_const_2", shape=(256,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d894_const_1"),
            self.get_constant("conv2d894_const_2"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D895(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d895_const_1", shape=(128, 256, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d895_const_2", shape=(128,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d895_const_1"),
            self.get_constant("conv2d895_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D896(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d896_const_1", shape=(128, 128, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d896_const_2", shape=(128,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d896_const_1"),
            self.get_constant("conv2d896_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D897(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d897_const_1", shape=(128, 512, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d897_const_2", shape=(128,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d897_const_1"),
            self.get_constant("conv2d897_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D898(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d898_const_1", shape=(256, 256, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d898_const_2", shape=(256,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d898_const_1"),
            self.get_constant("conv2d898_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D899(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d899_const_1", shape=(64, 256, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d899_const_2", shape=(64,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d899_const_1"),
            self.get_constant("conv2d899_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D900(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d900_const_1", shape=(64, 128, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d900_const_2", shape=(64,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d900_const_1"),
            self.get_constant("conv2d900_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D901(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d901_const_1", shape=(64, 64, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d901_const_2", shape=(64,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d901_const_1"),
            self.get_constant("conv2d901_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D902(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d902_const_1", shape=(64, 192, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d902_const_2", shape=(64,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d902_const_1"),
            self.get_constant("conv2d902_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D903(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d903_const_1", shape=(32, 64, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d903_const_2", shape=(32,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d903_const_1"),
            self.get_constant("conv2d903_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D904(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d904_const_1", shape=(32, 32, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d904_const_2", shape=(32,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d904_const_1"),
            self.get_constant("conv2d904_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D905(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d905_const_1", shape=(32, 96, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d905_const_2", shape=(32,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d905_const_1"),
            self.get_constant("conv2d905_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D906(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d906_const_1", shape=(4, 32, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d906_const_2", shape=(4,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d906_const_1"),
            self.get_constant("conv2d906_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D907(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d907_const_1", shape=(4, 64, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d907_const_2", shape=(4,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d907_const_1"),
            self.get_constant("conv2d907_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D908(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d908_const_1", shape=(4, 128, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d908_const_2", shape=(4,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d908_const_1"),
            self.get_constant("conv2d908_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D909(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d909_const_1", shape=(80, 32, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d909_const_2", shape=(80,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d909_const_1"),
            self.get_constant("conv2d909_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D910(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d910_const_1", shape=(80, 64, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d910_const_2", shape=(80,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d910_const_1"),
            self.get_constant("conv2d910_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D911(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d911_const_1", shape=(80, 128, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d911_const_2", shape=(80,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d911_const_1"),
            self.get_constant("conv2d911_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D912(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d912_const_1", shape=(48, 3, 3, 3), dtype=torch.float32)
        self.add_constant("conv2d912_const_2", shape=(48,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d912_const_1"),
            self.get_constant("conv2d912_const_2"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D913(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d913_const_1", shape=(96, 48, 3, 3), dtype=torch.float32)
        self.add_constant("conv2d913_const_2", shape=(96,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d913_const_1"),
            self.get_constant("conv2d913_const_2"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D914(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d914_const_1", shape=(64, 96, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d914_const_2", shape=(64,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d914_const_1"),
            self.get_constant("conv2d914_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D915(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d915_const_1", shape=(96, 128, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d915_const_2", shape=(96,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d915_const_1"),
            self.get_constant("conv2d915_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D916(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d916_const_1", shape=(192, 96, 3, 3), dtype=torch.float32)
        self.add_constant("conv2d916_const_2", shape=(192,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d916_const_1"),
            self.get_constant("conv2d916_const_2"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D917(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d917_const_1", shape=(128, 192, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d917_const_2", shape=(128,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d917_const_1"),
            self.get_constant("conv2d917_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D918(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d918_const_1", shape=(192, 256, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d918_const_2", shape=(192,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d918_const_1"),
            self.get_constant("conv2d918_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D919(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d919_const_1", shape=(384, 192, 3, 3), dtype=torch.float32)
        self.add_constant("conv2d919_const_2", shape=(384,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d919_const_1"),
            self.get_constant("conv2d919_const_2"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D920(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d920_const_1", shape=(256, 384, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d920_const_2", shape=(256,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d920_const_1"),
            self.get_constant("conv2d920_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D921(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d921_const_1", shape=(256, 256, 3, 3), dtype=torch.float32)
        self.add_constant("conv2d921_const_2", shape=(256,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d921_const_1"),
            self.get_constant("conv2d921_const_2"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D922(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d922_const_1", shape=(384, 512, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d922_const_2", shape=(384,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d922_const_1"),
            self.get_constant("conv2d922_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D923(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d923_const_1", shape=(768, 384, 3, 3), dtype=torch.float32)
        self.add_constant("conv2d923_const_2", shape=(768,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d923_const_1"),
            self.get_constant("conv2d923_const_2"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D924(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d924_const_1", shape=(512, 768, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d924_const_2", shape=(512,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d924_const_1"),
            self.get_constant("conv2d924_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D925(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d925_const_1", shape=(512, 512, 3, 3), dtype=torch.float32)
        self.add_constant("conv2d925_const_2", shape=(512,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d925_const_1"),
            self.get_constant("conv2d925_const_2"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D926(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d926_const_1", shape=(768, 1024, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d926_const_2", shape=(768,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d926_const_1"),
            self.get_constant("conv2d926_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D927(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d927_const_1", shape=(384, 768, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d927_const_2", shape=(384,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d927_const_1"),
            self.get_constant("conv2d927_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D928(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d928_const_1", shape=(768, 1536, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d928_const_2", shape=(768,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d928_const_1"),
            self.get_constant("conv2d928_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D929(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d929_const_1", shape=(192, 768, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d929_const_2", shape=(192,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d929_const_1"),
            self.get_constant("conv2d929_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D930(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d930_const_1", shape=(192, 384, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d930_const_2", shape=(192,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d930_const_1"),
            self.get_constant("conv2d930_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D931(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d931_const_1", shape=(192, 192, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d931_const_2", shape=(192,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d931_const_1"),
            self.get_constant("conv2d931_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D932(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d932_const_1", shape=(192, 192, 3, 3), dtype=torch.float32)
        self.add_constant("conv2d932_const_2", shape=(192,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d932_const_1"),
            self.get_constant("conv2d932_const_2"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D933(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d933_const_1", shape=(192, 576, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d933_const_2", shape=(192,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d933_const_1"),
            self.get_constant("conv2d933_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D934(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d934_const_1", shape=(96, 192, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d934_const_2", shape=(96,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d934_const_1"),
            self.get_constant("conv2d934_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D935(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d935_const_1", shape=(96, 96, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d935_const_2", shape=(96,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d935_const_1"),
            self.get_constant("conv2d935_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D936(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d936_const_1", shape=(96, 96, 3, 3), dtype=torch.float32)
        self.add_constant("conv2d936_const_2", shape=(96,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d936_const_1"),
            self.get_constant("conv2d936_const_2"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D937(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d937_const_1", shape=(96, 288, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d937_const_2", shape=(96,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d937_const_1"),
            self.get_constant("conv2d937_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D938(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d938_const_1", shape=(96, 96, 3, 3), dtype=torch.float32)
        self.add_constant("conv2d938_const_2", shape=(96,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d938_const_1"),
            self.get_constant("conv2d938_const_2"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D939(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d939_const_1", shape=(68, 96, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d939_const_2", shape=(68,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d939_const_1"),
            self.get_constant("conv2d939_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D940(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d940_const_1", shape=(1, 17, 1, 1), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d940_const_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D941(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d941_const_1", shape=(192, 192, 3, 3), dtype=torch.float32)
        self.add_constant("conv2d941_const_2", shape=(192,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d941_const_1"),
            self.get_constant("conv2d941_const_2"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D942(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d942_const_1", shape=(68, 192, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d942_const_2", shape=(68,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d942_const_1"),
            self.get_constant("conv2d942_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D943(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d943_const_1", shape=(384, 384, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d943_const_2", shape=(384,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d943_const_1"),
            self.get_constant("conv2d943_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D944(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d944_const_1", shape=(384, 384, 3, 3), dtype=torch.float32)
        self.add_constant("conv2d944_const_2", shape=(384,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d944_const_1"),
            self.get_constant("conv2d944_const_2"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D945(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d945_const_1", shape=(68, 384, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d945_const_2", shape=(68,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d945_const_1"),
            self.get_constant("conv2d945_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D946(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d946_const_1", shape=(80, 96, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d946_const_2", shape=(80,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d946_const_1"),
            self.get_constant("conv2d946_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D947(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d947_const_1", shape=(80, 192, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d947_const_2", shape=(80,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d947_const_1"),
            self.get_constant("conv2d947_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D948(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d948_const_1", shape=(80, 384, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d948_const_2", shape=(80,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d948_const_1"),
            self.get_constant("conv2d948_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D949(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d949_const_1", shape=(32, 3, 3, 3), dtype=torch.float32)
        self.add_constant("conv2d949_const_2", shape=(32,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d949_const_1"),
            self.get_constant("conv2d949_const_2"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D950(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d950_const_1", shape=(512, 256, 3, 3), dtype=torch.float32)
        self.add_constant("conv2d950_const_2", shape=(512,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d950_const_1"),
            self.get_constant("conv2d950_const_2"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D951(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d951_const_1", shape=(256, 512, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d951_const_2", shape=(256,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d951_const_1"),
            self.get_constant("conv2d951_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D952(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d952_const_1", shape=(256, 1024, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d952_const_2", shape=(256,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d952_const_1"),
            self.get_constant("conv2d952_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D953(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d953_const_1", shape=(512, 512, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d953_const_2", shape=(512,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d953_const_1"),
            self.get_constant("conv2d953_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D954(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d954_const_1", shape=(128, 384, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d954_const_2", shape=(128,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d954_const_1"),
            self.get_constant("conv2d954_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D955(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d955_const_1", shape=(4, 256, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d955_const_2", shape=(4,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d955_const_1"),
            self.get_constant("conv2d955_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D956(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d956_const_1", shape=(80, 256, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d956_const_2", shape=(80,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d956_const_1"),
            self.get_constant("conv2d956_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D957(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d957_const_1", shape=(64, 3, 3, 3), dtype=torch.float32)
        self.add_constant("conv2d957_const_2", shape=(64,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d957_const_1"),
            self.get_constant("conv2d957_const_2"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D958(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d958_const_1", shape=(1024, 512, 3, 3), dtype=torch.float32)
        self.add_constant("conv2d958_const_2", shape=(1024,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d958_const_1"),
            self.get_constant("conv2d958_const_2"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D959(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d959_const_1", shape=(512, 1024, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d959_const_2", shape=(512,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d959_const_1"),
            self.get_constant("conv2d959_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D960(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d960_const_1", shape=(1024, 1024, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d960_const_2", shape=(1024,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d960_const_1"),
            self.get_constant("conv2d960_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D961(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d961_const_1", shape=(1024, 2048, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d961_const_2", shape=(1024,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d961_const_1"),
            self.get_constant("conv2d961_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D962(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d962_const_1", shape=(256, 768, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d962_const_2", shape=(256,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d962_const_1"),
            self.get_constant("conv2d962_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D963(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d963_const_1", shape=(68, 128, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d963_const_2", shape=(68,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d963_const_1"),
            self.get_constant("conv2d963_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D964(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d964_const_1", shape=(68, 256, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d964_const_2", shape=(68,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d964_const_1"),
            self.get_constant("conv2d964_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D965(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d965_const_1", shape=(68, 512, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d965_const_2", shape=(68,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d965_const_1"),
            self.get_constant("conv2d965_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D966(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("conv2d966_const_1", shape=(80, 512, 1, 1), dtype=torch.float32)
        self.add_constant("conv2d966_const_2", shape=(80,), dtype=torch.float32)

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_constant("conv2d966_const_1"),
            self.get_constant("conv2d966_const_2"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D967(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d967.weight_1",
            forge.Parameter(*(48, 12, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d967.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D968(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d968.weight_1",
            forge.Parameter(*(48, 48, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d968.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D969(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d969.weight_1",
            forge.Parameter(*(768, 384, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d969.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D970(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d970.weight_1",
            forge.Parameter(*(768, 1536, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d970.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D971(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d971.weight_1",
            forge.Parameter(*(192, 192, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d971.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D972(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d972.weight_1",
            forge.Parameter(*(4, 192, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d972.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D973(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d973.weight_1",
            forge.Parameter(*(1, 192, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d973.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D974(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d974.weight_1",
            forge.Parameter(*(80, 192, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d974.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D975(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d975.weight_1",
            forge.Parameter(*(384, 384, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d975.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D976(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d976.weight_1",
            forge.Parameter(*(16, 12, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d976.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D977(ForgeModule):
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
            groups=32,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D978(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d978.weight_1",
            forge.Parameter(*(4, 64, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d978.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D979(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d979.weight_1",
            forge.Parameter(*(1, 64, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d979.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D980(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d980.weight_1",
            forge.Parameter(*(80, 64, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d980.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D981(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d981.weight_1",
            forge.Parameter(*(1024, 512, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d981.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D982(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d982.weight_1",
            forge.Parameter(*(4, 256, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d982.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D983(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d983.weight_1",
            forge.Parameter(*(1, 256, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d983.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D984(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d984.weight_1",
            forge.Parameter(*(80, 256, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d984.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D985(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d985.weight_1",
            forge.Parameter(*(64, 12, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d985.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D986(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d986.weight_1",
            forge.Parameter(*(80, 12, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d986.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D987(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d987.weight_1",
            forge.Parameter(*(80, 80, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d987.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D988(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d988.weight_1",
            forge.Parameter(*(160, 320, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d988.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D989(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d989.weight_1",
            forge.Parameter(*(320, 320, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d989.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D990(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d990.weight_1",
            forge.Parameter(*(640, 320, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d990.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D991(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d991.weight_1",
            forge.Parameter(*(320, 640, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d991.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D992(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d992.weight_1",
            forge.Parameter(*(640, 640, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d992.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D993(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d993.weight_1",
            forge.Parameter(*(1280, 640, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d993.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D994(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d994.weight_1",
            forge.Parameter(*(1280, 2560, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d994.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D995(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d995.weight_1",
            forge.Parameter(*(640, 640, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d995.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D996(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d996.weight_1",
            forge.Parameter(*(1280, 1280, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d996.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D997(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d997.weight_1",
            forge.Parameter(*(320, 1280, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d997.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D998(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d998.weight_1",
            forge.Parameter(*(160, 640, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d998.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D999(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d999.weight_1",
            forge.Parameter(*(320, 320, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d999.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D1000(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d1000.weight_1",
            forge.Parameter(*(4, 320, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d1000.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D1001(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d1001.weight_1",
            forge.Parameter(*(1, 320, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d1001.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D1002(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d1002.weight_1",
            forge.Parameter(*(80, 320, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d1002.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D1003(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d1003.weight_1",
            forge.Parameter(*(640, 640, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d1003.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D1004(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d1004.weight_1",
            forge.Parameter(*(32, 12, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d1004.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D1005(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d1005.weight_1",
            forge.Parameter(*(4, 128, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d1005.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D1006(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d1006.weight_1",
            forge.Parameter(*(1, 128, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d1006.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D1007(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d1007.weight_1",
            forge.Parameter(*(80, 128, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d1007.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D1008(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d1008.weight_1",
            forge.Parameter(*(24, 12, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d1008.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D1009(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d1009.weight_1",
            forge.Parameter(*(48, 24, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d1009.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D1010(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d1010.weight_1",
            forge.Parameter(*(24, 24, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d1010.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D1011(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d1011.weight_1",
            forge.Parameter(*(1, 96, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d1011.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D1012(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d1012.weight_1",
            forge.Parameter(*(80, 96, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d1012.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (Conv2D0, [((1, 768, 1, 128), torch.float32), ((768, 192, 1, 1), torch.float32)]),
    (Conv2D1, [((1, 768, 128, 1), torch.float32), ((768, 768, 1, 1), torch.float32)]),
    (Conv2D0, [((1, 768, 1, 128), torch.float32), ((3072, 192, 1, 1), torch.float32)]),
    (Conv2D0, [((1, 3072, 1, 128), torch.float32), ((768, 768, 1, 1), torch.float32)]),
    (Conv2D2, [((1, 3, 224, 224), torch.float32)]),
    (Conv2D3, [((1, 64, 27, 27), torch.float32)]),
    (Conv2D4, [((1, 192, 13, 13), torch.float32)]),
    (Conv2D5, [((1, 384, 13, 13), torch.float32)]),
    (Conv2D6, [((1, 256, 13, 13), torch.float32)]),
    (Conv2D7, [((1, 1, 28, 28), torch.float32)]),
    (Conv2D8, [((1, 16, 14, 14), torch.float32)]),
    (Conv2D9, [((1, 3, 224, 224), torch.float32)]),
    (Conv2D10, [((1, 3, 224, 224), torch.float32)]),
    (Conv2D11, [((1, 3, 224, 224), torch.float32)]),
    (Conv2D12, [((1, 3, 224, 224), torch.float32)]),
    (Conv2D13, [((1, 64, 56, 56), torch.float32)]),
    (Conv2D14, [((1, 128, 56, 56), torch.float32)]),
    (Conv2D15, [((1, 128, 56, 56), torch.float32)]),
    (Conv2D16, [((1, 96, 56, 56), torch.float32)]),
    (Conv2D17, [((1, 128, 56, 56), torch.float32)]),
    (Conv2D18, [((1, 160, 56, 56), torch.float32)]),
    (Conv2D19, [((1, 192, 56, 56), torch.float32)]),
    (Conv2D20, [((1, 224, 56, 56), torch.float32)]),
    (Conv2D21, [((1, 256, 56, 56), torch.float32)]),
    (Conv2D17, [((1, 128, 28, 28), torch.float32)]),
    (Conv2D14, [((1, 128, 28, 28), torch.float32)]),
    (Conv2D18, [((1, 160, 28, 28), torch.float32)]),
    (Conv2D19, [((1, 192, 28, 28), torch.float32)]),
    (Conv2D20, [((1, 224, 28, 28), torch.float32)]),
    (Conv2D21, [((1, 256, 28, 28), torch.float32)]),
    (Conv2D22, [((1, 288, 28, 28), torch.float32)]),
    (Conv2D23, [((1, 320, 28, 28), torch.float32)]),
    (Conv2D24, [((1, 352, 28, 28), torch.float32)]),
    (Conv2D25, [((1, 384, 28, 28), torch.float32)]),
    (Conv2D26, [((1, 416, 28, 28), torch.float32)]),
    (Conv2D27, [((1, 448, 28, 28), torch.float32)]),
    (Conv2D28, [((1, 480, 28, 28), torch.float32)]),
    (Conv2D29, [((1, 512, 28, 28), torch.float32)]),
    (Conv2D21, [((1, 256, 14, 14), torch.float32)]),
    (Conv2D14, [((1, 128, 14, 14), torch.float32)]),
    (Conv2D22, [((1, 288, 14, 14), torch.float32)]),
    (Conv2D23, [((1, 320, 14, 14), torch.float32)]),
    (Conv2D24, [((1, 352, 14, 14), torch.float32)]),
    (Conv2D25, [((1, 384, 14, 14), torch.float32)]),
    (Conv2D26, [((1, 416, 14, 14), torch.float32)]),
    (Conv2D27, [((1, 448, 14, 14), torch.float32)]),
    (Conv2D28, [((1, 480, 14, 14), torch.float32)]),
    (Conv2D30, [((1, 512, 14, 14), torch.float32)]),
    (Conv2D31, [((1, 544, 14, 14), torch.float32)]),
    (Conv2D32, [((1, 576, 14, 14), torch.float32)]),
    (Conv2D33, [((1, 608, 14, 14), torch.float32)]),
    (Conv2D34, [((1, 640, 14, 14), torch.float32)]),
    (Conv2D35, [((1, 672, 14, 14), torch.float32)]),
    (Conv2D36, [((1, 704, 14, 14), torch.float32)]),
    (Conv2D37, [((1, 736, 14, 14), torch.float32)]),
    (Conv2D38, [((1, 768, 14, 14), torch.float32)]),
    (Conv2D39, [((1, 800, 14, 14), torch.float32)]),
    (Conv2D40, [((1, 832, 14, 14), torch.float32)]),
    (Conv2D41, [((1, 864, 14, 14), torch.float32)]),
    (Conv2D42, [((1, 896, 14, 14), torch.float32)]),
    (Conv2D43, [((1, 928, 14, 14), torch.float32)]),
    (Conv2D44, [((1, 960, 14, 14), torch.float32)]),
    (Conv2D45, [((1, 992, 14, 14), torch.float32)]),
    (Conv2D46, [((1, 1024, 14, 14), torch.float32)]),
    (Conv2D47, [((1, 1056, 14, 14), torch.float32)]),
    (Conv2D48, [((1, 1088, 14, 14), torch.float32)]),
    (Conv2D49, [((1, 1120, 14, 14), torch.float32)]),
    (Conv2D50, [((1, 1152, 14, 14), torch.float32)]),
    (Conv2D51, [((1, 1184, 14, 14), torch.float32)]),
    (Conv2D52, [((1, 1216, 14, 14), torch.float32)]),
    (Conv2D53, [((1, 1248, 14, 14), torch.float32)]),
    (Conv2D54, [((1, 1280, 14, 14), torch.float32)]),
    (Conv2D55, [((1, 1312, 14, 14), torch.float32)]),
    (Conv2D56, [((1, 1344, 14, 14), torch.float32)]),
    (Conv2D57, [((1, 1376, 14, 14), torch.float32)]),
    (Conv2D58, [((1, 1408, 14, 14), torch.float32)]),
    (Conv2D59, [((1, 1440, 14, 14), torch.float32)]),
    (Conv2D60, [((1, 1472, 14, 14), torch.float32)]),
    (Conv2D61, [((1, 1504, 14, 14), torch.float32)]),
    (Conv2D62, [((1, 1536, 14, 14), torch.float32)]),
    (Conv2D63, [((1, 1568, 14, 14), torch.float32)]),
    (Conv2D64, [((1, 1600, 14, 14), torch.float32)]),
    (Conv2D65, [((1, 1632, 14, 14), torch.float32)]),
    (Conv2D66, [((1, 1664, 14, 14), torch.float32)]),
    (Conv2D67, [((1, 1696, 14, 14), torch.float32)]),
    (Conv2D68, [((1, 1728, 14, 14), torch.float32)]),
    (Conv2D69, [((1, 1760, 14, 14), torch.float32)]),
    (Conv2D70, [((1, 1792, 14, 14), torch.float32)]),
    (Conv2D42, [((1, 896, 7, 7), torch.float32)]),
    (Conv2D14, [((1, 128, 7, 7), torch.float32)]),
    (Conv2D43, [((1, 928, 7, 7), torch.float32)]),
    (Conv2D44, [((1, 960, 7, 7), torch.float32)]),
    (Conv2D45, [((1, 992, 7, 7), torch.float32)]),
    (Conv2D46, [((1, 1024, 7, 7), torch.float32)]),
    (Conv2D47, [((1, 1056, 7, 7), torch.float32)]),
    (Conv2D48, [((1, 1088, 7, 7), torch.float32)]),
    (Conv2D49, [((1, 1120, 7, 7), torch.float32)]),
    (Conv2D50, [((1, 1152, 7, 7), torch.float32)]),
    (Conv2D51, [((1, 1184, 7, 7), torch.float32)]),
    (Conv2D52, [((1, 1216, 7, 7), torch.float32)]),
    (Conv2D53, [((1, 1248, 7, 7), torch.float32)]),
    (Conv2D54, [((1, 1280, 7, 7), torch.float32)]),
    (Conv2D55, [((1, 1312, 7, 7), torch.float32)]),
    (Conv2D56, [((1, 1344, 7, 7), torch.float32)]),
    (Conv2D57, [((1, 1376, 7, 7), torch.float32)]),
    (Conv2D58, [((1, 1408, 7, 7), torch.float32)]),
    (Conv2D59, [((1, 1440, 7, 7), torch.float32)]),
    (Conv2D60, [((1, 1472, 7, 7), torch.float32)]),
    (Conv2D61, [((1, 1504, 7, 7), torch.float32)]),
    (Conv2D62, [((1, 1536, 7, 7), torch.float32)]),
    (Conv2D63, [((1, 1568, 7, 7), torch.float32)]),
    (Conv2D64, [((1, 1600, 7, 7), torch.float32)]),
    (Conv2D65, [((1, 1632, 7, 7), torch.float32)]),
    (Conv2D66, [((1, 1664, 7, 7), torch.float32)]),
    (Conv2D67, [((1, 1696, 7, 7), torch.float32)]),
    (Conv2D68, [((1, 1728, 7, 7), torch.float32)]),
    (Conv2D69, [((1, 1760, 7, 7), torch.float32)]),
    (Conv2D71, [((1, 1792, 7, 7), torch.float32)]),
    (Conv2D72, [((1, 1824, 7, 7), torch.float32)]),
    (Conv2D73, [((1, 1856, 7, 7), torch.float32)]),
    (Conv2D74, [((1, 1888, 7, 7), torch.float32)]),
    (Conv2D75, [((1, 1024, 14, 14), torch.float32)]),
    (Conv2D30, [((1, 512, 7, 7), torch.float32)]),
    (Conv2D31, [((1, 544, 7, 7), torch.float32)]),
    (Conv2D32, [((1, 576, 7, 7), torch.float32)]),
    (Conv2D33, [((1, 608, 7, 7), torch.float32)]),
    (Conv2D34, [((1, 640, 7, 7), torch.float32)]),
    (Conv2D35, [((1, 672, 7, 7), torch.float32)]),
    (Conv2D36, [((1, 704, 7, 7), torch.float32)]),
    (Conv2D37, [((1, 736, 7, 7), torch.float32)]),
    (Conv2D38, [((1, 768, 7, 7), torch.float32)]),
    (Conv2D39, [((1, 800, 7, 7), torch.float32)]),
    (Conv2D40, [((1, 832, 7, 7), torch.float32)]),
    (Conv2D41, [((1, 864, 7, 7), torch.float32)]),
    (Conv2D76, [((1, 3, 224, 224), torch.float32)]),
    (Conv2D77, [((1, 96, 56, 56), torch.float32)]),
    (Conv2D78, [((1, 192, 56, 56), torch.float32)]),
    (Conv2D79, [((1, 144, 56, 56), torch.float32)]),
    (Conv2D80, [((1, 192, 56, 56), torch.float32)]),
    (Conv2D81, [((1, 240, 56, 56), torch.float32)]),
    (Conv2D82, [((1, 288, 56, 56), torch.float32)]),
    (Conv2D83, [((1, 336, 56, 56), torch.float32)]),
    (Conv2D84, [((1, 384, 56, 56), torch.float32)]),
    (Conv2D80, [((1, 192, 28, 28), torch.float32)]),
    (Conv2D78, [((1, 192, 28, 28), torch.float32)]),
    (Conv2D81, [((1, 240, 28, 28), torch.float32)]),
    (Conv2D82, [((1, 288, 28, 28), torch.float32)]),
    (Conv2D83, [((1, 336, 28, 28), torch.float32)]),
    (Conv2D84, [((1, 384, 28, 28), torch.float32)]),
    (Conv2D85, [((1, 432, 28, 28), torch.float32)]),
    (Conv2D86, [((1, 480, 28, 28), torch.float32)]),
    (Conv2D87, [((1, 528, 28, 28), torch.float32)]),
    (Conv2D88, [((1, 576, 28, 28), torch.float32)]),
    (Conv2D89, [((1, 624, 28, 28), torch.float32)]),
    (Conv2D90, [((1, 672, 28, 28), torch.float32)]),
    (Conv2D91, [((1, 720, 28, 28), torch.float32)]),
    (Conv2D92, [((1, 768, 28, 28), torch.float32)]),
    (Conv2D84, [((1, 384, 14, 14), torch.float32)]),
    (Conv2D78, [((1, 192, 14, 14), torch.float32)]),
    (Conv2D85, [((1, 432, 14, 14), torch.float32)]),
    (Conv2D86, [((1, 480, 14, 14), torch.float32)]),
    (Conv2D87, [((1, 528, 14, 14), torch.float32)]),
    (Conv2D88, [((1, 576, 14, 14), torch.float32)]),
    (Conv2D89, [((1, 624, 14, 14), torch.float32)]),
    (Conv2D90, [((1, 672, 14, 14), torch.float32)]),
    (Conv2D91, [((1, 720, 14, 14), torch.float32)]),
    (Conv2D93, [((1, 768, 14, 14), torch.float32)]),
    (Conv2D94, [((1, 816, 14, 14), torch.float32)]),
    (Conv2D95, [((1, 864, 14, 14), torch.float32)]),
    (Conv2D96, [((1, 912, 14, 14), torch.float32)]),
    (Conv2D97, [((1, 960, 14, 14), torch.float32)]),
    (Conv2D98, [((1, 1008, 14, 14), torch.float32)]),
    (Conv2D99, [((1, 1056, 14, 14), torch.float32)]),
    (Conv2D100, [((1, 1104, 14, 14), torch.float32)]),
    (Conv2D101, [((1, 1152, 14, 14), torch.float32)]),
    (Conv2D102, [((1, 1200, 14, 14), torch.float32)]),
    (Conv2D103, [((1, 1248, 14, 14), torch.float32)]),
    (Conv2D104, [((1, 1296, 14, 14), torch.float32)]),
    (Conv2D105, [((1, 1344, 14, 14), torch.float32)]),
    (Conv2D106, [((1, 1392, 14, 14), torch.float32)]),
    (Conv2D107, [((1, 1440, 14, 14), torch.float32)]),
    (Conv2D108, [((1, 1488, 14, 14), torch.float32)]),
    (Conv2D109, [((1, 1536, 14, 14), torch.float32)]),
    (Conv2D110, [((1, 1584, 14, 14), torch.float32)]),
    (Conv2D111, [((1, 1632, 14, 14), torch.float32)]),
    (Conv2D112, [((1, 1680, 14, 14), torch.float32)]),
    (Conv2D113, [((1, 1728, 14, 14), torch.float32)]),
    (Conv2D114, [((1, 1776, 14, 14), torch.float32)]),
    (Conv2D115, [((1, 1824, 14, 14), torch.float32)]),
    (Conv2D116, [((1, 1872, 14, 14), torch.float32)]),
    (Conv2D117, [((1, 1920, 14, 14), torch.float32)]),
    (Conv2D118, [((1, 1968, 14, 14), torch.float32)]),
    (Conv2D119, [((1, 2016, 14, 14), torch.float32)]),
    (Conv2D120, [((1, 2064, 14, 14), torch.float32)]),
    (Conv2D121, [((1, 2112, 14, 14), torch.float32)]),
    (Conv2D99, [((1, 1056, 7, 7), torch.float32)]),
    (Conv2D78, [((1, 192, 7, 7), torch.float32)]),
    (Conv2D100, [((1, 1104, 7, 7), torch.float32)]),
    (Conv2D101, [((1, 1152, 7, 7), torch.float32)]),
    (Conv2D102, [((1, 1200, 7, 7), torch.float32)]),
    (Conv2D103, [((1, 1248, 7, 7), torch.float32)]),
    (Conv2D104, [((1, 1296, 7, 7), torch.float32)]),
    (Conv2D105, [((1, 1344, 7, 7), torch.float32)]),
    (Conv2D106, [((1, 1392, 7, 7), torch.float32)]),
    (Conv2D107, [((1, 1440, 7, 7), torch.float32)]),
    (Conv2D108, [((1, 1488, 7, 7), torch.float32)]),
    (Conv2D109, [((1, 1536, 7, 7), torch.float32)]),
    (Conv2D110, [((1, 1584, 7, 7), torch.float32)]),
    (Conv2D111, [((1, 1632, 7, 7), torch.float32)]),
    (Conv2D112, [((1, 1680, 7, 7), torch.float32)]),
    (Conv2D113, [((1, 1728, 7, 7), torch.float32)]),
    (Conv2D114, [((1, 1776, 7, 7), torch.float32)]),
    (Conv2D115, [((1, 1824, 7, 7), torch.float32)]),
    (Conv2D116, [((1, 1872, 7, 7), torch.float32)]),
    (Conv2D117, [((1, 1920, 7, 7), torch.float32)]),
    (Conv2D118, [((1, 1968, 7, 7), torch.float32)]),
    (Conv2D119, [((1, 2016, 7, 7), torch.float32)]),
    (Conv2D120, [((1, 2064, 7, 7), torch.float32)]),
    (Conv2D122, [((1, 2112, 7, 7), torch.float32)]),
    (Conv2D123, [((1, 2160, 7, 7), torch.float32)]),
    (Conv2D124, [((1, 1280, 14, 14), torch.float32)]),
    (Conv2D125, [((1, 3, 224, 224), torch.float32)]),
    (Conv2D126, [((1, 16, 224, 224), torch.float32)]),
    (Conv2D127, [((1, 16, 224, 224), torch.float32)]),
    (Conv2D128, [((1, 32, 112, 112), torch.float32)]),
    (Conv2D129, [((1, 64, 112, 112), torch.float32)]),
    (Conv2D130, [((1, 64, 56, 56), torch.float32)]),
    (Conv2D128, [((1, 32, 56, 56), torch.float32)]),
    (Conv2D131, [((1, 64, 56, 56), torch.float32)]),
    (Conv2D129, [((1, 64, 56, 56), torch.float32)]),
    (Conv2D132, [((1, 128, 56, 56), torch.float32)]),
    (Conv2D130, [((1, 64, 28, 28), torch.float32)]),
    (Conv2D131, [((1, 64, 28, 28), torch.float32)]),
    (Conv2D132, [((1, 128, 28, 28), torch.float32)]),
    (Conv2D133, [((1, 256, 28, 28), torch.float32)]),
    (Conv2D13, [((1, 64, 28, 28), torch.float32)]),
    (Conv2D134, [((1, 128, 28, 28), torch.float32)]),
    (Conv2D17, [((1, 128, 14, 14), torch.float32)]),
    (Conv2D13, [((1, 64, 14, 14), torch.float32)]),
    (Conv2D135, [((1, 128, 14, 14), torch.float32)]),
    (Conv2D136, [((1, 128, 14, 14), torch.float32)]),
    (Conv2D137, [((1, 256, 14, 14), torch.float32)]),
    (Conv2D138, [((1, 256, 7, 7), torch.float32)]),
    (Conv2D136, [((1, 128, 7, 7), torch.float32)]),
    (Conv2D139, [((1, 256, 7, 7), torch.float32)]),
    (Conv2D140, [((1, 640, 7, 7), torch.float32)]),
    (Conv2D141, [((1, 256, 1, 1), torch.float32)]),
    (Conv2D142, [((1, 32, 112, 112), torch.float32)]),
    (Conv2D143, [((1, 32, 112, 112), torch.float32)]),
    (Conv2D144, [((1, 32, 112, 112), torch.float32)]),
    (Conv2D145, [((1, 64, 56, 56), torch.float32)]),
    (Conv2D144, [((1, 32, 56, 56), torch.float32)]),
    (Conv2D143, [((1, 32, 56, 56), torch.float32)]),
    (Conv2D128, [((1, 32, 28, 28), torch.float32)]),
    (Conv2D145, [((1, 64, 28, 28), torch.float32)]),
    (Conv2D144, [((1, 32, 28, 28), torch.float32)]),
    (Conv2D143, [((1, 32, 28, 28), torch.float32)]),
    (Conv2D146, [((1, 64, 28, 28), torch.float32)]),
    (Conv2D147, [((1, 64, 28, 28), torch.float32)]),
    (Conv2D132, [((1, 128, 14, 14), torch.float32)]),
    (Conv2D147, [((1, 64, 14, 14), torch.float32)]),
    (Conv2D148, [((1, 128, 14, 14), torch.float32)]),
    (Conv2D149, [((1, 128, 14, 14), torch.float32)]),
    (Conv2D21, [((1, 256, 7, 7), torch.float32)]),
    (Conv2D149, [((1, 128, 7, 7), torch.float32)]),
    (Conv2D150, [((1, 32, 112, 112), torch.float32)]),
    (Conv2D151, [((1, 256, 112, 112), torch.float32)]),
    (Conv2D152, [((1, 32, 56, 56), torch.float32)]),
    (Conv2D136, [((1, 128, 56, 56), torch.float32)]),
    (Conv2D153, [((1, 256, 56, 56), torch.float32)]),
    (Conv2D154, [((1, 128, 56, 56), torch.float32)]),
    (Conv2D155, [((1, 512, 56, 56), torch.float32)]),
    (Conv2D136, [((1, 128, 28, 28), torch.float32)]),
    (Conv2D156, [((1, 256, 28, 28), torch.float32)]),
    (Conv2D157, [((1, 512, 28, 28), torch.float32)]),
    (Conv2D158, [((1, 768, 28, 28), torch.float32)]),
    (Conv2D159, [((1, 1152, 28, 28), torch.float32)]),
    (Conv2D160, [((1, 256, 28, 28), torch.float32)]),
    (Conv2D161, [((1, 1024, 28, 28), torch.float32)]),
    (Conv2D156, [((1, 256, 14, 14), torch.float32)]),
    (Conv2D162, [((1, 512, 14, 14), torch.float32)]),
    (Conv2D163, [((1, 1024, 14, 14), torch.float32)]),
    (Conv2D164, [((1, 1536, 14, 14), torch.float32)]),
    (Conv2D165, [((1, 2048, 14, 14), torch.float32)]),
    (Conv2D166, [((1, 2816, 14, 14), torch.float32)]),
    (Conv2D167, [((1, 512, 14, 14), torch.float32)]),
    (Conv2D168, [((1, 2048, 14, 14), torch.float32)]),
    (Conv2D169, [((1, 2048, 7, 7), torch.float32)]),
    (Conv2D162, [((1, 512, 7, 7), torch.float32)]),
    (Conv2D170, [((1, 1024, 7, 7), torch.float32)]),
    (Conv2D171, [((1, 2048, 7, 7), torch.float32)]),
    (Conv2D172, [((1, 2560, 7, 7), torch.float32)]),
    (Conv2D173, [((1, 1024, 1, 1), torch.float32)]),
    (Conv2D146, [((1, 64, 112, 112), torch.float32)]),
    (Conv2D147, [((1, 64, 112, 112), torch.float32)]),
    (Conv2D147, [((1, 64, 56, 56), torch.float32)]),
    (Conv2D146, [((1, 64, 56, 56), torch.float32)]),
    (Conv2D25, [((1, 384, 56, 56), torch.float32)]),
    (Conv2D148, [((1, 128, 56, 56), torch.float32)]),
    (Conv2D149, [((1, 128, 56, 56), torch.float32)]),
    (Conv2D149, [((1, 128, 28, 28), torch.float32)]),
    (Conv2D148, [((1, 128, 28, 28), torch.float32)]),
    (Conv2D138, [((1, 256, 28, 28), torch.float32)]),
    (Conv2D174, [((1, 256, 28, 28), torch.float32)]),
    (Conv2D6, [((1, 256, 28, 28), torch.float32)]),
    (Conv2D29, [((1, 512, 14, 14), torch.float32)]),
    (Conv2D6, [((1, 256, 14, 14), torch.float32)]),
    (Conv2D175, [((1, 2560, 14, 14), torch.float32)]),
    (Conv2D176, [((1, 3328, 14, 14), torch.float32)]),
    (Conv2D177, [((1, 512, 14, 14), torch.float32)]),
    (Conv2D178, [((1, 512, 14, 14), torch.float32)]),
    (Conv2D179, [((1, 512, 14, 14), torch.float32)]),
    (Conv2D75, [((1, 1024, 7, 7), torch.float32)]),
    (Conv2D179, [((1, 512, 7, 7), torch.float32)]),
    (Conv2D180, [((1, 896, 28, 28), torch.float32)]),
    (Conv2D181, [((1, 2304, 14, 14), torch.float32)]),
    (Conv2D152, [((1, 32, 112, 112), torch.float32)]),
    (Conv2D182, [((1, 32, 112, 112), torch.float32)]),
    (Conv2D134, [((1, 128, 112, 112), torch.float32)]),
    (Conv2D135, [((1, 128, 56, 56), torch.float32)]),
    (Conv2D137, [((1, 256, 56, 56), torch.float32)]),
    (Conv2D139, [((1, 256, 56, 56), torch.float32)]),
    (Conv2D139, [((1, 256, 28, 28), torch.float32)]),
    (Conv2D183, [((1, 512, 28, 28), torch.float32)]),
    (Conv2D184, [((1, 512, 28, 28), torch.float32)]),
    (Conv2D184, [((1, 512, 14, 14), torch.float32)]),
    (Conv2D185, [((1, 1024, 14, 14), torch.float32)]),
    (Conv2D186, [((1, 1024, 14, 14), torch.float32)]),
    (Conv2D187, [((1, 1024, 7, 7), torch.float32)]),
    (Conv2D186, [((1, 1024, 7, 7), torch.float32)]),
    (Conv2D188, [((1, 32, 112, 112), torch.float32)]),
    (Conv2D189, [((1, 64, 56, 56), torch.float32)]),
    (Conv2D190, [((1, 128, 28, 28), torch.float32)]),
    (Conv2D180, [((1, 896, 14, 14), torch.float32)]),
    (Conv2D191, [((1, 256, 14, 14), torch.float32)]),
    (Conv2D156, [((1, 256, 7, 7), torch.float32)]),
    (Conv2D192, [((1, 1280, 7, 7), torch.float32)]),
    (Conv2D193, [((1, 512, 1, 1), torch.float32)]),
    (Conv2D194, [((1, 3, 224, 224), torch.float32)]),
    (Conv2D195, [((1, 48, 112, 112), torch.float32), ((48, 1, 3, 3), torch.float32)]),
    (Conv2D196, [((1, 48, 112, 112), torch.float32), ((48, 1, 3, 3), torch.float32)]),
    (Conv2D197, [((1, 48, 1, 1), torch.float32)]),
    (Conv2D198, [((1, 12, 1, 1), torch.float32)]),
    (Conv2D199, [((1, 48, 112, 112), torch.float32)]),
    (Conv2D200, [((1, 24, 112, 112), torch.float32), ((24, 1, 3, 3), torch.float32)]),
    (Conv2D201, [((1, 24, 1, 1), torch.float32)]),
    (Conv2D202, [((1, 6, 1, 1), torch.float32)]),
    (Conv2D203, [((1, 24, 112, 112), torch.float32)]),
    (Conv2D204, [((1, 24, 112, 112), torch.float32)]),
    (Conv2D205, [((1, 144, 112, 112), torch.float32), ((144, 1, 3, 3), torch.float32)]),
    (Conv2D206, [((1, 144, 1, 1), torch.float32)]),
    (Conv2D207, [((1, 6, 1, 1), torch.float32)]),
    (Conv2D208, [((1, 144, 56, 56), torch.float32)]),
    (Conv2D209, [((1, 32, 56, 56), torch.float32)]),
    (Conv2D210, [((1, 192, 56, 56), torch.float32), ((192, 1, 3, 3), torch.float32)]),
    (Conv2D211, [((1, 192, 1, 1), torch.float32)]),
    (Conv2D212, [((1, 8, 1, 1), torch.float32)]),
    (Conv2D213, [((1, 192, 56, 56), torch.float32)]),
    (Conv2D214, [((1, 192, 56, 56), torch.float32), ((192, 1, 5, 5), torch.float32)]),
    (Conv2D215, [((1, 192, 28, 28), torch.float32)]),
    (Conv2D216, [((1, 56, 28, 28), torch.float32)]),
    (Conv2D217, [((1, 336, 28, 28), torch.float32), ((336, 1, 5, 5), torch.float32)]),
    (Conv2D218, [((1, 336, 1, 1), torch.float32)]),
    (Conv2D219, [((1, 14, 1, 1), torch.float32)]),
    (Conv2D220, [((1, 336, 28, 28), torch.float32)]),
    (Conv2D221, [((1, 336, 28, 28), torch.float32), ((336, 1, 3, 3), torch.float32)]),
    (Conv2D222, [((1, 336, 14, 14), torch.float32)]),
    (Conv2D223, [((1, 112, 14, 14), torch.float32)]),
    (Conv2D224, [((1, 672, 14, 14), torch.float32), ((672, 1, 3, 3), torch.float32)]),
    (Conv2D225, [((1, 672, 1, 1), torch.float32)]),
    (Conv2D226, [((1, 28, 1, 1), torch.float32)]),
    (Conv2D227, [((1, 672, 14, 14), torch.float32)]),
    (Conv2D228, [((1, 672, 14, 14), torch.float32), ((672, 1, 5, 5), torch.float32)]),
    (Conv2D229, [((1, 672, 14, 14), torch.float32), ((672, 1, 5, 5), torch.float32)]),
    (Conv2D230, [((1, 672, 14, 14), torch.float32)]),
    (Conv2D231, [((1, 160, 14, 14), torch.float32)]),
    (Conv2D232, [((1, 960, 14, 14), torch.float32), ((960, 1, 5, 5), torch.float32)]),
    (Conv2D233, [((1, 960, 14, 14), torch.float32), ((960, 1, 5, 5), torch.float32)]),
    (Conv2D234, [((1, 960, 1, 1), torch.float32)]),
    (Conv2D235, [((1, 40, 1, 1), torch.float32)]),
    (Conv2D236, [((1, 960, 14, 14), torch.float32)]),
    (Conv2D237, [((1, 960, 7, 7), torch.float32)]),
    (Conv2D238, [((1, 272, 7, 7), torch.float32)]),
    (Conv2D239, [((1, 1632, 7, 7), torch.float32), ((1632, 1, 5, 5), torch.float32)]),
    (Conv2D240, [((1, 1632, 1, 1), torch.float32)]),
    (Conv2D241, [((1, 68, 1, 1), torch.float32)]),
    (Conv2D242, [((1, 1632, 7, 7), torch.float32)]),
    (Conv2D243, [((1, 1632, 7, 7), torch.float32), ((1632, 1, 3, 3), torch.float32)]),
    (Conv2D244, [((1, 1632, 7, 7), torch.float32)]),
    (Conv2D245, [((1, 448, 7, 7), torch.float32)]),
    (Conv2D246, [((1, 2688, 7, 7), torch.float32), ((2688, 1, 3, 3), torch.float32)]),
    (Conv2D247, [((1, 2688, 1, 1), torch.float32)]),
    (Conv2D248, [((1, 112, 1, 1), torch.float32)]),
    (Conv2D249, [((1, 2688, 7, 7), torch.float32)]),
    (Conv2D250, [((1, 448, 7, 7), torch.float32)]),
    (Conv2D194, [((1, 3, 320, 320), torch.float32)]),
    (Conv2D195, [((1, 48, 160, 160), torch.float32), ((48, 1, 3, 3), torch.float32)]),
    (Conv2D199, [((1, 48, 160, 160), torch.float32)]),
    (Conv2D200, [((1, 24, 160, 160), torch.float32), ((24, 1, 3, 3), torch.float32)]),
    (Conv2D203, [((1, 24, 160, 160), torch.float32)]),
    (Conv2D204, [((1, 24, 160, 160), torch.float32)]),
    (Conv2D205, [((1, 144, 160, 160), torch.float32), ((144, 1, 3, 3), torch.float32)]),
    (Conv2D208, [((1, 144, 80, 80), torch.float32)]),
    (Conv2D209, [((1, 32, 80, 80), torch.float32)]),
    (Conv2D210, [((1, 192, 80, 80), torch.float32), ((192, 1, 3, 3), torch.float32)]),
    (Conv2D213, [((1, 192, 80, 80), torch.float32)]),
    (Conv2D214, [((1, 192, 80, 80), torch.float32), ((192, 1, 5, 5), torch.float32)]),
    (Conv2D215, [((1, 192, 40, 40), torch.float32)]),
    (Conv2D216, [((1, 56, 40, 40), torch.float32)]),
    (Conv2D217, [((1, 336, 40, 40), torch.float32), ((336, 1, 5, 5), torch.float32)]),
    (Conv2D220, [((1, 336, 40, 40), torch.float32)]),
    (Conv2D221, [((1, 336, 40, 40), torch.float32), ((336, 1, 3, 3), torch.float32)]),
    (Conv2D222, [((1, 336, 20, 20), torch.float32)]),
    (Conv2D223, [((1, 112, 20, 20), torch.float32)]),
    (Conv2D224, [((1, 672, 20, 20), torch.float32), ((672, 1, 3, 3), torch.float32)]),
    (Conv2D227, [((1, 672, 20, 20), torch.float32)]),
    (Conv2D228, [((1, 672, 20, 20), torch.float32), ((672, 1, 5, 5), torch.float32)]),
    (Conv2D230, [((1, 672, 20, 20), torch.float32)]),
    (Conv2D231, [((1, 160, 20, 20), torch.float32)]),
    (Conv2D232, [((1, 960, 20, 20), torch.float32), ((960, 1, 5, 5), torch.float32)]),
    (Conv2D233, [((1, 960, 20, 20), torch.float32), ((960, 1, 5, 5), torch.float32)]),
    (Conv2D236, [((1, 960, 20, 20), torch.float32)]),
    (Conv2D237, [((1, 960, 10, 10), torch.float32)]),
    (Conv2D238, [((1, 272, 10, 10), torch.float32)]),
    (Conv2D239, [((1, 1632, 10, 10), torch.float32), ((1632, 1, 5, 5), torch.float32)]),
    (Conv2D242, [((1, 1632, 10, 10), torch.float32)]),
    (Conv2D243, [((1, 1632, 10, 10), torch.float32), ((1632, 1, 3, 3), torch.float32)]),
    (Conv2D244, [((1, 1632, 10, 10), torch.float32)]),
    (Conv2D245, [((1, 448, 10, 10), torch.float32)]),
    (Conv2D246, [((1, 2688, 10, 10), torch.float32), ((2688, 1, 3, 3), torch.float32)]),
    (Conv2D249, [((1, 2688, 10, 10), torch.float32)]),
    (Conv2D250, [((1, 448, 10, 10), torch.float32)]),
    (Conv2D251, [((1, 3, 224, 224), torch.float32)]),
    (Conv2D252, [((1, 3, 224, 224), torch.float32)]),
    (Conv2D253, [((1, 32, 112, 112), torch.float32), ((32, 1, 3, 3), torch.float32)]),
    (Conv2D254, [((1, 32, 1, 1), torch.float32)]),
    (Conv2D255, [((1, 8, 1, 1), torch.float32)]),
    (Conv2D256, [((1, 32, 112, 112), torch.float32)]),
    (Conv2D257, [((1, 16, 112, 112), torch.float32)]),
    (Conv2D258, [((1, 96, 112, 112), torch.float32), ((96, 1, 3, 3), torch.float32)]),
    (Conv2D259, [((1, 96, 112, 112), torch.float32), ((96, 1, 3, 3), torch.float32)]),
    (Conv2D260, [((1, 96, 1, 1), torch.float32)]),
    (Conv2D261, [((1, 4, 1, 1), torch.float32)]),
    (Conv2D262, [((1, 96, 56, 56), torch.float32)]),
    (Conv2D204, [((1, 24, 56, 56), torch.float32)]),
    (Conv2D263, [((1, 144, 56, 56), torch.float32), ((144, 1, 3, 3), torch.float32)]),
    (Conv2D205, [((1, 144, 56, 56), torch.float32), ((144, 1, 3, 3), torch.float32)]),
    (Conv2D264, [((1, 144, 56, 56), torch.float32), ((144, 1, 3, 3), torch.float32)]),
    (Conv2D265, [((1, 144, 56, 56), torch.float32)]),
    (Conv2D266, [((1, 144, 56, 56), torch.float32), ((144, 1, 5, 5), torch.float32)]),
    (Conv2D267, [((1, 144, 28, 28), torch.float32)]),
    (Conv2D268, [((1, 40, 28, 28), torch.float32)]),
    (Conv2D269, [((1, 240, 28, 28), torch.float32), ((240, 1, 5, 5), torch.float32)]),
    (Conv2D270, [((1, 240, 1, 1), torch.float32)]),
    (Conv2D271, [((1, 10, 1, 1), torch.float32)]),
    (Conv2D272, [((1, 240, 28, 28), torch.float32)]),
    (Conv2D273, [((1, 240, 28, 28), torch.float32), ((240, 1, 3, 3), torch.float32)]),
    (Conv2D274, [((1, 240, 14, 14), torch.float32)]),
    (Conv2D275, [((1, 80, 14, 14), torch.float32)]),
    (Conv2D276, [((1, 480, 14, 14), torch.float32), ((480, 1, 3, 3), torch.float32)]),
    (Conv2D277, [((1, 480, 1, 1), torch.float32)]),
    (Conv2D278, [((1, 20, 1, 1), torch.float32)]),
    (Conv2D279, [((1, 480, 14, 14), torch.float32)]),
    (Conv2D280, [((1, 480, 14, 14), torch.float32), ((480, 1, 5, 5), torch.float32)]),
    (Conv2D281, [((1, 480, 14, 14), torch.float32)]),
    (Conv2D90, [((1, 672, 7, 7), torch.float32)]),
    (Conv2D282, [((1, 192, 7, 7), torch.float32)]),
    (Conv2D283, [((1, 1152, 7, 7), torch.float32), ((1152, 1, 5, 5), torch.float32)]),
    (Conv2D284, [((1, 1152, 1, 1), torch.float32)]),
    (Conv2D285, [((1, 48, 1, 1), torch.float32)]),
    (Conv2D286, [((1, 1152, 7, 7), torch.float32), ((1152, 1, 3, 3), torch.float32)]),
    (Conv2D287, [((1, 1152, 7, 7), torch.float32)]),
    (Conv2D288, [((1, 320, 7, 7), torch.float32)]),
    (Conv2D138, [((1, 256, 64, 64), torch.float32)]),
    (Conv2D29, [((1, 512, 16, 16), torch.float32)]),
    (Conv2D289, [((1, 2048, 8, 8), torch.float32)]),
    (Conv2D6, [((1, 256, 64, 64), torch.float32)]),
    (Conv2D6, [((1, 256, 16, 16), torch.float32)]),
    (Conv2D6, [((1, 256, 8, 8), torch.float32)]),
    (Conv2D290, [((1, 3, 224, 224), torch.float32)]),
    (Conv2D291, [((1, 16, 112, 112), torch.float32)]),
    (Conv2D292, [((1, 8, 112, 112), torch.float32), ((8, 1, 3, 3), torch.float32)]),
    (Conv2D293, [((1, 16, 112, 112), torch.float32)]),
    (Conv2D197, [((1, 48, 56, 56), torch.float32)]),
    (Conv2D294, [((1, 12, 56, 56), torch.float32), ((12, 1, 3, 3), torch.float32)]),
    (Conv2D295, [((1, 16, 112, 112), torch.float32), ((16, 1, 3, 3), torch.float32)]),
    (Conv2D296, [((1, 16, 112, 112), torch.float32), ((16, 1, 3, 3), torch.float32)]),
    (Conv2D293, [((1, 16, 56, 56), torch.float32)]),
    (Conv2D297, [((1, 24, 56, 56), torch.float32)]),
    (Conv2D298, [((1, 36, 56, 56), torch.float32), ((36, 1, 3, 3), torch.float32)]),
    (Conv2D299, [((1, 72, 56, 56), torch.float32)]),
    (Conv2D300, [((1, 72, 56, 56), torch.float32), ((72, 1, 5, 5), torch.float32)]),
    (Conv2D301, [((1, 72, 1, 1), torch.float32)]),
    (Conv2D302, [((1, 20, 1, 1), torch.float32)]),
    (Conv2D301, [((1, 72, 28, 28), torch.float32)]),
    (Conv2D303, [((1, 20, 28, 28), torch.float32), ((20, 1, 3, 3), torch.float32)]),
    (Conv2D304, [((1, 24, 56, 56), torch.float32), ((24, 1, 5, 5), torch.float32)]),
    (Conv2D305, [((1, 24, 28, 28), torch.float32)]),
    (Conv2D306, [((1, 40, 28, 28), torch.float32)]),
    (Conv2D307, [((1, 60, 28, 28), torch.float32), ((60, 1, 3, 3), torch.float32)]),
    (Conv2D308, [((1, 120, 1, 1), torch.float32)]),
    (Conv2D309, [((1, 32, 1, 1), torch.float32)]),
    (Conv2D310, [((1, 120, 28, 28), torch.float32)]),
    (Conv2D311, [((1, 40, 28, 28), torch.float32)]),
    (Conv2D312, [((1, 120, 28, 28), torch.float32), ((120, 1, 3, 3), torch.float32)]),
    (Conv2D272, [((1, 240, 14, 14), torch.float32)]),
    (Conv2D313, [((1, 40, 14, 14), torch.float32), ((40, 1, 3, 3), torch.float32)]),
    (Conv2D314, [((1, 40, 28, 28), torch.float32), ((40, 1, 3, 3), torch.float32)]),
    (Conv2D315, [((1, 40, 14, 14), torch.float32)]),
    (Conv2D316, [((1, 80, 14, 14), torch.float32)]),
    (Conv2D317, [((1, 100, 14, 14), torch.float32), ((100, 1, 3, 3), torch.float32)]),
    (Conv2D318, [((1, 200, 14, 14), torch.float32)]),
    (Conv2D319, [((1, 80, 14, 14), torch.float32)]),
    (Conv2D320, [((1, 92, 14, 14), torch.float32), ((92, 1, 3, 3), torch.float32)]),
    (Conv2D321, [((1, 184, 14, 14), torch.float32)]),
    (Conv2D322, [((1, 80, 14, 14), torch.float32)]),
    (Conv2D323, [((1, 240, 14, 14), torch.float32), ((240, 1, 3, 3), torch.float32)]),
    (Conv2D324, [((1, 480, 1, 1), torch.float32)]),
    (Conv2D325, [((1, 120, 1, 1), torch.float32)]),
    (Conv2D326, [((1, 480, 14, 14), torch.float32)]),
    (Conv2D327, [((1, 56, 14, 14), torch.float32), ((56, 1, 3, 3), torch.float32)]),
    (Conv2D328, [((1, 80, 14, 14), torch.float32), ((80, 1, 3, 3), torch.float32)]),
    (Conv2D329, [((1, 80, 14, 14), torch.float32)]),
    (Conv2D330, [((1, 112, 14, 14), torch.float32)]),
    (Conv2D331, [((1, 336, 14, 14), torch.float32), ((336, 1, 3, 3), torch.float32)]),
    (Conv2D332, [((1, 672, 1, 1), torch.float32)]),
    (Conv2D333, [((1, 168, 1, 1), torch.float32)]),
    (Conv2D334, [((1, 672, 14, 14), torch.float32)]),
    (Conv2D335, [((1, 672, 7, 7), torch.float32)]),
    (Conv2D328, [((1, 80, 7, 7), torch.float32), ((80, 1, 3, 3), torch.float32)]),
    (Conv2D336, [((1, 112, 14, 14), torch.float32), ((112, 1, 5, 5), torch.float32)]),
    (Conv2D337, [((1, 112, 7, 7), torch.float32)]),
    (Conv2D338, [((1, 160, 7, 7), torch.float32)]),
    (Conv2D276, [((1, 480, 7, 7), torch.float32), ((480, 1, 3, 3), torch.float32)]),
    (Conv2D339, [((1, 960, 7, 7), torch.float32)]),
    (Conv2D340, [((1, 960, 1, 1), torch.float32)]),
    (Conv2D341, [((1, 240, 1, 1), torch.float32)]),
    (Conv2D231, [((1, 160, 7, 7), torch.float32)]),
    (Conv2D342, [((1, 960, 1, 1), torch.float32)]),
    (Conv2D343, [((1, 64, 56, 56), torch.float32)]),
    (Conv2D1, [((1, 192, 28, 28), torch.float32), ((176, 192, 1, 1), torch.float32)]),
    (Conv2D344, [((1, 96, 28, 28), torch.float32)]),
    (Conv2D345, [((1, 16, 28, 28), torch.float32)]),
    (Conv2D213, [((1, 192, 28, 28), torch.float32)]),
    (Conv2D1, [((1, 256, 28, 28), torch.float32), ((288, 256, 1, 1), torch.float32)]),
    (Conv2D346, [((1, 128, 28, 28), torch.float32)]),
    (Conv2D347, [((1, 32, 28, 28), torch.float32)]),
    (Conv2D1, [((1, 480, 14, 14), torch.float32), ((304, 480, 1, 1), torch.float32)]),
    (Conv2D348, [((1, 96, 14, 14), torch.float32)]),
    (Conv2D349, [((1, 16, 14, 14), torch.float32)]),
    (Conv2D350, [((1, 480, 14, 14), torch.float32)]),
    (Conv2D1, [((1, 512, 14, 14), torch.float32), ((296, 512, 1, 1), torch.float32)]),
    (Conv2D351, [((1, 112, 14, 14), torch.float32)]),
    (Conv2D352, [((1, 24, 14, 14), torch.float32)]),
    (Conv2D353, [((1, 512, 14, 14), torch.float32)]),
    (Conv2D1, [((1, 512, 14, 14), torch.float32), ((280, 512, 1, 1), torch.float32)]),
    (Conv2D354, [((1, 128, 14, 14), torch.float32)]),
    (Conv2D190, [((1, 128, 14, 14), torch.float32)]),
    (Conv2D1, [((1, 512, 14, 14), torch.float32), ((288, 512, 1, 1), torch.float32)]),
    (Conv2D355, [((1, 144, 14, 14), torch.float32)]),
    (Conv2D356, [((1, 32, 14, 14), torch.float32)]),
    (Conv2D1, [((1, 528, 14, 14), torch.float32), ((448, 528, 1, 1), torch.float32)]),
    (Conv2D357, [((1, 160, 14, 14), torch.float32)]),
    (Conv2D358, [((1, 160, 14, 14), torch.float32)]),
    (Conv2D359, [((1, 32, 14, 14), torch.float32)]),
    (Conv2D360, [((1, 32, 14, 14), torch.float32)]),
    (Conv2D361, [((1, 528, 14, 14), torch.float32)]),
    (Conv2D1, [((1, 832, 7, 7), torch.float32), ((448, 832, 1, 1), torch.float32)]),
    (Conv2D357, [((1, 160, 7, 7), torch.float32)]),
    (Conv2D359, [((1, 32, 7, 7), torch.float32)]),
    (Conv2D1, [((1, 832, 7, 7), torch.float32), ((624, 832, 1, 1), torch.float32)]),
    (Conv2D4, [((1, 192, 7, 7), torch.float32)]),
    (Conv2D362, [((1, 48, 7, 7), torch.float32)]),
    (Conv2D363, [((1, 3, 224, 224), torch.float32)]),
    (Conv2D364, [((1, 3, 224, 224), torch.float32)]),
    (Conv2D365, [((1, 64, 56, 56), torch.float32)]),
    (Conv2D133, [((1, 256, 56, 56), torch.float32)]),
    (Conv2D366, [((1, 256, 56, 56), torch.float32)]),
    (Conv2D367, [((1, 18, 56, 56), torch.float32)]),
    (Conv2D368, [((1, 18, 56, 56), torch.float32)]),
    (Conv2D369, [((1, 256, 56, 56), torch.float32)]),
    (Conv2D370, [((1, 36, 28, 28), torch.float32)]),
    (Conv2D371, [((1, 36, 28, 28), torch.float32)]),
    (Conv2D372, [((1, 36, 28, 28), torch.float32)]),
    (Conv2D373, [((1, 18, 56, 56), torch.float32)]),
    (Conv2D374, [((1, 36, 28, 28), torch.float32)]),
    (Conv2D375, [((1, 72, 14, 14), torch.float32)]),
    (Conv2D376, [((1, 72, 14, 14), torch.float32)]),
    (Conv2D377, [((1, 72, 14, 14), torch.float32)]),
    (Conv2D378, [((1, 18, 28, 28), torch.float32)]),
    (Conv2D379, [((1, 72, 14, 14), torch.float32)]),
    (Conv2D380, [((1, 144, 7, 7), torch.float32)]),
    (Conv2D1, [((1, 144, 7, 7), torch.float32), ((126, 144, 1, 1), torch.float32)]),
    (Conv2D381, [((1, 18, 56, 56), torch.float32), ((72, 18, 3, 3), torch.float32)]),
    (Conv2D368, [((1, 18, 28, 28), torch.float32)]),
    (Conv2D382, [((1, 18, 14, 14), torch.float32)]),
    (Conv2D383, [((1, 36, 14, 14), torch.float32)]),
    (Conv2D384, [((1, 144, 7, 7), torch.float32)]),
    (Conv2D6, [((1, 256, 7, 7), torch.float32)]),
    (Conv2D160, [((1, 256, 7, 7), torch.float32)]),
    (Conv2D385, [((1, 144, 7, 7), torch.float32)]),
    (Conv2D386, [((1, 72, 14, 14), torch.float32)]),
    (Conv2D154, [((1, 128, 14, 14), torch.float32)]),
    (Conv2D387, [((1, 72, 14, 14), torch.float32)]),
    (Conv2D388, [((1, 36, 28, 28), torch.float32)]),
    (Conv2D365, [((1, 64, 28, 28), torch.float32)]),
    (Conv2D389, [((1, 36, 28, 28), torch.float32)]),
    (Conv2D390, [((1, 18, 56, 56), torch.float32)]),
    (Conv2D391, [((1, 18, 56, 56), torch.float32)]),
    (Conv2D190, [((1, 128, 56, 56), torch.float32)]),
    (Conv2D354, [((1, 128, 56, 56), torch.float32)]),
    (Conv2D191, [((1, 256, 28, 28), torch.float32)]),
    (Conv2D392, [((1, 256, 28, 28), torch.float32)]),
    (Conv2D393, [((1, 512, 14, 14), torch.float32)]),
    (Conv2D394, [((1, 256, 56, 56), torch.float32)]),
    (Conv2D395, [((1, 40, 56, 56), torch.float32)]),
    (Conv2D396, [((1, 40, 56, 56), torch.float32)]),
    (Conv2D397, [((1, 256, 56, 56), torch.float32)]),
    (Conv2D398, [((1, 80, 28, 28), torch.float32)]),
    (Conv2D399, [((1, 80, 28, 28), torch.float32)]),
    (Conv2D400, [((1, 80, 28, 28), torch.float32)]),
    (Conv2D401, [((1, 40, 56, 56), torch.float32)]),
    (Conv2D402, [((1, 80, 28, 28), torch.float32)]),
    (Conv2D403, [((1, 160, 14, 14), torch.float32)]),
    (Conv2D404, [((1, 160, 14, 14), torch.float32)]),
    (Conv2D405, [((1, 160, 14, 14), torch.float32)]),
    (Conv2D406, [((1, 40, 28, 28), torch.float32)]),
    (Conv2D407, [((1, 320, 7, 7), torch.float32)]),
    (Conv2D1, [((1, 320, 7, 7), torch.float32), ((280, 320, 1, 1), torch.float32)]),
    (Conv2D381, [((1, 40, 56, 56), torch.float32), ((160, 40, 3, 3), torch.float32)]),
    (Conv2D396, [((1, 40, 28, 28), torch.float32)]),
    (Conv2D408, [((1, 40, 14, 14), torch.float32)]),
    (Conv2D409, [((1, 80, 14, 14), torch.float32)]),
    (Conv2D410, [((1, 320, 7, 7), torch.float32)]),
    (Conv2D411, [((1, 320, 7, 7), torch.float32)]),
    (Conv2D18, [((1, 160, 14, 14), torch.float32)]),
    (Conv2D412, [((1, 160, 14, 14), torch.float32)]),
    (Conv2D413, [((1, 80, 28, 28), torch.float32)]),
    (Conv2D414, [((1, 80, 28, 28), torch.float32)]),
    (Conv2D415, [((1, 40, 56, 56), torch.float32)]),
    (Conv2D416, [((1, 40, 56, 56), torch.float32)]),
    (Conv2D417, [((1, 256, 56, 56), torch.float32)]),
    (Conv2D418, [((1, 256, 56, 56), torch.float32)]),
    (Conv2D419, [((1, 256, 56, 56), torch.float32)]),
    (Conv2D133, [((1, 256, 14, 14), torch.float32)]),
    (Conv2D420, [((1, 64, 28, 28), torch.float32)]),
    (Conv2D1, [((1, 512, 7, 7), torch.float32), ((448, 512, 1, 1), torch.float32)]),
    (Conv2D381, [((1, 64, 56, 56), torch.float32), ((256, 64, 3, 3), torch.float32)]),
    (Conv2D421, [((1, 64, 14, 14), torch.float32)]),
    (Conv2D422, [((1, 128, 14, 14), torch.float32)]),
    (Conv2D29, [((1, 512, 7, 7), torch.float32)]),
    (Conv2D423, [((1, 256, 56, 56), torch.float32)]),
    (Conv2D188, [((1, 32, 56, 56), torch.float32)]),
    (Conv2D189, [((1, 64, 28, 28), torch.float32)]),
    (Conv2D424, [((1, 128, 14, 14), torch.float32)]),
    (Conv2D360, [((1, 32, 28, 28), torch.float32)]),
    (Conv2D1, [((1, 256, 7, 7), torch.float32), ((224, 256, 1, 1), torch.float32)]),
    (Conv2D381, [((1, 32, 56, 56), torch.float32), ((128, 32, 3, 3), torch.float32)]),
    (Conv2D425, [((1, 32, 14, 14), torch.float32)]),
    (Conv2D420, [((1, 64, 14, 14), torch.float32)]),
    (Conv2D142, [((1, 32, 56, 56), torch.float32)]),
    (Conv2D426, [((1, 128, 56, 56), torch.float32)]),
    (Conv2D126, [((1, 16, 56, 56), torch.float32)]),
    (Conv2D427, [((1, 16, 56, 56), torch.float32)]),
    (Conv2D256, [((1, 32, 28, 28), torch.float32)]),
    (Conv2D127, [((1, 16, 56, 56), torch.float32)]),
    (Conv2D188, [((1, 32, 28, 28), torch.float32)]),
    (Conv2D428, [((1, 64, 14, 14), torch.float32)]),
    (Conv2D381, [((1, 16, 56, 56), torch.float32), ((64, 16, 3, 3), torch.float32)]),
    (Conv2D427, [((1, 16, 28, 28), torch.float32)]),
    (Conv2D429, [((1, 16, 14, 14), torch.float32)]),
    (Conv2D145, [((1, 64, 14, 14), torch.float32)]),
    (Conv2D430, [((1, 16, 28, 28), torch.float32)]),
    (Conv2D189, [((1, 64, 14, 14), torch.float32)]),
    (Conv2D431, [((1, 128, 7, 7), torch.float32)]),
    (Conv2D1, [((1, 128, 7, 7), torch.float32), ((112, 128, 1, 1), torch.float32)]),
    (Conv2D432, [((1, 64, 14, 14), torch.float32)]),
    (Conv2D150, [((1, 32, 28, 28), torch.float32)]),
    (Conv2D433, [((1, 16, 56, 56), torch.float32)]),
    (Conv2D434, [((1, 16, 56, 56), torch.float32)]),
    (Conv2D435, [((1, 256, 56, 56), torch.float32)]),
    (Conv2D436, [((1, 44, 56, 56), torch.float32)]),
    (Conv2D437, [((1, 44, 56, 56), torch.float32)]),
    (Conv2D438, [((1, 256, 56, 56), torch.float32)]),
    (Conv2D439, [((1, 88, 28, 28), torch.float32)]),
    (Conv2D440, [((1, 88, 28, 28), torch.float32)]),
    (Conv2D441, [((1, 88, 28, 28), torch.float32)]),
    (Conv2D442, [((1, 44, 56, 56), torch.float32)]),
    (Conv2D443, [((1, 88, 28, 28), torch.float32)]),
    (Conv2D444, [((1, 176, 14, 14), torch.float32)]),
    (Conv2D445, [((1, 176, 14, 14), torch.float32)]),
    (Conv2D446, [((1, 176, 14, 14), torch.float32)]),
    (Conv2D447, [((1, 44, 28, 28), torch.float32)]),
    (Conv2D448, [((1, 176, 14, 14), torch.float32)]),
    (Conv2D449, [((1, 352, 7, 7), torch.float32)]),
    (Conv2D1, [((1, 352, 7, 7), torch.float32), ((308, 352, 1, 1), torch.float32)]),
    (Conv2D381, [((1, 44, 56, 56), torch.float32), ((176, 44, 3, 3), torch.float32)]),
    (Conv2D437, [((1, 44, 28, 28), torch.float32)]),
    (Conv2D450, [((1, 44, 14, 14), torch.float32)]),
    (Conv2D451, [((1, 88, 14, 14), torch.float32)]),
    (Conv2D452, [((1, 352, 7, 7), torch.float32)]),
    (Conv2D453, [((1, 352, 7, 7), torch.float32)]),
    (Conv2D454, [((1, 176, 14, 14), torch.float32)]),
    (Conv2D455, [((1, 176, 14, 14), torch.float32)]),
    (Conv2D456, [((1, 88, 28, 28), torch.float32)]),
    (Conv2D457, [((1, 88, 28, 28), torch.float32)]),
    (Conv2D458, [((1, 44, 56, 56), torch.float32)]),
    (Conv2D459, [((1, 44, 56, 56), torch.float32)]),
    (Conv2D460, [((1, 256, 56, 56), torch.float32)]),
    (Conv2D461, [((1, 48, 56, 56), torch.float32)]),
    (Conv2D462, [((1, 48, 56, 56), torch.float32)]),
    (Conv2D463, [((1, 256, 56, 56), torch.float32)]),
    (Conv2D464, [((1, 96, 28, 28), torch.float32)]),
    (Conv2D465, [((1, 96, 28, 28), torch.float32)]),
    (Conv2D466, [((1, 96, 28, 28), torch.float32)]),
    (Conv2D467, [((1, 48, 56, 56), torch.float32)]),
    (Conv2D468, [((1, 96, 28, 28), torch.float32)]),
    (Conv2D469, [((1, 192, 14, 14), torch.float32)]),
    (Conv2D470, [((1, 192, 14, 14), torch.float32)]),
    (Conv2D471, [((1, 192, 14, 14), torch.float32)]),
    (Conv2D472, [((1, 48, 28, 28), torch.float32)]),
    (Conv2D473, [((1, 192, 14, 14), torch.float32)]),
    (Conv2D474, [((1, 384, 7, 7), torch.float32)]),
    (Conv2D1, [((1, 384, 7, 7), torch.float32), ((336, 384, 1, 1), torch.float32)]),
    (Conv2D381, [((1, 48, 56, 56), torch.float32), ((192, 48, 3, 3), torch.float32)]),
    (Conv2D462, [((1, 48, 28, 28), torch.float32)]),
    (Conv2D475, [((1, 48, 14, 14), torch.float32)]),
    (Conv2D476, [((1, 96, 14, 14), torch.float32)]),
    (Conv2D477, [((1, 384, 7, 7), torch.float32)]),
    (Conv2D478, [((1, 384, 7, 7), torch.float32)]),
    (Conv2D19, [((1, 192, 14, 14), torch.float32)]),
    (Conv2D479, [((1, 192, 14, 14), torch.float32)]),
    (Conv2D480, [((1, 96, 28, 28), torch.float32)]),
    (Conv2D481, [((1, 96, 28, 28), torch.float32)]),
    (Conv2D482, [((1, 48, 56, 56), torch.float32)]),
    (Conv2D483, [((1, 48, 56, 56), torch.float32)]),
    (Conv2D484, [((1, 256, 56, 56), torch.float32)]),
    (Conv2D485, [((1, 30, 56, 56), torch.float32)]),
    (Conv2D486, [((1, 30, 56, 56), torch.float32)]),
    (Conv2D487, [((1, 256, 56, 56), torch.float32)]),
    (Conv2D488, [((1, 60, 28, 28), torch.float32)]),
    (Conv2D489, [((1, 60, 28, 28), torch.float32)]),
    (Conv2D490, [((1, 60, 28, 28), torch.float32)]),
    (Conv2D491, [((1, 30, 56, 56), torch.float32)]),
    (Conv2D492, [((1, 60, 28, 28), torch.float32)]),
    (Conv2D493, [((1, 120, 14, 14), torch.float32)]),
    (Conv2D494, [((1, 120, 14, 14), torch.float32)]),
    (Conv2D495, [((1, 120, 14, 14), torch.float32)]),
    (Conv2D496, [((1, 30, 28, 28), torch.float32)]),
    (Conv2D497, [((1, 120, 14, 14), torch.float32)]),
    (Conv2D498, [((1, 240, 7, 7), torch.float32)]),
    (Conv2D1, [((1, 240, 7, 7), torch.float32), ((210, 240, 1, 1), torch.float32)]),
    (Conv2D381, [((1, 30, 56, 56), torch.float32), ((120, 30, 3, 3), torch.float32)]),
    (Conv2D486, [((1, 30, 28, 28), torch.float32)]),
    (Conv2D499, [((1, 30, 14, 14), torch.float32)]),
    (Conv2D500, [((1, 60, 14, 14), torch.float32)]),
    (Conv2D501, [((1, 240, 7, 7), torch.float32)]),
    (Conv2D502, [((1, 240, 7, 7), torch.float32)]),
    (Conv2D503, [((1, 120, 14, 14), torch.float32)]),
    (Conv2D504, [((1, 120, 14, 14), torch.float32)]),
    (Conv2D505, [((1, 60, 28, 28), torch.float32)]),
    (Conv2D506, [((1, 60, 28, 28), torch.float32)]),
    (Conv2D507, [((1, 30, 56, 56), torch.float32)]),
    (Conv2D508, [((1, 30, 56, 56), torch.float32)]),
    (Conv2D509, [((1, 3, 299, 299), torch.float32)]),
    (Conv2D251, [((1, 3, 299, 299), torch.float32)]),
    (Conv2D510, [((1, 32, 149, 149), torch.float32)]),
    (Conv2D356, [((1, 32, 147, 147), torch.float32)]),
    (Conv2D511, [((1, 64, 147, 147), torch.float32)]),
    (Conv2D512, [((1, 160, 73, 73), torch.float32)]),
    (Conv2D513, [((1, 64, 73, 73), torch.float32)]),
    (Conv2D514, [((1, 64, 73, 73), torch.float32)]),
    (Conv2D515, [((1, 64, 73, 73), torch.float32)]),
    (Conv2D516, [((1, 192, 71, 71), torch.float32)]),
    (Conv2D1, [((1, 384, 35, 35), torch.float32), ((224, 384, 1, 1), torch.float32)]),
    (Conv2D517, [((1, 64, 35, 35), torch.float32)]),
    (Conv2D464, [((1, 96, 35, 35), torch.float32)]),
    (Conv2D518, [((1, 384, 35, 35), torch.float32)]),
    (Conv2D519, [((1, 384, 35, 35), torch.float32)]),
    (Conv2D84, [((1, 384, 35, 35), torch.float32)]),
    (Conv2D520, [((1, 192, 35, 35), torch.float32)]),
    (Conv2D521, [((1, 224, 35, 35), torch.float32)]),
    (Conv2D1, [((1, 1024, 17, 17), torch.float32), ((768, 1024, 1, 1), torch.float32)]),
    (Conv2D522, [((1, 192, 17, 17), torch.float32)]),
    (Conv2D523, [((1, 224, 17, 17), torch.float32)]),
    (Conv2D524, [((1, 192, 17, 17), torch.float32)]),
    (Conv2D525, [((1, 224, 17, 17), torch.float32)]),
    (Conv2D526, [((1, 224, 17, 17), torch.float32)]),
    (Conv2D46, [((1, 1024, 17, 17), torch.float32)]),
    (Conv2D527, [((1, 1024, 17, 17), torch.float32)]),
    (Conv2D516, [((1, 192, 17, 17), torch.float32)]),
    (Conv2D528, [((1, 1024, 17, 17), torch.float32)]),
    (Conv2D529, [((1, 256, 17, 17), torch.float32)]),
    (Conv2D530, [((1, 256, 17, 17), torch.float32)]),
    (Conv2D531, [((1, 320, 17, 17), torch.float32)]),
    (Conv2D1, [((1, 1536, 8, 8), torch.float32), ((1024, 1536, 1, 1), torch.float32)]),
    (Conv2D532, [((1, 384, 8, 8), torch.float32)]),
    (Conv2D533, [((1, 384, 8, 8), torch.float32)]),
    (Conv2D534, [((1, 384, 8, 8), torch.float32)]),
    (Conv2D535, [((1, 448, 8, 8), torch.float32)]),
    (Conv2D536, [((1, 512, 8, 8), torch.float32)]),
    (Conv2D537, [((1, 512, 8, 8), torch.float32)]),
    (Conv2D538, [((1, 1536, 8, 8), torch.float32)]),
    (Conv2D539, [((1, 3, 224, 224), torch.float32)]),
    (Conv2D540, [((1, 3, 224, 224), torch.float32)]),
    (Conv2D541, [((1, 3, 224, 224), torch.float32)]),
    (Conv2D542, [((1, 3, 224, 224), torch.float32)]),
    (Conv2D543, [((1, 3, 224, 224), torch.float32)]),
    (Conv2D544, [((1, 3, 192, 192), torch.float32)]),
    (Conv2D200, [((1, 24, 96, 96), torch.float32), ((24, 1, 3, 3), torch.float32)]),
    (Conv2D545, [((1, 24, 96, 96), torch.float32)]),
    (Conv2D546, [((1, 48, 96, 96), torch.float32), ((48, 1, 3, 3), torch.float32)]),
    (Conv2D547, [((1, 48, 48, 48), torch.float32)]),
    (Conv2D548, [((1, 96, 48, 48), torch.float32), ((96, 1, 3, 3), torch.float32)]),
    (Conv2D259, [((1, 96, 48, 48), torch.float32), ((96, 1, 3, 3), torch.float32)]),
    (Conv2D549, [((1, 96, 48, 48), torch.float32)]),
    (Conv2D77, [((1, 96, 24, 24), torch.float32)]),
    (Conv2D210, [((1, 192, 24, 24), torch.float32), ((192, 1, 3, 3), torch.float32)]),
    (Conv2D550, [((1, 192, 24, 24), torch.float32), ((192, 1, 3, 3), torch.float32)]),
    (Conv2D80, [((1, 192, 24, 24), torch.float32)]),
    (Conv2D551, [((1, 192, 12, 12), torch.float32)]),
    (Conv2D552, [((1, 384, 12, 12), torch.float32), ((384, 1, 3, 3), torch.float32)]),
    (Conv2D553, [((1, 384, 12, 12), torch.float32), ((384, 1, 3, 3), torch.float32)]),
    (Conv2D554, [((1, 384, 12, 12), torch.float32)]),
    (Conv2D555, [((1, 384, 6, 6), torch.float32)]),
    (Conv2D556, [((1, 768, 6, 6), torch.float32), ((768, 1, 3, 3), torch.float32)]),
    (Conv2D557, [((1, 768, 6, 6), torch.float32)]),
    (Conv2D251, [((1, 3, 64, 64), torch.float32)]),
    (Conv2D253, [((1, 32, 32, 32), torch.float32), ((32, 1, 3, 3), torch.float32)]),
    (Conv2D128, [((1, 32, 32, 32), torch.float32)]),
    (Conv2D558, [((1, 64, 32, 32), torch.float32), ((64, 1, 3, 3), torch.float32)]),
    (Conv2D13, [((1, 64, 16, 16), torch.float32)]),
    (Conv2D559, [((1, 128, 16, 16), torch.float32), ((128, 1, 3, 3), torch.float32)]),
    (Conv2D560, [((1, 128, 16, 16), torch.float32), ((128, 1, 3, 3), torch.float32)]),
    (Conv2D17, [((1, 128, 16, 16), torch.float32)]),
    (Conv2D136, [((1, 128, 8, 8), torch.float32)]),
    (Conv2D561, [((1, 256, 8, 8), torch.float32), ((256, 1, 3, 3), torch.float32)]),
    (Conv2D562, [((1, 256, 8, 8), torch.float32), ((256, 1, 3, 3), torch.float32)]),
    (Conv2D138, [((1, 256, 8, 8), torch.float32)]),
    (Conv2D156, [((1, 256, 4, 4), torch.float32)]),
    (Conv2D563, [((1, 512, 4, 4), torch.float32), ((512, 1, 3, 3), torch.float32)]),
    (Conv2D564, [((1, 512, 4, 4), torch.float32), ((512, 1, 3, 3), torch.float32)]),
    (Conv2D177, [((1, 512, 4, 4), torch.float32)]),
    (Conv2D162, [((1, 512, 2, 2), torch.float32)]),
    (Conv2D565, [((1, 1024, 2, 2), torch.float32), ((1024, 1, 3, 3), torch.float32)]),
    (Conv2D187, [((1, 1024, 2, 2), torch.float32)]),
    (Conv2D566, [((1, 64, 112, 112), torch.float32), ((64, 1, 3, 3), torch.float32)]),
    (Conv2D558, [((1, 64, 112, 112), torch.float32), ((64, 1, 3, 3), torch.float32)]),
    (Conv2D567, [((1, 64, 112, 112), torch.float32), ((64, 1, 3, 3), torch.float32)]),
    (Conv2D559, [((1, 128, 56, 56), torch.float32), ((128, 1, 3, 3), torch.float32)]),
    (Conv2D568, [((1, 128, 56, 56), torch.float32), ((128, 1, 3, 3), torch.float32)]),
    (Conv2D561, [((1, 256, 28, 28), torch.float32), ((256, 1, 3, 3), torch.float32)]),
    (Conv2D569, [((1, 256, 28, 28), torch.float32), ((256, 1, 3, 3), torch.float32)]),
    (Conv2D563, [((1, 512, 14, 14), torch.float32), ((512, 1, 3, 3), torch.float32)]),
    (Conv2D570, [((1, 512, 14, 14), torch.float32), ((512, 1, 3, 3), torch.float32)]),
    (Conv2D565, [((1, 1024, 7, 7), torch.float32), ((1024, 1, 3, 3), torch.float32)]),
    (Conv2D208, [((1, 144, 28, 28), torch.float32)]),
    (Conv2D209, [((1, 32, 28, 28), torch.float32)]),
    (Conv2D210, [((1, 192, 28, 28), torch.float32), ((192, 1, 3, 3), torch.float32)]),
    (Conv2D571, [((1, 192, 28, 28), torch.float32), ((192, 1, 3, 3), torch.float32)]),
    (Conv2D550, [((1, 192, 28, 28), torch.float32), ((192, 1, 3, 3), torch.float32)]),
    (Conv2D572, [((1, 192, 14, 14), torch.float32)]),
    (Conv2D573, [((1, 64, 14, 14), torch.float32)]),
    (Conv2D552, [((1, 384, 14, 14), torch.float32), ((384, 1, 3, 3), torch.float32)]),
    (Conv2D574, [((1, 384, 14, 14), torch.float32)]),
    (Conv2D518, [((1, 384, 14, 14), torch.float32)]),
    (Conv2D575, [((1, 96, 14, 14), torch.float32)]),
    (Conv2D576, [((1, 576, 14, 14), torch.float32), ((576, 1, 3, 3), torch.float32)]),
    (Conv2D577, [((1, 576, 14, 14), torch.float32), ((576, 1, 3, 3), torch.float32)]),
    (Conv2D578, [((1, 576, 14, 14), torch.float32), ((576, 1, 3, 3), torch.float32)]),
    (Conv2D579, [((1, 576, 14, 14), torch.float32)]),
    (Conv2D580, [((1, 576, 7, 7), torch.float32)]),
    (Conv2D581, [((1, 960, 7, 7), torch.float32), ((960, 1, 3, 3), torch.float32)]),
    (Conv2D236, [((1, 960, 7, 7), torch.float32)]),
    (Conv2D582, [((1, 960, 7, 7), torch.float32)]),
    (Conv2D544, [((1, 3, 160, 160), torch.float32)]),
    (Conv2D200, [((1, 24, 80, 80), torch.float32), ((24, 1, 3, 3), torch.float32)]),
    (Conv2D583, [((1, 24, 80, 80), torch.float32)]),
    (Conv2D257, [((1, 16, 80, 80), torch.float32)]),
    (Conv2D259, [((1, 96, 80, 80), torch.float32), ((96, 1, 3, 3), torch.float32)]),
    (Conv2D262, [((1, 96, 40, 40), torch.float32)]),
    (Conv2D204, [((1, 24, 40, 40), torch.float32)]),
    (Conv2D263, [((1, 144, 40, 40), torch.float32), ((144, 1, 3, 3), torch.float32)]),
    (Conv2D264, [((1, 144, 40, 40), torch.float32), ((144, 1, 3, 3), torch.float32)]),
    (Conv2D265, [((1, 144, 40, 40), torch.float32)]),
    (Conv2D265, [((1, 144, 20, 20), torch.float32)]),
    (Conv2D204, [((1, 24, 20, 20), torch.float32)]),
    (Conv2D263, [((1, 144, 20, 20), torch.float32), ((144, 1, 3, 3), torch.float32)]),
    (Conv2D264, [((1, 144, 20, 20), torch.float32), ((144, 1, 3, 3), torch.float32)]),
    (Conv2D584, [((1, 144, 10, 10), torch.float32)]),
    (Conv2D585, [((1, 48, 10, 10), torch.float32)]),
    (Conv2D586, [((1, 288, 10, 10), torch.float32), ((288, 1, 3, 3), torch.float32)]),
    (Conv2D587, [((1, 288, 10, 10), torch.float32)]),
    (Conv2D588, [((1, 288, 10, 10), torch.float32)]),
    (Conv2D589, [((1, 72, 10, 10), torch.float32)]),
    (Conv2D590, [((1, 432, 10, 10), torch.float32), ((432, 1, 3, 3), torch.float32)]),
    (Conv2D591, [((1, 432, 10, 10), torch.float32), ((432, 1, 3, 3), torch.float32)]),
    (Conv2D592, [((1, 432, 10, 10), torch.float32)]),
    (Conv2D593, [((1, 432, 5, 5), torch.float32)]),
    (Conv2D594, [((1, 120, 5, 5), torch.float32)]),
    (Conv2D595, [((1, 720, 5, 5), torch.float32), ((720, 1, 3, 3), torch.float32)]),
    (Conv2D596, [((1, 720, 5, 5), torch.float32)]),
    (Conv2D597, [((1, 720, 5, 5), torch.float32)]),
    (Conv2D598, [((1, 240, 5, 5), torch.float32)]),
    (Conv2D599, [((1, 3, 96, 96), torch.float32)]),
    (Conv2D296, [((1, 16, 48, 48), torch.float32), ((16, 1, 3, 3), torch.float32)]),
    (Conv2D291, [((1, 16, 48, 48), torch.float32)]),
    (Conv2D600, [((1, 8, 48, 48), torch.float32)]),
    (Conv2D546, [((1, 48, 48, 48), torch.float32), ((48, 1, 3, 3), torch.float32)]),
    (Conv2D601, [((1, 48, 24, 24), torch.float32)]),
    (Conv2D600, [((1, 8, 24, 24), torch.float32)]),
    (Conv2D195, [((1, 48, 24, 24), torch.float32), ((48, 1, 3, 3), torch.float32)]),
    (Conv2D546, [((1, 48, 24, 24), torch.float32), ((48, 1, 3, 3), torch.float32)]),
    (Conv2D602, [((1, 48, 12, 12), torch.float32)]),
    (Conv2D257, [((1, 16, 12, 12), torch.float32)]),
    (Conv2D548, [((1, 96, 12, 12), torch.float32), ((96, 1, 3, 3), torch.float32)]),
    (Conv2D259, [((1, 96, 12, 12), torch.float32), ((96, 1, 3, 3), torch.float32)]),
    (Conv2D603, [((1, 96, 12, 12), torch.float32)]),
    (Conv2D262, [((1, 96, 6, 6), torch.float32)]),
    (Conv2D204, [((1, 24, 6, 6), torch.float32)]),
    (Conv2D263, [((1, 144, 6, 6), torch.float32), ((144, 1, 3, 3), torch.float32)]),
    (Conv2D265, [((1, 144, 6, 6), torch.float32)]),
    (Conv2D208, [((1, 144, 6, 6), torch.float32)]),
    (Conv2D209, [((1, 32, 6, 6), torch.float32)]),
    (Conv2D210, [((1, 192, 6, 6), torch.float32), ((192, 1, 3, 3), torch.float32)]),
    (Conv2D550, [((1, 192, 6, 6), torch.float32), ((192, 1, 3, 3), torch.float32)]),
    (Conv2D213, [((1, 192, 6, 6), torch.float32)]),
    (Conv2D215, [((1, 192, 3, 3), torch.float32)]),
    (Conv2D216, [((1, 56, 3, 3), torch.float32)]),
    (Conv2D331, [((1, 336, 3, 3), torch.float32), ((336, 1, 3, 3), torch.float32)]),
    (Conv2D220, [((1, 336, 3, 3), torch.float32)]),
    (Conv2D222, [((1, 336, 3, 3), torch.float32)]),
    (Conv2D604, [((1, 112, 3, 3), torch.float32)]),
    (Conv2D572, [((1, 192, 28, 28), torch.float32)]),
    (Conv2D573, [((1, 64, 28, 28), torch.float32)]),
    (Conv2D605, [((1, 384, 28, 28), torch.float32), ((384, 1, 3, 3), torch.float32)]),
    (Conv2D574, [((1, 384, 28, 28), torch.float32)]),
    (Conv2D518, [((1, 384, 28, 28), torch.float32)]),
    (Conv2D575, [((1, 96, 28, 28), torch.float32)]),
    (Conv2D606, [((1, 576, 28, 28), torch.float32), ((576, 1, 3, 3), torch.float32)]),
    (Conv2D579, [((1, 576, 28, 28), torch.float32)]),
    (Conv2D580, [((1, 576, 28, 28), torch.float32)]),
    (Conv2D231, [((1, 160, 28, 28), torch.float32)]),
    (Conv2D607, [((1, 960, 28, 28), torch.float32), ((960, 1, 3, 3), torch.float32)]),
    (Conv2D236, [((1, 960, 28, 28), torch.float32)]),
    (Conv2D582, [((1, 960, 28, 28), torch.float32)]),
    (Conv2D410, [((1, 320, 1, 1), torch.float32)]),
    (Conv2D410, [((1, 320, 28, 28), torch.float32)]),
    (Conv2D608, [((1, 256, 28, 28), torch.float32)]),
    (Conv2D291, [((1, 16, 1, 1), torch.float32)]),
    (Conv2D609, [((1, 8, 1, 1), torch.float32)]),
    (Conv2D610, [((1, 16, 56, 56), torch.float32)]),
    (Conv2D611, [((1, 16, 56, 56), torch.float32)]),
    (Conv2D612, [((1, 72, 56, 56), torch.float32), ((72, 1, 3, 3), torch.float32)]),
    (Conv2D613, [((1, 72, 56, 56), torch.float32), ((72, 1, 3, 3), torch.float32)]),
    (Conv2D614, [((1, 72, 28, 28), torch.float32)]),
    (Conv2D615, [((1, 24, 28, 28), torch.float32)]),
    (Conv2D616, [((1, 88, 28, 28), torch.float32), ((88, 1, 3, 3), torch.float32)]),
    (Conv2D617, [((1, 88, 28, 28), torch.float32)]),
    (Conv2D618, [((1, 24, 28, 28), torch.float32)]),
    (Conv2D619, [((1, 96, 28, 28), torch.float32), ((96, 1, 5, 5), torch.float32)]),
    (Conv2D262, [((1, 96, 1, 1), torch.float32)]),
    (Conv2D618, [((1, 24, 1, 1), torch.float32)]),
    (Conv2D620, [((1, 96, 14, 14), torch.float32)]),
    (Conv2D268, [((1, 40, 14, 14), torch.float32)]),
    (Conv2D269, [((1, 240, 14, 14), torch.float32), ((240, 1, 5, 5), torch.float32)]),
    (Conv2D621, [((1, 240, 1, 1), torch.float32)]),
    (Conv2D622, [((1, 64, 1, 1), torch.float32)]),
    (Conv2D311, [((1, 40, 14, 14), torch.float32)]),
    (Conv2D623, [((1, 120, 14, 14), torch.float32), ((120, 1, 5, 5), torch.float32)]),
    (Conv2D624, [((1, 120, 14, 14), torch.float32)]),
    (Conv2D625, [((1, 48, 14, 14), torch.float32)]),
    (Conv2D626, [((1, 144, 14, 14), torch.float32), ((144, 1, 5, 5), torch.float32)]),
    (Conv2D267, [((1, 144, 1, 1), torch.float32)]),
    (Conv2D627, [((1, 40, 1, 1), torch.float32)]),
    (Conv2D584, [((1, 144, 14, 14), torch.float32)]),
    (Conv2D585, [((1, 48, 14, 14), torch.float32)]),
    (Conv2D628, [((1, 288, 14, 14), torch.float32), ((288, 1, 5, 5), torch.float32)]),
    (Conv2D588, [((1, 288, 1, 1), torch.float32)]),
    (Conv2D629, [((1, 72, 1, 1), torch.float32)]),
    (Conv2D630, [((1, 288, 7, 7), torch.float32)]),
    (Conv2D575, [((1, 96, 7, 7), torch.float32)]),
    (Conv2D631, [((1, 576, 7, 7), torch.float32), ((576, 1, 5, 5), torch.float32)]),
    (Conv2D632, [((1, 576, 1, 1), torch.float32)]),
    (Conv2D633, [((1, 144, 1, 1), torch.float32)]),
    (Conv2D579, [((1, 576, 7, 7), torch.float32)]),
    (Conv2D634, [((1, 576, 1, 1), torch.float32)]),
    (Conv2D610, [((1, 16, 112, 112), torch.float32)]),
    (Conv2D635, [((1, 16, 112, 112), torch.float32)]),
    (Conv2D636, [((1, 64, 56, 56), torch.float32)]),
    (Conv2D637, [((1, 24, 56, 56), torch.float32)]),
    (Conv2D614, [((1, 72, 56, 56), torch.float32)]),
    (Conv2D614, [((1, 72, 1, 1), torch.float32)]),
    (Conv2D637, [((1, 24, 1, 1), torch.float32)]),
    (Conv2D638, [((1, 72, 28, 28), torch.float32)]),
    (Conv2D623, [((1, 120, 28, 28), torch.float32), ((120, 1, 5, 5), torch.float32)]),
    (Conv2D639, [((1, 120, 28, 28), torch.float32)]),
    (Conv2D640, [((1, 80, 14, 14), torch.float32)]),
    (Conv2D641, [((1, 200, 14, 14), torch.float32), ((200, 1, 3, 3), torch.float32)]),
    (Conv2D642, [((1, 200, 14, 14), torch.float32)]),
    (Conv2D643, [((1, 80, 14, 14), torch.float32)]),
    (Conv2D644, [((1, 184, 14, 14), torch.float32), ((184, 1, 3, 3), torch.float32)]),
    (Conv2D645, [((1, 184, 14, 14), torch.float32)]),
    (Conv2D230, [((1, 672, 7, 7), torch.float32)]),
    (Conv2D232, [((1, 960, 7, 7), torch.float32), ((960, 1, 5, 5), torch.float32)]),
    (Conv2D12, [((1, 3, 320, 1024), torch.float32)]),
    (Conv2D147, [((1, 64, 80, 256), torch.float32)]),
    (Conv2D189, [((1, 64, 80, 256), torch.float32)]),
    (Conv2D149, [((1, 128, 40, 128), torch.float32)]),
    (Conv2D646, [((1, 64, 80, 256), torch.float32)]),
    (Conv2D190, [((1, 128, 40, 128), torch.float32)]),
    (Conv2D6, [((1, 256, 20, 64), torch.float32)]),
    (Conv2D647, [((1, 128, 40, 128), torch.float32)]),
    (Conv2D191, [((1, 256, 20, 64), torch.float32)]),
    (Conv2D179, [((1, 512, 10, 32), torch.float32)]),
    (Conv2D648, [((1, 256, 20, 64), torch.float32)]),
    (Conv2D649, [((1, 512, 12, 34), torch.float32)]),
    (Conv2D649, [((1, 512, 22, 66), torch.float32)]),
    (Conv2D650, [((1, 256, 22, 66), torch.float32)]),
    (Conv2D650, [((1, 256, 42, 130), torch.float32)]),
    (Conv2D651, [((1, 128, 42, 130), torch.float32)]),
    (Conv2D651, [((1, 128, 82, 258), torch.float32)]),
    (Conv2D652, [((1, 64, 82, 258), torch.float32)]),
    (Conv2D653, [((1, 96, 162, 514), torch.float32)]),
    (Conv2D654, [((1, 32, 162, 514), torch.float32)]),
    (Conv2D655, [((1, 16, 322, 1026), torch.float32)]),
    (Conv2D656, [((1, 16, 322, 1026), torch.float32)]),
    (Conv2D12, [((1, 3, 192, 640), torch.float32)]),
    (Conv2D147, [((1, 64, 48, 160), torch.float32)]),
    (Conv2D189, [((1, 64, 48, 160), torch.float32)]),
    (Conv2D149, [((1, 128, 24, 80), torch.float32)]),
    (Conv2D646, [((1, 64, 48, 160), torch.float32)]),
    (Conv2D190, [((1, 128, 24, 80), torch.float32)]),
    (Conv2D6, [((1, 256, 12, 40), torch.float32)]),
    (Conv2D647, [((1, 128, 24, 80), torch.float32)]),
    (Conv2D191, [((1, 256, 12, 40), torch.float32)]),
    (Conv2D179, [((1, 512, 6, 20), torch.float32)]),
    (Conv2D648, [((1, 256, 12, 40), torch.float32)]),
    (Conv2D649, [((1, 512, 8, 22), torch.float32)]),
    (Conv2D649, [((1, 512, 14, 42), torch.float32)]),
    (Conv2D650, [((1, 256, 14, 42), torch.float32)]),
    (Conv2D650, [((1, 256, 26, 82), torch.float32)]),
    (Conv2D651, [((1, 128, 26, 82), torch.float32)]),
    (Conv2D651, [((1, 128, 50, 162), torch.float32)]),
    (Conv2D652, [((1, 64, 50, 162), torch.float32)]),
    (Conv2D653, [((1, 96, 98, 322), torch.float32)]),
    (Conv2D654, [((1, 32, 98, 322), torch.float32)]),
    (Conv2D655, [((1, 16, 194, 642), torch.float32)]),
    (Conv2D656, [((1, 16, 194, 642), torch.float32)]),
    (Conv2D657, [((1, 128, 28, 28), torch.float32)]),
    (Conv2D657, [((1, 128, 56, 56), torch.float32)]),
    (Conv2D658, [((1, 256, 14, 14), torch.float32)]),
    (Conv2D658, [((1, 256, 28, 28), torch.float32)]),
    (Conv2D659, [((1, 512, 7, 7), torch.float32)]),
    (Conv2D659, [((1, 512, 14, 14), torch.float32)]),
    (Conv2D660, [((1, 64, 56, 56), torch.float32), ((1792, 64, 3, 3), torch.float32)]),
    (Conv2D661, [((1, 256, 56, 56), torch.float32)]),
    (Conv2D662, [((1, 256, 56, 56), torch.float32)]),
    (Conv2D663, [((1, 256, 56, 56), torch.float32)]),
    (Conv2D664, [((1, 3, 224, 224), torch.float32)]),
    (Conv2D665, [((1, 3, 227, 227), torch.float32)]),
    (Conv2D666, [((1, 64, 27, 27), torch.float32)]),
    (Conv2D667, [((1, 192, 13, 13), torch.float32)]),
    (Conv2D668, [((1, 384, 13, 13), torch.float32)]),
    (Conv2D669, [((1, 256, 13, 13), torch.float32)]),
    (Conv2D670, [((1, 128, 112, 112), torch.float32)]),
    (Conv2D671, [((1, 128, 1, 1), torch.float32)]),
    (Conv2D672, [((1, 8, 1, 1), torch.float32)]),
    (Conv2D673, [((1, 128, 56, 56), torch.float32)]),
    (Conv2D424, [((1, 128, 1, 1), torch.float32)]),
    (Conv2D152, [((1, 32, 1, 1), torch.float32)]),
    (Conv2D674, [((1, 128, 56, 56), torch.float32)]),
    (Conv2D675, [((1, 128, 56, 56), torch.float32)]),
    (Conv2D676, [((1, 192, 56, 56), torch.float32)]),
    (Conv2D213, [((1, 192, 1, 1), torch.float32)]),
    (Conv2D209, [((1, 32, 1, 1), torch.float32)]),
    (Conv2D677, [((1, 192, 28, 28), torch.float32)]),
    (Conv2D470, [((1, 192, 1, 1), torch.float32)]),
    (Conv2D678, [((1, 48, 1, 1), torch.float32)]),
    (Conv2D479, [((1, 192, 28, 28), torch.float32)]),
    (Conv2D679, [((1, 192, 28, 28), torch.float32)]),
    (Conv2D680, [((1, 512, 28, 28), torch.float32)]),
    (Conv2D681, [((1, 512, 1, 1), torch.float32)]),
    (Conv2D682, [((1, 48, 1, 1), torch.float32)]),
    (Conv2D683, [((1, 512, 14, 14), torch.float32)]),
    (Conv2D30, [((1, 512, 1, 1), torch.float32)]),
    (Conv2D154, [((1, 128, 1, 1), torch.float32)]),
    (Conv2D684, [((1, 512, 14, 14), torch.float32)]),
    (Conv2D685, [((1, 512, 14, 14), torch.float32)]),
    (Conv2D686, [((1, 1088, 14, 14), torch.float32)]),
    (Conv2D48, [((1, 1088, 1, 1), torch.float32)]),
    (Conv2D687, [((1, 128, 1, 1), torch.float32)]),
    (Conv2D688, [((1, 1088, 7, 7), torch.float32)]),
    (Conv2D689, [((1, 1088, 7, 7), torch.float32)]),
    (Conv2D690, [((1, 1088, 1, 1), torch.float32)]),
    (Conv2D691, [((1, 272, 1, 1), torch.float32)]),
    (Conv2D154, [((1, 128, 28, 28), torch.float32)]),
    (Conv2D648, [((1, 256, 56, 56), torch.float32)]),
    (Conv2D156, [((1, 256, 56, 56), torch.float32)]),
    (Conv2D30, [((1, 512, 28, 28), torch.float32)]),
    (Conv2D160, [((1, 256, 14, 14), torch.float32)]),
    (Conv2D692, [((1, 512, 28, 28), torch.float32)]),
    (Conv2D162, [((1, 512, 28, 28), torch.float32)]),
    (Conv2D528, [((1, 1024, 14, 14), torch.float32)]),
    (Conv2D167, [((1, 512, 7, 7), torch.float32)]),
    (Conv2D693, [((1, 1024, 14, 14), torch.float32)]),
    (Conv2D170, [((1, 1024, 14, 14), torch.float32)]),
    (Conv2D165, [((1, 2048, 7, 7), torch.float32)]),
    (Conv2D138, [((1, 256, 56, 56), torch.float32)]),
    (Conv2D183, [((1, 512, 56, 56), torch.float32)]),
    (Conv2D177, [((1, 512, 28, 28), torch.float32)]),
    (Conv2D185, [((1, 1024, 28, 28), torch.float32)]),
    (Conv2D187, [((1, 1024, 14, 14), torch.float32)]),
    (Conv2D694, [((1, 2048, 14, 14), torch.float32)]),
    (Conv2D695, [((1, 2048, 7, 7), torch.float32)]),
    (Conv2D696, [((1, 2048, 7, 7), torch.float32)]),
    (Conv2D12, [((1, 3, 480, 640), torch.float32)]),
    (Conv2D147, [((1, 64, 120, 160), torch.float32)]),
    (Conv2D189, [((1, 64, 120, 160), torch.float32)]),
    (Conv2D149, [((1, 128, 60, 80), torch.float32)]),
    (Conv2D646, [((1, 64, 120, 160), torch.float32)]),
    (Conv2D190, [((1, 128, 60, 80), torch.float32)]),
    (Conv2D6, [((1, 256, 30, 40), torch.float32)]),
    (Conv2D647, [((1, 128, 60, 80), torch.float32)]),
    (Conv2D136, [((1, 128, 60, 80), torch.float32)]),
    (Conv2D191, [((1, 256, 30, 40), torch.float32)]),
    (Conv2D179, [((1, 512, 15, 20), torch.float32)]),
    (Conv2D648, [((1, 256, 30, 40), torch.float32)]),
    (Conv2D29, [((1, 512, 15, 20), torch.float32)]),
    (Conv2D138, [((1, 256, 30, 40), torch.float32)]),
    (Conv2D6, [((1, 256, 60, 80), torch.float32)]),
    (Conv2D174, [((1, 256, 60, 80), torch.float32)]),
    (Conv2D697, [((1, 256, 60, 80), torch.float32)]),
    (Conv2D697, [((1, 256, 30, 40), torch.float32)]),
    (Conv2D6, [((1, 256, 15, 20), torch.float32)]),
    (Conv2D697, [((1, 256, 15, 20), torch.float32)]),
    (Conv2D698, [((1, 512, 15, 20), torch.float32)]),
    (Conv2D6, [((1, 256, 8, 10), torch.float32)]),
    (Conv2D174, [((1, 256, 8, 10), torch.float32)]),
    (Conv2D697, [((1, 256, 8, 10), torch.float32)]),
    (Conv2D6, [((1, 256, 4, 5), torch.float32)]),
    (Conv2D697, [((1, 256, 4, 5), torch.float32)]),
    (Conv2D699, [((1, 256, 60, 80), torch.float32)]),
    (Conv2D699, [((1, 256, 30, 40), torch.float32)]),
    (Conv2D699, [((1, 256, 15, 20), torch.float32)]),
    (Conv2D699, [((1, 256, 8, 10), torch.float32)]),
    (Conv2D699, [((1, 256, 4, 5), torch.float32)]),
    (Conv2D130, [((1, 64, 120, 160), torch.float32)]),
    (Conv2D365, [((1, 64, 120, 160), torch.float32)]),
    (Conv2D133, [((1, 256, 120, 160), torch.float32)]),
    (Conv2D21, [((1, 256, 120, 160), torch.float32)]),
    (Conv2D148, [((1, 128, 120, 160), torch.float32)]),
    (Conv2D154, [((1, 128, 60, 80), torch.float32)]),
    (Conv2D648, [((1, 256, 120, 160), torch.float32)]),
    (Conv2D30, [((1, 512, 60, 80), torch.float32)]),
    (Conv2D29, [((1, 512, 60, 80), torch.float32)]),
    (Conv2D160, [((1, 256, 30, 40), torch.float32)]),
    (Conv2D692, [((1, 512, 60, 80), torch.float32)]),
    (Conv2D528, [((1, 1024, 30, 40), torch.float32)]),
    (Conv2D75, [((1, 1024, 30, 40), torch.float32)]),
    (Conv2D178, [((1, 512, 30, 40), torch.float32)]),
    (Conv2D167, [((1, 512, 15, 20), torch.float32)]),
    (Conv2D693, [((1, 1024, 30, 40), torch.float32)]),
    (Conv2D165, [((1, 2048, 15, 20), torch.float32)]),
    (Conv2D289, [((1, 2048, 15, 20), torch.float32)]),
    (Conv2D700, [((1, 2048, 15, 20), torch.float32)]),
    (Conv2D701, [((1, 3, 512, 512), torch.float32)]),
    (Conv2D702, [((1, 32, 128, 128), torch.float32)]),
    (Conv2D559, [((1, 128, 128, 128), torch.float32), ((128, 1, 3, 3), torch.float32)]),
    (Conv2D188, [((1, 32, 128, 128), torch.float32)]),
    (Conv2D356, [((1, 32, 128, 128), torch.float32)]),
    (Conv2D703, [((1, 64, 64, 64), torch.float32)]),
    (Conv2D561, [((1, 256, 64, 64), torch.float32), ((256, 1, 3, 3), torch.float32)]),
    (Conv2D704, [((1, 64, 64, 64), torch.float32)]),
    (Conv2D705, [((1, 160, 32, 32), torch.float32)]),
    (Conv2D706, [((1, 640, 32, 32), torch.float32), ((640, 1, 3, 3), torch.float32)]),
    (Conv2D707, [((1, 160, 32, 32), torch.float32)]),
    (Conv2D565, [((1, 1024, 16, 16), torch.float32), ((1024, 1, 3, 3), torch.float32)]),
    (Conv2D528, [((1, 1024, 128, 128), torch.float32)]),
    (Conv2D708, [((1, 256, 128, 128), torch.float32)]),
    (Conv2D709, [((1, 3, 512, 512), torch.float32)]),
    (Conv2D710, [((1, 64, 128, 128), torch.float32)]),
    (Conv2D561, [((1, 256, 128, 128), torch.float32), ((256, 1, 3, 3), torch.float32)]),
    (Conv2D189, [((1, 64, 128, 128), torch.float32)]),
    (Conv2D711, [((1, 128, 64, 64), torch.float32)]),
    (Conv2D563, [((1, 512, 64, 64), torch.float32), ((512, 1, 3, 3), torch.float32)]),
    (Conv2D712, [((1, 128, 64, 64), torch.float32)]),
    (Conv2D713, [((1, 320, 32, 32), torch.float32)]),
    (Conv2D714, [((1, 1280, 32, 32), torch.float32), ((1280, 1, 3, 3), torch.float32)]),
    (Conv2D715, [((1, 320, 32, 32), torch.float32)]),
    (Conv2D716, [((1, 2048, 16, 16), torch.float32), ((2048, 1, 3, 3), torch.float32)]),
    (Conv2D717, [((1, 3072, 128, 128), torch.float32)]),
    (Conv2D718, [((1, 768, 128, 128), torch.float32)]),
    (Conv2D12, [((1, 3, 300, 300), torch.float32)]),
    (Conv2D130, [((1, 64, 75, 75), torch.float32)]),
    (Conv2D147, [((1, 64, 75, 75), torch.float32)]),
    (Conv2D365, [((1, 64, 75, 75), torch.float32)]),
    (Conv2D133, [((1, 256, 75, 75), torch.float32)]),
    (Conv2D21, [((1, 256, 75, 75), torch.float32)]),
    (Conv2D148, [((1, 128, 75, 75), torch.float32)]),
    (Conv2D154, [((1, 128, 38, 38), torch.float32)]),
    (Conv2D648, [((1, 256, 75, 75), torch.float32)]),
    (Conv2D30, [((1, 512, 38, 38), torch.float32)]),
    (Conv2D149, [((1, 128, 38, 38), torch.float32)]),
    (Conv2D29, [((1, 512, 38, 38), torch.float32)]),
    (Conv2D6, [((1, 256, 38, 38), torch.float32)]),
    (Conv2D160, [((1, 256, 38, 38), torch.float32)]),
    (Conv2D162, [((1, 512, 38, 38), torch.float32)]),
    (Conv2D528, [((1, 1024, 38, 38), torch.float32)]),
    (Conv2D719, [((1, 1024, 38, 38), torch.float32)]),
    (Conv2D191, [((1, 256, 38, 38), torch.float32)]),
    (Conv2D720, [((1, 512, 19, 19), torch.float32)]),
    (Conv2D29, [((1, 512, 19, 19), torch.float32)]),
    (Conv2D191, [((1, 256, 19, 19), torch.float32)]),
    (Conv2D720, [((1, 512, 10, 10), torch.float32)]),
    (Conv2D30, [((1, 512, 10, 10), torch.float32)]),
    (Conv2D190, [((1, 128, 10, 10), torch.float32)]),
    (Conv2D721, [((1, 256, 5, 5), torch.float32)]),
    (Conv2D21, [((1, 256, 5, 5), torch.float32)]),
    (Conv2D722, [((1, 128, 5, 5), torch.float32)]),
    (Conv2D723, [((1, 256, 3, 3), torch.float32)]),
    (Conv2D21, [((1, 256, 3, 3), torch.float32)]),
    (Conv2D722, [((1, 128, 3, 3), torch.float32)]),
    (Conv2D723, [((1, 256, 1, 1), torch.float32)]),
    (Conv2D724, [((1, 1024, 38, 38), torch.float32)]),
    (Conv2D725, [((1, 512, 19, 19), torch.float32)]),
    (Conv2D725, [((1, 512, 10, 10), torch.float32)]),
    (Conv2D726, [((1, 256, 5, 5), torch.float32)]),
    (Conv2D727, [((1, 256, 3, 3), torch.float32)]),
    (Conv2D727, [((1, 256, 1, 1), torch.float32)]),
    (Conv2D728, [((1, 3, 256, 256), torch.float32)]),
    (Conv2D144, [((1, 32, 256, 256), torch.float32)]),
    (Conv2D147, [((1, 64, 128, 128), torch.float32)]),
    (Conv2D729, [((1, 64, 64, 64), torch.float32)]),
    (Conv2D149, [((1, 128, 64, 64), torch.float32)]),
    (Conv2D354, [((1, 128, 32, 32), torch.float32)]),
    (Conv2D6, [((1, 256, 32, 32), torch.float32)]),
    (Conv2D392, [((1, 256, 16, 16), torch.float32)]),
    (Conv2D179, [((1, 512, 16, 16), torch.float32)]),
    (Conv2D659, [((1, 512, 32, 32), torch.float32)]),
    (Conv2D658, [((1, 256, 64, 64), torch.float32)]),
    (Conv2D657, [((1, 128, 128, 128), torch.float32)]),
    (Conv2D730, [((1, 64, 256, 256), torch.float32)]),
    (Conv2D731, [((1, 32, 256, 256), torch.float32)]),
    (Conv2D732, [((1, 3072, 14, 14), torch.float32)]),
    (Conv2D733, [((1, 768, 28, 28), torch.float32)]),
    (Conv2D734, [((1, 384, 56, 56), torch.float32)]),
    (Conv2D14, [((1, 128, 112, 112), torch.float32)]),
    (Conv2D735, [((1, 32, 224, 224), torch.float32)]),
    (Conv2D736, [((1, 16, 224, 224), torch.float32)]),
    (Conv2D147, [((1, 64, 224, 224), torch.float32)]),
    (Conv2D729, [((1, 64, 112, 112), torch.float32)]),
    (Conv2D189, [((1, 64, 112, 112), torch.float32)]),
    (Conv2D149, [((1, 128, 112, 112), torch.float32)]),
    (Conv2D6, [((1, 256, 56, 56), torch.float32)]),
    (Conv2D174, [((1, 256, 56, 56), torch.float32)]),
    (Conv2D179, [((1, 512, 28, 28), torch.float32)]),
    (Conv2D178, [((1, 512, 28, 28), torch.float32)]),
    (Conv2D737, [((1, 1024, 28, 28), torch.float32)]),
    (Conv2D738, [((1, 512, 56, 56), torch.float32)]),
    (Conv2D417, [((1, 256, 112, 112), torch.float32)]),
    (Conv2D657, [((1, 128, 224, 224), torch.float32)]),
    (Conv2D739, [((1, 64, 224, 224), torch.float32)]),
    (Conv2D740, [((1, 512, 7, 7), torch.float32)]),
    (Conv2D741, [((1, 4096, 1, 1), torch.float32)]),
    (Conv2D158, [((1, 768, 56, 56), torch.float32)]),
    (Conv2D742, [((1, 256, 28, 28), torch.float32)]),
    (Conv2D403, [((1, 160, 28, 28), torch.float32)]),
    (Conv2D743, [((1, 1056, 28, 28), torch.float32)]),
    (Conv2D744, [((1, 512, 14, 14), torch.float32)]),
    (Conv2D745, [((1, 1472, 14, 14), torch.float32)]),
    (Conv2D746, [((1, 768, 14, 14), torch.float32)]),
    (Conv2D747, [((1, 1728, 14, 14), torch.float32)]),
    (Conv2D748, [((1, 768, 7, 7), torch.float32)]),
    (Conv2D749, [((1, 224, 7, 7), torch.float32)]),
    (Conv2D750, [((1, 1888, 7, 7), torch.float32)]),
    (Conv2D751, [((1, 1024, 7, 7), torch.float32)]),
    (Conv2D752, [((1, 2144, 7, 7), torch.float32)]),
    (Conv2D138, [((1, 256, 1, 1), torch.float32)]),
    (Conv2D177, [((1, 512, 1, 1), torch.float32)]),
    (Conv2D557, [((1, 768, 1, 1), torch.float32)]),
    (Conv2D187, [((1, 1024, 1, 1), torch.float32)]),
    (Conv2D753, [((1, 512, 28, 28), torch.float32)]),
    (Conv2D754, [((1, 1312, 28, 28), torch.float32)]),
    (Conv2D27, [((1, 448, 56, 56), torch.float32)]),
    (Conv2D755, [((1, 128, 28, 28), torch.float32)]),
    (Conv2D756, [((1, 528, 28, 28), torch.float32)]),
    (Conv2D757, [((1, 256, 14, 14), torch.float32)]),
    (Conv2D464, [((1, 96, 14, 14), torch.float32)]),
    (Conv2D758, [((1, 736, 14, 14), torch.float32)]),
    (Conv2D759, [((1, 384, 7, 7), torch.float32)]),
    (Conv2D760, [((1, 112, 7, 7), torch.float32)]),
    (Conv2D761, [((1, 944, 7, 7), torch.float32)]),
    (Conv2D130, [((1, 64, 112, 112), torch.float32)]),
    (Conv2D762, [((1, 448, 56, 56), torch.float32)]),
    (Conv2D763, [((1, 256, 28, 28), torch.float32)]),
    (Conv2D764, [((1, 160, 28, 28), torch.float32), ((160, 1, 3, 3), torch.float32)]),
    (Conv2D765, [((1, 160, 28, 28), torch.float32)]),
    (Conv2D766, [((1, 736, 28, 28), torch.float32)]),
    (Conv2D767, [((1, 512, 14, 14), torch.float32)]),
    (Conv2D210, [((1, 192, 14, 14), torch.float32), ((192, 1, 3, 3), torch.float32)]),
    (Conv2D80, [((1, 192, 14, 14), torch.float32)]),
    (Conv2D768, [((1, 1088, 14, 14), torch.float32)]),
    (Conv2D769, [((1, 768, 7, 7), torch.float32)]),
    (Conv2D770, [((1, 224, 7, 7), torch.float32), ((224, 1, 3, 3), torch.float32)]),
    (Conv2D771, [((1, 224, 7, 7), torch.float32)]),
    (Conv2D772, [((1, 1440, 7, 7), torch.float32)]),
    (Conv2D773, [((1, 1024, 14, 14), torch.float32)]),
    (Conv2D774, [((1, 1024, 7, 7), torch.float32)]),
    (Conv2D356, [((1, 32, 150, 150), torch.float32)]),
    (Conv2D567, [((1, 64, 150, 150), torch.float32), ((64, 1, 3, 3), torch.float32)]),
    (Conv2D13, [((1, 64, 150, 150), torch.float32)]),
    (Conv2D646, [((1, 64, 150, 150), torch.float32)]),
    (Conv2D559, [((1, 128, 150, 150), torch.float32), ((128, 1, 3, 3), torch.float32)]),
    (Conv2D560, [((1, 128, 150, 150), torch.float32), ((128, 1, 3, 3), torch.float32)]),
    (Conv2D17, [((1, 128, 150, 150), torch.float32)]),
    (Conv2D17, [((1, 128, 75, 75), torch.float32)]),
    (Conv2D559, [((1, 128, 75, 75), torch.float32), ((128, 1, 3, 3), torch.float32)]),
    (Conv2D136, [((1, 128, 75, 75), torch.float32)]),
    (Conv2D647, [((1, 128, 75, 75), torch.float32)]),
    (Conv2D561, [((1, 256, 75, 75), torch.float32), ((256, 1, 3, 3), torch.float32)]),
    (Conv2D562, [((1, 256, 75, 75), torch.float32), ((256, 1, 3, 3), torch.float32)]),
    (Conv2D138, [((1, 256, 75, 75), torch.float32)]),
    (Conv2D775, [((1, 256, 75, 75), torch.float32)]),
    (Conv2D138, [((1, 256, 38, 38), torch.float32)]),
    (Conv2D561, [((1, 256, 38, 38), torch.float32), ((256, 1, 3, 3), torch.float32)]),
    (Conv2D776, [((1, 256, 38, 38), torch.float32)]),
    (Conv2D777, [((1, 256, 38, 38), torch.float32)]),
    (Conv2D778, [((1, 728, 38, 38), torch.float32), ((728, 1, 3, 3), torch.float32)]),
    (Conv2D779, [((1, 728, 38, 38), torch.float32), ((728, 1, 3, 3), torch.float32)]),
    (Conv2D780, [((1, 728, 38, 38), torch.float32)]),
    (Conv2D781, [((1, 728, 38, 38), torch.float32)]),
    (Conv2D780, [((1, 728, 19, 19), torch.float32)]),
    (Conv2D778, [((1, 728, 19, 19), torch.float32), ((728, 1, 3, 3), torch.float32)]),
    (Conv2D782, [((1, 728, 19, 19), torch.float32)]),
    (Conv2D783, [((1, 728, 19, 19), torch.float32)]),
    (Conv2D784, [((1, 1024, 19, 19), torch.float32), ((1024, 1, 3, 3), torch.float32)]),
    (Conv2D187, [((1, 1024, 10, 10), torch.float32)]),
    (Conv2D565, [((1, 1024, 10, 10), torch.float32), ((1024, 1, 3, 3), torch.float32)]),
    (Conv2D785, [((1, 1024, 10, 10), torch.float32)]),
    (Conv2D786, [((1, 1536, 10, 10), torch.float32), ((1536, 1, 3, 3), torch.float32)]),
    (Conv2D787, [((1, 1536, 10, 10), torch.float32)]),
    (Conv2D788, [((1, 1536, 10, 10), torch.float32)]),
    (Conv2D789, [((1, 32, 149, 149), torch.float32)]),
    (Conv2D567, [((1, 64, 147, 147), torch.float32), ((64, 1, 3, 3), torch.float32)]),
    (Conv2D13, [((1, 64, 147, 147), torch.float32)]),
    (Conv2D646, [((1, 64, 147, 147), torch.float32)]),
    (Conv2D559, [((1, 128, 147, 147), torch.float32), ((128, 1, 3, 3), torch.float32)]),
    (Conv2D17, [((1, 128, 147, 147), torch.float32)]),
    (Conv2D559, [((1, 128, 74, 74), torch.float32), ((128, 1, 3, 3), torch.float32)]),
    (Conv2D136, [((1, 128, 74, 74), torch.float32)]),
    (Conv2D647, [((1, 128, 74, 74), torch.float32)]),
    (Conv2D561, [((1, 256, 74, 74), torch.float32), ((256, 1, 3, 3), torch.float32)]),
    (Conv2D138, [((1, 256, 74, 74), torch.float32)]),
    (Conv2D561, [((1, 256, 37, 37), torch.float32), ((256, 1, 3, 3), torch.float32)]),
    (Conv2D776, [((1, 256, 37, 37), torch.float32)]),
    (Conv2D777, [((1, 256, 37, 37), torch.float32)]),
    (Conv2D778, [((1, 728, 37, 37), torch.float32), ((728, 1, 3, 3), torch.float32)]),
    (Conv2D780, [((1, 728, 37, 37), torch.float32)]),
    (Conv2D790, [((1, 3, 640, 640), torch.float32)]),
    (Conv2D791, [((1, 80, 320, 320), torch.float32)]),
    (Conv2D792, [((1, 160, 160, 160), torch.float32)]),
    (Conv2D793, [((1, 80, 160, 160), torch.float32)]),
    (Conv2D794, [((1, 80, 160, 160), torch.float32)]),
    (Conv2D795, [((1, 160, 160, 160), torch.float32)]),
    (Conv2D796, [((1, 160, 160, 160), torch.float32)]),
    (Conv2D797, [((1, 320, 80, 80), torch.float32)]),
    (Conv2D795, [((1, 160, 80, 80), torch.float32)]),
    (Conv2D798, [((1, 160, 80, 80), torch.float32)]),
    (Conv2D799, [((1, 320, 80, 80), torch.float32)]),
    (Conv2D800, [((1, 320, 80, 80), torch.float32)]),
    (Conv2D801, [((1, 640, 40, 40), torch.float32)]),
    (Conv2D799, [((1, 320, 40, 40), torch.float32)]),
    (Conv2D802, [((1, 320, 40, 40), torch.float32)]),
    (Conv2D803, [((1, 320, 40, 40), torch.float32)]),
    (Conv2D804, [((1, 640, 40, 40), torch.float32)]),
    (Conv2D805, [((1, 640, 40, 40), torch.float32)]),
    (Conv2D806, [((1, 1280, 20, 20), torch.float32)]),
    (Conv2D804, [((1, 640, 20, 20), torch.float32)]),
    (Conv2D807, [((1, 640, 20, 20), torch.float32)]),
    (Conv2D808, [((1, 640, 20, 20), torch.float32)]),
    (Conv2D809, [((1, 1280, 20, 20), torch.float32)]),
    (Conv2D810, [((1, 2560, 20, 20), torch.float32)]),
    (Conv2D811, [((1, 1280, 40, 40), torch.float32)]),
    (Conv2D812, [((1, 640, 80, 80), torch.float32)]),
    (Conv2D813, [((1, 320, 80, 80), torch.float32)]),
    (Conv2D803, [((1, 320, 80, 80), torch.float32)]),
    (Conv2D814, [((1, 640, 40, 40), torch.float32)]),
    (Conv2D808, [((1, 640, 40, 40), torch.float32)]),
    (Conv2D815, [((1, 1280, 20, 20), torch.float32)]),
    (Conv2D816, [((1, 3, 480, 480), torch.float32)]),
    (Conv2D817, [((1, 48, 240, 240), torch.float32)]),
    (Conv2D818, [((1, 96, 120, 120), torch.float32)]),
    (Conv2D819, [((1, 48, 120, 120), torch.float32)]),
    (Conv2D820, [((1, 48, 120, 120), torch.float32)]),
    (Conv2D821, [((1, 96, 120, 120), torch.float32)]),
    (Conv2D822, [((1, 96, 120, 120), torch.float32)]),
    (Conv2D823, [((1, 192, 60, 60), torch.float32)]),
    (Conv2D821, [((1, 96, 60, 60), torch.float32)]),
    (Conv2D824, [((1, 96, 60, 60), torch.float32)]),
    (Conv2D825, [((1, 192, 60, 60), torch.float32)]),
    (Conv2D826, [((1, 192, 60, 60), torch.float32)]),
    (Conv2D827, [((1, 384, 30, 30), torch.float32)]),
    (Conv2D825, [((1, 192, 30, 30), torch.float32)]),
    (Conv2D828, [((1, 192, 30, 30), torch.float32)]),
    (Conv2D829, [((1, 384, 30, 30), torch.float32)]),
    (Conv2D830, [((1, 384, 30, 30), torch.float32)]),
    (Conv2D831, [((1, 768, 15, 15), torch.float32)]),
    (Conv2D829, [((1, 384, 15, 15), torch.float32)]),
    (Conv2D832, [((1, 384, 15, 15), torch.float32)]),
    (Conv2D833, [((1, 768, 15, 15), torch.float32)]),
    (Conv2D834, [((1, 1536, 15, 15), torch.float32)]),
    (Conv2D835, [((1, 768, 30, 30), torch.float32)]),
    (Conv2D836, [((1, 384, 60, 60), torch.float32)]),
    (Conv2D837, [((1, 192, 60, 60), torch.float32)]),
    (Conv2D838, [((1, 192, 60, 60), torch.float32)]),
    (Conv2D839, [((1, 384, 30, 30), torch.float32)]),
    (Conv2D840, [((1, 384, 30, 30), torch.float32)]),
    (Conv2D841, [((1, 768, 15, 15), torch.float32)]),
    (Conv2D790, [((1, 3, 320, 320), torch.float32)]),
    (Conv2D791, [((1, 80, 160, 160), torch.float32)]),
    (Conv2D792, [((1, 160, 80, 80), torch.float32)]),
    (Conv2D793, [((1, 80, 80, 80), torch.float32)]),
    (Conv2D794, [((1, 80, 80, 80), torch.float32)]),
    (Conv2D796, [((1, 160, 80, 80), torch.float32)]),
    (Conv2D797, [((1, 320, 40, 40), torch.float32)]),
    (Conv2D795, [((1, 160, 40, 40), torch.float32)]),
    (Conv2D798, [((1, 160, 40, 40), torch.float32)]),
    (Conv2D800, [((1, 320, 40, 40), torch.float32)]),
    (Conv2D801, [((1, 640, 20, 20), torch.float32)]),
    (Conv2D799, [((1, 320, 20, 20), torch.float32)]),
    (Conv2D802, [((1, 320, 20, 20), torch.float32)]),
    (Conv2D805, [((1, 640, 20, 20), torch.float32)]),
    (Conv2D806, [((1, 1280, 10, 10), torch.float32)]),
    (Conv2D804, [((1, 640, 10, 10), torch.float32)]),
    (Conv2D807, [((1, 640, 10, 10), torch.float32)]),
    (Conv2D809, [((1, 1280, 10, 10), torch.float32)]),
    (Conv2D810, [((1, 2560, 10, 10), torch.float32)]),
    (Conv2D811, [((1, 1280, 20, 20), torch.float32)]),
    (Conv2D812, [((1, 640, 40, 40), torch.float32)]),
    (Conv2D813, [((1, 320, 40, 40), torch.float32)]),
    (Conv2D814, [((1, 640, 20, 20), torch.float32)]),
    (Conv2D815, [((1, 1280, 10, 10), torch.float32)]),
    (Conv2D842, [((1, 3, 640, 640), torch.float32)]),
    (Conv2D843, [((1, 32, 320, 320), torch.float32)]),
    (Conv2D844, [((1, 64, 160, 160), torch.float32)]),
    (Conv2D845, [((1, 32, 160, 160), torch.float32)]),
    (Conv2D846, [((1, 32, 160, 160), torch.float32)]),
    (Conv2D847, [((1, 64, 160, 160), torch.float32)]),
    (Conv2D848, [((1, 64, 160, 160), torch.float32)]),
    (Conv2D849, [((1, 128, 80, 80), torch.float32)]),
    (Conv2D847, [((1, 64, 80, 80), torch.float32)]),
    (Conv2D850, [((1, 64, 80, 80), torch.float32)]),
    (Conv2D851, [((1, 64, 80, 80), torch.float32)]),
    (Conv2D852, [((1, 128, 80, 80), torch.float32)]),
    (Conv2D853, [((1, 128, 80, 80), torch.float32)]),
    (Conv2D854, [((1, 256, 40, 40), torch.float32)]),
    (Conv2D852, [((1, 128, 40, 40), torch.float32)]),
    (Conv2D855, [((1, 128, 40, 40), torch.float32)]),
    (Conv2D856, [((1, 128, 40, 40), torch.float32)]),
    (Conv2D857, [((1, 256, 40, 40), torch.float32)]),
    (Conv2D858, [((1, 256, 40, 40), torch.float32)]),
    (Conv2D859, [((1, 512, 20, 20), torch.float32)]),
    (Conv2D857, [((1, 256, 20, 20), torch.float32)]),
    (Conv2D669, [((1, 256, 20, 20), torch.float32)]),
    (Conv2D860, [((1, 256, 20, 20), torch.float32)]),
    (Conv2D861, [((1, 512, 20, 20), torch.float32)]),
    (Conv2D862, [((1, 1024, 20, 20), torch.float32)]),
    (Conv2D863, [((1, 512, 40, 40), torch.float32)]),
    (Conv2D864, [((1, 256, 80, 80), torch.float32)]),
    (Conv2D865, [((1, 128, 80, 80), torch.float32)]),
    (Conv2D856, [((1, 128, 80, 80), torch.float32)]),
    (Conv2D855, [((1, 128, 80, 80), torch.float32)]),
    (Conv2D866, [((1, 256, 40, 40), torch.float32)]),
    (Conv2D860, [((1, 256, 40, 40), torch.float32)]),
    (Conv2D669, [((1, 256, 40, 40), torch.float32)]),
    (Conv2D867, [((1, 512, 20, 20), torch.float32)]),
    (Conv2D868, [((1, 3, 320, 320), torch.float32)]),
    (Conv2D869, [((1, 512, 20, 20), torch.float32)]),
    (Conv2D862, [((1, 1024, 10, 10), torch.float32)]),
    (Conv2D861, [((1, 512, 10, 10), torch.float32)]),
    (Conv2D870, [((1, 512, 10, 10), torch.float32)]),
    (Conv2D871, [((1, 1024, 10, 10), torch.float32)]),
    (Conv2D872, [((1, 2048, 10, 10), torch.float32)]),
    (Conv2D873, [((1, 1024, 20, 20), torch.float32)]),
    (Conv2D874, [((1, 512, 20, 20), torch.float32)]),
    (Conv2D870, [((1, 512, 20, 20), torch.float32)]),
    (Conv2D875, [((1, 1024, 10, 10), torch.float32)]),
    (Conv2D868, [((1, 3, 640, 640), torch.float32)]),
    (Conv2D848, [((1, 64, 320, 320), torch.float32)]),
    (Conv2D849, [((1, 128, 160, 160), torch.float32)]),
    (Conv2D850, [((1, 64, 160, 160), torch.float32)]),
    (Conv2D852, [((1, 128, 160, 160), torch.float32)]),
    (Conv2D853, [((1, 128, 160, 160), torch.float32)]),
    (Conv2D854, [((1, 256, 80, 80), torch.float32)]),
    (Conv2D857, [((1, 256, 80, 80), torch.float32)]),
    (Conv2D858, [((1, 256, 80, 80), torch.float32)]),
    (Conv2D859, [((1, 512, 40, 40), torch.float32)]),
    (Conv2D861, [((1, 512, 40, 40), torch.float32)]),
    (Conv2D869, [((1, 512, 40, 40), torch.float32)]),
    (Conv2D871, [((1, 1024, 20, 20), torch.float32)]),
    (Conv2D872, [((1, 2048, 20, 20), torch.float32)]),
    (Conv2D873, [((1, 1024, 40, 40), torch.float32)]),
    (Conv2D863, [((1, 512, 80, 80), torch.float32)]),
    (Conv2D866, [((1, 256, 80, 80), torch.float32)]),
    (Conv2D860, [((1, 256, 80, 80), torch.float32)]),
    (Conv2D867, [((1, 512, 40, 40), torch.float32)]),
    (Conv2D874, [((1, 512, 40, 40), torch.float32)]),
    (Conv2D875, [((1, 1024, 20, 20), torch.float32)]),
    (Conv2D842, [((1, 3, 1280, 1280), torch.float32)]),
    (Conv2D843, [((1, 32, 640, 640), torch.float32)]),
    (Conv2D844, [((1, 64, 320, 320), torch.float32)]),
    (Conv2D845, [((1, 32, 320, 320), torch.float32)]),
    (Conv2D846, [((1, 32, 320, 320), torch.float32)]),
    (Conv2D847, [((1, 64, 320, 320), torch.float32)]),
    (Conv2D862, [((1, 1024, 40, 40), torch.float32)]),
    (Conv2D864, [((1, 256, 160, 160), torch.float32)]),
    (Conv2D865, [((1, 128, 160, 160), torch.float32)]),
    (Conv2D856, [((1, 128, 160, 160), torch.float32)]),
    (Conv2D876, [((1, 3, 640, 640), torch.float32)]),
    (Conv2D877, [((1, 16, 320, 320), torch.float32)]),
    (Conv2D878, [((1, 32, 160, 160), torch.float32)]),
    (Conv2D879, [((1, 16, 160, 160), torch.float32)]),
    (Conv2D880, [((1, 16, 160, 160), torch.float32)]),
    (Conv2D843, [((1, 32, 160, 160), torch.float32)]),
    (Conv2D844, [((1, 64, 80, 80), torch.float32)]),
    (Conv2D845, [((1, 32, 80, 80), torch.float32)]),
    (Conv2D846, [((1, 32, 80, 80), torch.float32)]),
    (Conv2D848, [((1, 64, 80, 80), torch.float32)]),
    (Conv2D849, [((1, 128, 40, 40), torch.float32)]),
    (Conv2D847, [((1, 64, 40, 40), torch.float32)]),
    (Conv2D850, [((1, 64, 40, 40), torch.float32)]),
    (Conv2D851, [((1, 64, 40, 40), torch.float32)]),
    (Conv2D853, [((1, 128, 40, 40), torch.float32)]),
    (Conv2D854, [((1, 256, 20, 20), torch.float32)]),
    (Conv2D852, [((1, 128, 20, 20), torch.float32)]),
    (Conv2D855, [((1, 128, 20, 20), torch.float32)]),
    (Conv2D856, [((1, 128, 20, 20), torch.float32)]),
    (Conv2D864, [((1, 256, 40, 40), torch.float32)]),
    (Conv2D881, [((1, 128, 80, 80), torch.float32)]),
    (Conv2D882, [((1, 64, 80, 80), torch.float32)]),
    (Conv2D865, [((1, 128, 40, 40), torch.float32)]),
    (Conv2D866, [((1, 256, 20, 20), torch.float32)]),
    (Conv2D842, [((1, 3, 480, 480), torch.float32)]),
    (Conv2D843, [((1, 32, 240, 240), torch.float32)]),
    (Conv2D844, [((1, 64, 120, 120), torch.float32)]),
    (Conv2D845, [((1, 32, 120, 120), torch.float32)]),
    (Conv2D846, [((1, 32, 120, 120), torch.float32)]),
    (Conv2D847, [((1, 64, 120, 120), torch.float32)]),
    (Conv2D848, [((1, 64, 120, 120), torch.float32)]),
    (Conv2D849, [((1, 128, 60, 60), torch.float32)]),
    (Conv2D847, [((1, 64, 60, 60), torch.float32)]),
    (Conv2D850, [((1, 64, 60, 60), torch.float32)]),
    (Conv2D851, [((1, 64, 60, 60), torch.float32)]),
    (Conv2D852, [((1, 128, 60, 60), torch.float32)]),
    (Conv2D853, [((1, 128, 60, 60), torch.float32)]),
    (Conv2D854, [((1, 256, 30, 30), torch.float32)]),
    (Conv2D852, [((1, 128, 30, 30), torch.float32)]),
    (Conv2D855, [((1, 128, 30, 30), torch.float32)]),
    (Conv2D856, [((1, 128, 30, 30), torch.float32)]),
    (Conv2D857, [((1, 256, 30, 30), torch.float32)]),
    (Conv2D858, [((1, 256, 30, 30), torch.float32)]),
    (Conv2D859, [((1, 512, 15, 15), torch.float32)]),
    (Conv2D857, [((1, 256, 15, 15), torch.float32)]),
    (Conv2D669, [((1, 256, 15, 15), torch.float32)]),
    (Conv2D861, [((1, 512, 15, 15), torch.float32)]),
    (Conv2D862, [((1, 1024, 15, 15), torch.float32)]),
    (Conv2D863, [((1, 512, 30, 30), torch.float32)]),
    (Conv2D864, [((1, 256, 60, 60), torch.float32)]),
    (Conv2D865, [((1, 128, 60, 60), torch.float32)]),
    (Conv2D856, [((1, 128, 60, 60), torch.float32)]),
    (Conv2D855, [((1, 128, 60, 60), torch.float32)]),
    (Conv2D866, [((1, 256, 30, 30), torch.float32)]),
    (Conv2D860, [((1, 256, 30, 30), torch.float32)]),
    (Conv2D669, [((1, 256, 30, 30), torch.float32)]),
    (Conv2D867, [((1, 512, 15, 15), torch.float32)]),
    (Conv2D790, [((1, 3, 480, 480), torch.float32)]),
    (Conv2D791, [((1, 80, 240, 240), torch.float32)]),
    (Conv2D792, [((1, 160, 120, 120), torch.float32)]),
    (Conv2D793, [((1, 80, 120, 120), torch.float32)]),
    (Conv2D794, [((1, 80, 120, 120), torch.float32)]),
    (Conv2D795, [((1, 160, 120, 120), torch.float32)]),
    (Conv2D796, [((1, 160, 120, 120), torch.float32)]),
    (Conv2D797, [((1, 320, 60, 60), torch.float32)]),
    (Conv2D795, [((1, 160, 60, 60), torch.float32)]),
    (Conv2D798, [((1, 160, 60, 60), torch.float32)]),
    (Conv2D799, [((1, 320, 60, 60), torch.float32)]),
    (Conv2D800, [((1, 320, 60, 60), torch.float32)]),
    (Conv2D801, [((1, 640, 30, 30), torch.float32)]),
    (Conv2D799, [((1, 320, 30, 30), torch.float32)]),
    (Conv2D802, [((1, 320, 30, 30), torch.float32)]),
    (Conv2D804, [((1, 640, 30, 30), torch.float32)]),
    (Conv2D805, [((1, 640, 30, 30), torch.float32)]),
    (Conv2D806, [((1, 1280, 15, 15), torch.float32)]),
    (Conv2D804, [((1, 640, 15, 15), torch.float32)]),
    (Conv2D807, [((1, 640, 15, 15), torch.float32)]),
    (Conv2D809, [((1, 1280, 15, 15), torch.float32)]),
    (Conv2D810, [((1, 2560, 15, 15), torch.float32)]),
    (Conv2D811, [((1, 1280, 30, 30), torch.float32)]),
    (Conv2D812, [((1, 640, 60, 60), torch.float32)]),
    (Conv2D813, [((1, 320, 60, 60), torch.float32)]),
    (Conv2D803, [((1, 320, 60, 60), torch.float32)]),
    (Conv2D814, [((1, 640, 30, 30), torch.float32)]),
    (Conv2D808, [((1, 640, 30, 30), torch.float32)]),
    (Conv2D815, [((1, 1280, 15, 15), torch.float32)]),
    (Conv2D876, [((1, 3, 480, 480), torch.float32)]),
    (Conv2D877, [((1, 16, 240, 240), torch.float32)]),
    (Conv2D878, [((1, 32, 120, 120), torch.float32)]),
    (Conv2D879, [((1, 16, 120, 120), torch.float32)]),
    (Conv2D880, [((1, 16, 120, 120), torch.float32)]),
    (Conv2D843, [((1, 32, 120, 120), torch.float32)]),
    (Conv2D844, [((1, 64, 60, 60), torch.float32)]),
    (Conv2D845, [((1, 32, 60, 60), torch.float32)]),
    (Conv2D846, [((1, 32, 60, 60), torch.float32)]),
    (Conv2D848, [((1, 64, 60, 60), torch.float32)]),
    (Conv2D849, [((1, 128, 30, 30), torch.float32)]),
    (Conv2D847, [((1, 64, 30, 30), torch.float32)]),
    (Conv2D850, [((1, 64, 30, 30), torch.float32)]),
    (Conv2D853, [((1, 128, 30, 30), torch.float32)]),
    (Conv2D854, [((1, 256, 15, 15), torch.float32)]),
    (Conv2D852, [((1, 128, 15, 15), torch.float32)]),
    (Conv2D855, [((1, 128, 15, 15), torch.float32)]),
    (Conv2D864, [((1, 256, 30, 30), torch.float32)]),
    (Conv2D881, [((1, 128, 60, 60), torch.float32)]),
    (Conv2D882, [((1, 64, 60, 60), torch.float32)]),
    (Conv2D865, [((1, 128, 30, 30), torch.float32)]),
    (Conv2D866, [((1, 256, 15, 15), torch.float32)]),
    (Conv2D868, [((1, 3, 480, 480), torch.float32)]),
    (Conv2D848, [((1, 64, 240, 240), torch.float32)]),
    (Conv2D849, [((1, 128, 120, 120), torch.float32)]),
    (Conv2D850, [((1, 64, 120, 120), torch.float32)]),
    (Conv2D852, [((1, 128, 120, 120), torch.float32)]),
    (Conv2D853, [((1, 128, 120, 120), torch.float32)]),
    (Conv2D854, [((1, 256, 60, 60), torch.float32)]),
    (Conv2D857, [((1, 256, 60, 60), torch.float32)]),
    (Conv2D858, [((1, 256, 60, 60), torch.float32)]),
    (Conv2D859, [((1, 512, 30, 30), torch.float32)]),
    (Conv2D861, [((1, 512, 30, 30), torch.float32)]),
    (Conv2D869, [((1, 512, 30, 30), torch.float32)]),
    (Conv2D870, [((1, 512, 15, 15), torch.float32)]),
    (Conv2D871, [((1, 1024, 15, 15), torch.float32)]),
    (Conv2D872, [((1, 2048, 15, 15), torch.float32)]),
    (Conv2D873, [((1, 1024, 30, 30), torch.float32)]),
    (Conv2D863, [((1, 512, 60, 60), torch.float32)]),
    (Conv2D866, [((1, 256, 60, 60), torch.float32)]),
    (Conv2D860, [((1, 256, 60, 60), torch.float32)]),
    (Conv2D867, [((1, 512, 30, 30), torch.float32)]),
    (Conv2D874, [((1, 512, 30, 30), torch.float32)]),
    (Conv2D875, [((1, 1024, 15, 15), torch.float32)]),
    (Conv2D816, [((1, 3, 640, 640), torch.float32)]),
    (Conv2D817, [((1, 48, 320, 320), torch.float32)]),
    (Conv2D818, [((1, 96, 160, 160), torch.float32)]),
    (Conv2D819, [((1, 48, 160, 160), torch.float32)]),
    (Conv2D820, [((1, 48, 160, 160), torch.float32)]),
    (Conv2D821, [((1, 96, 160, 160), torch.float32)]),
    (Conv2D822, [((1, 96, 160, 160), torch.float32)]),
    (Conv2D823, [((1, 192, 80, 80), torch.float32)]),
    (Conv2D821, [((1, 96, 80, 80), torch.float32)]),
    (Conv2D824, [((1, 96, 80, 80), torch.float32)]),
    (Conv2D825, [((1, 192, 80, 80), torch.float32)]),
    (Conv2D826, [((1, 192, 80, 80), torch.float32)]),
    (Conv2D827, [((1, 384, 40, 40), torch.float32)]),
    (Conv2D825, [((1, 192, 40, 40), torch.float32)]),
    (Conv2D828, [((1, 192, 40, 40), torch.float32)]),
    (Conv2D838, [((1, 192, 40, 40), torch.float32)]),
    (Conv2D829, [((1, 384, 40, 40), torch.float32)]),
    (Conv2D830, [((1, 384, 40, 40), torch.float32)]),
    (Conv2D831, [((1, 768, 20, 20), torch.float32)]),
    (Conv2D829, [((1, 384, 20, 20), torch.float32)]),
    (Conv2D832, [((1, 384, 20, 20), torch.float32)]),
    (Conv2D840, [((1, 384, 20, 20), torch.float32)]),
    (Conv2D833, [((1, 768, 20, 20), torch.float32)]),
    (Conv2D834, [((1, 1536, 20, 20), torch.float32)]),
    (Conv2D835, [((1, 768, 40, 40), torch.float32)]),
    (Conv2D836, [((1, 384, 80, 80), torch.float32)]),
    (Conv2D837, [((1, 192, 80, 80), torch.float32)]),
    (Conv2D838, [((1, 192, 80, 80), torch.float32)]),
    (Conv2D839, [((1, 384, 40, 40), torch.float32)]),
    (Conv2D840, [((1, 384, 40, 40), torch.float32)]),
    (Conv2D841, [((1, 768, 20, 20), torch.float32)]),
    (Conv2D816, [((1, 3, 320, 320), torch.float32)]),
    (Conv2D817, [((1, 48, 160, 160), torch.float32)]),
    (Conv2D818, [((1, 96, 80, 80), torch.float32)]),
    (Conv2D819, [((1, 48, 80, 80), torch.float32)]),
    (Conv2D820, [((1, 48, 80, 80), torch.float32)]),
    (Conv2D822, [((1, 96, 80, 80), torch.float32)]),
    (Conv2D823, [((1, 192, 40, 40), torch.float32)]),
    (Conv2D821, [((1, 96, 40, 40), torch.float32)]),
    (Conv2D824, [((1, 96, 40, 40), torch.float32)]),
    (Conv2D826, [((1, 192, 40, 40), torch.float32)]),
    (Conv2D827, [((1, 384, 20, 20), torch.float32)]),
    (Conv2D825, [((1, 192, 20, 20), torch.float32)]),
    (Conv2D828, [((1, 192, 20, 20), torch.float32)]),
    (Conv2D830, [((1, 384, 20, 20), torch.float32)]),
    (Conv2D831, [((1, 768, 10, 10), torch.float32)]),
    (Conv2D829, [((1, 384, 10, 10), torch.float32)]),
    (Conv2D832, [((1, 384, 10, 10), torch.float32)]),
    (Conv2D833, [((1, 768, 10, 10), torch.float32)]),
    (Conv2D834, [((1, 1536, 10, 10), torch.float32)]),
    (Conv2D835, [((1, 768, 20, 20), torch.float32)]),
    (Conv2D836, [((1, 384, 40, 40), torch.float32)]),
    (Conv2D837, [((1, 192, 40, 40), torch.float32)]),
    (Conv2D839, [((1, 384, 20, 20), torch.float32)]),
    (Conv2D841, [((1, 768, 10, 10), torch.float32)]),
    (Conv2D842, [((1, 3, 320, 320), torch.float32)]),
    (Conv2D858, [((1, 256, 20, 20), torch.float32)]),
    (Conv2D859, [((1, 512, 10, 10), torch.float32)]),
    (Conv2D857, [((1, 256, 10, 10), torch.float32)]),
    (Conv2D669, [((1, 256, 10, 10), torch.float32)]),
    (Conv2D863, [((1, 512, 20, 20), torch.float32)]),
    (Conv2D867, [((1, 512, 10, 10), torch.float32)]),
    (Conv2D876, [((1, 3, 320, 320), torch.float32)]),
    (Conv2D877, [((1, 16, 160, 160), torch.float32)]),
    (Conv2D878, [((1, 32, 80, 80), torch.float32)]),
    (Conv2D879, [((1, 16, 80, 80), torch.float32)]),
    (Conv2D880, [((1, 16, 80, 80), torch.float32)]),
    (Conv2D843, [((1, 32, 80, 80), torch.float32)]),
    (Conv2D844, [((1, 64, 40, 40), torch.float32)]),
    (Conv2D845, [((1, 32, 40, 40), torch.float32)]),
    (Conv2D846, [((1, 32, 40, 40), torch.float32)]),
    (Conv2D848, [((1, 64, 40, 40), torch.float32)]),
    (Conv2D849, [((1, 128, 20, 20), torch.float32)]),
    (Conv2D847, [((1, 64, 20, 20), torch.float32)]),
    (Conv2D850, [((1, 64, 20, 20), torch.float32)]),
    (Conv2D853, [((1, 128, 20, 20), torch.float32)]),
    (Conv2D854, [((1, 256, 10, 10), torch.float32)]),
    (Conv2D852, [((1, 128, 10, 10), torch.float32)]),
    (Conv2D855, [((1, 128, 10, 10), torch.float32)]),
    (Conv2D864, [((1, 256, 20, 20), torch.float32)]),
    (Conv2D881, [((1, 128, 40, 40), torch.float32)]),
    (Conv2D882, [((1, 64, 40, 40), torch.float32)]),
    (Conv2D865, [((1, 128, 20, 20), torch.float32)]),
    (Conv2D866, [((1, 256, 10, 10), torch.float32)]),
    (Conv2D883, [((1, 3, 448, 640), torch.float32)]),
    (Conv2D884, [((1, 16, 224, 320), torch.float32)]),
    (Conv2D885, [((1, 32, 112, 160), torch.float32)]),
    (Conv2D886, [((1, 32, 112, 160), torch.float32)]),
    (Conv2D887, [((1, 32, 112, 160), torch.float32)]),
    (Conv2D888, [((1, 64, 56, 80), torch.float32)]),
    (Conv2D889, [((1, 64, 56, 80), torch.float32)]),
    (Conv2D890, [((1, 64, 56, 80), torch.float32)]),
    (Conv2D891, [((1, 128, 28, 40), torch.float32)]),
    (Conv2D892, [((1, 128, 28, 40), torch.float32)]),
    (Conv2D893, [((1, 128, 28, 40), torch.float32)]),
    (Conv2D894, [((1, 256, 14, 20), torch.float32)]),
    (Conv2D895, [((1, 256, 14, 20), torch.float32)]),
    (Conv2D891, [((1, 128, 14, 20), torch.float32)]),
    (Conv2D896, [((1, 128, 14, 20), torch.float32)]),
    (Conv2D897, [((1, 512, 14, 20), torch.float32)]),
    (Conv2D898, [((1, 256, 14, 20), torch.float32)]),
    (Conv2D899, [((1, 256, 14, 20), torch.float32)]),
    (Conv2D900, [((1, 128, 28, 40), torch.float32)]),
    (Conv2D901, [((1, 64, 56, 80), torch.float32)]),
    (Conv2D902, [((1, 192, 28, 40), torch.float32)]),
    (Conv2D888, [((1, 64, 28, 40), torch.float32)]),
    (Conv2D889, [((1, 64, 28, 40), torch.float32)]),
    (Conv2D903, [((1, 64, 28, 40), torch.float32)]),
    (Conv2D903, [((1, 64, 56, 80), torch.float32)]),
    (Conv2D904, [((1, 32, 112, 160), torch.float32)]),
    (Conv2D905, [((1, 96, 56, 80), torch.float32)]),
    (Conv2D885, [((1, 32, 56, 80), torch.float32)]),
    (Conv2D886, [((1, 32, 56, 80), torch.float32)]),
    (Conv2D904, [((1, 32, 56, 80), torch.float32)]),
    (Conv2D906, [((1, 32, 56, 80), torch.float32)]),
    (Conv2D901, [((1, 64, 28, 40), torch.float32)]),
    (Conv2D907, [((1, 64, 28, 40), torch.float32)]),
    (Conv2D908, [((1, 128, 14, 20), torch.float32)]),
    (Conv2D909, [((1, 32, 56, 80), torch.float32)]),
    (Conv2D910, [((1, 64, 28, 40), torch.float32)]),
    (Conv2D911, [((1, 128, 14, 20), torch.float32)]),
    (Conv2D912, [((1, 3, 448, 640), torch.float32)]),
    (Conv2D913, [((1, 48, 224, 320), torch.float32)]),
    (Conv2D914, [((1, 96, 112, 160), torch.float32)]),
    (Conv2D888, [((1, 64, 112, 160), torch.float32)]),
    (Conv2D889, [((1, 64, 112, 160), torch.float32)]),
    (Conv2D915, [((1, 128, 112, 160), torch.float32)]),
    (Conv2D916, [((1, 96, 112, 160), torch.float32)]),
    (Conv2D917, [((1, 192, 56, 80), torch.float32)]),
    (Conv2D891, [((1, 128, 56, 80), torch.float32)]),
    (Conv2D892, [((1, 128, 56, 80), torch.float32)]),
    (Conv2D918, [((1, 256, 56, 80), torch.float32)]),
    (Conv2D919, [((1, 192, 56, 80), torch.float32)]),
    (Conv2D920, [((1, 384, 28, 40), torch.float32)]),
    (Conv2D894, [((1, 256, 28, 40), torch.float32)]),
    (Conv2D921, [((1, 256, 28, 40), torch.float32)]),
    (Conv2D922, [((1, 512, 28, 40), torch.float32)]),
    (Conv2D923, [((1, 384, 28, 40), torch.float32)]),
    (Conv2D924, [((1, 768, 14, 20), torch.float32)]),
    (Conv2D925, [((1, 512, 14, 20), torch.float32)]),
    (Conv2D926, [((1, 1024, 14, 20), torch.float32)]),
    (Conv2D927, [((1, 768, 14, 20), torch.float32)]),
    (Conv2D928, [((1, 1536, 14, 20), torch.float32)]),
    (Conv2D929, [((1, 768, 14, 20), torch.float32)]),
    (Conv2D930, [((1, 384, 28, 40), torch.float32)]),
    (Conv2D931, [((1, 192, 56, 80), torch.float32)]),
    (Conv2D932, [((1, 192, 56, 80), torch.float32)]),
    (Conv2D933, [((1, 576, 28, 40), torch.float32)]),
    (Conv2D917, [((1, 192, 28, 40), torch.float32)]),
    (Conv2D918, [((1, 256, 28, 40), torch.float32)]),
    (Conv2D934, [((1, 192, 28, 40), torch.float32)]),
    (Conv2D934, [((1, 192, 56, 80), torch.float32)]),
    (Conv2D935, [((1, 96, 112, 160), torch.float32)]),
    (Conv2D936, [((1, 96, 112, 160), torch.float32)]),
    (Conv2D937, [((1, 288, 56, 80), torch.float32)]),
    (Conv2D914, [((1, 96, 56, 80), torch.float32)]),
    (Conv2D915, [((1, 128, 56, 80), torch.float32)]),
    (Conv2D935, [((1, 96, 56, 80), torch.float32)]),
    (Conv2D938, [((1, 96, 56, 80), torch.float32)]),
    (Conv2D936, [((1, 96, 56, 80), torch.float32)]),
    (Conv2D939, [((1, 96, 56, 80), torch.float32)]),
    (Conv2D940, [((1, 17, 4, 4480), torch.float32)]),
    (Conv2D931, [((1, 192, 28, 40), torch.float32)]),
    (Conv2D941, [((1, 192, 28, 40), torch.float32)]),
    (Conv2D932, [((1, 192, 28, 40), torch.float32)]),
    (Conv2D942, [((1, 192, 28, 40), torch.float32)]),
    (Conv2D940, [((1, 17, 4, 1120), torch.float32)]),
    (Conv2D920, [((1, 384, 14, 20), torch.float32)]),
    (Conv2D922, [((1, 512, 14, 20), torch.float32)]),
    (Conv2D943, [((1, 384, 14, 20), torch.float32)]),
    (Conv2D944, [((1, 384, 14, 20), torch.float32)]),
    (Conv2D945, [((1, 384, 14, 20), torch.float32)]),
    (Conv2D940, [((1, 17, 4, 280), torch.float32)]),
    (Conv2D946, [((1, 96, 56, 80), torch.float32)]),
    (Conv2D947, [((1, 192, 28, 40), torch.float32)]),
    (Conv2D948, [((1, 384, 14, 20), torch.float32)]),
    (Conv2D949, [((1, 3, 448, 640), torch.float32)]),
    (Conv2D887, [((1, 32, 224, 320), torch.float32)]),
    (Conv2D890, [((1, 64, 112, 160), torch.float32)]),
    (Conv2D893, [((1, 128, 56, 80), torch.float32)]),
    (Conv2D950, [((1, 256, 28, 40), torch.float32)]),
    (Conv2D951, [((1, 512, 14, 20), torch.float32)]),
    (Conv2D952, [((1, 1024, 14, 20), torch.float32)]),
    (Conv2D953, [((1, 512, 14, 20), torch.float32)]),
    (Conv2D895, [((1, 256, 28, 40), torch.float32)]),
    (Conv2D896, [((1, 128, 56, 80), torch.float32)]),
    (Conv2D954, [((1, 384, 28, 40), torch.float32)]),
    (Conv2D900, [((1, 128, 56, 80), torch.float32)]),
    (Conv2D901, [((1, 64, 112, 160), torch.float32)]),
    (Conv2D902, [((1, 192, 56, 80), torch.float32)]),
    (Conv2D907, [((1, 64, 56, 80), torch.float32)]),
    (Conv2D896, [((1, 128, 28, 40), torch.float32)]),
    (Conv2D908, [((1, 128, 28, 40), torch.float32)]),
    (Conv2D955, [((1, 256, 14, 20), torch.float32)]),
    (Conv2D910, [((1, 64, 56, 80), torch.float32)]),
    (Conv2D911, [((1, 128, 28, 40), torch.float32)]),
    (Conv2D956, [((1, 256, 14, 20), torch.float32)]),
    (Conv2D957, [((1, 3, 448, 640), torch.float32)]),
    (Conv2D890, [((1, 64, 224, 320), torch.float32)]),
    (Conv2D900, [((1, 128, 112, 160), torch.float32)]),
    (Conv2D896, [((1, 128, 112, 160), torch.float32)]),
    (Conv2D893, [((1, 128, 112, 160), torch.float32)]),
    (Conv2D895, [((1, 256, 56, 80), torch.float32)]),
    (Conv2D898, [((1, 256, 56, 80), torch.float32)]),
    (Conv2D950, [((1, 256, 56, 80), torch.float32)]),
    (Conv2D951, [((1, 512, 28, 40), torch.float32)]),
    (Conv2D953, [((1, 512, 28, 40), torch.float32)]),
    (Conv2D958, [((1, 512, 28, 40), torch.float32)]),
    (Conv2D959, [((1, 1024, 14, 20), torch.float32)]),
    (Conv2D960, [((1, 1024, 14, 20), torch.float32)]),
    (Conv2D961, [((1, 2048, 14, 20), torch.float32)]),
    (Conv2D921, [((1, 256, 56, 80), torch.float32)]),
    (Conv2D962, [((1, 768, 28, 40), torch.float32)]),
    (Conv2D898, [((1, 256, 28, 40), torch.float32)]),
    (Conv2D892, [((1, 128, 112, 160), torch.float32)]),
    (Conv2D954, [((1, 384, 56, 80), torch.float32)]),
    (Conv2D963, [((1, 128, 56, 80), torch.float32)]),
    (Conv2D964, [((1, 256, 28, 40), torch.float32)]),
    (Conv2D965, [((1, 512, 14, 20), torch.float32)]),
    (Conv2D911, [((1, 128, 56, 80), torch.float32)]),
    (Conv2D956, [((1, 256, 28, 40), torch.float32)]),
    (Conv2D966, [((1, 512, 14, 20), torch.float32)]),
    (Conv2D967, [((1, 12, 320, 320), torch.float32)]),
    (Conv2D467, [((1, 48, 320, 320), torch.float32)]),
    (Conv2D466, [((1, 96, 160, 160), torch.float32)]),
    (Conv2D968, [((1, 48, 160, 160), torch.float32)]),
    (Conv2D461, [((1, 48, 160, 160), torch.float32)]),
    (Conv2D549, [((1, 96, 160, 160), torch.float32)]),
    (Conv2D468, [((1, 96, 160, 160), torch.float32)]),
    (Conv2D471, [((1, 192, 80, 80), torch.float32)]),
    (Conv2D549, [((1, 96, 80, 80), torch.float32)]),
    (Conv2D464, [((1, 96, 80, 80), torch.float32)]),
    (Conv2D80, [((1, 192, 80, 80), torch.float32)]),
    (Conv2D473, [((1, 192, 80, 80), torch.float32)]),
    (Conv2D84, [((1, 384, 40, 40), torch.float32)]),
    (Conv2D80, [((1, 192, 40, 40), torch.float32)]),
    (Conv2D469, [((1, 192, 40, 40), torch.float32)]),
    (Conv2D554, [((1, 384, 40, 40), torch.float32)]),
    (Conv2D969, [((1, 384, 40, 40), torch.float32)]),
    (Conv2D92, [((1, 768, 20, 20), torch.float32)]),
    (Conv2D970, [((1, 1536, 20, 20), torch.float32)]),
    (Conv2D554, [((1, 384, 20, 20), torch.float32)]),
    (Conv2D474, [((1, 384, 20, 20), torch.float32)]),
    (Conv2D557, [((1, 768, 20, 20), torch.float32)]),
    (Conv2D93, [((1, 768, 40, 40), torch.float32)]),
    (Conv2D518, [((1, 384, 80, 80), torch.float32)]),
    (Conv2D469, [((1, 192, 80, 80), torch.float32)]),
    (Conv2D971, [((1, 192, 80, 80), torch.float32)]),
    (Conv2D972, [((1, 192, 80, 80), torch.float32)]),
    (Conv2D973, [((1, 192, 80, 80), torch.float32)]),
    (Conv2D974, [((1, 192, 80, 80), torch.float32)]),
    (Conv2D972, [((1, 192, 40, 40), torch.float32)]),
    (Conv2D973, [((1, 192, 40, 40), torch.float32)]),
    (Conv2D974, [((1, 192, 40, 40), torch.float32)]),
    (Conv2D975, [((1, 384, 40, 40), torch.float32)]),
    (Conv2D93, [((1, 768, 20, 20), torch.float32)]),
    (Conv2D469, [((1, 192, 20, 20), torch.float32)]),
    (Conv2D972, [((1, 192, 20, 20), torch.float32)]),
    (Conv2D973, [((1, 192, 20, 20), torch.float32)]),
    (Conv2D974, [((1, 192, 20, 20), torch.float32)]),
    (Conv2D976, [((1, 12, 208, 208), torch.float32)]),
    (Conv2D295, [((1, 16, 208, 208), torch.float32), ((16, 1, 3, 3), torch.float32)]),
    (Conv2D433, [((1, 16, 104, 104), torch.float32)]),
    (Conv2D256, [((1, 32, 104, 104), torch.float32)]),
    (Conv2D610, [((1, 16, 104, 104), torch.float32)]),
    (Conv2D296, [((1, 16, 104, 104), torch.float32), ((16, 1, 3, 3), torch.float32)]),
    (Conv2D142, [((1, 32, 104, 104), torch.float32)]),
    (Conv2D977, [((1, 32, 104, 104), torch.float32), ((32, 1, 3, 3), torch.float32)]),
    (Conv2D128, [((1, 32, 52, 52), torch.float32)]),
    (Conv2D145, [((1, 64, 52, 52), torch.float32)]),
    (Conv2D142, [((1, 32, 52, 52), torch.float32)]),
    (Conv2D253, [((1, 32, 52, 52), torch.float32), ((32, 1, 3, 3), torch.float32)]),
    (Conv2D130, [((1, 64, 52, 52), torch.float32)]),
    (Conv2D558, [((1, 64, 52, 52), torch.float32), ((64, 1, 3, 3), torch.float32)]),
    (Conv2D567, [((1, 64, 52, 52), torch.float32), ((64, 1, 3, 3), torch.float32)]),
    (Conv2D13, [((1, 64, 26, 26), torch.float32)]),
    (Conv2D132, [((1, 128, 26, 26), torch.float32)]),
    (Conv2D130, [((1, 64, 26, 26), torch.float32)]),
    (Conv2D567, [((1, 64, 26, 26), torch.float32), ((64, 1, 3, 3), torch.float32)]),
    (Conv2D17, [((1, 128, 26, 26), torch.float32)]),
    (Conv2D560, [((1, 128, 26, 26), torch.float32), ((128, 1, 3, 3), torch.float32)]),
    (Conv2D136, [((1, 128, 13, 13), torch.float32)]),
    (Conv2D21, [((1, 256, 13, 13), torch.float32)]),
    (Conv2D29, [((1, 512, 13, 13), torch.float32)]),
    (Conv2D17, [((1, 128, 13, 13), torch.float32)]),
    (Conv2D559, [((1, 128, 13, 13), torch.float32), ((128, 1, 3, 3), torch.float32)]),
    (Conv2D138, [((1, 256, 13, 13), torch.float32)]),
    (Conv2D133, [((1, 256, 26, 26), torch.float32)]),
    (Conv2D424, [((1, 128, 52, 52), torch.float32)]),
    (Conv2D978, [((1, 64, 52, 52), torch.float32)]),
    (Conv2D979, [((1, 64, 52, 52), torch.float32)]),
    (Conv2D980, [((1, 64, 52, 52), torch.float32)]),
    (Conv2D978, [((1, 64, 26, 26), torch.float32)]),
    (Conv2D979, [((1, 64, 26, 26), torch.float32)]),
    (Conv2D980, [((1, 64, 26, 26), torch.float32)]),
    (Conv2D133, [((1, 256, 13, 13), torch.float32)]),
    (Conv2D567, [((1, 64, 13, 13), torch.float32), ((64, 1, 3, 3), torch.float32)]),
    (Conv2D130, [((1, 64, 13, 13), torch.float32)]),
    (Conv2D978, [((1, 64, 13, 13), torch.float32)]),
    (Conv2D979, [((1, 64, 13, 13), torch.float32)]),
    (Conv2D980, [((1, 64, 13, 13), torch.float32)]),
    (Conv2D728, [((1, 3, 640, 640), torch.float32)]),
    (Conv2D188, [((1, 32, 640, 640), torch.float32)]),
    (Conv2D145, [((1, 64, 320, 320), torch.float32)]),
    (Conv2D356, [((1, 32, 320, 320), torch.float32)]),
    (Conv2D188, [((1, 32, 320, 320), torch.float32)]),
    (Conv2D189, [((1, 64, 320, 320), torch.float32)]),
    (Conv2D132, [((1, 128, 160, 160), torch.float32)]),
    (Conv2D729, [((1, 64, 160, 160), torch.float32)]),
    (Conv2D189, [((1, 64, 160, 160), torch.float32)]),
    (Conv2D190, [((1, 128, 160, 160), torch.float32)]),
    (Conv2D21, [((1, 256, 80, 80), torch.float32)]),
    (Conv2D354, [((1, 128, 80, 80), torch.float32)]),
    (Conv2D190, [((1, 128, 80, 80), torch.float32)]),
    (Conv2D191, [((1, 256, 80, 80), torch.float32)]),
    (Conv2D29, [((1, 512, 40, 40), torch.float32)]),
    (Conv2D392, [((1, 256, 40, 40), torch.float32)]),
    (Conv2D191, [((1, 256, 40, 40), torch.float32)]),
    (Conv2D393, [((1, 512, 40, 40), torch.float32)]),
    (Conv2D75, [((1, 1024, 20, 20), torch.float32)]),
    (Conv2D981, [((1, 512, 20, 20), torch.float32)]),
    (Conv2D165, [((1, 2048, 20, 20), torch.float32)]),
    (Conv2D29, [((1, 512, 20, 20), torch.float32)]),
    (Conv2D158, [((1, 768, 40, 40), torch.float32)]),
    (Conv2D21, [((1, 256, 40, 40), torch.float32)]),
    (Conv2D25, [((1, 384, 80, 80), torch.float32)]),
    (Conv2D136, [((1, 128, 80, 80), torch.float32)]),
    (Conv2D6, [((1, 256, 80, 80), torch.float32)]),
    (Conv2D174, [((1, 256, 80, 80), torch.float32)]),
    (Conv2D982, [((1, 256, 80, 80), torch.float32)]),
    (Conv2D983, [((1, 256, 80, 80), torch.float32)]),
    (Conv2D984, [((1, 256, 80, 80), torch.float32)]),
    (Conv2D138, [((1, 256, 40, 40), torch.float32)]),
    (Conv2D6, [((1, 256, 40, 40), torch.float32)]),
    (Conv2D174, [((1, 256, 40, 40), torch.float32)]),
    (Conv2D982, [((1, 256, 40, 40), torch.float32)]),
    (Conv2D983, [((1, 256, 40, 40), torch.float32)]),
    (Conv2D984, [((1, 256, 40, 40), torch.float32)]),
    (Conv2D6, [((1, 256, 20, 20), torch.float32)]),
    (Conv2D982, [((1, 256, 20, 20), torch.float32)]),
    (Conv2D983, [((1, 256, 20, 20), torch.float32)]),
    (Conv2D984, [((1, 256, 20, 20), torch.float32)]),
    (Conv2D985, [((1, 12, 320, 320), torch.float32)]),
    (Conv2D130, [((1, 64, 160, 160), torch.float32)]),
    (Conv2D147, [((1, 64, 160, 160), torch.float32)]),
    (Conv2D17, [((1, 128, 160, 160), torch.float32)]),
    (Conv2D17, [((1, 128, 80, 80), torch.float32)]),
    (Conv2D149, [((1, 128, 80, 80), torch.float32)]),
    (Conv2D148, [((1, 128, 80, 80), torch.float32)]),
    (Conv2D138, [((1, 256, 80, 80), torch.float32)]),
    (Conv2D177, [((1, 512, 40, 40), torch.float32)]),
    (Conv2D169, [((1, 2048, 20, 20), torch.float32)]),
    (Conv2D177, [((1, 512, 20, 20), torch.float32)]),
    (Conv2D179, [((1, 512, 20, 20), torch.float32)]),
    (Conv2D187, [((1, 1024, 20, 20), torch.float32)]),
    (Conv2D528, [((1, 1024, 40, 40), torch.float32)]),
    (Conv2D30, [((1, 512, 80, 80), torch.float32)]),
    (Conv2D178, [((1, 512, 40, 40), torch.float32)]),
    (Conv2D528, [((1, 1024, 20, 20), torch.float32)]),
    (Conv2D986, [((1, 12, 320, 320), torch.float32)]),
    (Conv2D402, [((1, 80, 320, 320), torch.float32)]),
    (Conv2D405, [((1, 160, 160, 160), torch.float32)]),
    (Conv2D987, [((1, 80, 160, 160), torch.float32)]),
    (Conv2D398, [((1, 80, 160, 160), torch.float32)]),
    (Conv2D765, [((1, 160, 160, 160), torch.float32)]),
    (Conv2D358, [((1, 160, 160, 160), torch.float32)]),
    (Conv2D988, [((1, 320, 80, 80), torch.float32)]),
    (Conv2D765, [((1, 160, 80, 80), torch.float32)]),
    (Conv2D403, [((1, 160, 80, 80), torch.float32)]),
    (Conv2D989, [((1, 320, 80, 80), torch.float32)]),
    (Conv2D990, [((1, 320, 80, 80), torch.float32)]),
    (Conv2D991, [((1, 640, 40, 40), torch.float32)]),
    (Conv2D989, [((1, 320, 40, 40), torch.float32)]),
    (Conv2D407, [((1, 320, 40, 40), torch.float32)]),
    (Conv2D992, [((1, 640, 40, 40), torch.float32)]),
    (Conv2D993, [((1, 640, 40, 40), torch.float32)]),
    (Conv2D124, [((1, 1280, 20, 20), torch.float32)]),
    (Conv2D994, [((1, 2560, 20, 20), torch.float32)]),
    (Conv2D992, [((1, 640, 20, 20), torch.float32)]),
    (Conv2D995, [((1, 640, 20, 20), torch.float32)]),
    (Conv2D996, [((1, 1280, 20, 20), torch.float32)]),
    (Conv2D997, [((1, 1280, 40, 40), torch.float32)]),
    (Conv2D998, [((1, 640, 80, 80), torch.float32)]),
    (Conv2D407, [((1, 320, 80, 80), torch.float32)]),
    (Conv2D999, [((1, 320, 80, 80), torch.float32)]),
    (Conv2D1000, [((1, 320, 80, 80), torch.float32)]),
    (Conv2D1001, [((1, 320, 80, 80), torch.float32)]),
    (Conv2D1002, [((1, 320, 80, 80), torch.float32)]),
    (Conv2D1000, [((1, 320, 40, 40), torch.float32)]),
    (Conv2D1001, [((1, 320, 40, 40), torch.float32)]),
    (Conv2D1002, [((1, 320, 40, 40), torch.float32)]),
    (Conv2D1003, [((1, 640, 40, 40), torch.float32)]),
    (Conv2D997, [((1, 1280, 20, 20), torch.float32)]),
    (Conv2D407, [((1, 320, 20, 20), torch.float32)]),
    (Conv2D1000, [((1, 320, 20, 20), torch.float32)]),
    (Conv2D1001, [((1, 320, 20, 20), torch.float32)]),
    (Conv2D1002, [((1, 320, 20, 20), torch.float32)]),
    (Conv2D1004, [((1, 12, 320, 320), torch.float32)]),
    (Conv2D145, [((1, 64, 160, 160), torch.float32)]),
    (Conv2D142, [((1, 32, 160, 160), torch.float32)]),
    (Conv2D144, [((1, 32, 160, 160), torch.float32)]),
    (Conv2D132, [((1, 128, 80, 80), torch.float32)]),
    (Conv2D130, [((1, 64, 80, 80), torch.float32)]),
    (Conv2D147, [((1, 64, 80, 80), torch.float32)]),
    (Conv2D17, [((1, 128, 40, 40), torch.float32)]),
    (Conv2D149, [((1, 128, 40, 40), torch.float32)]),
    (Conv2D138, [((1, 256, 20, 20), torch.float32)]),
    (Conv2D30, [((1, 512, 40, 40), torch.float32)]),
    (Conv2D133, [((1, 256, 80, 80), torch.float32)]),
    (Conv2D1005, [((1, 128, 80, 80), torch.float32)]),
    (Conv2D1006, [((1, 128, 80, 80), torch.float32)]),
    (Conv2D1007, [((1, 128, 80, 80), torch.float32)]),
    (Conv2D1005, [((1, 128, 40, 40), torch.float32)]),
    (Conv2D1006, [((1, 128, 40, 40), torch.float32)]),
    (Conv2D1007, [((1, 128, 40, 40), torch.float32)]),
    (Conv2D30, [((1, 512, 20, 20), torch.float32)]),
    (Conv2D149, [((1, 128, 20, 20), torch.float32)]),
    (Conv2D1005, [((1, 128, 20, 20), torch.float32)]),
    (Conv2D1006, [((1, 128, 20, 20), torch.float32)]),
    (Conv2D1007, [((1, 128, 20, 20), torch.float32)]),
    (Conv2D1008, [((1, 12, 208, 208), torch.float32)]),
    (Conv2D1009, [((1, 24, 208, 208), torch.float32)]),
    (Conv2D199, [((1, 48, 104, 104), torch.float32)]),
    (Conv2D203, [((1, 24, 104, 104), torch.float32)]),
    (Conv2D1010, [((1, 24, 104, 104), torch.float32)]),
    (Conv2D968, [((1, 48, 104, 104), torch.float32)]),
    (Conv2D467, [((1, 48, 104, 104), torch.float32)]),
    (Conv2D466, [((1, 96, 52, 52), torch.float32)]),
    (Conv2D968, [((1, 48, 52, 52), torch.float32)]),
    (Conv2D461, [((1, 48, 52, 52), torch.float32)]),
    (Conv2D549, [((1, 96, 52, 52), torch.float32)]),
    (Conv2D468, [((1, 96, 52, 52), torch.float32)]),
    (Conv2D471, [((1, 192, 26, 26), torch.float32)]),
    (Conv2D549, [((1, 96, 26, 26), torch.float32)]),
    (Conv2D464, [((1, 96, 26, 26), torch.float32)]),
    (Conv2D80, [((1, 192, 26, 26), torch.float32)]),
    (Conv2D473, [((1, 192, 26, 26), torch.float32)]),
    (Conv2D84, [((1, 384, 13, 13), torch.float32)]),
    (Conv2D92, [((1, 768, 13, 13), torch.float32)]),
    (Conv2D80, [((1, 192, 13, 13), torch.float32)]),
    (Conv2D469, [((1, 192, 13, 13), torch.float32)]),
    (Conv2D554, [((1, 384, 13, 13), torch.float32)]),
    (Conv2D518, [((1, 384, 26, 26), torch.float32)]),
    (Conv2D470, [((1, 192, 52, 52), torch.float32)]),
    (Conv2D464, [((1, 96, 52, 52), torch.float32)]),
    (Conv2D465, [((1, 96, 52, 52), torch.float32)]),
    (Conv2D260, [((1, 96, 52, 52), torch.float32)]),
    (Conv2D1011, [((1, 96, 52, 52), torch.float32)]),
    (Conv2D1012, [((1, 96, 52, 52), torch.float32)]),
    (Conv2D260, [((1, 96, 26, 26), torch.float32)]),
    (Conv2D1011, [((1, 96, 26, 26), torch.float32)]),
    (Conv2D1012, [((1, 96, 26, 26), torch.float32)]),
    (Conv2D971, [((1, 192, 26, 26), torch.float32)]),
    (Conv2D518, [((1, 384, 13, 13), torch.float32)]),
    (Conv2D464, [((1, 96, 13, 13), torch.float32)]),
    (Conv2D260, [((1, 96, 13, 13), torch.float32)]),
    (Conv2D1011, [((1, 96, 13, 13), torch.float32)]),
    (Conv2D1012, [((1, 96, 13, 13), torch.float32)]),
]


@pytest.mark.push
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    forge_module, operand_shapes_dtypes = forge_module_and_shapes_dtypes

    inputs = [
        Tensor.create_from_shape(operand_shape, operand_dtype) for operand_shape, operand_dtype in operand_shapes_dtypes
    ]

    framework_model = forge_module(forge_module.__name__)
    framework_model.process_framework_parameters()

    for name, parameter in framework_model._parameters.items():
        parameter_tensor = Tensor.create_torch_tensor(
            shape=parameter.shape.get_pytorch_shape(), dtype=parameter.pt_data_format
        )
        framework_model.set_parameter(name, parameter_tensor)

    for name, constant in framework_model._constants.items():
        constant_tensor = Tensor.create_torch_tensor(
            shape=constant.shape.get_pytorch_shape(), dtype=constant.pt_data_format
        )
        framework_model.set_constant(name, constant_tensor)

    compiled_model = compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)
