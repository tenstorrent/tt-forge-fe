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


class Multiply0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply0_const_1", shape=(1,), dtype=torch.float32)

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_constant("multiply0_const_1"))
        return multiply_output_1


class Multiply1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, multiply_input_0, multiply_input_1):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, multiply_input_1)
        return multiply_output_1


class Multiply2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply2.weight_0", forge.Parameter(*(768,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, multiply_input_1):
        multiply_output_1 = forge.op.Multiply("", self.get_parameter("multiply2.weight_0"), multiply_input_1)
        return multiply_output_1


class Multiply3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply3_const_0", shape=(2, 1, 1, 13), dtype=torch.float32)

    def forward(self, multiply_input_1):
        multiply_output_1 = forge.op.Multiply("", self.get_constant("multiply3_const_0"), multiply_input_1)
        return multiply_output_1


class Multiply4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply4_const_0", shape=(2, 1, 7, 7), dtype=torch.float32)

    def forward(self, multiply_input_1):
        multiply_output_1 = forge.op.Multiply("", self.get_constant("multiply4_const_0"), multiply_input_1)
        return multiply_output_1


class Multiply5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply5_const_0", shape=(1, 1, 256, 256), dtype=torch.float32)

    def forward(self, multiply_input_1):
        multiply_output_1 = forge.op.Multiply("", self.get_constant("multiply5_const_0"), multiply_input_1)
        return multiply_output_1


class Multiply6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply6_const_1", shape=(1, 256, 1, 32), dtype=torch.float32)

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_constant("multiply6_const_1"))
        return multiply_output_1


class Multiply7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply7_const_0", shape=(1, 12, 128, 128), dtype=torch.float32)

    def forward(self, multiply_input_1):
        multiply_output_1 = forge.op.Multiply("", self.get_constant("multiply7_const_0"), multiply_input_1)
        return multiply_output_1


class Multiply8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply8_const_0", shape=(1, 12, 384, 384), dtype=torch.float32)

    def forward(self, multiply_input_1):
        multiply_output_1 = forge.op.Multiply("", self.get_constant("multiply8_const_0"), multiply_input_1)
        return multiply_output_1


class Multiply9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply9.weight_0",
            forge.Parameter(*(4096,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_1):
        multiply_output_1 = forge.op.Multiply("", self.get_parameter("multiply9.weight_0"), multiply_input_1)
        return multiply_output_1


class Multiply10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply10.weight_0",
            forge.Parameter(*(2048,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_1):
        multiply_output_1 = forge.op.Multiply("", self.get_parameter("multiply10.weight_0"), multiply_input_1)
        return multiply_output_1


class Multiply11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply11_const_0", shape=(4096,), dtype=torch.float32)

    def forward(self, multiply_input_1):
        multiply_output_1 = forge.op.Multiply("", self.get_constant("multiply11_const_0"), multiply_input_1)
        return multiply_output_1


class Multiply12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply12_const_0", shape=(1, 1, 32, 32), dtype=torch.float32)

    def forward(self, multiply_input_1):
        multiply_output_1 = forge.op.Multiply("", self.get_constant("multiply12_const_0"), multiply_input_1)
        return multiply_output_1


class Multiply13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply13.weight_0",
            forge.Parameter(*(1024,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_1):
        multiply_output_1 = forge.op.Multiply("", self.get_parameter("multiply13.weight_0"), multiply_input_1)
        return multiply_output_1


class Multiply14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply14.weight_0",
            forge.Parameter(*(1536,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_1):
        multiply_output_1 = forge.op.Multiply("", self.get_parameter("multiply14.weight_0"), multiply_input_1)
        return multiply_output_1


class Multiply15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply15.weight_0",
            forge.Parameter(*(3584,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_1):
        multiply_output_1 = forge.op.Multiply("", self.get_parameter("multiply15.weight_0"), multiply_input_1)
        return multiply_output_1


class Multiply16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply16.weight_0",
            forge.Parameter(*(896,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_1):
        multiply_output_1 = forge.op.Multiply("", self.get_parameter("multiply16.weight_0"), multiply_input_1)
        return multiply_output_1


class Multiply17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply17.weight_0",
            forge.Parameter(*(512,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_1):
        multiply_output_1 = forge.op.Multiply("", self.get_parameter("multiply17.weight_0"), multiply_input_1)
        return multiply_output_1


class Multiply18(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply18_const_0", shape=(1,), dtype=torch.float32)

    def forward(self, multiply_input_1):
        multiply_output_1 = forge.op.Multiply("", self.get_constant("multiply18_const_0"), multiply_input_1)
        return multiply_output_1


class Multiply19(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply19.weight_1", forge.Parameter(*(64,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply19.weight_1"))
        return multiply_output_1


class Multiply20(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply20.weight_1",
            forge.Parameter(*(128,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply20.weight_1"))
        return multiply_output_1


class Multiply21(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply21.weight_1", forge.Parameter(*(96,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply21.weight_1"))
        return multiply_output_1


class Multiply22(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply22.weight_1",
            forge.Parameter(*(160,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply22.weight_1"))
        return multiply_output_1


class Multiply23(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply23.weight_1",
            forge.Parameter(*(192,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply23.weight_1"))
        return multiply_output_1


class Multiply24(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply24.weight_1",
            forge.Parameter(*(224,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply24.weight_1"))
        return multiply_output_1


class Multiply25(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply25.weight_1",
            forge.Parameter(*(256,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply25.weight_1"))
        return multiply_output_1


class Multiply26(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply26.weight_1",
            forge.Parameter(*(288,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply26.weight_1"))
        return multiply_output_1


class Multiply27(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply27.weight_1",
            forge.Parameter(*(320,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply27.weight_1"))
        return multiply_output_1


class Multiply28(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply28.weight_1",
            forge.Parameter(*(352,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply28.weight_1"))
        return multiply_output_1


class Multiply29(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply29.weight_1",
            forge.Parameter(*(384,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply29.weight_1"))
        return multiply_output_1


class Multiply30(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply30.weight_1",
            forge.Parameter(*(416,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply30.weight_1"))
        return multiply_output_1


class Multiply31(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply31.weight_1",
            forge.Parameter(*(448,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply31.weight_1"))
        return multiply_output_1


class Multiply32(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply32.weight_1",
            forge.Parameter(*(480,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply32.weight_1"))
        return multiply_output_1


class Multiply33(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply33.weight_1",
            forge.Parameter(*(512,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply33.weight_1"))
        return multiply_output_1


class Multiply34(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply34.weight_1",
            forge.Parameter(*(544,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply34.weight_1"))
        return multiply_output_1


class Multiply35(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply35.weight_1",
            forge.Parameter(*(576,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply35.weight_1"))
        return multiply_output_1


class Multiply36(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply36.weight_1",
            forge.Parameter(*(608,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply36.weight_1"))
        return multiply_output_1


class Multiply37(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply37.weight_1",
            forge.Parameter(*(640,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply37.weight_1"))
        return multiply_output_1


class Multiply38(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply38.weight_1",
            forge.Parameter(*(672,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply38.weight_1"))
        return multiply_output_1


class Multiply39(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply39.weight_1",
            forge.Parameter(*(704,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply39.weight_1"))
        return multiply_output_1


class Multiply40(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply40.weight_1",
            forge.Parameter(*(736,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply40.weight_1"))
        return multiply_output_1


class Multiply41(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply41.weight_1",
            forge.Parameter(*(768,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply41.weight_1"))
        return multiply_output_1


class Multiply42(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply42.weight_1",
            forge.Parameter(*(800,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply42.weight_1"))
        return multiply_output_1


class Multiply43(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply43.weight_1",
            forge.Parameter(*(832,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply43.weight_1"))
        return multiply_output_1


class Multiply44(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply44.weight_1",
            forge.Parameter(*(864,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply44.weight_1"))
        return multiply_output_1


class Multiply45(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply45.weight_1",
            forge.Parameter(*(896,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply45.weight_1"))
        return multiply_output_1


class Multiply46(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply46.weight_1",
            forge.Parameter(*(928,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply46.weight_1"))
        return multiply_output_1


class Multiply47(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply47.weight_1",
            forge.Parameter(*(960,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply47.weight_1"))
        return multiply_output_1


class Multiply48(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply48.weight_1",
            forge.Parameter(*(992,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply48.weight_1"))
        return multiply_output_1


class Multiply49(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply49.weight_1",
            forge.Parameter(*(1024,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply49.weight_1"))
        return multiply_output_1


class Multiply50(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply50.weight_1",
            forge.Parameter(*(1056,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply50.weight_1"))
        return multiply_output_1


class Multiply51(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply51.weight_1",
            forge.Parameter(*(1088,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply51.weight_1"))
        return multiply_output_1


class Multiply52(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply52.weight_1",
            forge.Parameter(*(1120,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply52.weight_1"))
        return multiply_output_1


class Multiply53(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply53.weight_1",
            forge.Parameter(*(1152,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply53.weight_1"))
        return multiply_output_1


class Multiply54(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply54.weight_1",
            forge.Parameter(*(1184,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply54.weight_1"))
        return multiply_output_1


class Multiply55(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply55.weight_1",
            forge.Parameter(*(1216,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply55.weight_1"))
        return multiply_output_1


class Multiply56(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply56.weight_1",
            forge.Parameter(*(1248,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply56.weight_1"))
        return multiply_output_1


class Multiply57(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply57.weight_1",
            forge.Parameter(*(1280,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply57.weight_1"))
        return multiply_output_1


class Multiply58(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply58.weight_1",
            forge.Parameter(*(1312,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply58.weight_1"))
        return multiply_output_1


class Multiply59(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply59.weight_1",
            forge.Parameter(*(1344,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply59.weight_1"))
        return multiply_output_1


class Multiply60(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply60.weight_1",
            forge.Parameter(*(1376,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply60.weight_1"))
        return multiply_output_1


class Multiply61(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply61.weight_1",
            forge.Parameter(*(1408,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply61.weight_1"))
        return multiply_output_1


class Multiply62(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply62.weight_1",
            forge.Parameter(*(1440,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply62.weight_1"))
        return multiply_output_1


class Multiply63(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply63.weight_1",
            forge.Parameter(*(1472,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply63.weight_1"))
        return multiply_output_1


class Multiply64(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply64.weight_1",
            forge.Parameter(*(1504,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply64.weight_1"))
        return multiply_output_1


class Multiply65(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply65.weight_1",
            forge.Parameter(*(1536,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply65.weight_1"))
        return multiply_output_1


class Multiply66(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply66.weight_1",
            forge.Parameter(*(1568,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply66.weight_1"))
        return multiply_output_1


class Multiply67(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply67.weight_1",
            forge.Parameter(*(1600,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply67.weight_1"))
        return multiply_output_1


class Multiply68(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply68.weight_1",
            forge.Parameter(*(1632,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply68.weight_1"))
        return multiply_output_1


class Multiply69(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply69.weight_1",
            forge.Parameter(*(1664,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply69.weight_1"))
        return multiply_output_1


class Multiply70(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply70.weight_1",
            forge.Parameter(*(1696,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply70.weight_1"))
        return multiply_output_1


class Multiply71(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply71.weight_1",
            forge.Parameter(*(1728,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply71.weight_1"))
        return multiply_output_1


class Multiply72(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply72.weight_1",
            forge.Parameter(*(1760,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply72.weight_1"))
        return multiply_output_1


class Multiply73(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply73.weight_1",
            forge.Parameter(*(1792,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply73.weight_1"))
        return multiply_output_1


class Multiply74(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply74.weight_1",
            forge.Parameter(*(1824,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply74.weight_1"))
        return multiply_output_1


class Multiply75(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply75.weight_1",
            forge.Parameter(*(1856,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply75.weight_1"))
        return multiply_output_1


class Multiply76(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply76.weight_1",
            forge.Parameter(*(1888,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply76.weight_1"))
        return multiply_output_1


class Multiply77(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply77.weight_1",
            forge.Parameter(*(1920,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply77.weight_1"))
        return multiply_output_1


class Multiply78(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply78.weight_1",
            forge.Parameter(*(144,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply78.weight_1"))
        return multiply_output_1


class Multiply79(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply79.weight_1",
            forge.Parameter(*(240,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply79.weight_1"))
        return multiply_output_1


class Multiply80(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply80.weight_1",
            forge.Parameter(*(336,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply80.weight_1"))
        return multiply_output_1


class Multiply81(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply81.weight_1",
            forge.Parameter(*(432,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply81.weight_1"))
        return multiply_output_1


class Multiply82(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply82.weight_1",
            forge.Parameter(*(528,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply82.weight_1"))
        return multiply_output_1


class Multiply83(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply83.weight_1",
            forge.Parameter(*(624,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply83.weight_1"))
        return multiply_output_1


class Multiply84(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply84.weight_1",
            forge.Parameter(*(720,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply84.weight_1"))
        return multiply_output_1


class Multiply85(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply85.weight_1",
            forge.Parameter(*(816,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply85.weight_1"))
        return multiply_output_1


class Multiply86(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply86.weight_1",
            forge.Parameter(*(912,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply86.weight_1"))
        return multiply_output_1


class Multiply87(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply87.weight_1",
            forge.Parameter(*(1008,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply87.weight_1"))
        return multiply_output_1


class Multiply88(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply88.weight_1",
            forge.Parameter(*(1104,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply88.weight_1"))
        return multiply_output_1


class Multiply89(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply89.weight_1",
            forge.Parameter(*(1200,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply89.weight_1"))
        return multiply_output_1


class Multiply90(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply90.weight_1",
            forge.Parameter(*(1296,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply90.weight_1"))
        return multiply_output_1


class Multiply91(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply91.weight_1",
            forge.Parameter(*(1392,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply91.weight_1"))
        return multiply_output_1


class Multiply92(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply92.weight_1",
            forge.Parameter(*(1488,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply92.weight_1"))
        return multiply_output_1


class Multiply93(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply93.weight_1",
            forge.Parameter(*(1584,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply93.weight_1"))
        return multiply_output_1


class Multiply94(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply94.weight_1",
            forge.Parameter(*(1680,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply94.weight_1"))
        return multiply_output_1


class Multiply95(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply95.weight_1",
            forge.Parameter(*(1776,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply95.weight_1"))
        return multiply_output_1


class Multiply96(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply96.weight_1",
            forge.Parameter(*(1872,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply96.weight_1"))
        return multiply_output_1


class Multiply97(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply97.weight_1",
            forge.Parameter(*(1968,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply97.weight_1"))
        return multiply_output_1


class Multiply98(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply98.weight_1",
            forge.Parameter(*(2016,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply98.weight_1"))
        return multiply_output_1


class Multiply99(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply99.weight_1",
            forge.Parameter(*(2064,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply99.weight_1"))
        return multiply_output_1


class Multiply100(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply100.weight_1",
            forge.Parameter(*(2112,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply100.weight_1"))
        return multiply_output_1


class Multiply101(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply101.weight_1",
            forge.Parameter(*(2160,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply101.weight_1"))
        return multiply_output_1


class Multiply102(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply102.weight_1",
            forge.Parameter(*(2208,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply102.weight_1"))
        return multiply_output_1


class Multiply103(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply103.weight_1",
            forge.Parameter(*(16,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply103.weight_1"))
        return multiply_output_1


class Multiply104(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply104.weight_1",
            forge.Parameter(*(32,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply104.weight_1"))
        return multiply_output_1


class Multiply105(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply105.weight_1",
            forge.Parameter(*(2048,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply105.weight_1"))
        return multiply_output_1


class Multiply106(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply106.weight_1",
            forge.Parameter(*(48,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply106.weight_1"))
        return multiply_output_1


class Multiply107(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply107.weight_1",
            forge.Parameter(*(24,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply107.weight_1"))
        return multiply_output_1


class Multiply108(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply108.weight_1",
            forge.Parameter(*(56,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply108.weight_1"))
        return multiply_output_1


class Multiply109(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply109.weight_1",
            forge.Parameter(*(112,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply109.weight_1"))
        return multiply_output_1


class Multiply110(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply110.weight_1",
            forge.Parameter(*(272,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply110.weight_1"))
        return multiply_output_1


class Multiply111(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply111.weight_1",
            forge.Parameter(*(2688,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply111.weight_1"))
        return multiply_output_1


class Multiply112(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply112.weight_1",
            forge.Parameter(*(40,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply112.weight_1"))
        return multiply_output_1


class Multiply113(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply113.weight_1",
            forge.Parameter(*(80,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply113.weight_1"))
        return multiply_output_1


class Multiply114(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply114.weight_1", forge.Parameter(*(8,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply114.weight_1"))
        return multiply_output_1


class Multiply115(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply115.weight_1",
            forge.Parameter(*(12,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply115.weight_1"))
        return multiply_output_1


class Multiply116(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply116.weight_1",
            forge.Parameter(*(36,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply116.weight_1"))
        return multiply_output_1


class Multiply117(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply117.weight_1",
            forge.Parameter(*(72,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply117.weight_1"))
        return multiply_output_1


class Multiply118(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply118.weight_1",
            forge.Parameter(*(20,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply118.weight_1"))
        return multiply_output_1


class Multiply119(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply119.weight_1",
            forge.Parameter(*(60,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply119.weight_1"))
        return multiply_output_1


class Multiply120(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply120.weight_1",
            forge.Parameter(*(120,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply120.weight_1"))
        return multiply_output_1


class Multiply121(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply121.weight_1",
            forge.Parameter(*(100,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply121.weight_1"))
        return multiply_output_1


class Multiply122(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply122.weight_1",
            forge.Parameter(*(92,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply122.weight_1"))
        return multiply_output_1


class Multiply123(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply123.weight_1",
            forge.Parameter(*(208,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply123.weight_1"))
        return multiply_output_1


class Multiply124(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply124.weight_1",
            forge.Parameter(*(18,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply124.weight_1"))
        return multiply_output_1


class Multiply125(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply125.weight_1",
            forge.Parameter(*(44,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply125.weight_1"))
        return multiply_output_1


class Multiply126(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply126.weight_1",
            forge.Parameter(*(88,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply126.weight_1"))
        return multiply_output_1


class Multiply127(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply127.weight_1",
            forge.Parameter(*(176,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply127.weight_1"))
        return multiply_output_1


class Multiply128(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply128.weight_1",
            forge.Parameter(*(30,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply128.weight_1"))
        return multiply_output_1


class Multiply129(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply129.weight_1",
            forge.Parameter(*(200,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply129.weight_1"))
        return multiply_output_1


class Multiply130(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply130.weight_1",
            forge.Parameter(*(184,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply130.weight_1"))
        return multiply_output_1


class Multiply131(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply131.weight_1",
            forge.Parameter(*(728,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply131.weight_1"))
        return multiply_output_1


class Multiply132(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply132_const_1", shape=(1, 255, 80, 80), dtype=torch.float32)

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_constant("multiply132_const_1"))
        return multiply_output_1


class Multiply133(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply133_const_1", shape=(1, 255, 40, 40), dtype=torch.float32)

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_constant("multiply133_const_1"))
        return multiply_output_1


class Multiply134(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply134_const_1", shape=(1, 255, 20, 20), dtype=torch.float32)

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_constant("multiply134_const_1"))
        return multiply_output_1


class Multiply135(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply135_const_1", shape=(1, 255, 60, 60), dtype=torch.float32)

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_constant("multiply135_const_1"))
        return multiply_output_1


class Multiply136(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply136_const_1", shape=(1, 255, 30, 30), dtype=torch.float32)

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_constant("multiply136_const_1"))
        return multiply_output_1


class Multiply137(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply137_const_1", shape=(1, 255, 15, 15), dtype=torch.float32)

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_constant("multiply137_const_1"))
        return multiply_output_1


class Multiply138(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply138_const_1", shape=(1, 255, 10, 10), dtype=torch.float32)

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_constant("multiply138_const_1"))
        return multiply_output_1


class Multiply139(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply139_const_1", shape=(1, 255, 160, 160), dtype=torch.float32)

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_constant("multiply139_const_1"))
        return multiply_output_1


class Multiply140(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply140_const_1", shape=(5880, 1), dtype=torch.float32)

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_constant("multiply140_const_1"))
        return multiply_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (Multiply0, [((2, 1, 1024), torch.float32)]),
    (Multiply1, [((2, 13, 768), torch.float32), ((2, 13, 768), torch.float32)]),
    (Multiply1, [((2, 13, 768), torch.float32), ((2, 13, 1), torch.float32)]),
    (Multiply2, [((2, 13, 768), torch.float32)]),
    (Multiply0, [((2, 1, 1, 13), torch.float32)]),
    (Multiply1, [((2, 13, 1024), torch.float32), ((2, 13, 1), torch.float32)]),
    (Multiply1, [((2, 1, 1, 13), torch.float32), ((2, 1, 1, 13), torch.float32)]),
    (Multiply3, [((2, 1, 1, 13), torch.float32)]),
    (Multiply0, [((2, 1, 2048), torch.float32)]),
    (Multiply1, [((2, 13, 2048), torch.float32), ((2, 13, 1), torch.float32)]),
    (Multiply0, [((2, 1, 1536), torch.float32)]),
    (Multiply1, [((2, 13, 1536), torch.float32), ((2, 13, 1), torch.float32)]),
    (Multiply0, [((1, 2, 1024), torch.float32)]),
    (Multiply0, [((1, 2, 512), torch.float32)]),
    (Multiply0, [((1, 2, 1280), torch.float32)]),
    (Multiply0, [((1, 2, 768), torch.float32)]),
    (Multiply0, [((1, 2, 384), torch.float32)]),
    (Multiply0, [((2, 7, 512), torch.float32)]),
    (Multiply1, [((2, 1, 7, 7), torch.float32), ((2, 1, 7, 7), torch.float32)]),
    (Multiply4, [((2, 1, 7, 7), torch.float32)]),
    (Multiply0, [((2, 7, 2048), torch.float32)]),
    (Multiply1, [((2, 7, 2048), torch.float32), ((2, 7, 2048), torch.float32)]),
    (Multiply0, [((1, 12, 204, 204), torch.float32)]),
    (Multiply0, [((1, 1, 1, 204), torch.float32)]),
    (Multiply0, [((1, 12, 201, 201), torch.float32)]),
    (Multiply0, [((1, 1, 1, 201), torch.float32)]),
    (Multiply0, [((1, 16, 128, 128), torch.float32)]),
    (Multiply0, [((1, 1, 1, 128), torch.float32)]),
    (Multiply0, [((1, 64, 128, 128), torch.float32)]),
    (Multiply0, [((1, 12, 128, 128), torch.float32)]),
    (Multiply0, [((1, 256, 1024), torch.float32)]),
    (Multiply1, [((1, 1, 256, 256), torch.float32), ((1, 1, 256, 256), torch.float32)]),
    (Multiply5, [((1, 1, 256, 256), torch.float32)]),
    (Multiply0, [((1, 16, 384, 384), torch.float32)]),
    (Multiply6, [((1, 256, 16, 32), torch.float32)]),
    (Multiply0, [((1, 256, 16, 16), torch.float32)]),
    (Multiply0, [((1, 16, 256, 256), torch.float32)]),
    (Multiply0, [((1, 12, 128, 64), torch.float32)]),
    (Multiply1, [((1, 12, 128, 128), torch.float32), ((1, 12, 128, 128), torch.float32)]),
    (Multiply7, [((1, 12, 128, 128), torch.float32)]),
    (Multiply0, [((1, 12, 384, 64), torch.float32)]),
    (Multiply1, [((1, 12, 384, 384), torch.float32), ((1, 12, 384, 384), torch.float32)]),
    (Multiply8, [((1, 12, 384, 384), torch.float32)]),
    (Multiply1, [((1, 71, 6, 64), torch.float32), ((1, 1, 6, 64), torch.float32)]),
    (Multiply0, [((1, 71, 6, 32), torch.float32)]),
    (Multiply1, [((1, 1, 6, 64), torch.float32), ((1, 1, 6, 64), torch.float32)]),
    (Multiply0, [((1, 1, 6, 32), torch.float32)]),
    (Multiply0, [((1, 71, 6, 6), torch.float32)]),
    (Multiply1, [((1, 64, 334, 32), torch.float32), ((1, 1, 334, 32), torch.float32)]),
    (Multiply0, [((1, 64, 334, 16), torch.float32)]),
    (Multiply0, [((1, 64, 334, 334), torch.float32)]),
    (Multiply1, [((1, 334, 16384), torch.float32), ((1, 334, 16384), torch.float32)]),
    (Multiply0, [((1, 7, 2048), torch.float32)]),
    (Multiply1, [((1, 7, 2048), torch.float32), ((1, 7, 2048), torch.float32)]),
    (Multiply1, [((1, 7, 2048), torch.float32), ((1, 7, 1), torch.float32)]),
    (Multiply1, [((1, 7, 2048), torch.float32), ((2048,), torch.float32)]),
    (Multiply1, [((1, 8, 7, 256), torch.float32), ((1, 1, 7, 256), torch.float32)]),
    (Multiply0, [((1, 8, 7, 128), torch.float32)]),
    (Multiply1, [((1, 1, 7, 256), torch.float32), ((1, 1, 7, 256), torch.float32)]),
    (Multiply0, [((1, 1, 7, 128), torch.float32)]),
    (Multiply0, [((1, 8, 7, 7), torch.float32)]),
    (Multiply1, [((1, 7, 16384), torch.float32), ((1, 7, 16384), torch.float32)]),
    (Multiply0, [((1, 12, 256, 256), torch.float32)]),
    (Multiply0, [((1, 1, 256, 256), torch.float32)]),
    (Multiply0, [((1, 1, 1, 256), torch.float32)]),
    (Multiply0, [((1, 1, 32, 32), torch.float32)]),
    (Multiply1, [((1, 256, 4096), torch.float32), ((1, 256, 4096), torch.float32)]),
    (Multiply1, [((1, 256, 4096), torch.float32), ((1, 256, 1), torch.float32)]),
    (Multiply9, [((1, 256, 4096), torch.float32)]),
    (Multiply1, [((1, 32, 256, 128), torch.float32), ((1, 1, 256, 128), torch.float32)]),
    (Multiply0, [((1, 32, 256, 64), torch.float32)]),
    (Multiply1, [((1, 8, 256, 128), torch.float32), ((1, 1, 256, 128), torch.float32)]),
    (Multiply0, [((1, 8, 256, 64), torch.float32)]),
    (Multiply0, [((1, 32, 256, 256), torch.float32)]),
    (Multiply1, [((1, 256, 14336), torch.float32), ((1, 256, 14336), torch.float32)]),
    (Multiply1, [((1, 4, 2048), torch.float32), ((1, 4, 2048), torch.float32)]),
    (Multiply1, [((1, 4, 2048), torch.float32), ((1, 4, 1), torch.float32)]),
    (Multiply10, [((1, 4, 2048), torch.float32)]),
    (Multiply1, [((1, 32, 4, 64), torch.float32), ((1, 1, 4, 64), torch.float32)]),
    (Multiply0, [((1, 32, 4, 32), torch.float32)]),
    (Multiply1, [((1, 8, 4, 64), torch.float32), ((1, 1, 4, 64), torch.float32)]),
    (Multiply0, [((1, 8, 4, 32), torch.float32)]),
    (Multiply0, [((1, 32, 4, 4), torch.float32)]),
    (Multiply1, [((1, 4, 8192), torch.float32), ((1, 4, 8192), torch.float32)]),
    (Multiply1, [((1, 4, 4096), torch.float32), ((1, 4, 4096), torch.float32)]),
    (Multiply1, [((1, 4, 4096), torch.float32), ((1, 4, 1), torch.float32)]),
    (Multiply9, [((1, 4, 4096), torch.float32)]),
    (Multiply1, [((1, 32, 4, 128), torch.float32), ((1, 1, 4, 128), torch.float32)]),
    (Multiply0, [((1, 32, 4, 64), torch.float32)]),
    (Multiply1, [((1, 8, 4, 128), torch.float32), ((1, 1, 4, 128), torch.float32)]),
    (Multiply0, [((1, 8, 4, 64), torch.float32)]),
    (Multiply1, [((1, 4, 14336), torch.float32), ((1, 4, 14336), torch.float32)]),
    (Multiply1, [((1, 256, 2048), torch.float32), ((1, 256, 2048), torch.float32)]),
    (Multiply1, [((1, 256, 2048), torch.float32), ((1, 256, 1), torch.float32)]),
    (Multiply10, [((1, 256, 2048), torch.float32)]),
    (Multiply1, [((1, 32, 256, 64), torch.float32), ((1, 1, 256, 64), torch.float32)]),
    (Multiply0, [((1, 32, 256, 32), torch.float32)]),
    (Multiply1, [((1, 8, 256, 64), torch.float32), ((1, 1, 256, 64), torch.float32)]),
    (Multiply0, [((1, 8, 256, 32), torch.float32)]),
    (Multiply1, [((1, 256, 8192), torch.float32), ((1, 256, 8192), torch.float32)]),
    (Multiply1, [((1, 128, 4096), torch.float32), ((1, 128, 4096), torch.float32)]),
    (Multiply1, [((1, 128, 4096), torch.float32), ((1, 128, 1), torch.float32)]),
    (Multiply11, [((1, 128, 4096), torch.float32)]),
    (Multiply1, [((1, 32, 128, 128), torch.float32), ((1, 1, 128, 128), torch.float32)]),
    (Multiply0, [((1, 32, 128, 64), torch.float32)]),
    (Multiply1, [((1, 8, 128, 128), torch.float32), ((1, 1, 128, 128), torch.float32)]),
    (Multiply0, [((1, 8, 128, 64), torch.float32)]),
    (Multiply0, [((1, 32, 128, 128), torch.float32)]),
    (Multiply1, [((1, 128, 14336), torch.float32), ((1, 128, 14336), torch.float32)]),
    (Multiply1, [((1, 32), torch.float32), ((1, 32), torch.float32)]),
    (Multiply0, [((1, 32, 2048), torch.float32)]),
    (Multiply1, [((1, 1, 32, 32), torch.float32), ((1, 1, 32, 32), torch.float32)]),
    (Multiply12, [((1, 1, 32, 32), torch.float32)]),
    (Multiply1, [((1, 256), torch.float32), ((1, 256), torch.float32)]),
    (Multiply0, [((1, 256, 2048), torch.float32)]),
    (Multiply0, [((1, 32, 1024), torch.float32)]),
    (Multiply0, [((1, 32, 768), torch.float32)]),
    (Multiply0, [((1, 256, 768), torch.float32)]),
    (Multiply1, [((1, 32, 256, 32), torch.float32), ((1, 1, 256, 32), torch.float32)]),
    (Multiply0, [((1, 32, 256, 16), torch.float32)]),
    (Multiply1, [((1, 32, 12, 32), torch.float32), ((1, 1, 12, 32), torch.float32)]),
    (Multiply0, [((1, 32, 12, 16), torch.float32)]),
    (Multiply0, [((1, 32, 12, 12), torch.float32)]),
    (Multiply1, [((1, 32, 11, 32), torch.float32), ((1, 1, 11, 32), torch.float32)]),
    (Multiply0, [((1, 32, 11, 16), torch.float32)]),
    (Multiply0, [((1, 32, 11, 11), torch.float32)]),
    (Multiply1, [((1, 6, 1024), torch.float32), ((1, 6, 1024), torch.float32)]),
    (Multiply1, [((1, 6, 1024), torch.float32), ((1, 6, 1), torch.float32)]),
    (Multiply13, [((1, 6, 1024), torch.float32)]),
    (Multiply1, [((1, 16, 6, 64), torch.float32), ((1, 1, 6, 64), torch.float32)]),
    (Multiply0, [((1, 16, 6, 32), torch.float32)]),
    (Multiply0, [((1, 16, 6, 6), torch.float32)]),
    (Multiply1, [((1, 6, 2816), torch.float32), ((1, 6, 2816), torch.float32)]),
    (Multiply1, [((1, 29, 1024), torch.float32), ((1, 29, 1024), torch.float32)]),
    (Multiply1, [((1, 29, 1024), torch.float32), ((1, 29, 1), torch.float32)]),
    (Multiply13, [((1, 29, 1024), torch.float32)]),
    (Multiply1, [((1, 16, 29, 64), torch.float32), ((1, 1, 29, 64), torch.float32)]),
    (Multiply0, [((1, 16, 29, 32), torch.float32)]),
    (Multiply0, [((1, 16, 29, 29), torch.float32)]),
    (Multiply1, [((1, 29, 2816), torch.float32), ((1, 29, 2816), torch.float32)]),
    (Multiply1, [((1, 35, 1536), torch.float32), ((1, 35, 1536), torch.float32)]),
    (Multiply1, [((1, 35, 1536), torch.float32), ((1, 35, 1), torch.float32)]),
    (Multiply14, [((1, 35, 1536), torch.float32)]),
    (Multiply1, [((1, 12, 35, 128), torch.float32), ((1, 1, 35, 128), torch.float32)]),
    (Multiply0, [((1, 12, 35, 64), torch.float32)]),
    (Multiply1, [((1, 2, 35, 128), torch.float32), ((1, 1, 35, 128), torch.float32)]),
    (Multiply0, [((1, 2, 35, 64), torch.float32)]),
    (Multiply0, [((1, 12, 35, 35), torch.float32)]),
    (Multiply1, [((1, 35, 8960), torch.float32), ((1, 35, 8960), torch.float32)]),
    (Multiply1, [((1, 35, 3584), torch.float32), ((1, 35, 3584), torch.float32)]),
    (Multiply1, [((1, 35, 3584), torch.float32), ((1, 35, 1), torch.float32)]),
    (Multiply15, [((1, 35, 3584), torch.float32)]),
    (Multiply1, [((1, 28, 35, 128), torch.float32), ((1, 1, 35, 128), torch.float32)]),
    (Multiply0, [((1, 28, 35, 64), torch.float32)]),
    (Multiply1, [((1, 4, 35, 128), torch.float32), ((1, 1, 35, 128), torch.float32)]),
    (Multiply0, [((1, 4, 35, 64), torch.float32)]),
    (Multiply0, [((1, 28, 35, 35), torch.float32)]),
    (Multiply1, [((1, 35, 18944), torch.float32), ((1, 35, 18944), torch.float32)]),
    (Multiply1, [((1, 35, 2048), torch.float32), ((1, 35, 2048), torch.float32)]),
    (Multiply1, [((1, 35, 2048), torch.float32), ((1, 35, 1), torch.float32)]),
    (Multiply10, [((1, 35, 2048), torch.float32)]),
    (Multiply1, [((1, 16, 35, 128), torch.float32), ((1, 1, 35, 128), torch.float32)]),
    (Multiply0, [((1, 16, 35, 64), torch.float32)]),
    (Multiply0, [((1, 16, 35, 35), torch.float32)]),
    (Multiply1, [((1, 35, 11008), torch.float32), ((1, 35, 11008), torch.float32)]),
    (Multiply1, [((1, 35, 896), torch.float32), ((1, 35, 896), torch.float32)]),
    (Multiply1, [((1, 35, 896), torch.float32), ((1, 35, 1), torch.float32)]),
    (Multiply16, [((1, 35, 896), torch.float32)]),
    (Multiply1, [((1, 14, 35, 64), torch.float32), ((1, 1, 35, 64), torch.float32)]),
    (Multiply0, [((1, 14, 35, 32), torch.float32)]),
    (Multiply1, [((1, 2, 35, 64), torch.float32), ((1, 1, 35, 64), torch.float32)]),
    (Multiply0, [((1, 2, 35, 32), torch.float32)]),
    (Multiply0, [((1, 14, 35, 35), torch.float32)]),
    (Multiply1, [((1, 35, 4864), torch.float32), ((1, 35, 4864), torch.float32)]),
    (Multiply1, [((1, 29, 896), torch.float32), ((1, 29, 896), torch.float32)]),
    (Multiply1, [((1, 29, 896), torch.float32), ((1, 29, 1), torch.float32)]),
    (Multiply16, [((1, 29, 896), torch.float32)]),
    (Multiply1, [((1, 14, 29, 64), torch.float32), ((1, 1, 29, 64), torch.float32)]),
    (Multiply0, [((1, 14, 29, 32), torch.float32)]),
    (Multiply1, [((1, 2, 29, 64), torch.float32), ((1, 1, 29, 64), torch.float32)]),
    (Multiply0, [((1, 2, 29, 32), torch.float32)]),
    (Multiply0, [((1, 14, 29, 29), torch.float32)]),
    (Multiply1, [((1, 29, 4864), torch.float32), ((1, 29, 4864), torch.float32)]),
    (Multiply1, [((1, 39, 1536), torch.float32), ((1, 39, 1536), torch.float32)]),
    (Multiply1, [((1, 39, 1536), torch.float32), ((1, 39, 1), torch.float32)]),
    (Multiply14, [((1, 39, 1536), torch.float32)]),
    (Multiply1, [((1, 12, 39, 128), torch.float32), ((1, 1, 39, 128), torch.float32)]),
    (Multiply0, [((1, 12, 39, 64), torch.float32)]),
    (Multiply1, [((1, 2, 39, 128), torch.float32), ((1, 1, 39, 128), torch.float32)]),
    (Multiply0, [((1, 2, 39, 64), torch.float32)]),
    (Multiply0, [((1, 12, 39, 39), torch.float32)]),
    (Multiply1, [((1, 39, 8960), torch.float32), ((1, 39, 8960), torch.float32)]),
    (Multiply1, [((1, 29, 1536), torch.float32), ((1, 29, 1536), torch.float32)]),
    (Multiply1, [((1, 29, 1536), torch.float32), ((1, 29, 1), torch.float32)]),
    (Multiply14, [((1, 29, 1536), torch.float32)]),
    (Multiply1, [((1, 12, 29, 128), torch.float32), ((1, 1, 29, 128), torch.float32)]),
    (Multiply0, [((1, 12, 29, 64), torch.float32)]),
    (Multiply1, [((1, 2, 29, 128), torch.float32), ((1, 1, 29, 128), torch.float32)]),
    (Multiply0, [((1, 2, 29, 64), torch.float32)]),
    (Multiply0, [((1, 12, 29, 29), torch.float32)]),
    (Multiply1, [((1, 29, 8960), torch.float32), ((1, 29, 8960), torch.float32)]),
    (Multiply1, [((1, 29, 2048), torch.float32), ((1, 29, 2048), torch.float32)]),
    (Multiply1, [((1, 29, 2048), torch.float32), ((1, 29, 1), torch.float32)]),
    (Multiply10, [((1, 29, 2048), torch.float32)]),
    (Multiply1, [((1, 16, 29, 128), torch.float32), ((1, 1, 29, 128), torch.float32)]),
    (Multiply0, [((1, 16, 29, 64), torch.float32)]),
    (Multiply1, [((1, 29, 11008), torch.float32), ((1, 29, 11008), torch.float32)]),
    (Multiply1, [((1, 39, 2048), torch.float32), ((1, 39, 2048), torch.float32)]),
    (Multiply1, [((1, 39, 2048), torch.float32), ((1, 39, 1), torch.float32)]),
    (Multiply10, [((1, 39, 2048), torch.float32)]),
    (Multiply1, [((1, 16, 39, 128), torch.float32), ((1, 1, 39, 128), torch.float32)]),
    (Multiply0, [((1, 16, 39, 64), torch.float32)]),
    (Multiply0, [((1, 16, 39, 39), torch.float32)]),
    (Multiply1, [((1, 39, 11008), torch.float32), ((1, 39, 11008), torch.float32)]),
    (Multiply1, [((1, 39, 896), torch.float32), ((1, 39, 896), torch.float32)]),
    (Multiply1, [((1, 39, 896), torch.float32), ((1, 39, 1), torch.float32)]),
    (Multiply16, [((1, 39, 896), torch.float32)]),
    (Multiply1, [((1, 14, 39, 64), torch.float32), ((1, 1, 39, 64), torch.float32)]),
    (Multiply0, [((1, 14, 39, 32), torch.float32)]),
    (Multiply1, [((1, 2, 39, 64), torch.float32), ((1, 1, 39, 64), torch.float32)]),
    (Multiply0, [((1, 2, 39, 32), torch.float32)]),
    (Multiply0, [((1, 14, 39, 39), torch.float32)]),
    (Multiply1, [((1, 39, 4864), torch.float32), ((1, 39, 4864), torch.float32)]),
    (Multiply1, [((1, 39, 3584), torch.float32), ((1, 39, 3584), torch.float32)]),
    (Multiply1, [((1, 39, 3584), torch.float32), ((1, 39, 1), torch.float32)]),
    (Multiply15, [((1, 39, 3584), torch.float32)]),
    (Multiply1, [((1, 28, 39, 128), torch.float32), ((1, 1, 39, 128), torch.float32)]),
    (Multiply0, [((1, 28, 39, 64), torch.float32)]),
    (Multiply1, [((1, 4, 39, 128), torch.float32), ((1, 1, 39, 128), torch.float32)]),
    (Multiply0, [((1, 4, 39, 64), torch.float32)]),
    (Multiply0, [((1, 28, 39, 39), torch.float32)]),
    (Multiply1, [((1, 39, 18944), torch.float32), ((1, 39, 18944), torch.float32)]),
    (Multiply1, [((1, 29, 3584), torch.float32), ((1, 29, 3584), torch.float32)]),
    (Multiply1, [((1, 29, 3584), torch.float32), ((1, 29, 1), torch.float32)]),
    (Multiply15, [((1, 29, 3584), torch.float32)]),
    (Multiply1, [((1, 28, 29, 128), torch.float32), ((1, 1, 29, 128), torch.float32)]),
    (Multiply0, [((1, 28, 29, 64), torch.float32)]),
    (Multiply1, [((1, 4, 29, 128), torch.float32), ((1, 1, 29, 128), torch.float32)]),
    (Multiply0, [((1, 4, 29, 64), torch.float32)]),
    (Multiply0, [((1, 28, 29, 29), torch.float32)]),
    (Multiply1, [((1, 29, 18944), torch.float32), ((1, 29, 18944), torch.float32)]),
    (Multiply1, [((1, 128), torch.int32), ((1, 128), torch.int32)]),
    (Multiply1, [((1, 1, 512), torch.float32), ((1, 1, 512), torch.float32)]),
    (Multiply1, [((1, 1, 512), torch.float32), ((1, 1, 1), torch.float32)]),
    (Multiply17, [((1, 1, 512), torch.float32)]),
    (Multiply1, [((1, 1, 1024), torch.float32), ((1, 1, 1024), torch.float32)]),
    (Multiply1, [((1, 1, 768), torch.float32), ((1, 1, 768), torch.float32)]),
    (Multiply1, [((1, 1, 768), torch.float32), ((1, 1, 1), torch.float32)]),
    (Multiply2, [((1, 1, 768), torch.float32)]),
    (Multiply0, [((1, 1, 768), torch.float32)]),
    (Multiply1, [((1, 1, 1024), torch.float32), ((1, 1, 1), torch.float32)]),
    (Multiply13, [((1, 1, 1024), torch.float32)]),
    (Multiply0, [((1, 1, 1024), torch.float32)]),
    (Multiply1, [((1, 1, 2048), torch.float32), ((1, 1, 2048), torch.float32)]),
    (Multiply0, [((1, 1, 512), torch.float32)]),
    (Multiply1, [((1024, 72), torch.float32), ((1024, 72), torch.float32)]),
    (Multiply0, [((1, 12, 197, 197), torch.float32)]),
    (Multiply0, [((1, 3, 197, 197), torch.float32)]),
    (Multiply0, [((1, 6, 197, 197), torch.float32)]),
    (Multiply18, [((64,), torch.float32)]),
    (Multiply19, [((64,), torch.float32)]),
    (Multiply1, [((1, 64, 112, 112), torch.float32), ((64, 1, 1), torch.float32)]),
    (Multiply1, [((64,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((64,), torch.float32), ((64,), torch.float32)]),
    (Multiply1, [((1, 64, 56, 56), torch.float32), ((64, 1, 1), torch.float32)]),
    (Multiply18, [((128,), torch.float32)]),
    (Multiply20, [((128,), torch.float32)]),
    (Multiply1, [((1, 128, 56, 56), torch.float32), ((128, 1, 1), torch.float32)]),
    (Multiply1, [((128,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((128,), torch.float32), ((128,), torch.float32)]),
    (Multiply18, [((96,), torch.float32)]),
    (Multiply21, [((96,), torch.float32)]),
    (Multiply1, [((1, 96, 56, 56), torch.float32), ((96, 1, 1), torch.float32)]),
    (Multiply1, [((96,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((96,), torch.float32), ((96,), torch.float32)]),
    (Multiply18, [((160,), torch.float32)]),
    (Multiply22, [((160,), torch.float32)]),
    (Multiply1, [((1, 160, 56, 56), torch.float32), ((160, 1, 1), torch.float32)]),
    (Multiply1, [((160,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((160,), torch.float32), ((160,), torch.float32)]),
    (Multiply18, [((192,), torch.float32)]),
    (Multiply23, [((192,), torch.float32)]),
    (Multiply1, [((1, 192, 56, 56), torch.float32), ((192, 1, 1), torch.float32)]),
    (Multiply1, [((192,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((192,), torch.float32), ((192,), torch.float32)]),
    (Multiply18, [((224,), torch.float32)]),
    (Multiply24, [((224,), torch.float32)]),
    (Multiply1, [((1, 224, 56, 56), torch.float32), ((224, 1, 1), torch.float32)]),
    (Multiply1, [((224,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((224,), torch.float32), ((224,), torch.float32)]),
    (Multiply18, [((256,), torch.float32)]),
    (Multiply25, [((256,), torch.float32)]),
    (Multiply1, [((1, 256, 56, 56), torch.float32), ((256, 1, 1), torch.float32)]),
    (Multiply1, [((256,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((256,), torch.float32), ((256,), torch.float32)]),
    (Multiply1, [((1, 128, 28, 28), torch.float32), ((128, 1, 1), torch.float32)]),
    (Multiply1, [((1, 160, 28, 28), torch.float32), ((160, 1, 1), torch.float32)]),
    (Multiply1, [((1, 192, 28, 28), torch.float32), ((192, 1, 1), torch.float32)]),
    (Multiply1, [((1, 224, 28, 28), torch.float32), ((224, 1, 1), torch.float32)]),
    (Multiply1, [((1, 256, 28, 28), torch.float32), ((256, 1, 1), torch.float32)]),
    (Multiply18, [((288,), torch.float32)]),
    (Multiply26, [((288,), torch.float32)]),
    (Multiply1, [((1, 288, 28, 28), torch.float32), ((288, 1, 1), torch.float32)]),
    (Multiply1, [((288,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((288,), torch.float32), ((288,), torch.float32)]),
    (Multiply18, [((320,), torch.float32)]),
    (Multiply27, [((320,), torch.float32)]),
    (Multiply1, [((1, 320, 28, 28), torch.float32), ((320, 1, 1), torch.float32)]),
    (Multiply1, [((320,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((320,), torch.float32), ((320,), torch.float32)]),
    (Multiply18, [((352,), torch.float32)]),
    (Multiply28, [((352,), torch.float32)]),
    (Multiply1, [((1, 352, 28, 28), torch.float32), ((352, 1, 1), torch.float32)]),
    (Multiply1, [((352,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((352,), torch.float32), ((352,), torch.float32)]),
    (Multiply18, [((384,), torch.float32)]),
    (Multiply29, [((384,), torch.float32)]),
    (Multiply1, [((1, 384, 28, 28), torch.float32), ((384, 1, 1), torch.float32)]),
    (Multiply1, [((384,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((384,), torch.float32), ((384,), torch.float32)]),
    (Multiply18, [((416,), torch.float32)]),
    (Multiply30, [((416,), torch.float32)]),
    (Multiply1, [((1, 416, 28, 28), torch.float32), ((416, 1, 1), torch.float32)]),
    (Multiply1, [((416,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((416,), torch.float32), ((416,), torch.float32)]),
    (Multiply18, [((448,), torch.float32)]),
    (Multiply31, [((448,), torch.float32)]),
    (Multiply1, [((1, 448, 28, 28), torch.float32), ((448, 1, 1), torch.float32)]),
    (Multiply1, [((448,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((448,), torch.float32), ((448,), torch.float32)]),
    (Multiply18, [((480,), torch.float32)]),
    (Multiply32, [((480,), torch.float32)]),
    (Multiply1, [((1, 480, 28, 28), torch.float32), ((480, 1, 1), torch.float32)]),
    (Multiply1, [((480,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((480,), torch.float32), ((480,), torch.float32)]),
    (Multiply18, [((512,), torch.float32)]),
    (Multiply33, [((512,), torch.float32)]),
    (Multiply1, [((1, 512, 28, 28), torch.float32), ((512, 1, 1), torch.float32)]),
    (Multiply1, [((512,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((512,), torch.float32), ((512,), torch.float32)]),
    (Multiply1, [((1, 256, 14, 14), torch.float32), ((256, 1, 1), torch.float32)]),
    (Multiply1, [((1, 128, 14, 14), torch.float32), ((128, 1, 1), torch.float32)]),
    (Multiply1, [((1, 288, 14, 14), torch.float32), ((288, 1, 1), torch.float32)]),
    (Multiply1, [((1, 320, 14, 14), torch.float32), ((320, 1, 1), torch.float32)]),
    (Multiply1, [((1, 352, 14, 14), torch.float32), ((352, 1, 1), torch.float32)]),
    (Multiply1, [((1, 384, 14, 14), torch.float32), ((384, 1, 1), torch.float32)]),
    (Multiply1, [((1, 416, 14, 14), torch.float32), ((416, 1, 1), torch.float32)]),
    (Multiply1, [((1, 448, 14, 14), torch.float32), ((448, 1, 1), torch.float32)]),
    (Multiply1, [((1, 480, 14, 14), torch.float32), ((480, 1, 1), torch.float32)]),
    (Multiply1, [((1, 512, 14, 14), torch.float32), ((512, 1, 1), torch.float32)]),
    (Multiply18, [((544,), torch.float32)]),
    (Multiply34, [((544,), torch.float32)]),
    (Multiply1, [((1, 544, 14, 14), torch.float32), ((544, 1, 1), torch.float32)]),
    (Multiply1, [((544,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((544,), torch.float32), ((544,), torch.float32)]),
    (Multiply18, [((576,), torch.float32)]),
    (Multiply35, [((576,), torch.float32)]),
    (Multiply1, [((1, 576, 14, 14), torch.float32), ((576, 1, 1), torch.float32)]),
    (Multiply1, [((576,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((576,), torch.float32), ((576,), torch.float32)]),
    (Multiply18, [((608,), torch.float32)]),
    (Multiply36, [((608,), torch.float32)]),
    (Multiply1, [((1, 608, 14, 14), torch.float32), ((608, 1, 1), torch.float32)]),
    (Multiply1, [((608,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((608,), torch.float32), ((608,), torch.float32)]),
    (Multiply18, [((640,), torch.float32)]),
    (Multiply37, [((640,), torch.float32)]),
    (Multiply1, [((1, 640, 14, 14), torch.float32), ((640, 1, 1), torch.float32)]),
    (Multiply1, [((640,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((640,), torch.float32), ((640,), torch.float32)]),
    (Multiply18, [((672,), torch.float32)]),
    (Multiply38, [((672,), torch.float32)]),
    (Multiply1, [((1, 672, 14, 14), torch.float32), ((672, 1, 1), torch.float32)]),
    (Multiply1, [((672,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((672,), torch.float32), ((672,), torch.float32)]),
    (Multiply18, [((704,), torch.float32)]),
    (Multiply39, [((704,), torch.float32)]),
    (Multiply1, [((1, 704, 14, 14), torch.float32), ((704, 1, 1), torch.float32)]),
    (Multiply1, [((704,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((704,), torch.float32), ((704,), torch.float32)]),
    (Multiply18, [((736,), torch.float32)]),
    (Multiply40, [((736,), torch.float32)]),
    (Multiply1, [((1, 736, 14, 14), torch.float32), ((736, 1, 1), torch.float32)]),
    (Multiply1, [((736,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((736,), torch.float32), ((736,), torch.float32)]),
    (Multiply18, [((768,), torch.float32)]),
    (Multiply41, [((768,), torch.float32)]),
    (Multiply1, [((1, 768, 14, 14), torch.float32), ((768, 1, 1), torch.float32)]),
    (Multiply1, [((768,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((768,), torch.float32), ((768,), torch.float32)]),
    (Multiply18, [((800,), torch.float32)]),
    (Multiply42, [((800,), torch.float32)]),
    (Multiply1, [((1, 800, 14, 14), torch.float32), ((800, 1, 1), torch.float32)]),
    (Multiply1, [((800,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((800,), torch.float32), ((800,), torch.float32)]),
    (Multiply18, [((832,), torch.float32)]),
    (Multiply43, [((832,), torch.float32)]),
    (Multiply1, [((1, 832, 14, 14), torch.float32), ((832, 1, 1), torch.float32)]),
    (Multiply1, [((832,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((832,), torch.float32), ((832,), torch.float32)]),
    (Multiply18, [((864,), torch.float32)]),
    (Multiply44, [((864,), torch.float32)]),
    (Multiply1, [((1, 864, 14, 14), torch.float32), ((864, 1, 1), torch.float32)]),
    (Multiply1, [((864,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((864,), torch.float32), ((864,), torch.float32)]),
    (Multiply18, [((896,), torch.float32)]),
    (Multiply45, [((896,), torch.float32)]),
    (Multiply1, [((1, 896, 14, 14), torch.float32), ((896, 1, 1), torch.float32)]),
    (Multiply1, [((896,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((896,), torch.float32), ((896,), torch.float32)]),
    (Multiply18, [((928,), torch.float32)]),
    (Multiply46, [((928,), torch.float32)]),
    (Multiply1, [((1, 928, 14, 14), torch.float32), ((928, 1, 1), torch.float32)]),
    (Multiply1, [((928,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((928,), torch.float32), ((928,), torch.float32)]),
    (Multiply18, [((960,), torch.float32)]),
    (Multiply47, [((960,), torch.float32)]),
    (Multiply1, [((1, 960, 14, 14), torch.float32), ((960, 1, 1), torch.float32)]),
    (Multiply1, [((960,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((960,), torch.float32), ((960,), torch.float32)]),
    (Multiply18, [((992,), torch.float32)]),
    (Multiply48, [((992,), torch.float32)]),
    (Multiply1, [((1, 992, 14, 14), torch.float32), ((992, 1, 1), torch.float32)]),
    (Multiply1, [((992,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((992,), torch.float32), ((992,), torch.float32)]),
    (Multiply18, [((1024,), torch.float32)]),
    (Multiply49, [((1024,), torch.float32)]),
    (Multiply1, [((1, 1024, 14, 14), torch.float32), ((1024, 1, 1), torch.float32)]),
    (Multiply1, [((1024,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((1024,), torch.float32), ((1024,), torch.float32)]),
    (Multiply18, [((1056,), torch.float32)]),
    (Multiply50, [((1056,), torch.float32)]),
    (Multiply1, [((1, 1056, 14, 14), torch.float32), ((1056, 1, 1), torch.float32)]),
    (Multiply1, [((1056,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((1056,), torch.float32), ((1056,), torch.float32)]),
    (Multiply18, [((1088,), torch.float32)]),
    (Multiply51, [((1088,), torch.float32)]),
    (Multiply1, [((1, 1088, 14, 14), torch.float32), ((1088, 1, 1), torch.float32)]),
    (Multiply1, [((1088,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((1088,), torch.float32), ((1088,), torch.float32)]),
    (Multiply18, [((1120,), torch.float32)]),
    (Multiply52, [((1120,), torch.float32)]),
    (Multiply1, [((1, 1120, 14, 14), torch.float32), ((1120, 1, 1), torch.float32)]),
    (Multiply1, [((1120,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((1120,), torch.float32), ((1120,), torch.float32)]),
    (Multiply18, [((1152,), torch.float32)]),
    (Multiply53, [((1152,), torch.float32)]),
    (Multiply1, [((1, 1152, 14, 14), torch.float32), ((1152, 1, 1), torch.float32)]),
    (Multiply1, [((1152,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((1152,), torch.float32), ((1152,), torch.float32)]),
    (Multiply18, [((1184,), torch.float32)]),
    (Multiply54, [((1184,), torch.float32)]),
    (Multiply1, [((1, 1184, 14, 14), torch.float32), ((1184, 1, 1), torch.float32)]),
    (Multiply1, [((1184,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((1184,), torch.float32), ((1184,), torch.float32)]),
    (Multiply18, [((1216,), torch.float32)]),
    (Multiply55, [((1216,), torch.float32)]),
    (Multiply1, [((1, 1216, 14, 14), torch.float32), ((1216, 1, 1), torch.float32)]),
    (Multiply1, [((1216,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((1216,), torch.float32), ((1216,), torch.float32)]),
    (Multiply18, [((1248,), torch.float32)]),
    (Multiply56, [((1248,), torch.float32)]),
    (Multiply1, [((1, 1248, 14, 14), torch.float32), ((1248, 1, 1), torch.float32)]),
    (Multiply1, [((1248,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((1248,), torch.float32), ((1248,), torch.float32)]),
    (Multiply18, [((1280,), torch.float32)]),
    (Multiply57, [((1280,), torch.float32)]),
    (Multiply1, [((1, 1280, 14, 14), torch.float32), ((1280, 1, 1), torch.float32)]),
    (Multiply1, [((1280,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((1280,), torch.float32), ((1280,), torch.float32)]),
    (Multiply18, [((1312,), torch.float32)]),
    (Multiply58, [((1312,), torch.float32)]),
    (Multiply1, [((1, 1312, 14, 14), torch.float32), ((1312, 1, 1), torch.float32)]),
    (Multiply1, [((1312,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((1312,), torch.float32), ((1312,), torch.float32)]),
    (Multiply18, [((1344,), torch.float32)]),
    (Multiply59, [((1344,), torch.float32)]),
    (Multiply1, [((1, 1344, 14, 14), torch.float32), ((1344, 1, 1), torch.float32)]),
    (Multiply1, [((1344,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((1344,), torch.float32), ((1344,), torch.float32)]),
    (Multiply18, [((1376,), torch.float32)]),
    (Multiply60, [((1376,), torch.float32)]),
    (Multiply1, [((1, 1376, 14, 14), torch.float32), ((1376, 1, 1), torch.float32)]),
    (Multiply1, [((1376,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((1376,), torch.float32), ((1376,), torch.float32)]),
    (Multiply18, [((1408,), torch.float32)]),
    (Multiply61, [((1408,), torch.float32)]),
    (Multiply1, [((1, 1408, 14, 14), torch.float32), ((1408, 1, 1), torch.float32)]),
    (Multiply1, [((1408,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((1408,), torch.float32), ((1408,), torch.float32)]),
    (Multiply18, [((1440,), torch.float32)]),
    (Multiply62, [((1440,), torch.float32)]),
    (Multiply1, [((1, 1440, 14, 14), torch.float32), ((1440, 1, 1), torch.float32)]),
    (Multiply1, [((1440,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((1440,), torch.float32), ((1440,), torch.float32)]),
    (Multiply18, [((1472,), torch.float32)]),
    (Multiply63, [((1472,), torch.float32)]),
    (Multiply1, [((1, 1472, 14, 14), torch.float32), ((1472, 1, 1), torch.float32)]),
    (Multiply1, [((1472,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((1472,), torch.float32), ((1472,), torch.float32)]),
    (Multiply18, [((1504,), torch.float32)]),
    (Multiply64, [((1504,), torch.float32)]),
    (Multiply1, [((1, 1504, 14, 14), torch.float32), ((1504, 1, 1), torch.float32)]),
    (Multiply1, [((1504,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((1504,), torch.float32), ((1504,), torch.float32)]),
    (Multiply18, [((1536,), torch.float32)]),
    (Multiply65, [((1536,), torch.float32)]),
    (Multiply1, [((1, 1536, 14, 14), torch.float32), ((1536, 1, 1), torch.float32)]),
    (Multiply1, [((1536,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((1536,), torch.float32), ((1536,), torch.float32)]),
    (Multiply18, [((1568,), torch.float32)]),
    (Multiply66, [((1568,), torch.float32)]),
    (Multiply1, [((1, 1568, 14, 14), torch.float32), ((1568, 1, 1), torch.float32)]),
    (Multiply1, [((1568,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((1568,), torch.float32), ((1568,), torch.float32)]),
    (Multiply18, [((1600,), torch.float32)]),
    (Multiply67, [((1600,), torch.float32)]),
    (Multiply1, [((1, 1600, 14, 14), torch.float32), ((1600, 1, 1), torch.float32)]),
    (Multiply1, [((1600,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((1600,), torch.float32), ((1600,), torch.float32)]),
    (Multiply18, [((1632,), torch.float32)]),
    (Multiply68, [((1632,), torch.float32)]),
    (Multiply1, [((1, 1632, 14, 14), torch.float32), ((1632, 1, 1), torch.float32)]),
    (Multiply1, [((1632,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((1632,), torch.float32), ((1632,), torch.float32)]),
    (Multiply18, [((1664,), torch.float32)]),
    (Multiply69, [((1664,), torch.float32)]),
    (Multiply1, [((1, 1664, 14, 14), torch.float32), ((1664, 1, 1), torch.float32)]),
    (Multiply1, [((1664,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((1664,), torch.float32), ((1664,), torch.float32)]),
    (Multiply18, [((1696,), torch.float32)]),
    (Multiply70, [((1696,), torch.float32)]),
    (Multiply1, [((1, 1696, 14, 14), torch.float32), ((1696, 1, 1), torch.float32)]),
    (Multiply1, [((1696,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((1696,), torch.float32), ((1696,), torch.float32)]),
    (Multiply18, [((1728,), torch.float32)]),
    (Multiply71, [((1728,), torch.float32)]),
    (Multiply1, [((1, 1728, 14, 14), torch.float32), ((1728, 1, 1), torch.float32)]),
    (Multiply1, [((1728,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((1728,), torch.float32), ((1728,), torch.float32)]),
    (Multiply18, [((1760,), torch.float32)]),
    (Multiply72, [((1760,), torch.float32)]),
    (Multiply1, [((1, 1760, 14, 14), torch.float32), ((1760, 1, 1), torch.float32)]),
    (Multiply1, [((1760,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((1760,), torch.float32), ((1760,), torch.float32)]),
    (Multiply18, [((1792,), torch.float32)]),
    (Multiply73, [((1792,), torch.float32)]),
    (Multiply1, [((1, 1792, 14, 14), torch.float32), ((1792, 1, 1), torch.float32)]),
    (Multiply1, [((1792,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((1792,), torch.float32), ((1792,), torch.float32)]),
    (Multiply1, [((1, 896, 7, 7), torch.float32), ((896, 1, 1), torch.float32)]),
    (Multiply1, [((1, 128, 7, 7), torch.float32), ((128, 1, 1), torch.float32)]),
    (Multiply1, [((1, 928, 7, 7), torch.float32), ((928, 1, 1), torch.float32)]),
    (Multiply1, [((1, 960, 7, 7), torch.float32), ((960, 1, 1), torch.float32)]),
    (Multiply1, [((1, 992, 7, 7), torch.float32), ((992, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1024, 7, 7), torch.float32), ((1024, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1056, 7, 7), torch.float32), ((1056, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1088, 7, 7), torch.float32), ((1088, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1120, 7, 7), torch.float32), ((1120, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1152, 7, 7), torch.float32), ((1152, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1184, 7, 7), torch.float32), ((1184, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1216, 7, 7), torch.float32), ((1216, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1248, 7, 7), torch.float32), ((1248, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1280, 7, 7), torch.float32), ((1280, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1312, 7, 7), torch.float32), ((1312, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1344, 7, 7), torch.float32), ((1344, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1376, 7, 7), torch.float32), ((1376, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1408, 7, 7), torch.float32), ((1408, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1440, 7, 7), torch.float32), ((1440, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1472, 7, 7), torch.float32), ((1472, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1504, 7, 7), torch.float32), ((1504, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1536, 7, 7), torch.float32), ((1536, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1568, 7, 7), torch.float32), ((1568, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1600, 7, 7), torch.float32), ((1600, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1632, 7, 7), torch.float32), ((1632, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1664, 7, 7), torch.float32), ((1664, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1696, 7, 7), torch.float32), ((1696, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1728, 7, 7), torch.float32), ((1728, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1760, 7, 7), torch.float32), ((1760, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1792, 7, 7), torch.float32), ((1792, 1, 1), torch.float32)]),
    (Multiply18, [((1824,), torch.float32)]),
    (Multiply74, [((1824,), torch.float32)]),
    (Multiply1, [((1, 1824, 7, 7), torch.float32), ((1824, 1, 1), torch.float32)]),
    (Multiply1, [((1824,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((1824,), torch.float32), ((1824,), torch.float32)]),
    (Multiply18, [((1856,), torch.float32)]),
    (Multiply75, [((1856,), torch.float32)]),
    (Multiply1, [((1, 1856, 7, 7), torch.float32), ((1856, 1, 1), torch.float32)]),
    (Multiply1, [((1856,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((1856,), torch.float32), ((1856,), torch.float32)]),
    (Multiply18, [((1888,), torch.float32)]),
    (Multiply76, [((1888,), torch.float32)]),
    (Multiply1, [((1, 1888, 7, 7), torch.float32), ((1888, 1, 1), torch.float32)]),
    (Multiply1, [((1888,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((1888,), torch.float32), ((1888,), torch.float32)]),
    (Multiply18, [((1920,), torch.float32)]),
    (Multiply77, [((1920,), torch.float32)]),
    (Multiply1, [((1, 1920, 7, 7), torch.float32), ((1920, 1, 1), torch.float32)]),
    (Multiply1, [((1920,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((1920,), torch.float32), ((1920,), torch.float32)]),
    (Multiply1, [((1, 512, 7, 7), torch.float32), ((512, 1, 1), torch.float32)]),
    (Multiply1, [((1, 544, 7, 7), torch.float32), ((544, 1, 1), torch.float32)]),
    (Multiply1, [((1, 576, 7, 7), torch.float32), ((576, 1, 1), torch.float32)]),
    (Multiply1, [((1, 608, 7, 7), torch.float32), ((608, 1, 1), torch.float32)]),
    (Multiply1, [((1, 640, 7, 7), torch.float32), ((640, 1, 1), torch.float32)]),
    (Multiply1, [((1, 672, 7, 7), torch.float32), ((672, 1, 1), torch.float32)]),
    (Multiply1, [((1, 704, 7, 7), torch.float32), ((704, 1, 1), torch.float32)]),
    (Multiply1, [((1, 736, 7, 7), torch.float32), ((736, 1, 1), torch.float32)]),
    (Multiply1, [((1, 768, 7, 7), torch.float32), ((768, 1, 1), torch.float32)]),
    (Multiply1, [((1, 800, 7, 7), torch.float32), ((800, 1, 1), torch.float32)]),
    (Multiply1, [((1, 832, 7, 7), torch.float32), ((832, 1, 1), torch.float32)]),
    (Multiply1, [((1, 864, 7, 7), torch.float32), ((864, 1, 1), torch.float32)]),
    (Multiply1, [((1, 96, 112, 112), torch.float32), ((96, 1, 1), torch.float32)]),
    (Multiply18, [((144,), torch.float32)]),
    (Multiply78, [((144,), torch.float32)]),
    (Multiply1, [((1, 144, 56, 56), torch.float32), ((144, 1, 1), torch.float32)]),
    (Multiply1, [((144,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((144,), torch.float32), ((144,), torch.float32)]),
    (Multiply18, [((240,), torch.float32)]),
    (Multiply79, [((240,), torch.float32)]),
    (Multiply1, [((1, 240, 56, 56), torch.float32), ((240, 1, 1), torch.float32)]),
    (Multiply1, [((240,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((240,), torch.float32), ((240,), torch.float32)]),
    (Multiply1, [((1, 288, 56, 56), torch.float32), ((288, 1, 1), torch.float32)]),
    (Multiply18, [((336,), torch.float32)]),
    (Multiply80, [((336,), torch.float32)]),
    (Multiply1, [((1, 336, 56, 56), torch.float32), ((336, 1, 1), torch.float32)]),
    (Multiply1, [((336,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((336,), torch.float32), ((336,), torch.float32)]),
    (Multiply1, [((1, 384, 56, 56), torch.float32), ((384, 1, 1), torch.float32)]),
    (Multiply1, [((1, 240, 28, 28), torch.float32), ((240, 1, 1), torch.float32)]),
    (Multiply1, [((1, 336, 28, 28), torch.float32), ((336, 1, 1), torch.float32)]),
    (Multiply18, [((432,), torch.float32)]),
    (Multiply81, [((432,), torch.float32)]),
    (Multiply1, [((1, 432, 28, 28), torch.float32), ((432, 1, 1), torch.float32)]),
    (Multiply1, [((432,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((432,), torch.float32), ((432,), torch.float32)]),
    (Multiply18, [((528,), torch.float32)]),
    (Multiply82, [((528,), torch.float32)]),
    (Multiply1, [((1, 528, 28, 28), torch.float32), ((528, 1, 1), torch.float32)]),
    (Multiply1, [((528,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((528,), torch.float32), ((528,), torch.float32)]),
    (Multiply1, [((1, 576, 28, 28), torch.float32), ((576, 1, 1), torch.float32)]),
    (Multiply18, [((624,), torch.float32)]),
    (Multiply83, [((624,), torch.float32)]),
    (Multiply1, [((1, 624, 28, 28), torch.float32), ((624, 1, 1), torch.float32)]),
    (Multiply1, [((624,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((624,), torch.float32), ((624,), torch.float32)]),
    (Multiply1, [((1, 672, 28, 28), torch.float32), ((672, 1, 1), torch.float32)]),
    (Multiply18, [((720,), torch.float32)]),
    (Multiply84, [((720,), torch.float32)]),
    (Multiply1, [((1, 720, 28, 28), torch.float32), ((720, 1, 1), torch.float32)]),
    (Multiply1, [((720,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((720,), torch.float32), ((720,), torch.float32)]),
    (Multiply1, [((1, 768, 28, 28), torch.float32), ((768, 1, 1), torch.float32)]),
    (Multiply1, [((1, 192, 14, 14), torch.float32), ((192, 1, 1), torch.float32)]),
    (Multiply1, [((1, 432, 14, 14), torch.float32), ((432, 1, 1), torch.float32)]),
    (Multiply1, [((1, 528, 14, 14), torch.float32), ((528, 1, 1), torch.float32)]),
    (Multiply1, [((1, 624, 14, 14), torch.float32), ((624, 1, 1), torch.float32)]),
    (Multiply1, [((1, 720, 14, 14), torch.float32), ((720, 1, 1), torch.float32)]),
    (Multiply18, [((816,), torch.float32)]),
    (Multiply85, [((816,), torch.float32)]),
    (Multiply1, [((1, 816, 14, 14), torch.float32), ((816, 1, 1), torch.float32)]),
    (Multiply1, [((816,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((816,), torch.float32), ((816,), torch.float32)]),
    (Multiply18, [((912,), torch.float32)]),
    (Multiply86, [((912,), torch.float32)]),
    (Multiply1, [((1, 912, 14, 14), torch.float32), ((912, 1, 1), torch.float32)]),
    (Multiply1, [((912,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((912,), torch.float32), ((912,), torch.float32)]),
    (Multiply18, [((1008,), torch.float32)]),
    (Multiply87, [((1008,), torch.float32)]),
    (Multiply1, [((1, 1008, 14, 14), torch.float32), ((1008, 1, 1), torch.float32)]),
    (Multiply1, [((1008,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((1008,), torch.float32), ((1008,), torch.float32)]),
    (Multiply18, [((1104,), torch.float32)]),
    (Multiply88, [((1104,), torch.float32)]),
    (Multiply1, [((1, 1104, 14, 14), torch.float32), ((1104, 1, 1), torch.float32)]),
    (Multiply1, [((1104,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((1104,), torch.float32), ((1104,), torch.float32)]),
    (Multiply18, [((1200,), torch.float32)]),
    (Multiply89, [((1200,), torch.float32)]),
    (Multiply1, [((1, 1200, 14, 14), torch.float32), ((1200, 1, 1), torch.float32)]),
    (Multiply1, [((1200,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((1200,), torch.float32), ((1200,), torch.float32)]),
    (Multiply18, [((1296,), torch.float32)]),
    (Multiply90, [((1296,), torch.float32)]),
    (Multiply1, [((1, 1296, 14, 14), torch.float32), ((1296, 1, 1), torch.float32)]),
    (Multiply1, [((1296,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((1296,), torch.float32), ((1296,), torch.float32)]),
    (Multiply18, [((1392,), torch.float32)]),
    (Multiply91, [((1392,), torch.float32)]),
    (Multiply1, [((1, 1392, 14, 14), torch.float32), ((1392, 1, 1), torch.float32)]),
    (Multiply1, [((1392,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((1392,), torch.float32), ((1392,), torch.float32)]),
    (Multiply18, [((1488,), torch.float32)]),
    (Multiply92, [((1488,), torch.float32)]),
    (Multiply1, [((1, 1488, 14, 14), torch.float32), ((1488, 1, 1), torch.float32)]),
    (Multiply1, [((1488,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((1488,), torch.float32), ((1488,), torch.float32)]),
    (Multiply18, [((1584,), torch.float32)]),
    (Multiply93, [((1584,), torch.float32)]),
    (Multiply1, [((1, 1584, 14, 14), torch.float32), ((1584, 1, 1), torch.float32)]),
    (Multiply1, [((1584,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((1584,), torch.float32), ((1584,), torch.float32)]),
    (Multiply18, [((1680,), torch.float32)]),
    (Multiply94, [((1680,), torch.float32)]),
    (Multiply1, [((1, 1680, 14, 14), torch.float32), ((1680, 1, 1), torch.float32)]),
    (Multiply1, [((1680,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((1680,), torch.float32), ((1680,), torch.float32)]),
    (Multiply18, [((1776,), torch.float32)]),
    (Multiply95, [((1776,), torch.float32)]),
    (Multiply1, [((1, 1776, 14, 14), torch.float32), ((1776, 1, 1), torch.float32)]),
    (Multiply1, [((1776,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((1776,), torch.float32), ((1776,), torch.float32)]),
    (Multiply1, [((1, 1824, 14, 14), torch.float32), ((1824, 1, 1), torch.float32)]),
    (Multiply18, [((1872,), torch.float32)]),
    (Multiply96, [((1872,), torch.float32)]),
    (Multiply1, [((1, 1872, 14, 14), torch.float32), ((1872, 1, 1), torch.float32)]),
    (Multiply1, [((1872,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((1872,), torch.float32), ((1872,), torch.float32)]),
    (Multiply1, [((1, 1920, 14, 14), torch.float32), ((1920, 1, 1), torch.float32)]),
    (Multiply18, [((1968,), torch.float32)]),
    (Multiply97, [((1968,), torch.float32)]),
    (Multiply1, [((1, 1968, 14, 14), torch.float32), ((1968, 1, 1), torch.float32)]),
    (Multiply1, [((1968,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((1968,), torch.float32), ((1968,), torch.float32)]),
    (Multiply18, [((2016,), torch.float32)]),
    (Multiply98, [((2016,), torch.float32)]),
    (Multiply1, [((1, 2016, 14, 14), torch.float32), ((2016, 1, 1), torch.float32)]),
    (Multiply1, [((2016,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((2016,), torch.float32), ((2016,), torch.float32)]),
    (Multiply18, [((2064,), torch.float32)]),
    (Multiply99, [((2064,), torch.float32)]),
    (Multiply1, [((1, 2064, 14, 14), torch.float32), ((2064, 1, 1), torch.float32)]),
    (Multiply1, [((2064,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((2064,), torch.float32), ((2064,), torch.float32)]),
    (Multiply18, [((2112,), torch.float32)]),
    (Multiply100, [((2112,), torch.float32)]),
    (Multiply1, [((1, 2112, 14, 14), torch.float32), ((2112, 1, 1), torch.float32)]),
    (Multiply1, [((2112,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((2112,), torch.float32), ((2112,), torch.float32)]),
    (Multiply1, [((1, 192, 7, 7), torch.float32), ((192, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1104, 7, 7), torch.float32), ((1104, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1200, 7, 7), torch.float32), ((1200, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1296, 7, 7), torch.float32), ((1296, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1392, 7, 7), torch.float32), ((1392, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1488, 7, 7), torch.float32), ((1488, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1584, 7, 7), torch.float32), ((1584, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1680, 7, 7), torch.float32), ((1680, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1776, 7, 7), torch.float32), ((1776, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1872, 7, 7), torch.float32), ((1872, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1968, 7, 7), torch.float32), ((1968, 1, 1), torch.float32)]),
    (Multiply1, [((1, 2016, 7, 7), torch.float32), ((2016, 1, 1), torch.float32)]),
    (Multiply1, [((1, 2064, 7, 7), torch.float32), ((2064, 1, 1), torch.float32)]),
    (Multiply1, [((1, 2112, 7, 7), torch.float32), ((2112, 1, 1), torch.float32)]),
    (Multiply18, [((2160,), torch.float32)]),
    (Multiply101, [((2160,), torch.float32)]),
    (Multiply1, [((1, 2160, 7, 7), torch.float32), ((2160, 1, 1), torch.float32)]),
    (Multiply1, [((2160,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((2160,), torch.float32), ((2160,), torch.float32)]),
    (Multiply18, [((2208,), torch.float32)]),
    (Multiply102, [((2208,), torch.float32)]),
    (Multiply1, [((1, 2208, 7, 7), torch.float32), ((2208, 1, 1), torch.float32)]),
    (Multiply1, [((2208,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((2208,), torch.float32), ((2208,), torch.float32)]),
    (Multiply18, [((16,), torch.float32)]),
    (Multiply103, [((16,), torch.float32)]),
    (Multiply1, [((1, 16, 224, 224), torch.float32), ((16, 1, 1), torch.float32)]),
    (Multiply1, [((16,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((16,), torch.float32), ((16,), torch.float32)]),
    (Multiply18, [((32,), torch.float32)]),
    (Multiply104, [((32,), torch.float32)]),
    (Multiply1, [((1, 32, 112, 112), torch.float32), ((32, 1, 1), torch.float32)]),
    (Multiply1, [((32,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((32,), torch.float32), ((32,), torch.float32)]),
    (Multiply1, [((1, 64, 28, 28), torch.float32), ((64, 1, 1), torch.float32)]),
    (Multiply1, [((1, 256, 7, 7), torch.float32), ((256, 1, 1), torch.float32)]),
    (Multiply1, [((1, 32, 56, 56), torch.float32), ((32, 1, 1), torch.float32)]),
    (Multiply1, [((1, 32, 28, 28), torch.float32), ((32, 1, 1), torch.float32)]),
    (Multiply1, [((1, 64, 14, 14), torch.float32), ((64, 1, 1), torch.float32)]),
    (Multiply1, [((1, 256, 112, 112), torch.float32), ((256, 1, 1), torch.float32)]),
    (Multiply1, [((1, 512, 56, 56), torch.float32), ((512, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1024, 28, 28), torch.float32), ((1024, 1, 1), torch.float32)]),
    (Multiply18, [((2048,), torch.float32)]),
    (Multiply105, [((2048,), torch.float32)]),
    (Multiply1, [((1, 2048, 14, 14), torch.float32), ((2048, 1, 1), torch.float32)]),
    (Multiply1, [((2048,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((2048,), torch.float32), ((2048,), torch.float32)]),
    (Multiply1, [((1, 2048, 7, 7), torch.float32), ((2048, 1, 1), torch.float32)]),
    (Multiply1, [((1, 128, 112, 112), torch.float32), ((128, 1, 1), torch.float32)]),
    (Multiply18, [((48,), torch.float32)]),
    (Multiply106, [((48,), torch.float32)]),
    (Multiply1, [((1, 48, 112, 112), torch.float32), ((48, 1, 1), torch.float32)]),
    (Multiply1, [((48,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((48,), torch.float32), ((48,), torch.float32)]),
    (Multiply1, [((1, 48, 112, 112), torch.float32), ((1, 48, 112, 112), torch.float32)]),
    (Multiply1, [((1, 12, 1, 1), torch.float32), ((1, 12, 1, 1), torch.float32)]),
    (Multiply1, [((1, 48, 1, 1), torch.float32), ((1, 48, 112, 112), torch.float32)]),
    (Multiply18, [((24,), torch.float32)]),
    (Multiply107, [((24,), torch.float32)]),
    (Multiply1, [((1, 24, 112, 112), torch.float32), ((24, 1, 1), torch.float32)]),
    (Multiply1, [((24,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((24,), torch.float32), ((24,), torch.float32)]),
    (Multiply1, [((1, 24, 112, 112), torch.float32), ((1, 24, 112, 112), torch.float32)]),
    (Multiply1, [((1, 6, 1, 1), torch.float32), ((1, 6, 1, 1), torch.float32)]),
    (Multiply1, [((1, 24, 1, 1), torch.float32), ((1, 24, 112, 112), torch.float32)]),
    (Multiply1, [((1, 144, 112, 112), torch.float32), ((144, 1, 1), torch.float32)]),
    (Multiply1, [((1, 144, 112, 112), torch.float32), ((1, 144, 112, 112), torch.float32)]),
    (Multiply1, [((1, 144, 56, 56), torch.float32), ((1, 144, 56, 56), torch.float32)]),
    (Multiply1, [((1, 144, 1, 1), torch.float32), ((1, 144, 56, 56), torch.float32)]),
    (Multiply1, [((1, 192, 56, 56), torch.float32), ((1, 192, 56, 56), torch.float32)]),
    (Multiply1, [((1, 8, 1, 1), torch.float32), ((1, 8, 1, 1), torch.float32)]),
    (Multiply1, [((1, 192, 1, 1), torch.float32), ((1, 192, 56, 56), torch.float32)]),
    (Multiply1, [((1, 192, 28, 28), torch.float32), ((1, 192, 28, 28), torch.float32)]),
    (Multiply1, [((1, 192, 1, 1), torch.float32), ((1, 192, 28, 28), torch.float32)]),
    (Multiply18, [((56,), torch.float32)]),
    (Multiply108, [((56,), torch.float32)]),
    (Multiply1, [((1, 56, 28, 28), torch.float32), ((56, 1, 1), torch.float32)]),
    (Multiply1, [((56,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((56,), torch.float32), ((56,), torch.float32)]),
    (Multiply1, [((1, 336, 28, 28), torch.float32), ((1, 336, 28, 28), torch.float32)]),
    (Multiply1, [((1, 14, 1, 1), torch.float32), ((1, 14, 1, 1), torch.float32)]),
    (Multiply1, [((1, 336, 1, 1), torch.float32), ((1, 336, 28, 28), torch.float32)]),
    (Multiply1, [((1, 336, 14, 14), torch.float32), ((336, 1, 1), torch.float32)]),
    (Multiply1, [((1, 336, 14, 14), torch.float32), ((1, 336, 14, 14), torch.float32)]),
    (Multiply1, [((1, 336, 1, 1), torch.float32), ((1, 336, 14, 14), torch.float32)]),
    (Multiply18, [((112,), torch.float32)]),
    (Multiply109, [((112,), torch.float32)]),
    (Multiply1, [((1, 112, 14, 14), torch.float32), ((112, 1, 1), torch.float32)]),
    (Multiply1, [((112,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((112,), torch.float32), ((112,), torch.float32)]),
    (Multiply1, [((1, 672, 14, 14), torch.float32), ((1, 672, 14, 14), torch.float32)]),
    (Multiply1, [((1, 28, 1, 1), torch.float32), ((1, 28, 1, 1), torch.float32)]),
    (Multiply1, [((1, 672, 1, 1), torch.float32), ((1, 672, 14, 14), torch.float32)]),
    (Multiply1, [((1, 160, 14, 14), torch.float32), ((160, 1, 1), torch.float32)]),
    (Multiply1, [((1, 960, 14, 14), torch.float32), ((1, 960, 14, 14), torch.float32)]),
    (Multiply1, [((1, 40, 1, 1), torch.float32), ((1, 40, 1, 1), torch.float32)]),
    (Multiply1, [((1, 960, 1, 1), torch.float32), ((1, 960, 14, 14), torch.float32)]),
    (Multiply1, [((1, 960, 7, 7), torch.float32), ((1, 960, 7, 7), torch.float32)]),
    (Multiply1, [((1, 960, 1, 1), torch.float32), ((1, 960, 7, 7), torch.float32)]),
    (Multiply18, [((272,), torch.float32)]),
    (Multiply110, [((272,), torch.float32)]),
    (Multiply1, [((1, 272, 7, 7), torch.float32), ((272, 1, 1), torch.float32)]),
    (Multiply1, [((272,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((272,), torch.float32), ((272,), torch.float32)]),
    (Multiply1, [((1, 1632, 7, 7), torch.float32), ((1, 1632, 7, 7), torch.float32)]),
    (Multiply1, [((1, 68, 1, 1), torch.float32), ((1, 68, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1632, 1, 1), torch.float32), ((1, 1632, 7, 7), torch.float32)]),
    (Multiply1, [((1, 448, 7, 7), torch.float32), ((448, 1, 1), torch.float32)]),
    (Multiply18, [((2688,), torch.float32)]),
    (Multiply111, [((2688,), torch.float32)]),
    (Multiply1, [((1, 2688, 7, 7), torch.float32), ((2688, 1, 1), torch.float32)]),
    (Multiply1, [((2688,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((2688,), torch.float32), ((2688,), torch.float32)]),
    (Multiply1, [((1, 2688, 7, 7), torch.float32), ((1, 2688, 7, 7), torch.float32)]),
    (Multiply1, [((1, 112, 1, 1), torch.float32), ((1, 112, 1, 1), torch.float32)]),
    (Multiply1, [((1, 2688, 1, 1), torch.float32), ((1, 2688, 7, 7), torch.float32)]),
    (Multiply1, [((1, 1792, 7, 7), torch.float32), ((1, 1792, 7, 7), torch.float32)]),
    (Multiply1, [((1, 48, 160, 160), torch.float32), ((48, 1, 1), torch.float32)]),
    (Multiply1, [((1, 48, 160, 160), torch.float32), ((1, 48, 160, 160), torch.float32)]),
    (Multiply1, [((1, 48, 160, 160), torch.float32), ((1, 48, 1, 1), torch.float32)]),
    (Multiply1, [((1, 24, 160, 160), torch.float32), ((24, 1, 1), torch.float32)]),
    (Multiply1, [((1, 24, 160, 160), torch.float32), ((1, 24, 160, 160), torch.float32)]),
    (Multiply1, [((1, 24, 160, 160), torch.float32), ((1, 24, 1, 1), torch.float32)]),
    (Multiply1, [((1, 144, 160, 160), torch.float32), ((144, 1, 1), torch.float32)]),
    (Multiply1, [((1, 144, 160, 160), torch.float32), ((1, 144, 160, 160), torch.float32)]),
    (Multiply1, [((1, 144, 80, 80), torch.float32), ((144, 1, 1), torch.float32)]),
    (Multiply1, [((1, 144, 80, 80), torch.float32), ((1, 144, 80, 80), torch.float32)]),
    (Multiply1, [((1, 144, 80, 80), torch.float32), ((1, 144, 1, 1), torch.float32)]),
    (Multiply1, [((1, 32, 80, 80), torch.float32), ((32, 1, 1), torch.float32)]),
    (Multiply1, [((1, 192, 80, 80), torch.float32), ((192, 1, 1), torch.float32)]),
    (Multiply1, [((1, 192, 80, 80), torch.float32), ((1, 192, 80, 80), torch.float32)]),
    (Multiply1, [((1, 192, 80, 80), torch.float32), ((1, 192, 1, 1), torch.float32)]),
    (Multiply1, [((1, 192, 40, 40), torch.float32), ((192, 1, 1), torch.float32)]),
    (Multiply1, [((1, 192, 40, 40), torch.float32), ((1, 192, 40, 40), torch.float32)]),
    (Multiply1, [((1, 192, 40, 40), torch.float32), ((1, 192, 1, 1), torch.float32)]),
    (Multiply1, [((1, 56, 40, 40), torch.float32), ((56, 1, 1), torch.float32)]),
    (Multiply1, [((1, 336, 40, 40), torch.float32), ((336, 1, 1), torch.float32)]),
    (Multiply1, [((1, 336, 40, 40), torch.float32), ((1, 336, 40, 40), torch.float32)]),
    (Multiply1, [((1, 336, 40, 40), torch.float32), ((1, 336, 1, 1), torch.float32)]),
    (Multiply1, [((1, 336, 20, 20), torch.float32), ((336, 1, 1), torch.float32)]),
    (Multiply1, [((1, 336, 20, 20), torch.float32), ((1, 336, 20, 20), torch.float32)]),
    (Multiply1, [((1, 336, 20, 20), torch.float32), ((1, 336, 1, 1), torch.float32)]),
    (Multiply1, [((1, 112, 20, 20), torch.float32), ((112, 1, 1), torch.float32)]),
    (Multiply1, [((1, 672, 20, 20), torch.float32), ((672, 1, 1), torch.float32)]),
    (Multiply1, [((1, 672, 20, 20), torch.float32), ((1, 672, 20, 20), torch.float32)]),
    (Multiply1, [((1, 672, 20, 20), torch.float32), ((1, 672, 1, 1), torch.float32)]),
    (Multiply1, [((1, 160, 20, 20), torch.float32), ((160, 1, 1), torch.float32)]),
    (Multiply1, [((1, 960, 20, 20), torch.float32), ((960, 1, 1), torch.float32)]),
    (Multiply1, [((1, 960, 20, 20), torch.float32), ((1, 960, 20, 20), torch.float32)]),
    (Multiply1, [((1, 960, 20, 20), torch.float32), ((1, 960, 1, 1), torch.float32)]),
    (Multiply1, [((1, 960, 10, 10), torch.float32), ((960, 1, 1), torch.float32)]),
    (Multiply1, [((1, 960, 10, 10), torch.float32), ((1, 960, 10, 10), torch.float32)]),
    (Multiply1, [((1, 960, 10, 10), torch.float32), ((1, 960, 1, 1), torch.float32)]),
    (Multiply1, [((1, 272, 10, 10), torch.float32), ((272, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1632, 10, 10), torch.float32), ((1632, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1632, 10, 10), torch.float32), ((1, 1632, 10, 10), torch.float32)]),
    (Multiply1, [((1, 1632, 10, 10), torch.float32), ((1, 1632, 1, 1), torch.float32)]),
    (Multiply1, [((1, 448, 10, 10), torch.float32), ((448, 1, 1), torch.float32)]),
    (Multiply1, [((1, 2688, 10, 10), torch.float32), ((2688, 1, 1), torch.float32)]),
    (Multiply1, [((1, 2688, 10, 10), torch.float32), ((1, 2688, 10, 10), torch.float32)]),
    (Multiply1, [((1, 2688, 10, 10), torch.float32), ((1, 2688, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1792, 10, 10), torch.float32), ((1792, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1792, 10, 10), torch.float32), ((1, 1792, 10, 10), torch.float32)]),
    (Multiply1, [((1, 32, 112, 112), torch.float32), ((1, 32, 112, 112), torch.float32)]),
    (Multiply1, [((1, 32, 1, 1), torch.float32), ((1, 32, 112, 112), torch.float32)]),
    (Multiply1, [((1, 16, 112, 112), torch.float32), ((16, 1, 1), torch.float32)]),
    (Multiply1, [((1, 96, 112, 112), torch.float32), ((1, 96, 112, 112), torch.float32)]),
    (Multiply1, [((1, 96, 56, 56), torch.float32), ((1, 96, 56, 56), torch.float32)]),
    (Multiply1, [((1, 4, 1, 1), torch.float32), ((1, 4, 1, 1), torch.float32)]),
    (Multiply1, [((1, 96, 1, 1), torch.float32), ((1, 96, 56, 56), torch.float32)]),
    (Multiply1, [((1, 24, 56, 56), torch.float32), ((24, 1, 1), torch.float32)]),
    (Multiply1, [((1, 144, 28, 28), torch.float32), ((144, 1, 1), torch.float32)]),
    (Multiply1, [((1, 144, 28, 28), torch.float32), ((1, 144, 28, 28), torch.float32)]),
    (Multiply1, [((1, 144, 1, 1), torch.float32), ((1, 144, 28, 28), torch.float32)]),
    (Multiply18, [((40,), torch.float32)]),
    (Multiply112, [((40,), torch.float32)]),
    (Multiply1, [((1, 40, 28, 28), torch.float32), ((40, 1, 1), torch.float32)]),
    (Multiply1, [((40,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((40,), torch.float32), ((40,), torch.float32)]),
    (Multiply1, [((1, 240, 28, 28), torch.float32), ((1, 240, 28, 28), torch.float32)]),
    (Multiply1, [((1, 10, 1, 1), torch.float32), ((1, 10, 1, 1), torch.float32)]),
    (Multiply1, [((1, 240, 1, 1), torch.float32), ((1, 240, 28, 28), torch.float32)]),
    (Multiply1, [((1, 240, 14, 14), torch.float32), ((240, 1, 1), torch.float32)]),
    (Multiply1, [((1, 240, 14, 14), torch.float32), ((1, 240, 14, 14), torch.float32)]),
    (Multiply1, [((1, 240, 1, 1), torch.float32), ((1, 240, 14, 14), torch.float32)]),
    (Multiply18, [((80,), torch.float32)]),
    (Multiply113, [((80,), torch.float32)]),
    (Multiply1, [((1, 80, 14, 14), torch.float32), ((80, 1, 1), torch.float32)]),
    (Multiply1, [((80,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((80,), torch.float32), ((80,), torch.float32)]),
    (Multiply1, [((1, 480, 14, 14), torch.float32), ((1, 480, 14, 14), torch.float32)]),
    (Multiply1, [((1, 20, 1, 1), torch.float32), ((1, 20, 1, 1), torch.float32)]),
    (Multiply1, [((1, 480, 1, 1), torch.float32), ((1, 480, 14, 14), torch.float32)]),
    (Multiply1, [((1, 672, 7, 7), torch.float32), ((1, 672, 7, 7), torch.float32)]),
    (Multiply1, [((1, 672, 1, 1), torch.float32), ((1, 672, 7, 7), torch.float32)]),
    (Multiply1, [((1, 1152, 7, 7), torch.float32), ((1, 1152, 7, 7), torch.float32)]),
    (Multiply1, [((1, 48, 1, 1), torch.float32), ((1, 48, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1152, 1, 1), torch.float32), ((1, 1152, 7, 7), torch.float32)]),
    (Multiply1, [((1, 320, 7, 7), torch.float32), ((320, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1280, 7, 7), torch.float32), ((1, 1280, 7, 7), torch.float32)]),
    (Multiply1, [((1, 32, 112, 112), torch.float32), ((1, 32, 1, 1), torch.float32)]),
    (Multiply1, [((1, 96, 56, 56), torch.float32), ((1, 96, 1, 1), torch.float32)]),
    (Multiply1, [((1, 144, 56, 56), torch.float32), ((1, 144, 1, 1), torch.float32)]),
    (Multiply1, [((1, 144, 28, 28), torch.float32), ((1, 144, 1, 1), torch.float32)]),
    (Multiply1, [((1, 240, 28, 28), torch.float32), ((1, 240, 1, 1), torch.float32)]),
    (Multiply1, [((1, 240, 14, 14), torch.float32), ((1, 240, 1, 1), torch.float32)]),
    (Multiply1, [((1, 480, 14, 14), torch.float32), ((1, 480, 1, 1), torch.float32)]),
    (Multiply1, [((1, 672, 14, 14), torch.float32), ((1, 672, 1, 1), torch.float32)]),
    (Multiply1, [((1, 672, 7, 7), torch.float32), ((1, 672, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1152, 7, 7), torch.float32), ((1, 1152, 1, 1), torch.float32)]),
    (Multiply1, [((1, 256, 64, 64), torch.float32), ((256, 1, 1), torch.float32)]),
    (Multiply1, [((1, 256, 16, 16), torch.float32), ((256, 1, 1), torch.float32)]),
    (Multiply1, [((1, 256, 8, 8), torch.float32), ((256, 1, 1), torch.float32)]),
    (Multiply18, [((8,), torch.float32)]),
    (Multiply114, [((8,), torch.float32)]),
    (Multiply1, [((1, 8, 112, 112), torch.float32), ((8, 1, 1), torch.float32)]),
    (Multiply1, [((8,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((8,), torch.float32), ((8,), torch.float32)]),
    (Multiply1, [((1, 48, 56, 56), torch.float32), ((48, 1, 1), torch.float32)]),
    (Multiply18, [((12,), torch.float32)]),
    (Multiply115, [((12,), torch.float32)]),
    (Multiply1, [((1, 12, 56, 56), torch.float32), ((12, 1, 1), torch.float32)]),
    (Multiply1, [((12,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((12,), torch.float32), ((12,), torch.float32)]),
    (Multiply1, [((1, 16, 56, 56), torch.float32), ((16, 1, 1), torch.float32)]),
    (Multiply18, [((36,), torch.float32)]),
    (Multiply116, [((36,), torch.float32)]),
    (Multiply1, [((1, 36, 56, 56), torch.float32), ((36, 1, 1), torch.float32)]),
    (Multiply1, [((36,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((36,), torch.float32), ((36,), torch.float32)]),
    (Multiply18, [((72,), torch.float32)]),
    (Multiply117, [((72,), torch.float32)]),
    (Multiply1, [((1, 72, 28, 28), torch.float32), ((72, 1, 1), torch.float32)]),
    (Multiply1, [((72,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((72,), torch.float32), ((72,), torch.float32)]),
    (Multiply0, [((1, 72, 1, 1), torch.float32)]),
    (Multiply1, [((1, 72, 28, 28), torch.float32), ((1, 72, 1, 1), torch.float32)]),
    (Multiply18, [((20,), torch.float32)]),
    (Multiply118, [((20,), torch.float32)]),
    (Multiply1, [((1, 20, 28, 28), torch.float32), ((20, 1, 1), torch.float32)]),
    (Multiply1, [((20,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((20,), torch.float32), ((20,), torch.float32)]),
    (Multiply1, [((1, 24, 28, 28), torch.float32), ((24, 1, 1), torch.float32)]),
    (Multiply18, [((60,), torch.float32)]),
    (Multiply119, [((60,), torch.float32)]),
    (Multiply1, [((1, 60, 28, 28), torch.float32), ((60, 1, 1), torch.float32)]),
    (Multiply1, [((60,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((60,), torch.float32), ((60,), torch.float32)]),
    (Multiply0, [((1, 120, 1, 1), torch.float32)]),
    (Multiply1, [((1, 120, 28, 28), torch.float32), ((1, 120, 1, 1), torch.float32)]),
    (Multiply18, [((120,), torch.float32)]),
    (Multiply120, [((120,), torch.float32)]),
    (Multiply1, [((1, 120, 28, 28), torch.float32), ((120, 1, 1), torch.float32)]),
    (Multiply1, [((120,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((120,), torch.float32), ((120,), torch.float32)]),
    (Multiply1, [((1, 40, 14, 14), torch.float32), ((40, 1, 1), torch.float32)]),
    (Multiply18, [((100,), torch.float32)]),
    (Multiply121, [((100,), torch.float32)]),
    (Multiply1, [((1, 100, 14, 14), torch.float32), ((100, 1, 1), torch.float32)]),
    (Multiply1, [((100,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((100,), torch.float32), ((100,), torch.float32)]),
    (Multiply18, [((92,), torch.float32)]),
    (Multiply122, [((92,), torch.float32)]),
    (Multiply1, [((1, 92, 14, 14), torch.float32), ((92, 1, 1), torch.float32)]),
    (Multiply1, [((92,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((92,), torch.float32), ((92,), torch.float32)]),
    (Multiply0, [((1, 480, 1, 1), torch.float32)]),
    (Multiply1, [((1, 56, 14, 14), torch.float32), ((56, 1, 1), torch.float32)]),
    (Multiply0, [((1, 672, 1, 1), torch.float32)]),
    (Multiply1, [((1, 80, 7, 7), torch.float32), ((80, 1, 1), torch.float32)]),
    (Multiply1, [((1, 112, 7, 7), torch.float32), ((112, 1, 1), torch.float32)]),
    (Multiply1, [((1, 160, 7, 7), torch.float32), ((160, 1, 1), torch.float32)]),
    (Multiply1, [((1, 480, 7, 7), torch.float32), ((480, 1, 1), torch.float32)]),
    (Multiply0, [((1, 960, 1, 1), torch.float32)]),
    (Multiply1, [((1, 960, 7, 7), torch.float32), ((1, 960, 1, 1), torch.float32)]),
    (Multiply0, [((1, 1, 224, 224), torch.float32)]),
    (Multiply1, [((1, 176, 28, 28), torch.float32), ((176, 1, 1), torch.float32)]),
    (Multiply1, [((1, 96, 28, 28), torch.float32), ((96, 1, 1), torch.float32)]),
    (Multiply1, [((1, 304, 14, 14), torch.float32), ((304, 1, 1), torch.float32)]),
    (Multiply18, [((208,), torch.float32)]),
    (Multiply123, [((208,), torch.float32)]),
    (Multiply1, [((1, 208, 14, 14), torch.float32), ((208, 1, 1), torch.float32)]),
    (Multiply1, [((208,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((208,), torch.float32), ((208,), torch.float32)]),
    (Multiply1, [((1, 48, 14, 14), torch.float32), ((48, 1, 1), torch.float32)]),
    (Multiply1, [((1, 296, 14, 14), torch.float32), ((296, 1, 1), torch.float32)]),
    (Multiply1, [((1, 224, 14, 14), torch.float32), ((224, 1, 1), torch.float32)]),
    (Multiply1, [((1, 280, 14, 14), torch.float32), ((280, 1, 1), torch.float32)]),
    (Multiply1, [((1, 624, 7, 7), torch.float32), ((624, 1, 1), torch.float32)]),
    (Multiply1, [((1, 384, 7, 7), torch.float32), ((384, 1, 1), torch.float32)]),
    (Multiply18, [((18,), torch.float32)]),
    (Multiply124, [((18,), torch.float32)]),
    (Multiply1, [((1, 18, 56, 56), torch.float32), ((18, 1, 1), torch.float32)]),
    (Multiply1, [((18,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((18,), torch.float32), ((18,), torch.float32)]),
    (Multiply1, [((1, 36, 28, 28), torch.float32), ((36, 1, 1), torch.float32)]),
    (Multiply1, [((1, 18, 28, 28), torch.float32), ((18, 1, 1), torch.float32)]),
    (Multiply1, [((1, 72, 14, 14), torch.float32), ((72, 1, 1), torch.float32)]),
    (Multiply1, [((1, 18, 14, 14), torch.float32), ((18, 1, 1), torch.float32)]),
    (Multiply1, [((1, 36, 14, 14), torch.float32), ((36, 1, 1), torch.float32)]),
    (Multiply1, [((1, 144, 7, 7), torch.float32), ((144, 1, 1), torch.float32)]),
    (Multiply1, [((1, 126, 7, 7), torch.float32), ((126, 1, 1), torch.float32)]),
    (Multiply1, [((1, 40, 56, 56), torch.float32), ((40, 1, 1), torch.float32)]),
    (Multiply1, [((1, 80, 28, 28), torch.float32), ((80, 1, 1), torch.float32)]),
    (Multiply1, [((1, 280, 7, 7), torch.float32), ((280, 1, 1), torch.float32)]),
    (Multiply1, [((1, 32, 14, 14), torch.float32), ((32, 1, 1), torch.float32)]),
    (Multiply1, [((1, 224, 7, 7), torch.float32), ((224, 1, 1), torch.float32)]),
    (Multiply1, [((1, 16, 28, 28), torch.float32), ((16, 1, 1), torch.float32)]),
    (Multiply1, [((1, 16, 14, 14), torch.float32), ((16, 1, 1), torch.float32)]),
    (Multiply18, [((44,), torch.float32)]),
    (Multiply125, [((44,), torch.float32)]),
    (Multiply1, [((1, 44, 56, 56), torch.float32), ((44, 1, 1), torch.float32)]),
    (Multiply1, [((44,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((44,), torch.float32), ((44,), torch.float32)]),
    (Multiply18, [((88,), torch.float32)]),
    (Multiply126, [((88,), torch.float32)]),
    (Multiply1, [((1, 88, 28, 28), torch.float32), ((88, 1, 1), torch.float32)]),
    (Multiply1, [((88,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((88,), torch.float32), ((88,), torch.float32)]),
    (Multiply1, [((1, 44, 28, 28), torch.float32), ((44, 1, 1), torch.float32)]),
    (Multiply18, [((176,), torch.float32)]),
    (Multiply127, [((176,), torch.float32)]),
    (Multiply1, [((1, 176, 14, 14), torch.float32), ((176, 1, 1), torch.float32)]),
    (Multiply1, [((176,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((176,), torch.float32), ((176,), torch.float32)]),
    (Multiply1, [((1, 44, 14, 14), torch.float32), ((44, 1, 1), torch.float32)]),
    (Multiply1, [((1, 88, 14, 14), torch.float32), ((88, 1, 1), torch.float32)]),
    (Multiply1, [((1, 352, 7, 7), torch.float32), ((352, 1, 1), torch.float32)]),
    (Multiply1, [((1, 308, 7, 7), torch.float32), ((308, 1, 1), torch.float32)]),
    (Multiply1, [((1, 48, 28, 28), torch.float32), ((48, 1, 1), torch.float32)]),
    (Multiply1, [((1, 96, 14, 14), torch.float32), ((96, 1, 1), torch.float32)]),
    (Multiply1, [((1, 336, 7, 7), torch.float32), ((336, 1, 1), torch.float32)]),
    (Multiply18, [((30,), torch.float32)]),
    (Multiply128, [((30,), torch.float32)]),
    (Multiply1, [((1, 30, 56, 56), torch.float32), ((30, 1, 1), torch.float32)]),
    (Multiply1, [((30,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((30,), torch.float32), ((30,), torch.float32)]),
    (Multiply1, [((1, 30, 28, 28), torch.float32), ((30, 1, 1), torch.float32)]),
    (Multiply1, [((1, 120, 14, 14), torch.float32), ((120, 1, 1), torch.float32)]),
    (Multiply1, [((1, 30, 14, 14), torch.float32), ((30, 1, 1), torch.float32)]),
    (Multiply1, [((1, 60, 14, 14), torch.float32), ((60, 1, 1), torch.float32)]),
    (Multiply1, [((1, 240, 7, 7), torch.float32), ((240, 1, 1), torch.float32)]),
    (Multiply1, [((1, 210, 7, 7), torch.float32), ((210, 1, 1), torch.float32)]),
    (Multiply1, [((1, 32, 149, 149), torch.float32), ((32, 1, 1), torch.float32)]),
    (Multiply1, [((1, 32, 147, 147), torch.float32), ((32, 1, 1), torch.float32)]),
    (Multiply1, [((1, 64, 147, 147), torch.float32), ((64, 1, 1), torch.float32)]),
    (Multiply1, [((1, 96, 73, 73), torch.float32), ((96, 1, 1), torch.float32)]),
    (Multiply1, [((1, 64, 73, 73), torch.float32), ((64, 1, 1), torch.float32)]),
    (Multiply1, [((1, 96, 71, 71), torch.float32), ((96, 1, 1), torch.float32)]),
    (Multiply1, [((1, 192, 35, 35), torch.float32), ((192, 1, 1), torch.float32)]),
    (Multiply1, [((1, 224, 35, 35), torch.float32), ((224, 1, 1), torch.float32)]),
    (Multiply1, [((1, 96, 35, 35), torch.float32), ((96, 1, 1), torch.float32)]),
    (Multiply1, [((1, 384, 17, 17), torch.float32), ((384, 1, 1), torch.float32)]),
    (Multiply1, [((1, 256, 17, 17), torch.float32), ((256, 1, 1), torch.float32)]),
    (Multiply1, [((1, 768, 17, 17), torch.float32), ((768, 1, 1), torch.float32)]),
    (Multiply1, [((1, 224, 17, 17), torch.float32), ((224, 1, 1), torch.float32)]),
    (Multiply1, [((1, 192, 17, 17), torch.float32), ((192, 1, 1), torch.float32)]),
    (Multiply1, [((1, 128, 17, 17), torch.float32), ((128, 1, 1), torch.float32)]),
    (Multiply1, [((1, 192, 8, 8), torch.float32), ((192, 1, 1), torch.float32)]),
    (Multiply1, [((1, 320, 17, 17), torch.float32), ((320, 1, 1), torch.float32)]),
    (Multiply1, [((1, 320, 8, 8), torch.float32), ((320, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1024, 8, 8), torch.float32), ((1024, 1, 1), torch.float32)]),
    (Multiply1, [((1, 448, 8, 8), torch.float32), ((448, 1, 1), torch.float32)]),
    (Multiply1, [((1, 512, 8, 8), torch.float32), ((512, 1, 1), torch.float32)]),
    (Multiply1, [((1, 24, 96, 96), torch.float32), ((24, 1, 1), torch.float32)]),
    (Multiply1, [((1, 48, 96, 96), torch.float32), ((48, 1, 1), torch.float32)]),
    (Multiply1, [((1, 48, 48, 48), torch.float32), ((48, 1, 1), torch.float32)]),
    (Multiply1, [((1, 96, 48, 48), torch.float32), ((96, 1, 1), torch.float32)]),
    (Multiply1, [((1, 96, 24, 24), torch.float32), ((96, 1, 1), torch.float32)]),
    (Multiply1, [((1, 192, 24, 24), torch.float32), ((192, 1, 1), torch.float32)]),
    (Multiply1, [((1, 192, 12, 12), torch.float32), ((192, 1, 1), torch.float32)]),
    (Multiply1, [((1, 384, 12, 12), torch.float32), ((384, 1, 1), torch.float32)]),
    (Multiply1, [((1, 384, 6, 6), torch.float32), ((384, 1, 1), torch.float32)]),
    (Multiply1, [((1, 768, 6, 6), torch.float32), ((768, 1, 1), torch.float32)]),
    (Multiply1, [((1, 32, 32, 32), torch.float32), ((32, 1, 1), torch.float32)]),
    (Multiply1, [((1, 64, 32, 32), torch.float32), ((64, 1, 1), torch.float32)]),
    (Multiply1, [((1, 64, 16, 16), torch.float32), ((64, 1, 1), torch.float32)]),
    (Multiply1, [((1, 128, 16, 16), torch.float32), ((128, 1, 1), torch.float32)]),
    (Multiply1, [((1, 128, 8, 8), torch.float32), ((128, 1, 1), torch.float32)]),
    (Multiply1, [((1, 256, 4, 4), torch.float32), ((256, 1, 1), torch.float32)]),
    (Multiply1, [((1, 512, 4, 4), torch.float32), ((512, 1, 1), torch.float32)]),
    (Multiply1, [((1, 512, 2, 2), torch.float32), ((512, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1024, 2, 2), torch.float32), ((1024, 1, 1), torch.float32)]),
    (Multiply1, [((1, 24, 80, 80), torch.float32), ((24, 1, 1), torch.float32)]),
    (Multiply1, [((1, 16, 80, 80), torch.float32), ((16, 1, 1), torch.float32)]),
    (Multiply1, [((1, 96, 80, 80), torch.float32), ((96, 1, 1), torch.float32)]),
    (Multiply1, [((1, 96, 40, 40), torch.float32), ((96, 1, 1), torch.float32)]),
    (Multiply1, [((1, 24, 40, 40), torch.float32), ((24, 1, 1), torch.float32)]),
    (Multiply1, [((1, 144, 40, 40), torch.float32), ((144, 1, 1), torch.float32)]),
    (Multiply1, [((1, 144, 20, 20), torch.float32), ((144, 1, 1), torch.float32)]),
    (Multiply1, [((1, 24, 20, 20), torch.float32), ((24, 1, 1), torch.float32)]),
    (Multiply1, [((1, 144, 10, 10), torch.float32), ((144, 1, 1), torch.float32)]),
    (Multiply1, [((1, 48, 10, 10), torch.float32), ((48, 1, 1), torch.float32)]),
    (Multiply1, [((1, 288, 10, 10), torch.float32), ((288, 1, 1), torch.float32)]),
    (Multiply1, [((1, 72, 10, 10), torch.float32), ((72, 1, 1), torch.float32)]),
    (Multiply1, [((1, 432, 10, 10), torch.float32), ((432, 1, 1), torch.float32)]),
    (Multiply1, [((1, 432, 5, 5), torch.float32), ((432, 1, 1), torch.float32)]),
    (Multiply1, [((1, 120, 5, 5), torch.float32), ((120, 1, 1), torch.float32)]),
    (Multiply1, [((1, 720, 5, 5), torch.float32), ((720, 1, 1), torch.float32)]),
    (Multiply1, [((1, 240, 5, 5), torch.float32), ((240, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1280, 5, 5), torch.float32), ((1280, 1, 1), torch.float32)]),
    (Multiply1, [((1, 16, 48, 48), torch.float32), ((16, 1, 1), torch.float32)]),
    (Multiply1, [((1, 8, 48, 48), torch.float32), ((8, 1, 1), torch.float32)]),
    (Multiply1, [((1, 48, 24, 24), torch.float32), ((48, 1, 1), torch.float32)]),
    (Multiply1, [((1, 8, 24, 24), torch.float32), ((8, 1, 1), torch.float32)]),
    (Multiply1, [((1, 48, 12, 12), torch.float32), ((48, 1, 1), torch.float32)]),
    (Multiply1, [((1, 16, 12, 12), torch.float32), ((16, 1, 1), torch.float32)]),
    (Multiply1, [((1, 96, 12, 12), torch.float32), ((96, 1, 1), torch.float32)]),
    (Multiply1, [((1, 96, 6, 6), torch.float32), ((96, 1, 1), torch.float32)]),
    (Multiply1, [((1, 24, 6, 6), torch.float32), ((24, 1, 1), torch.float32)]),
    (Multiply1, [((1, 144, 6, 6), torch.float32), ((144, 1, 1), torch.float32)]),
    (Multiply1, [((1, 32, 6, 6), torch.float32), ((32, 1, 1), torch.float32)]),
    (Multiply1, [((1, 192, 6, 6), torch.float32), ((192, 1, 1), torch.float32)]),
    (Multiply1, [((1, 192, 3, 3), torch.float32), ((192, 1, 1), torch.float32)]),
    (Multiply1, [((1, 56, 3, 3), torch.float32), ((56, 1, 1), torch.float32)]),
    (Multiply1, [((1, 336, 3, 3), torch.float32), ((336, 1, 1), torch.float32)]),
    (Multiply1, [((1, 112, 3, 3), torch.float32), ((112, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1280, 3, 3), torch.float32), ((1280, 1, 1), torch.float32)]),
    (Multiply1, [((1, 960, 28, 28), torch.float32), ((960, 1, 1), torch.float32)]),
    (Multiply1, [((1, 256, 1, 1), torch.float32), ((256, 1, 1), torch.float32)]),
    (Multiply0, [((1, 16, 112, 112), torch.float32)]),
    (Multiply1, [((1, 16, 112, 112), torch.float32), ((1, 16, 112, 112), torch.float32)]),
    (Multiply0, [((1, 16, 1, 1), torch.float32)]),
    (Multiply1, [((1, 16, 56, 56), torch.float32), ((1, 16, 1, 1), torch.float32)]),
    (Multiply1, [((1, 72, 56, 56), torch.float32), ((72, 1, 1), torch.float32)]),
    (Multiply0, [((1, 96, 28, 28), torch.float32)]),
    (Multiply1, [((1, 96, 28, 28), torch.float32), ((1, 96, 28, 28), torch.float32)]),
    (Multiply0, [((1, 96, 14, 14), torch.float32)]),
    (Multiply1, [((1, 96, 14, 14), torch.float32), ((1, 96, 14, 14), torch.float32)]),
    (Multiply0, [((1, 96, 1, 1), torch.float32)]),
    (Multiply1, [((1, 96, 14, 14), torch.float32), ((1, 96, 1, 1), torch.float32)]),
    (Multiply0, [((1, 240, 14, 14), torch.float32)]),
    (Multiply0, [((1, 240, 1, 1), torch.float32)]),
    (Multiply0, [((1, 120, 14, 14), torch.float32)]),
    (Multiply1, [((1, 120, 14, 14), torch.float32), ((1, 120, 14, 14), torch.float32)]),
    (Multiply1, [((1, 120, 14, 14), torch.float32), ((1, 120, 1, 1), torch.float32)]),
    (Multiply1, [((1, 144, 14, 14), torch.float32), ((144, 1, 1), torch.float32)]),
    (Multiply0, [((1, 144, 14, 14), torch.float32)]),
    (Multiply1, [((1, 144, 14, 14), torch.float32), ((1, 144, 14, 14), torch.float32)]),
    (Multiply0, [((1, 144, 1, 1), torch.float32)]),
    (Multiply1, [((1, 144, 14, 14), torch.float32), ((1, 144, 1, 1), torch.float32)]),
    (Multiply0, [((1, 288, 14, 14), torch.float32)]),
    (Multiply1, [((1, 288, 14, 14), torch.float32), ((1, 288, 14, 14), torch.float32)]),
    (Multiply1, [((1, 288, 7, 7), torch.float32), ((288, 1, 1), torch.float32)]),
    (Multiply0, [((1, 288, 7, 7), torch.float32)]),
    (Multiply1, [((1, 288, 7, 7), torch.float32), ((1, 288, 7, 7), torch.float32)]),
    (Multiply0, [((1, 288, 1, 1), torch.float32)]),
    (Multiply1, [((1, 288, 7, 7), torch.float32), ((1, 288, 1, 1), torch.float32)]),
    (Multiply1, [((1, 96, 7, 7), torch.float32), ((96, 1, 1), torch.float32)]),
    (Multiply0, [((1, 576, 7, 7), torch.float32)]),
    (Multiply1, [((1, 576, 7, 7), torch.float32), ((1, 576, 7, 7), torch.float32)]),
    (Multiply0, [((1, 576, 1, 1), torch.float32)]),
    (Multiply1, [((1, 576, 7, 7), torch.float32), ((1, 576, 1, 1), torch.float32)]),
    (Multiply0, [((1, 1024, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1024, 1, 1), torch.float32), ((1, 1024, 1, 1), torch.float32)]),
    (Multiply0, [((1, 240, 28, 28), torch.float32)]),
    (Multiply18, [((200,), torch.float32)]),
    (Multiply129, [((200,), torch.float32)]),
    (Multiply1, [((1, 200, 14, 14), torch.float32), ((200, 1, 1), torch.float32)]),
    (Multiply1, [((200,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((200,), torch.float32), ((200,), torch.float32)]),
    (Multiply0, [((1, 200, 14, 14), torch.float32)]),
    (Multiply1, [((1, 200, 14, 14), torch.float32), ((1, 200, 14, 14), torch.float32)]),
    (Multiply18, [((184,), torch.float32)]),
    (Multiply130, [((184,), torch.float32)]),
    (Multiply1, [((1, 184, 14, 14), torch.float32), ((184, 1, 1), torch.float32)]),
    (Multiply1, [((184,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((184,), torch.float32), ((184,), torch.float32)]),
    (Multiply0, [((1, 184, 14, 14), torch.float32)]),
    (Multiply1, [((1, 184, 14, 14), torch.float32), ((1, 184, 14, 14), torch.float32)]),
    (Multiply0, [((1, 480, 14, 14), torch.float32)]),
    (Multiply0, [((1, 672, 14, 14), torch.float32)]),
    (Multiply0, [((1, 672, 7, 7), torch.float32)]),
    (Multiply0, [((1, 960, 7, 7), torch.float32)]),
    (Multiply0, [((1, 1280, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1280, 1, 1), torch.float32), ((1, 1280, 1, 1), torch.float32)]),
    (Multiply1, [((1, 72, 1, 1), torch.float32), ((1, 72, 28, 28), torch.float32)]),
    (Multiply1, [((1, 120, 1, 1), torch.float32), ((1, 120, 28, 28), torch.float32)]),
    (Multiply0, [((1, 1280), torch.float32)]),
    (Multiply1, [((1, 1280), torch.float32), ((1, 1280), torch.float32)]),
    (Multiply1, [((1, 16, 1, 1), torch.float32), ((1, 16, 56, 56), torch.float32)]),
    (Multiply1, [((1, 96, 1, 1), torch.float32), ((1, 96, 14, 14), torch.float32)]),
    (Multiply1, [((1, 120, 1, 1), torch.float32), ((1, 120, 14, 14), torch.float32)]),
    (Multiply1, [((1, 144, 1, 1), torch.float32), ((1, 144, 14, 14), torch.float32)]),
    (Multiply1, [((1, 288, 1, 1), torch.float32), ((1, 288, 7, 7), torch.float32)]),
    (Multiply1, [((1, 576, 1, 1), torch.float32), ((1, 576, 7, 7), torch.float32)]),
    (Multiply0, [((1, 1024), torch.float32)]),
    (Multiply1, [((1, 1024), torch.float32), ((1, 1024), torch.float32)]),
    (Multiply0, [((1, 3, 320, 1024), torch.float32)]),
    (Multiply1, [((1, 64, 160, 512), torch.float32), ((64, 1, 1), torch.float32)]),
    (Multiply1, [((1, 64, 80, 256), torch.float32), ((64, 1, 1), torch.float32)]),
    (Multiply1, [((1, 128, 40, 128), torch.float32), ((128, 1, 1), torch.float32)]),
    (Multiply1, [((1, 256, 20, 64), torch.float32), ((256, 1, 1), torch.float32)]),
    (Multiply1, [((1, 512, 10, 32), torch.float32), ((512, 1, 1), torch.float32)]),
    (Multiply0, [((1, 3, 192, 640), torch.float32)]),
    (Multiply1, [((1, 64, 96, 320), torch.float32), ((64, 1, 1), torch.float32)]),
    (Multiply1, [((1, 64, 48, 160), torch.float32), ((64, 1, 1), torch.float32)]),
    (Multiply1, [((1, 128, 24, 80), torch.float32), ((128, 1, 1), torch.float32)]),
    (Multiply1, [((1, 256, 12, 40), torch.float32), ((256, 1, 1), torch.float32)]),
    (Multiply1, [((1, 512, 6, 20), torch.float32), ((512, 1, 1), torch.float32)]),
    (Multiply0, [((1, 1, 512, 3025), torch.float32)]),
    (Multiply0, [((1, 8, 512, 512), torch.float32)]),
    (Multiply0, [((1, 1, 1, 512), torch.float32)]),
    (Multiply0, [((1, 1, 512, 50176), torch.float32)]),
    (Multiply1, [((1, 128, 56, 56), torch.float32), ((1, 128, 1, 1), torch.float32)]),
    (Multiply1, [((1, 192, 28, 28), torch.float32), ((1, 192, 1, 1), torch.float32)]),
    (Multiply1, [((1, 512, 14, 14), torch.float32), ((1, 512, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1088, 7, 7), torch.float32), ((1, 1088, 1, 1), torch.float32)]),
    (Multiply1, [((1, 64, 240, 320), torch.float32), ((64, 1, 1), torch.float32)]),
    (Multiply1, [((1, 64, 120, 160), torch.float32), ((64, 1, 1), torch.float32)]),
    (Multiply1, [((1, 128, 60, 80), torch.float32), ((128, 1, 1), torch.float32)]),
    (Multiply1, [((1, 256, 30, 40), torch.float32), ((256, 1, 1), torch.float32)]),
    (Multiply1, [((1, 512, 15, 20), torch.float32), ((512, 1, 1), torch.float32)]),
    (Multiply1, [((1, 256, 120, 160), torch.float32), ((256, 1, 1), torch.float32)]),
    (Multiply1, [((1, 128, 120, 160), torch.float32), ((128, 1, 1), torch.float32)]),
    (Multiply1, [((1, 512, 60, 80), torch.float32), ((512, 1, 1), torch.float32)]),
    (Multiply1, [((1, 256, 60, 80), torch.float32), ((256, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1024, 30, 40), torch.float32), ((1024, 1, 1), torch.float32)]),
    (Multiply1, [((1, 512, 30, 40), torch.float32), ((512, 1, 1), torch.float32)]),
    (Multiply1, [((1, 2048, 15, 20), torch.float32), ((2048, 1, 1), torch.float32)]),
    (Multiply0, [((1, 1, 16384, 256), torch.float32)]),
    (Multiply0, [((1, 2, 4096, 256), torch.float32)]),
    (Multiply0, [((1, 5, 1024, 256), torch.float32)]),
    (Multiply0, [((1, 8, 256, 256), torch.float32)]),
    (Multiply1, [((1, 256, 128, 128), torch.float32), ((256, 1, 1), torch.float32)]),
    (Multiply1, [((1, 768, 128, 128), torch.float32), ((768, 1, 1), torch.float32)]),
    (Multiply1, [((1, 64, 150, 150), torch.float32), ((64, 1, 1), torch.float32)]),
    (Multiply1, [((1, 64, 75, 75), torch.float32), ((64, 1, 1), torch.float32)]),
    (Multiply1, [((1, 256, 75, 75), torch.float32), ((256, 1, 1), torch.float32)]),
    (Multiply1, [((1, 128, 75, 75), torch.float32), ((128, 1, 1), torch.float32)]),
    (Multiply1, [((1, 128, 38, 38), torch.float32), ((128, 1, 1), torch.float32)]),
    (Multiply1, [((1, 512, 38, 38), torch.float32), ((512, 1, 1), torch.float32)]),
    (Multiply1, [((1, 256, 38, 38), torch.float32), ((256, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1024, 38, 38), torch.float32), ((1024, 1, 1), torch.float32)]),
    (Multiply1, [((1, 512, 19, 19), torch.float32), ((512, 1, 1), torch.float32)]),
    (Multiply1, [((1, 256, 19, 19), torch.float32), ((256, 1, 1), torch.float32)]),
    (Multiply1, [((1, 512, 10, 10), torch.float32), ((512, 1, 1), torch.float32)]),
    (Multiply1, [((1, 128, 10, 10), torch.float32), ((128, 1, 1), torch.float32)]),
    (Multiply1, [((1, 256, 5, 5), torch.float32), ((256, 1, 1), torch.float32)]),
    (Multiply1, [((1, 128, 5, 5), torch.float32), ((128, 1, 1), torch.float32)]),
    (Multiply1, [((1, 256, 3, 3), torch.float32), ((256, 1, 1), torch.float32)]),
    (Multiply1, [((1, 128, 3, 3), torch.float32), ((128, 1, 1), torch.float32)]),
    (Multiply1, [((1, 32, 256, 256), torch.float32), ((32, 1, 1), torch.float32)]),
    (Multiply1, [((1, 64, 128, 128), torch.float32), ((64, 1, 1), torch.float32)]),
    (Multiply1, [((1, 128, 64, 64), torch.float32), ((128, 1, 1), torch.float32)]),
    (Multiply1, [((1, 256, 32, 32), torch.float32), ((256, 1, 1), torch.float32)]),
    (Multiply1, [((1, 512, 16, 16), torch.float32), ((512, 1, 1), torch.float32)]),
    (Multiply1, [((1, 64, 224, 224), torch.float32), ((64, 1, 1), torch.float32)]),
    (Multiply0, [((1, 16, 197, 197), torch.float32)]),
    (Multiply0, [((1, 256, 1, 1), torch.float32)]),
    (Multiply1, [((1, 256, 56, 56), torch.float32), ((1, 256, 1, 1), torch.float32)]),
    (Multiply0, [((1, 512, 1, 1), torch.float32)]),
    (Multiply1, [((1, 512, 28, 28), torch.float32), ((1, 512, 1, 1), torch.float32)]),
    (Multiply0, [((1, 768, 1, 1), torch.float32)]),
    (Multiply1, [((1, 768, 14, 14), torch.float32), ((1, 768, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1024, 7, 7), torch.float32), ((1, 1024, 1, 1), torch.float32)]),
    (Multiply1, [((1, 32, 150, 150), torch.float32), ((32, 1, 1), torch.float32)]),
    (Multiply1, [((1, 128, 150, 150), torch.float32), ((128, 1, 1), torch.float32)]),
    (Multiply18, [((728,), torch.float32)]),
    (Multiply131, [((728,), torch.float32)]),
    (Multiply1, [((1, 728, 38, 38), torch.float32), ((728, 1, 1), torch.float32)]),
    (Multiply1, [((728,), torch.float32), ((1,), torch.float32)]),
    (Multiply1, [((728,), torch.float32), ((728,), torch.float32)]),
    (Multiply1, [((1, 728, 19, 19), torch.float32), ((728, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1024, 19, 19), torch.float32), ((1024, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1024, 10, 10), torch.float32), ((1024, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1536, 10, 10), torch.float32), ((1536, 1, 1), torch.float32)]),
    (Multiply1, [((1, 2048, 10, 10), torch.float32), ((2048, 1, 1), torch.float32)]),
    (Multiply1, [((1, 128, 147, 147), torch.float32), ((128, 1, 1), torch.float32)]),
    (Multiply1, [((1, 128, 74, 74), torch.float32), ((128, 1, 1), torch.float32)]),
    (Multiply1, [((1, 256, 74, 74), torch.float32), ((256, 1, 1), torch.float32)]),
    (Multiply1, [((1, 256, 37, 37), torch.float32), ((256, 1, 1), torch.float32)]),
    (Multiply1, [((1, 728, 37, 37), torch.float32), ((728, 1, 1), torch.float32)]),
    (Multiply1, [((1, 80, 320, 320), torch.float32), ((1, 80, 320, 320), torch.float32)]),
    (Multiply1, [((1, 160, 160, 160), torch.float32), ((1, 160, 160, 160), torch.float32)]),
    (Multiply1, [((1, 80, 160, 160), torch.float32), ((1, 80, 160, 160), torch.float32)]),
    (Multiply1, [((1, 320, 80, 80), torch.float32), ((1, 320, 80, 80), torch.float32)]),
    (Multiply1, [((1, 160, 80, 80), torch.float32), ((1, 160, 80, 80), torch.float32)]),
    (Multiply1, [((1, 640, 40, 40), torch.float32), ((1, 640, 40, 40), torch.float32)]),
    (Multiply1, [((1, 320, 40, 40), torch.float32), ((1, 320, 40, 40), torch.float32)]),
    (Multiply1, [((1, 1280, 20, 20), torch.float32), ((1, 1280, 20, 20), torch.float32)]),
    (Multiply1, [((1, 640, 20, 20), torch.float32), ((1, 640, 20, 20), torch.float32)]),
    (Multiply0, [((1, 255, 80, 80), torch.float32)]),
    (Multiply1, [((1, 255, 80, 80), torch.float32), ((1, 255, 80, 80), torch.float32)]),
    (Multiply132, [((1, 255, 80, 80), torch.float32)]),
    (Multiply0, [((1, 255, 40, 40), torch.float32)]),
    (Multiply1, [((1, 255, 40, 40), torch.float32), ((1, 255, 40, 40), torch.float32)]),
    (Multiply133, [((1, 255, 40, 40), torch.float32)]),
    (Multiply0, [((1, 255, 20, 20), torch.float32)]),
    (Multiply1, [((1, 255, 20, 20), torch.float32), ((1, 255, 20, 20), torch.float32)]),
    (Multiply134, [((1, 255, 20, 20), torch.float32)]),
    (Multiply1, [((1, 48, 240, 240), torch.float32), ((1, 48, 240, 240), torch.float32)]),
    (Multiply1, [((1, 96, 120, 120), torch.float32), ((1, 96, 120, 120), torch.float32)]),
    (Multiply1, [((1, 48, 120, 120), torch.float32), ((1, 48, 120, 120), torch.float32)]),
    (Multiply1, [((1, 192, 60, 60), torch.float32), ((1, 192, 60, 60), torch.float32)]),
    (Multiply1, [((1, 96, 60, 60), torch.float32), ((1, 96, 60, 60), torch.float32)]),
    (Multiply1, [((1, 384, 30, 30), torch.float32), ((1, 384, 30, 30), torch.float32)]),
    (Multiply1, [((1, 192, 30, 30), torch.float32), ((1, 192, 30, 30), torch.float32)]),
    (Multiply1, [((1, 768, 15, 15), torch.float32), ((1, 768, 15, 15), torch.float32)]),
    (Multiply1, [((1, 384, 15, 15), torch.float32), ((1, 384, 15, 15), torch.float32)]),
    (Multiply0, [((1, 255, 60, 60), torch.float32)]),
    (Multiply1, [((1, 255, 60, 60), torch.float32), ((1, 255, 60, 60), torch.float32)]),
    (Multiply135, [((1, 255, 60, 60), torch.float32)]),
    (Multiply0, [((1, 255, 30, 30), torch.float32)]),
    (Multiply1, [((1, 255, 30, 30), torch.float32), ((1, 255, 30, 30), torch.float32)]),
    (Multiply136, [((1, 255, 30, 30), torch.float32)]),
    (Multiply0, [((1, 255, 15, 15), torch.float32)]),
    (Multiply1, [((1, 255, 15, 15), torch.float32), ((1, 255, 15, 15), torch.float32)]),
    (Multiply137, [((1, 255, 15, 15), torch.float32)]),
    (Multiply1, [((1, 80, 80, 80), torch.float32), ((1, 80, 80, 80), torch.float32)]),
    (Multiply1, [((1, 160, 40, 40), torch.float32), ((1, 160, 40, 40), torch.float32)]),
    (Multiply1, [((1, 320, 20, 20), torch.float32), ((1, 320, 20, 20), torch.float32)]),
    (Multiply1, [((1, 1280, 10, 10), torch.float32), ((1, 1280, 10, 10), torch.float32)]),
    (Multiply1, [((1, 640, 10, 10), torch.float32), ((1, 640, 10, 10), torch.float32)]),
    (Multiply0, [((1, 255, 10, 10), torch.float32)]),
    (Multiply1, [((1, 255, 10, 10), torch.float32), ((1, 255, 10, 10), torch.float32)]),
    (Multiply138, [((1, 255, 10, 10), torch.float32)]),
    (Multiply1, [((1, 32, 320, 320), torch.float32), ((1, 32, 320, 320), torch.float32)]),
    (Multiply1, [((1, 64, 160, 160), torch.float32), ((1, 64, 160, 160), torch.float32)]),
    (Multiply1, [((1, 32, 160, 160), torch.float32), ((1, 32, 160, 160), torch.float32)]),
    (Multiply1, [((1, 128, 80, 80), torch.float32), ((1, 128, 80, 80), torch.float32)]),
    (Multiply1, [((1, 64, 80, 80), torch.float32), ((1, 64, 80, 80), torch.float32)]),
    (Multiply1, [((1, 256, 40, 40), torch.float32), ((1, 256, 40, 40), torch.float32)]),
    (Multiply1, [((1, 128, 40, 40), torch.float32), ((1, 128, 40, 40), torch.float32)]),
    (Multiply1, [((1, 512, 20, 20), torch.float32), ((1, 512, 20, 20), torch.float32)]),
    (Multiply1, [((1, 256, 20, 20), torch.float32), ((1, 256, 20, 20), torch.float32)]),
    (Multiply1, [((1, 1024, 10, 10), torch.float32), ((1, 1024, 10, 10), torch.float32)]),
    (Multiply1, [((1, 512, 10, 10), torch.float32), ((1, 512, 10, 10), torch.float32)]),
    (Multiply1, [((1, 64, 320, 320), torch.float32), ((1, 64, 320, 320), torch.float32)]),
    (Multiply1, [((1, 128, 160, 160), torch.float32), ((1, 128, 160, 160), torch.float32)]),
    (Multiply1, [((1, 256, 80, 80), torch.float32), ((1, 256, 80, 80), torch.float32)]),
    (Multiply1, [((1, 512, 40, 40), torch.float32), ((1, 512, 40, 40), torch.float32)]),
    (Multiply1, [((1, 1024, 20, 20), torch.float32), ((1, 1024, 20, 20), torch.float32)]),
    (Multiply1, [((1, 32, 640, 640), torch.float32), ((1, 32, 640, 640), torch.float32)]),
    (Multiply0, [((1, 255, 160, 160), torch.float32)]),
    (Multiply1, [((1, 255, 160, 160), torch.float32), ((1, 255, 160, 160), torch.float32)]),
    (Multiply139, [((1, 255, 160, 160), torch.float32)]),
    (Multiply1, [((1, 16, 320, 320), torch.float32), ((1, 16, 320, 320), torch.float32)]),
    (Multiply1, [((1, 16, 160, 160), torch.float32), ((1, 16, 160, 160), torch.float32)]),
    (Multiply1, [((1, 32, 80, 80), torch.float32), ((1, 32, 80, 80), torch.float32)]),
    (Multiply1, [((1, 64, 40, 40), torch.float32), ((1, 64, 40, 40), torch.float32)]),
    (Multiply1, [((1, 128, 20, 20), torch.float32), ((1, 128, 20, 20), torch.float32)]),
    (Multiply1, [((1, 32, 240, 240), torch.float32), ((1, 32, 240, 240), torch.float32)]),
    (Multiply1, [((1, 64, 120, 120), torch.float32), ((1, 64, 120, 120), torch.float32)]),
    (Multiply1, [((1, 32, 120, 120), torch.float32), ((1, 32, 120, 120), torch.float32)]),
    (Multiply1, [((1, 128, 60, 60), torch.float32), ((1, 128, 60, 60), torch.float32)]),
    (Multiply1, [((1, 64, 60, 60), torch.float32), ((1, 64, 60, 60), torch.float32)]),
    (Multiply1, [((1, 256, 30, 30), torch.float32), ((1, 256, 30, 30), torch.float32)]),
    (Multiply1, [((1, 128, 30, 30), torch.float32), ((1, 128, 30, 30), torch.float32)]),
    (Multiply1, [((1, 512, 15, 15), torch.float32), ((1, 512, 15, 15), torch.float32)]),
    (Multiply1, [((1, 256, 15, 15), torch.float32), ((1, 256, 15, 15), torch.float32)]),
    (Multiply1, [((1, 80, 240, 240), torch.float32), ((1, 80, 240, 240), torch.float32)]),
    (Multiply1, [((1, 160, 120, 120), torch.float32), ((1, 160, 120, 120), torch.float32)]),
    (Multiply1, [((1, 80, 120, 120), torch.float32), ((1, 80, 120, 120), torch.float32)]),
    (Multiply1, [((1, 320, 60, 60), torch.float32), ((1, 320, 60, 60), torch.float32)]),
    (Multiply1, [((1, 160, 60, 60), torch.float32), ((1, 160, 60, 60), torch.float32)]),
    (Multiply1, [((1, 640, 30, 30), torch.float32), ((1, 640, 30, 30), torch.float32)]),
    (Multiply1, [((1, 320, 30, 30), torch.float32), ((1, 320, 30, 30), torch.float32)]),
    (Multiply1, [((1, 1280, 15, 15), torch.float32), ((1, 1280, 15, 15), torch.float32)]),
    (Multiply1, [((1, 640, 15, 15), torch.float32), ((1, 640, 15, 15), torch.float32)]),
    (Multiply1, [((1, 16, 240, 240), torch.float32), ((1, 16, 240, 240), torch.float32)]),
    (Multiply1, [((1, 16, 120, 120), torch.float32), ((1, 16, 120, 120), torch.float32)]),
    (Multiply1, [((1, 32, 60, 60), torch.float32), ((1, 32, 60, 60), torch.float32)]),
    (Multiply1, [((1, 64, 30, 30), torch.float32), ((1, 64, 30, 30), torch.float32)]),
    (Multiply1, [((1, 128, 15, 15), torch.float32), ((1, 128, 15, 15), torch.float32)]),
    (Multiply1, [((1, 64, 240, 240), torch.float32), ((1, 64, 240, 240), torch.float32)]),
    (Multiply1, [((1, 128, 120, 120), torch.float32), ((1, 128, 120, 120), torch.float32)]),
    (Multiply1, [((1, 256, 60, 60), torch.float32), ((1, 256, 60, 60), torch.float32)]),
    (Multiply1, [((1, 512, 30, 30), torch.float32), ((1, 512, 30, 30), torch.float32)]),
    (Multiply1, [((1, 1024, 15, 15), torch.float32), ((1, 1024, 15, 15), torch.float32)]),
    (Multiply1, [((1, 48, 320, 320), torch.float32), ((1, 48, 320, 320), torch.float32)]),
    (Multiply1, [((1, 96, 160, 160), torch.float32), ((1, 96, 160, 160), torch.float32)]),
    (Multiply1, [((1, 96, 80, 80), torch.float32), ((1, 96, 80, 80), torch.float32)]),
    (Multiply1, [((1, 384, 40, 40), torch.float32), ((1, 384, 40, 40), torch.float32)]),
    (Multiply1, [((1, 768, 20, 20), torch.float32), ((1, 768, 20, 20), torch.float32)]),
    (Multiply1, [((1, 384, 20, 20), torch.float32), ((1, 384, 20, 20), torch.float32)]),
    (Multiply1, [((1, 48, 80, 80), torch.float32), ((1, 48, 80, 80), torch.float32)]),
    (Multiply1, [((1, 96, 40, 40), torch.float32), ((1, 96, 40, 40), torch.float32)]),
    (Multiply1, [((1, 192, 20, 20), torch.float32), ((1, 192, 20, 20), torch.float32)]),
    (Multiply1, [((1, 768, 10, 10), torch.float32), ((1, 768, 10, 10), torch.float32)]),
    (Multiply1, [((1, 384, 10, 10), torch.float32), ((1, 384, 10, 10), torch.float32)]),
    (Multiply1, [((1, 256, 10, 10), torch.float32), ((1, 256, 10, 10), torch.float32)]),
    (Multiply1, [((1, 16, 80, 80), torch.float32), ((1, 16, 80, 80), torch.float32)]),
    (Multiply1, [((1, 32, 40, 40), torch.float32), ((1, 32, 40, 40), torch.float32)]),
    (Multiply1, [((1, 64, 20, 20), torch.float32), ((1, 64, 20, 20), torch.float32)]),
    (Multiply1, [((1, 128, 10, 10), torch.float32), ((1, 128, 10, 10), torch.float32)]),
    (Multiply1, [((1, 32, 56, 80), torch.float32), ((1, 32, 56, 80), torch.float32)]),
    (Multiply1, [((1, 64, 28, 40), torch.float32), ((1, 64, 28, 40), torch.float32)]),
    (Multiply1, [((1, 128, 14, 20), torch.float32), ((1, 128, 14, 20), torch.float32)]),
    (Multiply0, [((1, 5880, 2), torch.float32)]),
    (Multiply140, [((1, 5880, 4), torch.float32)]),
    (Multiply1, [((1, 96, 56, 80), torch.float32), ((1, 96, 56, 80), torch.float32)]),
    (Multiply1, [((1, 192, 28, 40), torch.float32), ((1, 192, 28, 40), torch.float32)]),
    (Multiply1, [((1, 384, 14, 20), torch.float32), ((1, 384, 14, 20), torch.float32)]),
    (Multiply1, [((1, 64, 56, 80), torch.float32), ((1, 64, 56, 80), torch.float32)]),
    (Multiply1, [((1, 128, 28, 40), torch.float32), ((1, 128, 28, 40), torch.float32)]),
    (Multiply1, [((1, 256, 14, 20), torch.float32), ((1, 256, 14, 20), torch.float32)]),
    (Multiply1, [((1, 64, 224, 320), torch.float32), ((1, 64, 224, 320), torch.float32)]),
    (Multiply1, [((1, 128, 112, 160), torch.float32), ((1, 128, 112, 160), torch.float32)]),
    (Multiply1, [((1, 64, 112, 160), torch.float32), ((1, 64, 112, 160), torch.float32)]),
    (Multiply1, [((1, 256, 56, 80), torch.float32), ((1, 256, 56, 80), torch.float32)]),
    (Multiply1, [((1, 128, 56, 80), torch.float32), ((1, 128, 56, 80), torch.float32)]),
    (Multiply1, [((1, 512, 28, 40), torch.float32), ((1, 512, 28, 40), torch.float32)]),
    (Multiply1, [((1, 256, 28, 40), torch.float32), ((1, 256, 28, 40), torch.float32)]),
    (Multiply1, [((1, 1024, 14, 20), torch.float32), ((1, 1024, 14, 20), torch.float32)]),
    (Multiply1, [((1, 512, 14, 20), torch.float32), ((1, 512, 14, 20), torch.float32)]),
    (Multiply1, [((1, 48, 320, 320), torch.float32), ((48, 1, 1), torch.float32)]),
    (Multiply1, [((1, 96, 160, 160), torch.float32), ((96, 1, 1), torch.float32)]),
    (Multiply1, [((1, 384, 40, 40), torch.float32), ((384, 1, 1), torch.float32)]),
    (Multiply1, [((1, 768, 20, 20), torch.float32), ((768, 1, 1), torch.float32)]),
    (Multiply1, [((1, 384, 20, 20), torch.float32), ((384, 1, 1), torch.float32)]),
    (Multiply1, [((1, 192, 20, 20), torch.float32), ((192, 1, 1), torch.float32)]),
    (Multiply1, [((1, 16, 208, 208), torch.float32), ((16, 1, 1), torch.float32)]),
    (Multiply1, [((1, 16, 208, 208), torch.float32), ((1, 16, 208, 208), torch.float32)]),
    (Multiply1, [((1, 16, 104, 104), torch.float32), ((16, 1, 1), torch.float32)]),
    (Multiply1, [((1, 16, 104, 104), torch.float32), ((1, 16, 104, 104), torch.float32)]),
    (Multiply1, [((1, 32, 104, 104), torch.float32), ((32, 1, 1), torch.float32)]),
    (Multiply1, [((1, 32, 104, 104), torch.float32), ((1, 32, 104, 104), torch.float32)]),
    (Multiply1, [((1, 32, 52, 52), torch.float32), ((32, 1, 1), torch.float32)]),
    (Multiply1, [((1, 32, 52, 52), torch.float32), ((1, 32, 52, 52), torch.float32)]),
    (Multiply1, [((1, 64, 52, 52), torch.float32), ((64, 1, 1), torch.float32)]),
    (Multiply1, [((1, 64, 52, 52), torch.float32), ((1, 64, 52, 52), torch.float32)]),
    (Multiply1, [((1, 64, 26, 26), torch.float32), ((64, 1, 1), torch.float32)]),
    (Multiply1, [((1, 64, 26, 26), torch.float32), ((1, 64, 26, 26), torch.float32)]),
    (Multiply1, [((1, 128, 26, 26), torch.float32), ((128, 1, 1), torch.float32)]),
    (Multiply1, [((1, 128, 26, 26), torch.float32), ((1, 128, 26, 26), torch.float32)]),
    (Multiply1, [((1, 128, 13, 13), torch.float32), ((128, 1, 1), torch.float32)]),
    (Multiply1, [((1, 128, 13, 13), torch.float32), ((1, 128, 13, 13), torch.float32)]),
    (Multiply1, [((1, 256, 13, 13), torch.float32), ((256, 1, 1), torch.float32)]),
    (Multiply1, [((1, 256, 13, 13), torch.float32), ((1, 256, 13, 13), torch.float32)]),
    (Multiply1, [((1, 64, 13, 13), torch.float32), ((64, 1, 1), torch.float32)]),
    (Multiply1, [((1, 64, 13, 13), torch.float32), ((1, 64, 13, 13), torch.float32)]),
    (Multiply1, [((1, 32, 640, 640), torch.float32), ((32, 1, 1), torch.float32)]),
    (Multiply1, [((1, 64, 320, 320), torch.float32), ((64, 1, 1), torch.float32)]),
    (Multiply1, [((1, 32, 320, 320), torch.float32), ((32, 1, 1), torch.float32)]),
    (Multiply1, [((1, 128, 160, 160), torch.float32), ((128, 1, 1), torch.float32)]),
    (Multiply1, [((1, 64, 160, 160), torch.float32), ((64, 1, 1), torch.float32)]),
    (Multiply1, [((1, 256, 80, 80), torch.float32), ((256, 1, 1), torch.float32)]),
    (Multiply1, [((1, 128, 80, 80), torch.float32), ((128, 1, 1), torch.float32)]),
    (Multiply1, [((1, 512, 40, 40), torch.float32), ((512, 1, 1), torch.float32)]),
    (Multiply1, [((1, 256, 40, 40), torch.float32), ((256, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1024, 20, 20), torch.float32), ((1024, 1, 1), torch.float32)]),
    (Multiply1, [((1, 512, 20, 20), torch.float32), ((512, 1, 1), torch.float32)]),
    (Multiply1, [((1, 256, 20, 20), torch.float32), ((256, 1, 1), torch.float32)]),
    (Multiply1, [((1, 128, 40, 40), torch.float32), ((128, 1, 1), torch.float32)]),
    (Multiply1, [((1, 80, 320, 320), torch.float32), ((80, 1, 1), torch.float32)]),
    (Multiply1, [((1, 160, 160, 160), torch.float32), ((160, 1, 1), torch.float32)]),
    (Multiply1, [((1, 80, 160, 160), torch.float32), ((80, 1, 1), torch.float32)]),
    (Multiply1, [((1, 320, 80, 80), torch.float32), ((320, 1, 1), torch.float32)]),
    (Multiply1, [((1, 160, 80, 80), torch.float32), ((160, 1, 1), torch.float32)]),
    (Multiply1, [((1, 640, 40, 40), torch.float32), ((640, 1, 1), torch.float32)]),
    (Multiply1, [((1, 320, 40, 40), torch.float32), ((320, 1, 1), torch.float32)]),
    (Multiply1, [((1, 1280, 20, 20), torch.float32), ((1280, 1, 1), torch.float32)]),
    (Multiply1, [((1, 640, 20, 20), torch.float32), ((640, 1, 1), torch.float32)]),
    (Multiply1, [((1, 320, 20, 20), torch.float32), ((320, 1, 1), torch.float32)]),
    (Multiply1, [((1, 32, 160, 160), torch.float32), ((32, 1, 1), torch.float32)]),
    (Multiply1, [((1, 64, 80, 80), torch.float32), ((64, 1, 1), torch.float32)]),
    (Multiply1, [((1, 128, 20, 20), torch.float32), ((128, 1, 1), torch.float32)]),
    (Multiply1, [((1, 24, 208, 208), torch.float32), ((24, 1, 1), torch.float32)]),
    (Multiply1, [((1, 24, 208, 208), torch.float32), ((1, 24, 208, 208), torch.float32)]),
    (Multiply1, [((1, 48, 104, 104), torch.float32), ((48, 1, 1), torch.float32)]),
    (Multiply1, [((1, 48, 104, 104), torch.float32), ((1, 48, 104, 104), torch.float32)]),
    (Multiply1, [((1, 24, 104, 104), torch.float32), ((24, 1, 1), torch.float32)]),
    (Multiply1, [((1, 24, 104, 104), torch.float32), ((1, 24, 104, 104), torch.float32)]),
    (Multiply1, [((1, 96, 52, 52), torch.float32), ((96, 1, 1), torch.float32)]),
    (Multiply1, [((1, 96, 52, 52), torch.float32), ((1, 96, 52, 52), torch.float32)]),
    (Multiply1, [((1, 48, 52, 52), torch.float32), ((48, 1, 1), torch.float32)]),
    (Multiply1, [((1, 48, 52, 52), torch.float32), ((1, 48, 52, 52), torch.float32)]),
    (Multiply1, [((1, 192, 26, 26), torch.float32), ((192, 1, 1), torch.float32)]),
    (Multiply1, [((1, 192, 26, 26), torch.float32), ((1, 192, 26, 26), torch.float32)]),
    (Multiply1, [((1, 96, 26, 26), torch.float32), ((96, 1, 1), torch.float32)]),
    (Multiply1, [((1, 96, 26, 26), torch.float32), ((1, 96, 26, 26), torch.float32)]),
    (Multiply1, [((1, 384, 13, 13), torch.float32), ((384, 1, 1), torch.float32)]),
    (Multiply1, [((1, 384, 13, 13), torch.float32), ((1, 384, 13, 13), torch.float32)]),
    (Multiply1, [((1, 192, 13, 13), torch.float32), ((192, 1, 1), torch.float32)]),
    (Multiply1, [((1, 192, 13, 13), torch.float32), ((1, 192, 13, 13), torch.float32)]),
    (Multiply1, [((1, 96, 13, 13), torch.float32), ((96, 1, 1), torch.float32)]),
    (Multiply1, [((1, 96, 13, 13), torch.float32), ((1, 96, 13, 13), torch.float32)]),
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
