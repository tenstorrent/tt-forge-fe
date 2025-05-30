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
        self.add_constant("multiply1_const_0", shape=(1,), dtype=torch.float32)

    def forward(self, multiply_input_1):
        multiply_output_1 = forge.op.Multiply("", self.get_constant("multiply1_const_0"), multiply_input_1)
        return multiply_output_1


class Multiply2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply2.weight_1", forge.Parameter(*(16,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply2.weight_1"))
        return multiply_output_1


class Multiply3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, multiply_input_0, multiply_input_1):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, multiply_input_1)
        return multiply_output_1


class Multiply4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply4.weight_0", forge.Parameter(*(1,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, multiply_input_1):
        multiply_output_1 = forge.op.Multiply("", self.get_parameter("multiply4.weight_0"), multiply_input_1)
        return multiply_output_1


class Multiply5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply5.weight_1", forge.Parameter(*(24,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply5.weight_1"))
        return multiply_output_1


class Multiply6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply6.weight_1",
            forge.Parameter(*(1, 64, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply6.weight_1"))
        return multiply_output_1


class Multiply7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply7.weight_1",
            forge.Parameter(*(1, 256, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply7.weight_1"))
        return multiply_output_1


class Multiply8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply8.weight_1",
            forge.Parameter(*(1, 128, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply8.weight_1"))
        return multiply_output_1


class Multiply9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply9.weight_1",
            forge.Parameter(*(1, 512, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply9.weight_1"))
        return multiply_output_1


class Multiply10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply10.weight_1",
            forge.Parameter(*(1, 1024, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply10.weight_1"))
        return multiply_output_1


class Multiply11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply11.weight_1",
            forge.Parameter(*(1, 2048, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply11.weight_1"))
        return multiply_output_1


class Multiply12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply12.weight_1",
            forge.Parameter(*(768,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply12.weight_1"))
        return multiply_output_1


class Multiply13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply13.weight_1", forge.Parameter(*(60,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply13.weight_1"))
        return multiply_output_1


class Multiply14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply14.weight_1",
            forge.Parameter(*(120,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply14.weight_1"))
        return multiply_output_1


class Multiply15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply15.weight_1",
            forge.Parameter(*(480,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply15.weight_1"))
        return multiply_output_1


class Multiply16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply16.weight_1", forge.Parameter(*(64,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply16.weight_1"))
        return multiply_output_1


class Multiply17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply17.weight_1",
            forge.Parameter(*(256,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply17.weight_1"))
        return multiply_output_1


class Multiply18(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply18.weight_1",
            forge.Parameter(*(128,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply18.weight_1"))
        return multiply_output_1


class Multiply19(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply19.weight_1",
            forge.Parameter(*(512,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply19.weight_1"))
        return multiply_output_1


class Multiply20(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply20.weight_1",
            forge.Parameter(*(1024,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply20.weight_1"))
        return multiply_output_1


class Multiply21(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply21_const_1", shape=(1024,), dtype=torch.float32)

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_constant("multiply21_const_1"))
        return multiply_output_1


class Multiply22(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply22_const_1", shape=(256,), dtype=torch.float32)

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_constant("multiply22_const_1"))
        return multiply_output_1


class Multiply23(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply23.weight_1",
            forge.Parameter(*(2048,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply23.weight_1"))
        return multiply_output_1


class Multiply24(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply24_const_0", shape=(1, 1, 256, 256), dtype=torch.float32)

    def forward(self, multiply_input_1):
        multiply_output_1 = forge.op.Multiply("", self.get_constant("multiply24_const_0"), multiply_input_1)
        return multiply_output_1


class Multiply25(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply25_const_1", shape=(1, 256, 1, 32), dtype=torch.float32)

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_constant("multiply25_const_1"))
        return multiply_output_1


class Multiply26(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply26_const_0", shape=(1, 12, 128, 128), dtype=torch.float32)

    def forward(self, multiply_input_1):
        multiply_output_1 = forge.op.Multiply("", self.get_constant("multiply26_const_0"), multiply_input_1)
        return multiply_output_1


class Multiply27(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply27_const_0", shape=(1, 1, 32, 32), dtype=torch.float32)

    def forward(self, multiply_input_1):
        multiply_output_1 = forge.op.Multiply("", self.get_constant("multiply27_const_0"), multiply_input_1)
        return multiply_output_1


class Multiply28(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply28.weight_0",
            forge.Parameter(*(2048,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_1):
        multiply_output_1 = forge.op.Multiply("", self.get_parameter("multiply28.weight_0"), multiply_input_1)
        return multiply_output_1


class Multiply29(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply29_const_0", shape=(16, 1), dtype=torch.float32)

    def forward(self, multiply_input_1):
        multiply_output_1 = forge.op.Multiply("", self.get_constant("multiply29_const_0"), multiply_input_1)
        return multiply_output_1


class Multiply30(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply30_const_0", shape=(2, 1, 7, 7), dtype=torch.float32)

    def forward(self, multiply_input_1):
        multiply_output_1 = forge.op.Multiply("", self.get_constant("multiply30_const_0"), multiply_input_1)
        return multiply_output_1


class Multiply31(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply31_const_0", shape=(1, 12, 384, 384), dtype=torch.float32)

    def forward(self, multiply_input_1):
        multiply_output_1 = forge.op.Multiply("", self.get_constant("multiply31_const_0"), multiply_input_1)
        return multiply_output_1


class Multiply32(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply32_const_0", shape=(1, 1, 7, 7), dtype=torch.float32)

    def forward(self, multiply_input_1):
        multiply_output_1 = forge.op.Multiply("", self.get_constant("multiply32_const_0"), multiply_input_1)
        return multiply_output_1


class Multiply33(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply33_const_0", shape=(1, 1, 14, 20), dtype=torch.float32)

    def forward(self, multiply_input_1):
        multiply_output_1 = forge.op.Multiply("", self.get_constant("multiply33_const_0"), multiply_input_1)
        return multiply_output_1


class Multiply34(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply34_const_1", shape=(8, 1), dtype=torch.float32)

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_constant("multiply34_const_1"))
        return multiply_output_1


class Multiply35(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply35.weight_1",
            forge.Parameter(*(264, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply35.weight_1"))
        return multiply_output_1


class Multiply36(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply36.weight_1",
            forge.Parameter(*(128, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply36.weight_1"))
        return multiply_output_1


class Multiply37(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply37.weight_1",
            forge.Parameter(*(64, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply37.weight_1"))
        return multiply_output_1


class Multiply38(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply38.weight_1",
            forge.Parameter(*(32, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply38.weight_1"))
        return multiply_output_1


class Multiply39(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply39.weight_1",
            forge.Parameter(*(16, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply39.weight_1"))
        return multiply_output_1


class Multiply40(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply40_const_1", shape=(1, 8400), dtype=torch.float32)

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_constant("multiply40_const_1"))
        return multiply_output_1


class Multiply41(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply41.weight_1",
            forge.Parameter(*(312,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply41.weight_1"))
        return multiply_output_1


class Multiply42(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply42.weight_1", forge.Parameter(*(8,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply42.weight_1"))
        return multiply_output_1


class Multiply43(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply43.weight_1", forge.Parameter(*(40,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply43.weight_1"))
        return multiply_output_1


class Multiply44(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply44.weight_1", forge.Parameter(*(48,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply44.weight_1"))
        return multiply_output_1


class Multiply45(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply45.weight_1", forge.Parameter(*(72,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply45.weight_1"))
        return multiply_output_1


class Multiply46(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply46.weight_1",
            forge.Parameter(*(144,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply46.weight_1"))
        return multiply_output_1


class Multiply47(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply47.weight_1",
            forge.Parameter(*(288,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply47.weight_1"))
        return multiply_output_1


class Multiply48(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply48_const_1", shape=(1, 48), dtype=torch.float32)

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_constant("multiply48_const_1"))
        return multiply_output_1


class Multiply49(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply49.weight_1", forge.Parameter(*(96,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply49.weight_1"))
        return multiply_output_1


class Multiply50(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply50.weight_1",
            forge.Parameter(*(160,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply50.weight_1"))
        return multiply_output_1


class Multiply51(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply51.weight_1",
            forge.Parameter(*(192,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply51.weight_1"))
        return multiply_output_1


class Multiply52(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply52.weight_1",
            forge.Parameter(*(224,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply52.weight_1"))
        return multiply_output_1


class Multiply53(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply53.weight_1",
            forge.Parameter(*(320,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply53.weight_1"))
        return multiply_output_1


class Multiply54(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply54.weight_1",
            forge.Parameter(*(352,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply54.weight_1"))
        return multiply_output_1


class Multiply55(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply55.weight_1",
            forge.Parameter(*(384,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply55.weight_1"))
        return multiply_output_1


class Multiply56(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply56.weight_1",
            forge.Parameter(*(416,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply56.weight_1"))
        return multiply_output_1


class Multiply57(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply57.weight_1",
            forge.Parameter(*(448,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply57.weight_1"))
        return multiply_output_1


class Multiply58(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply58.weight_1",
            forge.Parameter(*(544,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply58.weight_1"))
        return multiply_output_1


class Multiply59(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply59.weight_1",
            forge.Parameter(*(576,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply59.weight_1"))
        return multiply_output_1


class Multiply60(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply60.weight_1",
            forge.Parameter(*(608,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply60.weight_1"))
        return multiply_output_1


class Multiply61(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply61.weight_1",
            forge.Parameter(*(640,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply61.weight_1"))
        return multiply_output_1


class Multiply62(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply62.weight_1",
            forge.Parameter(*(672,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply62.weight_1"))
        return multiply_output_1


class Multiply63(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply63.weight_1",
            forge.Parameter(*(704,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply63.weight_1"))
        return multiply_output_1


class Multiply64(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply64.weight_1",
            forge.Parameter(*(736,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply64.weight_1"))
        return multiply_output_1


class Multiply65(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply65.weight_1",
            forge.Parameter(*(800,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply65.weight_1"))
        return multiply_output_1


class Multiply66(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply66.weight_1",
            forge.Parameter(*(832,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply66.weight_1"))
        return multiply_output_1


class Multiply67(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply67.weight_1",
            forge.Parameter(*(864,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply67.weight_1"))
        return multiply_output_1


class Multiply68(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply68.weight_1",
            forge.Parameter(*(896,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply68.weight_1"))
        return multiply_output_1


class Multiply69(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply69.weight_1",
            forge.Parameter(*(928,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply69.weight_1"))
        return multiply_output_1


class Multiply70(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply70.weight_1",
            forge.Parameter(*(960,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply70.weight_1"))
        return multiply_output_1


class Multiply71(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply71.weight_1",
            forge.Parameter(*(992,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply71.weight_1"))
        return multiply_output_1


class Multiply72(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply72.weight_1", forge.Parameter(*(32,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply72.weight_1"))
        return multiply_output_1


class Multiply73(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply73.weight_1",
            forge.Parameter(*(1280,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply73.weight_1"))
        return multiply_output_1


class Multiply74(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply74.weight_0",
            forge.Parameter(*(1024,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_1):
        multiply_output_1 = forge.op.Multiply("", self.get_parameter("multiply74.weight_0"), multiply_input_1)
        return multiply_output_1


class Multiply75(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply75_const_1", shape=(1, 2048, 16), dtype=torch.float32)

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_constant("multiply75_const_1"))
        return multiply_output_1


class Multiply76(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply76.weight_0",
            forge.Parameter(*(768,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_1):
        multiply_output_1 = forge.op.Multiply("", self.get_parameter("multiply76.weight_0"), multiply_input_1)
        return multiply_output_1


class Multiply77(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply77_const_0", shape=(2, 1, 1, 13), dtype=torch.float32)

    def forward(self, multiply_input_1):
        multiply_output_1 = forge.op.Multiply("", self.get_constant("multiply77_const_0"), multiply_input_1)
        return multiply_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Multiply0,
        [((1, 12, 6, 6), torch.float32)],
        {
            "model_names": [
                "onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 1, 1, 6), torch.float32)],
        {
            "model_names": [
                "onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply1,
        [((16,), torch.float32)],
        {
            "model_names": [
                "TranslatedLayer",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply2,
        [((16,), torch.float32)],
        {
            "model_names": [
                "TranslatedLayer",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 16, 240, 240), torch.float32), ((16, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((16,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "TranslatedLayer",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((16,), torch.float32), ((16,), torch.float32)],
        {
            "model_names": [
                "TranslatedLayer",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (Multiply4, [((1, 16, 240, 240), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (Multiply0, [((1, 16, 240, 240), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (
        Multiply3,
        [((1, 16, 240, 240), torch.float32), ((1, 16, 240, 240), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99},
    ),
    (Multiply4, [((1, 32, 240, 240), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (Multiply0, [((1, 32, 240, 240), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (
        Multiply3,
        [((1, 32, 240, 240), torch.float32), ((1, 32, 240, 240), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99},
    ),
    (Multiply4, [((1, 32, 120, 120), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (Multiply4, [((1, 48, 120, 120), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (Multiply0, [((1, 48, 120, 120), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (
        Multiply3,
        [((1, 48, 120, 120), torch.float32), ((1, 48, 120, 120), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99},
    ),
    (Multiply4, [((1, 48, 60, 60), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (Multiply4, [((1, 96, 60, 60), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (Multiply0, [((1, 96, 60, 60), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (
        Multiply3,
        [((1, 96, 60, 60), torch.float32), ((1, 96, 60, 60), torch.float32)],
        {"model_names": ["TranslatedLayer", "onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (Multiply4, [((1, 96, 30, 30), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (Multiply4, [((1, 192, 30, 30), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (Multiply0, [((1, 192, 30, 30), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (
        Multiply3,
        [((1, 192, 30, 30), torch.float32), ((1, 192, 30, 30), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99},
    ),
    (Multiply4, [((1, 192, 15, 15), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (
        Multiply0,
        [((1, 192, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer", "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 192, 15, 15), torch.float32), ((1, 192, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99},
    ),
    (Multiply4, [((1, 384, 15, 15), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (Multiply0, [((1, 384, 15, 15), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (
        Multiply3,
        [((1, 384, 15, 15), torch.float32), ((1, 384, 15, 15), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 384, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer", "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 384, 15, 15), torch.float32), ((1, 384, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 96, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer", "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 96, 15, 15), torch.float32), ((1, 96, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 24, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer", "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 24, 15, 15), torch.float32), ((1, 24, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 96, 30, 30), torch.float32), ((1, 96, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 24, 30, 30), torch.float32), ((1, 24, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 96, 60, 60), torch.float32), ((1, 96, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer", "onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 24, 60, 60), torch.float32), ((1, 24, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 96, 120, 120), torch.float32), ((1, 96, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 24, 120, 120), torch.float32), ((1, 24, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99},
    ),
    (
        Multiply1,
        [((24,), torch.float32)],
        {
            "model_names": [
                "TranslatedLayer",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply5,
        [((24,), torch.float32)],
        {
            "model_names": [
                "TranslatedLayer",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 24, 120, 120), torch.float32), ((24, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((24,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "TranslatedLayer",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((24,), torch.float32), ((24,), torch.float32)],
        {
            "model_names": [
                "TranslatedLayer",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 24, 240, 240), torch.float32), ((24, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 100, 256), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply6,
        [((1, 64, 214, 320), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply6,
        [((1, 64, 107, 160), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply7,
        [((1, 256, 107, 160), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply8,
        [((1, 128, 107, 160), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply8,
        [((1, 128, 54, 80), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply9,
        [((1, 512, 54, 80), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply7,
        [((1, 256, 54, 80), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply7,
        [((1, 256, 27, 40), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply10,
        [((1, 1024, 27, 40), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply9,
        [((1, 512, 27, 40), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply9,
        [((1, 512, 14, 20), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply11,
        [((1, 2048, 14, 20), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 280, 256), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 40, 144, 144), torch.float32), ((1, 40, 144, 144), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 10, 1, 1), torch.float32), ((1, 10, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 40, 144, 144), torch.float32), ((1, 40, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 24, 144, 144), torch.float32), ((1, 24, 144, 144), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 6, 1, 1), torch.float32), ((1, 6, 1, 1), torch.float32)],
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
        },
    ),
    (
        Multiply3,
        [((1, 24, 144, 144), torch.float32), ((1, 24, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 144, 144, 144), torch.float32), ((1, 144, 144, 144), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 144, 72, 72), torch.float32), ((1, 144, 72, 72), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 144, 72, 72), torch.float32), ((1, 144, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 192, 72, 72), torch.float32), ((1, 192, 72, 72), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 8, 1, 1), torch.float32), ((1, 8, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 192, 72, 72), torch.float32), ((1, 192, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 192, 36, 36), torch.float32), ((1, 192, 36, 36), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 192, 36, 36), torch.float32), ((1, 192, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 288, 36, 36), torch.float32), ((1, 288, 36, 36), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 12, 1, 1), torch.float32), ((1, 12, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 288, 36, 36), torch.float32), ((1, 288, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 288, 18, 18), torch.float32), ((1, 288, 18, 18), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 288, 18, 18), torch.float32), ((1, 288, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 576, 18, 18), torch.float32), ((1, 576, 18, 18), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 24, 1, 1), torch.float32), ((1, 24, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 576, 18, 18), torch.float32), ((1, 576, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 816, 18, 18), torch.float32), ((1, 816, 18, 18), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 34, 1, 1), torch.float32), ((1, 34, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 816, 18, 18), torch.float32), ((1, 816, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 816, 9, 9), torch.float32), ((1, 816, 9, 9), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 816, 9, 9), torch.float32), ((1, 816, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 1392, 9, 9), torch.float32), ((1, 1392, 9, 9), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 58, 1, 1), torch.float32), ((1, 58, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 1392, 9, 9), torch.float32), ((1, 1392, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 2304, 9, 9), torch.float32), ((1, 2304, 9, 9), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 96, 1, 1), torch.float32), ((1, 96, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 2304, 9, 9), torch.float32), ((1, 2304, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 1536, 9, 9), torch.float32), ((1, 1536, 9, 9), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 1, 16384, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 2, 4096, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 5, 1024, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 8, 256, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 9, 768), torch.float32), ((1, 9, 768), torch.float32)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_ernie_1_0_mlm_padlenlp",
                "pd_ernie_1_0_qa_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_qa_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 9, 768), torch.float32), ((1, 9, 1), torch.float32)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_ernie_1_0_mlm_padlenlp",
                "pd_ernie_1_0_qa_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_qa_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply12,
        [((1, 9, 768), torch.float32)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_ernie_1_0_mlm_padlenlp",
                "pd_ernie_1_0_qa_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_qa_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 12, 9, 64), torch.float32)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_ernie_1_0_mlm_padlenlp",
                "pd_ernie_1_0_qa_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_qa_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 1, 1, 9), torch.float32)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_ernie_1_0_mlm_padlenlp",
                "pd_ernie_1_0_qa_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_qa_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 12, 9, 9), torch.float32), ((1, 12, 9, 1), torch.float32)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_ernie_1_0_mlm_padlenlp",
                "pd_ernie_1_0_qa_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_qa_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 11, 768), torch.float32), ((1, 11, 768), torch.float32)],
        {
            "model_names": [
                "pd_bert_chinese_roberta_base_qa_padlenlp",
                "pd_roberta_rbt4_ch_clm_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 11, 768), torch.float32), ((1, 11, 1), torch.float32)],
        {
            "model_names": [
                "pd_bert_chinese_roberta_base_qa_padlenlp",
                "pd_roberta_rbt4_ch_clm_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply12,
        [((1, 11, 768), torch.float32)],
        {
            "model_names": [
                "pd_bert_chinese_roberta_base_qa_padlenlp",
                "pd_roberta_rbt4_ch_clm_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 12, 11, 64), torch.float32)],
        {
            "model_names": [
                "pd_bert_chinese_roberta_base_qa_padlenlp",
                "pd_roberta_rbt4_ch_clm_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 1, 1, 11), torch.float32)],
        {
            "model_names": [
                "pd_bert_chinese_roberta_base_qa_padlenlp",
                "pd_albert_chinese_tiny_mlm_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 12, 11, 11), torch.float32), ((1, 12, 11, 1), torch.float32)],
        {
            "model_names": [
                "pd_bert_chinese_roberta_base_qa_padlenlp",
                "pd_roberta_rbt4_ch_clm_padlenlp",
                "pd_albert_chinese_tiny_mlm_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 16, 16, 50), torch.float32), ((16, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply4,
        [((1, 16, 16, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 16, 16, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 16, 16, 50), torch.float32), ((1, 16, 16, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply4,
        [((1, 32, 16, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 32, 16, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 32, 16, 50), torch.float32), ((1, 32, 16, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply4,
        [((1, 64, 16, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 64, 16, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 64, 16, 50), torch.float32), ((1, 64, 16, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply4,
        [((1, 64, 8, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 64, 8, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 64, 8, 50), torch.float32), ((1, 64, 8, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply4,
        [((1, 128, 8, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 128, 8, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 128, 8, 50), torch.float32), ((1, 128, 8, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply4,
        [((1, 128, 8, 25), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 128, 8, 25), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 128, 8, 25), torch.float32), ((1, 128, 8, 25), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply4,
        [((1, 240, 8, 25), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 240, 8, 25), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 240, 8, 25), torch.float32), ((1, 240, 8, 25), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply4,
        [((1, 240, 4, 25), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 240, 4, 25), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 240, 4, 25), torch.float32), ((1, 240, 4, 25), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 240, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 240, 4, 25), torch.float32), ((1, 240, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply4,
        [((1, 480, 4, 25), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 480, 4, 25), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 480, 4, 25), torch.float32), ((1, 480, 4, 25), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 480, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 480, 4, 25), torch.float32), ((1, 480, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply4,
        [((1, 480, 2, 25), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 480, 2, 25), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 480, 2, 25), torch.float32), ((1, 480, 2, 25), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply1,
        [((60,), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply13,
        [((60,), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 60, 1, 12), torch.float32), ((60, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((60,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((60,), torch.float32), ((60,), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 60, 1, 12), torch.float32), ((1, 60, 1, 12), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply1,
        [((120,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply14,
        [((120,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 120, 1, 12), torch.float32), ((120, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((120,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((120,), torch.float32), ((120,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 120, 1, 12), torch.float32), ((1, 120, 1, 12), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 12, 120), torch.float32), ((1, 12, 120), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 12, 120), torch.float32), ((1, 12, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply14,
        [((1, 12, 120), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 8, 12, 15), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 8, 12, 12), torch.float32), ((1, 8, 12, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 12, 240), torch.float32), ((1, 12, 240), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply1,
        [((480,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply15,
        [((480,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 480, 1, 12), torch.float32), ((480, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((480,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((480,), torch.float32), ((480,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 480, 1, 12), torch.float32), ((1, 480, 1, 12), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 12, 97), torch.float32), ((1, 12, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply1,
        [((64,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_resnet_18_img_cls_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply16,
        [((64,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_resnet_18_img_cls_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 64, 112, 112), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_resnet_18_img_cls_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((64,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_resnet_18_img_cls_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((64,), torch.float32), ((64,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_resnet_18_img_cls_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 64, 56, 56), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_resnet_18_img_cls_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply1,
        [((256,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_resnet_18_img_cls_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply17,
        [((256,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_resnet_18_img_cls_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 256, 56, 56), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((256,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_resnet_18_img_cls_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((256,), torch.float32), ((256,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_resnet_18_img_cls_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply1,
        [((128,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_resnet_18_img_cls_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply18,
        [((128,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_resnet_18_img_cls_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 128, 56, 56), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((128,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_resnet_18_img_cls_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((128,), torch.float32), ((128,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_resnet_18_img_cls_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 128, 28, 28), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_resnet_18_img_cls_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply1,
        [((512,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_resnet_18_img_cls_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply19,
        [((512,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_resnet_18_img_cls_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 512, 28, 28), torch.float32), ((512, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((512,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_resnet_18_img_cls_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((512,), torch.float32), ((512,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_resnet_18_img_cls_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 256, 28, 28), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 256, 14, 14), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_resnet_18_img_cls_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply1,
        [((1024,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply20,
        [((1024,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 1024, 14, 14), torch.float32), ((1024, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1024,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1024,), torch.float32), ((1024,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (Multiply21, [((1024,), torch.float32)], {"model_names": ["pd_resnet_101_img_cls_paddlemodels"], "pcc": 0.99}),
    (Multiply22, [((256,), torch.float32)], {"model_names": ["pd_resnet_101_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Multiply3,
        [((1, 512, 14, 14), torch.float32), ((512, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 512, 7, 7), torch.float32), ((512, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_resnet_18_img_cls_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply1,
        [((2048,), torch.float32)],
        {"model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply23,
        [((2048,), torch.float32)],
        {"model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 2048, 7, 7), torch.float32), ((2048, 1, 1), torch.float32)],
        {"model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((2048,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((2048,), torch.float32), ((2048,), torch.float32)],
        {"model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 12, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_albert_base_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "onnx_bert_bert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 1, 1, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 16, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 64, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 256, 1024), torch.float32)],
        {"model_names": ["pt_bart_facebook_bart_large_mnli_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 1, 256, 256), torch.float32), ((1, 1, 256, 256), torch.float32)],
        {
            "model_names": ["pt_bart_facebook_bart_large_mnli_seq_cls_hf", "pt_opt_facebook_opt_1_3b_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Multiply24,
        [((1, 1, 256, 256), torch.float32)],
        {
            "model_names": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_phi2_microsoft_phi_2_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply25,
        [((1, 256, 16, 32), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 256, 16, 16), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 16, 256, 256), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 12, 128, 64), torch.float32)],
        {
            "model_names": [
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 12, 128, 128), torch.float32), ((1, 12, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply26,
        [((1, 12, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 1, 32, 32), torch.float32)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply27,
        [((1, 1, 32, 32), torch.float32)],
        {
            "model_names": [
                "pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_bloom_bigscience_bloom_1b1_clm_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 256, 2048), torch.float32), ((1, 256, 2048), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 256, 2048), torch.float32), ((1, 256, 1), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply28,
        [((1, 256, 2048), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 32, 256, 64), torch.float32), ((1, 1, 256, 64), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 32, 256, 32), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 8, 256, 64), torch.float32), ((1, 1, 256, 64), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 8, 256, 32), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 32, 256, 256), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_clm_hf", "pt_phi2_microsoft_phi_2_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 256, 8192), torch.float32), ((1, 256, 8192), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 32), torch.int64), ((1, 32), torch.int64)],
        {
            "model_names": [
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_bloom_bigscience_bloom_1b1_clm_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (Multiply0, [((1, 32, 768), torch.float32)], {"model_names": ["pt_opt_facebook_opt_125m_qa_hf"], "pcc": 0.99}),
    (
        Multiply3,
        [((1, 1, 32, 32), torch.float32), ((1, 1, 32, 32), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_qa_hf", "pt_opt_facebook_opt_350m_qa_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 32, 256, 32), torch.float32), ((1, 1, 256, 32), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_clm_hf"], "pcc": 0.99},
    ),
    (Multiply0, [((1, 32, 256, 16), torch.float32)], {"model_names": ["pt_phi2_microsoft_phi_2_clm_hf"], "pcc": 0.99}),
    (
        Multiply0,
        [((1, 12, 204, 204), torch.float32)],
        {"model_names": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 1, 1, 204), torch.float32)],
        {"model_names": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 1, 768), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 1500, 768), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 16, 384, 384), torch.float32)],
        {
            "model_names": [
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 32, 112, 112), torch.float32), ((1, 32, 112, 112), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 32, 112, 112), torch.float32), ((1, 32, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 96, 112, 112), torch.float32), ((1, 96, 112, 112), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 96, 56, 56), torch.float32), ((1, 96, 56, 56), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 4, 1, 1), torch.float32), ((1, 4, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 96, 56, 56), torch.float32), ((1, 96, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 144, 56, 56), torch.float32), ((1, 144, 56, 56), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 144, 56, 56), torch.float32), ((1, 144, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 144, 28, 28), torch.float32), ((1, 144, 28, 28), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 144, 28, 28), torch.float32), ((1, 144, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 240, 28, 28), torch.float32), ((1, 240, 28, 28), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 240, 28, 28), torch.float32), ((1, 240, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 240, 14, 14), torch.float32), ((1, 240, 14, 14), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 240, 14, 14), torch.float32), ((1, 240, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 480, 14, 14), torch.float32), ((1, 480, 14, 14), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 20, 1, 1), torch.float32), ((1, 20, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 480, 14, 14), torch.float32), ((1, 480, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 672, 14, 14), torch.float32), ((1, 672, 14, 14), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 28, 1, 1), torch.float32), ((1, 28, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 672, 14, 14), torch.float32), ((1, 672, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 672, 7, 7), torch.float32), ((1, 672, 7, 7), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 672, 7, 7), torch.float32), ((1, 672, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 1152, 7, 7), torch.float32), ((1, 1152, 7, 7), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 48, 1, 1), torch.float32), ((1, 48, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 1152, 7, 7), torch.float32), ((1, 1152, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 1280, 7, 7), torch.float32), ((1, 1280, 7, 7), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 12, 13, 13), torch.float32)],
        {"model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 10, 768), torch.float32), ((1, 10, 768), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 10, 768), torch.float32), ((1, 10, 1), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Multiply12,
        [((1, 10, 768), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 12, 10, 64), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 1, 1, 10), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 12, 10, 10), torch.float32), ((1, 12, 10, 1), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 8, 768), torch.float32), ((1, 8, 768), torch.float32)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 8, 768), torch.float32), ((1, 8, 1), torch.float32)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply12,
        [((1, 8, 768), torch.float32)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 12, 8, 64), torch.float32)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 1, 1, 8), torch.float32)],
        {"model_names": ["pd_bert_bert_base_uncased_seq_cls_padlenlp"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 12, 8, 8), torch.float32), ((1, 12, 8, 1), torch.float32)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 16, 224, 224), torch.float32), ((16, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply4,
        [((1, 16, 224, 224), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 16, 224, 224), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 16, 224, 224), torch.float32), ((1, 16, 224, 224), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply4,
        [((1, 32, 224, 224), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 32, 224, 224), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 32, 224, 224), torch.float32), ((1, 32, 224, 224), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply4,
        [((1, 32, 112, 112), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply4,
        [((1, 48, 112, 112), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 48, 112, 112), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 48, 112, 112), torch.float32), ((1, 48, 112, 112), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply4,
        [((1, 48, 56, 56), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply4,
        [((1, 96, 56, 56), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 96, 56, 56), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply4,
        [((1, 96, 28, 28), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply4,
        [((1, 192, 28, 28), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 192, 28, 28), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 192, 28, 28), torch.float32), ((1, 192, 28, 28), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply4,
        [((1, 192, 14, 14), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 192, 14, 14), torch.float32), ((1, 192, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply4,
        [((1, 384, 14, 14), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 384, 14, 14), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 384, 14, 14), torch.float32), ((1, 384, 14, 14), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 384, 14, 14), torch.float32), ((1, 384, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 96, 14, 14), torch.float32), ((1, 96, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 24, 14, 14), torch.float32), ((1, 24, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 96, 28, 28), torch.float32), ((1, 96, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 24, 28, 28), torch.float32), ((1, 24, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 24, 56, 56), torch.float32), ((1, 24, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 96, 112, 112), torch.float32), ((1, 96, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 24, 112, 112), torch.float32), ((1, 24, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 24, 112, 112), torch.float32), ((24, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 24, 224, 224), torch.float32), ((24, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 11), torch.int64), ((1, 11), torch.int64)],
        {"model_names": ["pd_roberta_rbt4_ch_clm_padlenlp"], "pcc": 0.99},
    ),
    (Multiply0, [((1, 11), torch.float32)], {"model_names": ["pd_roberta_rbt4_ch_clm_padlenlp"], "pcc": 0.99}),
    (
        Multiply1,
        [((16, 32, 32), torch.float32)],
        {"model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99},
    ),
    (Multiply29, [((1, 1, 32), torch.float32)], {"model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99}),
    (
        Multiply0,
        [((1, 32, 6144), torch.float32)],
        {"model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 32, 6144), torch.float32), ((1, 32, 6144), torch.float32)],
        {"model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((2, 7, 512), torch.float32)],
        {"model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((2, 1, 7, 7), torch.float32), ((2, 1, 7, 7), torch.float32)],
        {"model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99},
    ),
    (
        Multiply30,
        [((2, 1, 7, 7), torch.float32)],
        {"model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((2, 7, 2048), torch.float32)],
        {"model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((2, 7, 2048), torch.float32), ((2, 7, 2048), torch.float32)],
        {"model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 12, 384, 64), torch.float32)],
        {"model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 12, 384, 384), torch.float32), ((1, 12, 384, 384), torch.float32)],
        {"model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"], "pcc": 0.99},
    ),
    (
        Multiply31,
        [((1, 12, 384, 384), torch.float32)],
        {"model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 12, 7, 7), torch.float32)],
        {
            "model_names": [
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 1, 7, 7), torch.float32)],
        {
            "model_names": [
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 1, 1, 7), torch.float32)],
        {
            "model_names": [
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 4, 2048), torch.float32), ((1, 4, 2048), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 4, 2048), torch.float32), ((1, 4, 1), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply28,
        [((1, 4, 2048), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 32, 4, 64), torch.float32), ((1, 1, 4, 64), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 32, 4, 32), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 8, 4, 64), torch.float32), ((1, 1, 4, 64), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 8, 4, 32), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 32, 4, 4), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 4, 8192), torch.float32), ((1, 4, 8192), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 256), torch.int64), ((1, 256), torch.int64)],
        {"model_names": ["pt_opt_facebook_opt_1_3b_clm_hf"], "pcc": 0.99},
    ),
    (Multiply0, [((1, 256, 2048), torch.float32)], {"model_names": ["pt_opt_facebook_opt_1_3b_clm_hf"], "pcc": 0.99}),
    (Multiply0, [((1, 32, 1024), torch.float32)], {"model_names": ["pt_opt_facebook_opt_350m_qa_hf"], "pcc": 0.99}),
    (
        Multiply3,
        [((1, 32, 7, 32), torch.float32), ((1, 1, 7, 32), torch.float32)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 32, 7, 16), torch.float32)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 32, 7, 7), torch.float32)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply32,
        [((1, 1, 7, 7), torch.float32)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 12, 201, 201), torch.float32)],
        {"model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 1, 1, 201), torch.float32)],
        {"model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 1, 384), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 1500, 384), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 1, 1, 64), torch.float32), ((1, 1, 1, 64), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 112, 112, 64), torch.float32), ((1, 1, 1, 64), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 56, 55, 64), torch.float32), ((1, 1, 1, 64), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 1, 1, 256), torch.float32), ((1, 1, 1, 256), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 56, 55, 256), torch.float32), ((1, 1, 1, 256), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 1, 1, 128), torch.float32), ((1, 1, 1, 128), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 56, 55, 128), torch.float32), ((1, 1, 1, 128), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 28, 28, 128), torch.float32), ((1, 1, 1, 128), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 1, 1, 512), torch.float32), ((1, 1, 1, 512), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 28, 28, 512), torch.float32), ((1, 1, 1, 512), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 28, 28, 256), torch.float32), ((1, 1, 1, 256), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 14, 14, 256), torch.float32), ((1, 1, 1, 256), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 1, 1, 1024), torch.float32), ((1, 1, 1, 1024), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 14, 14, 1024), torch.float32), ((1, 1, 1, 1024), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 14, 14, 512), torch.float32), ((1, 1, 1, 512), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 7, 7, 512), torch.float32), ((1, 1, 1, 512), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 1, 1, 2048), torch.float32), ((1, 1, 1, 2048), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 7, 7, 2048), torch.float32), ((1, 1, 1, 2048), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (Multiply0, [((1, 1, 1, 2048), torch.float32)], {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99}),
    (
        Multiply0,
        [((1, 100, 8, 32), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 100, 8, 32, 1), torch.float32), ((1, 1, 8, 32, 280), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Multiply33,
        [((1, 100, 8, 14, 20), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((100, 8, 9240), torch.float32), ((100, 8, 9240), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((100, 8, 9240), torch.float32), ((100, 8, 1), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Multiply34,
        [((100, 8, 9240), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Multiply35,
        [((100, 264, 14, 20), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((100, 8, 4480), torch.float32), ((100, 8, 4480), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((100, 8, 4480), torch.float32), ((100, 8, 1), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Multiply34,
        [((100, 8, 4480), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Multiply36,
        [((100, 128, 14, 20), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((100, 8, 8640), torch.float32), ((100, 8, 8640), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((100, 8, 8640), torch.float32), ((100, 8, 1), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Multiply34,
        [((100, 8, 8640), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Multiply37,
        [((100, 64, 27, 40), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((100, 8, 17280), torch.float32), ((100, 8, 17280), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((100, 8, 17280), torch.float32), ((100, 8, 1), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Multiply34,
        [((100, 8, 17280), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Multiply38,
        [((100, 32, 54, 80), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((100, 8, 34240), torch.float32), ((100, 8, 34240), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((100, 8, 34240), torch.float32), ((100, 8, 1), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Multiply34,
        [((100, 8, 34240), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Multiply39,
        [((100, 16, 107, 160), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 32, 120, 120), torch.float32), ((1, 32, 120, 120), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 32, 120, 120), torch.float32), ((1, 32, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 16, 120, 120), torch.float32), ((1, 16, 120, 120), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 16, 120, 120), torch.float32), ((1, 16, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 96, 120, 120), torch.float32), ((1, 96, 120, 120), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 144, 60, 60), torch.float32), ((1, 144, 60, 60), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 144, 60, 60), torch.float32), ((1, 144, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 144, 30, 30), torch.float32), ((1, 144, 30, 30), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 144, 30, 30), torch.float32), ((1, 144, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 240, 30, 30), torch.float32), ((1, 240, 30, 30), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 240, 30, 30), torch.float32), ((1, 240, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 240, 15, 15), torch.float32), ((1, 240, 15, 15), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 240, 15, 15), torch.float32), ((1, 240, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 480, 15, 15), torch.float32), ((1, 480, 15, 15), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 480, 15, 15), torch.float32), ((1, 480, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 672, 15, 15), torch.float32), ((1, 672, 15, 15), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 672, 15, 15), torch.float32), ((1, 672, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 672, 8, 8), torch.float32), ((1, 672, 8, 8), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 672, 8, 8), torch.float32), ((1, 672, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 1152, 8, 8), torch.float32), ((1, 1152, 8, 8), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 1152, 8, 8), torch.float32), ((1, 1152, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 1920, 8, 8), torch.float32), ((1, 1920, 8, 8), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 80, 1, 1), torch.float32), ((1, 80, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 1920, 8, 8), torch.float32), ((1, 1920, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 1280, 8, 8), torch.float32), ((1, 1280, 8, 8), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 48, 160, 160), torch.float32), ((1, 48, 160, 160), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 48, 160, 160), torch.float32), ((1, 48, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 24, 160, 160), torch.float32), ((1, 24, 160, 160), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 24, 160, 160), torch.float32), ((1, 24, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 144, 160, 160), torch.float32), ((1, 144, 160, 160), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 144, 80, 80), torch.float32), ((1, 144, 80, 80), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 144, 80, 80), torch.float32), ((1, 144, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 192, 80, 80), torch.float32), ((1, 192, 80, 80), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 192, 80, 80), torch.float32), ((1, 192, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 192, 40, 40), torch.float32), ((1, 192, 40, 40), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 192, 40, 40), torch.float32), ((1, 192, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 336, 40, 40), torch.float32), ((1, 336, 40, 40), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 14, 1, 1), torch.float32), ((1, 14, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 336, 40, 40), torch.float32), ((1, 336, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 336, 20, 20), torch.float32), ((1, 336, 20, 20), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 336, 20, 20), torch.float32), ((1, 336, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 672, 20, 20), torch.float32), ((1, 672, 20, 20), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 672, 20, 20), torch.float32), ((1, 672, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 960, 20, 20), torch.float32), ((1, 960, 20, 20), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 40, 1, 1), torch.float32), ((1, 40, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 960, 20, 20), torch.float32), ((1, 960, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 960, 10, 10), torch.float32), ((1, 960, 10, 10), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 960, 10, 10), torch.float32), ((1, 960, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 1632, 10, 10), torch.float32), ((1, 1632, 10, 10), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 68, 1, 1), torch.float32), ((1, 68, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 1632, 10, 10), torch.float32), ((1, 1632, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 2688, 10, 10), torch.float32), ((1, 2688, 10, 10), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 112, 1, 1), torch.float32), ((1, 112, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 2688, 10, 10), torch.float32), ((1, 2688, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 1792, 10, 10), torch.float32), ((1, 1792, 10, 10), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 12, 197, 197), torch.float32)],
        {"model_names": ["onnx_vit_base_google_vit_base_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 16, 320, 320), torch.float32), ((1, 16, 320, 320), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 32, 160, 160), torch.float32), ((1, 32, 160, 160), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 16, 160, 160), torch.float32), ((1, 16, 160, 160), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 64, 80, 80), torch.float32), ((1, 64, 80, 80), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 32, 80, 80), torch.float32), ((1, 32, 80, 80), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 128, 40, 40), torch.float32), ((1, 128, 40, 40), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 64, 40, 40), torch.float32), ((1, 64, 40, 40), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 256, 20, 20), torch.float32), ((1, 256, 20, 20), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 128, 20, 20), torch.float32), ((1, 128, 20, 20), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 80, 80, 80), torch.float32), ((1, 80, 80, 80), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 80, 40, 40), torch.float32), ((1, 80, 40, 40), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 64, 20, 20), torch.float32), ((1, 64, 20, 20), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 80, 20, 20), torch.float32), ((1, 80, 20, 20), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (Multiply0, [((1, 2, 8400), torch.float32)], {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99}),
    (Multiply40, [((1, 4, 8400), torch.float32)], {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99}),
    (
        Multiply3,
        [((1, 11, 128), torch.float32), ((1, 11, 128), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 11, 128), torch.float32), ((1, 11, 1), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Multiply18,
        [((1, 11, 128), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 12, 11, 11), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 11, 312), torch.float32), ((1, 11, 312), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 11, 312), torch.float32), ((1, 11, 1), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Multiply41,
        [((1, 11, 312), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 15, 768), torch.float32), ((1, 15, 768), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 15, 768), torch.float32), ((1, 15, 1), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"], "pcc": 0.99},
    ),
    (
        Multiply12,
        [((1, 15, 768), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 12, 15, 64), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 1, 1, 15), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 12, 15, 15), torch.float32), ((1, 12, 15, 1), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"], "pcc": 0.99},
    ),
    (
        Multiply1,
        [((8,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply42,
        [((8,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 8, 16, 50), torch.float32), ((8, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((8,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((8,), torch.float32), ((8,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 8, 16, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 8, 16, 50), torch.float32), ((1, 8, 16, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 8, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 8, 16, 50), torch.float32), ((1, 8, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply1,
        [((40,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply43,
        [((40,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 40, 16, 50), torch.float32), ((40, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((40,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((40,), torch.float32), ((40,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 40, 8, 50), torch.float32), ((40, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 16, 8, 50), torch.float32), ((16, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply1,
        [((48,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply44,
        [((48,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 48, 8, 50), torch.float32), ((48, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((48,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((48,), torch.float32), ((48,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 48, 8, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 48, 8, 50), torch.float32), ((1, 48, 8, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 48, 4, 50), torch.float32), ((48, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 48, 4, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 48, 4, 50), torch.float32), ((1, 48, 4, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 48, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 48, 4, 50), torch.float32), ((1, 48, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 24, 4, 50), torch.float32), ((24, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 120, 4, 50), torch.float32), ((120, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 120, 4, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 120, 4, 50), torch.float32), ((1, 120, 4, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 120, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 120, 4, 50), torch.float32), ((1, 120, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 64, 4, 50), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 64, 4, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 64, 4, 50), torch.float32), ((1, 64, 4, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 64, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 64, 4, 50), torch.float32), ((1, 64, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply1,
        [((72,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply45,
        [((72,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 72, 4, 50), torch.float32), ((72, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((72,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((72,), torch.float32), ((72,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 72, 4, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 72, 4, 50), torch.float32), ((1, 72, 4, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 72, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 72, 4, 50), torch.float32), ((1, 72, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply1,
        [((144,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply46,
        [((144,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 144, 4, 50), torch.float32), ((144, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((144,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((144,), torch.float32), ((144,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 144, 4, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 144, 4, 50), torch.float32), ((1, 144, 4, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 144, 2, 50), torch.float32), ((144, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 144, 2, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 144, 2, 50), torch.float32), ((1, 144, 2, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 144, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 144, 2, 50), torch.float32), ((1, 144, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 48, 2, 50), torch.float32), ((48, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply1,
        [((288,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply47,
        [((288,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 288, 2, 50), torch.float32), ((288, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((288,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((288,), torch.float32), ((288,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 288, 2, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 288, 2, 50), torch.float32), ((1, 288, 2, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 288, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 288, 2, 50), torch.float32), ((1, 288, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply48,
        [((1, 48), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 48), torch.float32), ((1, 48), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 25, 6625), torch.float32), ((1, 25, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 2, 1280), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf", "onnx_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 1500, 1280), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf", "onnx_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 32, 128, 128), torch.float32), ((1, 32, 128, 128), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 32, 128, 128), torch.float32), ((1, 32, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 16, 128, 128), torch.float32), ((1, 16, 128, 128), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 16, 128, 128), torch.float32), ((1, 16, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 96, 128, 128), torch.float32), ((1, 96, 128, 128), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 96, 64, 64), torch.float32), ((1, 96, 64, 64), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 96, 64, 64), torch.float32), ((1, 96, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 144, 64, 64), torch.float32), ((1, 144, 64, 64), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 144, 64, 64), torch.float32), ((1, 144, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 144, 32, 32), torch.float32), ((1, 144, 32, 32), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 144, 32, 32), torch.float32), ((1, 144, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 288, 32, 32), torch.float32), ((1, 288, 32, 32), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 288, 32, 32), torch.float32), ((1, 288, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 288, 16, 16), torch.float32), ((1, 288, 16, 16), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 288, 16, 16), torch.float32), ((1, 288, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 528, 16, 16), torch.float32), ((1, 528, 16, 16), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 22, 1, 1), torch.float32), ((1, 22, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 528, 16, 16), torch.float32), ((1, 528, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 720, 16, 16), torch.float32), ((1, 720, 16, 16), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 30, 1, 1), torch.float32), ((1, 30, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 720, 16, 16), torch.float32), ((1, 720, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 720, 8, 8), torch.float32), ((1, 720, 8, 8), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 720, 8, 8), torch.float32), ((1, 720, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 1248, 8, 8), torch.float32), ((1, 1248, 8, 8), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 52, 1, 1), torch.float32), ((1, 52, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 1248, 8, 8), torch.float32), ((1, 1248, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 2112, 8, 8), torch.float32), ((1, 2112, 8, 8), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 88, 1, 1), torch.float32), ((1, 88, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 2112, 8, 8), torch.float32), ((1, 2112, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 1408, 8, 8), torch.float32), ((1, 1408, 8, 8), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 48, 224, 224), torch.float32), ((1, 48, 224, 224), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 48, 224, 224), torch.float32), ((1, 48, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 24, 224, 224), torch.float32), ((1, 24, 224, 224), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 24, 224, 224), torch.float32), ((1, 24, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 144, 224, 224), torch.float32), ((1, 144, 224, 224), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 144, 112, 112), torch.float32), ((1, 144, 112, 112), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 144, 112, 112), torch.float32), ((1, 144, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 240, 112, 112), torch.float32), ((1, 240, 112, 112), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 240, 112, 112), torch.float32), ((1, 240, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 240, 56, 56), torch.float32), ((1, 240, 56, 56), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 240, 56, 56), torch.float32), ((1, 240, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 384, 56, 56), torch.float32), ((1, 384, 56, 56), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 16, 1, 1), torch.float32), ((1, 16, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 384, 56, 56), torch.float32), ((1, 384, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 384, 28, 28), torch.float32), ((1, 384, 28, 28), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 384, 28, 28), torch.float32), ((1, 384, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 768, 28, 28), torch.float32), ((1, 768, 28, 28), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 32, 1, 1), torch.float32), ((1, 32, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 768, 28, 28), torch.float32), ((1, 768, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 1056, 28, 28), torch.float32), ((1, 1056, 28, 28), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 44, 1, 1), torch.float32), ((1, 44, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 1056, 28, 28), torch.float32), ((1, 1056, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 1056, 14, 14), torch.float32), ((1, 1056, 14, 14), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 1056, 14, 14), torch.float32), ((1, 1056, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 1824, 14, 14), torch.float32), ((1, 1824, 14, 14), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 76, 1, 1), torch.float32), ((1, 76, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 1824, 14, 14), torch.float32), ((1, 1824, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 3072, 14, 14), torch.float32), ((1, 3072, 14, 14), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 128, 1, 1), torch.float32), ((1, 128, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 3072, 14, 14), torch.float32), ((1, 3072, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 2048, 14, 14), torch.float32), ((1, 2048, 14, 14), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((64, 3, 64, 32), torch.float32), ((64, 3, 64, 32), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((64, 3, 64, 64), torch.float32), ((3, 1, 1), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((3, 64, 64), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((16, 6, 64, 32), torch.float32), ((16, 6, 64, 32), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((16, 6, 64, 64), torch.float32), ((6, 1, 1), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((6, 64, 64), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((4, 12, 64, 32), torch.float32), ((4, 12, 64, 32), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((4, 12, 64, 64), torch.float32), ((12, 1, 1), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((12, 64, 64), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 24, 64, 32), torch.float32), ((1, 24, 64, 32), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 24, 64, 64), torch.float32), ((24, 1, 1), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((24, 64, 64), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (Multiply0, [((1, 9216), torch.float32)], {"model_names": ["pd_alexnet_base_img_cls_paddlemodels"], "pcc": 0.99}),
    (Multiply0, [((1, 4096), torch.float32)], {"model_names": ["pd_alexnet_base_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Multiply1,
        [((96,), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Multiply49,
        [((96,), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 96, 56, 56), torch.float32), ((96, 1, 1), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((96,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((96,), torch.float32), ((96,), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Multiply1,
        [((160,), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Multiply50,
        [((160,), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 160, 56, 56), torch.float32), ((160, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((160,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((160,), torch.float32), ((160,), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Multiply1,
        [((192,), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Multiply51,
        [((192,), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 192, 56, 56), torch.float32), ((192, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((192,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((192,), torch.float32), ((192,), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (Multiply1, [((224,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (Multiply52, [((224,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Multiply3,
        [((1, 224, 56, 56), torch.float32), ((224, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((224,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((224,), torch.float32), ((224,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 160, 28, 28), torch.float32), ((160, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 192, 28, 28), torch.float32), ((192, 1, 1), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 224, 28, 28), torch.float32), ((224, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 288, 28, 28), torch.float32), ((288, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply1,
        [((320,), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Multiply53,
        [((320,), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 320, 28, 28), torch.float32), ((320, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((320,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((320,), torch.float32), ((320,), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (Multiply1, [((352,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (Multiply54, [((352,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Multiply3,
        [((1, 352, 28, 28), torch.float32), ((352, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((352,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((352,), torch.float32), ((352,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply1,
        [((384,), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Multiply55,
        [((384,), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 384, 28, 28), torch.float32), ((384, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((384,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((384,), torch.float32), ((384,), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (Multiply1, [((416,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (Multiply56, [((416,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Multiply3,
        [((1, 416, 28, 28), torch.float32), ((416, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((416,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((416,), torch.float32), ((416,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (Multiply1, [((448,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (Multiply57, [((448,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Multiply3,
        [((1, 448, 28, 28), torch.float32), ((448, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((448,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((448,), torch.float32), ((448,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 480, 28, 28), torch.float32), ((480, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 128, 14, 14), torch.float32), ((128, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 288, 14, 14), torch.float32), ((288, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 320, 14, 14), torch.float32), ((320, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 352, 14, 14), torch.float32), ((352, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 384, 14, 14), torch.float32), ((384, 1, 1), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 416, 14, 14), torch.float32), ((416, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 448, 14, 14), torch.float32), ((448, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 480, 14, 14), torch.float32), ((480, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (Multiply1, [((544,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (Multiply58, [((544,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Multiply3,
        [((1, 544, 14, 14), torch.float32), ((544, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((544,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((544,), torch.float32), ((544,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply1,
        [((576,), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Multiply59,
        [((576,), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 576, 14, 14), torch.float32), ((576, 1, 1), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((576,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((576,), torch.float32), ((576,), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (Multiply1, [((608,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (Multiply60, [((608,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Multiply3,
        [((1, 608, 14, 14), torch.float32), ((608, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((608,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((608,), torch.float32), ((608,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (Multiply1, [((640,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (Multiply61, [((640,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Multiply3,
        [((1, 640, 14, 14), torch.float32), ((640, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((640,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((640,), torch.float32), ((640,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (Multiply1, [((672,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (Multiply62, [((672,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Multiply3,
        [((1, 672, 14, 14), torch.float32), ((672, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((672,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((672,), torch.float32), ((672,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (Multiply1, [((704,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (Multiply63, [((704,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Multiply3,
        [((1, 704, 14, 14), torch.float32), ((704, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((704,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((704,), torch.float32), ((704,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (Multiply1, [((736,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (Multiply64, [((736,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Multiply3,
        [((1, 736, 14, 14), torch.float32), ((736, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((736,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((736,), torch.float32), ((736,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (Multiply1, [((768,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (Multiply12, [((768,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Multiply3,
        [((1, 768, 14, 14), torch.float32), ((768, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((768,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((768,), torch.float32), ((768,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (Multiply1, [((800,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (Multiply65, [((800,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Multiply3,
        [((1, 800, 14, 14), torch.float32), ((800, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((800,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((800,), torch.float32), ((800,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (Multiply1, [((832,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (Multiply66, [((832,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Multiply3,
        [((1, 832, 14, 14), torch.float32), ((832, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((832,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((832,), torch.float32), ((832,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (Multiply1, [((864,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (Multiply67, [((864,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Multiply3,
        [((1, 864, 14, 14), torch.float32), ((864, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((864,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((864,), torch.float32), ((864,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (Multiply1, [((896,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (Multiply68, [((896,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Multiply3,
        [((1, 896, 14, 14), torch.float32), ((896, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((896,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((896,), torch.float32), ((896,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (Multiply1, [((928,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (Multiply69, [((928,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Multiply3,
        [((1, 928, 14, 14), torch.float32), ((928, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((928,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((928,), torch.float32), ((928,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply1,
        [((960,), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Multiply70,
        [((960,), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 960, 14, 14), torch.float32), ((960, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((960,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((960,), torch.float32), ((960,), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (Multiply1, [((992,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (Multiply71, [((992,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Multiply3,
        [((1, 992, 14, 14), torch.float32), ((992, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((992,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((992,), torch.float32), ((992,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 128, 7, 7), torch.float32), ((128, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 544, 7, 7), torch.float32), ((544, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 576, 7, 7), torch.float32), ((576, 1, 1), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 608, 7, 7), torch.float32), ((608, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 640, 7, 7), torch.float32), ((640, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 672, 7, 7), torch.float32), ((672, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 704, 7, 7), torch.float32), ((704, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 736, 7, 7), torch.float32), ((736, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 768, 7, 7), torch.float32), ((768, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 800, 7, 7), torch.float32), ((800, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 832, 7, 7), torch.float32), ((832, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 864, 7, 7), torch.float32), ((864, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 896, 7, 7), torch.float32), ((896, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 928, 7, 7), torch.float32), ((928, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 960, 7, 7), torch.float32), ((960, 1, 1), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 992, 7, 7), torch.float32), ((992, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 1024, 7, 7), torch.float32), ((1024, 1, 1), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv1_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Multiply1,
        [((32,), torch.float32)],
        {
            "model_names": ["pd_mobilenetv1_basic_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Multiply72,
        [((32,), torch.float32)],
        {
            "model_names": ["pd_mobilenetv1_basic_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 32, 112, 112), torch.float32), ((32, 1, 1), torch.float32)],
        {
            "model_names": ["pd_mobilenetv1_basic_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((32,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": ["pd_mobilenetv1_basic_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((32,), torch.float32), ((32,), torch.float32)],
        {
            "model_names": ["pd_mobilenetv1_basic_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Multiply3,
        [((1, 16, 112, 112), torch.float32), ((16, 1, 1), torch.float32)],
        {"model_names": ["pd_mobilenetv2_basic_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 96, 112, 112), torch.float32), ((96, 1, 1), torch.float32)],
        {"model_names": ["pd_mobilenetv2_basic_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 24, 56, 56), torch.float32), ((24, 1, 1), torch.float32)],
        {"model_names": ["pd_mobilenetv2_basic_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 144, 56, 56), torch.float32), ((144, 1, 1), torch.float32)],
        {"model_names": ["pd_mobilenetv2_basic_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 144, 28, 28), torch.float32), ((144, 1, 1), torch.float32)],
        {"model_names": ["pd_mobilenetv2_basic_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 32, 28, 28), torch.float32), ((32, 1, 1), torch.float32)],
        {"model_names": ["pd_mobilenetv2_basic_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 192, 14, 14), torch.float32), ((192, 1, 1), torch.float32)],
        {"model_names": ["pd_mobilenetv2_basic_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 64, 14, 14), torch.float32), ((64, 1, 1), torch.float32)],
        {"model_names": ["pd_mobilenetv2_basic_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 96, 14, 14), torch.float32), ((96, 1, 1), torch.float32)],
        {"model_names": ["pd_mobilenetv2_basic_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 160, 7, 7), torch.float32), ((160, 1, 1), torch.float32)],
        {"model_names": ["pd_mobilenetv2_basic_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 320, 7, 7), torch.float32), ((320, 1, 1), torch.float32)],
        {"model_names": ["pd_mobilenetv2_basic_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply1,
        [((1280,), torch.float32)],
        {"model_names": ["pd_mobilenetv2_basic_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply73,
        [((1280,), torch.float32)],
        {"model_names": ["pd_mobilenetv2_basic_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 1280, 7, 7), torch.float32), ((1280, 1, 1), torch.float32)],
        {"model_names": ["pd_mobilenetv2_basic_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1280,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_mobilenetv2_basic_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1280,), torch.float32), ((1280,), torch.float32)],
        {"model_names": ["pd_mobilenetv2_basic_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 25, 97), torch.float32), ((1, 25, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 12, 14, 14), torch.float32)],
        {"model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 1, 1, 14), torch.float32)],
        {"model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 588, 2048), torch.float32), ((1, 588, 2048), torch.float32)],
        {"model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 588, 2048), torch.float32), ((1, 588, 1), torch.float32)],
        {"model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"], "pcc": 0.99},
    ),
    (
        Multiply28,
        [((1, 588, 2048), torch.float32)],
        {"model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 16, 588, 128), torch.float32), ((1, 1, 588, 128), torch.float32)],
        {"model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 16, 588, 64), torch.float32)],
        {"model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 16, 588, 588), torch.float32)],
        {"model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 588, 5504), torch.float32), ((1, 588, 5504), torch.float32)],
        {"model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 1, 256, 256), torch.float32)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((2048, 16), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 6, 1024), torch.float32), ((1, 6, 1024), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 6, 1024), torch.float32), ((1, 6, 1), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply74,
        [((1, 6, 1024), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 2048, 6), torch.float32), ((1, 2048, 6), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 2048, 1, 16), torch.float32), ((1, 2048, 6, 1), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply75,
        [((1, 2048, 16), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 2048, 6, 1), torch.float32), ((1, 1, 6, 16), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 2048, 6, 16), torch.float32), ((1, 2048, 6, 1), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 2048, 16), torch.float32), ((1, 2048, 16), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((1, 2048, 6), torch.float32), ((1, 2048, 1), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((2, 1, 1536), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((2, 13, 768), torch.float32), ((2, 13, 768), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((2, 13, 768), torch.float32), ((2, 13, 1), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Multiply76,
        [((2, 13, 768), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((2, 1, 1, 13), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((2, 13, 1536), torch.float32), ((2, 13, 1), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Multiply3,
        [((2, 1, 1, 13), torch.float32), ((2, 1, 1, 13), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Multiply77,
        [((2, 1, 1, 13), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 1, 512), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 1500, 512), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    record_forge_op_name("Multiply")

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
