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

    def forward(self, multiply_input_0, multiply_input_1):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, multiply_input_1)
        return multiply_output_1


class Multiply1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply1.weight_1", forge.Parameter(*(128,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply1.weight_1"))
        return multiply_output_1


class Multiply2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply2_const_1", shape=(1,), dtype=torch.float32)

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_constant("multiply2_const_1"))
        return multiply_output_1


class Multiply3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply3.weight_1", forge.Parameter(*(312,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply3.weight_1"))
        return multiply_output_1


class Multiply4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply4.weight_1", forge.Parameter(*(64,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply4.weight_1"))
        return multiply_output_1


class Multiply5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply5.weight_1", forge.Parameter(*(256,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply5.weight_1"))
        return multiply_output_1


class Multiply6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply6.weight_1", forge.Parameter(*(512,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply6.weight_1"))
        return multiply_output_1


class Multiply7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply7.weight_1",
            forge.Parameter(*(1024,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply7.weight_1"))
        return multiply_output_1


class Multiply8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply8.weight_1",
            forge.Parameter(*(2048,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply8.weight_1"))
        return multiply_output_1


class Multiply9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply9_const_0", shape=(1, 1, 128, 128), dtype=torch.float32)

    def forward(self, multiply_input_1):
        multiply_output_1 = forge.op.Multiply("", self.get_constant("multiply9_const_0"), multiply_input_1)
        return multiply_output_1


class Multiply10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply10.weight_1",
            forge.Parameter(*(16,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply10.weight_1"))
        return multiply_output_1


class Multiply11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply11.weight_1",
            forge.Parameter(*(32,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply11.weight_1"))
        return multiply_output_1


class Multiply12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply12.weight_1",
            forge.Parameter(*(64,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply12.weight_1"))
        return multiply_output_1


class Multiply13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply13.weight_1",
            forge.Parameter(*(128,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply13.weight_1"))
        return multiply_output_1


class Multiply14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply14.weight_1",
            forge.Parameter(*(256,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply14.weight_1"))
        return multiply_output_1


class Multiply15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply15.weight_1",
            forge.Parameter(*(512,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply15.weight_1"))
        return multiply_output_1


class Multiply16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply16.weight_1",
            forge.Parameter(*(1024,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply16.weight_1"))
        return multiply_output_1


class Multiply17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply17.weight_1",
            forge.Parameter(*(40,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply17.weight_1"))
        return multiply_output_1


class Multiply18(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply18.weight_1",
            forge.Parameter(*(24,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply18.weight_1"))
        return multiply_output_1


class Multiply19(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply19.weight_1",
            forge.Parameter(*(144,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply19.weight_1"))
        return multiply_output_1


class Multiply20(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply20.weight_1",
            forge.Parameter(*(192,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply20.weight_1"))
        return multiply_output_1


class Multiply21(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply21.weight_1",
            forge.Parameter(*(48,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply21.weight_1"))
        return multiply_output_1


class Multiply22(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply22.weight_1",
            forge.Parameter(*(288,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply22.weight_1"))
        return multiply_output_1


class Multiply23(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply23.weight_1",
            forge.Parameter(*(96,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply23.weight_1"))
        return multiply_output_1


class Multiply24(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply24.weight_1",
            forge.Parameter(*(576,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply24.weight_1"))
        return multiply_output_1


class Multiply25(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply25.weight_1",
            forge.Parameter(*(136,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply25.weight_1"))
        return multiply_output_1


class Multiply26(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply26.weight_1",
            forge.Parameter(*(816,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply26.weight_1"))
        return multiply_output_1


class Multiply27(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply27.weight_1",
            forge.Parameter(*(232,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply27.weight_1"))
        return multiply_output_1


class Multiply28(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply28.weight_1",
            forge.Parameter(*(1392,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply28.weight_1"))
        return multiply_output_1


class Multiply29(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply29.weight_1",
            forge.Parameter(*(384,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply29.weight_1"))
        return multiply_output_1


class Multiply30(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply30.weight_1",
            forge.Parameter(*(2304,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply30.weight_1"))
        return multiply_output_1


class Multiply31(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply31.weight_1",
            forge.Parameter(*(1536,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply31.weight_1"))
        return multiply_output_1


class Multiply32(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply32.weight_1",
            forge.Parameter(*(56,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply32.weight_1"))
        return multiply_output_1


class Multiply33(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply33.weight_1",
            forge.Parameter(*(336,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply33.weight_1"))
        return multiply_output_1


class Multiply34(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply34.weight_1",
            forge.Parameter(*(112,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply34.weight_1"))
        return multiply_output_1


class Multiply35(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply35.weight_1",
            forge.Parameter(*(672,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply35.weight_1"))
        return multiply_output_1


class Multiply36(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply36.weight_1",
            forge.Parameter(*(160,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply36.weight_1"))
        return multiply_output_1


class Multiply37(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply37.weight_1",
            forge.Parameter(*(960,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply37.weight_1"))
        return multiply_output_1


class Multiply38(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply38.weight_1",
            forge.Parameter(*(272,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply38.weight_1"))
        return multiply_output_1


class Multiply39(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply39.weight_1",
            forge.Parameter(*(1632,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply39.weight_1"))
        return multiply_output_1


class Multiply40(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply40.weight_1",
            forge.Parameter(*(448,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply40.weight_1"))
        return multiply_output_1


class Multiply41(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply41.weight_1",
            forge.Parameter(*(2688,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply41.weight_1"))
        return multiply_output_1


class Multiply42(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply42.weight_1",
            forge.Parameter(*(1792,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply42.weight_1"))
        return multiply_output_1


class Multiply43(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply43.weight_1",
            forge.Parameter(*(240,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply43.weight_1"))
        return multiply_output_1


class Multiply44(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply44.weight_1",
            forge.Parameter(*(72,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply44.weight_1"))
        return multiply_output_1


class Multiply45(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply45.weight_1",
            forge.Parameter(*(432,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply45.weight_1"))
        return multiply_output_1


class Multiply46(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply46.weight_1",
            forge.Parameter(*(864,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply46.weight_1"))
        return multiply_output_1


class Multiply47(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply47.weight_1",
            forge.Parameter(*(200,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply47.weight_1"))
        return multiply_output_1


class Multiply48(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply48.weight_1",
            forge.Parameter(*(1200,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply48.weight_1"))
        return multiply_output_1


class Multiply49(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply49.weight_1",
            forge.Parameter(*(344,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply49.weight_1"))
        return multiply_output_1


class Multiply50(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply50.weight_1",
            forge.Parameter(*(2064,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply50.weight_1"))
        return multiply_output_1


class Multiply51(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply51.weight_1",
            forge.Parameter(*(3456,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply51.weight_1"))
        return multiply_output_1


class Multiply52(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply52.weight_1",
            forge.Parameter(*(30,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply52.weight_1"))
        return multiply_output_1


class Multiply53(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply53.weight_1",
            forge.Parameter(*(60,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply53.weight_1"))
        return multiply_output_1


class Multiply54(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply54.weight_1",
            forge.Parameter(*(120,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply54.weight_1"))
        return multiply_output_1


class Multiply55(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply55.weight_1",
            forge.Parameter(*(2048,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply55.weight_1"))
        return multiply_output_1


class Multiply56(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply56.weight_0",
            forge.Parameter(*(2048,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_1):
        multiply_output_1 = forge.op.Multiply("", self.get_parameter("multiply56.weight_0"), multiply_input_1)
        return multiply_output_1


class Multiply57(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply57_const_0", shape=(1, 1, 256, 256), dtype=torch.float32)

    def forward(self, multiply_input_1):
        multiply_output_1 = forge.op.Multiply("", self.get_constant("multiply57_const_0"), multiply_input_1)
        return multiply_output_1


class Multiply58(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply58.weight_1",
            forge.Parameter(*(528,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply58.weight_1"))
        return multiply_output_1


class Multiply59(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply59.weight_1",
            forge.Parameter(*(1056,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply59.weight_1"))
        return multiply_output_1


class Multiply60(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply60.weight_1",
            forge.Parameter(*(2904,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply60.weight_1"))
        return multiply_output_1


class Multiply61(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply61.weight_1",
            forge.Parameter(*(7392,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply61.weight_1"))
        return multiply_output_1


class Multiply62(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply62.weight_1",
            forge.Parameter(*(224,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply62.weight_1"))
        return multiply_output_1


class Multiply63(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply63.weight_1",
            forge.Parameter(*(896,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply63.weight_1"))
        return multiply_output_1


class Multiply64(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply64.weight_1",
            forge.Parameter(*(2016,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply64.weight_1"))
        return multiply_output_1


class Multiply65(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply65_const_1", shape=(1,), dtype=torch.bfloat16)

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_constant("multiply65_const_1"))
        return multiply_output_1


class Multiply66(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply66.weight_1",
            forge.Parameter(*(768,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply66.weight_1"))
        return multiply_output_1


class Multiply67(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply67.weight_1",
            forge.Parameter(*(80,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply67.weight_1"))
        return multiply_output_1


class Multiply68(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply68.weight_1",
            forge.Parameter(*(728,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply68.weight_1"))
        return multiply_output_1


class Multiply69(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply69_const_1", shape=(1, 255, 80, 80), dtype=torch.float32)

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_constant("multiply69_const_1"))
        return multiply_output_1


class Multiply70(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply70_const_1", shape=(1, 255, 40, 40), dtype=torch.float32)

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_constant("multiply70_const_1"))
        return multiply_output_1


class Multiply71(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply71_const_1", shape=(1, 255, 20, 20), dtype=torch.float32)

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_constant("multiply71_const_1"))
        return multiply_output_1


class Multiply72(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply72_const_1", shape=(1, 255, 60, 60), dtype=torch.float32)

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_constant("multiply72_const_1"))
        return multiply_output_1


class Multiply73(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply73_const_1", shape=(1, 255, 30, 30), dtype=torch.float32)

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_constant("multiply73_const_1"))
        return multiply_output_1


class Multiply74(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply74_const_1", shape=(1, 255, 15, 15), dtype=torch.float32)

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_constant("multiply74_const_1"))
        return multiply_output_1


class Multiply75(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply75.weight_1",
            forge.Parameter(*(320,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply75.weight_1"))
        return multiply_output_1


class Multiply76(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply76.weight_1",
            forge.Parameter(*(640,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply76.weight_1"))
        return multiply_output_1


class Multiply77(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply77_const_1", shape=(1, 8400), dtype=torch.bfloat16)

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_constant("multiply77_const_1"))
        return multiply_output_1


class Multiply78(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply78.weight_1",
            forge.Parameter(*(768,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply78.weight_1"))
        return multiply_output_1


class Multiply79(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply79.weight_0",
            forge.Parameter(*(1024,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_1):
        multiply_output_1 = forge.op.Multiply("", self.get_parameter("multiply79.weight_0"), multiply_input_1)
        return multiply_output_1


class Multiply80(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply80.weight_1",
            forge.Parameter(*(352,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply80.weight_1"))
        return multiply_output_1


class Multiply81(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply81.weight_1",
            forge.Parameter(*(416,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply81.weight_1"))
        return multiply_output_1


class Multiply82(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply82.weight_1",
            forge.Parameter(*(480,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply82.weight_1"))
        return multiply_output_1


class Multiply83(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply83.weight_1",
            forge.Parameter(*(544,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply83.weight_1"))
        return multiply_output_1


class Multiply84(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply84.weight_1",
            forge.Parameter(*(608,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply84.weight_1"))
        return multiply_output_1


class Multiply85(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply85.weight_1",
            forge.Parameter(*(704,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply85.weight_1"))
        return multiply_output_1


class Multiply86(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply86.weight_1",
            forge.Parameter(*(736,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply86.weight_1"))
        return multiply_output_1


class Multiply87(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply87.weight_1",
            forge.Parameter(*(800,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply87.weight_1"))
        return multiply_output_1


class Multiply88(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply88.weight_1",
            forge.Parameter(*(832,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply88.weight_1"))
        return multiply_output_1


class Multiply89(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply89.weight_1",
            forge.Parameter(*(928,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply89.weight_1"))
        return multiply_output_1


class Multiply90(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply90.weight_1",
            forge.Parameter(*(992,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply90.weight_1"))
        return multiply_output_1


class Multiply91(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply91.weight_1",
            forge.Parameter(*(1344,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply91.weight_1"))
        return multiply_output_1


class Multiply92(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply92.weight_1",
            forge.Parameter(*(3840,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply92.weight_1"))
        return multiply_output_1


class Multiply93(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply93.weight_1",
            forge.Parameter(*(2560,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply93.weight_1"))
        return multiply_output_1


class Multiply94(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply94.weight_1",
            forge.Parameter(*(8,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply94.weight_1"))
        return multiply_output_1


class Multiply95(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply95.weight_1",
            forge.Parameter(*(12,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply95.weight_1"))
        return multiply_output_1


class Multiply96(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply96.weight_1",
            forge.Parameter(*(36,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply96.weight_1"))
        return multiply_output_1


class Multiply97(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply97.weight_1",
            forge.Parameter(*(20,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply97.weight_1"))
        return multiply_output_1


class Multiply98(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply98.weight_1",
            forge.Parameter(*(100,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply98.weight_1"))
        return multiply_output_1


class Multiply99(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply99.weight_1",
            forge.Parameter(*(92,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply99.weight_1"))
        return multiply_output_1


class Multiply100(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply100.weight_1",
            forge.Parameter(*(18,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply100.weight_1"))
        return multiply_output_1


class Multiply101(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply101.weight_1",
            forge.Parameter(*(720,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply101.weight_1"))
        return multiply_output_1


class Multiply102(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply102.weight_1",
            forge.Parameter(*(1280,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply102.weight_1"))
        return multiply_output_1


class Multiply103(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply103.weight_1",
            forge.Parameter(*(184,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply103.weight_1"))
        return multiply_output_1


class Multiply104(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply104.weight_0",
            forge.Parameter(*(1024,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_1):
        multiply_output_1 = forge.op.Multiply("", self.get_parameter("multiply104.weight_0"), multiply_input_1)
        return multiply_output_1


class Multiply105(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply105_const_0", shape=(1, 1, 6, 6), dtype=torch.float32)

    def forward(self, multiply_input_1):
        multiply_output_1 = forge.op.Multiply("", self.get_constant("multiply105_const_0"), multiply_input_1)
        return multiply_output_1


class Multiply106(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply106.weight_0",
            forge.Parameter(*(1536,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_1):
        multiply_output_1 = forge.op.Multiply("", self.get_parameter("multiply106.weight_0"), multiply_input_1)
        return multiply_output_1


class Multiply107(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply107_const_0", shape=(1, 1, 35, 35), dtype=torch.float32)

    def forward(self, multiply_input_1):
        multiply_output_1 = forge.op.Multiply("", self.get_constant("multiply107_const_0"), multiply_input_1)
        return multiply_output_1


class Multiply108(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply108_const_0", shape=(1, 1, 29, 29), dtype=torch.float32)

    def forward(self, multiply_input_1):
        multiply_output_1 = forge.op.Multiply("", self.get_constant("multiply108_const_0"), multiply_input_1)
        return multiply_output_1


class Multiply109(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply109.weight_1",
            forge.Parameter(*(1232,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply109.weight_1"))
        return multiply_output_1


class Multiply110(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply110.weight_1",
            forge.Parameter(*(3024,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply110.weight_1"))
        return multiply_output_1


class Multiply111(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply111.weight_1",
            forge.Parameter(*(696,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply111.weight_1"))
        return multiply_output_1


class Multiply112(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply112.weight_1",
            forge.Parameter(*(3712,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply112.weight_1"))
        return multiply_output_1


class Multiply113(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply113.weight_1",
            forge.Parameter(*(2520,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply113.weight_1"))
        return multiply_output_1


class Multiply114(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply114.weight_1",
            forge.Parameter(*(1008,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply114.weight_1"))
        return multiply_output_1


class Multiply115(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply115.weight_0",
            forge.Parameter(*(768,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_1):
        multiply_output_1 = forge.op.Multiply("", self.get_parameter("multiply115.weight_0"), multiply_input_1)
        return multiply_output_1


class Multiply116(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply116_const_1", shape=(1, 255, 10, 10), dtype=torch.float32)

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_constant("multiply116_const_1"))
        return multiply_output_1


class Multiply117(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply117.weight_0",
            forge.Parameter(*(768,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_1):
        multiply_output_1 = forge.op.Multiply("", self.get_parameter("multiply117.weight_0"), multiply_input_1)
        return multiply_output_1


class Multiply118(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply118.weight_1",
            forge.Parameter(*(176,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply118.weight_1"))
        return multiply_output_1


class Multiply119(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply119.weight_1",
            forge.Parameter(*(304,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply119.weight_1"))
        return multiply_output_1


class Multiply120(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply120.weight_1",
            forge.Parameter(*(1824,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply120.weight_1"))
        return multiply_output_1


class Multiply121(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply121.weight_1",
            forge.Parameter(*(3072,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply121.weight_1"))
        return multiply_output_1


class Multiply122(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply122.weight_1",
            forge.Parameter(*(88,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply122.weight_1"))
        return multiply_output_1


class Multiply123(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply123.weight_0",
            forge.Parameter(*(896,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_1):
        multiply_output_1 = forge.op.Multiply("", self.get_parameter("multiply123.weight_0"), multiply_input_1)
        return multiply_output_1


class Multiply124(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply124_const_0", shape=(1,), dtype=torch.bfloat16)

    def forward(self, multiply_input_1):
        multiply_output_1 = forge.op.Multiply("", self.get_constant("multiply124_const_0"), multiply_input_1)
        return multiply_output_1


class Multiply125(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply125_const_1", shape=(5880, 1), dtype=torch.bfloat16)

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_constant("multiply125_const_1"))
        return multiply_output_1


class Multiply126(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply126_const_1", shape=(1, 8400), dtype=torch.float32)

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_constant("multiply126_const_1"))
        return multiply_output_1


class Multiply127(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply127.weight_1", forge.Parameter(*(8,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply127.weight_1"))
        return multiply_output_1


class Multiply128(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply128.weight_1",
            forge.Parameter(*(40,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply128.weight_1"))
        return multiply_output_1


class Multiply129(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply129.weight_1",
            forge.Parameter(*(16,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply129.weight_1"))
        return multiply_output_1


class Multiply130(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply130.weight_1",
            forge.Parameter(*(48,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply130.weight_1"))
        return multiply_output_1


class Multiply131(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply131.weight_1",
            forge.Parameter(*(24,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply131.weight_1"))
        return multiply_output_1


class Multiply132(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply132.weight_1",
            forge.Parameter(*(120,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply132.weight_1"))
        return multiply_output_1


class Multiply133(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply133.weight_1",
            forge.Parameter(*(72,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply133.weight_1"))
        return multiply_output_1


class Multiply134(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply134.weight_1",
            forge.Parameter(*(144,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply134.weight_1"))
        return multiply_output_1


class Multiply135(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply135.weight_1",
            forge.Parameter(*(288,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply135.weight_1"))
        return multiply_output_1


class Multiply136(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply136_const_1", shape=(1, 48), dtype=torch.float32)

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_constant("multiply136_const_1"))
        return multiply_output_1


class Multiply137(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply137_const_1", shape=(1, 256, 1, 32), dtype=torch.float32)

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_constant("multiply137_const_1"))
        return multiply_output_1


class Multiply138(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply138.weight_1",
            forge.Parameter(*(1088,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply138.weight_1"))
        return multiply_output_1


class Multiply139(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply139.weight_1",
            forge.Parameter(*(1120,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply139.weight_1"))
        return multiply_output_1


class Multiply140(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply140.weight_1",
            forge.Parameter(*(1152,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply140.weight_1"))
        return multiply_output_1


class Multiply141(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply141.weight_1",
            forge.Parameter(*(1184,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply141.weight_1"))
        return multiply_output_1


class Multiply142(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply142.weight_1",
            forge.Parameter(*(1216,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply142.weight_1"))
        return multiply_output_1


class Multiply143(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply143.weight_1",
            forge.Parameter(*(1248,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply143.weight_1"))
        return multiply_output_1


class Multiply144(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply144.weight_1",
            forge.Parameter(*(1312,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply144.weight_1"))
        return multiply_output_1


class Multiply145(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply145.weight_1",
            forge.Parameter(*(1376,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply145.weight_1"))
        return multiply_output_1


class Multiply146(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply146.weight_1",
            forge.Parameter(*(1408,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply146.weight_1"))
        return multiply_output_1


class Multiply147(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply147.weight_1",
            forge.Parameter(*(1440,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply147.weight_1"))
        return multiply_output_1


class Multiply148(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply148.weight_1",
            forge.Parameter(*(1472,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply148.weight_1"))
        return multiply_output_1


class Multiply149(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply149.weight_1",
            forge.Parameter(*(1504,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply149.weight_1"))
        return multiply_output_1


class Multiply150(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply150.weight_1",
            forge.Parameter(*(1568,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply150.weight_1"))
        return multiply_output_1


class Multiply151(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply151.weight_1",
            forge.Parameter(*(1600,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply151.weight_1"))
        return multiply_output_1


class Multiply152(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply152.weight_1",
            forge.Parameter(*(1664,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply152.weight_1"))
        return multiply_output_1


class Multiply153(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply153.weight_1",
            forge.Parameter(*(1696,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply153.weight_1"))
        return multiply_output_1


class Multiply154(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply154.weight_1",
            forge.Parameter(*(1728,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply154.weight_1"))
        return multiply_output_1


class Multiply155(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply155.weight_1",
            forge.Parameter(*(1760,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply155.weight_1"))
        return multiply_output_1


class Multiply156(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply156.weight_1",
            forge.Parameter(*(1856,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply156.weight_1"))
        return multiply_output_1


class Multiply157(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply157.weight_1",
            forge.Parameter(*(1888,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply157.weight_1"))
        return multiply_output_1


class Multiply158(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply158.weight_1",
            forge.Parameter(*(1920,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply158.weight_1"))
        return multiply_output_1


class Multiply159(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply159.weight_1",
            forge.Parameter(*(208,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply159.weight_1"))
        return multiply_output_1


class Multiply160(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply160.weight_1",
            forge.Parameter(*(2112,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply160.weight_1"))
        return multiply_output_1


class Multiply161(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("multiply161_const_0", shape=(32,), dtype=torch.int32)

    def forward(self, multiply_input_1):
        multiply_output_1 = forge.op.Multiply("", self.get_constant("multiply161_const_0"), multiply_input_1)
        return multiply_output_1


class Multiply162(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply162.weight_0",
            forge.Parameter(*(512,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, multiply_input_1):
        multiply_output_1 = forge.op.Multiply("", self.get_parameter("multiply162.weight_0"), multiply_input_1)
        return multiply_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Multiply0,
        [((1, 11, 128), torch.float32), ((1, 11, 128), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99},
    ),
    (Multiply1, [((1, 11, 128), torch.float32)], {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99}),
    (
        Multiply2,
        [((1, 12, 11, 11), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 1, 1, 11), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 11, 312), torch.float32), ((1, 11, 312), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99},
    ),
    (Multiply3, [((1, 11, 312), torch.float32)], {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99}),
    (
        Multiply4,
        [((64,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "tf_resnet_resnet50_img_cls_keras",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 64, 112, 112), torch.float32), ((64, 1, 1), torch.float32)],
        {"model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((64,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "tf_resnet_resnet50_img_cls_keras",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((64,), torch.float32), ((64,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "tf_resnet_resnet50_img_cls_keras",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 64, 56, 56), torch.float32), ((64, 1, 1), torch.float32)],
        {"model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply5,
        [((256,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "tf_resnet_resnet50_img_cls_keras",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 256, 56, 56), torch.float32), ((256, 1, 1), torch.float32)],
        {"model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((256,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "tf_resnet_resnet50_img_cls_keras",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((256,), torch.float32), ((256,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "tf_resnet_resnet50_img_cls_keras",
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
                "tf_resnet_resnet50_img_cls_keras",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 128, 56, 56), torch.float32), ((128, 1, 1), torch.float32)],
        {"model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((128,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "tf_resnet_resnet50_img_cls_keras",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((128,), torch.float32), ((128,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "tf_resnet_resnet50_img_cls_keras",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 128, 28, 28), torch.float32), ((128, 1, 1), torch.float32)],
        {"model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply6,
        [((512,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "tf_resnet_resnet50_img_cls_keras",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 512, 28, 28), torch.float32), ((512, 1, 1), torch.float32)],
        {"model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((512,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "tf_resnet_resnet50_img_cls_keras",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((512,), torch.float32), ((512,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "tf_resnet_resnet50_img_cls_keras",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 256, 28, 28), torch.float32), ((256, 1, 1), torch.float32)],
        {"model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 256, 14, 14), torch.float32), ((256, 1, 1), torch.float32)],
        {"model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply7,
        [((1024,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "tf_resnet_resnet50_img_cls_keras",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 1024, 14, 14), torch.float32), ((1024, 1, 1), torch.float32)],
        {"model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1024,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "tf_resnet_resnet50_img_cls_keras",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1024,), torch.float32), ((1024,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "tf_resnet_resnet50_img_cls_keras",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 512, 14, 14), torch.float32), ((512, 1, 1), torch.float32)],
        {"model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 512, 7, 7), torch.float32), ((512, 1, 1), torch.float32)],
        {"model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply8,
        [((2048,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "tf_resnet_resnet50_img_cls_keras",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 2048, 7, 7), torch.float32), ((2048, 1, 1), torch.float32)],
        {"model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((2048,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "tf_resnet_resnet50_img_cls_keras",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((2048,), torch.float32), ((2048,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "tf_resnet_resnet50_img_cls_keras",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply2,
        [((1, 16, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 1, 128, 128), torch.float32), ((1, 1, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply9,
        [((1, 1, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply10,
        [((16,), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 16, 224, 224), torch.bfloat16), ((16, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((16,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((16,), torch.bfloat16), ((16,), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply11,
        [((32,), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_regnet_regnet_y_128gf_img_cls_torchvision",
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_160_img_cls_hf",
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_regnet_regnet_x_32gf_img_cls_torchvision",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
                "pt_yolo_v4_default_obj_det_github",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_xception_xception_img_cls_timm",
                "pt_yolo_v3_default_obj_det_github",
                "pt_yolov9_default_obj_det_github",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 32, 112, 112), torch.bfloat16), ((32, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_160_img_cls_hf",
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_regnet_regnet_x_32gf_img_cls_torchvision",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((32,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_regnet_regnet_y_128gf_img_cls_torchvision",
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_160_img_cls_hf",
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_regnet_regnet_x_32gf_img_cls_torchvision",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
                "pt_yolo_v4_default_obj_det_github",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_xception_xception_img_cls_timm",
                "pt_yolo_v3_default_obj_det_github",
                "pt_yolov9_default_obj_det_github",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((32,), torch.bfloat16), ((32,), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_regnet_regnet_y_128gf_img_cls_torchvision",
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_160_img_cls_hf",
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_regnet_regnet_x_32gf_img_cls_torchvision",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
                "pt_yolo_v4_default_obj_det_github",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_xception_xception_img_cls_timm",
                "pt_yolo_v3_default_obj_det_github",
                "pt_yolov9_default_obj_det_github",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply12,
        [((64,), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilenetv3_ssd_resnet34_img_cls_torchvision",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_yolo_v4_default_obj_det_github",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_xception_xception_img_cls_timm",
                "pt_yolo_v3_default_obj_det_github",
                "pt_yolov9_default_obj_det_github",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 64, 112, 112), torch.bfloat16), ((64, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((64,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilenetv3_ssd_resnet34_img_cls_torchvision",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_yolo_v4_default_obj_det_github",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_xception_xception_img_cls_timm",
                "pt_yolo_v3_default_obj_det_github",
                "pt_yolov9_default_obj_det_github",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((64,), torch.bfloat16), ((64,), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilenetv3_ssd_resnet34_img_cls_torchvision",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_yolo_v4_default_obj_det_github",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_xception_xception_img_cls_timm",
                "pt_yolo_v3_default_obj_det_github",
                "pt_yolov9_default_obj_det_github",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 64, 56, 56), torch.bfloat16), ((64, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply13,
        [((128,), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilenetv3_ssd_resnet34_img_cls_torchvision",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_yolo_v4_default_obj_det_github",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_xception_xception_img_cls_timm",
                "pt_yolo_v3_default_obj_det_github",
                "pt_yolov9_default_obj_det_github",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 128, 56, 56), torch.bfloat16), ((128, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((128,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilenetv3_ssd_resnet34_img_cls_torchvision",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_yolo_v4_default_obj_det_github",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_xception_xception_img_cls_timm",
                "pt_yolo_v3_default_obj_det_github",
                "pt_yolov9_default_obj_det_github",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((128,), torch.bfloat16), ((128,), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilenetv3_ssd_resnet34_img_cls_torchvision",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_yolo_v4_default_obj_det_github",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_xception_xception_img_cls_timm",
                "pt_yolo_v3_default_obj_det_github",
                "pt_yolov9_default_obj_det_github",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 128, 28, 28), torch.bfloat16), ((128, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply14,
        [((256,), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilenetv3_ssd_resnet34_img_cls_torchvision",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_fpn_base_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_yolo_v4_default_obj_det_github",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_xception_xception_img_cls_timm",
                "pt_yolo_v3_default_obj_det_github",
                "pt_yolov9_default_obj_det_github",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 256, 28, 28), torch.bfloat16), ((256, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_resnet_50_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((256,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilenetv3_ssd_resnet34_img_cls_torchvision",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_fpn_base_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_yolo_v4_default_obj_det_github",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_xception_xception_img_cls_timm",
                "pt_yolo_v3_default_obj_det_github",
                "pt_yolov9_default_obj_det_github",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((256,), torch.bfloat16), ((256,), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilenetv3_ssd_resnet34_img_cls_torchvision",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_fpn_base_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_yolo_v4_default_obj_det_github",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_xception_xception_img_cls_timm",
                "pt_yolo_v3_default_obj_det_github",
                "pt_yolov9_default_obj_det_github",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 256, 14, 14), torch.bfloat16), ((256, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply15,
        [((512,), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilenetv3_ssd_resnet34_img_cls_torchvision",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_yolo_v4_default_obj_det_github",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_yolo_v3_default_obj_det_github",
                "pt_yolov9_default_obj_det_github",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 512, 14, 14), torch.bfloat16), ((512, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((512,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilenetv3_ssd_resnet34_img_cls_torchvision",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_yolo_v4_default_obj_det_github",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_yolo_v3_default_obj_det_github",
                "pt_yolov9_default_obj_det_github",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((512,), torch.bfloat16), ((512,), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilenetv3_ssd_resnet34_img_cls_torchvision",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_yolo_v4_default_obj_det_github",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_yolo_v3_default_obj_det_github",
                "pt_yolov9_default_obj_det_github",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 512, 7, 7), torch.bfloat16), ((512, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply16,
        [((1024,), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_yolo_v4_default_obj_det_github",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_xception_xception_img_cls_timm",
                "pt_yolo_v3_default_obj_det_github",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1024, 7, 7), torch.bfloat16), ((1024, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1024,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_yolo_v4_default_obj_det_github",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_xception_xception_img_cls_timm",
                "pt_yolo_v3_default_obj_det_github",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1024,), torch.bfloat16), ((1024,), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_yolo_v4_default_obj_det_github",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_xception_xception_img_cls_timm",
                "pt_yolo_v3_default_obj_det_github",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply2,
        [((1, 12, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply17,
        [((40,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 40, 112, 112), torch.bfloat16), ((40, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((40,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((40,), torch.bfloat16), ((40,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 40, 112, 112), torch.bfloat16), ((1, 40, 112, 112), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b3_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 10, 1, 1), torch.bfloat16), ((1, 10, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 40, 1, 1), torch.bfloat16), ((1, 40, 112, 112), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b3_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply18,
        [((24,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 24, 112, 112), torch.bfloat16), ((24, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((24,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((24,), torch.bfloat16), ((24,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 24, 112, 112), torch.bfloat16), ((1, 24, 112, 112), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 6, 1, 1), torch.bfloat16), ((1, 6, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 24, 1, 1), torch.bfloat16), ((1, 24, 112, 112), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply19,
        [((144,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 144, 112, 112), torch.bfloat16), ((144, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((144,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((144,), torch.bfloat16), ((144,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 144, 112, 112), torch.bfloat16), ((1, 144, 112, 112), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 144, 56, 56), torch.bfloat16), ((144, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 144, 56, 56), torch.bfloat16), ((1, 144, 56, 56), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 144, 1, 1), torch.bfloat16), ((1, 144, 56, 56), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 32, 56, 56), torch.bfloat16), ((32, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply20,
        [((192,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 192, 56, 56), torch.bfloat16), ((192, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((192,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((192,), torch.bfloat16), ((192,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 192, 56, 56), torch.bfloat16), ((1, 192, 56, 56), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 8, 1, 1), torch.bfloat16), ((1, 8, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 192, 1, 1), torch.bfloat16), ((1, 192, 56, 56), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 192, 28, 28), torch.bfloat16), ((192, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 192, 28, 28), torch.bfloat16), ((1, 192, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 192, 1, 1), torch.bfloat16), ((1, 192, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply21,
        [((48,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 48, 28, 28), torch.bfloat16), ((48, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((48,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((48,), torch.bfloat16), ((48,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply22,
        [((288,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 288, 28, 28), torch.bfloat16), ((288, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((288,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((288,), torch.bfloat16), ((288,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 288, 28, 28), torch.bfloat16), ((1, 288, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 12, 1, 1), torch.bfloat16), ((1, 12, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 288, 1, 1), torch.bfloat16), ((1, 288, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 288, 14, 14), torch.bfloat16), ((288, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 288, 14, 14), torch.bfloat16), ((1, 288, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 288, 1, 1), torch.bfloat16), ((1, 288, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply23,
        [((96,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 96, 14, 14), torch.bfloat16), ((96, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((96,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((96,), torch.bfloat16), ((96,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply24,
        [((576,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 576, 14, 14), torch.bfloat16), ((576, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((576,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((576,), torch.bfloat16), ((576,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 576, 14, 14), torch.bfloat16), ((1, 576, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b3_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 24, 1, 1), torch.bfloat16), ((1, 24, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b3_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 576, 1, 1), torch.bfloat16), ((1, 576, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b3_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply25,
        [((136,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 136, 14, 14), torch.bfloat16), ((136, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b3_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((136,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((136,), torch.bfloat16), ((136,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply26,
        [((816,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 816, 14, 14), torch.bfloat16), ((816, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b3_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((816,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((816,), torch.bfloat16), ((816,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 816, 14, 14), torch.bfloat16), ((1, 816, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b3_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 34, 1, 1), torch.bfloat16), ((1, 34, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b3_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 816, 1, 1), torch.bfloat16), ((1, 816, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b3_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 816, 7, 7), torch.bfloat16), ((816, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b3_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 816, 7, 7), torch.bfloat16), ((1, 816, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b3_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 816, 1, 1), torch.bfloat16), ((1, 816, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b3_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply27,
        [((232,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 232, 7, 7), torch.bfloat16), ((232, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b3_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((232,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((232,), torch.bfloat16), ((232,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply28,
        [((1392,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1392, 7, 7), torch.bfloat16), ((1392, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b3_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1392,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1392,), torch.bfloat16), ((1392,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1392, 7, 7), torch.bfloat16), ((1, 1392, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b3_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 58, 1, 1), torch.bfloat16), ((1, 58, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b3_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1392, 1, 1), torch.bfloat16), ((1, 1392, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b3_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply29,
        [((384,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 384, 7, 7), torch.bfloat16), ((384, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((384,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((384,), torch.bfloat16), ((384,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply30,
        [((2304,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 2304, 7, 7), torch.bfloat16), ((2304, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((2304,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((2304,), torch.bfloat16), ((2304,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 2304, 7, 7), torch.bfloat16), ((1, 2304, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 96, 1, 1), torch.bfloat16), ((1, 96, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 2304, 1, 1), torch.bfloat16), ((1, 2304, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply31,
        [((1536,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1536, 7, 7), torch.bfloat16), ((1536, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1536,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1536,), torch.bfloat16), ((1536,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1536, 7, 7), torch.bfloat16), ((1, 1536, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b3_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 48, 112, 112), torch.bfloat16), ((48, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 48, 112, 112), torch.bfloat16), ((1, 48, 112, 112), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 48, 1, 1), torch.bfloat16), ((1, 48, 112, 112), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply32,
        [((56,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 56, 28, 28), torch.bfloat16), ((56, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((56,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((56,), torch.bfloat16), ((56,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply33,
        [((336,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_regnet_regnet_x_32gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 336, 28, 28), torch.bfloat16), ((336, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((336,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_regnet_regnet_x_32gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((336,), torch.bfloat16), ((336,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_regnet_regnet_x_32gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 336, 28, 28), torch.bfloat16), ((1, 336, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 14, 1, 1), torch.bfloat16), ((1, 14, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 336, 1, 1), torch.bfloat16), ((1, 336, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 336, 14, 14), torch.bfloat16), ((336, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 336, 14, 14), torch.bfloat16), ((1, 336, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 336, 1, 1), torch.bfloat16), ((1, 336, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply34,
        [((112,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 112, 14, 14), torch.bfloat16), ((112, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((112,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((112,), torch.bfloat16), ((112,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply35,
        [((672,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_regnet_regnet_x_32gf_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 672, 14, 14), torch.bfloat16), ((672, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((672,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_regnet_regnet_x_32gf_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((672,), torch.bfloat16), ((672,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_regnet_regnet_x_32gf_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 672, 14, 14), torch.bfloat16), ((1, 672, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 28, 1, 1), torch.bfloat16), ((1, 28, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 672, 1, 1), torch.bfloat16), ((1, 672, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply36,
        [((160,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_yolov8_yolov8x_obj_det_github",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 160, 14, 14), torch.bfloat16), ((160, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((160,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_yolov8_yolov8x_obj_det_github",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((160,), torch.bfloat16), ((160,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_yolov8_yolov8x_obj_det_github",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply37,
        [((960,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 960, 14, 14), torch.bfloat16), ((960, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((960,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((960,), torch.bfloat16), ((960,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 960, 14, 14), torch.bfloat16), ((1, 960, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 40, 1, 1), torch.bfloat16), ((1, 40, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 960, 1, 1), torch.bfloat16), ((1, 960, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 960, 7, 7), torch.bfloat16), ((960, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 960, 7, 7), torch.bfloat16), ((1, 960, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 960, 1, 1), torch.bfloat16), ((1, 960, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply38,
        [((272,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 272, 7, 7), torch.bfloat16), ((272, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((272,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((272,), torch.bfloat16), ((272,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply39,
        [((1632,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1632, 7, 7), torch.bfloat16), ((1632, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1632,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1632,), torch.bfloat16), ((1632,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1632, 7, 7), torch.bfloat16), ((1, 1632, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 68, 1, 1), torch.bfloat16), ((1, 68, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1632, 1, 1), torch.bfloat16), ((1, 1632, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply40,
        [((448,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_regnet_facebook_regnet_y_160_img_cls_hf",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 448, 7, 7), torch.bfloat16), ((448, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((448,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_regnet_facebook_regnet_y_160_img_cls_hf",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((448,), torch.bfloat16), ((448,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_regnet_facebook_regnet_y_160_img_cls_hf",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply41,
        [((2688,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 2688, 7, 7), torch.bfloat16), ((2688, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((2688,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((2688,), torch.bfloat16), ((2688,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 2688, 7, 7), torch.bfloat16), ((1, 2688, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 112, 1, 1), torch.bfloat16), ((1, 112, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 2688, 1, 1), torch.bfloat16), ((1, 2688, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply42,
        [((1792,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1792, 7, 7), torch.bfloat16), ((1792, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1792,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1792,), torch.bfloat16), ((1792,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1792, 7, 7), torch.bfloat16), ((1, 1792, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 56, 112, 112), torch.bfloat16), ((56, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 56, 112, 112), torch.bfloat16), ((1, 56, 112, 112), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 56, 1, 1), torch.bfloat16), ((1, 56, 112, 112), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 32, 112, 112), torch.bfloat16), ((1, 32, 112, 112), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 32, 1, 1), torch.bfloat16), ((1, 32, 112, 112), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 192, 112, 112), torch.bfloat16), ((192, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 192, 112, 112), torch.bfloat16), ((1, 192, 112, 112), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 40, 56, 56), torch.bfloat16), ((40, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply43,
        [((240,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 240, 56, 56), torch.bfloat16), ((240, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((240,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((240,), torch.bfloat16), ((240,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 240, 56, 56), torch.bfloat16), ((1, 240, 56, 56), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 240, 1, 1), torch.bfloat16), ((1, 240, 56, 56), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 240, 28, 28), torch.bfloat16), ((240, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 240, 28, 28), torch.bfloat16), ((1, 240, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 240, 1, 1), torch.bfloat16), ((1, 240, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply44,
        [((72,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 72, 28, 28), torch.bfloat16), ((72, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((72,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((72,), torch.bfloat16), ((72,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply45,
        [((432,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 432, 28, 28), torch.bfloat16), ((432, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((432,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((432,), torch.bfloat16), ((432,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 432, 28, 28), torch.bfloat16), ((1, 432, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 18, 1, 1), torch.bfloat16), ((1, 18, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 432, 1, 1), torch.bfloat16), ((1, 432, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 432, 14, 14), torch.bfloat16), ((432, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 432, 14, 14), torch.bfloat16), ((1, 432, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 432, 1, 1), torch.bfloat16), ((1, 432, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 144, 14, 14), torch.bfloat16), ((144, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply46,
        [((864,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 864, 14, 14), torch.bfloat16), ((864, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((864,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((864,), torch.bfloat16), ((864,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 864, 14, 14), torch.bfloat16), ((1, 864, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 36, 1, 1), torch.bfloat16), ((1, 36, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 864, 1, 1), torch.bfloat16), ((1, 864, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply47,
        [((200,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 200, 14, 14), torch.bfloat16), ((200, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((200,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((200,), torch.bfloat16), ((200,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply48,
        [((1200,), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1200, 14, 14), torch.bfloat16), ((1200, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1200,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1200,), torch.bfloat16), ((1200,), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1200, 14, 14), torch.bfloat16), ((1, 1200, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 50, 1, 1), torch.bfloat16), ((1, 50, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1200, 1, 1), torch.bfloat16), ((1, 1200, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1200, 7, 7), torch.bfloat16), ((1200, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1200, 7, 7), torch.bfloat16), ((1, 1200, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1200, 1, 1), torch.bfloat16), ((1, 1200, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply49,
        [((344,), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 344, 7, 7), torch.bfloat16), ((344, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((344,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((344,), torch.bfloat16), ((344,), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply50,
        [((2064,), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 2064, 7, 7), torch.bfloat16), ((2064, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((2064,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((2064,), torch.bfloat16), ((2064,), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 2064, 7, 7), torch.bfloat16), ((1, 2064, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 86, 1, 1), torch.bfloat16), ((1, 86, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 2064, 1, 1), torch.bfloat16), ((1, 2064, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 576, 7, 7), torch.bfloat16), ((576, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply51,
        [((3456,), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 3456, 7, 7), torch.bfloat16), ((3456, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((3456,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((3456,), torch.bfloat16), ((3456,), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 3456, 7, 7), torch.bfloat16), ((1, 3456, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 144, 1, 1), torch.bfloat16), ((1, 144, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 3456, 1, 1), torch.bfloat16), ((1, 3456, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 256, 56, 56), torch.bfloat16), ((256, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply52,
        [((30,), torch.bfloat16)],
        {"model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 30, 56, 56), torch.bfloat16), ((30, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((30,), torch.bfloat16), ((1,), torch.bfloat16)],
        {"model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((30,), torch.bfloat16), ((30,), torch.bfloat16)],
        {"model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply53,
        [((60,), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm", "pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 60, 28, 28), torch.bfloat16), ((60, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm", "pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((60,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm", "pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((60,), torch.bfloat16), ((60,), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm", "pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 30, 28, 28), torch.bfloat16), ((30, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply54,
        [((120,), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 120, 14, 14), torch.bfloat16), ((120, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((120,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((120,), torch.bfloat16), ((120,), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 30, 14, 14), torch.bfloat16), ((30, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 60, 14, 14), torch.bfloat16), ((60, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 240, 7, 7), torch.bfloat16), ((240, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 210, 7, 7), torch.bfloat16), ((210, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_hrnet_hrnet_w30_pose_estimation_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 120, 28, 28), torch.bfloat16), ((120, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 256, 7, 7), torch.bfloat16), ((256, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 128, 14, 14), torch.bfloat16), ((128, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 64, 28, 28), torch.bfloat16), ((64, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply55,
        [((2048,), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 2048, 7, 7), torch.bfloat16), ((2048, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((2048,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((2048,), torch.bfloat16), ((2048,), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 4, 2048), torch.float32), ((1, 4, 2048), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 4, 2048), torch.float32), ((1, 4, 1), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply56,
        [((1, 4, 2048), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 32, 4, 64), torch.float32), ((1, 1, 4, 64), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 32, 4, 32), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 8, 4, 64), torch.float32), ((1, 1, 4, 64), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 8, 4, 32), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 32, 4, 4), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 4, 8192), torch.float32), ((1, 4, 8192), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 64, 160, 160), torch.bfloat16), ((64, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_ssd_resnet34_img_cls_torchvision",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_yolov9_default_obj_det_github",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 64, 80, 80), torch.bfloat16), ((64, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_ssd_resnet34_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_yolov9_default_obj_det_github",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 128, 40, 40), torch.bfloat16), ((128, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_ssd_resnet34_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_yolov9_default_obj_det_github",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 256, 20, 20), torch.bfloat16), ((256, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_ssd_resnet34_img_cls_torchvision",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_yolov9_default_obj_det_github",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 512, 10, 10), torch.bfloat16), ((512, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_ssd_resnet34_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 32, 256, 32), torch.float32), ((1, 1, 256, 32), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99},
    ),
    (Multiply2, [((1, 32, 256, 16), torch.float32)], {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99}),
    (Multiply2, [((1, 32, 256, 256), torch.float32)], {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99}),
    (
        Multiply0,
        [((1, 1, 256, 256), torch.float32), ((1, 1, 256, 256), torch.float32)],
        {
            "model_names": [
                "pt_phi1_microsoft_phi_1_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply57,
        [((1, 1, 256, 256), torch.float32)],
        {
            "model_names": [
                "pt_phi1_microsoft_phi_1_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 32, 5, 32), torch.float32), ((1, 1, 5, 32), torch.float32)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 32, 5, 16), torch.float32)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 32, 5, 5), torch.float32)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 32, 192, 192), torch.bfloat16), ((32, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply58,
        [((528,), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_regnet_y_128gf_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 528, 96, 96), torch.bfloat16), ((528, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((528,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_regnet_y_128gf_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((528,), torch.bfloat16), ((528,), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_regnet_y_128gf_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 528, 192, 192), torch.bfloat16), ((528, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 528, 1, 1), torch.bfloat16), ((1, 528, 96, 96), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply59,
        [((1056,), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_regnet_y_128gf_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1056, 48, 48), torch.bfloat16), ((1056, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1056,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_regnet_y_128gf_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1056,), torch.bfloat16), ((1056,), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_regnet_y_128gf_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1056, 96, 96), torch.bfloat16), ((1056, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1056, 1, 1), torch.bfloat16), ((1, 1056, 48, 48), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply60,
        [((2904,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 2904, 24, 24), torch.bfloat16), ((2904, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((2904,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((2904,), torch.bfloat16), ((2904,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 2904, 48, 48), torch.bfloat16), ((2904, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 2904, 1, 1), torch.bfloat16), ((1, 2904, 24, 24), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply61,
        [((7392,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 7392, 12, 12), torch.bfloat16), ((7392, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((7392,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((7392,), torch.bfloat16), ((7392,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 7392, 24, 24), torch.bfloat16), ((7392, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 7392, 1, 1), torch.bfloat16), ((1, 7392, 12, 12), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply62,
        [((224,), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_160_img_cls_hf",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 224, 56, 56), torch.bfloat16), ((224, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_160_img_cls_hf",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((224,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_160_img_cls_hf",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((224,), torch.bfloat16), ((224,), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_160_img_cls_hf",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 224, 112, 112), torch.bfloat16), ((224, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_8gf_img_cls_torchvision", "pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 224, 1, 1), torch.bfloat16), ((1, 224, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 448, 28, 28), torch.bfloat16), ((448, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_160_img_cls_hf",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 448, 56, 56), torch.bfloat16), ((448, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_8gf_img_cls_torchvision", "pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 448, 1, 1), torch.bfloat16), ((1, 448, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply63,
        [((896,), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 896, 14, 14), torch.bfloat16), ((896, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((896,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((896,), torch.bfloat16), ((896,), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 896, 28, 28), torch.bfloat16), ((896, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 896, 1, 1), torch.bfloat16), ((1, 896, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply64,
        [((2016,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 2016, 7, 7), torch.bfloat16), ((2016, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((2016,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((2016,), torch.bfloat16), ((2016,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 2016, 14, 14), torch.bfloat16), ((2016, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 2016, 1, 1), torch.bfloat16), ((1, 2016, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 512, 28, 28), torch.bfloat16), ((512, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1024, 14, 14), torch.bfloat16), ((1024, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 64, 240, 320), torch.bfloat16), ((64, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf", "pt_yolo_v4_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 64, 120, 160), torch.bfloat16), ((64, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf", "pt_yolo_v4_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 256, 120, 160), torch.bfloat16), ((256, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 128, 120, 160), torch.bfloat16), ((128, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf", "pt_yolo_v4_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 128, 60, 80), torch.bfloat16), ((128, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf", "pt_yolo_v4_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 512, 60, 80), torch.bfloat16), ((512, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 256, 60, 80), torch.bfloat16), ((256, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf", "pt_yolo_v4_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 256, 30, 40), torch.bfloat16), ((256, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf", "pt_yolo_v4_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1024, 30, 40), torch.bfloat16), ((1024, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 512, 30, 40), torch.bfloat16), ((512, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf", "pt_yolo_v4_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 512, 15, 20), torch.bfloat16), ((512, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf", "pt_yolo_v4_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 2048, 15, 20), torch.bfloat16), ((2048, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_retinanet_retinanet_rn50fpn_obj_det_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply65,
        [((64, 3, 49, 32), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply65,
        [((16, 6, 49, 32), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply65,
        [((4, 12, 49, 32), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply65,
        [((1, 24, 49, 32), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply65,
        [((1, 12, 197, 197), torch.bfloat16)],
        {
            "model_names": [
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_vit_vit_b_16_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply65,
        [((1, 16, 50, 50), torch.bfloat16)],
        {"model_names": ["pt_vit_vit_l_32_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 256, 56, 56), torch.bfloat16), ((1, 256, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 160, 28, 28), torch.bfloat16), ((160, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 512, 28, 28), torch.bfloat16), ((1, 512, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 192, 14, 14), torch.bfloat16), ((192, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply66,
        [((768,), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 768, 14, 14), torch.bfloat16), ((768, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((768,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((768,), torch.bfloat16), ((768,), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 768, 14, 14), torch.bfloat16), ((1, 768, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 224, 7, 7), torch.bfloat16), ((224, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1024, 7, 7), torch.bfloat16), ((1, 1024, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply67,
        [((80,), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_yolov8_yolov8x_obj_det_github",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 80, 28, 28), torch.bfloat16), ((80, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_vovnet_vovnet27s_img_cls_osmr", "pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((80,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_yolov8_yolov8x_obj_det_github",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((80,), torch.bfloat16), ((80,), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_yolov8_yolov8x_obj_det_github",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 384, 14, 14), torch.bfloat16), ((384, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 112, 7, 7), torch.bfloat16), ((112, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 32, 150, 150), torch.bfloat16), ((32, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 64, 150, 150), torch.bfloat16), ((64, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_xception_xception41_img_cls_timm", "pt_xception_xception71_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 128, 150, 150), torch.bfloat16), ((128, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_xception_xception41_img_cls_timm", "pt_xception_xception71_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 128, 75, 75), torch.bfloat16), ((128, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_xception_xception41_img_cls_timm", "pt_xception_xception71_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 256, 75, 75), torch.bfloat16), ((256, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_xception_xception41_img_cls_timm", "pt_xception_xception71_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 256, 38, 38), torch.bfloat16), ((256, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_xception_xception41_img_cls_timm", "pt_xception_xception71_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply68,
        [((728,), torch.bfloat16)],
        {
            "model_names": [
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 728, 38, 38), torch.bfloat16), ((728, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_xception_xception41_img_cls_timm", "pt_xception_xception71_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((728,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((728,), torch.bfloat16), ((728,), torch.bfloat16)],
        {
            "model_names": [
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 728, 19, 19), torch.bfloat16), ((728, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1024, 19, 19), torch.bfloat16), ((1024, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1024, 10, 10), torch.bfloat16), ((1024, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1536, 10, 10), torch.bfloat16), ((1536, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 2048, 10, 10), torch.bfloat16), ((2048, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_xception_xception_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 64, 320, 320), torch.float32), ((1, 64, 320, 320), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 128, 160, 160), torch.float32), ((1, 128, 160, 160), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 64, 160, 160), torch.float32), ((1, 64, 160, 160), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 256, 80, 80), torch.float32), ((1, 256, 80, 80), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 128, 80, 80), torch.float32), ((1, 128, 80, 80), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 512, 40, 40), torch.float32), ((1, 512, 40, 40), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 256, 40, 40), torch.float32), ((1, 256, 40, 40), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 1024, 20, 20), torch.float32), ((1, 1024, 20, 20), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 512, 20, 20), torch.float32), ((1, 512, 20, 20), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 255, 80, 80), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 255, 80, 80), torch.float32), ((1, 255, 80, 80), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply69,
        [((1, 255, 80, 80), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply2,
        [((1, 255, 40, 40), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 255, 40, 40), torch.float32), ((1, 255, 40, 40), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply70,
        [((1, 255, 40, 40), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply2,
        [((1, 255, 20, 20), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 255, 20, 20), torch.float32), ((1, 255, 20, 20), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply71,
        [((1, 255, 20, 20), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 16, 240, 240), torch.float32), ((1, 16, 240, 240), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 32, 120, 120), torch.float32), ((1, 32, 120, 120), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 16, 120, 120), torch.float32), ((1, 16, 120, 120), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 64, 60, 60), torch.float32), ((1, 64, 60, 60), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 32, 60, 60), torch.float32), ((1, 32, 60, 60), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 128, 30, 30), torch.float32), ((1, 128, 30, 30), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 64, 30, 30), torch.float32), ((1, 64, 30, 30), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 256, 15, 15), torch.float32), ((1, 256, 15, 15), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 128, 15, 15), torch.float32), ((1, 128, 15, 15), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 255, 60, 60), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 255, 60, 60), torch.float32), ((1, 255, 60, 60), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Multiply72,
        [((1, 255, 60, 60), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 255, 30, 30), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 255, 30, 30), torch.float32), ((1, 255, 30, 30), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Multiply73,
        [((1, 255, 30, 30), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 255, 15, 15), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 255, 15, 15), torch.float32), ((1, 255, 15, 15), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Multiply74,
        [((1, 255, 15, 15), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 80, 320, 320), torch.bfloat16), ((80, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolov8_yolov8x_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 80, 320, 320), torch.bfloat16), ((1, 80, 320, 320), torch.bfloat16)],
        {"model_names": ["pt_yolov8_yolov8x_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 160, 160, 160), torch.bfloat16), ((160, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolov8_yolov8x_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 160, 160, 160), torch.bfloat16), ((1, 160, 160, 160), torch.bfloat16)],
        {"model_names": ["pt_yolov8_yolov8x_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 80, 160, 160), torch.bfloat16), ((80, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolov8_yolov8x_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 80, 160, 160), torch.bfloat16), ((1, 80, 160, 160), torch.bfloat16)],
        {"model_names": ["pt_yolov8_yolov8x_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply75,
        [((320,), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov8_yolov8x_obj_det_github",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 320, 80, 80), torch.bfloat16), ((320, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolov8_yolov8x_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((320,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov8_yolov8x_obj_det_github",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((320,), torch.bfloat16), ((320,), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov8_yolov8x_obj_det_github",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 320, 80, 80), torch.bfloat16), ((1, 320, 80, 80), torch.bfloat16)],
        {"model_names": ["pt_yolov8_yolov8x_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 160, 80, 80), torch.bfloat16), ((160, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolov8_yolov8x_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 160, 80, 80), torch.bfloat16), ((1, 160, 80, 80), torch.bfloat16)],
        {"model_names": ["pt_yolov8_yolov8x_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply76,
        [((640,), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov8_yolov8x_obj_det_github",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 640, 40, 40), torch.bfloat16), ((640, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolov8_yolov8x_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((640,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov8_yolov8x_obj_det_github",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((640,), torch.bfloat16), ((640,), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov8_yolov8x_obj_det_github",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 640, 40, 40), torch.bfloat16), ((1, 640, 40, 40), torch.bfloat16)],
        {"model_names": ["pt_yolov8_yolov8x_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 320, 40, 40), torch.bfloat16), ((320, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolov8_yolov8x_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 320, 40, 40), torch.bfloat16), ((1, 320, 40, 40), torch.bfloat16)],
        {"model_names": ["pt_yolov8_yolov8x_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 640, 20, 20), torch.bfloat16), ((640, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolov8_yolov8x_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 640, 20, 20), torch.bfloat16), ((1, 640, 20, 20), torch.bfloat16)],
        {"model_names": ["pt_yolov8_yolov8x_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 320, 20, 20), torch.bfloat16), ((320, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolov8_yolov8x_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 320, 20, 20), torch.bfloat16), ((1, 320, 20, 20), torch.bfloat16)],
        {"model_names": ["pt_yolov8_yolov8x_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 80, 80, 80), torch.bfloat16), ((80, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolov8_yolov8x_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 80, 80, 80), torch.bfloat16), ((1, 80, 80, 80), torch.bfloat16)],
        {"model_names": ["pt_yolov8_yolov8x_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 80, 40, 40), torch.bfloat16), ((80, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolov8_yolov8x_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 80, 40, 40), torch.bfloat16), ((1, 80, 40, 40), torch.bfloat16)],
        {"model_names": ["pt_yolov8_yolov8x_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 80, 20, 20), torch.bfloat16), ((80, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolov8_yolov8x_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 80, 20, 20), torch.bfloat16), ((1, 80, 20, 20), torch.bfloat16)],
        {"model_names": ["pt_yolov8_yolov8x_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply77,
        [((1, 4, 8400), torch.bfloat16)],
        {
            "model_names": ["pt_yolov8_yolov8x_obj_det_github", "pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 64, 320, 320), torch.bfloat16), ((64, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolov9_default_obj_det_github",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 64, 320, 320), torch.bfloat16), ((1, 64, 320, 320), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_l_obj_det_torchhub", "pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 128, 160, 160), torch.bfloat16), ((128, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolov9_default_obj_det_github",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 128, 160, 160), torch.bfloat16), ((1, 128, 160, 160), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_l_obj_det_torchhub", "pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 64, 160, 160), torch.bfloat16), ((1, 64, 160, 160), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_l_obj_det_torchhub", "pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 256, 80, 80), torch.bfloat16), ((256, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_yolov9_default_obj_det_github",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 256, 80, 80), torch.bfloat16), ((1, 256, 80, 80), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_l_obj_det_torchhub", "pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 128, 80, 80), torch.bfloat16), ((128, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_yolov9_default_obj_det_github",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 128, 80, 80), torch.bfloat16), ((1, 128, 80, 80), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_l_obj_det_torchhub", "pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 512, 40, 40), torch.bfloat16), ((512, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_yolov9_default_obj_det_github",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 512, 40, 40), torch.bfloat16), ((1, 512, 40, 40), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_l_obj_det_torchhub", "pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 256, 40, 40), torch.bfloat16), ((256, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_yolov9_default_obj_det_github",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 256, 40, 40), torch.bfloat16), ((1, 256, 40, 40), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_l_obj_det_torchhub", "pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1024, 20, 20), torch.bfloat16), ((1024, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1024, 20, 20), torch.bfloat16), ((1, 1024, 20, 20), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 512, 20, 20), torch.bfloat16), ((512, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_yolov9_default_obj_det_github",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 512, 20, 20), torch.bfloat16), ((1, 512, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_l_obj_det_torchhub", "pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 256, 20, 20), torch.bfloat16), ((1, 256, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_l_obj_det_torchhub", "pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 48, 320, 320), torch.bfloat16), ((48, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_m_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 48, 320, 320), torch.bfloat16), ((1, 48, 320, 320), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_m_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 96, 160, 160), torch.bfloat16), ((96, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_m_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 96, 160, 160), torch.bfloat16), ((1, 96, 160, 160), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_m_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 48, 160, 160), torch.bfloat16), ((48, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_m_obj_det_torchhub", "pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 48, 160, 160), torch.bfloat16), ((1, 48, 160, 160), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_m_obj_det_torchhub", "pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 192, 80, 80), torch.bfloat16), ((192, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_m_obj_det_torchhub", "pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 192, 80, 80), torch.bfloat16), ((1, 192, 80, 80), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_m_obj_det_torchhub", "pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 96, 80, 80), torch.bfloat16), ((96, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 96, 80, 80), torch.bfloat16), ((1, 96, 80, 80), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_m_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 384, 40, 40), torch.bfloat16), ((384, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_m_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 384, 40, 40), torch.bfloat16), ((1, 384, 40, 40), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_m_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 192, 40, 40), torch.bfloat16), ((192, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_m_obj_det_torchhub", "pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 192, 40, 40), torch.bfloat16), ((1, 192, 40, 40), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_m_obj_det_torchhub", "pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 768, 20, 20), torch.bfloat16), ((768, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_m_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 768, 20, 20), torch.bfloat16), ((1, 768, 20, 20), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_m_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 384, 20, 20), torch.bfloat16), ((384, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_m_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 384, 20, 20), torch.bfloat16), ((1, 384, 20, 20), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_m_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 192, 20, 20), torch.bfloat16), ((192, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_m_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 192, 20, 20), torch.bfloat16), ((1, 192, 20, 20), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_m_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 16384, 256), torch.float32), ((1, 16384, 256), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 16384, 256), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 4096, 512), torch.float32), ((1, 4096, 512), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 4096, 512), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 1024, 1280), torch.float32), ((1, 1024, 1280), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 1024, 1280), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 256, 2048), torch.float32), ((1, 256, 2048), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 256, 2048), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 9), torch.int64), ((1, 9), torch.int64)],
        {"model_names": ["pd_roberta_rbt4_ch_seq_cls_padlenlp"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 9, 768), torch.float32), ((1, 9, 768), torch.float32)],
        {
            "model_names": [
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply78,
        [((1, 9, 768), torch.float32)],
        {
            "model_names": [
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply2,
        [((1, 12, 9, 64), torch.float32)],
        {
            "model_names": [
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (Multiply2, [((1, 9), torch.float32)], {"model_names": ["pd_roberta_rbt4_ch_seq_cls_padlenlp"], "pcc": 0.99}),
    (
        Multiply65,
        [((1, 16, 197, 197), torch.bfloat16)],
        {
            "model_names": [
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_vit_l_16_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply79,
        [((1, 197, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 96, 56, 56), torch.bfloat16), ((96, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 160, 56, 56), torch.bfloat16), ((160, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 224, 28, 28), torch.bfloat16), ((224, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 320, 28, 28), torch.bfloat16), ((320, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply80,
        [((352,), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 352, 28, 28), torch.bfloat16), ((352, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((352,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((352,), torch.bfloat16), ((352,), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 384, 28, 28), torch.bfloat16), ((384, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply81,
        [((416,), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 416, 28, 28), torch.bfloat16), ((416, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((416,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((416,), torch.bfloat16), ((416,), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply82,
        [((480,), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 480, 28, 28), torch.bfloat16), ((480, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((480,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((480,), torch.bfloat16), ((480,), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 320, 14, 14), torch.bfloat16), ((320, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 352, 14, 14), torch.bfloat16), ((352, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 416, 14, 14), torch.bfloat16), ((416, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 448, 14, 14), torch.bfloat16), ((448, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 480, 14, 14), torch.bfloat16), ((480, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply83,
        [((544,), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 544, 14, 14), torch.bfloat16), ((544, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((544,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((544,), torch.bfloat16), ((544,), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply84,
        [((608,), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 608, 14, 14), torch.bfloat16), ((608, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((608,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((608,), torch.bfloat16), ((608,), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 640, 14, 14), torch.bfloat16), ((640, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply85,
        [((704,), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 704, 14, 14), torch.bfloat16), ((704, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((704,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((704,), torch.bfloat16), ((704,), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply86,
        [((736,), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 736, 14, 14), torch.bfloat16), ((736, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((736,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((736,), torch.bfloat16), ((736,), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply87,
        [((800,), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 800, 14, 14), torch.bfloat16), ((800, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((800,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((800,), torch.bfloat16), ((800,), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply88,
        [((832,), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 832, 14, 14), torch.bfloat16), ((832, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((832,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((832,), torch.bfloat16), ((832,), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply89,
        [((928,), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 928, 14, 14), torch.bfloat16), ((928, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((928,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((928,), torch.bfloat16), ((928,), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply90,
        [((992,), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 992, 14, 14), torch.bfloat16), ((992, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((992,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((992,), torch.bfloat16), ((992,), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 128, 7, 7), torch.bfloat16), ((128, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 544, 7, 7), torch.bfloat16), ((544, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 608, 7, 7), torch.bfloat16), ((608, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 640, 7, 7), torch.bfloat16), ((640, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 672, 7, 7), torch.bfloat16), ((672, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 704, 7, 7), torch.bfloat16), ((704, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 736, 7, 7), torch.bfloat16), ((736, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 768, 7, 7), torch.bfloat16), ((768, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 800, 7, 7), torch.bfloat16), ((800, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 832, 7, 7), torch.bfloat16), ((832, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 864, 7, 7), torch.bfloat16), ((864, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 896, 7, 7), torch.bfloat16), ((896, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 928, 7, 7), torch.bfloat16), ((928, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 992, 7, 7), torch.bfloat16), ((992, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 256, 112, 112), torch.bfloat16), ((256, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_dla_dla102x2_visual_bb_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 512, 56, 56), torch.bfloat16), ((512, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1024, 28, 28), torch.bfloat16), ((1024, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 2048, 14, 14), torch.bfloat16), ((2048, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 32, 28, 28), torch.bfloat16), ((32, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 64, 14, 14), torch.bfloat16), ((64, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 48, 160, 160), torch.bfloat16), ((1, 48, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 24, 160, 160), torch.bfloat16), ((24, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 24, 160, 160), torch.bfloat16), ((1, 24, 160, 160), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 24, 160, 160), torch.bfloat16), ((1, 24, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 144, 160, 160), torch.bfloat16), ((144, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 144, 160, 160), torch.bfloat16), ((1, 144, 160, 160), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 144, 80, 80), torch.bfloat16), ((144, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 144, 80, 80), torch.bfloat16), ((1, 144, 80, 80), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 144, 80, 80), torch.bfloat16), ((1, 144, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 32, 80, 80), torch.bfloat16), ((32, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 192, 80, 80), torch.bfloat16), ((1, 192, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 192, 40, 40), torch.bfloat16), ((1, 192, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 56, 40, 40), torch.bfloat16), ((56, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 336, 40, 40), torch.bfloat16), ((336, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 336, 40, 40), torch.bfloat16), ((1, 336, 40, 40), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 336, 40, 40), torch.bfloat16), ((1, 336, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 336, 20, 20), torch.bfloat16), ((336, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 336, 20, 20), torch.bfloat16), ((1, 336, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 336, 20, 20), torch.bfloat16), ((1, 336, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 112, 20, 20), torch.bfloat16), ((112, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 672, 20, 20), torch.bfloat16), ((672, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 672, 20, 20), torch.bfloat16), ((1, 672, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 672, 20, 20), torch.bfloat16), ((1, 672, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 160, 20, 20), torch.bfloat16), ((160, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 960, 20, 20), torch.bfloat16), ((960, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 960, 20, 20), torch.bfloat16), ((1, 960, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 960, 20, 20), torch.bfloat16), ((1, 960, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 960, 10, 10), torch.bfloat16), ((960, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 960, 10, 10), torch.bfloat16), ((1, 960, 10, 10), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 960, 10, 10), torch.bfloat16), ((1, 960, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 272, 10, 10), torch.bfloat16), ((272, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1632, 10, 10), torch.bfloat16), ((1632, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1632, 10, 10), torch.bfloat16), ((1, 1632, 10, 10), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1632, 10, 10), torch.bfloat16), ((1, 1632, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 448, 10, 10), torch.bfloat16), ((448, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 2688, 10, 10), torch.bfloat16), ((2688, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 2688, 10, 10), torch.bfloat16), ((1, 2688, 10, 10), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 2688, 10, 10), torch.bfloat16), ((1, 2688, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1792, 10, 10), torch.bfloat16), ((1792, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1792, 10, 10), torch.bfloat16), ((1, 1792, 10, 10), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 64, 112, 112), torch.bfloat16), ((1, 64, 112, 112), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 16, 1, 1), torch.bfloat16), ((1, 16, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 64, 1, 1), torch.bfloat16), ((1, 64, 112, 112), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 48, 56, 56), torch.bfloat16), ((48, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 288, 56, 56), torch.bfloat16), ((288, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 288, 56, 56), torch.bfloat16), ((1, 288, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 288, 1, 1), torch.bfloat16), ((1, 288, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 480, 28, 28), torch.bfloat16), ((1, 480, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 20, 1, 1), torch.bfloat16), ((1, 20, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 480, 1, 1), torch.bfloat16), ((1, 480, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 480, 14, 14), torch.bfloat16), ((1, 480, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 480, 1, 1), torch.bfloat16), ((1, 480, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 224, 14, 14), torch.bfloat16), ((224, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply91,
        [((1344,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_regnet_regnet_x_32gf_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1344, 14, 14), torch.bfloat16), ((1344, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_regnet_regnet_x_32gf_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1344,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_regnet_regnet_x_32gf_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1344,), torch.bfloat16), ((1344,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_regnet_regnet_x_32gf_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1344, 14, 14), torch.bfloat16), ((1, 1344, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 56, 1, 1), torch.bfloat16), ((1, 56, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1344, 1, 1), torch.bfloat16), ((1, 1344, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1344, 7, 7), torch.bfloat16), ((1344, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1344, 7, 7), torch.bfloat16), ((1, 1344, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1344, 1, 1), torch.bfloat16), ((1, 1344, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply92,
        [((3840,), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 3840, 7, 7), torch.bfloat16), ((3840, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((3840,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((3840,), torch.bfloat16), ((3840,), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 3840, 7, 7), torch.bfloat16), ((1, 3840, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 160, 1, 1), torch.bfloat16), ((1, 160, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 3840, 1, 1), torch.bfloat16), ((1, 3840, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply93,
        [((2560,), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 2560, 7, 7), torch.bfloat16), ((2560, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((2560,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((2560,), torch.bfloat16), ((2560,), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 2560, 7, 7), torch.bfloat16), ((1, 2560, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 256, 64, 64), torch.bfloat16), ((256, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_fpn_base_img_cls_torchvision", "pt_yolo_v3_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 256, 16, 16), torch.bfloat16), ((256, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_fpn_base_img_cls_torchvision", "pt_yolo_v3_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 256, 8, 8), torch.bfloat16), ((256, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_fpn_base_img_cls_torchvision",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 16, 112, 112), torch.bfloat16), ((16, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply94,
        [((8,), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 8, 112, 112), torch.bfloat16), ((8, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((8,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((8,), torch.bfloat16), ((8,), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply95,
        [((12,), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 12, 56, 56), torch.bfloat16), ((12, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((12,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((12,), torch.bfloat16), ((12,), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 16, 56, 56), torch.bfloat16), ((16, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 24, 56, 56), torch.bfloat16), ((24, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply96,
        [((36,), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 36, 56, 56), torch.bfloat16), ((36, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((36,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((36,), torch.bfloat16), ((36,), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 72, 28, 28), torch.bfloat16), ((1, 72, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply97,
        [((20,), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 20, 28, 28), torch.bfloat16), ((20, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((20,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((20,), torch.bfloat16), ((20,), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 24, 28, 28), torch.bfloat16), ((24, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 40, 28, 28), torch.bfloat16), ((40, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 120, 28, 28), torch.bfloat16), ((1, 120, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 240, 14, 14), torch.bfloat16), ((240, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 40, 14, 14), torch.bfloat16), ((40, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 80, 14, 14), torch.bfloat16), ((80, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply98,
        [((100,), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 100, 14, 14), torch.bfloat16), ((100, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((100,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((100,), torch.bfloat16), ((100,), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply99,
        [((92,), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 92, 14, 14), torch.bfloat16), ((92, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((92,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((92,), torch.bfloat16), ((92,), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 480, 14, 14), torch.bfloat16), ((1, 480, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 56, 14, 14), torch.bfloat16), ((56, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 672, 14, 14), torch.bfloat16), ((1, 672, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 672, 7, 7), torch.bfloat16), ((1, 672, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 80, 7, 7), torch.bfloat16), ((80, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 160, 7, 7), torch.bfloat16), ((160, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 480, 7, 7), torch.bfloat16), ((480, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_ghostnet_ghostnet_100_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 960, 7, 7), torch.bfloat16), ((1, 960, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 16, 28, 28), torch.bfloat16), ((16, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 16, 14, 14), torch.bfloat16), ((16, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 32, 14, 14), torch.bfloat16), ((32, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply100,
        [((18,), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 18, 56, 56), torch.bfloat16), ((18, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((18,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((18,), torch.bfloat16), ((18,), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 36, 28, 28), torch.bfloat16), ((36, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 18, 28, 28), torch.bfloat16), ((18, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 72, 14, 14), torch.bfloat16), ((72, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 18, 14, 14), torch.bfloat16), ((18, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 36, 14, 14), torch.bfloat16), ((36, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 144, 7, 7), torch.bfloat16), ((144, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 126, 7, 7), torch.bfloat16), ((126, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 24, 96, 96), torch.bfloat16), ((24, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 48, 96, 96), torch.bfloat16), ((48, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 48, 48, 48), torch.bfloat16), ((48, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 96, 48, 48), torch.bfloat16), ((96, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 96, 24, 24), torch.bfloat16), ((96, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 192, 24, 24), torch.bfloat16), ((192, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 192, 12, 12), torch.bfloat16), ((192, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 384, 12, 12), torch.bfloat16), ((384, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 384, 6, 6), torch.bfloat16), ((384, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 768, 6, 6), torch.bfloat16), ((768, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 24, 80, 80), torch.bfloat16), ((24, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 16, 80, 80), torch.bfloat16), ((16, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 96, 40, 40), torch.bfloat16), ((96, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 24, 40, 40), torch.bfloat16), ((24, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 144, 40, 40), torch.bfloat16), ((144, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 144, 20, 20), torch.bfloat16), ((144, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 24, 20, 20), torch.bfloat16), ((24, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 144, 10, 10), torch.bfloat16), ((144, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 48, 10, 10), torch.bfloat16), ((48, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 288, 10, 10), torch.bfloat16), ((288, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 72, 10, 10), torch.bfloat16), ((72, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 432, 10, 10), torch.bfloat16), ((432, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 432, 5, 5), torch.bfloat16), ((432, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 120, 5, 5), torch.bfloat16), ((120, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply101,
        [((720,), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 720, 5, 5), torch.bfloat16), ((720, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((720,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((720,), torch.bfloat16), ((720,), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 240, 5, 5), torch.bfloat16), ((240, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply102,
        [((1280,), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1280, 5, 5), torch.bfloat16), ((1280, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1280,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1280,), torch.bfloat16), ((1280,), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 96, 112, 112), torch.bfloat16), ((96, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 144, 28, 28), torch.bfloat16), ((144, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 320, 7, 7), torch.bfloat16), ((320, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_mobilenet_v2_img_cls_torchvision", "pt_mobilenetv2_basic_img_cls_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1280, 7, 7), torch.bfloat16), ((1280, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 16, 112, 112), torch.bfloat16), ((1, 16, 112, 112), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 72, 56, 56), torch.bfloat16), ((72, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 240, 14, 14), torch.bfloat16), ((1, 240, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 200, 14, 14), torch.bfloat16), ((1, 200, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply103,
        [((184,), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 184, 14, 14), torch.bfloat16), ((184, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((184,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((184,), torch.bfloat16), ((184,), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 184, 14, 14), torch.bfloat16), ((1, 184, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 672, 7, 7), torch.bfloat16), ((1, 672, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1280, 1, 1), torch.bfloat16), ((1, 1280, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 64, 96, 320), torch.bfloat16), ((64, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 64, 48, 160), torch.bfloat16), ((64, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 128, 24, 80), torch.bfloat16), ((128, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 256, 12, 40), torch.bfloat16), ((256, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 512, 6, 20), torch.bfloat16), ((512, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 32, 11, 32), torch.float32), ((1, 1, 11, 32), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 32, 11, 16), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 32, 11, 11), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 6, 1024), torch.float32), ((1, 6, 1024), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 6, 1024), torch.float32), ((1, 6, 1), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply104,
        [((1, 6, 1024), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 16, 6, 64), torch.float32), ((1, 1, 6, 64), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 16, 6, 32), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 16, 6, 6), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 1, 6, 6), torch.float32), ((1, 1, 6, 6), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply105,
        [((1, 1, 6, 6), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 6, 2816), torch.float32), ((1, 6, 2816), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 35, 1536), torch.float32), ((1, 35, 1536), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 35, 1536), torch.float32), ((1, 35, 1), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply106,
        [((1, 35, 1536), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 12, 35, 128), torch.float32), ((1, 1, 35, 128), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 12, 35, 64), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 2, 35, 128), torch.float32), ((1, 1, 35, 128), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 2, 35, 64), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 12, 35, 35), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 1, 35, 35), torch.float32), ((1, 1, 35, 35), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply107,
        [((1, 1, 35, 35), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 35, 8960), torch.float32), ((1, 35, 8960), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 29, 1536), torch.float32), ((1, 29, 1536), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 29, 1536), torch.float32), ((1, 29, 1), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply106,
        [((1, 29, 1536), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 12, 29, 128), torch.float32), ((1, 1, 29, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 12, 29, 64), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 2, 29, 128), torch.float32), ((1, 1, 29, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 2, 29, 64), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 12, 29, 29), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 1, 29, 29), torch.float32), ((1, 1, 29, 29), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf", "pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply108,
        [((1, 1, 29, 29), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf", "pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 29, 8960), torch.float32), ((1, 29, 8960), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 224, 56, 56), torch.bfloat16), ((1, 224, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 448, 28, 28), torch.bfloat16), ((1, 448, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply109,
        [((1232,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1232, 28, 28), torch.bfloat16), ((1232, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1232,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1232,), torch.bfloat16), ((1232,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1232, 14, 14), torch.bfloat16), ((1232, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1232, 14, 14), torch.bfloat16), ((1, 1232, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply110,
        [((3024,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 3024, 14, 14), torch.bfloat16), ((3024, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((3024,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((3024,), torch.bfloat16), ((3024,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 3024, 7, 7), torch.bfloat16), ((3024, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 3024, 7, 7), torch.bfloat16), ((1, 3024, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 232, 112, 112), torch.bfloat16), ((232, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 232, 56, 56), torch.bfloat16), ((232, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 232, 56, 56), torch.bfloat16), ((1, 232, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply111,
        [((696,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 696, 56, 56), torch.bfloat16), ((696, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((696,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((696,), torch.bfloat16), ((696,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 696, 28, 28), torch.bfloat16), ((696, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 696, 28, 28), torch.bfloat16), ((1, 696, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1392, 28, 28), torch.bfloat16), ((1392, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1392, 14, 14), torch.bfloat16), ((1392, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1392, 14, 14), torch.bfloat16), ((1, 1392, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply112,
        [((3712,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 3712, 14, 14), torch.bfloat16), ((3712, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((3712,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((3712,), torch.bfloat16), ((3712,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 3712, 7, 7), torch.bfloat16), ((3712, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 3712, 7, 7), torch.bfloat16), ((1, 3712, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 336, 56, 56), torch.bfloat16), ((336, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 336, 112, 112), torch.bfloat16), ((336, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 672, 28, 28), torch.bfloat16), ((672, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 672, 56, 56), torch.bfloat16), ((672, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1344, 28, 28), torch.bfloat16), ((1344, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply113,
        [((2520,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 2520, 7, 7), torch.bfloat16), ((2520, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((2520,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((2520,), torch.bfloat16), ((2520,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 2520, 14, 14), torch.bfloat16), ((2520, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply114,
        [((1008,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_3_2gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1008, 7, 7), torch.bfloat16), ((1008, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_3_2gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1008,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_3_2gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1008,), torch.bfloat16), ((1008,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_3_2gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1008, 14, 14), torch.bfloat16), ((1008, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_3_2gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((64, 3, 64, 32), torch.float32), ((64, 3, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((64, 3, 64, 64), torch.float32), ((3, 1, 1), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 3, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((16, 6, 64, 32), torch.float32), ((16, 6, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((16, 6, 64, 64), torch.float32), ((6, 1, 1), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 6, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((4, 12, 64, 32), torch.float32), ((4, 12, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((4, 12, 64, 64), torch.float32), ((12, 1, 1), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 12, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 24, 64, 32), torch.float32), ((1, 24, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 24, 64, 64), torch.float32), ((24, 1, 1), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 24, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 513, 768), torch.float32), ((1, 513, 768), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 513, 768), torch.float32), ((1, 513, 1), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (Multiply115, [((1, 513, 768), torch.float32)], {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99}),
    (
        Multiply0,
        [((1, 61, 768), torch.float32), ((1, 61, 768), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 61, 768), torch.float32), ((1, 61, 1), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (Multiply115, [((1, 61, 768), torch.float32)], {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99}),
    (Multiply2, [((1, 513, 768), torch.float32)], {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99}),
    (
        Multiply0,
        [((1, 64, 224, 224), torch.bfloat16), ((64, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_vgg_bn_vgg19b_obj_det_osmr"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 128, 112, 112), torch.bfloat16), ((128, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_vgg_bn_vgg19b_obj_det_osmr"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 32, 480, 640), torch.bfloat16), ((32, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 32, 480, 640), torch.bfloat16), ((1, 32, 480, 640), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 64, 240, 320), torch.bfloat16), ((1, 64, 240, 320), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 32, 240, 320), torch.bfloat16), ((32, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 32, 240, 320), torch.bfloat16), ((1, 32, 240, 320), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 128, 120, 160), torch.bfloat16), ((1, 128, 120, 160), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 64, 120, 160), torch.bfloat16), ((1, 64, 120, 160), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 256, 60, 80), torch.bfloat16), ((1, 256, 60, 80), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 128, 60, 80), torch.bfloat16), ((1, 128, 60, 80), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 512, 30, 40), torch.bfloat16), ((1, 512, 30, 40), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 256, 30, 40), torch.bfloat16), ((1, 256, 30, 40), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 1024, 15, 20), torch.bfloat16), ((1024, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 1024, 15, 20), torch.bfloat16), ((1, 1024, 15, 20), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 512, 15, 20), torch.bfloat16), ((1, 512, 15, 20), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 256, 15, 20), torch.bfloat16), ((256, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 128, 30, 40), torch.bfloat16), ((128, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 80, 160, 160), torch.float32), ((1, 80, 160, 160), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 160, 80, 80), torch.float32), ((1, 160, 80, 80), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 80, 80, 80), torch.float32), ((1, 80, 80, 80), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_320x320", "onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 320, 40, 40), torch.float32), ((1, 320, 40, 40), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 160, 40, 40), torch.float32), ((1, 160, 40, 40), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 640, 20, 20), torch.float32), ((1, 640, 20, 20), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 320, 20, 20), torch.float32), ((1, 320, 20, 20), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 1280, 10, 10), torch.float32), ((1, 1280, 10, 10), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 640, 10, 10), torch.float32), ((1, 640, 10, 10), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 255, 10, 10), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 255, 10, 10), torch.float32), ((1, 255, 10, 10), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply116,
        [((1, 255, 10, 10), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply117,
        [((1, 197, 768), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply65,
        [((1, 3, 197, 197), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 384, 28, 28), torch.bfloat16), ((1, 384, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 384, 1, 1), torch.bfloat16), ((1, 384, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b5_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 384, 14, 14), torch.bfloat16), ((1, 384, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b5_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 384, 1, 1), torch.bfloat16), ((1, 384, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b5_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 768, 14, 14), torch.bfloat16), ((1, 768, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b5_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 32, 1, 1), torch.bfloat16), ((1, 32, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 768, 1, 1), torch.bfloat16), ((1, 768, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b5_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply118,
        [((176,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 176, 14, 14), torch.bfloat16), ((176, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b5_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((176,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((176,), torch.bfloat16), ((176,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1056, 14, 14), torch.bfloat16), ((1056, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1056, 14, 14), torch.bfloat16), ((1, 1056, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 44, 1, 1), torch.bfloat16), ((1, 44, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1056, 1, 1), torch.bfloat16), ((1, 1056, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b5_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1056, 7, 7), torch.bfloat16), ((1056, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1056, 7, 7), torch.bfloat16), ((1, 1056, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b5_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1056, 1, 1), torch.bfloat16), ((1, 1056, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b5_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply119,
        [((304,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 304, 7, 7), torch.bfloat16), ((304, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b5_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((304,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((304,), torch.bfloat16), ((304,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply120,
        [((1824,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1824, 7, 7), torch.bfloat16), ((1824, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1824,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1824,), torch.bfloat16), ((1824,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1824, 7, 7), torch.bfloat16), ((1, 1824, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b5_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 76, 1, 1), torch.bfloat16), ((1, 76, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1824, 1, 1), torch.bfloat16), ((1, 1824, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b5_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply121,
        [((3072,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 3072, 7, 7), torch.bfloat16), ((3072, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b5_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((3072,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((3072,), torch.bfloat16), ((3072,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 3072, 7, 7), torch.bfloat16), ((1, 3072, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b5_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 128, 1, 1), torch.bfloat16), ((1, 128, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 3072, 1, 1), torch.bfloat16), ((1, 3072, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b5_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 2048, 7, 7), torch.bfloat16), ((1, 2048, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b5_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 24, 150, 150), torch.bfloat16), ((24, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 144, 150, 150), torch.bfloat16), ((144, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 144, 75, 75), torch.bfloat16), ((144, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 32, 75, 75), torch.bfloat16), ((32, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 192, 75, 75), torch.bfloat16), ((192, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 192, 38, 38), torch.bfloat16), ((192, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 48, 38, 38), torch.bfloat16), ((48, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 288, 38, 38), torch.bfloat16), ((288, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 288, 19, 19), torch.bfloat16), ((288, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 96, 19, 19), torch.bfloat16), ((96, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 576, 19, 19), torch.bfloat16), ((576, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 136, 19, 19), torch.bfloat16), ((136, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 816, 19, 19), torch.bfloat16), ((816, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 816, 10, 10), torch.bfloat16), ((816, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 232, 10, 10), torch.bfloat16), ((232, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1392, 10, 10), torch.bfloat16), ((1392, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 384, 10, 10), torch.bfloat16), ((384, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1280, 10, 10), torch.bfloat16), ((1280, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 32, 149, 149), torch.bfloat16), ((32, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 32, 147, 147), torch.bfloat16), ((32, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 64, 147, 147), torch.bfloat16), ((64, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 96, 73, 73), torch.bfloat16), ((96, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 64, 73, 73), torch.bfloat16), ((64, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 96, 71, 71), torch.bfloat16), ((96, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 192, 35, 35), torch.bfloat16), ((192, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 224, 35, 35), torch.bfloat16), ((224, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 96, 35, 35), torch.bfloat16), ((96, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 384, 17, 17), torch.bfloat16), ((384, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 256, 17, 17), torch.bfloat16), ((256, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 768, 17, 17), torch.bfloat16), ((768, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 224, 17, 17), torch.bfloat16), ((224, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 192, 17, 17), torch.bfloat16), ((192, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 128, 17, 17), torch.bfloat16), ((128, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 192, 8, 8), torch.bfloat16), ((192, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 320, 17, 17), torch.bfloat16), ((320, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 320, 8, 8), torch.bfloat16), ((320, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1024, 8, 8), torch.bfloat16), ((1024, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 448, 8, 8), torch.bfloat16), ((448, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 512, 8, 8), torch.bfloat16), ((512, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_inception_inception_v4_tf_in1k_img_cls_timm", "pt_inception_inception_v4_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 96, 28, 28), torch.bfloat16), ((96, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 576, 28, 28), torch.bfloat16), ((576, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 960, 28, 28), torch.bfloat16), ((960, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 256, 1, 1), torch.bfloat16), ((256, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 16, 56, 56), torch.bfloat16), ((1, 16, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply122,
        [((88,), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 88, 28, 28), torch.bfloat16), ((88, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((88,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((88,), torch.bfloat16), ((88,), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 96, 28, 28), torch.bfloat16), ((1, 96, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 96, 14, 14), torch.bfloat16), ((1, 96, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 96, 14, 14), torch.bfloat16), ((1, 96, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 240, 14, 14), torch.bfloat16), ((1, 240, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 120, 14, 14), torch.bfloat16), ((1, 120, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 120, 14, 14), torch.bfloat16), ((1, 120, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 48, 14, 14), torch.bfloat16), ((48, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 144, 14, 14), torch.bfloat16), ((1, 144, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 144, 14, 14), torch.bfloat16), ((1, 144, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 288, 7, 7), torch.bfloat16), ((288, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 288, 7, 7), torch.bfloat16), ((1, 288, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 288, 7, 7), torch.bfloat16), ((1, 288, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 96, 7, 7), torch.bfloat16), ((96, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 576, 7, 7), torch.bfloat16), ((1, 576, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 576, 7, 7), torch.bfloat16), ((1, 576, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1024, 1, 1), torch.bfloat16), ((1, 1024, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 29, 896), torch.float32), ((1, 29, 896), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 29, 896), torch.float32), ((1, 29, 1), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply123,
        [((1, 29, 896), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 14, 29, 64), torch.float32), ((1, 1, 29, 64), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 14, 29, 32), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 2, 29, 64), torch.float32), ((1, 1, 29, 64), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 2, 29, 32), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 14, 29, 29), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 29, 4864), torch.float32), ((1, 29, 4864), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 768, 128, 128), torch.bfloat16), ((768, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 128, 147, 147), torch.bfloat16), ((128, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_xception_xception_img_cls_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 128, 74, 74), torch.bfloat16), ((128, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_xception_xception_img_cls_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 256, 74, 74), torch.bfloat16), ((256, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_xception_xception_img_cls_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 256, 37, 37), torch.bfloat16), ((256, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_xception_xception_img_cls_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 728, 37, 37), torch.bfloat16), ((728, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_xception_xception_img_cls_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (Multiply2, [((1, 256, 1024), torch.float32)], {"model_names": ["pt_xglm_facebook_xglm_564m_clm_hf"], "pcc": 0.99}),
    (
        Multiply0,
        [((1, 32, 512, 512), torch.bfloat16), ((32, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v3_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 64, 256, 256), torch.bfloat16), ((64, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v3_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 32, 256, 256), torch.bfloat16), ((32, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v3_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 128, 128, 128), torch.bfloat16), ((128, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v3_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 64, 128, 128), torch.bfloat16), ((64, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v3_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 128, 64, 64), torch.bfloat16), ((128, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v3_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 512, 32, 32), torch.bfloat16), ((512, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v3_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 256, 32, 32), torch.bfloat16), ((256, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v3_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 1024, 16, 16), torch.bfloat16), ((1024, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v3_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 512, 16, 16), torch.bfloat16), ((512, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v3_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 128, 32, 32), torch.bfloat16), ((128, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolo_v3_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 48, 320, 320), torch.float32), ((1, 48, 320, 320), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 96, 160, 160), torch.float32), ((1, 96, 160, 160), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 48, 160, 160), torch.float32), ((1, 48, 160, 160), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 192, 80, 80), torch.float32), ((1, 192, 80, 80), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 96, 80, 80), torch.float32), ((1, 96, 80, 80), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 384, 40, 40), torch.float32), ((1, 384, 40, 40), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 192, 40, 40), torch.float32), ((1, 192, 40, 40), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 768, 20, 20), torch.float32), ((1, 768, 20, 20), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 384, 20, 20), torch.float32), ((1, 384, 20, 20), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 32, 160, 160), torch.float32), ((1, 32, 160, 160), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5s_img_cls_torchhub_320x320", "onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 64, 80, 80), torch.float32), ((1, 64, 80, 80), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5s_img_cls_torchhub_320x320", "onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 32, 80, 80), torch.float32), ((1, 32, 80, 80), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5s_img_cls_torchhub_320x320", "onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 128, 40, 40), torch.float32), ((1, 128, 40, 40), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5s_img_cls_torchhub_320x320", "onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 64, 40, 40), torch.float32), ((1, 64, 40, 40), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5s_img_cls_torchhub_320x320", "onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 256, 20, 20), torch.float32), ((1, 256, 20, 20), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5s_img_cls_torchhub_320x320", "onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 128, 20, 20), torch.float32), ((1, 128, 20, 20), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5s_img_cls_torchhub_320x320", "onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 512, 10, 10), torch.float32), ((1, 512, 10, 10), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5s_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 256, 10, 10), torch.float32), ((1, 256, 10, 10), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5s_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 64, 224, 320), torch.bfloat16), ((1, 64, 224, 320), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 128, 112, 160), torch.bfloat16), ((1, 128, 112, 160), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 64, 112, 160), torch.bfloat16), ((1, 64, 112, 160), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply124,
        [((1, 64, 112, 160), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 256, 56, 80), torch.bfloat16), ((1, 256, 56, 80), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 128, 56, 80), torch.bfloat16), ((1, 128, 56, 80), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply124,
        [((1, 128, 56, 80), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 512, 28, 40), torch.bfloat16), ((1, 512, 28, 40), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 256, 28, 40), torch.bfloat16), ((1, 256, 28, 40), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply124,
        [((1, 256, 28, 40), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 1024, 14, 20), torch.bfloat16), ((1, 1024, 14, 20), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 512, 14, 20), torch.bfloat16), ((1, 512, 14, 20), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply124,
        [((1, 512, 14, 20), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 128, 28, 40), torch.bfloat16), ((1, 128, 28, 40), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply124,
        [((1, 128, 28, 40), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 64, 56, 80), torch.bfloat16), ((1, 64, 56, 80), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply124,
        [((1, 64, 56, 80), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 256, 14, 20), torch.bfloat16), ((1, 256, 14, 20), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply124,
        [((1, 256, 14, 20), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply125,
        [((1, 5880, 4), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 32, 160, 160), torch.bfloat16), ((32, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolov9_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 32, 160, 160), torch.bfloat16), ((1, 32, 160, 160), torch.bfloat16)],
        {"model_names": ["pt_yolov9_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 256, 160, 160), torch.bfloat16), ((256, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolov9_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 256, 160, 160), torch.bfloat16), ((1, 256, 160, 160), torch.bfloat16)],
        {"model_names": ["pt_yolov9_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 64, 80, 80), torch.bfloat16), ((1, 64, 80, 80), torch.bfloat16)],
        {"model_names": ["pt_yolov9_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 512, 80, 80), torch.bfloat16), ((512, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolov9_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 512, 80, 80), torch.bfloat16), ((1, 512, 80, 80), torch.bfloat16)],
        {"model_names": ["pt_yolov9_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 128, 40, 40), torch.bfloat16), ((1, 128, 40, 40), torch.bfloat16)],
        {"model_names": ["pt_yolov9_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 128, 20, 20), torch.bfloat16), ((128, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolov9_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 128, 20, 20), torch.bfloat16), ((1, 128, 20, 20), torch.bfloat16)],
        {"model_names": ["pt_yolov9_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 64, 40, 40), torch.bfloat16), ((64, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolov9_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 64, 40, 40), torch.bfloat16), ((1, 64, 40, 40), torch.bfloat16)],
        {"model_names": ["pt_yolov9_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 64, 20, 20), torch.bfloat16), ((64, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolov9_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 64, 20, 20), torch.bfloat16), ((1, 64, 20, 20), torch.bfloat16)],
        {"model_names": ["pt_yolov9_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 32, 640, 640), torch.bfloat16), ((32, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_darknet_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 32, 320, 320), torch.bfloat16), ((32, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_darknet_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Multiply0,
        [((1, 112, 109, 64), torch.float32), ((64,), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 56, 54, 256), torch.float32), ((256,), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 56, 54, 64), torch.float32), ((64,), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 28, 27, 512), torch.float32), ((512,), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 28, 27, 128), torch.float32), ((128,), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 14, 14, 1024), torch.float32), ((1024,), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 14, 14, 256), torch.float32), ((256,), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 7, 7, 2048), torch.float32), ((2048,), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 7, 7, 512), torch.float32), ((512,), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 1, 1, 9), torch.float32)],
        {
            "model_names": ["pd_bert_chinese_roberta_base_mlm_padlenlp", "pd_bert_bert_base_uncased_mlm_padlenlp"],
            "pcc": 0.99,
        },
    ),
    (
        Multiply2,
        [((1, 16, 112, 112), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 16, 112, 112), torch.float32), ((1, 16, 112, 112), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 16, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 16, 1, 1), torch.float32), ((1, 16, 56, 56), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 96, 28, 28), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 96, 28, 28), torch.float32), ((1, 96, 28, 28), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 96, 14, 14), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 96, 14, 14), torch.float32), ((1, 96, 14, 14), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 96, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 96, 1, 1), torch.float32), ((1, 96, 14, 14), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 240, 14, 14), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 240, 14, 14), torch.float32), ((1, 240, 14, 14), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 240, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 240, 1, 1), torch.float32), ((1, 240, 14, 14), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 120, 14, 14), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 120, 14, 14), torch.float32), ((1, 120, 14, 14), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 120, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 120, 1, 1), torch.float32), ((1, 120, 14, 14), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 144, 14, 14), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 144, 14, 14), torch.float32), ((1, 144, 14, 14), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 144, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 144, 1, 1), torch.float32), ((1, 144, 14, 14), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 288, 14, 14), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 288, 14, 14), torch.float32), ((1, 288, 14, 14), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 288, 7, 7), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 288, 7, 7), torch.float32), ((1, 288, 7, 7), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 288, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Multiply0,
        [((1, 288, 1, 1), torch.float32), ((1, 288, 7, 7), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 576, 7, 7), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 576, 7, 7), torch.float32), ((1, 576, 7, 7), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 576, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 576, 1, 1), torch.float32), ((1, 576, 7, 7), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 1024), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 1024), torch.float32), ((1, 1024), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 3, 197, 64), torch.float32)],
        {"model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 3, 64, 197), torch.float32)],
        {"model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 197, 768), torch.float32), ((1, 197, 768), torch.float32)],
        {"model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 197, 768), torch.float32)],
        {"model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 16, 320, 320), torch.float32), ((1, 16, 320, 320), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 16, 160, 160), torch.float32), ((1, 16, 160, 160), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 80, 40, 40), torch.float32), ((1, 80, 40, 40), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 64, 20, 20), torch.float32), ((1, 64, 20, 20), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 80, 20, 20), torch.float32), ((1, 80, 20, 20), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Multiply126,
        [((1, 4, 8400), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (Multiply2, [((1, 9216), torch.float32)], {"model_names": ["pd_alexnet_base_img_cls_paddlemodels"], "pcc": 0.99}),
    (Multiply2, [((1, 4096), torch.float32)], {"model_names": ["pd_alexnet_base_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Multiply127,
        [((8,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 8, 16, 50), torch.float32), ((8, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((8,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((8,), torch.float32), ((8,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 8, 16, 50), torch.float32), ((1, 8, 16, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 8, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 8, 16, 50), torch.float32), ((1, 8, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply128,
        [((40,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 40, 16, 50), torch.float32), ((40, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((40,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((40,), torch.float32), ((40,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 40, 8, 50), torch.float32), ((40, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply129,
        [((16,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 16, 8, 50), torch.float32), ((16, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((16,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((16,), torch.float32), ((16,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply130,
        [((48,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 48, 8, 50), torch.float32), ((48, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((48,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((48,), torch.float32), ((48,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 48, 8, 50), torch.float32), ((1, 48, 8, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 48, 4, 50), torch.float32), ((48, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 48, 4, 50), torch.float32), ((1, 48, 4, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 48, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 48, 4, 50), torch.float32), ((1, 48, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply131,
        [((24,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 24, 4, 50), torch.float32), ((24, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((24,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((24,), torch.float32), ((24,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply132,
        [((120,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 120, 4, 50), torch.float32), ((120, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((120,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((120,), torch.float32), ((120,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 120, 4, 50), torch.float32), ((1, 120, 4, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 120, 4, 50), torch.float32), ((1, 120, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 64, 4, 50), torch.float32), ((64, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 64, 4, 50), torch.float32), ((1, 64, 4, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 64, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 64, 4, 50), torch.float32), ((1, 64, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply133,
        [((72,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 72, 4, 50), torch.float32), ((72, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((72,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((72,), torch.float32), ((72,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 72, 4, 50), torch.float32), ((1, 72, 4, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 72, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 72, 4, 50), torch.float32), ((1, 72, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply134,
        [((144,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 144, 4, 50), torch.float32), ((144, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((144,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((144,), torch.float32), ((144,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 144, 4, 50), torch.float32), ((1, 144, 4, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 144, 2, 50), torch.float32), ((144, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 144, 2, 50), torch.float32), ((1, 144, 2, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 144, 2, 50), torch.float32), ((1, 144, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 48, 2, 50), torch.float32), ((48, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply135,
        [((288,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 288, 2, 50), torch.float32), ((288, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((288,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((288,), torch.float32), ((288,), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 288, 2, 50), torch.float32), ((1, 288, 2, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 288, 2, 50), torch.float32), ((1, 288, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply136,
        [((1, 48), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 48), torch.float32), ((1, 48), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Multiply137,
        [((1, 256, 16, 32), torch.float32)],
        {"model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 256, 16, 16), torch.float32)],
        {"model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply138,
        [((1088,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1088, 14, 14), torch.bfloat16), ((1088, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1088,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1088,), torch.bfloat16), ((1088,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply139,
        [((1120,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1120, 14, 14), torch.bfloat16), ((1120, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1120,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1120,), torch.bfloat16), ((1120,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply140,
        [((1152,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1152, 14, 14), torch.bfloat16), ((1152, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1152,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1152,), torch.bfloat16), ((1152,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply141,
        [((1184,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1184, 14, 14), torch.bfloat16), ((1184, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1184,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1184,), torch.bfloat16), ((1184,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply142,
        [((1216,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1216, 14, 14), torch.bfloat16), ((1216, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1216,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1216,), torch.bfloat16), ((1216,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply143,
        [((1248,), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1248, 14, 14), torch.bfloat16), ((1248, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1248,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1248,), torch.bfloat16), ((1248,), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1280, 14, 14), torch.bfloat16), ((1280, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply144,
        [((1312,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1312, 14, 14), torch.bfloat16), ((1312, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1312,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1312,), torch.bfloat16), ((1312,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply145,
        [((1376,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1376, 14, 14), torch.bfloat16), ((1376, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1376,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1376,), torch.bfloat16), ((1376,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply146,
        [((1408,), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1408, 14, 14), torch.bfloat16), ((1408, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1408,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1408,), torch.bfloat16), ((1408,), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply147,
        [((1440,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1440, 14, 14), torch.bfloat16), ((1440, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1440,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1440,), torch.bfloat16), ((1440,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply148,
        [((1472,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1472, 14, 14), torch.bfloat16), ((1472, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1472,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1472,), torch.bfloat16), ((1472,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply149,
        [((1504,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1504, 14, 14), torch.bfloat16), ((1504, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1504,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1504,), torch.bfloat16), ((1504,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1536, 14, 14), torch.bfloat16), ((1536, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply150,
        [((1568,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1568, 14, 14), torch.bfloat16), ((1568, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1568,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1568,), torch.bfloat16), ((1568,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply151,
        [((1600,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1600, 14, 14), torch.bfloat16), ((1600, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1600,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1600,), torch.bfloat16), ((1600,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1632, 14, 14), torch.bfloat16), ((1632, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply152,
        [((1664,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1664, 14, 14), torch.bfloat16), ((1664, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1664,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1664,), torch.bfloat16), ((1664,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply153,
        [((1696,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1696, 14, 14), torch.bfloat16), ((1696, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1696,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1696,), torch.bfloat16), ((1696,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply154,
        [((1728,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1728, 14, 14), torch.bfloat16), ((1728, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1728,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1728,), torch.bfloat16), ((1728,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply155,
        [((1760,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1760, 14, 14), torch.bfloat16), ((1760, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1760,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1760,), torch.bfloat16), ((1760,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1792, 14, 14), torch.bfloat16), ((1792, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1088, 7, 7), torch.bfloat16), ((1088, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1120, 7, 7), torch.bfloat16), ((1120, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1152, 7, 7), torch.bfloat16), ((1152, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1184, 7, 7), torch.bfloat16), ((1184, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1216, 7, 7), torch.bfloat16), ((1216, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1248, 7, 7), torch.bfloat16), ((1248, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1312, 7, 7), torch.bfloat16), ((1312, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1376, 7, 7), torch.bfloat16), ((1376, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1408, 7, 7), torch.bfloat16), ((1408, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1440, 7, 7), torch.bfloat16), ((1440, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1472, 7, 7), torch.bfloat16), ((1472, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1504, 7, 7), torch.bfloat16), ((1504, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1568, 7, 7), torch.bfloat16), ((1568, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1600, 7, 7), torch.bfloat16), ((1600, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1664, 7, 7), torch.bfloat16), ((1664, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1696, 7, 7), torch.bfloat16), ((1696, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1728, 7, 7), torch.bfloat16), ((1728, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1760, 7, 7), torch.bfloat16), ((1760, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply156,
        [((1856,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1856, 7, 7), torch.bfloat16), ((1856, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1856,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1856,), torch.bfloat16), ((1856,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply157,
        [((1888,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1888, 7, 7), torch.bfloat16), ((1888, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1888,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1888,), torch.bfloat16), ((1888,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply158,
        [((1920,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1920, 7, 7), torch.bfloat16), ((1920, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1920,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1920,), torch.bfloat16), ((1920,), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 4, 1, 1), torch.bfloat16), ((1, 4, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 16, 1, 1), torch.bfloat16), ((1, 16, 112, 112), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 96, 112, 112), torch.bfloat16), ((1, 96, 112, 112), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 96, 56, 56), torch.bfloat16), ((1, 96, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 96, 1, 1), torch.bfloat16), ((1, 96, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 144, 28, 28), torch.bfloat16), ((1, 144, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 144, 1, 1), torch.bfloat16), ((1, 144, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 88, 14, 14), torch.bfloat16), ((88, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 528, 14, 14), torch.bfloat16), ((528, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 528, 14, 14), torch.bfloat16), ((1, 528, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 22, 1, 1), torch.bfloat16), ((1, 22, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 528, 1, 1), torch.bfloat16), ((1, 528, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 720, 14, 14), torch.bfloat16), ((720, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 720, 14, 14), torch.bfloat16), ((1, 720, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 30, 1, 1), torch.bfloat16), ((1, 30, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 720, 1, 1), torch.bfloat16), ((1, 720, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 720, 7, 7), torch.bfloat16), ((720, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 720, 7, 7), torch.bfloat16), ((1, 720, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 720, 1, 1), torch.bfloat16), ((1, 720, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply159,
        [((208,), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 208, 7, 7), torch.bfloat16), ((208, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((208,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((208,), torch.bfloat16), ((208,), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1248, 7, 7), torch.bfloat16), ((1, 1248, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 52, 1, 1), torch.bfloat16), ((1, 52, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1248, 1, 1), torch.bfloat16), ((1, 1248, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 352, 7, 7), torch.bfloat16), ((352, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply160,
        [((2112,), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 2112, 7, 7), torch.bfloat16), ((2112, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((2112,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((2112,), torch.bfloat16), ((2112,), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 2112, 7, 7), torch.bfloat16), ((1, 2112, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 88, 1, 1), torch.bfloat16), ((1, 88, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 2112, 1, 1), torch.bfloat16), ((1, 2112, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1408, 7, 7), torch.bfloat16), ((1, 1408, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 48, 224, 224), torch.bfloat16), ((48, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 48, 224, 224), torch.bfloat16), ((1, 48, 224, 224), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 48, 224, 224), torch.bfloat16), ((1, 48, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 24, 224, 224), torch.bfloat16), ((24, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 24, 224, 224), torch.bfloat16), ((1, 24, 224, 224), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 24, 224, 224), torch.bfloat16), ((1, 24, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 144, 224, 224), torch.bfloat16), ((144, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 144, 224, 224), torch.bfloat16), ((1, 144, 224, 224), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 144, 112, 112), torch.bfloat16), ((1, 144, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 240, 112, 112), torch.bfloat16), ((240, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 240, 112, 112), torch.bfloat16), ((1, 240, 112, 112), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 240, 112, 112), torch.bfloat16), ((1, 240, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 240, 56, 56), torch.bfloat16), ((1, 240, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 384, 56, 56), torch.bfloat16), ((384, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 384, 56, 56), torch.bfloat16), ((1, 384, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 384, 56, 56), torch.bfloat16), ((1, 384, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 384, 28, 28), torch.bfloat16), ((1, 384, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 768, 28, 28), torch.bfloat16), ((768, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 768, 28, 28), torch.bfloat16), ((1, 768, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 768, 28, 28), torch.bfloat16), ((1, 768, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 176, 28, 28), torch.bfloat16), ((176, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1056, 28, 28), torch.bfloat16), ((1056, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1056, 28, 28), torch.bfloat16), ((1, 1056, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1056, 28, 28), torch.bfloat16), ((1, 1056, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1056, 14, 14), torch.bfloat16), ((1, 1056, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 304, 14, 14), torch.bfloat16), ((304, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1824, 14, 14), torch.bfloat16), ((1824, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1824, 14, 14), torch.bfloat16), ((1, 1824, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 1824, 14, 14), torch.bfloat16), ((1, 1824, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 3072, 14, 14), torch.bfloat16), ((3072, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 3072, 14, 14), torch.bfloat16), ((1, 3072, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 3072, 14, 14), torch.bfloat16), ((1, 3072, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply0,
        [((1, 2048, 14, 14), torch.bfloat16), ((1, 2048, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply2,
        [((1, 12, 256, 256), torch.float32)],
        {"model_names": ["pt_gpt_gpt2_text_gen_hf", "pt_opt_facebook_opt_125m_clm_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 256), torch.int64), ((1, 256), torch.int64)],
        {"model_names": ["pt_opt_facebook_opt_125m_clm_hf"], "pcc": 0.99},
    ),
    (Multiply2, [((1, 256, 768), torch.float32)], {"model_names": ["pt_opt_facebook_opt_125m_clm_hf"], "pcc": 0.99}),
    (
        Multiply0,
        [((1, 32), torch.int64), ((1, 32), torch.int64)],
        {"model_names": ["pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99},
    ),
    (Multiply2, [((1, 32, 768), torch.float32)], {"model_names": ["pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99}),
    (
        Multiply2,
        [((1, 12, 32, 32), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99},
    ),
    (Multiply161, [((1, 32), torch.int32)], {"model_names": ["pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99}),
    (
        Multiply0,
        [((1, 32, 12, 32), torch.float32), ((1, 1, 12, 32), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 32, 12, 16), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply2,
        [((1, 32, 12, 12), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 513, 512), torch.float32), ((1, 513, 512), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 513, 512), torch.float32), ((1, 513, 1), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (Multiply162, [((1, 513, 512), torch.float32)], {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99}),
    (
        Multiply0,
        [((1, 61, 512), torch.float32), ((1, 61, 512), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Multiply0,
        [((1, 61, 512), torch.float32), ((1, 61, 1), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (Multiply162, [((1, 61, 512), torch.float32)], {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99}),
    (Multiply2, [((1, 513, 512), torch.float32)], {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99}),
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
