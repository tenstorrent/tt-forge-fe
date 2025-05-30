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


class Add0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, add_input_0, add_input_1):
        add_output_1 = forge.op.Add("", add_input_0, add_input_1)
        return add_output_1


class Add1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add1.weight_0", forge.Parameter(*(768,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add1.weight_0"), add_input_1)
        return add_output_1


class Add2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add2.weight_0", forge.Parameter(*(3072,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add2.weight_0"), add_input_1)
        return add_output_1


class Add3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add3.weight_1", forge.Parameter(*(768,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add3.weight_1"))
        return add_output_1


class Add4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add4.weight_1", forge.Parameter(*(16,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add4.weight_1"))
        return add_output_1


class Add5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add5.weight_1", forge.Parameter(*(1,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add5.weight_1"))
        return add_output_1


class Add6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add6_const_1", shape=(1,), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add6_const_1"))
        return add_output_1


class Add7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add7.weight_1", forge.Parameter(*(24,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add7.weight_1"))
        return add_output_1


class Add8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add8_const_0", shape=(1, 100, 256), dtype=torch.float32)

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_constant("add8_const_0"), add_input_1)
        return add_output_1


class Add9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add9.weight_0", forge.Parameter(*(256,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add9.weight_0"), add_input_1)
        return add_output_1


class Add10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add10.weight_1",
            forge.Parameter(*(1, 64, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add10.weight_1"))
        return add_output_1


class Add11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add11.weight_1",
            forge.Parameter(*(1, 256, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add11.weight_1"))
        return add_output_1


class Add12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add12.weight_1",
            forge.Parameter(*(1, 128, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add12.weight_1"))
        return add_output_1


class Add13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add13.weight_1",
            forge.Parameter(*(1, 512, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add13.weight_1"))
        return add_output_1


class Add14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add14.weight_1",
            forge.Parameter(*(1, 1024, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add14.weight_1"))
        return add_output_1


class Add15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add15.weight_1",
            forge.Parameter(*(1, 2048, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add15.weight_1"))
        return add_output_1


class Add16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add16_const_1", shape=(1, 280, 256), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add16_const_1"))
        return add_output_1


class Add17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add17_const_1", shape=(1, 1, 280, 280), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add17_const_1"))
        return add_output_1


class Add18(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add18.weight_0", forge.Parameter(*(2048,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add18.weight_0"), add_input_1)
        return add_output_1


class Add19(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add19_const_1", shape=(1, 1, 100, 280), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add19_const_1"))
        return add_output_1


class Add20(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add20.weight_0", forge.Parameter(*(92,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add20.weight_0"), add_input_1)
        return add_output_1


class Add21(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add21.weight_0", forge.Parameter(*(4,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add21.weight_0"), add_input_1)
        return add_output_1


class Add22(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add22.weight_1", forge.Parameter(*(1000,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add22.weight_1"))
        return add_output_1


class Add23(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add23.weight_0", forge.Parameter(*(64,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add23.weight_0"), add_input_1)
        return add_output_1


class Add24(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add24.weight_0", forge.Parameter(*(128,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add24.weight_0"), add_input_1)
        return add_output_1


class Add25(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add25.weight_0", forge.Parameter(*(512,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add25.weight_0"), add_input_1)
        return add_output_1


class Add26(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add26.weight_0", forge.Parameter(*(320,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add26.weight_0"), add_input_1)
        return add_output_1


class Add27(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add27.weight_0", forge.Parameter(*(1280,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add27.weight_0"), add_input_1)
        return add_output_1


class Add28(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add28.weight_1", forge.Parameter(*(3072,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add28.weight_1"))
        return add_output_1


class Add29(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add29.weight_1", forge.Parameter(*(30522,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add29.weight_1"))
        return add_output_1


class Add30(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add30.weight_1", forge.Parameter(*(2,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add30.weight_1"))
        return add_output_1


class Add31(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add31.weight_1", forge.Parameter(*(18000,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add31.weight_1"))
        return add_output_1


class Add32(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add32.weight_1", forge.Parameter(*(60,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add32.weight_1"))
        return add_output_1


class Add33(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add33.weight_1", forge.Parameter(*(120,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add33.weight_1"))
        return add_output_1


class Add34(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add34.weight_1", forge.Parameter(*(360,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add34.weight_1"))
        return add_output_1


class Add35(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add35.weight_1", forge.Parameter(*(240,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add35.weight_1"))
        return add_output_1


class Add36(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add36.weight_1", forge.Parameter(*(480,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add36.weight_1"))
        return add_output_1


class Add37(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add37.weight_1", forge.Parameter(*(97,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add37.weight_1"))
        return add_output_1


class Add38(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add38.weight_1", forge.Parameter(*(64,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add38.weight_1"))
        return add_output_1


class Add39(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add39.weight_1", forge.Parameter(*(256,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add39.weight_1"))
        return add_output_1


class Add40(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add40.weight_1", forge.Parameter(*(128,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add40.weight_1"))
        return add_output_1


class Add41(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add41.weight_1", forge.Parameter(*(512,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add41.weight_1"))
        return add_output_1


class Add42(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add42.weight_1", forge.Parameter(*(1024,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add42.weight_1"))
        return add_output_1


class Add43(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add43.weight_1", forge.Parameter(*(2048,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add43.weight_1"))
        return add_output_1


class Add44(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add44.weight_1", forge.Parameter(*(30000,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add44.weight_1"))
        return add_output_1


class Add45(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add45.weight_1", forge.Parameter(*(8192,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add45.weight_1"))
        return add_output_1


class Add46(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add46.weight_1", forge.Parameter(*(4096,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add46.weight_1"))
        return add_output_1


class Add47(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add47.weight_1", forge.Parameter(*(16384,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add47.weight_1"))
        return add_output_1


class Add48(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add48_const_1", shape=(1, 1, 256, 256), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add48_const_1"))
        return add_output_1


class Add49(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add49_const_1", shape=(1, 1, 1, 128), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add49_const_1"))
        return add_output_1


class Add50(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add50.weight_1", forge.Parameter(*(9,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add50.weight_1"))
        return add_output_1


class Add51(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add51_const_0", shape=(1, 1, 256, 256), dtype=torch.float32)

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_constant("add51_const_0"), add_input_1)
        return add_output_1


class Add52(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add52.weight_1", forge.Parameter(*(51200,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add52.weight_1"))
        return add_output_1


class Add53(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add53_const_0", shape=(1, 1, 32, 32), dtype=torch.float32)

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_constant("add53_const_0"), add_input_1)
        return add_output_1


class Add54(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add54_const_1", shape=(1,), dtype=torch.int64)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add54_const_1"))
        return add_output_1


class Add55(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add55.weight_1", forge.Parameter(*(2560,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add55.weight_1"))
        return add_output_1


class Add56(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add56.weight_1", forge.Parameter(*(10240,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add56.weight_1"))
        return add_output_1


class Add57(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add57.weight_1",
            forge.Parameter(*(1500, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add57.weight_1"))
        return add_output_1


class Add58(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add58.weight_0", forge.Parameter(*(1024,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add58.weight_0"), add_input_1)
        return add_output_1


class Add59(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add59_const_1", shape=(1, 1, 1, 384), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add59_const_1"))
        return add_output_1


class Add60(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add60.weight_0", forge.Parameter(*(4096,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add60.weight_0"), add_input_1)
        return add_output_1


class Add61(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add61.weight_0", forge.Parameter(*(384,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add61.weight_0"), add_input_1)
        return add_output_1


class Add62(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add62_const_1", shape=(1, 1, 1, 13), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add62_const_1"))
        return add_output_1


class Add63(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add63.weight_0", forge.Parameter(*(1536,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add63.weight_0"), add_input_1)
        return add_output_1


class Add64(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add64.weight_1", forge.Parameter(*(384,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add64.weight_1"))
        return add_output_1


class Add65(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add65.weight_1", forge.Parameter(*(32000,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add65.weight_1"))
        return add_output_1


class Add66(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add66.weight_1", forge.Parameter(*(21128,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add66.weight_1"))
        return add_output_1


class Add67(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add67.weight_1", forge.Parameter(*(4608,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add67.weight_1"))
        return add_output_1


class Add68(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add68.weight_1", forge.Parameter(*(1536,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add68.weight_1"))
        return add_output_1


class Add69(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add69.weight_1", forge.Parameter(*(6144,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add69.weight_1"))
        return add_output_1


class Add70(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add70_const_1", shape=(2, 1, 7, 7), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add70_const_1"))
        return add_output_1


class Add71(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add71_const_1", shape=(1, 1, 4, 4), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add71_const_1"))
        return add_output_1


class Add72(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add72_const_0", shape=(1, 1, 7, 7), dtype=torch.float32)

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_constant("add72_const_0"), add_input_1)
        return add_output_1


class Add73(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add73.weight_1", forge.Parameter(*(3129,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add73.weight_1"))
        return add_output_1


class Add74(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add74.weight_1",
            forge.Parameter(*(1500, 384), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add74.weight_1"))
        return add_output_1


class Add75(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add75.weight_0", forge.Parameter(*(251,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add75.weight_0"), add_input_1)
        return add_output_1


class Add76(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add76_const_0", shape=(1,), dtype=torch.float32)

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_constant("add76_const_0"), add_input_1)
        return add_output_1


class Add77(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add77_const_1", shape=(8, 1), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add77_const_1"))
        return add_output_1


class Add78(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add78.weight_1",
            forge.Parameter(*(264, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add78.weight_1"))
        return add_output_1


class Add79(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add79.weight_1",
            forge.Parameter(*(128, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add79.weight_1"))
        return add_output_1


class Add80(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add80.weight_1", forge.Parameter(*(64, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add80.weight_1"))
        return add_output_1


class Add81(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add81.weight_1", forge.Parameter(*(32, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add81.weight_1"))
        return add_output_1


class Add82(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add82.weight_1", forge.Parameter(*(16, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add82.weight_1"))
        return add_output_1


class Add83(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add83.weight_0", forge.Parameter(*(32,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add83.weight_0"), add_input_1)
        return add_output_1


class Add84(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add84.weight_0", forge.Parameter(*(160,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add84.weight_0"), add_input_1)
        return add_output_1


class Add85(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add85.weight_0", forge.Parameter(*(640,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add85.weight_0"), add_input_1)
        return add_output_1


class Add86(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add86.weight_1",
            forge.Parameter(*(1, 197, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add86.weight_1"))
        return add_output_1


class Add87(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add87_const_0", shape=(1, 2, 8400), dtype=torch.float32)

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_constant("add87_const_0"), add_input_1)
        return add_output_1


class Add88(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add88.weight_1", forge.Parameter(*(312,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add88.weight_1"))
        return add_output_1


class Add89(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add89.weight_1", forge.Parameter(*(1248,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add89.weight_1"))
        return add_output_1


class Add90(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add90.weight_1", forge.Parameter(*(8,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add90.weight_1"))
        return add_output_1


class Add91(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add91.weight_1", forge.Parameter(*(40,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add91.weight_1"))
        return add_output_1


class Add92(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add92.weight_1", forge.Parameter(*(48,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add92.weight_1"))
        return add_output_1


class Add93(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add93.weight_1", forge.Parameter(*(72,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add93.weight_1"))
        return add_output_1


class Add94(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add94.weight_1", forge.Parameter(*(144,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add94.weight_1"))
        return add_output_1


class Add95(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add95.weight_1", forge.Parameter(*(288,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add95.weight_1"))
        return add_output_1


class Add96(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add96.weight_1", forge.Parameter(*(192,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add96.weight_1"))
        return add_output_1


class Add97(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add97.weight_1", forge.Parameter(*(6625,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add97.weight_1"))
        return add_output_1


class Add98(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add98.weight_1", forge.Parameter(*(1280,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add98.weight_1"))
        return add_output_1


class Add99(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add99_const_1", shape=(1, 1, 2, 2), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add99_const_1"))
        return add_output_1


class Add100(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add100_const_1", shape=(1500, 1280), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add100_const_1"))
        return add_output_1


class Add101(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add101.weight_1", forge.Parameter(*(5120,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add101.weight_1"))
        return add_output_1


class Add102(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add102.weight_0", forge.Parameter(*(30522,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add102.weight_0"), add_input_1)
        return add_output_1


class Add103(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add103.weight_0", forge.Parameter(*(96,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add103.weight_0"), add_input_1)
        return add_output_1


class Add104(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add104_const_1", shape=(64, 1, 64, 64), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add104_const_1"))
        return add_output_1


class Add105(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add105.weight_0", forge.Parameter(*(192,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add105.weight_0"), add_input_1)
        return add_output_1


class Add106(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add106_const_1", shape=(16, 1, 64, 64), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add106_const_1"))
        return add_output_1


class Add107(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add107_const_1", shape=(4, 1, 64, 64), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add107_const_1"))
        return add_output_1


class Add108(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add108.weight_1",
            forge.Parameter(*(1500, 1280), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add108.weight_1"))
        return add_output_1


class Add109(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add109.weight_0", forge.Parameter(*(5120,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_1):
        add_output_1 = forge.op.Add("", self.get_parameter("add109.weight_0"), add_input_1)
        return add_output_1


class Add110(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add110_const_1", shape=(1, 1, 1, 8), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add110_const_1"))
        return add_output_1


class Add111(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add111.weight_1", forge.Parameter(*(96,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add111.weight_1"))
        return add_output_1


class Add112(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add112.weight_1", forge.Parameter(*(160,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add112.weight_1"))
        return add_output_1


class Add113(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add113.weight_1", forge.Parameter(*(224,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add113.weight_1"))
        return add_output_1


class Add114(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add114.weight_1", forge.Parameter(*(320,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add114.weight_1"))
        return add_output_1


class Add115(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add115.weight_1", forge.Parameter(*(352,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add115.weight_1"))
        return add_output_1


class Add116(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add116.weight_1", forge.Parameter(*(416,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add116.weight_1"))
        return add_output_1


class Add117(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add117.weight_1", forge.Parameter(*(448,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add117.weight_1"))
        return add_output_1


class Add118(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add118.weight_1", forge.Parameter(*(544,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add118.weight_1"))
        return add_output_1


class Add119(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add119.weight_1", forge.Parameter(*(576,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add119.weight_1"))
        return add_output_1


class Add120(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add120.weight_1", forge.Parameter(*(608,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add120.weight_1"))
        return add_output_1


class Add121(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add121.weight_1", forge.Parameter(*(640,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add121.weight_1"))
        return add_output_1


class Add122(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add122.weight_1", forge.Parameter(*(672,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add122.weight_1"))
        return add_output_1


class Add123(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add123.weight_1", forge.Parameter(*(704,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add123.weight_1"))
        return add_output_1


class Add124(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add124.weight_1", forge.Parameter(*(736,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add124.weight_1"))
        return add_output_1


class Add125(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add125.weight_1", forge.Parameter(*(800,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add125.weight_1"))
        return add_output_1


class Add126(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add126.weight_1", forge.Parameter(*(832,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add126.weight_1"))
        return add_output_1


class Add127(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add127.weight_1", forge.Parameter(*(864,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add127.weight_1"))
        return add_output_1


class Add128(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add128.weight_1", forge.Parameter(*(896,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add128.weight_1"))
        return add_output_1


class Add129(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add129.weight_1", forge.Parameter(*(928,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add129.weight_1"))
        return add_output_1


class Add130(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add130.weight_1", forge.Parameter(*(960,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add130.weight_1"))
        return add_output_1


class Add131(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add131.weight_1", forge.Parameter(*(992,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add131.weight_1"))
        return add_output_1


class Add132(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add132.weight_1", forge.Parameter(*(32,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add132.weight_1"))
        return add_output_1


class Add133(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add133_const_1", shape=(1, 1, 588, 588), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add133_const_1"))
        return add_output_1


class Add134(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add134.weight_1", forge.Parameter(*(28996,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add134.weight_1"))
        return add_output_1


class Add135(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add135.weight_1",
            forge.Parameter(*(1500, 512), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add135.weight_1"))
        return add_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Add0,
        [((1, 6, 768), torch.float32), ((1, 6, 768), torch.float32)],
        {
            "model_names": [
                "onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add1,
        [((1, 6, 768), torch.float32)],
        {
            "model_names": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 12, 6, 6), torch.float32), ((1, 1, 1, 6), torch.float32)],
        {
            "model_names": [
                "onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add2,
        [((1, 6, 3072), torch.float32)],
        {
            "model_names": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Add3,
        [((1, 768), torch.float32)],
        {
            "model_names": [
                "onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_vilt_dandelin_vilt_b32_mlm_mlm_hf",
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
                "pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf",
                "pd_bert_bert_base_japanese_seq_cls_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
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
        Add4,
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
        Add0,
        [((1, 16, 240, 240), torch.float32), ((16, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99},
    ),
    (Add5, [((1, 16, 240, 240), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (Add6, [((1, 16, 240, 240), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (
        Add0,
        [((1, 32, 240, 240), torch.float32), ((32, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99},
    ),
    (Add5, [((1, 32, 240, 240), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (Add6, [((1, 32, 240, 240), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (
        Add0,
        [((1, 32, 120, 120), torch.float32), ((32, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer", "onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (Add5, [((1, 32, 120, 120), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (
        Add0,
        [((1, 48, 120, 120), torch.float32), ((48, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99},
    ),
    (Add5, [((1, 48, 120, 120), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (Add6, [((1, 48, 120, 120), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (
        Add0,
        [((1, 48, 60, 60), torch.float32), ((48, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99},
    ),
    (Add5, [((1, 48, 60, 60), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (
        Add0,
        [((1, 96, 60, 60), torch.float32), ((96, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer", "onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (Add5, [((1, 96, 60, 60), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (Add6, [((1, 96, 60, 60), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (
        Add0,
        [((1, 96, 30, 30), torch.float32), ((96, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99},
    ),
    (Add5, [((1, 96, 30, 30), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (
        Add0,
        [((1, 192, 30, 30), torch.float32), ((192, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99},
    ),
    (Add5, [((1, 192, 30, 30), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (Add6, [((1, 192, 30, 30), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (
        Add0,
        [((1, 192, 15, 15), torch.float32), ((192, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99},
    ),
    (Add5, [((1, 192, 15, 15), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (
        Add0,
        [((1, 48, 1, 1), torch.float32), ((48, 1, 1), torch.float32)],
        {
            "model_names": [
                "TranslatedLayer",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 192, 1, 1), torch.float32), ((192, 1, 1), torch.float32)],
        {
            "model_names": [
                "TranslatedLayer",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add6,
        [((1, 192, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer", "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 384, 15, 15), torch.float32), ((384, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99},
    ),
    (Add5, [((1, 384, 15, 15), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (Add6, [((1, 384, 15, 15), torch.float32)], {"model_names": ["TranslatedLayer"], "pcc": 0.99}),
    (
        Add0,
        [((1, 96, 1, 1), torch.float32), ((96, 1, 1), torch.float32)],
        {
            "model_names": [
                "TranslatedLayer",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 384, 1, 1), torch.float32), ((384, 1, 1), torch.float32)],
        {
            "model_names": [
                "TranslatedLayer",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add6,
        [((1, 384, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer", "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 360, 15, 15), torch.float32), ((360, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 1, 1), torch.float32), ((24, 1, 1), torch.float32)],
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
        },
    ),
    (
        Add6,
        [((1, 96, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer", "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 15, 15), torch.float32), ((1, 96, 15, 15), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 6, 1, 1), torch.float32), ((6, 1, 1), torch.float32)],
        {
            "model_names": [
                "TranslatedLayer",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add6,
        [((1, 24, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer", "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 15, 15), torch.float32), ((1, 24, 15, 15), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 42, 30, 30), torch.float32), ((42, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 30, 30), torch.float32), ((1, 96, 30, 30), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 30, 30), torch.float32), ((1, 24, 30, 30), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 18, 60, 60), torch.float32), ((18, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 60, 60), torch.float32), ((1, 96, 60, 60), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 60, 60), torch.float32), ((1, 24, 60, 60), torch.float32)],
        {"model_names": ["TranslatedLayer", "onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 12, 120, 120), torch.float32), ((12, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 120, 120), torch.float32), ((1, 96, 120, 120), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 120, 120), torch.float32), ((1, 24, 120, 120), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99},
    ),
    (
        Add0,
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
        Add7,
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
        Add0,
        [((1, 24, 120, 120), torch.float32), ((24, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 240, 240), torch.float32), ((24, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1, 480, 480), torch.float32), ((1, 1, 1), torch.float32)],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99},
    ),
    (
        Add8,
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
        Add9,
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
        Add0,
        [((1, 100, 256), torch.float32), ((1, 100, 256), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add10,
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
        Add10,
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
        Add11,
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
        Add0,
        [((1, 256, 107, 160), torch.float32), ((1, 256, 107, 160), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add12,
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
        Add12,
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
        Add13,
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
        Add0,
        [((1, 512, 54, 80), torch.float32), ((1, 512, 54, 80), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add11,
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
        Add11,
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
        Add14,
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
        Add0,
        [((1, 1024, 27, 40), torch.float32), ((1, 1024, 27, 40), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add13,
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
        Add13,
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
        Add15,
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
        Add0,
        [((1, 2048, 14, 20), torch.float32), ((1, 2048, 14, 20), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 14, 20), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add16,
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
        Add9,
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
        Add17,
        [((1, 8, 280, 280), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 280, 256), torch.float32), ((1, 280, 256), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add18,
        [((1, 280, 2048), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add19,
        [((1, 8, 100, 280), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add18,
        [((1, 100, 2048), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add20,
        [((1, 100, 92), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_obj_det_hf"], "pcc": 0.99},
    ),
    (
        Add21,
        [((1, 100, 4), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 16, 224, 224), torch.float32), ((16, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla46x_c_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "onnx_dla_dla34_visual_bb_torchvision",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 112, 112), torch.float32), ((32, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla46x_c_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_dla_dla34_visual_bb_torchvision",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 112, 112), torch.float32), ((256, 1, 1), torch.float32)],
        {"model_names": ["onnx_dla_dla102x2_visual_bb_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 56, 56), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 56, 56), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_vovnet_vovnet27s_obj_det_osmr",
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 56, 56), torch.float32), ((1, 128, 56, 56), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 512, 56, 56), torch.float32), ((512, 1, 1), torch.float32)],
        {"model_names": ["onnx_dla_dla102x2_visual_bb_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 512, 28, 28), torch.float32), ((512, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 28, 28), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_vovnet_vovnet27s_obj_det_osmr",
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 28, 28), torch.float32), ((1, 256, 28, 28), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1024, 28, 28), torch.float32), ((1024, 1, 1), torch.float32)],
        {"model_names": ["onnx_dla_dla102x2_visual_bb_torchvision"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1024, 14, 14), torch.float32), ((1024, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 512, 14, 14), torch.float32), ((512, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 512, 14, 14), torch.float32), ((1, 512, 14, 14), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 2048, 14, 14), torch.float32), ((2048, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 2048, 7, 7), torch.float32), ((2048, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1024, 7, 7), torch.float32), ((1024, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1024, 7, 7), torch.float32), ((1, 1024, 7, 7), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1000, 1, 1), torch.float32), ((1000, 1, 1), torch.float32)],
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
        },
    ),
    (
        Add0,
        [((1, 64, 112, 112), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla46x_c_visual_bb_torchvision",
                "onnx_vovnet_vovnet27s_obj_det_osmr",
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_dla_dla169_visual_bb_torchvision",
                "pd_resnet_18_img_cls_paddlemodels",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 56, 56), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla46x_c_visual_bb_torchvision",
                "onnx_vovnet_vovnet27s_obj_det_osmr",
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_dla_dla60_visual_bb_torchvision",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_dla_dla169_visual_bb_torchvision",
                "pd_resnet_18_img_cls_paddlemodels",
                "onnx_dla_dla34_visual_bb_torchvision",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 56, 56), torch.float32), ((1, 64, 56, 56), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla46x_c_visual_bb_torchvision",
                "pd_resnet_18_img_cls_paddlemodels",
                "onnx_dla_dla34_visual_bb_torchvision",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 28, 28), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_dla_dla46x_c_visual_bb_torchvision", "onnx_dla_dla60x_c_visual_bb_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 28, 28), torch.float32), ((1, 64, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_dla_dla46x_c_visual_bb_torchvision", "onnx_dla_dla60x_c_visual_bb_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 28, 28), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla46x_c_visual_bb_torchvision",
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_dla_dla60_visual_bb_torchvision",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_dla_dla169_visual_bb_torchvision",
                "pd_resnet_18_img_cls_paddlemodels",
                "onnx_dla_dla34_visual_bb_torchvision",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 14, 14), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla46x_c_visual_bb_torchvision",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 14, 14), torch.float32), ((1, 128, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_dla_dla46x_c_visual_bb_torchvision", "onnx_dla_dla60x_c_visual_bb_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 14, 14), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla46x_c_visual_bb_torchvision",
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_dla_dla60_visual_bb_torchvision",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_dla_dla169_visual_bb_torchvision",
                "pd_resnet_18_img_cls_paddlemodels",
                "onnx_dla_dla34_visual_bb_torchvision",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 7, 7), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_dla_dla46x_c_visual_bb_torchvision", "onnx_dla_dla60x_c_visual_bb_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 7, 7), torch.float32), ((1, 256, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_dla_dla46x_c_visual_bb_torchvision", "onnx_dla_dla60x_c_visual_bb_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 40, 144, 144), torch.float32), ((40, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 10, 1, 1), torch.float32), ((10, 1, 1), torch.float32)],
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
        Add0,
        [((1, 40, 1, 1), torch.float32), ((40, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 24, 144, 144), torch.float32), ((24, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
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
        Add0,
        [((1, 144, 144, 144), torch.float32), ((144, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 144, 72, 72), torch.float32), ((144, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 144, 1, 1), torch.float32), ((144, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 72, 72), torch.float32), ((32, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 192, 72, 72), torch.float32), ((192, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 8, 1, 1), torch.float32), ((8, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 72, 72), torch.float32), ((1, 32, 72, 72), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 192, 36, 36), torch.float32), ((192, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 48, 36, 36), torch.float32), ((48, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 288, 36, 36), torch.float32), ((288, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 12, 1, 1), torch.float32), ((12, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 288, 1, 1), torch.float32), ((288, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 48, 36, 36), torch.float32), ((1, 48, 36, 36), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 288, 18, 18), torch.float32), ((288, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 96, 18, 18), torch.float32), ((96, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 576, 18, 18), torch.float32), ((576, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 576, 1, 1), torch.float32), ((576, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 96, 18, 18), torch.float32), ((1, 96, 18, 18), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 136, 18, 18), torch.float32), ((136, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 816, 18, 18), torch.float32), ((816, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 34, 1, 1), torch.float32), ((34, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 816, 1, 1), torch.float32), ((816, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 136, 18, 18), torch.float32), ((1, 136, 18, 18), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 816, 9, 9), torch.float32), ((816, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 232, 9, 9), torch.float32), ((232, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1392, 9, 9), torch.float32), ((1392, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 58, 1, 1), torch.float32), ((58, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1392, 1, 1), torch.float32), ((1392, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 232, 9, 9), torch.float32), ((1, 232, 9, 9), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 384, 9, 9), torch.float32), ((384, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 2304, 9, 9), torch.float32), ((2304, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 2304, 1, 1), torch.float32), ((2304, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 384, 9, 9), torch.float32), ((1, 384, 9, 9), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1536, 9, 9), torch.float32), ((1536, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add22,
        [((1, 1000), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_vovnet_vovnet27s_obj_det_osmr",
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
                "pd_resnet_18_img_cls_paddlemodels",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pd_alexnet_base_img_cls_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 128, 128), torch.float32), ((64, 1, 1), torch.float32)],
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
        },
    ),
    (
        Add23,
        [((1, 16384, 64), torch.float32)],
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
        },
    ),
    (
        Add0,
        [((1, 64, 16, 16), torch.float32), ((64, 1, 1), torch.float32)],
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
        Add23,
        [((1, 256, 64), torch.float32)],
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
        Add0,
        [((1, 16384, 64), torch.float32), ((1, 16384, 64), torch.float32)],
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
        },
    ),
    (
        Add9,
        [((1, 16384, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 128, 128), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 64, 64), torch.float32), ((128, 1, 1), torch.float32)],
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
        },
    ),
    (
        Add24,
        [((1, 4096, 128), torch.float32)],
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
        },
    ),
    (
        Add0,
        [((1, 128, 16, 16), torch.float32), ((128, 1, 1), torch.float32)],
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
        },
    ),
    (
        Add24,
        [((1, 256, 128), torch.float32)],
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
        },
    ),
    (
        Add0,
        [((1, 4096, 128), torch.float32), ((1, 4096, 128), torch.float32)],
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
        },
    ),
    (
        Add25,
        [((1, 4096, 512), torch.float32)],
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
        },
    ),
    (
        Add0,
        [((1, 512, 64, 64), torch.float32), ((512, 1, 1), torch.float32)],
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
        },
    ),
    (
        Add0,
        [((1, 320, 32, 32), torch.float32), ((320, 1, 1), torch.float32)],
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
        },
    ),
    (
        Add26,
        [((1, 1024, 320), torch.float32)],
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
        },
    ),
    (
        Add0,
        [((1, 320, 16, 16), torch.float32), ((320, 1, 1), torch.float32)],
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
        },
    ),
    (
        Add26,
        [((1, 256, 320), torch.float32)],
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
        },
    ),
    (
        Add0,
        [((1, 1024, 320), torch.float32), ((1, 1024, 320), torch.float32)],
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
        },
    ),
    (
        Add27,
        [((1, 1024, 1280), torch.float32)],
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
        },
    ),
    (
        Add0,
        [((1, 1280, 32, 32), torch.float32), ((1280, 1, 1), torch.float32)],
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
        },
    ),
    (
        Add0,
        [((1, 512, 16, 16), torch.float32), ((512, 1, 1), torch.float32)],
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
        },
    ),
    (
        Add25,
        [((1, 256, 512), torch.float32)],
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
        },
    ),
    (
        Add0,
        [((1, 256, 512), torch.float32), ((1, 256, 512), torch.float32)],
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
        },
    ),
    (
        Add18,
        [((1, 256, 2048), torch.float32)],
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
        },
    ),
    (
        Add0,
        [((1, 2048, 16, 16), torch.float32), ((2048, 1, 1), torch.float32)],
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
        },
    ),
    (
        Add1,
        [((1, 256, 768), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add1,
        [((1, 1024, 768), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add1,
        [((1, 4096, 768), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add1,
        [((1, 16384, 768), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 768, 128, 128), torch.float32), ((768, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 150, 128, 128), torch.float32), ((150, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 80, 28, 28), torch.float32), ((80, 1, 1), torch.float32)],
        {"model_names": ["onnx_vovnet_vovnet27s_obj_det_osmr"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 14, 14), torch.float32), ((96, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_vovnet_vovnet27s_obj_det_osmr",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 384, 14, 14), torch.float32), ((384, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_vovnet_vovnet27s_obj_det_osmr",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 112, 7, 7), torch.float32), ((112, 1, 1), torch.float32)],
        {"model_names": ["onnx_vovnet_vovnet27s_obj_det_osmr"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 512, 7, 7), torch.float32), ((512, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_vovnet_vovnet27s_obj_det_osmr",
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_dla_dla60_visual_bb_torchvision",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_dla_dla169_visual_bb_torchvision",
                "pd_resnet_18_img_cls_paddlemodels",
                "onnx_dla_dla34_visual_bb_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
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
        Add6,
        [((1, 9, 1), torch.float32)],
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
        Add3,
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
        Add6,
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
        Add0,
        [((1, 12, 9, 9), torch.float32), ((1, 1, 1, 9), torch.float32)],
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
        Add28,
        [((1, 9, 3072), torch.float32)],
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
    (Add29, [((1, 9, 30522), torch.float32)], {"model_names": ["pd_bert_bert_base_uncased_mlm_padlenlp"], "pcc": 0.99}),
    (
        Add0,
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
        Add6,
        [((1, 11, 1), torch.float32)],
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
        Add3,
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
        Add6,
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
        Add0,
        [((1, 12, 11, 11), torch.float32), ((1, 1, 1, 11), torch.float32)],
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
        Add28,
        [((1, 11, 3072), torch.float32)],
        {
            "model_names": [
                "pd_bert_chinese_roberta_base_qa_padlenlp",
                "pd_roberta_rbt4_ch_clm_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (Add30, [((1, 11, 2), torch.float32)], {"model_names": ["pd_bert_chinese_roberta_base_qa_padlenlp"], "pcc": 0.99}),
    (Add31, [((1, 9, 18000), torch.float32)], {"model_names": ["pd_ernie_1_0_mlm_padlenlp"], "pcc": 0.99}),
    (
        Add0,
        [((1, 16, 16, 50), torch.float32), ((16, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add5,
        [((1, 16, 16, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add6,
        [((1, 16, 16, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 16, 50), torch.float32), ((32, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add5,
        [((1, 32, 16, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add6,
        [((1, 32, 16, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 16, 50), torch.float32), ((64, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add5,
        [((1, 64, 16, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add6,
        [((1, 64, 16, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 8, 50), torch.float32), ((64, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add5,
        [((1, 64, 8, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add6,
        [((1, 64, 8, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 8, 50), torch.float32), ((128, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add5,
        [((1, 128, 8, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add6,
        [((1, 128, 8, 50), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 8, 25), torch.float32), ((128, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add5,
        [((1, 128, 8, 25), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add6,
        [((1, 128, 8, 25), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 240, 8, 25), torch.float32), ((240, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add5,
        [((1, 240, 8, 25), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add6,
        [((1, 240, 8, 25), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 240, 4, 25), torch.float32), ((240, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add5,
        [((1, 240, 4, 25), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add6,
        [((1, 240, 4, 25), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 60, 1, 1), torch.float32), ((60, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 240, 1, 1), torch.float32), ((240, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add6,
        [((1, 240, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 480, 4, 25), torch.float32), ((480, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add5,
        [((1, 480, 4, 25), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add6,
        [((1, 480, 4, 25), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 120, 1, 1), torch.float32), ((120, 1, 1), torch.float32)],
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
        Add0,
        [((1, 480, 1, 1), torch.float32), ((480, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add6,
        [((1, 480, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 480, 2, 25), torch.float32), ((480, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add5,
        [((1, 480, 2, 25), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add6,
        [((1, 480, 2, 25), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((60,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add32,
        [((60,), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 60, 1, 12), torch.float32), ((60, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
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
        Add33,
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
        Add0,
        [((1, 120, 1, 12), torch.float32), ((120, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add6,
        [((1, 12, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add33,
        [((1, 12, 120), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add34,
        [((1, 12, 360), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 12, 120), torch.float32), ((1, 12, 120), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add35,
        [((1, 12, 240), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
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
        Add36,
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
        Add0,
        [((1, 480, 1, 12), torch.float32), ((480, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add37,
        [((1, 12, 97), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
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
        Add38,
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
        Add0,
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
        Add39,
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
        Add0,
        [((1, 256, 56, 56), torch.float32), ((1, 256, 56, 56), torch.float32)],
        {"model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
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
        Add40,
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
        Add0,
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
        Add41,
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
        Add0,
        [((1, 512, 28, 28), torch.float32), ((1, 512, 28, 28), torch.float32)],
        {"model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
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
        Add42,
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
        Add0,
        [((1, 1024, 14, 14), torch.float32), ((1, 1024, 14, 14), torch.float32)],
        {"model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((2048,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add43,
        [((2048,), torch.float32)],
        {"model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 2048, 7, 7), torch.float32), ((1, 2048, 7, 7), torch.float32)],
        {"model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 128), torch.float32), ((1, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add3,
        [((1, 128, 768), torch.float32)],
        {
            "model_names": [
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_albert_base_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 12, 128, 128), torch.float32), ((1, 1, 1, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_albert_base_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 768), torch.float32), ((1, 128, 768), torch.float32)],
        {
            "model_names": [
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_albert_base_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "onnx_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add28,
        [((1, 128, 3072), torch.float32)],
        {
            "model_names": [
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_albert_base_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add30,
        [((1, 128, 2), torch.float32)],
        {
            "model_names": [
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_large_v2_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add40,
        [((1, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xlarge_v1_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add44,
        [((1, 128, 30000), torch.float32)],
        {
            "model_names": [
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xlarge_v1_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add43,
        [((1, 128, 2048), torch.float32)],
        {"model_names": ["pt_albert_xlarge_v2_token_cls_hf", "pt_albert_xlarge_v1_mlm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 128, 128), torch.float32), ((1, 1, 1, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 2048), torch.float32), ((1, 128, 2048), torch.float32)],
        {"model_names": ["pt_albert_xlarge_v2_token_cls_hf", "pt_albert_xlarge_v1_mlm_hf"], "pcc": 0.99},
    ),
    (
        Add45,
        [((1, 128, 8192), torch.float32)],
        {"model_names": ["pt_albert_xlarge_v2_token_cls_hf", "pt_albert_xlarge_v1_mlm_hf"], "pcc": 0.99},
    ),
    (
        Add46,
        [((1, 128, 4096), torch.float32)],
        {
            "model_names": [
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 128, 128), torch.float32), ((1, 1, 1, 128), torch.float32)],
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
        Add0,
        [((1, 128, 4096), torch.float32), ((1, 128, 4096), torch.float32)],
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
        Add47,
        [((1, 128, 16384), torch.float32)],
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
        Add0,
        [((1, 256, 1024), torch.float32), ((1, 256, 1024), torch.float32)],
        {
            "model_names": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add42,
        [((1, 256, 1024), torch.float32)],
        {
            "model_names": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add48,
        [((1, 16, 256, 256), torch.float32)],
        {"model_names": ["pt_bart_facebook_bart_large_mnli_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1, 256, 256), torch.float32), ((1, 1, 256, 256), torch.float32)],
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
        Add0,
        [((1, 16, 256, 256), torch.float32), ((1, 1, 256, 256), torch.float32)],
        {
            "model_names": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add46,
        [((1, 256, 4096), torch.float32)],
        {
            "model_names": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 1024), torch.float32), ((1, 128, 1024), torch.float32)],
        {
            "model_names": [
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add42,
        [((1, 128, 1024), torch.float32)],
        {
            "model_names": [
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add49,
        [((1, 16, 128, 128), torch.float32)],
        {"model_names": ["pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Add50,
        [((1, 128, 9), torch.float32)],
        {
            "model_names": [
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add3,
        [((1, 6, 768), torch.float32)],
        {
            "model_names": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Add28,
        [((1, 6, 3072), torch.float32)],
        {
            "model_names": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 16, 32), torch.float32), ((1, 256, 16, 32), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add51,
        [((1, 1, 1, 256), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_phi2_microsoft_phi_2_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add52,
        [((1, 256, 51200), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_phi2_microsoft_phi_2_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
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
        Add29,
        [((1, 128, 30522), torch.float32)],
        {"model_names": ["pt_distilbert_distilbert_base_uncased_mlm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((128, 1), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader"], "pcc": 0.99},
    ),
    (
        Add5,
        [((1, 1), torch.float32)],
        {"model_names": ["pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 2048), torch.float32), ((1, 32, 2048), torch.float32)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 32, 32), torch.float32), ((1, 1, 32, 32), torch.float32)],
        {
            "model_names": [
                "pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf",
                "pt_bloom_bigscience_bloom_1b1_clm_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add53,
        [((1, 1, 1, 32), torch.float32)],
        {
            "model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf", "pt_bloom_bigscience_bloom_1b1_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1, 32, 32), torch.float32), ((1, 1, 32, 32), torch.float32)],
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
        Add43,
        [((1, 32, 2048), torch.float32)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add45,
        [((1, 32, 8192), torch.float32)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf"], "pcc": 0.99},
    ),
    (Add6, [((1, 256, 1), torch.float32)], {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 32, 256, 64), torch.float32), ((1, 32, 256, 64), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 8, 256, 64), torch.float32), ((1, 8, 256, 64), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 256, 256), torch.float32), ((1, 1, 256, 256), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_phi2_microsoft_phi_2_clm_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 2048), torch.float32), ((1, 256, 2048), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add54,
        [((1, 32), torch.int64)],
        {"model_names": ["pt_opt_facebook_opt_125m_qa_hf", "pt_opt_facebook_opt_350m_qa_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 768), torch.float32), ((1, 32, 768), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_qa_hf"], "pcc": 0.99},
    ),
    (Add3, [((1, 32, 768), torch.float32)], {"model_names": ["pt_opt_facebook_opt_125m_qa_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 12, 32, 32), torch.float32), ((1, 1, 32, 32), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_qa_hf"], "pcc": 0.99},
    ),
    (Add28, [((32, 3072), torch.float32)], {"model_names": ["pt_opt_facebook_opt_125m_qa_hf"], "pcc": 0.99}),
    (Add3, [((32, 768), torch.float32)], {"model_names": ["pt_opt_facebook_opt_125m_qa_hf"], "pcc": 0.99}),
    (
        Add0,
        [((32, 768), torch.float32), ((32, 768), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_qa_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((32, 1), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_qa_hf", "pt_opt_facebook_opt_350m_qa_hf"], "pcc": 0.99},
    ),
    (Add55, [((1, 256, 2560), torch.float32)], {"model_names": ["pt_phi2_microsoft_phi_2_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 32, 256, 32), torch.float32), ((1, 32, 256, 32), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_clm_hf"], "pcc": 0.99},
    ),
    (Add56, [((1, 256, 10240), torch.float32)], {"model_names": ["pt_phi2_microsoft_phi_2_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 256, 2560), torch.float32), ((1, 256, 2560), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_clm_hf"], "pcc": 0.99},
    ),
    (Add3, [((1, 204, 768), torch.float32)], {"model_names": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 12, 204, 204), torch.float32), ((1, 1, 1, 204), torch.float32)],
        {"model_names": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 204, 768), torch.float32), ((1, 204, 768), torch.float32)],
        {"model_names": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"], "pcc": 0.99},
    ),
    (Add28, [((1, 204, 3072), torch.float32)], {"model_names": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 1, 768), torch.float32), ((1, 1, 768), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add3,
        [((1, 1, 768), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 768, 3000), torch.float32), ((768, 1), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 768, 1500), torch.float32), ((768, 1), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add57,
        [((1, 1500, 768), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add3,
        [((1, 1500, 768), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1500, 768), torch.float32), ((1, 1500, 768), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add28,
        [((1, 1500, 3072), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add28,
        [((1, 1, 3072), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 384, 1024), torch.float32), ((1, 384, 1024), torch.float32)],
        {
            "model_names": [
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add58,
        [((1, 384, 1024), torch.float32)],
        {"model_names": ["onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf"], "pcc": 0.99},
    ),
    (
        Add59,
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
        Add60,
        [((1, 384, 4096), torch.float32)],
        {"model_names": ["onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((384, 1), torch.float32), ((1,), torch.float32)],
        {
            "model_names": [
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 112, 112), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_dla_dla102x_visual_bb_torchvision", "onnx_dla_dla60x_visual_bb_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 1, 1), torch.float32), ((32, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 16, 112, 112), torch.float32), ((16, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 96, 112, 112), torch.float32), ((96, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 96, 56, 56), torch.float32), ((96, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 4, 1, 1), torch.float32), ((4, 1, 1), torch.float32)],
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
        Add0,
        [((1, 24, 56, 56), torch.float32), ((24, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 144, 56, 56), torch.float32), ((144, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 24, 56, 56), torch.float32), ((1, 24, 56, 56), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 144, 28, 28), torch.float32), ((144, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 40, 28, 28), torch.float32), ((40, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 240, 28, 28), torch.float32), ((240, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 40, 28, 28), torch.float32), ((1, 40, 28, 28), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 240, 14, 14), torch.float32), ((240, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 80, 14, 14), torch.float32), ((80, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 480, 14, 14), torch.float32), ((480, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm", "pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 20, 1, 1), torch.float32), ((20, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 80, 14, 14), torch.float32), ((1, 80, 14, 14), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 112, 14, 14), torch.float32), ((112, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 672, 14, 14), torch.float32), ((672, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm", "pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 28, 1, 1), torch.float32), ((28, 1, 1), torch.float32)],
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
        Add0,
        [((1, 672, 1, 1), torch.float32), ((672, 1, 1), torch.float32)],
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
        Add0,
        [((1, 112, 14, 14), torch.float32), ((1, 112, 14, 14), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 672, 7, 7), torch.float32), ((672, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm", "pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 192, 7, 7), torch.float32), ((192, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1152, 7, 7), torch.float32), ((1152, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1152, 1, 1), torch.float32), ((1152, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 192, 7, 7), torch.float32), ((1, 192, 7, 7), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 320, 7, 7), torch.float32), ((320, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1280, 7, 7), torch.float32), ((1280, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 13, 384), torch.float32), ((1, 13, 384), torch.float32)],
        {"model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add61,
        [((1, 13, 384), torch.float32)],
        {"model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add62,
        [((1, 12, 13, 13), torch.float32)],
        {"model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add63,
        [((1, 13, 1536), torch.float32)],
        {"model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add64,
        [((1, 384), torch.float32)],
        {"model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 8, 112, 112), torch.float32), ((8, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 112, 112), torch.float32), ((48, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 48, 56, 56), torch.float32), ((48, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 16, 56, 56), torch.float32), ((16, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 56, 56), torch.float32), ((1, 16, 56, 56), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 28, 28), torch.float32), ((96, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 16, 28, 28), torch.float32), ((16, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 28, 28), torch.float32), ((1, 16, 28, 28), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 14, 14), torch.float32), ((32, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 192, 14, 14), torch.float32), ((192, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 14, 14), torch.float32), ((1, 32, 14, 14), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 14, 14), torch.float32), ((48, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 288, 14, 14), torch.float32), ((288, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm", "pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 48, 14, 14), torch.float32), ((1, 48, 14, 14), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 288, 7, 7), torch.float32), ((288, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 80, 7, 7), torch.float32), ((80, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 480, 7, 7), torch.float32), ((480, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 80, 7, 7), torch.float32), ((1, 80, 7, 7), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_050_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 160, 7, 7), torch.float32), ((160, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 160, 28, 28), torch.float32), ((160, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet_v1_57_obj_det_torchhub", "pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 768, 14, 14), torch.float32), ((768, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 768, 14, 14), torch.float32), ((1, 768, 14, 14), torch.float32)],
        {"model_names": ["onnx_vovnet_vovnet_v1_57_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 224, 7, 7), torch.float32), ((224, 1, 1), torch.float32)],
        {"model_names": ["onnx_vovnet_vovnet_v1_57_obj_det_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 10, 768), torch.float32), ((1, 10, 768), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99},
    ),
    (Add6, [((1, 10, 1), torch.float32)], {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99}),
    (Add3, [((1, 10, 768), torch.float32)], {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99}),
    (Add6, [((1, 1, 1, 10), torch.float32)], {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99}),
    (
        Add0,
        [((1, 12, 10, 10), torch.float32), ((1, 1, 1, 10), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Add28,
        [((1, 10, 3072), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Add65,
        [((1, 10, 32000), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Add0,
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
        Add6,
        [((1, 8, 1), torch.float32)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add3,
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
        Add6,
        [((1, 1, 1, 8), torch.float32)],
        {"model_names": ["pd_bert_bert_base_uncased_seq_cls_padlenlp"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 12, 8, 8), torch.float32), ((1, 1, 1, 8), torch.float32)],
        {"model_names": ["pd_bert_bert_base_uncased_seq_cls_padlenlp"], "pcc": 0.99},
    ),
    (
        Add28,
        [((1, 8, 3072), torch.float32)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add30,
        [((1, 2), torch.float32)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
                "pd_bert_bert_base_japanese_seq_cls_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add30,
        [((1, 9, 2), torch.float32)],
        {"model_names": ["pd_ernie_1_0_qa_padlenlp", "pd_bert_bert_base_uncased_qa_padlenlp"], "pcc": 0.99},
    ),
    (
        Add5,
        [((1, 16, 224, 224), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add6,
        [((1, 16, 224, 224), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 224, 224), torch.float32), ((32, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add5,
        [((1, 32, 224, 224), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add6,
        [((1, 32, 224, 224), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add5,
        [((1, 32, 112, 112), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add5,
        [((1, 48, 112, 112), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add6,
        [((1, 48, 112, 112), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add5,
        [((1, 48, 56, 56), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add5,
        [((1, 96, 56, 56), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add6,
        [((1, 96, 56, 56), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add5,
        [((1, 96, 28, 28), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 192, 28, 28), torch.float32), ((192, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add5,
        [((1, 192, 28, 28), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add6,
        [((1, 192, 28, 28), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add5,
        [((1, 192, 14, 14), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add5,
        [((1, 384, 14, 14), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add6,
        [((1, 384, 14, 14), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 360, 14, 14), torch.float32), ((360, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 14, 14), torch.float32), ((1, 96, 14, 14), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 24, 14, 14), torch.float32), ((1, 24, 14, 14), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 42, 28, 28), torch.float32), ((42, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 28, 28), torch.float32), ((1, 96, 28, 28), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 28, 28), torch.float32), ((1, 24, 28, 28), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 18, 56, 56), torch.float32), ((18, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 56, 56), torch.float32), ((1, 96, 56, 56), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 12, 112, 112), torch.float32), ((12, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 112, 112), torch.float32), ((1, 96, 112, 112), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 112, 112), torch.float32), ((1, 24, 112, 112), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 112, 112), torch.float32), ((24, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 224, 224), torch.float32), ((24, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1, 448, 448), torch.float32), ((1, 1, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add66,
        [((1, 11, 21128), torch.float32)],
        {"model_names": ["pd_roberta_rbt4_ch_clm_padlenlp", "pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99},
    ),
    (Add67, [((1, 32, 4608), torch.float32)], {"model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((16, 32, 32), torch.float32), ((16, 1, 32), torch.float32)],
        {"model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99},
    ),
    (Add68, [((1, 32, 1536), torch.float32)], {"model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 32, 1536), torch.float32), ((1, 32, 1536), torch.float32)],
        {"model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99},
    ),
    (Add69, [((1, 32, 6144), torch.float32)], {"model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99}),
    (Add6, [((1, 32, 6144), torch.float32)], {"model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((2, 7, 512), torch.float32), ((1, 7, 512), torch.float32)],
        {"model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99},
    ),
    (
        Add41,
        [((2, 7, 512), torch.float32)],
        {"model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99},
    ),
    (
        Add70,
        [((2, 8, 7, 7), torch.float32)],
        {"model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99},
    ),
    (
        Add0,
        [((2, 1, 7, 7), torch.float32), ((2, 1, 7, 7), torch.float32)],
        {"model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99},
    ),
    (
        Add0,
        [((2, 8, 7, 7), torch.float32), ((2, 1, 7, 7), torch.float32)],
        {"model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99},
    ),
    (
        Add0,
        [((2, 7, 512), torch.float32), ((2, 7, 512), torch.float32)],
        {"model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99},
    ),
    (
        Add43,
        [((2, 7, 2048), torch.float32)],
        {"model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 384, 768), torch.float32), ((1, 384, 768), torch.float32)],
        {"model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"], "pcc": 0.99},
    ),
    (
        Add3,
        [((1, 384, 768), torch.float32)],
        {"model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 12, 384, 384), torch.float32), ((1, 12, 384, 384), torch.float32)],
        {"model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"], "pcc": 0.99},
    ),
    (
        Add28,
        [((1, 384, 3072), torch.float32)],
        {"model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 7, 768), torch.float32), ((1, 7, 768), torch.float32)],
        {
            "model_names": [
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((7, 768), torch.float32), ((768,), torch.float32)],
        {
            "model_names": [
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 12, 7, 7), torch.float32), ((1, 1, 7, 7), torch.float32)],
        {
            "model_names": [
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 12, 7, 7), torch.float32), ((1, 1, 1, 7), torch.float32)],
        {
            "model_names": [
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add3,
        [((7, 768), torch.float32)],
        {
            "model_names": [
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add28,
        [((7, 3072), torch.float32)],
        {
            "model_names": [
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add6,
        [((1, 4, 1), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 4, 64), torch.float32), ((1, 32, 4, 64), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 8, 4, 64), torch.float32), ((1, 8, 4, 64), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add71,
        [((1, 32, 4, 4), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 4, 2048), torch.float32), ((1, 4, 2048), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (Add54, [((1, 256), torch.int64)], {"model_names": ["pt_opt_facebook_opt_1_3b_clm_hf"], "pcc": 0.99}),
    (
        Add43,
        [((1, 256, 2048), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_1_3b_clm_hf", "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf"], "pcc": 0.99},
    ),
    (Add45, [((256, 8192), torch.float32)], {"model_names": ["pt_opt_facebook_opt_1_3b_clm_hf"], "pcc": 0.99}),
    (Add43, [((256, 2048), torch.float32)], {"model_names": ["pt_opt_facebook_opt_1_3b_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((256, 2048), torch.float32), ((256, 2048), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_1_3b_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 1024), torch.float32), ((1, 32, 1024), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_350m_qa_hf"], "pcc": 0.99},
    ),
    (Add42, [((1, 32, 1024), torch.float32)], {"model_names": ["pt_opt_facebook_opt_350m_qa_hf"], "pcc": 0.99}),
    (Add46, [((32, 4096), torch.float32)], {"model_names": ["pt_opt_facebook_opt_350m_qa_hf"], "pcc": 0.99}),
    (Add42, [((32, 1024), torch.float32)], {"model_names": ["pt_opt_facebook_opt_350m_qa_hf"], "pcc": 0.99}),
    (
        Add0,
        [((32, 1024), torch.float32), ((32, 1024), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_350m_qa_hf"], "pcc": 0.99},
    ),
    (Add43, [((1, 7, 2048), torch.float32)], {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 32, 7, 32), torch.float32), ((1, 32, 7, 32), torch.float32)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_clm_hf"], "pcc": 0.99},
    ),
    (Add72, [((1, 1, 1, 7), torch.float32)], {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 1, 7, 7), torch.float32), ((1, 1, 7, 7), torch.float32)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 7, 7), torch.float32), ((1, 1, 7, 7), torch.float32)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_clm_hf"], "pcc": 0.99},
    ),
    (Add45, [((1, 7, 8192), torch.float32)], {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 7, 2048), torch.float32), ((1, 7, 2048), torch.float32)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_clm_hf"], "pcc": 0.99},
    ),
    (Add52, [((1, 7, 51200), torch.float32)], {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_clm_hf"], "pcc": 0.99}),
    (
        Add3,
        [((1, 201, 768), torch.float32)],
        {"model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 12, 201, 201), torch.float32), ((1, 1, 1, 201), torch.float32)],
        {"model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 201, 768), torch.float32), ((1, 201, 768), torch.float32)],
        {"model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"], "pcc": 0.99},
    ),
    (
        Add28,
        [((1, 201, 3072), torch.float32)],
        {"model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"], "pcc": 0.99},
    ),
    (
        Add68,
        [((1, 1536), torch.float32)],
        {"model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"], "pcc": 0.99},
    ),
    (
        Add73,
        [((1, 3129), torch.float32)],
        {"model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1, 384), torch.float32), ((1, 1, 384), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add64,
        [((1, 1, 384), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 384, 3000), torch.float32), ((384, 1), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 384, 1500), torch.float32), ((384, 1), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add74,
        [((1, 1500, 384), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add64,
        [((1, 1500, 384), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1500, 384), torch.float32), ((1, 1500, 384), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add68,
        [((1, 1500, 1536), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add68,
        [((1, 1, 1536), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (Add6, [((1, 1, 1, 64), torch.float32)], {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 112, 112, 64), torch.float32), ((1, 1, 1, 64), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 56, 55, 64), torch.float32), ((1, 1, 1, 64), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (Add6, [((1, 1, 1, 256), torch.float32)], {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 56, 55, 256), torch.float32), ((1, 1, 1, 256), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 56, 55, 256), torch.float32), ((1, 56, 55, 256), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (Add6, [((1, 1, 1, 128), torch.float32)], {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 56, 55, 128), torch.float32), ((1, 1, 1, 128), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 28, 28, 128), torch.float32), ((1, 1, 1, 128), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (Add6, [((1, 1, 1, 512), torch.float32)], {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 28, 28, 512), torch.float32), ((1, 1, 1, 512), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 28, 28, 512), torch.float32), ((1, 28, 28, 512), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 28, 28, 256), torch.float32), ((1, 1, 1, 256), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 14, 14, 256), torch.float32), ((1, 1, 1, 256), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (Add6, [((1, 1, 1, 1024), torch.float32)], {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 14, 14, 1024), torch.float32), ((1, 1, 1, 1024), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 14, 14, 1024), torch.float32), ((1, 14, 14, 1024), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 14, 14, 512), torch.float32), ((1, 1, 1, 512), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 7, 7, 512), torch.float32), ((1, 1, 1, 512), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (Add6, [((1, 1, 1, 2048), torch.float32)], {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 7, 7, 2048), torch.float32), ((1, 1, 1, 2048), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 7, 7, 2048), torch.float32), ((1, 7, 7, 2048), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1000), torch.float32), ((1, 1000), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add75,
        [((1, 100, 251), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 107, 160), torch.float32), ((32, 1, 1), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 54, 80), torch.float32), ((64, 1, 1), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 27, 40), torch.float32), ((128, 1, 1), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add76,
        [((1, 100, 8, 14, 20), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((100, 264, 14, 20), torch.float32), ((264, 1, 1), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add6,
        [((100, 8, 1), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add77,
        [((100, 8, 9240), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add78,
        [((100, 264, 14, 20), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((100, 128, 14, 20), torch.float32), ((128, 1, 1), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add77,
        [((100, 8, 4480), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add79,
        [((100, 128, 14, 20), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((100, 128, 27, 40), torch.float32), ((100, 128, 27, 40), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((100, 64, 27, 40), torch.float32), ((64, 1, 1), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add77,
        [((100, 8, 8640), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add80,
        [((100, 64, 27, 40), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((100, 64, 54, 80), torch.float32), ((100, 64, 54, 80), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((100, 32, 54, 80), torch.float32), ((32, 1, 1), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add77,
        [((100, 8, 17280), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add81,
        [((100, 32, 54, 80), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((100, 32, 107, 160), torch.float32), ((100, 32, 107, 160), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((100, 16, 107, 160), torch.float32), ((16, 1, 1), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add77,
        [((100, 8, 34240), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add82,
        [((100, 16, 107, 160), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((100, 1, 107, 160), torch.float32), ((1, 1, 1), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 120, 120), torch.float32), ((16, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 1, 1), torch.float32), ((16, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 16, 120, 120), torch.float32), ((1, 16, 120, 120), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 120, 120), torch.float32), ((96, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 60, 60), torch.float32), ((24, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 144, 60, 60), torch.float32), ((144, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 144, 30, 30), torch.float32), ((144, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 40, 30, 30), torch.float32), ((40, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 240, 30, 30), torch.float32), ((240, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 40, 30, 30), torch.float32), ((1, 40, 30, 30), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 240, 15, 15), torch.float32), ((240, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 80, 15, 15), torch.float32), ((80, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 480, 15, 15), torch.float32), ((480, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 80, 15, 15), torch.float32), ((1, 80, 15, 15), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 112, 15, 15), torch.float32), ((112, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 672, 15, 15), torch.float32), ((672, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 112, 15, 15), torch.float32), ((1, 112, 15, 15), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 672, 8, 8), torch.float32), ((672, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 192, 8, 8), torch.float32), ((192, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1152, 8, 8), torch.float32), ((1152, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 192, 8, 8), torch.float32), ((1, 192, 8, 8), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 320, 8, 8), torch.float32), ((320, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1920, 8, 8), torch.float32), ((1920, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 80, 1, 1), torch.float32), ((80, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1920, 1, 1), torch.float32), ((1920, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 320, 8, 8), torch.float32), ((1, 320, 8, 8), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1280, 8, 8), torch.float32), ((1280, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 160, 160), torch.float32), ((48, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 160, 160), torch.float32), ((24, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 160, 160), torch.float32), ((1, 24, 160, 160), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 144, 160, 160), torch.float32), ((144, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 144, 80, 80), torch.float32), ((144, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 80, 80), torch.float32), ((32, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm", "onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 192, 80, 80), torch.float32), ((192, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 80, 80), torch.float32), ((1, 32, 80, 80), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm", "onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 192, 40, 40), torch.float32), ((192, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 56, 40, 40), torch.float32), ((56, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 336, 40, 40), torch.float32), ((336, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 14, 1, 1), torch.float32), ((14, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 336, 1, 1), torch.float32), ((336, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 56, 40, 40), torch.float32), ((1, 56, 40, 40), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 336, 20, 20), torch.float32), ((336, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 112, 20, 20), torch.float32), ((112, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 672, 20, 20), torch.float32), ((672, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 112, 20, 20), torch.float32), ((1, 112, 20, 20), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 160, 20, 20), torch.float32), ((160, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 960, 20, 20), torch.float32), ((960, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 960, 1, 1), torch.float32), ((960, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 160, 20, 20), torch.float32), ((1, 160, 20, 20), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 960, 10, 10), torch.float32), ((960, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 272, 10, 10), torch.float32), ((272, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1632, 10, 10), torch.float32), ((1632, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 68, 1, 1), torch.float32), ((68, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1632, 1, 1), torch.float32), ((1632, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 272, 10, 10), torch.float32), ((1, 272, 10, 10), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 448, 10, 10), torch.float32), ((448, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 2688, 10, 10), torch.float32), ((2688, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 112, 1, 1), torch.float32), ((112, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 2688, 1, 1), torch.float32), ((2688, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 448, 10, 10), torch.float32), ((1, 448, 10, 10), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1792, 10, 10), torch.float32), ((1792, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 28, 28), torch.float32), ((32, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 28, 28), torch.float32), ((1, 32, 28, 28), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 14, 14), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 14, 14), torch.float32), ((1, 64, 14, 14), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 576, 14, 14), torch.float32), ((576, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 576, 7, 7), torch.float32), ((576, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 960, 7, 7), torch.float32), ((960, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 160, 7, 7), torch.float32), ((1, 160, 7, 7), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 128, 128), torch.float32), ((32, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add83,
        [((1, 16384, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 32, 16, 16), torch.float32), ((32, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add83,
        [((1, 256, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 16384, 32), torch.float32), ((1, 16384, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add24,
        [((1, 16384, 128), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 128, 128, 128), torch.float32), ((128, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 64, 64), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add23,
        [((1, 4096, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 4096, 64), torch.float32), ((1, 4096, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add9,
        [((1, 4096, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 64, 64), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 160, 32, 32), torch.float32), ((160, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add84,
        [((1, 1024, 160), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 160, 16, 16), torch.float32), ((160, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add84,
        [((1, 256, 160), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1024, 160), torch.float32), ((1, 1024, 160), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add85,
        [((1, 1024, 640), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 640, 32, 32), torch.float32), ((640, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 16, 16), torch.float32), ((256, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add9,
        [((1, 256, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 256), torch.float32), ((1, 256, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add58,
        [((1, 256, 1024), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1024, 16, 16), torch.float32), ((1024, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add86,
        [((1, 197, 768), torch.float32)],
        {"model_names": ["onnx_vit_base_google_vit_base_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add1,
        [((1, 197, 768), torch.float32)],
        {"model_names": ["onnx_vit_base_google_vit_base_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 197, 768), torch.float32), ((1, 197, 768), torch.float32)],
        {"model_names": ["onnx_vit_base_google_vit_base_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add2,
        [((1, 197, 3072), torch.float32)],
        {"model_names": ["onnx_vit_base_google_vit_base_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 320, 320), torch.float32), ((16, 1, 1), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 32, 160, 160), torch.float32), ((32, 1, 1), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 160, 160), torch.float32), ((16, 1, 1), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 160, 160), torch.float32), ((1, 16, 160, 160), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 80, 80), torch.float32), ((64, 1, 1), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 40, 40), torch.float32), ((128, 1, 1), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 40, 40), torch.float32), ((64, 1, 1), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 40, 40), torch.float32), ((1, 64, 40, 40), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 20, 20), torch.float32), ((256, 1, 1), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 20, 20), torch.float32), ((128, 1, 1), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 20, 20), torch.float32), ((1, 128, 20, 20), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 80, 80, 80), torch.float32), ((80, 1, 1), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 80, 40, 40), torch.float32), ((80, 1, 1), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 20, 20), torch.float32), ((64, 1, 1), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 80, 20, 20), torch.float32), ((80, 1, 1), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (Add87, [((1, 2, 8400), torch.float32)], {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99}),
    (
        Add0,
        [((1, 2, 8400), torch.float32), ((1, 2, 8400), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 11, 128), torch.float32), ((1, 11, 128), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99},
    ),
    (Add40, [((1, 11, 128), torch.float32)], {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99}),
    (Add88, [((1, 11, 312), torch.float32)], {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99}),
    (
        Add0,
        [((1, 11, 312), torch.float32), ((1, 11, 312), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99},
    ),
    (Add89, [((1, 11, 1248), torch.float32)], {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99}),
    (
        Add0,
        [((1, 15, 768), torch.float32), ((1, 15, 768), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"], "pcc": 0.99},
    ),
    (
        Add6,
        [((1, 15, 1), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"], "pcc": 0.99},
    ),
    (
        Add3,
        [((1, 15, 768), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"], "pcc": 0.99},
    ),
    (
        Add6,
        [((1, 1, 1, 15), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 12, 15, 15), torch.float32), ((1, 1, 1, 15), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"], "pcc": 0.99},
    ),
    (
        Add28,
        [((1, 15, 3072), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"], "pcc": 0.99},
    ),
    (
        Add66,
        [((1, 9, 21128), torch.float32)],
        {"model_names": ["pd_bert_chinese_roberta_base_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Add0,
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
        Add90,
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
        Add0,
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
        Add6,
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
        Add0,
        [((1, 2, 1, 1), torch.float32), ((2, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add6,
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
        Add0,
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
        Add91,
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
        Add0,
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
        Add0,
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
        Add0,
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
        Add0,
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
        Add92,
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
        Add0,
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
        Add0,
        [((1, 16, 8, 50), torch.float32), ((1, 16, 8, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add6,
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
        Add0,
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
        Add6,
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
        Add6,
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
        Add0,
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
        Add0,
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
        Add6,
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
        Add0,
        [((1, 30, 1, 1), torch.float32), ((30, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add6,
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
        Add0,
        [((1, 24, 4, 50), torch.float32), ((1, 24, 4, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
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
        Add6,
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
        Add0,
        [((1, 64, 1, 1), torch.float32), ((64, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add6,
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
        Add0,
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
        Add93,
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
        Add0,
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
        Add6,
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
        Add0,
        [((1, 18, 1, 1), torch.float32), ((18, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 72, 1, 1), torch.float32), ((72, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add6,
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
        Add0,
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
        Add94,
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
        Add0,
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
        Add6,
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
        Add0,
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
        Add6,
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
        Add0,
        [((1, 36, 1, 1), torch.float32), ((36, 1, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add6,
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
        Add0,
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
        Add0,
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
        Add95,
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
        Add0,
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
        Add6,
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
        Add6,
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
        Add0,
        [((1, 48, 2, 50), torch.float32), ((1, 48, 2, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 192), torch.float32), ((1, 192), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add96,
        [((1, 192), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
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
        Add97,
        [((1, 25, 6625), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 28, 28), torch.float32), ((1, 128, 28, 28), torch.float32)],
        {
            "model_names": [
                "pd_resnet_18_img_cls_paddlemodels",
                "onnx_dla_dla34_visual_bb_torchvision",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 256, 14, 14), torch.float32), ((1, 256, 14, 14), torch.float32)],
        {
            "model_names": [
                "pd_resnet_18_img_cls_paddlemodels",
                "onnx_dla_dla34_visual_bb_torchvision",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 512, 7, 7), torch.float32), ((1, 512, 7, 7), torch.float32)],
        {
            "model_names": [
                "pd_resnet_18_img_cls_paddlemodels",
                "onnx_dla_dla34_visual_bb_torchvision",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 2, 1280), torch.float32), ((1, 2, 1280), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf", "onnx_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Add98,
        [((1, 2, 1280), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"], "pcc": 0.99},
    ),
    (
        Add99,
        [((1, 20, 2, 2), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf", "onnx_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1280, 3000), torch.float32), ((1280, 1), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf", "onnx_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 1280, 1500), torch.float32), ((1280, 1), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf", "onnx_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Add100,
        [((1, 1500, 1280), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"], "pcc": 0.99},
    ),
    (
        Add98,
        [((1, 1500, 1280), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1500, 1280), torch.float32), ((1, 1500, 1280), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf", "onnx_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Add101,
        [((1, 1500, 5120), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"], "pcc": 0.99},
    ),
    (
        Add101,
        [((1, 2, 5120), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"], "pcc": 0.99},
    ),
    (Add1, [((1, 128, 768), torch.float32)], {"model_names": ["onnx_bert_bert_base_uncased_mlm_hf"], "pcc": 0.99}),
    (Add49, [((1, 12, 128, 128), torch.float32)], {"model_names": ["onnx_bert_bert_base_uncased_mlm_hf"], "pcc": 0.99}),
    (Add2, [((1, 128, 3072), torch.float32)], {"model_names": ["onnx_bert_bert_base_uncased_mlm_hf"], "pcc": 0.99}),
    (Add102, [((1, 128, 30522), torch.float32)], {"model_names": ["onnx_bert_bert_base_uncased_mlm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 16, 128, 128), torch.float32), ((16, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 128, 128), torch.float32), ((1, 16, 128, 128), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 128, 128), torch.float32), ((96, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 96, 64, 64), torch.float32), ((96, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 24, 64, 64), torch.float32), ((24, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 144, 64, 64), torch.float32), ((144, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 64, 64), torch.float32), ((1, 24, 64, 64), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 144, 32, 32), torch.float32), ((144, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 32, 32), torch.float32), ((48, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 288, 32, 32), torch.float32), ((288, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 32, 32), torch.float32), ((1, 48, 32, 32), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 288, 16, 16), torch.float32), ((288, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 88, 16, 16), torch.float32), ((88, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 528, 16, 16), torch.float32), ((528, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 22, 1, 1), torch.float32), ((22, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 528, 1, 1), torch.float32), ((528, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 88, 16, 16), torch.float32), ((1, 88, 16, 16), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 120, 16, 16), torch.float32), ((120, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 720, 16, 16), torch.float32), ((720, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 720, 1, 1), torch.float32), ((720, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 120, 16, 16), torch.float32), ((1, 120, 16, 16), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 720, 8, 8), torch.float32), ((720, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 208, 8, 8), torch.float32), ((208, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1248, 8, 8), torch.float32), ((1248, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 52, 1, 1), torch.float32), ((52, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1248, 1, 1), torch.float32), ((1248, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 208, 8, 8), torch.float32), ((1, 208, 8, 8), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 352, 8, 8), torch.float32), ((352, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 2112, 8, 8), torch.float32), ((2112, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 88, 1, 1), torch.float32), ((88, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 2112, 1, 1), torch.float32), ((2112, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 352, 8, 8), torch.float32), ((1, 352, 8, 8), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1408, 8, 8), torch.float32), ((1408, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 48, 224, 224), torch.float32), ((48, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 24, 224, 224), torch.float32), ((1, 24, 224, 224), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 144, 224, 224), torch.float32), ((144, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 144, 112, 112), torch.float32), ((144, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 40, 112, 112), torch.float32), ((40, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 240, 112, 112), torch.float32), ((240, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 40, 112, 112), torch.float32), ((1, 40, 112, 112), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 240, 56, 56), torch.float32), ((240, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 384, 56, 56), torch.float32), ((384, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 384, 28, 28), torch.float32), ((384, 1, 1), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm", "pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 768, 28, 28), torch.float32), ((768, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 768, 1, 1), torch.float32), ((768, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 176, 28, 28), torch.float32), ((176, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1056, 28, 28), torch.float32), ((1056, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 44, 1, 1), torch.float32), ((44, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1056, 1, 1), torch.float32), ((1056, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 176, 28, 28), torch.float32), ((1, 176, 28, 28), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1056, 14, 14), torch.float32), ((1056, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 304, 14, 14), torch.float32), ((304, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1824, 14, 14), torch.float32), ((1824, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 76, 1, 1), torch.float32), ((76, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1824, 1, 1), torch.float32), ((1824, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 304, 14, 14), torch.float32), ((1, 304, 14, 14), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 3072, 14, 14), torch.float32), ((3072, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 1, 1), torch.float32), ((128, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 3072, 1, 1), torch.float32), ((3072, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 72, 14, 14), torch.float32), ((72, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 432, 14, 14), torch.float32), ((432, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 72, 14, 14), torch.float32), ((1, 72, 14, 14), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 104, 14, 14), torch.float32), ((104, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 624, 14, 14), torch.float32), ((624, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 104, 14, 14), torch.float32), ((1, 104, 14, 14), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 624, 7, 7), torch.float32), ((624, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 176, 7, 7), torch.float32), ((176, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1056, 7, 7), torch.float32), ((1056, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 176, 7, 7), torch.float32), ((1, 176, 7, 7), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 352, 7, 7), torch.float32), ((352, 1, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Add9,
        [((1, 1024, 256), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Add103,
        [((64, 64, 96), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add25,
        [((1, 15, 15, 512), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((64, 3, 64, 64), torch.float32), ((1, 3, 64, 64), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 4096, 96), torch.float32), ((1, 4096, 96), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add61,
        [((1, 4096, 384), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add103,
        [((1, 4096, 96), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add104,
        [((1, 64, 3, 64, 64), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add105,
        [((16, 64, 192), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((16, 6, 64, 64), torch.float32), ((1, 6, 64, 64), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1024, 192), torch.float32), ((1, 1024, 192), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add105,
        [((1, 1024, 192), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add106,
        [((1, 16, 6, 64, 64), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add61,
        [((4, 64, 384), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((4, 12, 64, 64), torch.float32), ((1, 12, 64, 64), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 384), torch.float32), ((1, 256, 384), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add63,
        [((1, 256, 1536), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add61,
        [((1, 256, 384), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add107,
        [((1, 4, 12, 64, 64), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add1,
        [((1, 64, 768), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 768), torch.float32), ((1, 64, 768), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add2,
        [((1, 64, 3072), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Add27,
        [((1, 2, 1280), torch.float32)],
        {"model_names": ["onnx_whisper_openai_whisper_large_v3_clm_hf"], "pcc": 0.99},
    ),
    (
        Add108,
        [((1, 1500, 1280), torch.float32)],
        {"model_names": ["onnx_whisper_openai_whisper_large_v3_clm_hf"], "pcc": 0.99},
    ),
    (
        Add27,
        [((1, 1500, 1280), torch.float32)],
        {"model_names": ["onnx_whisper_openai_whisper_large_v3_clm_hf"], "pcc": 0.99},
    ),
    (
        Add109,
        [((1, 1500, 5120), torch.float32)],
        {"model_names": ["onnx_whisper_openai_whisper_large_v3_clm_hf"], "pcc": 0.99},
    ),
    (
        Add109,
        [((1, 2, 5120), torch.float32)],
        {"model_names": ["onnx_whisper_openai_whisper_large_v3_clm_hf"], "pcc": 0.99},
    ),
    (Add46, [((1, 4096), torch.float32)], {"model_names": ["pd_alexnet_base_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Add110,
        [((1, 12, 8, 8), torch.float32)],
        {"model_names": ["pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp"], "pcc": 0.99},
    ),
    (
        Add0,
        [((96,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Add111,
        [((96,), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((160,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Add112,
        [((160,), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 160, 56, 56), torch.float32), ((160, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((192,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Add96,
        [((192,), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 192, 56, 56), torch.float32), ((192, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((224,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (Add113, [((224,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Add0,
        [((1, 224, 56, 56), torch.float32), ((224, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 224, 28, 28), torch.float32), ((224, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 288, 28, 28), torch.float32), ((288, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((320,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Add114,
        [((320,), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 320, 28, 28), torch.float32), ((320, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((352,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (Add115, [((352,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Add0,
        [((1, 352, 28, 28), torch.float32), ((352, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((384,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Add64,
        [((384,), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((416,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (Add116, [((416,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Add0,
        [((1, 416, 28, 28), torch.float32), ((416, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((448,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (Add117, [((448,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Add0,
        [((1, 448, 28, 28), torch.float32), ((448, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 480, 28, 28), torch.float32), ((480, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 320, 14, 14), torch.float32), ((320, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 352, 14, 14), torch.float32), ((352, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 416, 14, 14), torch.float32), ((416, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 448, 14, 14), torch.float32), ((448, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((544,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (Add118, [((544,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Add0,
        [((1, 544, 14, 14), torch.float32), ((544, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((576,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Add119,
        [((576,), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((608,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (Add120, [((608,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Add0,
        [((1, 608, 14, 14), torch.float32), ((608, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((640,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (Add121, [((640,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Add0,
        [((1, 640, 14, 14), torch.float32), ((640, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((672,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (Add122, [((672,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Add0,
        [((704,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (Add123, [((704,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Add0,
        [((1, 704, 14, 14), torch.float32), ((704, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((736,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (Add124, [((736,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Add0,
        [((1, 736, 14, 14), torch.float32), ((736, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((768,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (Add3, [((768,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Add0,
        [((800,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (Add125, [((800,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Add0,
        [((1, 800, 14, 14), torch.float32), ((800, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((832,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (Add126, [((832,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Add0,
        [((1, 832, 14, 14), torch.float32), ((832, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((864,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (Add127, [((864,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Add0,
        [((1, 864, 14, 14), torch.float32), ((864, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((896,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (Add128, [((896,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Add0,
        [((1, 896, 14, 14), torch.float32), ((896, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((928,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (Add129, [((928,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Add0,
        [((1, 928, 14, 14), torch.float32), ((928, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((960,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Add130,
        [((960,), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 960, 14, 14), torch.float32), ((960, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((992,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (Add131, [((992,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Add0,
        [((1, 992, 14, 14), torch.float32), ((992, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 128, 7, 7), torch.float32), ((128, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 544, 7, 7), torch.float32), ((544, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 608, 7, 7), torch.float32), ((608, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 640, 7, 7), torch.float32), ((640, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 704, 7, 7), torch.float32), ((704, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 736, 7, 7), torch.float32), ((736, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 768, 7, 7), torch.float32), ((768, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 800, 7, 7), torch.float32), ((800, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 832, 7, 7), torch.float32), ((832, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 864, 7, 7), torch.float32), ((864, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 896, 7, 7), torch.float32), ((896, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 928, 7, 7), torch.float32), ((928, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 992, 7, 7), torch.float32), ((992, 1, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((32,), torch.float32), ((1,), torch.float32)],
        {
            "model_names": ["pd_mobilenetv1_basic_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Add132,
        [((32,), torch.float32)],
        {
            "model_names": ["pd_mobilenetv1_basic_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1280,), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pd_mobilenetv2_basic_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (Add98, [((1280,), torch.float32)], {"model_names": ["pd_mobilenetv2_basic_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Add37,
        [((1, 25, 97), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 14, 128), torch.float32), ((1, 14, 128), torch.float32)],
        {"model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"], "pcc": 0.99},
    ),
    (
        Add3,
        [((1, 14, 768), torch.float32)],
        {"model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 12, 14, 14), torch.float32), ((1, 1, 1, 14), torch.float32)],
        {"model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 14, 768), torch.float32), ((1, 14, 768), torch.float32)],
        {"model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"], "pcc": 0.99},
    ),
    (
        Add28,
        [((1, 14, 3072), torch.float32)],
        {"model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((14, 1), torch.float32), ((1,), torch.float32)],
        {"model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"], "pcc": 0.99},
    ),
    (
        Add42,
        [((1, 384, 1024), torch.float32)],
        {"model_names": ["pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf"], "pcc": 0.99},
    ),
    (
        Add46,
        [((1, 384, 4096), torch.float32)],
        {"model_names": ["pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf"], "pcc": 0.99},
    ),
    (
        Add6,
        [((1, 588, 1), torch.float32)],
        {"model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 16, 588, 128), torch.float32), ((1, 16, 588, 128), torch.float32)],
        {"model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"], "pcc": 0.99},
    ),
    (
        Add133,
        [((1, 16, 588, 588), torch.float32)],
        {"model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 588, 2048), torch.float32), ((1, 588, 2048), torch.float32)],
        {"model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"], "pcc": 0.99},
    ),
    (
        Add134,
        [((1, 128, 28996), torch.float32)],
        {"model_names": ["pt_distilbert_distilbert_base_cased_mlm_hf"], "pcc": 0.99},
    ),
    (
        Add45,
        [((1, 256, 8192), torch.float32)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf"], "pcc": 0.99},
    ),
    (Add6, [((1, 6, 1), torch.float32)], {"model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"], "pcc": 0.99}),
    (
        Add0,
        [((1, 2048, 1, 9), torch.float32), ((2048, 1, 1), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"], "pcc": 0.99},
    ),
    (
        Add43,
        [((1, 6, 2048), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"], "pcc": 0.99},
    ),
    (
        Add6,
        [((1, 6, 2048), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 2048, 16), torch.float32), ((1, 2048, 16), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 2048, 6), torch.float32), ((1, 2048, 6), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 6, 1024), torch.float32), ((1, 6, 1024), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((2, 1, 1536), torch.float32), ((2, 1, 1536), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((2, 1, 1536), torch.float32), ((1, 1536), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Add6,
        [((2, 13, 1), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 12, 13, 13), torch.float32), ((2, 1, 1, 13), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((2, 12, 13, 13), torch.float32), ((2, 12, 13, 13), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((2, 13, 768), torch.float32), ((2, 13, 768), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Add68,
        [((2, 13, 1536), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((2, 1, 1, 13), torch.float32), ((2, 1, 1, 13), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((2, 24, 1, 13), torch.float32), ((2, 1, 1, 13), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1, 512), torch.float32), ((1, 1, 512), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add41,
        [((1, 1, 512), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 512, 3000), torch.float32), ((512, 1), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 512, 1500), torch.float32), ((512, 1), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add135,
        [((1, 1500, 512), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add41,
        [((1, 1500, 512), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 1500, 512), torch.float32), ((1, 1500, 512), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add43,
        [((1, 1500, 2048), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Add43,
        [((1, 1, 2048), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    record_forge_op_name("Add")

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
